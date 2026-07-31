from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_task2_fixed_anchor_protocol import (
    aggregate_rows,
    evaluate_prediction,
    fixed_anchor_indices,
    load_pickle,
    resolve_full_run_dir,
    resolve_sample_run_dir,
    resolve_subset_file,
)
from run_task2_reviewer_baselines import DEFAULT_PRETRAIN_ROOT, SOURCE_DATA_ROOT


class ConditionalMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def build_feature_rows(tasks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for task in tasks:
        for step, freq in zip(task["steps"], task["freqs"]):
            features.append([float(step), float(task["WL"]), float(task["Retention"]), float(task["PEC"])])
            targets.append([np.log1p(max(float(freq), 0.0))])
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def transform_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(features, dtype=np.float32) - mean) / std).astype(np.float32)


def task_features(task: dict) -> np.ndarray:
    return np.asarray(
        [[float(step), float(task["WL"]), float(task["Retention"]), float(task["PEC"])] for step in task["steps"]],
        dtype=np.float32,
    )


def train_conditional_mlp(source_tasks: list[dict], epochs: int, lr: float) -> tuple[ConditionalMLP, np.ndarray, np.ndarray]:
    torch.manual_seed(42)
    torch.set_num_threads(1)
    features, targets = build_feature_rows(source_tasks)
    mean = features.mean(axis=0, keepdims=True)
    std = np.maximum(features.std(axis=0, keepdims=True), 1.0e-6)
    x_train = torch.as_tensor(transform_features(features, mean, std), dtype=torch.float32)
    y_train = torch.as_tensor(targets, dtype=torch.float32)
    model = ConditionalMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1.0e-5)
    criterion = nn.MSELoss()
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best_loss = float("inf")
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, std


def calibrated_prediction(model: ConditionalMLP, mean: np.ndarray, std: np.ndarray, task: dict, anchor_indices: list[int]) -> np.ndarray:
    features = transform_features(task_features(task), mean, std)
    with torch.no_grad():
        pred_log = model(torch.as_tensor(features, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
    anchor_idx = np.asarray(anchor_indices, dtype=np.int64)
    anchor_pred = pred_log[anchor_idx]
    anchor_true = np.log1p(np.clip(np.asarray(task["freqs"], dtype=np.float64)[anchor_idx], a_min=0.0, a_max=None))
    design = np.column_stack([anchor_pred, np.ones_like(anchor_pred)])
    try:
        coef, *_ = np.linalg.lstsq(design, anchor_true, rcond=None)
        calibrated_log = coef[0] * pred_log + coef[1]
    except Exception:
        calibrated_log = pred_log
    return np.clip(np.expm1(np.clip(calibrated_log, a_min=-20.0, a_max=12.0)), a_min=0.0, a_max=None)


def evaluate_family(pretrain_root: Path, result_root: Path, final_summary: pd.DataFrame, model_index: int, run_kind: str, budgets: list[int], epochs: int, lr: float):
    run_dir = resolve_sample_run_dir(result_root, model_index) if run_kind == "sample" else resolve_full_run_dir(result_root, final_summary, model_index)
    source_task_path = resolve_subset_file(run_dir, "source_tasks_*.pkl") if run_kind == "sample" else run_dir / "source_tasks.pkl"
    target_task_path = resolve_subset_file(run_dir, "target_tasks_*.pkl") if run_kind == "sample" else run_dir / "target_tasks.pkl"
    source_tasks = load_pickle(source_task_path)
    target_tasks = load_pickle(target_task_path)
    model, mean, std = train_conditional_mlp(source_tasks, epochs=epochs, lr=lr)

    summary_rows = []
    task_rows = []
    for budget in budgets:
        rows = []
        for task in target_tasks:
            anchor_indices = fixed_anchor_indices(task, budget)
            anchor_set = set(anchor_indices)
            test_indices = [index for index in range(int(task["num_points"])) if index not in anchor_set]
            prediction = calibrated_prediction(model, mean, std, task, anchor_indices)
            rows.append(evaluate_prediction(task, test_indices, prediction))
        summary = aggregate_rows(model_index, "conditional_mlp_calibrated", budget, rows)
        summary["run_kind"] = run_kind
        summary["conditional_epochs"] = int(epochs)
        summary["conditional_lr"] = float(lr)
        summary_rows.append(summary)
        for row in rows:
            task_rows.append(
                {
                    "model_index": int(model_index),
                    "curve_family": f"P{model_index - 1}",
                    "label": summary["label"],
                    "budget": int(budget),
                    "method": "conditional_mlp_calibrated",
                    "run_kind": run_kind,
                    **{key: row[key] for key in ("task_id", "WL", "Retention", "PEC", "test_r2", "test_rmse", "test_mae", "abs_peak_step_error", "tail_rmse_pct_peak")},
                }
            )
    return summary_rows, task_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-anchor conditional MLP baseline.")
    parser.add_argument("--pretrain-root", default=str(DEFAULT_PRETRAIN_ROOT))
    parser.add_argument("--models", default="3,5,8")
    parser.add_argument("--run-kind", choices=["sample", "full"], default="sample")
    parser.add_argument("--budgets", default="5,7,9")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--output", default="task2_fixed_anchor_conditional_mlp.csv")
    parser.add_argument("--task-output", default="task2_fixed_anchor_conditional_mlp_task_metrics.csv")
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root).resolve()
    result_root = pretrain_root / "artifacts" / "curve_transfer_batch"
    final_summary = pd.read_csv(result_root / "final_summary.csv")
    model_indices = [int(part.strip()) for part in args.models.split(",") if part.strip()]
    budgets = [int(part.strip()) for part in args.budgets.split(",") if part.strip()]

    all_summary_rows = []
    all_task_rows = []
    for model_index in model_indices:
        print(f"RUN conditional MLP model_{model_index}")
        summary_rows, task_rows = evaluate_family(pretrain_root, result_root, final_summary, model_index, args.run_kind, budgets, args.epochs, args.lr)
        all_summary_rows.extend(summary_rows)
        all_task_rows.extend(task_rows)

    SOURCE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = SOURCE_DATA_ROOT / args.output
    task_path = SOURCE_DATA_ROOT / args.task_output
    pd.DataFrame(all_summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(all_task_rows).to_csv(task_path, index=False)
    print(f"wrote {summary_path}")
    print(f"wrote {task_path}")


if __name__ == "__main__":
    main()
