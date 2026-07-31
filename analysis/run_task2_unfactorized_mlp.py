from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_task2_condition_ablation import resolve_full_run_dir, resolve_sample_run_dir, task_bundle_paths
from run_task2_reviewer_baselines import (
    DEFAULT_PRETRAIN_ROOT,
    SOURCE_DATA_ROOT,
    STATE_LABELS,
    curve_shape_stats,
    load_pickle,
    regression_stats,
)


class UnfactorizedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, values):
        return self.network(values)


def tail_indices(length: int) -> np.ndarray:
    tail_count = max(2, int(math.ceil(0.2 * length)))
    return np.unique(np.concatenate([np.arange(tail_count), np.arange(max(0, length - tail_count), length)])).astype(np.int64)


def symmetric_tail_kl(truth: np.ndarray, prediction: np.ndarray) -> float:
    indices = tail_indices(len(truth))
    eps = 1.0e-9
    true_tail = np.clip(np.asarray(truth, dtype=np.float64)[indices], a_min=0.0, a_max=None) + eps
    pred_tail = np.clip(np.asarray(prediction, dtype=np.float64)[indices], a_min=0.0, a_max=None) + eps
    true_prob = true_tail / np.sum(true_tail)
    pred_prob = pred_tail / np.sum(pred_tail)
    forward = np.sum(true_prob * np.log(true_prob / pred_prob))
    reverse = np.sum(pred_prob * np.log(pred_prob / true_prob))
    return float(0.5 * (forward + reverse))


def scaled_task_features(task: dict, run_dir: Path) -> np.ndarray:
    step_scaler = load_pickle(run_dir / "step_scaler.pkl")
    retention_scaler = load_pickle(run_dir / "retention_scaler.pkl")
    pec_scaler = load_pickle(run_dir / "pec_scaler.pkl")
    steps = np.asarray(task["steps"], dtype=np.float32).reshape(-1, 1)
    step_scaled = step_scaler.transform(steps).astype(np.float32)
    wl_scaled = np.full_like(step_scaled, 2.0 * (float(task["WL"]) / 255.0) - 1.0, dtype=np.float32)
    retention_scaled_value = float(retention_scaler.transform([[float(task["Retention"])]])[0, 0])
    pec_scaled_value = float(pec_scaler.transform([[float(task["PEC"])]])[0, 0])
    retention_scaled = np.full_like(step_scaled, retention_scaled_value, dtype=np.float32)
    pec_scaled = np.full_like(step_scaled, pec_scaled_value, dtype=np.float32)
    return np.hstack([step_scaled, wl_scaled, retention_scaled, pec_scaled]).astype(np.float32)


def log_targets(task: dict) -> np.ndarray:
    freqs = np.asarray(task["freqs"], dtype=np.float32)
    return np.log1p(np.clip(freqs, a_min=0.0, a_max=None)).reshape(-1, 1).astype(np.float32)


def build_rows(tasks: list[dict], run_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for task in tasks:
        task_features = scaled_task_features(task, run_dir)
        task_targets = log_targets(task)
        if split_name == "all":
            indices = np.arange(int(task["num_points"]), dtype=np.int64)
        elif split_name == "train_val":
            indices = np.asarray(task["splits"]["train"] + task["splits"]["val"], dtype=np.int64)
        else:
            indices = np.asarray(task["splits"][split_name], dtype=np.int64)
        features.append(task_features[indices])
        targets.append(task_targets[indices])
    return np.vstack(features).astype(np.float32), np.vstack(targets).astype(np.float32)


def train_model(
    model: UnfactorizedMLP,
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    x_tensor = torch.as_tensor(features, dtype=torch.float32)
    y_tensor = torch.as_tensor(targets, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.MSELoss()
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best_loss = float("inf")
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        loss = criterion(model(x_tensor), y_tensor)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)


def predict_task(model: UnfactorizedMLP, task: dict, run_dir: Path) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_log = model(torch.as_tensor(scaled_task_features(task, run_dir), dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
    return np.clip(np.expm1(np.clip(pred_log, a_min=-20.0, a_max=12.0)), a_min=0.0, a_max=None)


def evaluate_model(model_index: int, target_tasks: list[dict], model: UnfactorizedMLP, run_dir: Path) -> tuple[dict, list[dict]]:
    rows = []
    for task in target_tasks:
        prediction = predict_task(model, task, run_dir)
        truth = np.asarray(task["freqs"], dtype=np.float64)
        steps = np.asarray(task["steps"], dtype=np.float64)
        test_indices = np.asarray(task["splits"]["test"], dtype=np.int64)
        stats = regression_stats(truth[test_indices], prediction[test_indices])
        shape = curve_shape_stats(steps, truth, prediction)
        rows.append(
            {
                "model_index": int(model_index),
                "curve_family": f"P{model_index - 1}",
                "label": STATE_LABELS[model_index],
                "method": "unfactorized_mlp",
                "task_id": int(task["task_id"]),
                "WL": int(task["WL"]),
                "Retention": int(task["Retention"]),
                "PEC": int(task["PEC"]),
                "test_r2": stats["r2"],
                "test_rmse": stats["rmse"],
                "test_mae": stats["mae"],
                **{key: stats[key] for key in ("n", "sse", "mae_sum", "sum_y", "sum_y2")},
                **shape,
                "tail_symmetric_kl": symmetric_tail_kl(truth, prediction),
            }
        )

    total_n = int(sum(row["n"] for row in rows))
    total_sse = float(sum(row["sse"] for row in rows))
    total_mae = float(sum(row["mae_sum"] for row in rows))
    total_sum_y = float(sum(row["sum_y"] for row in rows))
    total_sum_y2 = float(sum(row["sum_y2"] for row in rows))
    ss_tot = float(total_sum_y2 - (total_sum_y**2) / max(total_n, 1))
    task_r2 = np.asarray([row["test_r2"] for row in rows if row["test_r2"] is not None], dtype=np.float64)
    summary = {
        "model_index": int(model_index),
        "curve_family": f"P{model_index - 1}",
        "label": STATE_LABELS[model_index],
        "method": "unfactorized_mlp",
        "task_count": int(len(rows)),
        "point_r2": float(1.0 - total_sse / ss_tot) if ss_tot > 0.0 else np.nan,
        "task_mean_r2": float(np.mean(task_r2)),
        "task_median_r2": float(np.median(task_r2)),
        "point_rmse": float(math.sqrt(total_sse / max(total_n, 1))),
        "point_mae": float(total_mae / max(total_n, 1)),
        "frac_test_r2_gt_0_9": float(np.mean(task_r2 > 0.9)),
        "frac_test_r2_lt_0": float(np.mean(task_r2 < 0.0)),
        "p05_test_r2": float(np.quantile(task_r2, 0.05)),
        "mean_test_rmse": float(np.mean([row["test_rmse"] for row in rows])),
        "mean_test_mae": float(np.mean([row["test_mae"] for row in rows])),
        "mean_abs_peak_step_error": float(np.mean([row["abs_peak_step_error"] for row in rows])),
        "mean_tail_rmse_pct_peak": float(np.mean([row["tail_rmse_pct_peak"] for row in rows])),
        "mean_tail_symmetric_kl": float(np.mean([row["tail_symmetric_kl"] for row in rows])),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an unfactorized [step, WL, retention, PEC] MLP baseline.")
    parser.add_argument("--pretrain-root", default=str(DEFAULT_PRETRAIN_ROOT))
    parser.add_argument("--models", default="3,5,8")
    parser.add_argument("--run-kind", choices=["sample", "full"], default="sample")
    parser.add_argument("--source-epochs", type=int, default=1500)
    parser.add_argument("--target-epochs", type=int, default=1500)
    parser.add_argument("--source-lr", type=float, default=1.0e-3)
    parser.add_argument("--target-lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--target-split", choices=["train", "train_val"], default="train_val")
    parser.add_argument("--output", default="task2_unfactorized_mlp_representative.csv")
    parser.add_argument("--task-output", default="task2_unfactorized_mlp_representative_task_metrics.csv")
    args = parser.parse_args()

    torch.manual_seed(20260602)
    torch.set_num_threads(1)

    pretrain_root = Path(args.pretrain_root).resolve()
    result_root = pretrain_root / "artifacts" / "curve_transfer_batch"
    final_summary = pd.read_csv(result_root / "final_summary.csv")
    model_indices = [int(part.strip()) for part in args.models.split(",") if part.strip()]
    SOURCE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    task_rows = []
    for model_index in model_indices:
        run_dir = (
            resolve_sample_run_dir(result_root, model_index)
            if args.run_kind == "sample"
            else resolve_full_run_dir(result_root, final_summary, model_index)
        )
        source_task_path, target_task_path = task_bundle_paths(run_dir, args.run_kind)
        source_tasks = load_pickle(source_task_path)
        target_tasks = load_pickle(target_task_path)

        model = UnfactorizedMLP()
        source_features, source_targets = build_rows(source_tasks, run_dir, "all")
        train_model(model, source_features, source_targets, args.source_epochs, args.source_lr, args.weight_decay)
        target_features, target_targets = build_rows(target_tasks, run_dir, args.target_split)
        train_model(model, target_features, target_targets, args.target_epochs, args.target_lr, args.weight_decay)

        summary, rows = evaluate_model(model_index, target_tasks, model, run_dir)
        summary["run_kind"] = args.run_kind
        summary["source_epochs"] = int(args.source_epochs)
        summary["target_epochs"] = int(args.target_epochs)
        summary["target_split"] = args.target_split
        summary_rows.append(summary)
        task_rows.extend(rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = SOURCE_DATA_ROOT / args.output
    task_path = SOURCE_DATA_ROOT / args.task_output
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(task_rows).to_csv(task_path, index=False)
    print(f"wrote {summary_path}")
    print(f"wrote {task_path}")


if __name__ == "__main__":
    main()
