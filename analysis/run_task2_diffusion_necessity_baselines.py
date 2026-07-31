from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from run_task2_condition_ablation import (
    build_finetune_config,
    condition_paths,
    resolve_full_run_dir,
    resolve_sample_run_dir,
    summarize_ablation,
)
from run_task2_fixed_anchor_protocol import nearest_source_indices
from run_task2_reviewer_baselines import (
    DEFAULT_PRETRAIN_ROOT,
    SOURCE_DATA_ROOT,
    STATE_LABELS,
    load_mlp_regressor,
    load_pickle,
)


class Hypernetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, out_dim),
        )

    def forward(self, values):
        return self.network(values)


def resolve_subset_file(run_dir: Path, pattern: str) -> Path:
    matches = sorted((run_dir / "subsets").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing subset file pattern {pattern} under {run_dir / 'subsets'}")
    return matches[0]


def task_bundle_paths(run_dir: Path, run_kind: str) -> tuple[Path, Path]:
    if run_kind == "sample":
        return (
            resolve_subset_file(run_dir, "source_tasks_*.pkl"),
            resolve_subset_file(run_dir, "target_tasks_*.pkl"),
        )
    return run_dir / "source_tasks.pkl", run_dir / "target_tasks.pkl"


def transform_freqs(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(values, dtype=np.float32), a_min=0.0, a_max=None)).reshape(-1, 1)


def build_source_arrays(source_tasks: list[dict], step_scaler) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for task in source_tasks:
        x_scaled = step_scaler.transform(np.asarray(task["steps"], dtype=np.float32).reshape(-1, 1)).astype(np.float32)
        y_log = transform_freqs(np.asarray(task["freqs"], dtype=np.float32))
        x_rows.append(x_scaled)
        y_rows.append(y_log)
    return np.vstack(x_rows).astype(np.float32), np.vstack(y_rows).astype(np.float32)


def train_pooled_source_mlp(
    source_tasks: list[dict],
    step_scaler,
    pretrain_root: Path,
    epochs: int,
    lr: float,
) -> np.ndarray:
    torch.manual_seed(42)
    torch.set_num_threads(1)
    mlp_regressor = load_mlp_regressor(pretrain_root)
    model = mlp_regressor(in_dim=1, hidden_dims=(32, 16), out_dim=1, activation="tanh")
    x_train, y_train = build_source_arrays(source_tasks, step_scaler)
    x_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1.0e-6)
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
    return parameters_to_vector(model.parameters()).detach().cpu().numpy().astype(np.float32)


def condition_features(frame: pd.DataFrame, run_dir: Path) -> np.ndarray:
    retention_scaler = load_pickle(run_dir / "retention_scaler.pkl")
    pec_scaler = load_pickle(run_dir / "pec_scaler.pkl")
    wl = frame["WL"].astype(float).to_numpy().reshape(-1, 1)
    wl_scaled = 2.0 * (wl / 255.0) - 1.0
    retention_scaled = retention_scaler.transform(frame[["Retention"]]).astype(np.float32)
    pec_scaled = pec_scaler.transform(frame[["PEC"]]).astype(np.float32)
    return np.hstack([wl_scaled, retention_scaled, pec_scaled]).astype(np.float32)


def train_hypernetwork(
    source_conditions: pd.DataFrame,
    target_conditions: pd.DataFrame,
    source_params: np.ndarray,
    run_dir: Path,
    epochs: int,
    lr: float,
) -> np.ndarray:
    torch.manual_seed(31415)
    torch.set_num_threads(1)
    source_features = condition_features(source_conditions, run_dir)
    target_features = condition_features(target_conditions, run_dir)
    source_params = np.asarray(source_params, dtype=np.float32)
    param_mean = source_params.mean(axis=0, keepdims=True)
    param_std = np.maximum(source_params.std(axis=0, keepdims=True), 1.0e-6)
    y_train = (source_params - param_mean) / param_std
    x_tensor = torch.as_tensor(source_features, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    model = Hypernetwork(in_dim=source_features.shape[1], out_dim=source_params.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1.0e-5)
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
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.as_tensor(target_features, dtype=torch.float32)).detach().cpu().numpy()
    return (pred_scaled * param_std + param_mean).astype(np.float32)


def baseline_parameter_matrix(
    method: str,
    run_dir: Path,
    run_kind: str,
    pretrain_root: Path,
    source_conditions: pd.DataFrame,
    target_conditions: pd.DataFrame,
    source_tasks: list[dict],
    target_tasks: list[dict],
    source_params: np.ndarray,
    pooled_epochs: int,
    pooled_lr: float,
    hyper_epochs: int,
    hyper_lr: float,
) -> np.ndarray:
    if method == "diffusion_transfer":
        return np.load(sorted(run_dir.glob("sampleSeq_RealParams_*.npy"))[0]).reshape(len(target_tasks), -1).astype(np.float32)
    if method == "source_pooled_mlp":
        step_scaler = load_pickle(run_dir / "step_scaler.pkl")
        vector = train_pooled_source_mlp(source_tasks, step_scaler, pretrain_root, epochs=pooled_epochs, lr=pooled_lr)
        return np.repeat(vector.reshape(1, -1), len(target_tasks), axis=0).astype(np.float32)
    if method == "nearest_parameter_transfer":
        indices = nearest_source_indices(source_conditions, target_tasks)
        return np.asarray(source_params[indices], dtype=np.float32)
    if method == "hypernetwork_parameter":
        return train_hypernetwork(
            source_conditions=source_conditions,
            target_conditions=target_conditions,
            source_params=source_params,
            run_dir=run_dir,
            epochs=hyper_epochs,
            lr=hyper_lr,
        )
    raise ValueError(f"Unsupported baseline method: {method}")


def run_finetune(
    pretrain_root: Path,
    run_dir: Path,
    run_kind: str,
    method: str,
    strategy: str,
    output_dir: Path,
    params_path: Path,
    workers: int,
) -> None:
    sys.path.insert(0, str(pretrain_root))
    from curve_task_workflow import finetune_target_domain_models  # noqa: WPS433

    source_condition_path, target_condition_path = condition_paths(run_dir, run_kind)
    config = build_finetune_config(
        pretrain_root=pretrain_root,
        run_dir=run_dir,
        run_kind=run_kind,
        output_dir=output_dir,
        source_condition_csv=source_condition_path,
        target_condition_csv=target_condition_path,
        strategy=strategy,
        workers=workers,
    )
    finetune_target_domain_models(config, params_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 2 diffusion-necessity baselines.")
    parser.add_argument("--pretrain-root", default=str(DEFAULT_PRETRAIN_ROOT))
    parser.add_argument("--models", default="3,5,8")
    parser.add_argument("--run-kind", choices=["sample", "full"], default="sample")
    parser.add_argument(
        "--methods",
        default="scratch_only,source_pooled_mlp,nearest_parameter_transfer,hypernetwork_parameter,diffusion_transfer",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pooled-epochs", type=int, default=1200)
    parser.add_argument("--pooled-lr", type=float, default=1.0e-3)
    parser.add_argument("--hyper-epochs", type=int, default=2500)
    parser.add_argument("--hyper-lr", type=float, default=1.0e-3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", default="task2_diffusion_necessity_representative.csv")
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root).resolve()
    result_root = pretrain_root / "artifacts" / "curve_transfer_batch"
    final_summary = pd.read_csv(result_root / "final_summary.csv")
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    model_indices = [int(part.strip()) for part in args.models.split(",") if part.strip()]
    SOURCE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_index in model_indices:
        run_dir = (
            resolve_sample_run_dir(result_root, model_index)
            if args.run_kind == "sample"
            else resolve_full_run_dir(result_root, final_summary, model_index)
        )
        source_task_path, target_task_path = task_bundle_paths(run_dir, args.run_kind)
        source_tasks = load_pickle(source_task_path)
        target_tasks = load_pickle(target_task_path)
        source_condition_path, target_condition_path = condition_paths(run_dir, args.run_kind)
        source_conditions = pd.read_csv(source_condition_path)
        target_conditions = pd.read_csv(target_condition_path)
        source_params = np.load(run_dir / "source_params.npy").reshape(len(source_tasks), -1).astype(np.float32)
        diffusion_params = np.load(sorted(run_dir.glob("sampleSeq_RealParams_*.npy"))[0]).reshape(len(target_tasks), -1).astype(np.float32)

        for method in methods:
            output_dir = run_dir / "diffusion_necessity" / method
            output_dir.mkdir(parents=True, exist_ok=True)
            params_path = output_dir / "init_params.npy"
            strategy = "scratch_only" if method == "scratch_only" else "generated_only"

            if method == "scratch_only":
                if args.force or not params_path.exists():
                    np.save(params_path, diffusion_params)
            elif args.force or not params_path.exists():
                params = baseline_parameter_matrix(
                    method=method,
                    run_dir=run_dir,
                    run_kind=args.run_kind,
                    pretrain_root=pretrain_root,
                    source_conditions=source_conditions,
                    target_conditions=target_conditions,
                    source_tasks=source_tasks,
                    target_tasks=target_tasks,
                    source_params=source_params,
                    pooled_epochs=args.pooled_epochs,
                    pooled_lr=args.pooled_lr,
                    hyper_epochs=args.hyper_epochs,
                    hyper_lr=args.hyper_lr,
                )
                np.save(params_path, params.astype(np.float32))

            report_path = output_dir / "target_transfer_report.json"
            if args.force or not report_path.exists():
                run_finetune(
                    pretrain_root=pretrain_root,
                    run_dir=run_dir,
                    run_kind=args.run_kind,
                    method=method,
                    strategy=strategy,
                    output_dir=output_dir,
                    params_path=params_path,
                    workers=args.workers,
                )

            row = summarize_ablation(
                model_index=model_index,
                run_kind=args.run_kind,
                ablation=method,
                output_dir=output_dir,
                run_dir=run_dir,
                pretrain_root=pretrain_root,
                diffusion_sample_path=params_path,
            )
            row["method"] = method
            row["strategy"] = strategy
            row["label"] = STATE_LABELS[model_index]
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2))

    output_path = SOURCE_DATA_ROOT / args.output
    pd.DataFrame(rows).sort_values(["model_index", "method"]).to_csv(output_path, index=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
