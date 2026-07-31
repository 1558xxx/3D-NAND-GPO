from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from run_task2_reviewer_baselines import (
    DEFAULT_PRETRAIN_ROOT,
    SOURCE_DATA_ROOT,
    STATE_LABELS,
    curve_shape_stats,
    gaussian_predict,
    load_mlp_regressor,
    regression_stats,
    sort_xy,
    spline_predict,
)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def resolve_sample_run_dir(result_root: Path, model_index: int) -> Path:
    model_dir = result_root / f"model_{model_index}" / "sample"
    matches = sorted(path for path in model_dir.glob("*") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"Missing sample run directory under {model_dir}")
    return matches[0]


def resolve_full_run_dir(result_root: Path, final_summary: pd.DataFrame, model_index: int) -> Path:
    candidate = str(final_summary.loc[final_summary["model_index"] == model_index, "candidate_name"].iloc[0])
    return result_root / f"model_{model_index}" / "full" / candidate


def resolve_subset_file(run_dir: Path, pattern: str) -> Path:
    matches = sorted((run_dir / "subsets").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing subset file pattern {pattern} under {run_dir / 'subsets'}")
    return matches[0]


def fixed_anchor_indices(task: dict, budget: int) -> list[int]:
    length = int(task["num_points"])
    budget = min(int(budget), length)
    if budget >= length:
        return list(range(length))

    peak_index = int(np.argmax(np.asarray(task["freqs"], dtype=np.float64)))
    left_slots = budget // 2
    right_slots = budget - left_slots - 1
    left = np.linspace(0, peak_index, num=left_slots + 1)
    right = np.linspace(peak_index, length - 1, num=right_slots + 1)
    indices = [int(round(value)) for value in left[:-1]] + [peak_index] + [int(round(value)) for value in right[1:]]
    selected = sorted({max(0, min(length - 1, index)) for index in indices})
    if len(selected) < budget:
        fallback = [int(round(value)) for value in np.linspace(0, length - 1, num=budget)]
        for index in fallback:
            selected.append(max(0, min(length - 1, index)))
            selected = sorted(set(selected))
            if len(selected) == budget:
                break
    return selected[:budget]


def inverse_freqs(values: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(np.asarray(values, dtype=np.float32), a_min=-20.0, a_max=12.0)).astype(np.float32)


def transform_freqs(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(values, dtype=np.float32), a_min=0.0, a_max=None)).reshape(-1, 1)


def predict_from_params(task: dict, parameter_vector: np.ndarray, step_scaler, mlp_regressor) -> np.ndarray:
    model = mlp_regressor(in_dim=1, hidden_dims=(32, 16), out_dim=1, activation="tanh")
    vector_to_parameters(torch.as_tensor(np.asarray(parameter_vector, dtype=np.float32).reshape(-1)), model.parameters())
    model.eval()
    x_scaled = step_scaler.transform(np.asarray(task["steps"], dtype=np.float32).reshape(-1, 1)).astype(np.float32)
    with torch.no_grad():
        pred_log = model(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy().reshape(-1, 1)
    return np.clip(inverse_freqs(pred_log).reshape(-1), a_min=0.0, a_max=None)


def fit_prior_regularized_mlp(
    task: dict,
    anchor_indices: list[int],
    init_vector: np.ndarray,
    step_scaler,
    mlp_regressor,
    prior_lambda: float,
    epochs: int,
    lr: float,
) -> np.ndarray:
    torch.manual_seed(42 + int(task["task_id"]))
    model = mlp_regressor(in_dim=1, hidden_dims=(32, 16), out_dim=1, activation="tanh")
    init_tensor = torch.as_tensor(np.asarray(init_vector, dtype=np.float32).reshape(-1))
    vector_to_parameters(init_tensor.clone(), model.parameters())
    x_all = step_scaler.transform(np.asarray(task["steps"], dtype=np.float32).reshape(-1, 1)).astype(np.float32)
    y_all = transform_freqs(np.asarray(task["freqs"], dtype=np.float32))
    x_train = torch.as_tensor(x_all[np.asarray(anchor_indices, dtype=np.int64)], dtype=torch.float32)
    y_train = torch.as_tensor(y_all[np.asarray(anchor_indices, dtype=np.int64)], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best_loss = float("inf")
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        predictions = model(x_train)
        data_loss = torch.mean((predictions - y_train) ** 2)
        current_vector = parameters_to_vector(model.parameters())
        prior_loss = torch.mean((current_vector - init_tensor) ** 2)
        loss = data_loss + float(prior_lambda) * prior_loss
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


def nearest_source_indices(source_conditions: pd.DataFrame, target_tasks: list[dict]) -> np.ndarray:
    source_matrix = source_conditions[["WL", "Retention", "PEC"]].astype(float).to_numpy()
    source_scale = np.maximum(source_matrix.std(axis=0), 1.0)
    indices = []
    for task in target_tasks:
        target = np.asarray([task["WL"], task["Retention"], task["PEC"]], dtype=np.float64)
        distance = np.sum(((source_matrix - target) / source_scale) ** 2, axis=1)
        indices.append(int(np.argmin(distance)))
    return np.asarray(indices, dtype=np.int64)


def evaluate_prediction(task: dict, test_indices: list[int], prediction: np.ndarray) -> dict:
    steps = np.asarray(task["steps"], dtype=np.float64)
    truth = np.asarray(task["freqs"], dtype=np.float64)
    test_idx = np.asarray(test_indices, dtype=np.int64)
    stats = regression_stats(truth[test_idx], prediction[test_idx])
    return {
        "task_id": int(task["task_id"]),
        "WL": int(task["WL"]),
        "Retention": int(task["Retention"]),
        "PEC": int(task["PEC"]),
        "test_r2": stats["r2"],
        "test_rmse": stats["rmse"],
        "test_mae": stats["mae"],
        **{key: stats[key] for key in ("n", "sse", "mae_sum", "sum_y", "sum_y2")},
        **curve_shape_stats(steps, truth, prediction),
    }


def aggregate_rows(model_index: int, method: str, budget: int, rows: list[dict]) -> dict:
    total_n = int(sum(row["n"] for row in rows))
    total_sse = float(sum(row["sse"] for row in rows))
    total_mae = float(sum(row["mae_sum"] for row in rows))
    total_sum_y = float(sum(row["sum_y"] for row in rows))
    total_sum_y2 = float(sum(row["sum_y2"] for row in rows))
    ss_tot = float(total_sum_y2 - (total_sum_y ** 2) / max(total_n, 1))
    task_r2 = np.asarray([row["test_r2"] for row in rows if row["test_r2"] is not None], dtype=np.float64)
    return {
        "model_index": int(model_index),
        "curve_family": f"P{model_index - 1}",
        "label": STATE_LABELS[model_index],
        "budget": int(budget),
        "method": method,
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
    }


def evaluate_family(
    pretrain_root: Path,
    result_root: Path,
    final_summary: pd.DataFrame,
    model_index: int,
    run_kind: str,
    budgets: list[int],
    prior_lambda: float,
    epochs: int,
    lr: float,
) -> tuple[list[dict], list[dict]]:
    run_dir = resolve_sample_run_dir(result_root, model_index) if run_kind == "sample" else resolve_full_run_dir(result_root, final_summary, model_index)
    target_task_path = resolve_subset_file(run_dir, "target_tasks_*.pkl") if run_kind == "sample" else run_dir / "target_tasks.pkl"
    source_condition_path = resolve_subset_file(run_dir, "source_conditions_*.csv") if run_kind == "sample" else run_dir / "source_conditions.csv"
    tasks = load_pickle(target_task_path)
    step_scaler = load_pickle(run_dir / "step_scaler.pkl")
    diffusion_params = np.load(sorted(run_dir.glob("sampleSeq_RealParams_*.npy"))[0])
    source_params = np.load(run_dir / "source_params.npy")
    source_conditions = pd.read_csv(source_condition_path)
    source_nearest = nearest_source_indices(source_conditions, tasks)
    mlp_regressor = load_mlp_regressor(pretrain_root)

    summary_rows = []
    task_rows = []
    for budget in budgets:
        method_rows = {
            "gaussian": [],
            "spline": [],
            "source_pretrained_mlp": [],
            "proposed_prior_regularized": [],
        }
        for task_index, task in enumerate(tasks):
            anchor_indices = fixed_anchor_indices(task, budget)
            anchor_set = set(anchor_indices)
            test_indices = [index for index in range(int(task["num_points"])) if index not in anchor_set]
            steps = np.asarray(task["steps"], dtype=np.float64)
            freqs = np.asarray(task["freqs"], dtype=np.float64)
            anchor_x, anchor_y = sort_xy(steps[np.asarray(anchor_indices, dtype=np.int64)], freqs[np.asarray(anchor_indices, dtype=np.int64)])

            classical_predictions = {
                "gaussian": gaussian_predict(anchor_x, anchor_y, steps),
                "spline": spline_predict(anchor_x, anchor_y, steps),
            }
            for method, prediction in classical_predictions.items():
                row = evaluate_prediction(task, test_indices, prediction)
                method_rows[method].append(row)

            source_vector = fit_prior_regularized_mlp(
                task,
                anchor_indices,
                source_params[source_nearest[task_index]],
                step_scaler,
                mlp_regressor,
                prior_lambda,
                epochs,
                lr,
            )
            source_prediction = predict_from_params(task, source_vector, step_scaler, mlp_regressor)
            method_rows["source_pretrained_mlp"].append(evaluate_prediction(task, test_indices, source_prediction))

            proposed_vector = fit_prior_regularized_mlp(
                task,
                anchor_indices,
                diffusion_params[task_index],
                step_scaler,
                mlp_regressor,
                prior_lambda,
                epochs,
                lr,
            )
            proposed_prediction = predict_from_params(task, proposed_vector, step_scaler, mlp_regressor)
            method_rows["proposed_prior_regularized"].append(evaluate_prediction(task, test_indices, proposed_prediction))

        for method, rows in method_rows.items():
            summary = aggregate_rows(model_index, method, budget, rows)
            summary["run_kind"] = run_kind
            summary["prior_lambda"] = float(prior_lambda)
            summary["epochs"] = int(epochs)
            summary["lr"] = float(lr)
            summary_rows.append(summary)
            for row in rows:
                task_rows.append(
                    {
                        "model_index": int(model_index),
                        "curve_family": f"P{model_index - 1}",
                        "label": STATE_LABELS[model_index],
                        "budget": int(budget),
                        "method": method,
                        "run_kind": run_kind,
                        **{key: row[key] for key in ("task_id", "WL", "Retention", "PEC", "test_r2", "test_rmse", "test_mae", "abs_peak_step_error", "tail_rmse_pct_peak")},
                    }
                )
        print(json.dumps(summary_rows[-len(method_rows):], ensure_ascii=False, indent=2))
    return summary_rows, task_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-anchor sparse acquisition protocol for Task 2.")
    parser.add_argument("--pretrain-root", default=str(DEFAULT_PRETRAIN_ROOT))
    parser.add_argument("--models", default="3,5,8")
    parser.add_argument("--run-kind", choices=["sample", "full"], default="sample")
    parser.add_argument("--budgets", default="5,7,9")
    parser.add_argument("--prior-lambda", type=float, default=1.0e-2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--output", default="task2_fixed_anchor_protocol.csv")
    parser.add_argument("--task-output", default="task2_fixed_anchor_protocol_task_metrics.csv")
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root).resolve()
    result_root = pretrain_root / "artifacts" / "curve_transfer_batch"
    final_summary = pd.read_csv(result_root / "final_summary.csv")
    model_indices = [int(part.strip()) for part in args.models.split(",") if part.strip()]
    budgets = [int(part.strip()) for part in args.budgets.split(",") if part.strip()]

    all_summary_rows = []
    all_task_rows = []
    for model_index in model_indices:
        print(f"RUN model_{model_index} {args.run_kind}")
        summary_rows, task_rows = evaluate_family(
            pretrain_root,
            result_root,
            final_summary,
            model_index,
            args.run_kind,
            budgets,
            args.prior_lambda,
            args.epochs,
            args.lr,
        )
        all_summary_rows.extend(summary_rows)
        all_task_rows.extend(task_rows)
        pd.DataFrame(all_summary_rows).to_csv(SOURCE_DATA_ROOT / args.output, index=False)
        pd.DataFrame(all_task_rows).to_csv(SOURCE_DATA_ROOT / args.task_output, index=False)

    SOURCE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = SOURCE_DATA_ROOT / args.output
    task_path = SOURCE_DATA_ROOT / args.task_output
    pd.DataFrame(all_summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(all_task_rows).to_csv(task_path, index=False)
    print(f"wrote {summary_path}")
    print(f"wrote {task_path}")


if __name__ == "__main__":
    main()
