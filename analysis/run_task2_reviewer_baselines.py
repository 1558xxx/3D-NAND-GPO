from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from torch.nn.utils import vector_to_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_ROOT = PROJECT_ROOT / "paper_data"
DEFAULT_PRETRAIN_ROOT = PROJECT_ROOT / "Pretrain"
STATE_LABELS = {2: "P1 / M2", 3: "P2 / M3", 4: "P3 / M4", 5: "P4 / M5", 6: "P5 / M6", 7: "P6 / M7", 8: "P7 / M8"}


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    error = y_pred - y_true
    sse = float(np.sum(error ** 2))
    mae_sum = float(np.sum(np.abs(error)))
    sum_y = float(np.sum(y_true))
    sum_y2 = float(np.sum(y_true ** 2))
    n = int(y_true.size)
    ss_tot = float(sum_y2 - (sum_y ** 2) / max(n, 1))
    r2 = float(1.0 - sse / ss_tot) if ss_tot > 0.0 else None
    return {
        "n": n,
        "sse": sse,
        "mae_sum": mae_sum,
        "sum_y": sum_y,
        "sum_y2": sum_y2,
        "rmse": float(math.sqrt(sse / max(n, 1))),
        "mae": float(mae_sum / max(n, 1)),
        "r2": r2,
    }


def gaussian_curve(x_values, amplitude, mean, sigma, baseline):
    sigma = max(float(sigma), 1e-3)
    return baseline + amplitude * np.exp(-0.5 * ((x_values - mean) / sigma) ** 2)


def gaussian_predict(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
    weights = np.clip(train_y, 1.0, None)
    amplitude0 = max(float(train_y.max() - train_y.min()), 1.0)
    mean0 = float(np.average(train_x, weights=weights))
    sigma0 = max(float(np.sqrt(np.average((train_x - mean0) ** 2, weights=weights))), 1.0)
    baseline0 = max(float(train_y.min()), 0.0)
    upper_y = float(train_y.max())
    bounds = (
        [0.0, float(train_x.min()) - 10.0, 0.3, 0.0],
        [upper_y * 3.0 + 1.0, float(train_x.max()) + 10.0, 100.0, upper_y * 2.0 + 1.0],
    )
    try:
        params, _ = curve_fit(
            gaussian_curve,
            train_x,
            train_y,
            p0=[amplitude0, mean0, sigma0, baseline0],
            bounds=bounds,
            maxfev=10000,
        )
        pred = gaussian_curve(eval_x, *params)
    except Exception:
        pred = gaussian_curve(eval_x, amplitude0, mean0, sigma0, baseline0)
    return np.clip(np.asarray(pred, dtype=np.float64), a_min=0.0, a_max=None)


def spline_predict(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
    if train_x.size >= 3:
        pred = CubicSpline(train_x, train_y, bc_type="natural", extrapolate=True)(eval_x)
    else:
        pred = np.interp(eval_x, train_x, train_y)
    return np.clip(np.asarray(pred, dtype=np.float64), a_min=0.0, a_max=None)


def sort_xy(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x_values)
    return np.asarray(x_values[order], dtype=np.float64), np.asarray(y_values[order], dtype=np.float64)


def tail_indices(length: int) -> np.ndarray:
    tail_count = max(2, int(math.ceil(0.2 * length)))
    return np.unique(np.concatenate([np.arange(tail_count), np.arange(max(0, length - tail_count), length)])).astype(np.int64)


def curve_shape_stats(steps: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> dict:
    true_peak_index = int(np.argmax(truth))
    pred_peak_index = int(np.argmax(pred))
    peak_height = max(float(np.max(truth)), 1.0)
    tail_idx = tail_indices(len(truth))
    tail_rmse = float(np.sqrt(np.mean((pred[tail_idx] - truth[tail_idx]) ** 2)))
    return {
        "abs_peak_step_error": abs(float(steps[pred_peak_index] - steps[true_peak_index])),
        "tail_rmse_pct_peak": 100.0 * tail_rmse / peak_height,
    }


def evaluate_classical_task(payload: tuple[dict, str]) -> dict:
    task, method = payload
    steps = np.asarray(task["steps"], dtype=np.float64)
    freqs = np.asarray(task["freqs"], dtype=np.float64)
    train_idx = np.asarray(task["splits"]["train"] + task["splits"]["val"], dtype=np.int64)
    test_idx = np.asarray(task["splits"]["test"], dtype=np.int64)
    train_x, train_y = sort_xy(steps[train_idx], freqs[train_idx])
    test_x = steps[test_idx]
    test_y = freqs[test_idx]
    if method == "gaussian":
        test_pred = gaussian_predict(train_x, train_y, test_x)
        full_pred = gaussian_predict(train_x, train_y, steps)
    elif method == "spline":
        test_pred = spline_predict(train_x, train_y, test_x)
        full_pred = spline_predict(train_x, train_y, steps)
    else:
        raise ValueError(f"Unsupported method: {method}")
    test_stats = regression_stats(test_y, test_pred)
    shape_stats = curve_shape_stats(steps, freqs, full_pred)
    return {
        "task_id": int(task["task_id"]),
        "WL": int(task["WL"]),
        "Retention": int(task["Retention"]),
        "PEC": int(task["PEC"]),
        "test_r2": test_stats["r2"],
        "test_rmse": test_stats["rmse"],
        "test_mae": test_stats["mae"],
        **{key: test_stats[key] for key in ("n", "sse", "mae_sum", "sum_y", "sum_y2")},
        **shape_stats,
    }


def aggregate_rows(model_index: int, method: str, rows: list[dict]) -> dict:
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


def inverse_freqs(values: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(np.asarray(values, dtype=np.float32), a_min=-20.0, a_max=12.0)).astype(np.float32)


def load_mlp_regressor(pretrain_root: Path):
    sys.path.insert(0, str(pretrain_root))
    from Models import MLPRegressor  # noqa: WPS433

    return MLPRegressor


def reconstruct_proposed_curve(task: dict, parameter_vector: np.ndarray, step_scaler, mlp_regressor) -> np.ndarray:
    model = mlp_regressor(in_dim=1, hidden_dims=(32, 16), out_dim=1, activation="tanh")
    vector_to_parameters(torch.as_tensor(np.asarray(parameter_vector, dtype=np.float32).reshape(-1)), model.parameters())
    model.eval()
    x_scaled = step_scaler.transform(np.asarray(task["steps"], dtype=np.float32).reshape(-1, 1)).astype(np.float32)
    with torch.no_grad():
        pred_log = model(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy().reshape(-1, 1)
    return np.clip(inverse_freqs(pred_log).reshape(-1), a_min=0.0, a_max=None)


def proposed_summary(model_index: int, run_dir: Path, tasks: list[dict], pretrain_root: Path) -> dict:
    report = json.loads((run_dir / "target_transfer_report.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "target_task_metrics.csv")
    params = np.load(run_dir / "target_finetuned_params.npy")
    step_scaler = load_pickle(run_dir / "step_scaler.pkl")
    mlp_regressor = load_mlp_regressor(pretrain_root)
    shape_rows = []
    for task_index, task in enumerate(tasks):
        pred = reconstruct_proposed_curve(task, params[task_index], step_scaler, mlp_regressor)
        truth = np.asarray(task["freqs"], dtype=np.float32)
        steps = np.asarray(task["steps"], dtype=np.float32)
        shape_rows.append(curve_shape_stats(steps, truth, pred))
    return {
        "model_index": int(model_index),
        "curve_family": f"P{model_index - 1}",
        "label": STATE_LABELS[model_index],
        "method": "proposed_validation_selected",
        "task_count": int(len(metrics)),
        "point_r2": float(report["test"]["point_level"]["r2"]),
        "task_mean_r2": float(report["test"]["task_level"]["mean_r2"]),
        "task_median_r2": float(report["test"]["task_level"]["median_r2"]),
        "point_rmse": float(report["test"]["point_level"]["rmse"]),
        "point_mae": float(report["test"]["point_level"]["mae"]),
        "frac_test_r2_gt_0_9": float((metrics["test_r2"] > 0.9).mean()),
        "frac_test_r2_lt_0": float((metrics["test_r2"] < 0.0).mean()),
        "p05_test_r2": float(metrics["test_r2"].quantile(0.05)),
        "mean_test_rmse": float(metrics["test_rmse"].mean()),
        "mean_test_mae": float(metrics["test_mae"].mean()),
        "mean_abs_peak_step_error": float(np.mean([row["abs_peak_step_error"] for row in shape_rows])),
        "mean_tail_rmse_pct_peak": float(np.mean([row["tail_rmse_pct_peak"] for row in shape_rows])),
    }


def resolve_run_dir(result_root: Path, final_summary: pd.DataFrame, model_index: int) -> Path:
    candidate_name = str(final_summary.loc[final_summary["model_index"] == model_index, "candidate_name"].iloc[0])
    return result_root / f"model_{model_index}" / "full" / candidate_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-family reviewer-requested classical baselines.")
    parser.add_argument("--pretrain-root", default=str(DEFAULT_PRETRAIN_ROOT))
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--models", default="2,3,4,5,6,7,8")
    parser.add_argument("--methods", default="gaussian,spline")
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root).resolve()
    result_root = pretrain_root / "artifacts" / "curve_transfer_batch"
    final_summary = pd.read_csv(result_root / "final_summary.csv")
    model_indices = [int(part.strip()) for part in args.models.split(",") if part.strip()]
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    SOURCE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    task_rows = []
    for model_index in model_indices:
        run_dir = resolve_run_dir(result_root, final_summary, model_index)
        tasks = load_pickle(run_dir / "target_tasks.pkl")
        print(f"model_{model_index}: loaded {len(tasks)} tasks from {run_dir}")
        summary_rows.append(proposed_summary(model_index, run_dir, tasks, pretrain_root))
        for method in methods:
            payloads = [(task, method) for task in tasks]
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                rows = list(executor.map(evaluate_classical_task, payloads, chunksize=64))
            method_summary = aggregate_rows(model_index, method, rows)
            summary_rows.append(method_summary)
            for row in rows:
                task_rows.append(
                    {
                        "model_index": int(model_index),
                        "curve_family": f"P{model_index - 1}",
                        "label": STATE_LABELS[model_index],
                        "method": method,
                        "task_id": row["task_id"],
                        "WL": row["WL"],
                        "Retention": row["Retention"],
                        "PEC": row["PEC"],
                        "test_r2": row["test_r2"],
                        "test_rmse": row["test_rmse"],
                        "test_mae": row["test_mae"],
                        "abs_peak_step_error": row["abs_peak_step_error"],
                        "tail_rmse_pct_peak": row["tail_rmse_pct_peak"],
                    }
                )
            print(json.dumps(method_summary, ensure_ascii=False, indent=2))

    summary_df = pd.DataFrame(summary_rows).sort_values(["model_index", "method"]).reset_index(drop=True)
    task_df = pd.DataFrame(task_rows).sort_values(["model_index", "method", "task_id"]).reset_index(drop=True)
    summary_path = SOURCE_DATA_ROOT / "task2_full_family_baselines.csv"
    task_path = SOURCE_DATA_ROOT / "task2_full_family_baseline_task_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    task_df.to_csv(task_path, index=False)
    print(f"wrote {summary_path}")
    print(f"wrote {task_path}")


if __name__ == "__main__":
    main()
