from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_task2_reviewer_baselines import (
    DEFAULT_PRETRAIN_ROOT,
    SOURCE_DATA_ROOT,
    STATE_LABELS,
    curve_shape_stats,
    gaussian_predict,
    load_mlp_regressor,
    load_pickle,
    reconstruct_proposed_curve,
    regression_stats,
    sort_xy,
    spline_predict,
)


def resolve_run_dir(result_root: Path, final_summary: pd.DataFrame, model_index: int) -> Path:
    candidate = str(final_summary.loc[final_summary["model_index"] == model_index, "candidate_name"].iloc[0])
    return result_root / f"model_{model_index}" / "full" / candidate


def candidate_predictions(task: dict, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.asarray(task["steps"], dtype=np.float64)
    freqs = np.asarray(task["freqs"], dtype=np.float64)
    train_idx = np.asarray(task["splits"]["train"], dtype=np.int64)
    val_idx = np.asarray(task["splits"]["val"], dtype=np.int64)
    observed_idx = np.asarray(task["splits"]["train"] + task["splits"]["val"], dtype=np.int64)

    train_x, train_y = sort_xy(steps[train_idx], freqs[train_idx])
    observed_x, observed_y = sort_xy(steps[observed_idx], freqs[observed_idx])

    if method == "gaussian":
        val_pred = gaussian_predict(train_x, train_y, steps[val_idx])
        test_full_pred = gaussian_predict(observed_x, observed_y, steps)
    elif method == "spline":
        val_pred = spline_predict(train_x, train_y, steps[val_idx])
        test_full_pred = spline_predict(observed_x, observed_y, steps)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return val_pred, test_full_pred, freqs


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def aggregate_selector(model_index: int, method: str, rows: list[dict]) -> dict:
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


def evaluate_model(model_index: int, run_dir: Path, pretrain_root: Path) -> tuple[list[dict], list[dict]]:
    tasks = load_pickle(run_dir / "target_tasks.pkl")
    params = np.load(run_dir / "target_finetuned_params.npy")
    step_scaler = load_pickle(run_dir / "step_scaler.pkl")
    mlp_regressor = load_mlp_regressor(pretrain_root)

    selector_rows = {"hybrid_classical_val_selected": [], "hybrid_all_val_selected": []}
    task_rows = []
    selection_counts = {key: {} for key in selector_rows}

    for task_index, task in enumerate(tasks):
        steps = np.asarray(task["steps"], dtype=np.float64)
        freqs = np.asarray(task["freqs"], dtype=np.float64)
        val_idx = np.asarray(task["splits"]["val"], dtype=np.int64)
        test_idx = np.asarray(task["splits"]["test"], dtype=np.int64)

        candidate_full_predictions = {}
        candidate_val_scores = {}

        for method in ("gaussian", "spline"):
            val_pred, full_pred, _ = candidate_predictions(task, method)
            candidate_full_predictions[method] = full_pred
            candidate_val_scores[method] = mse(freqs[val_idx], val_pred)

        proposed_full_pred = reconstruct_proposed_curve(task, params[task_index], step_scaler, mlp_regressor)
        candidate_full_predictions["proposed_validation_selected"] = proposed_full_pred
        candidate_val_scores["proposed_validation_selected"] = mse(freqs[val_idx], proposed_full_pred[val_idx])

        selections = {
            "hybrid_classical_val_selected": min(("gaussian", "spline"), key=lambda name: candidate_val_scores[name]),
            "hybrid_all_val_selected": min(candidate_val_scores, key=candidate_val_scores.get),
        }

        for selector_name, selected_method in selections.items():
            selection_counts[selector_name][selected_method] = selection_counts[selector_name].get(selected_method, 0) + 1
            full_pred = np.clip(candidate_full_predictions[selected_method], a_min=0.0, a_max=None)
            test_stats = regression_stats(freqs[test_idx], full_pred[test_idx])
            shape_stats = curve_shape_stats(steps, freqs, full_pred)
            row = {
                "task_id": int(task["task_id"]),
                "WL": int(task["WL"]),
                "Retention": int(task["Retention"]),
                "PEC": int(task["PEC"]),
                "method": selector_name,
                "selected_method": selected_method,
                "test_r2": test_stats["r2"],
                "test_rmse": test_stats["rmse"],
                "test_mae": test_stats["mae"],
                **{key: test_stats[key] for key in ("n", "sse", "mae_sum", "sum_y", "sum_y2")},
                **shape_stats,
            }
            selector_rows[selector_name].append(row)
            task_rows.append(
                {
                    "model_index": int(model_index),
                    "curve_family": f"P{model_index - 1}",
                    "label": STATE_LABELS[model_index],
                    "method": selector_name,
                    "selected_method": selected_method,
                    "task_id": int(task["task_id"]),
                    "WL": int(task["WL"]),
                    "Retention": int(task["Retention"]),
                    "PEC": int(task["PEC"]),
                    "test_r2": test_stats["r2"],
                    "test_rmse": test_stats["rmse"],
                    "test_mae": test_stats["mae"],
                    **shape_stats,
                }
            )

    summary_rows = []
    for selector_name, rows in selector_rows.items():
        summary = aggregate_selector(model_index, selector_name, rows)
        for candidate in ("gaussian", "spline", "proposed_validation_selected"):
            summary[f"selected_{candidate}_frac"] = selection_counts[selector_name].get(candidate, 0) / max(len(tasks), 1)
        summary_rows.append(summary)
    return summary_rows, task_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation-selected hybrid classical/proposed policies.")
    parser.add_argument("--pretrain-root", default=str(DEFAULT_PRETRAIN_ROOT))
    parser.add_argument("--models", default="2,3,4,5,6,7,8")
    parser.add_argument("--output", default="task2_hybrid_selector.csv")
    parser.add_argument("--task-output", default="task2_hybrid_selector_task_metrics.csv")
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root).resolve()
    result_root = pretrain_root / "artifacts" / "curve_transfer_batch"
    final_summary = pd.read_csv(result_root / "final_summary.csv")
    model_indices = [int(part.strip()) for part in args.models.split(",") if part.strip()]

    summary_rows = []
    task_rows = []
    for model_index in model_indices:
        run_dir = resolve_run_dir(result_root, final_summary, model_index)
        print(f"model_{model_index}: {run_dir}")
        model_summary, model_tasks = evaluate_model(model_index, run_dir, pretrain_root)
        summary_rows.extend(model_summary)
        task_rows.extend(model_tasks)

    SOURCE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = SOURCE_DATA_ROOT / args.output
    task_path = SOURCE_DATA_ROOT / args.task_output
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(task_rows).to_csv(task_path, index=False)
    print(f"wrote {summary_path}")
    print(f"wrote {task_path}")


if __name__ == "__main__":
    main()
