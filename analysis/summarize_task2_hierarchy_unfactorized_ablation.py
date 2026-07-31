from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_task2_condition_ablation import task_bundle_paths
from run_task2_reviewer_baselines import (
    DEFAULT_PRETRAIN_ROOT,
    SOURCE_DATA_ROOT,
    curve_shape_stats,
    load_mlp_regressor,
    load_pickle,
    reconstruct_proposed_curve,
)


METHOD_ORDER = [
    "full_proposed",
    "remove_pec",
    "remove_wl",
    "remove_retention",
    "shuffle_pec",
    "source_pooled_mlp",
    "nearest_parameter_transfer",
    "hypernetwork_parameter",
    "unfactorized_mlp",
    "scratch_only",
]

METHOD_LABELS = {
    "full_proposed": "Full proposed",
    "remove_pec": "Remove PEC",
    "remove_wl": "Remove WL",
    "remove_retention": "Remove retention",
    "shuffle_pec": "Shuffle PEC",
    "source_pooled_mlp": "Source-pretrained MLP",
    "nearest_parameter_transfer": "Nearest transfer",
    "hypernetwork_parameter": "Hypernetwork",
    "unfactorized_mlp": "Unfactorized MLP",
    "scratch_only": "Scratch-only",
}


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


def find_run_dir(artifact_dir: Path) -> Path:
    for candidate in [artifact_dir, *artifact_dir.parents]:
        if (candidate / "step_scaler.pkl").exists():
            return candidate
    raise FileNotFoundError(f"Could not find run directory above {artifact_dir}")


def parameter_shape_metrics(row: pd.Series, pretrain_root: Path) -> dict:
    artifact_dir = Path(str(row["artifact_dir"]))
    run_dir = find_run_dir(artifact_dir)
    target_task_path = task_bundle_paths(run_dir, str(row.get("run_kind", "sample")))[1]
    tasks = load_pickle(target_task_path)
    params = np.load(artifact_dir / "target_finetuned_params.npy")
    if params.ndim == 3:
        params = params.reshape(params.shape[0], -1)
    step_scaler = load_pickle(run_dir / "step_scaler.pkl")
    mlp_regressor = load_mlp_regressor(pretrain_root)

    peak_errors = []
    tail_rmse = []
    tail_kl = []
    for task_index, task in enumerate(tasks):
        prediction = reconstruct_proposed_curve(task, params[task_index], step_scaler, mlp_regressor)
        truth = np.asarray(task["freqs"], dtype=np.float64)
        steps = np.asarray(task["steps"], dtype=np.float64)
        shape = curve_shape_stats(steps, truth, prediction)
        peak_errors.append(shape["abs_peak_step_error"])
        tail_rmse.append(shape["tail_rmse_pct_peak"])
        tail_kl.append(symmetric_tail_kl(truth, prediction))
    return {
        "mean_abs_peak_step_error": float(np.mean(peak_errors)),
        "mean_tail_rmse_pct_peak": float(np.mean(tail_rmse)),
        "mean_tail_symmetric_kl": float(np.mean(tail_kl)),
    }


def condition_rows(path: Path, pretrain_root: Path) -> list[dict]:
    frame = pd.read_csv(path)
    mapping = {
        "Full": "full_proposed",
        "-PEC": "remove_pec",
        "-WL": "remove_wl",
        "-Retention": "remove_retention",
        "PEC shuffled": "shuffle_pec",
    }
    rows = []
    for _, row in frame.loc[frame["model"].isin(mapping)].iterrows():
        method = mapping[str(row["model"])]
        shape = parameter_shape_metrics(row, pretrain_root)
        rows.append(
            {
                "model_index": int(row["model_index"]),
                "label": row["label"],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "point_r2": float(row["point_r2"]),
                "task_mean_r2": float(row["task_mean_r2"]),
                "task_median_r2": float(row["task_median_r2"]),
                "p05_test_r2": float(row["p05_test_r2"]),
                "frac_test_r2_gt_0_9": float(row["frac_test_r2_gt_0_9"]),
                "frac_test_r2_lt_0": float(row["frac_test_r2_lt_0"]),
                **shape,
            }
        )
    return rows


def transfer_rows(path: Path, pretrain_root: Path) -> list[dict]:
    frame = pd.read_csv(path)
    keep = {"source_pooled_mlp", "nearest_parameter_transfer", "hypernetwork_parameter", "scratch_only"}
    rows = []
    for _, row in frame.loc[frame["method"].isin(keep)].iterrows():
        method = str(row["method"])
        shape = parameter_shape_metrics(row, pretrain_root)
        rows.append(
            {
                "model_index": int(row["model_index"]),
                "label": row["label"],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "point_r2": float(row["point_r2"]),
                "task_mean_r2": float(row["task_mean_r2"]),
                "task_median_r2": float(row["task_median_r2"]),
                "p05_test_r2": float(row["p05_test_r2"]),
                "frac_test_r2_gt_0_9": float(row["frac_test_r2_gt_0_9"]),
                "frac_test_r2_lt_0": float(row["frac_test_r2_lt_0"]),
                **shape,
            }
        )
    return rows


def unfactorized_rows(summary_path: Path, task_path: Path) -> list[dict]:
    summary = pd.read_csv(summary_path)
    task_metrics = pd.read_csv(task_path)
    rows = []
    for _, row in summary.iterrows():
        family_tasks = task_metrics.loc[task_metrics["model_index"] == int(row["model_index"])]
        method = "unfactorized_mlp"
        rows.append(
            {
                "model_index": int(row["model_index"]),
                "label": row["label"],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "point_r2": float(row["point_r2"]),
                "task_mean_r2": float(row["task_mean_r2"]),
                "task_median_r2": float(row["task_median_r2"]),
                "p05_test_r2": float(row["p05_test_r2"]),
                "frac_test_r2_gt_0_9": float(row["frac_test_r2_gt_0_9"]),
                "frac_test_r2_lt_0": float(row["frac_test_r2_lt_0"]),
                "mean_abs_peak_step_error": float(row["mean_abs_peak_step_error"]),
                "mean_tail_rmse_pct_peak": float(row["mean_tail_rmse_pct_peak"]),
                "mean_tail_symmetric_kl": float(family_tasks["tail_symmetric_kl"].mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a representative hierarchy/unfactorized ablation table.")
    parser.add_argument("--pretrain-root", default=str(DEFAULT_PRETRAIN_ROOT))
    parser.add_argument("--condition-input", default="task2_condition_ablation_representative.csv")
    parser.add_argument("--transfer-input", default="task2_diffusion_necessity_representative.csv")
    parser.add_argument("--unfactorized-input", default="task2_unfactorized_mlp_representative.csv")
    parser.add_argument("--unfactorized-task-input", default="task2_unfactorized_mlp_representative_task_metrics.csv")
    parser.add_argument("--output-detail", default="task2_hierarchy_unfactorized_ablation_representative.csv")
    parser.add_argument("--output-summary", default="task2_hierarchy_unfactorized_ablation_representative_summary.csv")
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root).resolve()
    rows = []
    rows.extend(condition_rows(SOURCE_DATA_ROOT / args.condition_input, pretrain_root))
    rows.extend(transfer_rows(SOURCE_DATA_ROOT / args.transfer_input, pretrain_root))
    rows.extend(unfactorized_rows(SOURCE_DATA_ROOT / args.unfactorized_input, SOURCE_DATA_ROOT / args.unfactorized_task_input))

    detail = pd.DataFrame(rows)
    detail["order"] = detail["method"].map({method: index for index, method in enumerate(METHOD_ORDER)})
    detail = detail.sort_values(["order", "model_index"]).reset_index(drop=True)
    detail_path = SOURCE_DATA_ROOT / args.output_detail
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby(["order", "method", "method_label"], as_index=False)
        .agg(
            point_r2=("point_r2", "mean"),
            task_mean_r2=("task_mean_r2", "mean"),
            task_median_r2=("task_median_r2", "mean"),
            p05_test_r2=("p05_test_r2", "mean"),
            frac_test_r2_gt_0_9=("frac_test_r2_gt_0_9", "mean"),
            frac_test_r2_lt_0=("frac_test_r2_lt_0", "mean"),
            mean_abs_peak_step_error=("mean_abs_peak_step_error", "mean"),
            mean_tail_rmse_pct_peak=("mean_tail_rmse_pct_peak", "mean"),
            mean_tail_symmetric_kl=("mean_tail_symmetric_kl", "mean"),
        )
        .sort_values("order")
    )
    family_task = detail.pivot(index="method", columns="label", values="task_mean_r2").reset_index()
    summary = summary.merge(family_task, on="method", how="left")
    summary_path = SOURCE_DATA_ROOT / args.output_summary
    summary.to_csv(summary_path, index=False)
    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
