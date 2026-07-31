import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from curve_task_workflow import finetune_target_domain_models
from main import load_local_config


PRETRAIN_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = PRETRAIN_ROOT / "artifacts" / "curve_transfer_batch"


def load_final_summary():
    return pd.read_csv(RESULT_ROOT / "final_summary.csv")


def resolve_run_directory(model_index):
    model_dir = RESULT_ROOT / f"model_{model_index}" / "sample"
    matches = sorted(path for path in model_dir.glob("*") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"Missing sample run directory under {model_dir}")
    return matches[0]


def resolve_subset_file(run_dir, pattern):
    matches = sorted((run_dir / "subsets").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing subset file pattern {pattern} under {run_dir / 'subsets'}")
    return matches[0]


def load_pickle(path_like):
    with Path(path_like).open("rb") as handle:
        return pickle.load(handle)


def save_pickle(obj, path_like):
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def select_evenly_spaced_indices(length, count):
    count = int(count)
    if count <= 0:
        return np.empty((0,), dtype=np.int64)

    candidates = np.arange(length, dtype=np.int64)
    if count >= len(candidates):
        return candidates.copy()

    raw_positions = np.linspace(0, len(candidates) - 1, num=count)
    selected = []
    used = set()
    for position in raw_positions:
        idx = int(round(position))
        idx = max(0, min(idx, len(candidates) - 1))
        if idx not in used:
            selected.append(idx)
            used.add(idx)

    if len(selected) < count:
        for idx in candidates.tolist():
            if idx in used:
                continue
            selected.append(idx)
            used.add(idx)
            if len(selected) == count:
                break

    return np.asarray(sorted(selected), dtype=np.int64)


def gaussian_curve(x, amplitude, mean, sigma, baseline):
    sigma = max(float(sigma), 1e-3)
    return baseline + amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def gaussian_predict(train_x, train_y, eval_x):
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    eval_x = np.asarray(eval_x, dtype=np.float64)

    weights = np.clip(train_y, 1.0, None)
    amplitude0 = max(float(train_y.max() - train_y.min()), 1.0)
    mean0 = float(np.average(train_x, weights=weights))
    sigma0 = max(float(np.sqrt(np.average((train_x - mean0) ** 2, weights=weights))), 1.0)
    baseline0 = max(float(train_y.min()), 0.0)

    bounds = (
        [0.0, float(train_x.min()) - 10.0, 0.3, 0.0],
        [float(train_y.max()) * 3.0 + 1.0, float(train_x.max()) + 10.0, 100.0, float(train_y.max()) * 2.0 + 1.0],
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


def spline_predict(train_x, train_y, eval_x):
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    eval_x = np.asarray(eval_x, dtype=np.float64)

    if len(train_x) >= 3:
        spline = CubicSpline(train_x, train_y, bc_type="natural", extrapolate=True)
        pred = spline(eval_x)
    else:
        pred = np.interp(eval_x, train_x, train_y)
    return np.clip(np.asarray(pred, dtype=np.float64), a_min=0.0, a_max=None)


def regression_summary(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else None
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2) if r2 is not None else None,
        "num_points": int(len(y_true)),
    }


def aggregate_task_metrics(task_rows):
    y_true_all = []
    y_pred_all = []
    for row in task_rows:
        y_true_all.extend(row["y_true"])
        y_pred_all.extend(row["y_pred"])

    point_metrics = regression_summary(np.asarray(y_true_all, dtype=np.float64), np.asarray(y_pred_all, dtype=np.float64))
    task_r2 = [row["test_r2"] for row in task_rows if row["test_r2"] is not None]
    return {
        "point_r2": float(point_metrics["r2"]),
        "point_rmse": float(point_metrics["rmse"]),
        "point_mae": float(point_metrics["mae"]),
        "task_mean_r2": float(np.mean(task_r2)) if task_r2 else None,
        "task_median_r2": float(np.median(task_r2)) if task_r2 else None,
        "frac_test_r2_gt_0_9": float(np.mean(np.asarray(task_r2) > 0.9)) if task_r2 else None,
        "frac_test_r2_lt_0": float(np.mean(np.asarray(task_r2) < 0.0)) if task_r2 else None,
        "mean_test_rmse": float(np.mean([row["test_rmse"] for row in task_rows])),
        "mean_test_mae": float(np.mean([row["test_mae"] for row in task_rows])),
    }


def build_sparse_task_bundle(tasks, observed_points):
    modified = []
    for task in tasks:
        task_copy = copy.deepcopy(task)
        task_length = int(task_copy["num_points"])
        train_count = min(int(observed_points), max(2, task_length - 2))
        train_indices = select_evenly_spaced_indices(task_length, train_count)
        test_indices = np.asarray([idx for idx in range(task_length) if idx not in set(train_indices.tolist())], dtype=np.int64)
        task_copy["splits"] = {
            "train": train_indices.tolist(),
            "val": [],
            "test": test_indices.tolist(),
        }
        modified.append(task_copy)
    return modified


def evaluate_classical_baselines(tasks):
    baseline_rows = []
    for method_name in ("spline", "gaussian"):
        task_rows = []
        for task in tasks:
            steps = np.asarray(task["steps"], dtype=np.float64)
            freqs = np.asarray(task["freqs"], dtype=np.float64)
            train_idx = np.asarray(task["splits"]["train"], dtype=np.int64)
            test_idx = np.asarray(task["splits"]["test"], dtype=np.int64)

            train_x = steps[train_idx]
            train_y = freqs[train_idx]
            test_x = steps[test_idx]
            test_y = freqs[test_idx]

            if method_name == "spline":
                pred = spline_predict(train_x, train_y, test_x)
            else:
                pred = gaussian_predict(train_x, train_y, test_x)

            summary = regression_summary(test_y, pred)
            task_rows.append(
                {
                    "task_id": int(task["task_id"]),
                    "method": method_name,
                    "WL": int(task["WL"]),
                    "Retention": int(task["Retention"]),
                    "PEC": int(task["PEC"]),
                    "test_r2": summary["r2"],
                    "test_rmse": summary["rmse"],
                    "test_mae": summary["mae"],
                    "y_true": test_y.tolist(),
                    "y_pred": pred.tolist(),
                }
            )

        aggregate = aggregate_task_metrics(task_rows)
        aggregate["method"] = method_name
        aggregate["task_rows"] = task_rows
        baseline_rows.append(aggregate)
    return baseline_rows


def build_config(model_index, observed_points, output_tag):
    config = load_local_config("config_curve_transfer_sample.yaml")
    config["_config_dir"] = str(PRETRAIN_ROOT)

    run_dir = resolve_run_directory(model_index)
    data_config = config["data"]
    train_config = config["train"]

    data_config["csv_path"] = str((PRETRAIN_ROOT.parent / "Data" / "split_by_model" / f"model_{model_index}.csv").resolve())
    data_config["artifacts_dir"] = str(run_dir)
    data_config["step_scaler_path"] = str(run_dir / "step_scaler.pkl")
    data_config["retention_scaler_path"] = str(run_dir / "retention_scaler.pkl")
    data_config["pec_scaler_path"] = str(run_dir / "pec_scaler.pkl")
    data_config["wl_vocab_path"] = str(run_dir / "wl_vocab.json")
    data_config["source_task_bundle_path"] = str(resolve_subset_file(run_dir, "source_tasks_*.pkl"))
    data_config["source_condition_csv"] = str(resolve_subset_file(run_dir, "source_conditions_*.csv"))
    data_config["target_condition_csv"] = str(resolve_subset_file(run_dir, "target_conditions_*.csv"))

    output_dir = run_dir / "sparsity" / f"{output_tag}_{observed_points:02d}pts"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_config["target_task_bundle_path"] = str(output_dir / f"target_tasks_obs{observed_points:02d}.pkl")

    train_config["target_init_strategy"] = "generated_only"
    train_config["target_scratch_retries"] = 0
    train_config["target_finetuned_params_path"] = str(output_dir / "target_finetuned_params.npy")
    train_config["target_metrics_path"] = str(output_dir / "target_task_metrics.csv")
    train_config["target_report_path"] = str(output_dir / "target_transfer_report.json")
    train_config["source_parameter_vector_path"] = str(run_dir / "source_params.npy")
    train_config["source_metrics_path"] = str(run_dir / "source_task_metrics.csv")
    train_config["source_summary_path"] = str(run_dir / "source_summary.json")

    diffusion_matches = sorted(run_dir.glob("sampleSeq_RealParams_*.npy"))
    if not diffusion_matches:
        raise FileNotFoundError(f"Missing diffusion samples under {run_dir}")

    return config, run_dir, output_dir, diffusion_matches[0]


def summarize_ours(metrics_path, report_path, observed_points, model_index):
    metrics = pd.read_csv(metrics_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return {
        "model_index": int(model_index),
        "curve_family": f"P{model_index - 1}",
        "method": "physics_guided_diffusion",
        "observed_points": int(observed_points),
        "task_count": int(len(metrics)),
        "point_r2": float(report["test"]["point_level"]["r2"]),
        "task_mean_r2": float(report["test"]["task_level"]["mean_r2"]),
        "task_median_r2": float(report["test"]["task_level"]["median_r2"]),
        "point_rmse": float(report["test"]["point_level"]["rmse"]),
        "point_mae": float(report["test"]["point_level"]["mae"]),
        "frac_test_r2_gt_0_9": float((metrics["test_r2"] > 0.9).mean()),
        "frac_test_r2_lt_0": float((metrics["test_r2"] < 0.0).mean()),
        "mean_test_rmse": float(metrics["test_rmse"].mean()),
        "mean_test_mae": float(metrics["test_mae"].mean()),
    }


def write_per_task_rows(rows, output_path):
    if not rows:
        return
    serializable = []
    for row in rows:
        serializable.append(
            {
                "task_id": row["task_id"],
                "method": row["method"],
                "WL": row["WL"],
                "Retention": row["Retention"],
                "PEC": row["PEC"],
                "test_r2": row["test_r2"],
                "test_rmse": row["test_rmse"],
                "test_mae": row["test_mae"],
            }
        )
    pd.DataFrame(serializable).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Benchmark task2 under extreme target sparsity on an existing sample run.")
    parser.add_argument("--model_index", type=int, default=3)
    parser.add_argument("--observed_points", default="3,5,7,9,11,13")
    parser.add_argument("--output_tag", default="reviewer_depth")
    args = parser.parse_args()

    observed_points_list = [int(part.strip()) for part in str(args.observed_points).split(",") if part.strip()]
    config, run_dir, _, diffusion_sample_path = build_config(
        model_index=args.model_index,
        observed_points=observed_points_list[0],
        output_tag=args.output_tag,
    )
    benchmark_root = run_dir / "sparsity" / args.output_tag
    benchmark_root.mkdir(parents=True, exist_ok=True)

    original_tasks = load_pickle(resolve_subset_file(run_dir, "target_tasks_*.pkl"))
    summary_rows = []
    per_task_rows = []

    for observed_points in observed_points_list:
        config, run_dir, output_dir, diffusion_sample_path = build_config(
            model_index=args.model_index,
            observed_points=observed_points,
            output_tag=args.output_tag,
        )
        sparse_tasks = build_sparse_task_bundle(original_tasks, observed_points)
        save_pickle(sparse_tasks, config["data"]["target_task_bundle_path"])

        finetune_target_domain_models(config, diffusion_sample_path)
        ours_summary = summarize_ours(
            metrics_path=Path(config["train"]["target_metrics_path"]),
            report_path=Path(config["train"]["target_report_path"]),
            observed_points=observed_points,
            model_index=args.model_index,
        )
        summary_rows.append(ours_summary)

        baseline_summaries = evaluate_classical_baselines(sparse_tasks)
        for baseline_summary in baseline_summaries:
            summary_rows.append(
                {
                    "model_index": int(args.model_index),
                    "curve_family": f"P{args.model_index - 1}",
                    "method": baseline_summary["method"],
                    "observed_points": int(observed_points),
                    "task_count": int(len(sparse_tasks)),
                    "point_r2": baseline_summary["point_r2"],
                    "task_mean_r2": baseline_summary["task_mean_r2"],
                    "task_median_r2": baseline_summary["task_median_r2"],
                    "point_rmse": baseline_summary["point_rmse"],
                    "point_mae": baseline_summary["point_mae"],
                    "frac_test_r2_gt_0_9": baseline_summary["frac_test_r2_gt_0_9"],
                    "frac_test_r2_lt_0": baseline_summary["frac_test_r2_lt_0"],
                    "mean_test_rmse": baseline_summary["mean_test_rmse"],
                    "mean_test_mae": baseline_summary["mean_test_mae"],
                }
            )
            for row in baseline_summary["task_rows"]:
                row["observed_points"] = int(observed_points)
                per_task_rows.append(row)

        ours_metrics = pd.read_csv(config["train"]["target_metrics_path"])
        ours_metrics = ours_metrics.assign(method="physics_guided_diffusion", observed_points=int(observed_points))
        per_task_rows.extend(
            [
                {
                    "task_id": int(row.task_id),
                    "method": "physics_guided_diffusion",
                    "WL": int(row.WL),
                    "Retention": int(row.Retention),
                    "PEC": int(row.PEC),
                    "test_r2": float(row.test_r2),
                    "test_rmse": float(row.test_rmse),
                    "test_mae": float(row.test_mae),
                    "observed_points": int(observed_points),
                }
                for row in ours_metrics.itertuples(index=False)
            ]
        )

        print(json.dumps(summary_rows[-3:], ensure_ascii=False, indent=2))

    summary_df = pd.DataFrame(summary_rows).sort_values(["observed_points", "method"]).reset_index(drop=True)
    summary_path = benchmark_root / "sparsity_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    write_per_task_rows(per_task_rows, benchmark_root / "sparsity_benchmark_task_metrics.csv")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
