from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_task2_reviewer_baselines import (
    DEFAULT_PRETRAIN_ROOT,
    SOURCE_DATA_ROOT,
    STATE_LABELS,
    curve_shape_stats,
    load_mlp_regressor,
    load_pickle,
    reconstruct_proposed_curve,
    regression_stats,
)


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


def condition_paths(run_dir: Path, run_kind: str) -> tuple[Path, Path]:
    if run_kind == "sample":
        return (
            resolve_subset_file(run_dir, "source_conditions_*.csv"),
            resolve_subset_file(run_dir, "target_conditions_*.csv"),
        )
    return run_dir / "source_conditions.csv", run_dir / "target_conditions.csv"


def task_bundle_paths(run_dir: Path, run_kind: str) -> tuple[Path, Path]:
    if run_kind == "sample":
        return (
            resolve_subset_file(run_dir, "source_tasks_*.pkl"),
            resolve_subset_file(run_dir, "target_tasks_*.pkl"),
        )
    return run_dir / "source_tasks.pkl", run_dir / "target_tasks.pkl"


def write_condition_ablation(
    ablation: str,
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> tuple[Path, Path]:
    source = source_frame.copy()
    target = target_frame.copy()
    pooled = pd.concat([source_frame, target_frame], ignore_index=True)

    if ablation in {"full", "full_rerun"}:
        pass
    elif ablation == "remove_pec":
        source["PEC"] = float(source_frame["PEC"].median())
        target["PEC"] = float(source_frame["PEC"].median())
    elif ablation == "remove_wl":
        source["WL"] = int(source_frame["WL"].mode().iloc[0])
        target["WL"] = int(source_frame["WL"].mode().iloc[0])
    elif ablation == "remove_retention":
        source["Retention"] = float(source_frame["Retention"].median())
        target["Retention"] = float(source_frame["Retention"].median())
    elif ablation == "shuffle_pec":
        rng = np.random.default_rng(seed)
        source["PEC"] = rng.permutation(source["PEC"].to_numpy())
        target["PEC"] = rng.permutation(target["PEC"].to_numpy())
    elif ablation == "shuffle_target_pec":
        rng = np.random.default_rng(seed)
        target["PEC"] = rng.permutation(target["PEC"].to_numpy())
    elif ablation == "remove_all_stress":
        source["WL"] = int(source_frame["WL"].mode().iloc[0])
        target["WL"] = int(source_frame["WL"].mode().iloc[0])
        source["Retention"] = float(source_frame["Retention"].median())
        target["Retention"] = float(source_frame["Retention"].median())
        source["PEC"] = float(source_frame["PEC"].median())
        target["PEC"] = float(source_frame["PEC"].median())
    elif ablation == "pooled_neutral":
        source["WL"] = int(pooled["WL"].mode().iloc[0])
        target["WL"] = int(pooled["WL"].mode().iloc[0])
        source["Retention"] = float(pooled["Retention"].median())
        target["Retention"] = float(pooled["Retention"].median())
        source["PEC"] = float(pooled["PEC"].median())
        target["PEC"] = float(pooled["PEC"].median())
    else:
        raise ValueError(f"Unsupported condition ablation: {ablation}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source_conditions.csv"
    target_path = output_dir / "target_conditions.csv"
    source.to_csv(source_path, index=False)
    target.to_csv(target_path, index=False)
    return source_path, target_path


def run_gpd(
    python_executable: str,
    pretrain_root: Path,
    run_dir: Path,
    source_condition_csv: Path,
    target_condition_csv: Path,
    exp_index: int,
    gpd_epochs: int,
    output_dir: Path,
) -> Path:
    gpd_dir = pretrain_root.parent / "GPD"
    command = [
        python_executable,
        "1Dmain.py",
        "--modeldim",
        "16",
        "--epochs",
        str(int(gpd_epochs)),
        "--expIndex",
        str(int(exp_index)),
        "--diffusionstep",
        "10",
        "--denoise",
        "Trans3",
        "--trainbatchsize",
        "1",
        "--samplebatchsize",
        "64",
        "--repeat_num",
        "1",
        "--train_param_path",
        str(run_dir / "source_params.npy"),
        "--train_condition_csv",
        str(source_condition_csv),
        "--sample_condition_csv",
        str(target_condition_csv),
        "--retention_scaler_path",
        str(run_dir / "retention_scaler.pkl"),
        "--pec_scaler_path",
        str(run_dir / "pec_scaler.pkl"),
        "--wl_vocab_path",
        str(run_dir / "wl_vocab.json"),
    ]
    subprocess.run(command, cwd=str(gpd_dir), check=True)
    sample_path = gpd_dir / "Output" / f"sampleSeq_RealParams_{exp_index}.npy"
    if not sample_path.exists():
        raise FileNotFoundError(f"GPD did not create {sample_path}")
    snapshot_path = output_dir / sample_path.name
    shutil.copy2(sample_path, snapshot_path)
    return snapshot_path


def build_finetune_config(
    pretrain_root: Path,
    run_dir: Path,
    run_kind: str,
    output_dir: Path,
    source_condition_csv: Path,
    target_condition_csv: Path,
    strategy: str,
    workers: int,
) -> dict:
    sys.path.insert(0, str(pretrain_root))
    from main import load_local_config  # noqa: WPS433

    template_name = "config_curve_transfer_sample.yaml" if run_kind == "sample" else "config_curve_transfer.yaml"
    config = load_local_config(template_name)
    config["_config_dir"] = str(pretrain_root)

    source_task_bundle_path, target_task_bundle_path = task_bundle_paths(run_dir, run_kind)
    data_config = config["data"]
    data_config["artifacts_dir"] = str(run_dir)
    data_config["source_task_bundle_path"] = str(source_task_bundle_path)
    data_config["target_task_bundle_path"] = str(target_task_bundle_path)
    data_config["source_condition_csv"] = str(source_condition_csv)
    data_config["target_condition_csv"] = str(target_condition_csv)
    data_config["step_scaler_path"] = str(run_dir / "step_scaler.pkl")
    data_config["retention_scaler_path"] = str(run_dir / "retention_scaler.pkl")
    data_config["pec_scaler_path"] = str(run_dir / "pec_scaler.pkl")
    data_config["wl_vocab_path"] = str(run_dir / "wl_vocab.json")

    train_config = config["train"]
    train_config["target_init_strategy"] = strategy
    train_config["target_scratch_retries"] = 0 if strategy == "generated_only" else int(train_config.get("target_scratch_retries", 1))
    train_config["parallel_workers"] = int(workers)
    train_config["source_parameter_vector_path"] = str(run_dir / "source_params.npy")
    train_config["source_metrics_path"] = str(run_dir / "source_task_metrics.csv")
    train_config["source_summary_path"] = str(run_dir / "source_summary.json")
    train_config["target_finetuned_params_path"] = str(output_dir / "target_finetuned_params.npy")
    train_config["target_metrics_path"] = str(output_dir / "target_task_metrics.csv")
    train_config["target_report_path"] = str(output_dir / "target_transfer_report.json")
    return config


def aggregate_param_predictions(
    model_index: int,
    ablation: str,
    tasks: list[dict],
    params: np.ndarray,
    run_dir: Path,
    pretrain_root: Path,
    prefix: str,
) -> dict:
    step_scaler = load_pickle(run_dir / "step_scaler.pkl")
    mlp_regressor = load_mlp_regressor(pretrain_root)
    rows = []
    shape_rows = []
    params = np.asarray(params)
    if params.ndim == 3:
        params = params.reshape(params.shape[0], -1)
    for task_index, task in enumerate(tasks):
        prediction = reconstruct_proposed_curve(task, params[task_index], step_scaler, mlp_regressor)
        test_indices = np.asarray(task["splits"]["test"], dtype=np.int64)
        truth = np.asarray(task["freqs"], dtype=np.float64)
        stats = regression_stats(truth[test_indices], prediction[test_indices])
        rows.append(stats)
        shape_rows.append(curve_shape_stats(np.asarray(task["steps"], dtype=np.float64), truth, prediction))

    total_n = int(sum(row["n"] for row in rows))
    total_sse = float(sum(row["sse"] for row in rows))
    total_sum_y = float(sum(row["sum_y"] for row in rows))
    total_sum_y2 = float(sum(row["sum_y2"] for row in rows))
    ss_tot = float(total_sum_y2 - (total_sum_y**2) / max(total_n, 1))
    task_r2 = np.asarray([row["r2"] for row in rows if row["r2"] is not None], dtype=np.float64)
    return {
        f"{prefix}_point_r2": float(1.0 - total_sse / ss_tot) if ss_tot > 0.0 else np.nan,
        f"{prefix}_task_mean_r2": float(np.mean(task_r2)),
        f"{prefix}_task_median_r2": float(np.median(task_r2)),
        f"{prefix}_point_rmse": float(np.sqrt(total_sse / max(total_n, 1))),
        f"{prefix}_frac_test_r2_gt_0_9": float(np.mean(task_r2 > 0.9)),
        f"{prefix}_frac_test_r2_lt_0": float(np.mean(task_r2 < 0.0)),
        f"{prefix}_mean_abs_peak_step_error": float(np.mean([row["abs_peak_step_error"] for row in shape_rows])),
        f"{prefix}_mean_tail_rmse_pct_peak": float(np.mean([row["tail_rmse_pct_peak"] for row in shape_rows])),
    }


def summarize_ablation(
    model_index: int,
    run_kind: str,
    ablation: str,
    output_dir: Path,
    run_dir: Path,
    pretrain_root: Path,
    diffusion_sample_path: Path,
) -> dict:
    report = json.loads((output_dir / "target_transfer_report.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(output_dir / "target_task_metrics.csv")
    target_task_path = task_bundle_paths(run_dir, run_kind)[1]
    tasks = load_pickle(target_task_path)
    generated_params = np.load(diffusion_sample_path)
    finetuned_params = np.load(output_dir / "target_finetuned_params.npy")
    row = {
        "model_index": int(model_index),
        "curve_family": f"P{model_index - 1}",
        "label": STATE_LABELS[model_index],
        "run_kind": run_kind,
        "ablation": ablation,
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
        "artifact_dir": str(output_dir),
    }
    row.update(aggregate_param_predictions(model_index, ablation, tasks, generated_params, run_dir, pretrain_root, "zero_shot"))
    row.update(aggregate_param_predictions(model_index, ablation, tasks, finetuned_params, run_dir, pretrain_root, "adapted_shape"))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 2 condition-removal and condition-shuffle ablations.")
    parser.add_argument("--pretrain-root", default=str(DEFAULT_PRETRAIN_ROOT))
    parser.add_argument("--models", default="3")
    parser.add_argument("--run-kind", choices=["sample", "full"], default="sample")
    parser.add_argument("--ablations", default="full,remove_pec,remove_wl,remove_retention,shuffle_pec")
    parser.add_argument("--strategy", choices=["generated_only", "validation_selected"], default="generated_only")
    parser.add_argument("--gpd-epochs", type=int, default=100)
    parser.add_argument("--exp-base", type=int, default=78100)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rerun-full", action="store_true")
    parser.add_argument("--output", default="task2_condition_ablation_model3.csv")
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root).resolve()
    result_root = pretrain_root / "artifacts" / "curve_transfer_batch"
    final_summary = pd.read_csv(result_root / "final_summary.csv")
    model_indices = [int(part.strip()) for part in args.models.split(",") if part.strip()]
    ablations = [part.strip() for part in args.ablations.split(",") if part.strip()]
    SOURCE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(pretrain_root))
    from curve_task_workflow import finetune_target_domain_models  # noqa: WPS433

    rows = []
    for model_index in model_indices:
        run_dir = (
            resolve_sample_run_dir(result_root, model_index)
            if args.run_kind == "sample"
            else resolve_full_run_dir(result_root, final_summary, model_index)
        )
        source_condition_path, target_condition_path = condition_paths(run_dir, args.run_kind)
        source_frame = pd.read_csv(source_condition_path)
        target_frame = pd.read_csv(target_condition_path)
        original_diffusion = sorted(run_dir.glob("sampleSeq_RealParams_*.npy"))[0]

        for ablation_index, ablation in enumerate(ablations, start=1):
            output_dir = run_dir / "condition_ablation" / f"{args.strategy}_{ablation}"
            output_dir.mkdir(parents=True, exist_ok=True)
            ablated_source_path, ablated_target_path = write_condition_ablation(
                ablation,
                source_frame,
                target_frame,
                output_dir,
                args.seed + model_index * 100 + ablation_index,
            )

            should_reuse_full = ablation == "full" and not args.rerun_full
            if should_reuse_full:
                diffusion_sample_path = original_diffusion
            else:
                diffusion_sample_path = output_dir / f"sampleSeq_RealParams_{args.exp_base + model_index * 100 + ablation_index}.npy"
                if args.force or not diffusion_sample_path.exists():
                    diffusion_sample_path = run_gpd(
                        python_executable=sys.executable,
                        pretrain_root=pretrain_root,
                        run_dir=run_dir,
                        source_condition_csv=ablated_source_path,
                        target_condition_csv=ablated_target_path,
                        exp_index=args.exp_base + model_index * 100 + ablation_index,
                        gpd_epochs=args.gpd_epochs,
                        output_dir=output_dir,
                    )

            report_path = output_dir / "target_transfer_report.json"
            if args.force or not report_path.exists():
                config = build_finetune_config(
                    pretrain_root=pretrain_root,
                    run_dir=run_dir,
                    run_kind=args.run_kind,
                    output_dir=output_dir,
                    source_condition_csv=ablated_source_path,
                    target_condition_csv=ablated_target_path,
                    strategy=args.strategy,
                    workers=args.workers,
                )
                finetune_target_domain_models(config, diffusion_sample_path)

            row = summarize_ablation(
                model_index=model_index,
                run_kind=args.run_kind,
                ablation=ablation,
                output_dir=output_dir,
                run_dir=run_dir,
                pretrain_root=pretrain_root,
                diffusion_sample_path=diffusion_sample_path,
            )
            row["gpd_epochs"] = int(args.gpd_epochs)
            row["strategy"] = args.strategy
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2))

    output_path = SOURCE_DATA_ROOT / args.output
    pd.DataFrame(rows).sort_values(["model_index", "ablation"]).to_csv(output_path, index=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
