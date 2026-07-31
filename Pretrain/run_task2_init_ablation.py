import argparse
import json
from pathlib import Path

import pandas as pd

from curve_task_workflow import finetune_target_domain_models
from main import load_local_config


PRETRAIN_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = PRETRAIN_ROOT / "artifacts" / "curve_transfer_batch"


def load_final_summary():
    return pd.read_csv(RESULT_ROOT / "final_summary.csv")


def resolve_full_candidate(model_index):
    summary = load_final_summary()
    row = summary.loc[summary["model_index"] == model_index]
    if row.empty:
        raise ValueError("Missing final summary row for model_{}".format(model_index))
    return str(row.iloc[0]["candidate_name"])


def resolve_run_directory(model_index, run_kind):
    model_dir = RESULT_ROOT / "model_{}".format(model_index) / run_kind
    if run_kind == "full":
        candidate_name = resolve_full_candidate(model_index)
        run_dir = model_dir / candidate_name
        if not run_dir.exists():
            raise FileNotFoundError("Missing full run directory: {}".format(run_dir))
        return run_dir

    matches = sorted(model_dir.glob("*"))
    matches = [path for path in matches if path.is_dir()]
    if not matches:
        raise FileNotFoundError("Missing sample run directory under {}".format(model_dir))
    return matches[0]


def resolve_subset_file(run_dir, pattern):
    matches = sorted((run_dir / "subsets").glob(pattern))
    if not matches:
        raise FileNotFoundError("Missing subset file pattern {} under {}".format(pattern, run_dir / "subsets"))
    return matches[0]


def build_config(model_index, run_kind, strategy, output_tag):
    template_name = "config_curve_transfer_sample.yaml" if run_kind == "sample" else "config_curve_transfer.yaml"
    config = load_local_config(template_name)
    config["_config_dir"] = str(PRETRAIN_ROOT)

    run_dir = resolve_run_directory(model_index, run_kind)
    data_config = config["data"]
    train_config = config["train"]

    data_config["csv_path"] = str((PRETRAIN_ROOT.parent / "Data" / "split_by_model" / "model_{}.csv".format(model_index)).resolve())
    data_config["artifacts_dir"] = str(run_dir)
    data_config["step_scaler_path"] = str(run_dir / "step_scaler.pkl")
    data_config["retention_scaler_path"] = str(run_dir / "retention_scaler.pkl")
    data_config["pec_scaler_path"] = str(run_dir / "pec_scaler.pkl")
    data_config["wl_vocab_path"] = str(run_dir / "wl_vocab.json")

    if run_kind == "sample":
        data_config["source_task_bundle_path"] = str(resolve_subset_file(run_dir, "source_tasks_*.pkl"))
        data_config["target_task_bundle_path"] = str(resolve_subset_file(run_dir, "target_tasks_*.pkl"))
        data_config["source_condition_csv"] = str(resolve_subset_file(run_dir, "source_conditions_*.csv"))
        data_config["target_condition_csv"] = str(resolve_subset_file(run_dir, "target_conditions_*.csv"))
    else:
        data_config["source_task_bundle_path"] = str(run_dir / "source_tasks.pkl")
        data_config["target_task_bundle_path"] = str(run_dir / "target_tasks.pkl")
        data_config["source_condition_csv"] = str(run_dir / "source_conditions.csv")
        data_config["target_condition_csv"] = str(run_dir / "target_conditions.csv")

    output_dir = run_dir / "ablation" / "{}_{}".format(strategy, output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config["target_init_strategy"] = str(strategy)
    if strategy == "scratch_only":
        train_config["target_scratch_retries"] = max(1, int(train_config.get("target_scratch_retries", 1)))
    train_config["target_finetuned_params_path"] = str(output_dir / "target_finetuned_params.npy")
    train_config["target_metrics_path"] = str(output_dir / "target_task_metrics.csv")
    train_config["target_report_path"] = str(output_dir / "target_transfer_report.json")
    train_config["source_parameter_vector_path"] = str(run_dir / "source_params.npy")
    train_config["source_metrics_path"] = str(run_dir / "source_task_metrics.csv")
    train_config["source_summary_path"] = str(run_dir / "source_summary.json")

    diffusion_matches = sorted(run_dir.glob("sampleSeq_RealParams_*.npy"))
    if not diffusion_matches:
        raise FileNotFoundError("Missing diffusion samples under {}".format(run_dir))

    return config, run_dir, output_dir, diffusion_matches[0]


def summarize_results(metrics_path, report_path, output_dir, model_index, run_kind, strategy):
    metrics = pd.read_csv(metrics_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    summary = {
        "model_index": int(model_index),
        "curve_family": "P{}".format(model_index - 1),
        "run_kind": run_kind,
        "strategy": strategy,
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
    }
    summary_path = output_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run a lightweight task2 initialization ablation on existing artifacts.")
    parser.add_argument("--model_index", type=int, default=3)
    parser.add_argument("--run_kind", choices=["sample", "full"], default="sample")
    parser.add_argument(
        "--strategy",
        choices=["validation_selected", "generated_only", "scratch_only"],
        required=True,
    )
    parser.add_argument("--output_tag", default="pilot")
    args = parser.parse_args()

    config, run_dir, output_dir, diffusion_sample_path = build_config(
        model_index=args.model_index,
        run_kind=args.run_kind,
        strategy=args.strategy,
        output_tag=args.output_tag,
    )

    print("run_dir:", run_dir)
    print("output_dir:", output_dir)
    print("diffusion_sample_path:", diffusion_sample_path)
    finetune_target_domain_models(config, diffusion_sample_path)
    summarize_results(
        metrics_path=Path(config["train"]["target_metrics_path"]),
        report_path=Path(config["train"]["target_report_path"]),
        output_dir=output_dir,
        model_index=args.model_index,
        run_kind=args.run_kind,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()
