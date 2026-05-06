import argparse
import json
import sys
from pathlib import Path

import numpy as np

from curve_task_workflow import run_curve_transfer_full_pipeline
from main import load_local_config


def resolve_path(base_dir, path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_json(path_like):
    with Path(path_like).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parameter_row_count(path_like):
    array = np.load(path_like)
    if array.ndim == 1:
        return 1, int(array.shape[0])
    return int(array.shape[0]), int(array.reshape(array.shape[0], -1).shape[1])


def build_summary(config):
    base_dir = Path(config["_config_dir"])
    data_config = config["data"]
    train_config = config["train"]
    gpd_config = config["gpd"]

    domain_summary_path = resolve_path(base_dir, data_config["domain_summary_path"])
    source_summary_path = resolve_path(base_dir, train_config["source_summary_path"])
    target_report_path = resolve_path(base_dir, train_config["target_report_path"])
    source_parameter_path = resolve_path(base_dir, train_config["source_parameter_vector_path"])
    target_parameter_path = resolve_path(base_dir, train_config["target_finetuned_params_path"])

    exp_index = int(gpd_config.get("exp_index", 9501))
    diffusion_sample_path = (base_dir.parent / "GPD" / "Output" / "sampleSeq_RealParams_{}.npy".format(exp_index)).resolve()

    domain_summary = load_json(domain_summary_path)
    source_summary = load_json(source_summary_path)
    target_report = load_json(target_report_path)

    source_rows, source_dim = parameter_row_count(source_parameter_path)
    generated_rows, generated_dim = parameter_row_count(diffusion_sample_path)
    target_rows, target_dim = parameter_row_count(target_parameter_path)

    one_group_one_network_check = {
        "group_key": ["WL", "Retention", "PEC"],
        "source_task_count": int(domain_summary["source_task_count"]),
        "target_task_count": int(domain_summary["target_task_count"]),
        "source_parameter_rows": source_rows,
        "generated_parameter_rows": generated_rows,
        "target_finetuned_parameter_rows": target_rows,
        "parameter_dim": {
            "source": source_dim,
            "generated": generated_dim,
            "target_finetuned": target_dim,
        },
        "verified": bool(
            source_rows == int(domain_summary["source_task_count"])
            and generated_rows == int(domain_summary["target_task_count"])
            and target_rows == int(domain_summary["target_task_count"])
            and source_dim == generated_dim == target_dim
        ),
    }

    summary = {
        "csv_path": domain_summary["csv_path"],
        "domain_summary": domain_summary,
        "source_pretrain_summary": source_summary,
        "target_transfer_report": target_report,
        "one_group_one_network_check": one_group_one_network_check,
        "artifacts": {
            "source_params": str(source_parameter_path),
            "diffusion_samples": str(diffusion_sample_path),
            "target_finetuned_params": str(target_parameter_path),
            "target_metrics_csv": str(resolve_path(base_dir, train_config["target_metrics_path"])),
            "target_report_json": str(target_report_path),
        },
    }
    return summary


def print_key_metrics(summary):
    target_report = summary["target_transfer_report"]
    check = summary["one_group_one_network_check"]
    test_point = target_report["test"]["point_level"]
    test_task = target_report["test"]["task_level"]

    print("model_2 full transfer summary")
    print("csv: {}".format(summary["csv_path"]))
    print(
        "tasks: source={} target={} verified_one_group_one_network={}".format(
            check["source_task_count"],
            check["target_task_count"],
            check["verified"],
        )
    )
    print(
        "test point-level: mse={:.6f} rmse={:.6f} mae={:.6f} r2={:.6f}".format(
            test_point["mse"],
            test_point["rmse"],
            test_point["mae"],
            test_point["r2"],
        )
    )
    print(
        "test task-level: mean_rmse={:.6f} mean_mae={:.6f} mean_r2={:.6f}".format(
            test_task["mean_rmse"],
            test_task["mean_mae"],
            test_task["mean_r2"],
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Run the full model_2 curve transfer pipeline.")
    parser.add_argument("--config_filename", default="config_curve_transfer.yaml")
    parser.add_argument("--skip_pipeline", action="store_true", help="Only summarize existing artifacts.")
    parser.add_argument("--summary_path", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    config = load_local_config(args.config_filename)
    if not args.skip_pipeline:
        run_curve_transfer_full_pipeline(config, python_executable=sys.executable)

    summary = build_summary(config)
    base_dir = Path(config["_config_dir"])
    summary_path = (
        resolve_path(base_dir, args.summary_path)
        if args.summary_path
        else resolve_path(base_dir, config["data"]["artifacts_dir"]) / "model2_full_transfer_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print_key_metrics(summary)
    print("summary saved to {}".format(summary_path))


if __name__ == "__main__":
    main()
