import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xlwt

from curve_task_workflow import run_curve_transfer_full_pipeline
from main import load_local_config
from run_model2_full_transfer import build_summary
from run_model2_sample_transfer import apply_overrides, run_sample


SAMPLE_CANDIDATES = [
    {
        "name": "tanh_32x16_e140_g100_r1",
        "activation": "tanh",
        "hidden_dims": [32, 16],
        "source_epochs": 140,
        "target_epochs": 140,
        "gpd_epochs": 100,
        "modeldim": 16,
        "diffusionstep": 10,
        "train_batchsize": 1,
        "sample_batchsize": 64,
        "target_scratch_retries": 1,
        "step_scaler_scope": "source_only",
    },
    {
        "name": "tanh_32x16_e160_g120_r1",
        "activation": "tanh",
        "hidden_dims": [32, 16],
        "source_epochs": 160,
        "target_epochs": 160,
        "gpd_epochs": 120,
        "modeldim": 16,
        "diffusionstep": 10,
        "train_batchsize": 1,
        "sample_batchsize": 64,
        "target_scratch_retries": 1,
        "step_scaler_scope": "source_only",
    },
    {
        "name": "tanh_32x16_e180_g120_r2",
        "activation": "tanh",
        "hidden_dims": [32, 16],
        "source_epochs": 180,
        "target_epochs": 180,
        "gpd_epochs": 120,
        "modeldim": 16,
        "diffusionstep": 10,
        "train_batchsize": 1,
        "sample_batchsize": 64,
        "target_scratch_retries": 2,
        "step_scaler_scope": "source_only",
    },
    {
        "name": "tanh_32x16_e140_g100_r2_allsteps",
        "activation": "tanh",
        "hidden_dims": [32, 16],
        "source_epochs": 140,
        "target_epochs": 140,
        "gpd_epochs": 100,
        "modeldim": 16,
        "diffusionstep": 10,
        "train_batchsize": 1,
        "sample_batchsize": 64,
        "target_scratch_retries": 2,
        "step_scaler_scope": "all_tasks",
    },
    {
        "name": "silu_32x16_e140_g100_r1",
        "activation": "silu",
        "hidden_dims": [32, 16],
        "source_epochs": 140,
        "target_epochs": 140,
        "gpd_epochs": 100,
        "modeldim": 16,
        "diffusionstep": 10,
        "train_batchsize": 1,
        "sample_batchsize": 64,
        "target_scratch_retries": 1,
        "step_scaler_scope": "source_only",
    },
]


def resolve_path(base_dir, path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def deep_copy_config(config):
    return json.loads(json.dumps(config))


def configure_model_run(config, model_index, stage_name, candidate, exp_index, artifacts_root):
    configured = deep_copy_config(config)
    base_dir = Path(configured["_config_dir"]).resolve()
    model_dir = artifacts_root / "model_{}".format(model_index) / stage_name / candidate["name"]
    model_dir.mkdir(parents=True, exist_ok=True)

    configured["data"]["csv_path"] = str((base_dir.parent / "Data" / "split_by_model" / "model_{}.csv".format(model_index)).resolve())
    configured["data"]["artifacts_dir"] = str(model_dir)
    configured["data"]["source_task_bundle_path"] = str(model_dir / "source_tasks.pkl")
    configured["data"]["target_task_bundle_path"] = str(model_dir / "target_tasks.pkl")
    configured["data"]["source_condition_csv"] = str(model_dir / "source_conditions.csv")
    configured["data"]["target_condition_csv"] = str(model_dir / "target_conditions.csv")
    configured["data"]["domain_summary_path"] = str(model_dir / "domain_summary.json")
    configured["data"]["step_scaler_path"] = str(model_dir / "step_scaler.pkl")
    configured["data"]["retention_scaler_path"] = str(model_dir / "retention_scaler.pkl")
    configured["data"]["pec_scaler_path"] = str(model_dir / "pec_scaler.pkl")
    configured["data"]["wl_vocab_path"] = str(model_dir / "wl_vocab.json")
    configured["data"]["step_scaler_scope"] = candidate["step_scaler_scope"]

    configured["train"]["source_parameter_vector_path"] = str(model_dir / "source_params.npy")
    configured["train"]["source_metrics_path"] = str(model_dir / "source_task_metrics.csv")
    configured["train"]["source_summary_path"] = str(model_dir / "source_summary.json")
    configured["train"]["target_finetuned_params_path"] = str(model_dir / "target_finetuned_params.npy")
    configured["train"]["target_metrics_path"] = str(model_dir / "target_task_metrics.csv")
    configured["train"]["target_report_path"] = str(model_dir / "target_transfer_report.json")
    configured["train"]["target_scratch_retries"] = int(candidate["target_scratch_retries"])
    configured["gpd"]["exp_index"] = int(exp_index)

    apply_overrides(
        configured,
        activation=candidate["activation"],
        hidden_dims=candidate["hidden_dims"],
        source_epochs=candidate["source_epochs"],
        target_epochs=candidate["target_epochs"],
        gpd_epochs=candidate["gpd_epochs"],
        exp_index=exp_index,
        train_batchsize=candidate["train_batchsize"],
        sample_batchsize=candidate["sample_batchsize"],
        modeldim=candidate["modeldim"],
        diffusionstep=candidate["diffusionstep"],
    )
    return configured


def summarize_run(config, stage_name, candidate_name, model_index, run_kind):
    summary = build_summary(config)
    base_dir = Path(config["_config_dir"]).resolve()
    data_dir = resolve_path(base_dir, config["data"]["artifacts_dir"])
    report = summary["target_transfer_report"]
    task_metrics_path = resolve_path(base_dir, config["train"]["target_metrics_path"])
    task_metrics = pd.read_csv(task_metrics_path)
    effective_source_task_count = int(summary["domain_summary"]["source_task_count"])
    effective_target_task_count = int(summary["domain_summary"]["target_task_count"])

    if run_kind == "sample":
        sample_check_path = data_dir / "sample_run_check.json"
        if sample_check_path.exists():
            with sample_check_path.open("r", encoding="utf-8") as handle:
                sample_check = json.load(handle)
            effective_source_task_count = int(sample_check["source_limit"])
            effective_target_task_count = int(sample_check["target_limit"])
            one_group = summary["one_group_one_network_check"]
            one_group["source_task_count"] = effective_source_task_count
            one_group["target_task_count"] = effective_target_task_count
            one_group["verified"] = bool(
                int(one_group["source_parameter_rows"]) == effective_source_task_count
                and int(one_group["generated_parameter_rows"]) == effective_target_task_count
                and int(one_group["target_finetuned_parameter_rows"]) == effective_target_task_count
                and int(one_group["parameter_dim"]["source"]) == int(one_group["parameter_dim"]["generated"])
                and int(one_group["parameter_dim"]["source"]) == int(one_group["parameter_dim"]["target_finetuned"])
            )

    summary["meta"] = {
        "model_index": int(model_index),
        "run_kind": run_kind,
        "stage_name": stage_name,
        "candidate_name": candidate_name,
        "artifacts_dir": str(data_dir),
        "effective_source_task_count": effective_source_task_count,
        "effective_target_task_count": effective_target_task_count,
        "frac_test_r2_gt_0_9": float((task_metrics["test_r2"] > 0.9).mean()) if "test_r2" in task_metrics else None,
        "frac_test_r2_lt_0": float((task_metrics["test_r2"] < 0).mean()) if "test_r2" in task_metrics else None,
        "min_test_r2": float(task_metrics["test_r2"].min()) if "test_r2" in task_metrics else None,
        "p05_test_r2": float(task_metrics["test_r2"].quantile(0.05)) if "test_r2" in task_metrics else None,
        "point_test_r2": float(report["test"]["point_level"]["r2"]),
        "task_mean_test_r2": float(report["test"]["task_level"]["mean_r2"]),
    }

    diffusion_source = Path(summary["artifacts"]["diffusion_samples"])
    diffusion_snapshot = data_dir / diffusion_source.name
    if diffusion_source.exists() and not diffusion_snapshot.exists():
        shutil.copy2(diffusion_source, diffusion_snapshot)
    summary["artifacts"]["diffusion_samples_snapshot"] = str(diffusion_snapshot)

    summary_path = data_dir / "{}_summary.json".format(run_kind)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def compute_r2_score(summary):
    report = summary["target_transfer_report"]
    point_r2 = float(report["test"]["point_level"]["r2"])
    task_r2 = float(report["test"]["task_level"]["mean_r2"])
    return {
        "point_r2": point_r2,
        "task_mean_r2": task_r2,
        "combined_r2": (point_r2 + task_r2) / 2.0,
    }


def passes_threshold(summary, threshold):
    score = compute_r2_score(summary)
    return score["point_r2"] >= threshold and score["task_mean_r2"] >= threshold


def flatten_summary_row(summary, status_label, elapsed_seconds):
    report = summary["target_transfer_report"]
    row = {
        "model_index": summary["meta"]["model_index"],
        "run_kind": summary["meta"]["run_kind"],
        "stage_name": summary["meta"]["stage_name"],
        "candidate_name": summary["meta"]["candidate_name"],
        "status": status_label,
        "elapsed_seconds": float(elapsed_seconds),
        "csv_path": summary["csv_path"],
        "source_task_count": summary["meta"]["effective_source_task_count"],
        "target_task_count": summary["meta"]["effective_target_task_count"],
        "verified_one_group_one_network": summary["one_group_one_network_check"]["verified"],
        "parameter_dim": summary["one_group_one_network_check"]["parameter_dim"]["source"],
        "test_point_r2": report["test"]["point_level"]["r2"],
        "test_point_rmse": report["test"]["point_level"]["rmse"],
        "test_point_mae": report["test"]["point_level"]["mae"],
        "test_task_mean_r2": report["test"]["task_level"]["mean_r2"],
        "test_task_median_r2": report["test"]["task_level"]["median_r2"],
        "val_point_r2": report["val"]["point_level"]["r2"],
        "val_task_mean_r2": report["val"]["task_level"]["mean_r2"],
        "train_point_r2": report["train"]["point_level"]["r2"],
        "train_task_mean_r2": report["train"]["task_level"]["mean_r2"],
        "frac_test_r2_gt_0_9": summary["meta"]["frac_test_r2_gt_0_9"],
        "frac_test_r2_lt_0": summary["meta"]["frac_test_r2_lt_0"],
        "min_test_r2": summary["meta"]["min_test_r2"],
        "p05_test_r2": summary["meta"]["p05_test_r2"],
        "artifacts_dir": summary["meta"]["artifacts_dir"],
        "target_metrics_csv": summary["artifacts"]["target_metrics_csv"],
        "target_report_json": summary["artifacts"]["target_report_json"],
        "diffusion_samples_snapshot": summary["artifacts"]["diffusion_samples_snapshot"],
    }
    return row


def save_state(state, state_path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def save_workbook(state, workbook_path):
    workbook = xlwt.Workbook()

    def write_sheet(sheet_name, rows):
        sheet = workbook.add_sheet(sheet_name[:31])
        if not rows:
            sheet.write(0, 0, "empty")
            return
        columns = list(rows[0].keys())
        for col_index, column in enumerate(columns):
            sheet.write(0, col_index, column)
        for row_index, row in enumerate(rows, start=1):
            for col_index, column in enumerate(columns):
                value = row.get(column)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                elif value is None:
                    value = ""
                elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    value = str(value)
                sheet.write(row_index, col_index, value)

    final_rows = []
    sample_rows = []
    full_rows = []
    for model_state in state["models"]:
        for row in model_state.get("sample_attempt_rows", []):
            sample_rows.append(row)
        for row in model_state.get("full_attempt_rows", []):
            full_rows.append(row)
        if model_state.get("final_row"):
            final_rows.append(model_state["final_row"])

    write_sheet("final_summary", final_rows)
    write_sheet("sample_attempts", sample_rows)
    write_sheet("full_attempts", full_rows)
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(str(workbook_path))


def log(message):
    print("[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message), flush=True)


def run_sample_attempt(base_config, model_index, candidate, attempt_index, sample_source_limit, sample_target_limit, sample_seed, artifacts_root):
    exp_index = 52000 + model_index * 100 + attempt_index
    config = configure_model_run(
        base_config,
        model_index=model_index,
        stage_name="sample",
        candidate=candidate,
        exp_index=exp_index,
        artifacts_root=artifacts_root,
    )
    start = time.time()
    run_sample(config, sample_source_limit, sample_target_limit, sample_seed)
    summary = summarize_run(
        config,
        stage_name="sample",
        candidate_name=candidate["name"],
        model_index=model_index,
        run_kind="sample",
    )
    elapsed = time.time() - start
    return summary, flatten_summary_row(summary, "completed", elapsed)


def run_full_attempt(base_config, model_index, candidate, attempt_index, artifacts_root):
    exp_index = 62000 + model_index * 100 + attempt_index
    config = configure_model_run(
        base_config,
        model_index=model_index,
        stage_name="full",
        candidate=candidate,
        exp_index=exp_index,
        artifacts_root=artifacts_root,
    )
    start = time.time()
    run_curve_transfer_full_pipeline(config, python_executable=sys.executable)
    summary = summarize_run(
        config,
        stage_name="full",
        candidate_name=candidate["name"],
        model_index=model_index,
        run_kind="full",
    )
    elapsed = time.time() - start
    return summary, flatten_summary_row(summary, "completed", elapsed)


def choose_full_candidate(sample_summaries):
    ranked = sorted(
        sample_summaries,
        key=lambda item: compute_r2_score(item["summary"])["combined_r2"],
        reverse=True,
    )
    return [item["candidate"] for item in ranked]


def main():
    parser = argparse.ArgumentParser(description="Tune and run full curve-transfer workflows for model_2 to model_8.")
    parser.add_argument("--start_model", type=int, default=2)
    parser.add_argument("--end_model", type=int, default=8)
    parser.add_argument("--sample_source_limit", type=int, default=150)
    parser.add_argument("--sample_target_limit", type=int, default=150)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_stop_threshold", type=float, default=0.95)
    parser.add_argument("--full_threshold", type=float, default=0.9)
    parser.add_argument("--artifacts_subdir", default="curve_transfer_batch")
    parser.add_argument("--workbook_name", default="curve_transfer_batch_results.xls")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    artifacts_root = (base_dir / "artifacts" / args.artifacts_subdir).resolve()
    state_path = artifacts_root / "batch_state.json"
    workbook_path = artifacts_root / args.workbook_name

    sample_base_config = load_local_config("config_curve_transfer_sample.yaml")
    full_base_config = load_local_config("config_curve_transfer.yaml")

    state = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "models": [],
    }
    save_state(state, state_path)

    for model_index in range(args.start_model, args.end_model + 1):
        model_state = {
            "model_index": int(model_index),
            "sample_attempt_rows": [],
            "full_attempt_rows": [],
            "sample_selected_candidate": None,
            "full_selected_candidate": None,
            "final_status": "pending",
            "final_row": None,
        }
        state["models"].append(model_state)
        save_state(state, state_path)

        log("model_{}: starting sample tuning".format(model_index))
        sample_summaries = []
        for attempt_index, candidate in enumerate(SAMPLE_CANDIDATES, start=1):
            log("model_{}: sample attempt {} with {}".format(model_index, attempt_index, candidate["name"]))
            try:
                summary, row = run_sample_attempt(
                    sample_base_config,
                    model_index=model_index,
                    candidate=candidate,
                    attempt_index=attempt_index,
                    sample_source_limit=args.sample_source_limit,
                    sample_target_limit=args.sample_target_limit,
                    sample_seed=args.sample_seed,
                    artifacts_root=artifacts_root,
                )
                row["threshold_pass"] = passes_threshold(summary, args.full_threshold)
                model_state["sample_attempt_rows"].append(row)
                sample_summaries.append({"candidate": candidate, "summary": summary})
                save_state(state, state_path)
                save_workbook(state, workbook_path)
                score = compute_r2_score(summary)
                log(
                    "model_{}: sample {} point_r2={:.4f} task_mean_r2={:.4f}".format(
                        model_index,
                        candidate["name"],
                        score["point_r2"],
                        score["task_mean_r2"],
                    )
                )
                if passes_threshold(summary, args.sample_stop_threshold):
                    log("model_{}: sample tuning reached stop threshold with {}".format(model_index, candidate["name"]))
                    break
            except Exception as exc:
                error_row = {
                    "model_index": model_index,
                    "run_kind": "sample",
                    "stage_name": "sample",
                    "candidate_name": candidate["name"],
                    "status": "failed",
                    "elapsed_seconds": None,
                    "error": str(exc),
                }
                model_state["sample_attempt_rows"].append(error_row)
                save_state(state, state_path)
                save_workbook(state, workbook_path)
                log("model_{}: sample attempt {} failed: {}".format(model_index, candidate["name"], exc))

        if not sample_summaries:
            model_state["final_status"] = "failed_no_sample"
            save_state(state, state_path)
            save_workbook(state, workbook_path)
            log("model_{}: no successful sample attempts, skipping full run".format(model_index))
            continue

        full_candidate_order = choose_full_candidate(sample_summaries)
        model_state["sample_selected_candidate"] = full_candidate_order[0]["name"]
        save_state(state, state_path)

        full_success = False
        for attempt_index, candidate in enumerate(full_candidate_order, start=1):
            log("model_{}: full attempt {} with {}".format(model_index, attempt_index, candidate["name"]))
            try:
                summary, row = run_full_attempt(
                    full_base_config,
                    model_index=model_index,
                    candidate=candidate,
                    attempt_index=attempt_index,
                    artifacts_root=artifacts_root,
                )
                row["threshold_pass"] = passes_threshold(summary, args.full_threshold)
                model_state["full_attempt_rows"].append(row)
                save_state(state, state_path)
                save_workbook(state, workbook_path)
                score = compute_r2_score(summary)
                log(
                    "model_{}: full {} point_r2={:.4f} task_mean_r2={:.4f}".format(
                        model_index,
                        candidate["name"],
                        score["point_r2"],
                        score["task_mean_r2"],
                    )
                )
                if passes_threshold(summary, args.full_threshold):
                    model_state["full_selected_candidate"] = candidate["name"]
                    model_state["final_status"] = "passed"
                    model_state["final_row"] = row
                    full_success = True
                    log("model_{}: passed full threshold with {}".format(model_index, candidate["name"]))
                    break
            except Exception as exc:
                error_row = {
                    "model_index": model_index,
                    "run_kind": "full",
                    "stage_name": "full",
                    "candidate_name": candidate["name"],
                    "status": "failed",
                    "elapsed_seconds": None,
                    "error": str(exc),
                }
                model_state["full_attempt_rows"].append(error_row)
                save_state(state, state_path)
                save_workbook(state, workbook_path)
                log("model_{}: full attempt {} failed: {}".format(model_index, candidate["name"], exc))

        if not full_success:
            completed_rows = [row for row in model_state["full_attempt_rows"] if row.get("status") == "completed"]
            if completed_rows:
                best_row = max(
                    completed_rows,
                    key=lambda row: float(row.get("test_point_r2", -1e9)) + float(row.get("test_task_mean_r2", -1e9)),
                )
                model_state["final_row"] = best_row
                model_state["full_selected_candidate"] = best_row["candidate_name"]
                model_state["final_status"] = "best_below_threshold"
                log("model_{}: full runs completed but stayed below threshold".format(model_index))
            else:
                model_state["final_status"] = "failed_no_full"

        save_state(state, state_path)
        save_workbook(state, workbook_path)

    log("batch run complete")
    save_state(state, state_path)
    save_workbook(state, workbook_path)


if __name__ == "__main__":
    main()
