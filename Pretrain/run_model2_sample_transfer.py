import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from curve_task_workflow import (
    _save_pickle,
    finetune_target_domain_models,
    prepare_curve_domains,
    train_source_domain_models,
)
from main import load_local_config


def resolve_path(base_dir, path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def write_subset_csv(input_path, output_path, task_ids):
    frame = pd.read_csv(input_path)
    frame = frame[frame["task_id"].isin(task_ids)].copy()
    frame = frame.sort_values("task_id").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def load_pickle(path_like):
    with Path(path_like).open("rb") as handle:
        return pickle.load(handle)


def select_task_subset(tasks, limit, seed):
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(tasks), size=limit, replace=False))
    subset = [tasks[int(index)] for index in chosen]
    task_ids = [int(task["task_id"]) for task in subset]
    return subset, task_ids


def prepare_subset_artifacts(config, source_limit, target_limit, seed):
    base_dir = Path(config["_config_dir"])
    data_config = config["data"]
    artifacts_dir = resolve_path(base_dir, data_config["artifacts_dir"])
    sample_dir = artifacts_dir / "subsets"
    sample_dir.mkdir(parents=True, exist_ok=True)

    source_tasks = load_pickle(resolve_path(base_dir, data_config["source_task_bundle_path"]))
    target_tasks = load_pickle(resolve_path(base_dir, data_config["target_task_bundle_path"]))
    source_subset, source_task_ids = select_task_subset(source_tasks, source_limit, seed)
    target_subset, target_task_ids = select_task_subset(target_tasks, target_limit, seed + 1)

    source_bundle_path = sample_dir / "source_tasks_{}_seed{}.pkl".format(source_limit, seed)
    target_bundle_path = sample_dir / "target_tasks_{}_seed{}.pkl".format(target_limit, seed + 1)
    _save_pickle(source_subset, source_bundle_path)
    _save_pickle(target_subset, target_bundle_path)

    source_condition_csv = sample_dir / "source_conditions_{}_seed{}.csv".format(source_limit, seed)
    target_condition_csv = sample_dir / "target_conditions_{}_seed{}.csv".format(target_limit, seed + 1)
    write_subset_csv(resolve_path(base_dir, data_config["source_condition_csv"]), source_condition_csv, source_task_ids)
    write_subset_csv(resolve_path(base_dir, data_config["target_condition_csv"]), target_condition_csv, target_task_ids)

    subset_config = json.loads(json.dumps(config))
    subset_config["data"]["source_task_bundle_path"] = str(source_bundle_path)
    subset_config["data"]["target_task_bundle_path"] = str(target_bundle_path)
    subset_config["data"]["source_condition_csv"] = str(source_condition_csv)
    subset_config["data"]["target_condition_csv"] = str(target_condition_csv)
    subset_config["data"]["artifacts_dir"] = str(artifacts_dir)
    return subset_config, {
        "source_task_ids": source_task_ids,
        "target_task_ids": target_task_ids,
        "source_bundle_path": str(source_bundle_path),
        "target_bundle_path": str(target_bundle_path),
        "source_condition_csv": str(source_condition_csv),
        "target_condition_csv": str(target_condition_csv),
    }


def parameter_shape(path_like):
    array = np.load(path_like)
    return tuple(int(value) for value in array.shape)


def load_json(path_like):
    with Path(path_like).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def print_metric_summary(report):
    test = report["test"]
    print("sample transfer test point-level:", test["point_level"])
    print("sample transfer test task-level:", test["task_level"])


def apply_overrides(
    config,
    activation=None,
    hidden_dims=None,
    source_epochs=None,
    target_epochs=None,
    gpd_epochs=None,
    exp_index=None,
    train_batchsize=None,
    sample_batchsize=None,
    modeldim=None,
    diffusionstep=None,
):
    if activation:
        config["model"]["activation"] = str(activation)
    if hidden_dims:
        config["model"]["hidden_dims"] = [int(value) for value in hidden_dims]
    if source_epochs is not None:
        config["train"]["source_epochs"] = int(source_epochs)
    if target_epochs is not None:
        config["train"]["target_epochs"] = int(target_epochs)
    if gpd_epochs is not None:
        config["gpd"]["epochs"] = int(gpd_epochs)
    if exp_index is not None:
        config["gpd"]["exp_index"] = int(exp_index)
    if train_batchsize is not None:
        config["gpd"]["trainbatchsize"] = int(train_batchsize)
    if sample_batchsize is not None:
        config["gpd"]["samplebatchsize"] = int(sample_batchsize)
    if modeldim is not None:
        config["gpd"]["modeldim"] = int(modeldim)
    if diffusionstep is not None:
        config["gpd"]["diffusionstep"] = int(diffusionstep)
    return config


def run_sample(config, source_limit, target_limit, seed):
    base_dir = Path(config["_config_dir"])

    if source_limit <= 0 or target_limit <= 0:
        raise ValueError("source_limit and target_limit must be positive")

    prepare_curve_domains(config)
    subset_config, subset_info = prepare_subset_artifacts(config, source_limit, target_limit, seed)
    data_config = subset_config["data"]
    train_config = subset_config["train"]
    gpd_config = subset_config["gpd"]
    train_source_domain_models(subset_config)

    gpd_dir = (base_dir.parent / "GPD").resolve()
    exp_index = int(gpd_config.get("exp_index", 9511))
    command = [
        sys.executable,
        "-u",
        "1Dmain.py",
        "--modeldim",
        str(int(gpd_config.get("modeldim", 16))),
        "--epochs",
        str(int(gpd_config.get("epochs", 80))),
        "--expIndex",
        str(exp_index),
        "--diffusionstep",
        str(int(gpd_config.get("diffusionstep", 10))),
        "--denoise",
        str(gpd_config.get("denoise", "Trans3")),
        "--trainbatchsize",
        str(int(gpd_config.get("trainbatchsize", 1))),
        "--samplebatchsize",
        str(int(gpd_config.get("samplebatchsize", 64))),
        "--repeat_num",
        str(int(gpd_config.get("repeat_num", 1))),
        "--train_param_path",
        str(resolve_path(base_dir, train_config["source_parameter_vector_path"])),
        "--train_condition_csv",
        str(resolve_path(base_dir, data_config["source_condition_csv"])),
        "--sample_condition_csv",
        str(resolve_path(base_dir, data_config["target_condition_csv"])),
        "--retention_scaler_path",
        str(resolve_path(base_dir, data_config["retention_scaler_path"])),
        "--pec_scaler_path",
        str(resolve_path(base_dir, data_config["pec_scaler_path"])),
        "--wl_vocab_path",
        str(resolve_path(base_dir, data_config["wl_vocab_path"])),
    ]
    subprocess.run(command, cwd=str(gpd_dir), check=True)

    diffusion_sample_path = gpd_dir / "Output" / "sampleSeq_RealParams_{}.npy".format(exp_index)
    finetune_target_domain_models(subset_config, diffusion_sample_path)

    report = load_json(resolve_path(base_dir, train_config["target_report_path"]))
    source_shape = parameter_shape(resolve_path(base_dir, train_config["source_parameter_vector_path"]))
    generated_shape = parameter_shape(diffusion_sample_path)
    finetuned_shape = parameter_shape(resolve_path(base_dir, train_config["target_finetuned_params_path"]))
    check = {
        "source_limit": int(source_limit),
        "target_limit": int(target_limit),
        "seed": int(seed),
        "source_params_shape": source_shape,
        "generated_params_shape": generated_shape,
        "target_finetuned_params_shape": finetuned_shape,
        "sample_only": True,
    }
    check.update(subset_info)
    artifacts_dir = resolve_path(base_dir, data_config["artifacts_dir"])
    with (artifacts_dir / "sample_run_check.json").open("w", encoding="utf-8") as handle:
        json.dump(check, handle, ensure_ascii=False, indent=2)

    print(json.dumps(check, ensure_ascii=False, indent=2))
    print_metric_summary(report)


def main():
    parser = argparse.ArgumentParser(description="Run a limited model_2 transfer experiment only.")
    parser.add_argument("--config_filename", default="config_curve_transfer_sample.yaml")
    parser.add_argument("--source_limit", type=int, default=100)
    parser.add_argument("--target_limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activation", default="", type=str)
    parser.add_argument("--hidden_dims", nargs="*", type=int, default=None)
    parser.add_argument("--source_epochs", type=int, default=None)
    parser.add_argument("--target_epochs", type=int, default=None)
    parser.add_argument("--gpd_epochs", type=int, default=None)
    parser.add_argument("--exp_index", type=int, default=None)
    parser.add_argument("--train_batchsize", type=int, default=None)
    parser.add_argument("--sample_batchsize", type=int, default=None)
    parser.add_argument("--modeldim", type=int, default=None)
    parser.add_argument("--diffusionstep", type=int, default=None)
    args = parser.parse_args()

    config = load_local_config(args.config_filename)
    config = apply_overrides(
        config,
        activation=args.activation or None,
        hidden_dims=args.hidden_dims,
        source_epochs=args.source_epochs,
        target_epochs=args.target_epochs,
        gpd_epochs=args.gpd_epochs,
        exp_index=args.exp_index,
        train_batchsize=args.train_batchsize,
        sample_batchsize=args.sample_batchsize,
        modeldim=args.modeldim,
        diffusionstep=args.diffusionstep,
    )
    run_sample(config, args.source_limit, args.target_limit, args.seed)


if __name__ == "__main__":
    main()
