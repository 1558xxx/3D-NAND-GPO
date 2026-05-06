import copy
import csv
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from Models import MLPRegressor


CURVE_COLUMNS = ["step", "WL", "Retention", "PEC", "freq"]
_PARALLEL_WORKER_CONTEXT = {}


def _resolve_path(base_dir, path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = (Path(base_dir) / path).resolve()
    return path


def _ensure_parent(path_like):
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_pickle(obj, path_like):
    path = _ensure_parent(path_like)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def _load_pickle(path_like):
    with Path(path_like).open("rb") as handle:
        return pickle.load(handle)


def _save_json(obj, path_like):
    path = _ensure_parent(path_like)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def _load_curve_frame(csv_path):
    frame = pd.read_csv(csv_path).copy()
    if frame.shape[1] < 5:
        raise ValueError("Expected at least 5 columns in curve CSV, got {}".format(frame.shape[1]))

    selected = frame.iloc[:, :5].copy()
    selected.columns = CURVE_COLUMNS
    selected["step"] = selected["step"].astype(np.int64)
    selected["WL"] = selected["WL"].astype(np.int64)
    selected["Retention"] = selected["Retention"].astype(np.int64)
    selected["PEC"] = selected["PEC"].astype(np.int64)
    selected["freq"] = selected["freq"].astype(np.float32)
    return selected


def _aggregate_curve_rows(frame):
    aggregated = (
        frame.groupby(["WL", "Retention", "PEC", "step"], as_index=False)
        .agg(freq=("freq", "mean"), raw_count=("freq", "size"))
        .sort_values(["WL", "Retention", "PEC", "step"])
        .reset_index(drop=True)
    )
    return aggregated


def _select_evenly_spaced_indices(candidates, count):
    candidates = np.asarray(sorted(candidates), dtype=np.int64)
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if count >= len(candidates):
        return candidates.copy()

    raw_positions = np.linspace(0, len(candidates) - 1, num=count)
    selected = []
    used = set()
    for position in raw_positions:
        idx = int(round(position))
        idx = max(0, min(idx, len(candidates) - 1))
        if idx not in used:
            selected.append(candidates[idx])
            used.add(idx)

    if len(selected) < count:
        for idx, value in enumerate(candidates):
            if idx in used:
                continue
            selected.append(value)
            used.add(idx)
            if len(selected) == count:
                break

    return np.asarray(sorted(selected), dtype=np.int64)


def _split_task_indices(task_length, split_config):
    train_ratio = float(split_config.get("train", 0.6))
    val_ratio = float(split_config.get("val", 0.2))
    test_ratio = float(split_config.get("test", 0.2))

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("Task split ratios must sum to 1.0, got {}".format(ratio_sum))
    if task_length < 6:
        raise ValueError("Task length {} is too small for train/val/test splitting".format(task_length))

    n_test = max(2, int(round(task_length * test_ratio)))
    n_val = max(2, int(round(task_length * val_ratio)))
    n_train = task_length - n_test - n_val

    while n_train < 2 and n_val > 2:
        n_val -= 1
        n_train += 1
    while n_train < 2 and n_test > 2:
        n_test -= 1
        n_train += 1

    if n_train < 2:
        raise ValueError("Task length {} cannot satisfy split constraints".format(task_length))

    all_indices = np.arange(task_length, dtype=np.int64)
    boundary_train = {0, task_length - 1}
    candidate_indices = np.asarray(
        [index for index in all_indices if index not in boundary_train],
        dtype=np.int64,
    )

    if len(candidate_indices) < n_test + n_val:
        raise ValueError("Task length {} cannot satisfy interior split constraints".format(task_length))

    test_indices = _select_evenly_spaced_indices(candidate_indices, n_test)
    remaining = [index for index in candidate_indices if index not in set(test_indices.tolist())]
    val_indices = _select_evenly_spaced_indices(remaining, n_val)
    train_indices = np.asarray(
        [
            index
            for index in all_indices
            if index not in set(test_indices.tolist()) and index not in set(val_indices.tolist())
        ],
        dtype=np.int64,
    )

    return {
        "train": np.asarray(sorted(train_indices.tolist()), dtype=np.int64),
        "val": np.asarray(sorted(val_indices.tolist()), dtype=np.int64),
        "test": np.asarray(sorted(test_indices.tolist()), dtype=np.int64),
    }


def _build_task_records(aggregated_frame, source_retentions, target_retentions, min_task_points, target_split_config):
    source_tasks = []
    target_tasks = []
    duplicate_rows_averaged = int((aggregated_frame["raw_count"] > 1).sum())

    grouped = aggregated_frame.groupby(["WL", "Retention", "PEC"], sort=True)
    source_id = 0
    target_id = 0

    for (wl, retention, pec), group in grouped:
        group = group.sort_values("step").reset_index(drop=True)
        if len(group) < min_task_points:
            continue

        task = {
            "WL": int(wl),
            "Retention": int(retention),
            "PEC": int(pec),
            "steps": group["step"].astype(np.int64).tolist(),
            "freqs": group["freq"].astype(np.float32).tolist(),
            "raw_rows": int(group["raw_count"].sum()),
            "num_points": int(len(group)),
        }

        if retention in source_retentions:
            task["task_id"] = int(source_id)
            source_id += 1
            source_tasks.append(task)
        elif retention in target_retentions:
            task["task_id"] = int(target_id)
            task["splits"] = {
                split_name: split_indices.tolist()
                for split_name, split_indices in _split_task_indices(len(group), target_split_config).items()
            }
            target_id += 1
            target_tasks.append(task)

    summary = {
        "source_task_count": int(len(source_tasks)),
        "target_task_count": int(len(target_tasks)),
        "duplicate_condition_step_rows_averaged": duplicate_rows_averaged,
    }
    return source_tasks, target_tasks, summary


def _task_records_to_condition_frame(tasks):
    records = []
    for task in tasks:
        records.append(
            {
                "task_id": int(task["task_id"]),
                "WL": int(task["WL"]),
                "Retention": int(task["Retention"]),
                "PEC": int(task["PEC"]),
                "num_points": int(task["num_points"]),
            }
        )
    return pd.DataFrame.from_records(records)


def _fit_and_save_domain_scalers(frame, source_tasks, target_tasks, data_config, base_dir):
    task_conditions = _task_records_to_condition_frame(source_tasks + target_tasks)
    wl_values = sorted(task_conditions["WL"].dropna().astype(int).unique().tolist())
    source_task_conditions = _task_records_to_condition_frame(source_tasks)
    step_scaler_scope = str(data_config.get("step_scaler_scope", "source_only")).lower()

    retention_scaler = MinMaxScaler(feature_range=(-1, 1))
    retention_scaler.fit(task_conditions[["Retention"]].to_numpy(dtype=np.float32))
    _save_pickle(retention_scaler, _resolve_path(base_dir, data_config["retention_scaler_path"]))

    pec_scaler = MinMaxScaler(feature_range=(-1, 1))
    pec_scaler.fit(task_conditions[["PEC"]].to_numpy(dtype=np.float32))
    _save_pickle(pec_scaler, _resolve_path(base_dir, data_config["pec_scaler_path"]))

    step_scaler = MinMaxScaler(feature_range=(-1, 1))
    if step_scaler_scope == "source_only":
        source_steps = []
        for task in source_tasks:
            source_steps.extend(task["steps"])
        step_scaler.fit(np.asarray(source_steps, dtype=np.float32).reshape(-1, 1))
    elif step_scaler_scope == "all_tasks":
        step_scaler.fit(frame[["step"]].to_numpy(dtype=np.float32))
    else:
        raise ValueError("Unsupported step_scaler_scope: {}".format(step_scaler_scope))
    _save_pickle(step_scaler, _resolve_path(base_dir, data_config["step_scaler_path"]))

    wl_vocab_path = _resolve_path(base_dir, data_config["wl_vocab_path"])
    _ensure_parent(wl_vocab_path)
    with wl_vocab_path.open("w", encoding="utf-8") as handle:
        json.dump(wl_values, handle, ensure_ascii=False, indent=2)

    return {
        "retention_scaler": retention_scaler,
        "pec_scaler": pec_scaler,
        "step_scaler": step_scaler,
        "wl_values": wl_values,
        "step_scaler_scope": step_scaler_scope,
        "source_retention_min": float(source_task_conditions["Retention"].min()) if not source_task_conditions.empty else None,
        "source_retention_max": float(source_task_conditions["Retention"].max()) if not source_task_conditions.empty else None,
    }


def prepare_curve_domains(config):
    base_dir = config["_config_dir"]
    data_config = config.get("data", {})

    frame = _load_curve_frame(_resolve_path(base_dir, data_config["csv_path"]))
    aggregated_frame = _aggregate_curve_rows(frame)

    source_retentions = set(int(value) for value in data_config.get("source_retentions", [1, 2, 3]))
    target_retentions = set(int(value) for value in data_config.get("target_retentions", [4, 5, 6]))
    min_task_points = int(data_config.get("min_task_points", 10))
    target_split_config = data_config.get("target_task_split", {"train": 0.6, "val": 0.2, "test": 0.2})

    source_tasks, target_tasks, summary = _build_task_records(
        aggregated_frame=aggregated_frame,
        source_retentions=source_retentions,
        target_retentions=target_retentions,
        min_task_points=min_task_points,
        target_split_config=target_split_config,
    )

    scaler_info = _fit_and_save_domain_scalers(
        frame=aggregated_frame,
        source_tasks=source_tasks,
        target_tasks=target_tasks,
        data_config=data_config,
        base_dir=base_dir,
    )

    source_bundle_path = _resolve_path(base_dir, data_config["source_task_bundle_path"])
    target_bundle_path = _resolve_path(base_dir, data_config["target_task_bundle_path"])
    _save_pickle(source_tasks, source_bundle_path)
    _save_pickle(target_tasks, target_bundle_path)

    source_condition_csv = _resolve_path(base_dir, data_config["source_condition_csv"])
    target_condition_csv = _resolve_path(base_dir, data_config["target_condition_csv"])
    _ensure_parent(source_condition_csv)
    _ensure_parent(target_condition_csv)
    _task_records_to_condition_frame(source_tasks).to_csv(source_condition_csv, index=False, encoding="utf-8-sig")
    _task_records_to_condition_frame(target_tasks).to_csv(target_condition_csv, index=False, encoding="utf-8-sig")

    domain_summary = {
        "csv_path": str(_resolve_path(base_dir, data_config["csv_path"])),
        "source_retentions": sorted(source_retentions),
        "target_retentions": sorted(target_retentions),
        "min_task_points": min_task_points,
        "source_task_count": summary["source_task_count"],
        "target_task_count": summary["target_task_count"],
        "duplicate_condition_step_rows_averaged": summary["duplicate_condition_step_rows_averaged"],
        "source_num_points_mean": float(np.mean([task["num_points"] for task in source_tasks])) if source_tasks else 0.0,
        "target_num_points_mean": float(np.mean([task["num_points"] for task in target_tasks])) if target_tasks else 0.0,
        "step_scaler_scope": scaler_info["step_scaler_scope"],
        "condition_scaler_scope": "all_tasks",
    }
    _save_json(domain_summary, _resolve_path(base_dir, data_config["domain_summary_path"]))
    print(json.dumps(domain_summary, ensure_ascii=False, indent=2))


def _build_model(model_config):
    return MLPRegressor(
        in_dim=int(model_config.get("in_dim", 1)),
        hidden_dims=tuple(model_config.get("hidden_dims", [32, 16])),
        out_dim=int(model_config.get("out_dim", 1)),
        activation=str(model_config.get("activation", "relu")),
    )


def _transform_steps(step_scaler, steps):
    array = np.asarray(steps, dtype=np.float32).reshape(-1, 1)
    return step_scaler.transform(array).astype(np.float32)


def _transform_freqs(freqs):
    array = np.asarray(freqs, dtype=np.float32).reshape(-1, 1)
    return np.log1p(array).astype(np.float32)


def _inverse_freqs(values):
    clipped = np.clip(np.asarray(values, dtype=np.float32), a_min=-20.0, a_max=12.0)
    return np.expm1(clipped).astype(np.float32)


def _regression_summary(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
        "num_points": int(len(y_true)),
    }


def _evaluate_split(model, x_all, y_all, indices, device):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return None

    with torch.no_grad():
        x_tensor = torch.as_tensor(x_all[indices], dtype=torch.float32, device=device)
        predictions = model(x_tensor).detach().cpu().numpy()

    y_true_log = y_all[indices].reshape(-1, 1)
    y_pred_log = np.nan_to_num(
        predictions.reshape(-1, 1),
        nan=0.0,
        posinf=12.0,
        neginf=-20.0,
    )
    y_true = _inverse_freqs(y_true_log).reshape(-1)
    y_pred = np.clip(_inverse_freqs(y_pred_log).reshape(-1), a_min=0.0, a_max=None)
    metrics = _regression_summary(y_true, y_pred)
    metrics["y_true"] = y_true.tolist()
    metrics["y_pred"] = y_pred.tolist()
    return metrics


def _fit_single_task_model(
    task,
    model_config,
    train_config,
    step_scaler,
    device,
    init_vector=None,
    use_task_splits=False,
    seed_offset=0,
):
    seed = int(train_config.get("seed", 42)) + int(seed_offset)
    torch.manual_seed(seed)

    model = _build_model(model_config).to(device)
    if init_vector is not None:
        init_tensor = torch.as_tensor(init_vector, dtype=torch.float32, device=device).reshape(-1)
        expected_dim = parameters_to_vector(model.parameters()).numel()
        if init_tensor.numel() != expected_dim:
            raise ValueError("Parameter vector size mismatch: {} vs {}".format(init_tensor.numel(), expected_dim))
        vector_to_parameters(init_tensor, model.parameters())

    x_all = _transform_steps(step_scaler, task["steps"])
    y_all = _transform_freqs(task["freqs"])
    all_indices = np.arange(len(task["steps"]), dtype=np.int64)

    if use_task_splits:
        splits = {name: np.asarray(indices, dtype=np.int64) for name, indices in task["splits"].items()}
        train_indices = splits["train"]
        val_indices = splits["val"]
        test_indices = splits["test"]
        max_epochs = int(train_config.get("target_epochs", 200))
    else:
        train_indices = all_indices
        val_indices = np.empty((0,), dtype=np.int64)
        test_indices = np.empty((0,), dtype=np.int64)
        max_epochs = int(train_config.get("source_epochs", 200))

    optimizer_name = str(train_config.get("optimizer", "lbfgs")).lower()
    lr = float(train_config.get("lr", 1e-2))
    weight_decay = float(train_config.get("weight_decay", 0.0))
    lbfgs_lr = float(train_config.get("lbfgs_lr", 0.8))
    lbfgs_max_iter = int(train_config.get("lbfgs_max_iter", 5))
    lbfgs_history_size = int(train_config.get("lbfgs_history_size", 20))
    patience = int(train_config.get("patience", 40))
    min_delta = float(train_config.get("min_delta", 1e-6))

    if optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer for curve tasks: {}".format(optimizer_name))
    criterion = nn.MSELoss()

    x_train = torch.as_tensor(x_all[train_indices], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_all[train_indices], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(x_all[val_indices], dtype=torch.float32, device=device) if val_indices.size else None
    y_val = torch.as_tensor(y_all[val_indices], dtype=torch.float32, device=device) if val_indices.size else None

    best_state = copy.deepcopy(model.state_dict())
    model.eval()
    with torch.no_grad():
        initial_train_loss = float(criterion(model(x_train), y_train).item())
        if x_val is not None and y_val is not None:
            initial_score = float(criterion(model(x_val), y_val).item())
        else:
            initial_score = initial_train_loss

    best_score = initial_score if np.isfinite(initial_score) else float("inf")
    best_epoch = 0
    no_improve_count = 0
    final_train_loss = initial_train_loss if np.isfinite(initial_train_loss) else None

    for epoch in range(max_epochs):
        model.train()
        if optimizer_name == "lbfgs":
            def closure():
                optimizer.zero_grad()
                train_predictions = model(x_train)
                train_loss_inner = criterion(train_predictions, y_train)
                train_loss_inner.backward()
                return train_loss_inner

            train_loss = optimizer.step(closure)
            final_train_loss = float(train_loss.item())
        else:
            optimizer.zero_grad()
            train_predictions = model(x_train)
            train_loss = criterion(train_predictions, y_train)
            train_loss.backward()
            optimizer.step()
            final_train_loss = float(train_loss.item())

        parameters_are_finite = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
        if not np.isfinite(final_train_loss) or not parameters_are_finite:
            model.load_state_dict(best_state)
            break

        monitor_loss = final_train_loss
        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_predictions = model(x_val)
                monitor_loss = float(criterion(val_predictions, y_val).item())

        if not np.isfinite(monitor_loss):
            model.load_state_dict(best_state)
            break

        if monitor_loss + min_delta < best_score:
            best_score = monitor_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            break

    model.load_state_dict(best_state)
    parameter_vector = parameters_to_vector(model.parameters()).detach().cpu().numpy().astype(np.float32)

    metrics = {
        "task_id": int(task["task_id"]),
        "WL": int(task["WL"]),
        "Retention": int(task["Retention"]),
        "PEC": int(task["PEC"]),
        "num_points": int(task["num_points"]),
        "best_epoch": int(best_epoch),
        "best_monitor_loss": float(best_score),
        "final_train_loss": float(final_train_loss if final_train_loss is not None else best_score),
        "train": _evaluate_split(model, x_all, y_all, train_indices, device),
        "val": _evaluate_split(model, x_all, y_all, val_indices, device) if val_indices.size else None,
        "test": _evaluate_split(model, x_all, y_all, test_indices, device) if test_indices.size else None,
    }
    return parameter_vector, metrics


def _fit_target_task_with_selection(
    task,
    model_config,
    train_config,
    step_scaler,
    device,
    generated_init_vector,
):
    scratch_retries = int(train_config.get("target_scratch_retries", 0))
    candidate_specs = [("generated", generated_init_vector, 0)]
    for retry_index in range(scratch_retries):
        candidate_specs.append(("scratch", None, retry_index + 1))

    best_vector = None
    best_metrics = None
    best_score = None

    for init_mode, init_vector, seed_offset in candidate_specs:
        parameter_vector, metrics = _fit_single_task_model(
            task=task,
            model_config=model_config,
            train_config=train_config,
            step_scaler=step_scaler,
            device=device,
            init_vector=init_vector,
            use_task_splits=True,
            seed_offset=seed_offset,
        )
        metrics["init_mode"] = init_mode
        metrics["seed_offset"] = int(seed_offset)
        candidate_score = float(metrics["best_monitor_loss"])

        if best_score is None or candidate_score < best_score:
            best_vector = parameter_vector
            best_metrics = metrics
            best_score = candidate_score

    return best_vector, best_metrics


def _resolve_parallel_workers(train_config, task_count, device):
    raw_value = train_config.get("parallel_workers", 1)
    if raw_value is None:
        return 1

    if isinstance(raw_value, str):
        raw_text = raw_value.strip().lower()
        if raw_text in {"", "1", "none"}:
            return 1
        if raw_text == "auto":
            cpu_total = os.cpu_count() or 1
            return max(1, min(task_count, max(1, cpu_total // 2)))
        raw_value = int(raw_text)

    worker_count = max(1, min(task_count, int(raw_value)))
    if worker_count > 1 and device.type != "cpu":
        raise ValueError("parallel_workers > 1 is only supported with CPU task training")
    return worker_count


def _parallel_worker_init(model_config, train_config, step_scaler, device_name, worker_mode):
    global _PARALLEL_WORKER_CONTEXT
    torch.set_num_threads(1)
    _PARALLEL_WORKER_CONTEXT = {
        "model_config": model_config,
        "train_config": train_config,
        "step_scaler": step_scaler,
        "device_name": device_name,
        "worker_mode": worker_mode,
    }


def _parallel_fit_worker(payload):
    task = payload["task"]
    init_vector = payload.get("init_vector")
    generated_init_vector = payload.get("generated_init_vector")
    seed_offset = int(payload.get("seed_offset", 0))

    context = _PARALLEL_WORKER_CONTEXT
    device = torch.device(context["device_name"])
    if context["worker_mode"] == "source":
        return _fit_single_task_model(
            task=task,
            model_config=context["model_config"],
            train_config=context["train_config"],
            step_scaler=context["step_scaler"],
            device=device,
            init_vector=init_vector,
            use_task_splits=False,
            seed_offset=seed_offset,
        )
    if context["worker_mode"] == "target":
        return _fit_target_task_with_selection(
            task=task,
            model_config=context["model_config"],
            train_config=context["train_config"],
            step_scaler=context["step_scaler"],
            device=device,
            generated_init_vector=generated_init_vector,
        )
    raise ValueError("Unsupported worker_mode: {}".format(context["worker_mode"]))


def _process_pool_chunksize(task_count, worker_count):
    return max(1, task_count // max(worker_count * 8, 1))


def _load_task_bundle(path_like):
    return _load_pickle(path_like)


def _save_metrics_csv(records, path_like):
    if not records:
        return

    rows = []
    for record in records:
        row = {
            "task_id": record["task_id"],
            "WL": record["WL"],
            "Retention": record["Retention"],
            "PEC": record["PEC"],
            "num_points": record["num_points"],
            "init_mode": record.get("init_mode", ""),
            "seed_offset": record.get("seed_offset", 0),
            "best_epoch": record["best_epoch"],
            "best_monitor_loss": record["best_monitor_loss"],
            "final_train_loss": record["final_train_loss"],
        }
        for split_name in ("train", "val", "test"):
            split_metrics = record.get(split_name)
            if not split_metrics:
                continue
            row["{}_mse".format(split_name)] = split_metrics["mse"]
            row["{}_rmse".format(split_name)] = split_metrics["rmse"]
            row["{}_mae".format(split_name)] = split_metrics["mae"]
            row["{}_r2".format(split_name)] = split_metrics["r2"]
            row["{}_points".format(split_name)] = split_metrics["num_points"]
        rows.append(row)

    fieldnames = list(rows[0].keys())
    output_path = _ensure_parent(path_like)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_source_domain_models(config, limit_tasks=None):
    base_dir = config["_config_dir"]
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    train_config = config.get("train", {})

    source_tasks = _load_task_bundle(_resolve_path(base_dir, data_config["source_task_bundle_path"]))
    if limit_tasks is not None:
        source_tasks = source_tasks[:limit_tasks]

    step_scaler = _load_pickle(_resolve_path(base_dir, data_config["step_scaler_path"]))
    device = _select_torch_device(train_config)
    print("source domain training device: {}".format(device))

    parameter_vectors = []
    metrics_records = []

    total_tasks = len(source_tasks)
    worker_count = _resolve_parallel_workers(train_config, total_tasks, device)
    if worker_count == 1:
        for index, task in enumerate(source_tasks, start=1):
            parameter_vector, metrics = _fit_single_task_model(
                task=task,
                model_config=model_config,
                train_config=train_config,
                step_scaler=step_scaler,
                device=device,
                init_vector=None,
                use_task_splits=False,
            )
            parameter_vectors.append(parameter_vector)
            metrics_records.append(metrics)

            if index == 1 or index % 200 == 0 or index == total_tasks:
                print("trained source tasks {}/{}".format(index, total_tasks))
    else:
        print("source domain parallel workers: {}".format(worker_count))
        payloads = [{"task": task, "seed_offset": 0} for task in source_tasks]
        chunksize = _process_pool_chunksize(total_tasks, worker_count)
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_parallel_worker_init,
            initargs=(model_config, train_config, step_scaler, str(device), "source"),
        ) as executor:
            for index, (parameter_vector, metrics) in enumerate(
                executor.map(_parallel_fit_worker, payloads, chunksize=chunksize),
                start=1,
            ):
                parameter_vectors.append(parameter_vector)
                metrics_records.append(metrics)
                if index == 1 or index % 200 == 0 or index == total_tasks:
                    print("trained source tasks {}/{}".format(index, total_tasks))

    parameter_matrix = np.stack(parameter_vectors, axis=0).astype(np.float32)
    np.save(_resolve_path(base_dir, train_config["source_parameter_vector_path"]), parameter_matrix)
    _save_metrics_csv(metrics_records, _resolve_path(base_dir, train_config["source_metrics_path"]))

    summary = {
        "source_task_count": int(total_tasks),
        "parameter_dim": int(parameter_matrix.shape[1]),
        "mean_train_mse": float(np.mean([record["train"]["mse"] for record in metrics_records])),
        "mean_train_rmse": float(np.mean([record["train"]["rmse"] for record in metrics_records])),
        "mean_train_mae": float(np.mean([record["train"]["mae"] for record in metrics_records])),
        "mean_train_r2": float(np.mean([record["train"]["r2"] for record in metrics_records if record["train"]["r2"] is not None])),
    }
    _save_json(summary, _resolve_path(base_dir, train_config["source_summary_path"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_parameter_matrix(parameter_path):
    matrix = np.load(parameter_path)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim == 2:
        pass
    elif matrix.ndim == 3:
        matrix = matrix.reshape(matrix.shape[0], -1)
    else:
        raise ValueError("Unsupported parameter tensor shape: {}".format(matrix.shape))
    return matrix


def _select_torch_device(train_config):
    device_name = str(train_config.get("device", "cpu")).lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device=cuda, but CUDA is not available")
    return torch.device(device_name)


def _aggregate_target_report(metrics_records):
    report = {
        "task_count": int(len(metrics_records)),
    }
    for split_name in ("train", "val", "test"):
        available = [record[split_name] for record in metrics_records if record.get(split_name) is not None]
        if not available:
            report[split_name] = None
            continue

        y_true = []
        y_pred = []
        for split_metrics in available:
            y_true.extend(split_metrics["y_true"])
            y_pred.extend(split_metrics["y_pred"])

        point_metrics = _regression_summary(np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32))
        task_r2 = [split_metrics["r2"] for split_metrics in available if split_metrics["r2"] is not None]
        report[split_name] = {
            "point_level": point_metrics,
            "task_level": {
                "mean_mse": float(np.mean([split_metrics["mse"] for split_metrics in available])),
                "mean_rmse": float(np.mean([split_metrics["rmse"] for split_metrics in available])),
                "mean_mae": float(np.mean([split_metrics["mae"] for split_metrics in available])),
                "mean_r2": float(np.mean(task_r2)) if task_r2 else None,
                "median_r2": float(np.median(task_r2)) if task_r2 else None,
            },
        }
    return report


def finetune_target_domain_models(config, diffusion_sample_path, limit_tasks=None):
    base_dir = config["_config_dir"]
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    train_config = config.get("train", {})

    target_tasks = _load_task_bundle(_resolve_path(base_dir, data_config["target_task_bundle_path"]))
    if limit_tasks is not None:
        target_tasks = target_tasks[:limit_tasks]

    step_scaler = _load_pickle(_resolve_path(base_dir, data_config["step_scaler_path"]))
    parameter_matrix = _load_parameter_matrix(diffusion_sample_path)
    if parameter_matrix.shape[0] not in {1, len(target_tasks)}:
        raise ValueError(
            "Generated parameter rows {} do not match target task count {}".format(
                parameter_matrix.shape[0],
                len(target_tasks),
            )
        )

    if parameter_matrix.shape[0] == 1 and len(target_tasks) > 1:
        parameter_matrix = np.repeat(parameter_matrix, len(target_tasks), axis=0)

    device = _select_torch_device(train_config)
    print("target domain finetuning device: {}".format(device))
    metrics_records = []
    finetuned_vectors = []

    total_tasks = len(target_tasks)
    worker_count = _resolve_parallel_workers(train_config, total_tasks, device)
    if worker_count == 1:
        for index, task in enumerate(target_tasks, start=1):
            parameter_vector, metrics = _fit_target_task_with_selection(
                task=task,
                model_config=model_config,
                train_config=train_config,
                step_scaler=step_scaler,
                device=device,
                generated_init_vector=parameter_matrix[index - 1],
            )
            finetuned_vectors.append(parameter_vector)
            metrics_records.append(metrics)

            if index == 1 or index % 200 == 0 or index == total_tasks:
                print("finetuned target tasks {}/{}".format(index, total_tasks))
    else:
        print("target domain parallel workers: {}".format(worker_count))
        payloads = [
            {
                "task": task,
                "generated_init_vector": parameter_matrix[index - 1],
            }
            for index, task in enumerate(target_tasks, start=1)
        ]
        chunksize = _process_pool_chunksize(total_tasks, worker_count)
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_parallel_worker_init,
            initargs=(model_config, train_config, step_scaler, str(device), "target"),
        ) as executor:
            for index, (parameter_vector, metrics) in enumerate(
                executor.map(_parallel_fit_worker, payloads, chunksize=chunksize),
                start=1,
            ):
                finetuned_vectors.append(parameter_vector)
                metrics_records.append(metrics)
                if index == 1 or index % 200 == 0 or index == total_tasks:
                    print("finetuned target tasks {}/{}".format(index, total_tasks))

    finetuned_matrix = np.stack(finetuned_vectors, axis=0).astype(np.float32)
    np.save(_resolve_path(base_dir, train_config["target_finetuned_params_path"]), finetuned_matrix)
    _save_metrics_csv(metrics_records, _resolve_path(base_dir, train_config["target_metrics_path"]))

    report = _aggregate_target_report(metrics_records)
    report["target_task_count"] = int(total_tasks)
    report["parameter_dim"] = int(finetuned_matrix.shape[1])
    _save_json(report, _resolve_path(base_dir, train_config["target_report_path"]))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_curve_transfer_full_pipeline(config, python_executable=None):
    prepare_curve_domains(config)
    train_source_domain_models(config)

    base_dir = Path(config["_config_dir"])
    data_config = config.get("data", {})
    train_config = config.get("train", {})
    gpd_config = config.get("gpd", {})

    python_executable = python_executable or sys.executable
    gpd_dir = (base_dir.parent / "GPD").resolve()
    exp_index = int(gpd_config.get("exp_index", 9401))

    command = [
        python_executable,
        "1Dmain.py",
        "--modeldim",
        str(int(gpd_config.get("modeldim", 16))),
        "--epochs",
        str(int(gpd_config.get("epochs", 200))),
        "--expIndex",
        str(exp_index),
        "--diffusionstep",
        str(int(gpd_config.get("diffusionstep", 20))),
        "--denoise",
        str(gpd_config.get("denoise", "Trans3")),
        "--trainbatchsize",
        str(int(gpd_config.get("trainbatchsize", 1))),
        "--samplebatchsize",
        str(int(gpd_config.get("samplebatchsize", 64))),
        "--repeat_num",
        str(int(gpd_config.get("repeat_num", 1))),
        "--train_param_path",
        str(_resolve_path(base_dir, train_config["source_parameter_vector_path"])),
        "--train_condition_csv",
        str(_resolve_path(base_dir, data_config["source_condition_csv"])),
        "--sample_condition_csv",
        str(_resolve_path(base_dir, data_config["target_condition_csv"])),
        "--retention_scaler_path",
        str(_resolve_path(base_dir, data_config["retention_scaler_path"])),
        "--pec_scaler_path",
        str(_resolve_path(base_dir, data_config["pec_scaler_path"])),
        "--wl_vocab_path",
        str(_resolve_path(base_dir, data_config["wl_vocab_path"])),
    ]
    subprocess.run(command, cwd=str(gpd_dir), check=True)

    diffusion_sample_path = gpd_dir / "Output" / "sampleSeq_RealParams_{}.npy".format(exp_index)
    finetune_target_domain_models(config, diffusion_sample_path)
