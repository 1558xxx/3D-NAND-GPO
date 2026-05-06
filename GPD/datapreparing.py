import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_path(path_like):
    return Path(path_like).expanduser().resolve()


def _load_pickle(path_like):
    with _resolve_path(path_like).open("rb") as handle:
        return pickle.load(handle)


def _load_wl_mapping(wl_vocab_path, frame):
    if wl_vocab_path and _resolve_path(wl_vocab_path).exists():
        with _resolve_path(wl_vocab_path).open("r", encoding="utf-8") as handle:
            wl_values = json.load(handle)
    else:
        wl_values = sorted(pd.Series(frame["WL"]).dropna().unique().tolist())

    mapping = {str(value): index for index, value in enumerate(wl_values)}
    return mapping, len(wl_values)


def _normalize_conditions(frame, retention_scaler_path, pec_scaler_path, wl_vocab_path=None):
    required_columns = ["WL", "Retention", "PEC"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError("Condition CSV is missing required columns: {}".format(", ".join(missing)))

    retention_scaler = _load_pickle(retention_scaler_path)
    pec_scaler = _load_pickle(pec_scaler_path)
    wl_mapping, num_wl = _load_wl_mapping(wl_vocab_path, frame)

    wl_indices = frame["WL"].astype(str).map(wl_mapping)
    if wl_indices.isnull().any():
        unknown_values = frame.loc[wl_indices.isnull(), "WL"].drop_duplicates().tolist()
        raise ValueError("Found WL values that are not in the saved vocabulary: {}".format(unknown_values))

    retention_norm = retention_scaler.transform(frame[["Retention"]]).astype(np.float32).reshape(-1)
    pec_norm = pec_scaler.transform(frame[["PEC"]]).astype(np.float32).reshape(-1)

    condition = np.column_stack(
        [
            wl_indices.to_numpy(dtype=np.float32),
            retention_norm,
            pec_norm,
        ]
    ).astype(np.float32)
    return condition, num_wl


def _load_parameter_sequences(parameter_path):
    params = np.load(_resolve_path(parameter_path))
    params = np.asarray(params, dtype=np.float32)

    if params.ndim == 1:
        params = params.reshape(1, 1, -1)
    elif params.ndim == 2:
        params = params[:, None, :]
    elif params.ndim == 3:
        params = params.reshape(params.shape[0], 1, -1)
    else:
        raise ValueError("Unsupported parameter tensor shape: {}".format(params.shape))

    return params


def _align_parameter_rows(params, target_rows):
    current_rows = int(params.shape[0])
    target_rows = int(target_rows)

    if current_rows == target_rows:
        return params

    if current_rows == 1 and target_rows > 1:
        return np.repeat(params, target_rows, axis=0)

    raise ValueError(
        "Parameter sample count does not match condition count: {} vs {}".format(current_rows, target_rows)
    )


def datapreparing(
    train_param_path,
    train_condition_csv,
    retention_scaler_path,
    pec_scaler_path,
    wl_vocab_path=None,
    sample_condition_csv=None,
    sample_target_param_path=None,
    repeat_num=1,
):
    train_parameters = _load_parameter_sequences(train_param_path)
    train_frame = pd.read_csv(_resolve_path(train_condition_csv))
    train_condition, num_wl = _normalize_conditions(
        frame=train_frame,
        retention_scaler_path=retention_scaler_path,
        pec_scaler_path=pec_scaler_path,
        wl_vocab_path=wl_vocab_path,
    )
    train_parameters = _align_parameter_rows(train_parameters, train_condition.shape[0])

    repeat_num = max(int(repeat_num), 1)
    if repeat_num > 1:
        train_parameters = np.repeat(train_parameters, repeat_num, axis=0)
        train_condition = np.repeat(train_condition, repeat_num, axis=0)

    sample_condition_csv = sample_condition_csv or train_condition_csv
    sample_frame = pd.read_csv(_resolve_path(sample_condition_csv))
    sample_condition, _ = _normalize_conditions(
        frame=sample_frame,
        retention_scaler_path=retention_scaler_path,
        pec_scaler_path=pec_scaler_path,
        wl_vocab_path=wl_vocab_path,
    )

    gen_target = None
    if sample_target_param_path:
        gen_target = _load_parameter_sequences(sample_target_param_path)
    elif _resolve_path(sample_condition_csv) == _resolve_path(train_condition_csv):
        gen_target = _load_parameter_sequences(train_param_path)

    if gen_target is not None:
        gen_target = _align_parameter_rows(gen_target, sample_condition.shape[0])

    metadata = {
        "num_wl": num_wl,
        "parameter_dim": int(train_parameters.shape[-1]),
        "train_size": int(train_parameters.shape[0]),
        "sample_size": int(sample_condition.shape[0]),
    }
    return train_parameters, train_condition, sample_condition, gen_target, metadata
