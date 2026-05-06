import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset


FEATURE_COLUMNS = ["\u6b65\u957f", "WL", "Retention", "PEC"]
TARGET_COLUMN = "\u9891\u6570"


class NandRegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def _resolve_path(path_like):
    return Path(path_like).expanduser().resolve()


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


def save_wl_vocab(values, path_like):
    path = _ensure_parent(path_like)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(values), handle, ensure_ascii=False, indent=2)


def load_wl_vocab(path_like):
    with Path(path_like).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_saved_scaler(path_like):
    return _load_pickle(path_like)


def inverse_transform_column(values, scaler):
    array = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(array).reshape(-1)


def _validate_columns(frame, feature_columns, target_column):
    required = list(feature_columns) + [target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError("CSV 缺少必要列: {}".format(", ".join(missing)))


def _fit_or_load_scalers(frame, scale, retention_scaler_path, pec_scaler_path):
    if not scale:
        return None, None

    retention_path = _resolve_path(retention_scaler_path)
    pec_path = _resolve_path(pec_scaler_path)

    retention_scaler = MinMaxScaler(feature_range=(-1, 1))
    pec_scaler = MinMaxScaler(feature_range=(-1, 1))

    retention_scaler.fit(frame[["Retention"]])
    pec_scaler.fit(frame[["PEC"]])

    _save_pickle(retention_scaler, retention_path)
    _save_pickle(pec_scaler, pec_path)
    return retention_scaler, pec_scaler


def _fit_feature_scaler(frame, feature_columns, feature_scaler_path=None, feature_scale_method="standard"):
    if not feature_scaler_path or feature_scale_method in {None, "", "none"}:
        return None

    scaler_path = _resolve_path(feature_scaler_path)
    if feature_scale_method == "standard":
        scaler = StandardScaler()
    elif feature_scale_method == "minmax":
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError("Unsupported feature_scale_method: {}".format(feature_scale_method))

    scaler.fit(frame[feature_columns])
    _save_pickle(scaler, scaler_path)
    return scaler


def preprocess_dataframe(
    csv_path,
    feature_columns=None,
    target_column=TARGET_COLUMN,
    scale=True,
    retention_scaler_path="artifacts/retention_scaler.pkl",
    pec_scaler_path="artifacts/pec_scaler.pkl",
    wl_vocab_path="artifacts/wl_vocab.json",
    processed_csv_path=None,
    feature_scaler_path=None,
    feature_scale_method="standard",
):
    feature_columns = feature_columns or FEATURE_COLUMNS
    frame = pd.read_csv(_resolve_path(csv_path)).copy()
    _validate_columns(frame, feature_columns, target_column)

    retention_scaler, pec_scaler = _fit_or_load_scalers(
        frame,
        scale,
        retention_scaler_path,
        pec_scaler_path,
    )

    processed = frame.copy()
    if scale:
        processed["Retention"] = (
            retention_scaler.transform(frame[["Retention"]]).astype(np.float32).reshape(-1)
        )
        processed["PEC"] = (
            pec_scaler.transform(frame[["PEC"]]).astype(np.float32).reshape(-1)
        )

    wl_values = sorted(pd.Series(frame["WL"]).dropna().unique().tolist())
    save_wl_vocab(wl_values, _resolve_path(wl_vocab_path))
    wl_mapping = {str(value): index for index, value in enumerate(wl_values)}
    processed["WL_index"] = frame["WL"].astype(str).map(wl_mapping).astype(np.int64)

    feature_scaler = _fit_feature_scaler(
        frame=frame,
        feature_columns=feature_columns,
        feature_scaler_path=feature_scaler_path,
        feature_scale_method=feature_scale_method,
    )

    if processed_csv_path:
        processed_path = _resolve_path(processed_csv_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(processed_path, index=False, encoding="utf-8-sig")

    feature_frame = frame[feature_columns].to_numpy(dtype=np.float32)
    if feature_scaler is not None:
        feature_frame = feature_scaler.transform(feature_frame).astype(np.float32)

    features = feature_frame
    features = np.expand_dims(features, axis=1)
    targets = processed[[target_column]].to_numpy(dtype=np.float32)
    conditions = np.column_stack(
        [
            processed["WL_index"].to_numpy(dtype=np.float32),
            processed["Retention"].to_numpy(dtype=np.float32),
            processed["PEC"].to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)

    metadata = {
        "frame": processed,
        "retention_scaler": retention_scaler,
        "pec_scaler": pec_scaler,
        "feature_scaler": feature_scaler,
        "wl_values": wl_values,
        "conditions": conditions,
    }
    return features, targets, metadata


def _split_indices(length, split_config=None, seed=42):
    split_config = split_config or {}
    train_ratio = float(split_config.get("train", 0.7))
    val_ratio = float(split_config.get("val", 0.15))
    test_ratio = float(split_config.get("test", 0.15))

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("train/val/test 划分比例之和必须为 1.0，当前为 {}".format(ratio_sum))

    rng = np.random.default_rng(seed)
    indices = rng.permutation(length)

    train_end = int(length * train_ratio)
    val_end = train_end + int(length * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    return train_indices, val_indices, test_indices


def _make_loader(features, targets, indices, batch_size, shuffle):
    if len(indices) == 0:
        return None
    dataset = NandRegressionDataset(features[indices], targets[indices])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_dataloaders(
    csv_path,
    batch_size=32,
    feature_columns=None,
    target_column=TARGET_COLUMN,
    scale=True,
    split_config=None,
    seed=42,
    retention_scaler_path="artifacts/retention_scaler.pkl",
    pec_scaler_path="artifacts/pec_scaler.pkl",
    wl_vocab_path="artifacts/wl_vocab.json",
    processed_csv_path=None,
    feature_scaler_path=None,
    feature_scale_method="standard",
):
    features, targets, metadata = preprocess_dataframe(
        csv_path=csv_path,
        feature_columns=feature_columns,
        target_column=target_column,
        scale=scale,
        retention_scaler_path=retention_scaler_path,
        pec_scaler_path=pec_scaler_path,
        wl_vocab_path=wl_vocab_path,
        processed_csv_path=processed_csv_path,
        feature_scaler_path=feature_scaler_path,
        feature_scale_method=feature_scale_method,
    )

    train_indices, val_indices, test_indices = _split_indices(
        length=features.shape[0],
        split_config=split_config,
        seed=seed,
    )

    loaders = {
        "train": _make_loader(features, targets, train_indices, batch_size, True),
        "val": _make_loader(features, targets, val_indices, batch_size, False),
        "test": _make_loader(features, targets, test_indices, batch_size, False),
    }
    metadata["indices"] = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    metadata["features"] = features
    metadata["targets"] = targets
    return loaders, metadata
