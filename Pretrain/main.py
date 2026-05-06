import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from curve_task_workflow import (
    finetune_target_domain_models,
    prepare_curve_domains,
    run_curve_transfer_full_pipeline,
    train_source_domain_models,
)
from datasets import build_dataloaders
from Models import MLPRegressor


def resolve_path(base_dir, path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_local_config(config_filename):
    config_path = Path(config_filename)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parent / config_filename).resolve()

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.full_load(handle)

    base_dir = config_path.parent
    config["_config_dir"] = str(base_dir)
    data_config = config.get("data", {})
    train_config = config.get("train", {})

    path_keys = [
        "csv_path",
        "retention_scaler_path",
        "pec_scaler_path",
        "feature_scaler_path",
        "wl_vocab_path",
        "processed_csv_path",
        "artifacts_dir",
    ]
    for key in path_keys:
        if key in data_config:
            data_config[key] = str(resolve_path(base_dir, data_config[key]))

    train_path_keys = ["checkpoint_path", "parameter_vector_path", "metrics_path"]
    for key in train_path_keys:
        if key in train_config:
            train_config[key] = str(resolve_path(base_dir, train_config[key]))

    return config


def build_model(model_config):
    model_type = model_config.get("type", "MLP")
    if model_type != "MLP":
        raise ValueError("当前仅支持 MLP，收到 model.type={}".format(model_type))

    return MLPRegressor(
        in_dim=model_config.get("in_dim", 4),
        hidden_dims=tuple(model_config.get("hidden_dims", [64, 32])),
        out_dim=model_config.get("out_dim", 1),
        activation=model_config.get("activation", "relu"),
    )


def run_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    if dataloader is None:
        return None

    model.eval()
    total_loss = 0.0
    total_count = 0
    predictions_all = []
    targets_all = []

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        predictions = model(features)
        loss = criterion(predictions, targets)

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

        predictions_all.append(predictions.detach().cpu())
        targets_all.append(targets.detach().cpu())

    metrics = {
        "loss": total_loss / max(total_count, 1),
        "predictions": torch.cat(predictions_all, dim=0).numpy() if predictions_all else np.empty((0, 1)),
        "targets": torch.cat(targets_all, dim=0).numpy() if targets_all else np.empty((0, 1)),
    }
    return metrics


def summarize_regression_metrics(metrics):
    predictions = metrics.get("predictions")
    targets = metrics.get("targets")
    if predictions is None or targets is None or len(predictions) == 0:
        return {
            "loss": metrics.get("loss"),
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2": None,
        }

    y_pred = np.asarray(predictions).reshape(-1)
    y_true = np.asarray(targets).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "loss": float(metrics.get("loss")),
        "mse": float(mse),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_artifacts(model, train_config, metrics):
    checkpoint_path = Path(train_config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metrics": metrics}, checkpoint_path)

    parameter_vector_path = Path(train_config["parameter_vector_path"])
    parameter_vector_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_vector = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    np.save(parameter_vector_path, parameter_vector)

    metrics_path = Path(train_config["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"predictions", "targets"}
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable_metrics, handle, ensure_ascii=False, indent=2)


def load_parameter_vector(parameter_path, sample_index=0):
    parameter_array = np.load(parameter_path)
    if parameter_array.ndim >= 2:
        parameter_array = parameter_array.reshape(parameter_array.shape[0], -1)
        if sample_index >= parameter_array.shape[0]:
            raise IndexError(
                "sample_index={} 超出参数样本数量 {}".format(sample_index, parameter_array.shape[0])
            )
        parameter_array = parameter_array[sample_index]
    return torch.as_tensor(parameter_array, dtype=torch.float32)


def maybe_initialize_from_generated_parameters(model, diffusion_sample_path, sample_index, device):
    if not diffusion_sample_path:
        return

    parameter_vector = load_parameter_vector(diffusion_sample_path, sample_index=sample_index).to(device)
    expected_dim = parameters_to_vector(model.parameters()).numel()
    if parameter_vector.numel() != expected_dim:
        raise ValueError(
            "扩散参数长度不匹配: 期望 {}, 实际 {}".format(expected_dim, parameter_vector.numel())
        )
    vector_to_parameters(parameter_vector, model.parameters())


def train_model(args):
    config = load_local_config(args.config_filename)
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    train_config = config.get("train", {}).copy()

    if args.epochs is not None:
        train_config["epochs"] = args.epochs

    loaders, metadata = build_dataloaders(
        csv_path=data_config["csv_path"],
        batch_size=int(data_config.get("batch_size", 32)),
        feature_columns=data_config.get("feature_columns"),
        target_column=data_config.get("target_column", "\u9891\u6570"),
        scale=bool(data_config.get("scale", True)),
        split_config=data_config.get("split"),
        seed=int(data_config.get("seed", 42)),
        retention_scaler_path=data_config["retention_scaler_path"],
        pec_scaler_path=data_config["pec_scaler_path"],
        feature_scaler_path=data_config.get("feature_scaler_path"),
        feature_scale_method=data_config.get("feature_scale_method", "standard"),
        wl_vocab_path=data_config["wl_vocab_path"],
        processed_csv_path=data_config.get("processed_csv_path"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config).to(device)

    if args.mode == "finetune":
        maybe_initialize_from_generated_parameters(
            model=model,
            diffusion_sample_path=args.diffusion_sample_path,
            sample_index=args.sample_index,
            device=device,
        )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.get("lr", 1e-4)))

    epochs = int(train_config.get("epochs", 100))
    print_every = int(train_config.get("print_every", 10))

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        val_loss = val_metrics["loss"] if val_metrics is not None else train_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % print_every == 0 or epoch == 0 or epoch + 1 == epochs:
            print(
                "Epoch {}/{} | train_loss={:.6f} | val_loss={:.6f}".format(
                    epoch + 1,
                    epochs,
                    train_loss,
                    val_loss,
                )
            )

    model.load_state_dict(best_state)

    val_metrics = evaluate(model, loaders["val"], criterion, device) or {"loss": None}
    test_metrics = evaluate(model, loaders["test"], criterion, device) or {"loss": None}
    val_summary = summarize_regression_metrics(val_metrics)
    test_summary = summarize_regression_metrics(test_metrics)
    final_metrics = {
        "mode": args.mode,
        "train_loss_last_epoch": float(train_loss),
        "best_val_loss": float(best_val_loss),
        "val_loss": val_summary["loss"],
        "val_mse": val_summary["mse"],
        "val_rmse": val_summary["rmse"],
        "val_mae": val_summary["mae"],
        "val_r2": val_summary["r2"],
        "test_loss": test_summary["loss"],
        "test_mse": test_summary["mse"],
        "test_rmse": test_summary["rmse"],
        "test_mae": test_summary["mae"],
        "test_r2": test_summary["r2"],
        "num_wl": len(metadata["wl_values"]),
        "parameter_dim": int(parameters_to_vector(model.parameters()).numel()),
        "dataset_size": int(metadata["features"].shape[0]),
        "train_size": int(len(metadata["indices"]["train"])),
        "val_size": int(len(metadata["indices"]["val"])),
        "test_size": int(len(metadata["indices"]["test"])),
    }

    save_artifacts(model, train_config, final_metrics)
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


def extract_parameter_vector(args):
    config = load_local_config(args.config_filename)
    model_config = config.get("model", {})
    train_config = config.get("train", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config).to(device)

    checkpoint = torch.load(train_config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    parameter_vector = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    output_path = Path(train_config["parameter_vector_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, parameter_vector)
    print("saved parameter vector to {}".format(output_path))


def main():
    parser = argparse.ArgumentParser(description="3D NAND 参数回归训练入口")
    parser.add_argument("--config_filename", default="config.yaml", type=str)
    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "finetune",
            "extract",
            "prepare_domains",
            "train_source_domain",
            "finetune_target_domain",
            "full_curve_transfer",
        ],
        default="train",
    )
    parser.add_argument("--diffusion_sample_path", default="", type=str)
    parser.add_argument("--sample_index", default=0, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--limit_source_tasks", default=None, type=int)
    parser.add_argument("--limit_target_tasks", default=None, type=int)
    args = parser.parse_args()

    if args.mode == "prepare_domains":
        config = load_local_config(args.config_filename)
        prepare_curve_domains(config)
    elif args.mode == "train_source_domain":
        config = load_local_config(args.config_filename)
        train_source_domain_models(config, limit_tasks=args.limit_source_tasks)
    elif args.mode == "finetune_target_domain":
        config = load_local_config(args.config_filename)
        if not args.diffusion_sample_path:
            raise ValueError("--diffusion_sample_path is required for finetune_target_domain")
        finetune_target_domain_models(
            config,
            diffusion_sample_path=args.diffusion_sample_path,
            limit_tasks=args.limit_target_tasks,
        )
    elif args.mode == "full_curve_transfer":
        config = load_local_config(args.config_filename)
        run_curve_transfer_full_pipeline(config)
    elif args.mode == "extract":
        extract_parameter_vector(args)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
