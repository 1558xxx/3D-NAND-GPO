"""
Collect trained MLP checkpoints into a parameter-vector matrix for diffusion training.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.utils import parameters_to_vector

CURRENT_DIR = Path(__file__).resolve().parent
PRETRAIN_DIR = CURRENT_DIR.parent
if str(PRETRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_DIR))

from Models import MLPRegressor


def resolve_path(base_dir, path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (CURRENT_DIR / config_path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.full_load(handle)
    return config, config_path.parent


def build_model(model_config):
    return MLPRegressor(
        in_dim=model_config.get("in_dim", 4),
        hidden_dims=tuple(model_config.get("hidden_dims", [64, 32])),
        out_dim=model_config.get("out_dim", 1),
    )


def load_state_dict_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("无法识别的 checkpoint 格式: {}".format(checkpoint_path))


def main():
    parser = argparse.ArgumentParser(description="Extract parameter vectors from MLP checkpoints")
    parser.add_argument("--config", default="../config.yaml", type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--suffix", default=".pt", type=str)
    args = parser.parse_args()

    config, config_dir = load_config(args.config)
    model_config = config.get("model", {})

    checkpoint_dir = resolve_path(config_dir, args.checkpoint_dir)
    output_path = resolve_path(config_dir, args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameter_vectors = []
    checkpoint_files = sorted(checkpoint_dir.rglob("*{}".format(args.suffix)))
    if not checkpoint_files:
        raise FileNotFoundError("未在 {} 下找到后缀为 {} 的 checkpoint".format(checkpoint_dir, args.suffix))

    for checkpoint_path in checkpoint_files:
        model = build_model(model_config)
        model.load_state_dict(load_state_dict_from_checkpoint(checkpoint_path))
        parameter_vector = parameters_to_vector(model.parameters()).detach().cpu().numpy()
        parameter_vectors.append(parameter_vector)

    stacked = np.stack(parameter_vectors, axis=0).astype(np.float32)
    np.save(output_path, stacked)
    print("saved {} parameter vectors to {}".format(stacked.shape[0], output_path))


if __name__ == "__main__":
    main()
