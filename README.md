# 3D-NAND-GPO

3D NAND flash parameter regression and parameter generation pipeline built on top of the original GPD codebase.

This repository has been adapted from the original spatio-temporal GPD framework to a 3D NAND setting. The active workflow is no longer traffic or crowd-flow forecasting. Instead, it focuses on:

- supervised regression from process and device features to target frequency
- parameter-vector extraction from the regression model
- diffusion-based generation of model parameters under NAND conditions
- optional fine-tuning from generated parameters

## Overview

The current 3D NAND workflow uses these inputs:

- features: `步长`, `WL`, `Retention`, `PEC`
- target: `频数`

The migration includes these core changes:

- `Pretrain/datasets.py` now reads CSV data, builds `DataLoader`s, and reshapes inputs to `(N, 1, 4)`
- `Pretrain/Models/mlp_model.py` defines the regression backbone as `4 -> 64 -> 32 -> 1`
- `Pretrain/main.py` trains or fine-tunes the regression model with `nn.MSELoss` and `Adam(lr=1e-4)`
- parameter extraction and parameter loading use `torch.nn.utils.parameters_to_vector` and `vector_to_parameters`
- diffusion conditioning removes KG embeddings and uses:
  - `WL` as an embedding
  - `Retention` normalized to `[-1, 1]`
  - `PEC` normalized to `[-1, 1]`
- transformer conditioning is injected by concatenating the condition vector to the noisy parameter tokens

## Active Files

The main 3D NAND path is:

- `Pretrain/config.yaml`
- `Pretrain/datasets.py`
- `Pretrain/main.py`
- `Pretrain/Models/mlp_model.py`
- `Pretrain/PrepareParams/model2tensor.py`
- `GPD/datapreparing.py`
- `GPD/1Dmain.py`
- `GPD/TimeTransformer/transformer.py`
- `GPD/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py`

Some legacy graph-model files from the upstream project are still present for reference, but they are not part of the active 3D NAND pipeline.

## Project Structure

```text
GPD-master/
|-- Data/
|-- Pretrain/
|   |-- Models/
|   |   `-- mlp_model.py
|   |-- PrepareParams/
|   |   `-- model2tensor.py
|   |-- config.yaml
|   |-- datasets.py
|   `-- main.py
|-- GPD/
|   |-- 1Dmain.py
|   |-- datapreparing.py
|   |-- TimeTransformer/
|   `-- denoising_diffusion_pytorch/
|-- assets/
|-- requirements.txt
`-- README.md
```

## Environment

- Python >= 3.8
- PyTorch
- pandas
- scikit-learn
- PyYAML
- accelerate
- einops
- ema-pytorch
- tqdm

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data Format

The default regression config points to:

```text
Data/nand_regression.csv
```

The CSV should contain at least these columns:

```text
步长, WL, Retention, PEC, 频数
```

Notes:

- `Retention` and `PEC` are normalized independently and their scalers are saved for reuse
- `WL` is also mapped to an index vocabulary for diffusion conditioning
- generated artifacts are written under `Pretrain/artifacts/`

## Default Regression Config

`Pretrain/config.yaml` is configured for:

- `features: 4`
- `target_dim: 1`
- `batch_size: 32`
- `scale: true`
- `hidden_dims: [64, 32]`
- `in_dim: 4`
- `out_dim: 1`
- `num_nodes: 1`

## Workflow

### 1. Train the MLP regressor

From the repository root:

```bash
cd Pretrain
python main.py --mode train
```

This will:

- read the CSV specified in `Pretrain/config.yaml`
- split the dataset into train/val/test
- train the MLP regressor
- save the best checkpoint
- export a parameter vector for the trained model

Main outputs:

- `Pretrain/artifacts/mlp_regressor.pt`
- `Pretrain/artifacts/mlp_regressor_params.npy`
- `Pretrain/artifacts/processed_conditions.csv`
- `Pretrain/artifacts/retention_scaler.pkl`
- `Pretrain/artifacts/pec_scaler.pkl`
- `Pretrain/artifacts/wl_vocab.json`

### 2. Collect parameter vectors for diffusion training

If you have multiple checkpoints and want to build a parameter dataset:

```bash
cd Pretrain
python PrepareParams/model2tensor.py --checkpoint_dir ./artifacts --output_path ./PrepareParams/model_params.npy
```

This stacks model parameters into a 2D array of shape:

```text
(num_models, parameter_dim)
```

### 3. Train the diffusion model

From the repository root:

```bash
cd GPD
python 1Dmain.py --expIndex 888 --epochs 20000 --diffusionstep 500 --denoise Trans3
```

By default, this script reads:

- parameter vectors from `../Pretrain/PrepareParams/model_params.npy`
- processed condition CSV from `../Pretrain/artifacts/processed_conditions.csv`
- saved scalers and `WL` vocabulary from `../Pretrain/artifacts/`

Condition construction is:

- `WL -> nn.Embedding(num_wl, 8)`
- `Retention -> normalized to [-1, 1]`
- `PEC -> normalized to [-1, 1]`

The final condition vector is:

```text
[WL_embedding, Retention_norm, PEC_norm]
```

Generated samples are saved to:

```text
GPD/Output/sampleSeq_RealParams_<expIndex>.npy
```

### 4. Fine-tune from generated parameters

After diffusion sampling:

```bash
cd Pretrain
python main.py --mode finetune --diffusion_sample_path ../GPD/Output/sampleSeq_RealParams_888.npy
```

Optional arguments:

- `--sample_index`: choose which generated parameter vector to load when the `.npy` file contains multiple samples
- `--epochs`: override the epoch count from `config.yaml`

## Notes

- Large data files, `*.npy`, model weights, logs, and outputs are ignored by `.gitignore` by default
- The repository currently tracks code and lightweight assets only
- If you want to version large datasets or checkpoints, use Git LFS instead of ordinary Git

## Migration Summary

Compared with the original upstream GPD repository, this version changes the problem formulation from graph forecasting to 3D NAND regression:

- graph-structured time-series input has been replaced by tabular NAND features
- the backbone is now a lightweight MLP regressor
- KG conditioning has been removed
- diffusion conditions now depend on `WL`, `Retention`, and `PEC`
- parameter serialization is handled through official PyTorch vector utilities

## Acknowledgment

This repository is adapted from the original GPD project and reworked for a 3D NAND flash parameter-regression and parameter-generation workflow.
