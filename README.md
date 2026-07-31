# Device-Condition-Informed Diffusion Parameter Transfer for 3-D NAND

This repository contains the code and processed numerical data associated with the manuscript **“Device-Condition-Informed Diffusion Parameter Transfer for Sparse Reconstruction of 3-D NAND Threshold-Voltage Distributions.”** The study investigates sparse reconstruction of threshold-voltage distributions for the seven programmed TLC states (P1–P7) using a device-condition-informed diffusion parameter prior and target-device adaptation.

## Repository contents

- `Data/split_by_model/`: processed device-condition data organized by device/model identifier.
- `GPD/`: conditional diffusion parameter-generation implementation and supporting model components.
- `Pretrain/`: source-model training, target adaptation, and sparsity/initialization benchmarks.
- `analysis/`: fixed-budget evaluation, comparison baselines, ablation studies, and figure-data processing scripts used for the manuscript.
- `paper_data/`: processed numerical data directly supporting the reported tables and figures.

The repository intentionally excludes the manuscript source, submission forms, local execution logs, temporary files, and unrelated project material.

## Manuscript result mapping

| Manuscript item | Supporting files |
| --- | --- |
| Per-state reconstruction and comparison results | `paper_data/task2_scale_summary.csv`, `paper_data/task2_full_family_baselines.csv` |
| Fixed-budget 7/9-point protocol | `paper_data/task2_fixed_anchor_protocol_representative_with_conditional.csv`, `paper_data/task2_fixed_anchor_protocol_representative_task_metrics.csv` |
| Hierarchical transfer ablation | `paper_data/task2_hierarchy_unfactorized_ablation_representative_summary.csv`, `paper_data/task2_full_family_baseline_task_metrics.csv` |
| Predictor attribution | `paper_data/task1_shap_feature_importance.csv`, `paper_data/task1_shap_explained_samples.csv`, `paper_data/task1_shap_run_metadata.json` |
| PEC-dependent reconstruction case study | `paper_data/task2_physical_consistency_curves.csv`, `paper_data/task2_physical_consistency_ladder.csv` |
| Extreme-sparsity analysis | `paper_data/task2_extreme_sparsity_model3.csv` |
| Residual analysis | `paper_data/task2_residual_heatmap.csv` |

## Environment

Python 3.9 or a compatible environment is recommended. Install the main dependencies with:

```bash
pip install -r requirements.txt
```

PyTorch should be installed with the CPU or CUDA build appropriate for the local system. The versions in `requirements.txt` reflect the main development environment.

## Main workflows

1. Prepare or inspect the processed device data under `Data/split_by_model/`.
2. Train or evaluate the diffusion parameter generator using the scripts in `GPD/`.
3. Run source training and target adaptation through `Pretrain/curve_task_workflow.py` and the task-specific entry points in `Pretrain/`.
4. Reproduce fixed-budget comparisons and ablations with the scripts in `analysis/`.
5. Use the processed outputs in `paper_data/` to verify the numerical values reported in the manuscript.

`analysis/make_task2_depth_figures.py` reads the processed inputs from `paper_data/` and writes regenerated graphics to `generated_figures/`.

The exact experiment commands depend on the local data and checkpoint locations. Each entry-point script exposes its configurable paths and experiment settings through command-line arguments or constants near the top of the file.

## Interpretation notes

- SHAP values in this study quantify predictor attribution and are not interpreted as causal physical mechanisms.
- The fixed-budget evaluation uses controlled representative subsets with peak-informed anchor selection, as described in the manuscript.
- Performance claims are limited to the reported datasets, device conditions, and evaluation protocols.

## Authors

Chunru Xiong, Zongzheng Li, Qiang Li, and Haihua Hu.
