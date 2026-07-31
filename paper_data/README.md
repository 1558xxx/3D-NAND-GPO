# Processed manuscript data

This directory contains processed numerical outputs used to verify the tables, plots, and case studies reported in the associated manuscript. The files are provided as CSV or JSON so that the reported values can be inspected without rerunning all training stages.

## File groups

- `task1_shap_*`: predictor-attribution values and run metadata.
- `task2_scale_summary.csv`: per-state reconstruction summary.
- `task2_full_family_baselines.csv`: aggregate comparison with the evaluated baseline families.
- `task2_full_family_baseline_task_metrics.csv`: task-level comparison metrics used for aggregate statistics and ablation checks.
- `task2_fixed_anchor_protocol_representative_*`: controlled 7/9-point fixed-budget evaluation.
- `task2_hierarchy_unfactorized_ablation_representative_summary.csv`: hierarchy and transfer ablation results.
- `task2_diffusion_necessity_representative_summary.csv`: diffusion-prior necessity comparison.
- `task2_extreme_sparsity_model3.csv`: extreme-sparsity evaluation for the representative device.
- `task2_physical_consistency_*`: PEC-dependent reconstruction case study.
- `task2_residual_heatmap.csv`: residual values used for the reported residual analysis.

These files contain processed research measurements and derived evaluation metrics. They do not contain manuscript text, author correspondence, submission records, or local execution metadata.
