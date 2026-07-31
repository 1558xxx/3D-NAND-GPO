from __future__ import annotations

import argparse

import pandas as pd

from run_task2_reviewer_baselines import SOURCE_DATA_ROOT


METHOD_ORDER = [
    "scratch_only",
    "source_pooled_mlp",
    "nearest_parameter_transfer",
    "hypernetwork_parameter",
    "diffusion_transfer",
]

METHOD_LABELS = {
    "scratch_only": "Scratch-only",
    "source_pooled_mlp": "Source-pretrained pooled MLP",
    "nearest_parameter_transfer": "Nearest parameter transfer",
    "hypernetwork_parameter": "Direct hypernetwork parameter",
    "diffusion_transfer": "Diffusion transfer",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Task 2 diffusion-necessity baselines.")
    parser.add_argument("--input", default="task2_diffusion_necessity_representative.csv")
    parser.add_argument("--output", default="task2_diffusion_necessity_representative_summary.csv")
    args = parser.parse_args()

    frame = pd.read_csv(SOURCE_DATA_ROOT / args.input)
    summary = (
        frame.groupby("method")
        .agg(
            point_r2=("point_r2", "mean"),
            task_mean_r2=("task_mean_r2", "mean"),
            task_median_r2=("task_median_r2", "mean"),
            p05_test_r2=("p05_test_r2", "mean"),
            frac_test_r2_gt_0_9=("frac_test_r2_gt_0_9", "mean"),
            frac_test_r2_lt_0=("frac_test_r2_lt_0", "mean"),
            tail_rmse_pct_peak=("adapted_shape_mean_tail_rmse_pct_peak", "mean"),
            peak_step_error=("adapted_shape_mean_abs_peak_step_error", "mean"),
        )
        .reset_index()
    )
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    summary["order"] = summary["method"].map({method: index for index, method in enumerate(METHOD_ORDER)})

    family_task_r2 = frame.pivot(index="method", columns="label", values="task_mean_r2").reset_index()
    summary = summary.merge(family_task_r2, on="method", how="left")
    summary = summary[
        [
            "order",
            "method",
            "method_label",
            "P2 / M3",
            "P4 / M5",
            "P7 / M8",
            "point_r2",
            "task_mean_r2",
            "task_median_r2",
            "p05_test_r2",
            "frac_test_r2_gt_0_9",
            "frac_test_r2_lt_0",
            "tail_rmse_pct_peak",
            "peak_step_error",
        ]
    ].sort_values("order")

    output_path = SOURCE_DATA_ROOT / args.output
    summary.to_csv(output_path, index=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
