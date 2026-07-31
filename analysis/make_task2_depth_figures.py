from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from torch.nn.utils import vector_to_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ROOT = PROJECT_ROOT / "Pretrain"
RESULT_ROOT = PRETRAIN_ROOT / "artifacts" / "curve_transfer_batch"
PAPER_DATA_ROOT = PROJECT_ROOT / "paper_data"
FIGURE_ROOT = PROJECT_ROOT / "generated_figures"
SOURCE_DATA_ROOT = PAPER_DATA_ROOT
PHYSICAL_MODEL_INDEX = 3
PHYSICAL_WL = 71
PHYSICAL_RETENTION = 6
PHYSICAL_CASE_PECS = [0, 6000, 10000]

sys.path.insert(0, str(PRETRAIN_ROOT))
from Models import MLPRegressor  # noqa: E402


COLORS = {
    "ink": "#272727",
    "muted": "#6B7280",
    "grid": "#E8EBEE",
    "truth": "#2B2B2B",
    "method": "#0F4D92",
    "method_soft": "#DCEAF8",
    "accent": "#D4842A",
    "accent_soft": "#F6E7D4",
    "green": "#4F8A4C",
    "red": "#B64342",
    "purple": "#7A67C7",
}

METHOD_LABELS = {
    "physics_guided_diffusion": "Condition-informed parameter transfer",
    "spline": "Spline interpolation",
    "gaussian": "Gaussian fitting",
}

METHOD_COLORS = {
    "physics_guided_diffusion": COLORS["method"],
    "spline": COLORS["green"],
    "gaussian": COLORS["red"],
}


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["font.size"] = 8
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["legend.frameon"] = False


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
    )


def save_figure(fig, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIGURE_ROOT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.05)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def crop_white_margins(image: np.ndarray, threshold: float = 0.985, pad: int = 10) -> np.ndarray:
    if image.ndim == 2:
        non_white = image < threshold
    else:
        rgb = image[..., :3]
        non_white = np.any(rgb < threshold, axis=-1)

    coords = np.argwhere(non_white)
    if coords.size == 0:
        return image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    y0 = max(int(y0) - pad, 0)
    x0 = max(int(x0) - pad, 0)
    y1 = min(int(y1) + pad + 1, image.shape[0])
    x1 = min(int(x1) + pad + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def find_column(df: pd.DataFrame, candidates: list[str], fallback_index: int | None = None) -> str:
    normalized = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    if fallback_index is not None:
        return str(df.columns[fallback_index])
    raise KeyError(f"None of the candidate columns are present: {candidates}")


def draw_task1_global_cue(ax) -> None:
    importance = pd.read_csv(PAPER_DATA_ROOT / "task1_shap_feature_importance.csv")
    importance = importance.sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)
    colors = [COLORS["method_soft"] if feature != "Step" else COLORS["method"] for feature in importance["feature"]]

    ax.barh(importance["feature"], importance["mean_abs_shap"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Controlled global SHAP attribution", loc="left", fontsize=8.7, pad=5)
    ax.set_xlabel("mean(|SHAP value|)", fontsize=7.2)
    ax.tick_params(axis="both", labelsize=7.0, length=2.5, pad=2)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0.0, importance["mean_abs_shap"].max() * 1.12)


def draw_task1_step_cue(ax) -> None:
    samples = pd.read_csv(PAPER_DATA_ROOT / "task1_shap_explained_samples.csv")
    step_col = find_column(samples, ["Step", "step", "步长"], fallback_index=0)
    pec_col = find_column(samples, ["PEC", "pec", "擦写次数"], fallback_index=4)

    plot_df = samples[[step_col, pec_col, "shap_step"]].copy()
    plot_df[step_col] = pd.to_numeric(plot_df[step_col], errors="coerce")
    plot_df[pec_col] = pd.to_numeric(plot_df[pec_col], errors="coerce")
    plot_df["shap_step"] = pd.to_numeric(plot_df["shap_step"], errors="coerce")
    plot_df = plot_df.dropna().sort_values(step_col)

    summary = (
        plot_df.groupby(step_col, as_index=False)["shap_step"]
        .agg(["median", "min", "max"])
        .reset_index()
        .sort_values(step_col)
    )
    scatter = ax.scatter(
        plot_df[step_col],
        plot_df["shap_step"],
        c=plot_df[pec_col],
        cmap="cool",
        s=8,
        alpha=0.50,
        linewidths=0,
    )
    ax.plot(summary[step_col], summary["median"], color=COLORS["ink"], linewidth=1.2)
    ax.fill_between(
        summary[step_col],
        summary["min"],
        summary["max"],
        color=COLORS["method_soft"],
        alpha=0.32,
        linewidth=0,
    )
    ax.axhline(0.0, color=COLORS["muted"], linewidth=0.7, linestyle=(0, (2, 3)))
    ax.set_title("RRV-coordinate SHAP dependence", loc="left", fontsize=8.7, pad=5)
    ax.set_xlabel("RRV scan index", fontsize=7.2)
    ax.set_ylabel("SHAP value for RRV scan index", fontsize=7.2)
    ax.tick_params(axis="both", labelsize=7.0, length=2.5, pad=2)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.050, pad=0.025)
    cbar.ax.tick_params(labelsize=6.7, length=2.0, pad=1)


def load_final_summary() -> pd.DataFrame:
    return pd.read_csv(RESULT_ROOT / "final_summary.csv")


def resolve_run_dir(model_index: int, run_kind: str = "full") -> Path:
    state = load_final_summary()
    if run_kind == "full":
        candidate_name = state.loc[state["model_index"] == model_index, "candidate_name"].iloc[0]
        return RESULT_ROOT / f"model_{model_index}" / "full" / str(candidate_name)

    model_dir = RESULT_ROOT / f"model_{model_index}" / "sample"
    matches = sorted(path for path in model_dir.glob("*") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"Missing sample run directory under {model_dir}")
    return matches[0]


def inverse_freqs(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), a_min=-20.0, a_max=12.0)
    return np.expm1(clipped).astype(np.float32)


def reconstruct_curve(task: dict, parameter_vector: np.ndarray, step_scaler) -> np.ndarray:
    model = MLPRegressor(in_dim=1, hidden_dims=(32, 16), out_dim=1, activation="tanh")
    vector_to_parameters(torch.as_tensor(np.asarray(parameter_vector, dtype=np.float32).reshape(-1)), model.parameters())
    model.eval()
    x_scaled = step_scaler.transform(np.asarray(task["steps"], dtype=np.float32).reshape(-1, 1)).astype(np.float32)
    with torch.no_grad():
        pred_log = model(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy().reshape(-1, 1)
    return np.clip(inverse_freqs(pred_log).reshape(-1), a_min=0.0, a_max=None)


def asymmetry_score(values: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    peak_index = int(np.argmax(values))
    left = float(values[:peak_index].sum()) if peak_index > 0 else 0.0
    right = float(values[peak_index + 1 :].sum()) if peak_index + 1 < len(values) else 0.0
    asymmetry = (right - left) / (left + right + 1e-6)
    return float(asymmetry), peak_index


def build_physical_consistency_assets() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = resolve_run_dir(PHYSICAL_MODEL_INDEX, run_kind="full")
    metrics = pd.read_csv(run_dir / "target_task_metrics.csv")
    tasks = load_pickle(run_dir / "target_tasks.pkl")
    step_scaler = load_pickle(run_dir / "step_scaler.pkl")
    finetuned = np.load(run_dir / "target_finetuned_params.npy")
    generated = np.load(sorted(run_dir.glob("sampleSeq_RealParams_*.npy"))[0]).reshape(len(tasks), -1)

    ladder = metrics.loc[(metrics["Retention"] == PHYSICAL_RETENTION) & (metrics["WL"] == PHYSICAL_WL)].sort_values("PEC")
    stress_rows = []
    curve_rows = []
    for row in ladder.itertuples(index=False):
        task = tasks[int(row.task_id)]
        prediction = reconstruct_curve(task, finetuned[int(row.task_id)], step_scaler)
        truth = np.asarray(task["freqs"], dtype=np.float32)
        true_asym, true_peak_index = asymmetry_score(truth)
        pred_asym, pred_peak_index = asymmetry_score(prediction)
        drift_norm = float(
            np.linalg.norm(finetuned[int(row.task_id)] - generated[int(row.task_id)])
            / max(np.linalg.norm(generated[int(row.task_id)]), 1e-6)
        )

        stress_rows.append(
            {
                "model_index": PHYSICAL_MODEL_INDEX,
                "curve_family": f"P{PHYSICAL_MODEL_INDEX - 1}",
                "WL": PHYSICAL_WL,
                "Retention": PHYSICAL_RETENTION,
                "PEC": int(row.PEC),
                "task_id": int(row.task_id),
                "test_r2": float(row.test_r2),
                "true_peak_step": int(task["steps"][true_peak_index]),
                "pred_peak_step": int(task["steps"][pred_peak_index]),
                "peak_step_error": int(task["steps"][pred_peak_index] - task["steps"][true_peak_index]),
                "true_asymmetry": true_asym,
                "pred_asymmetry": pred_asym,
                "parameter_refinement_norm": drift_norm,
            }
        )

        if int(row.PEC) in PHYSICAL_CASE_PECS:
            train_indices = set(task["splits"]["train"])
            for index, (step_value, true_value, pred_value) in enumerate(zip(task["steps"], truth, prediction)):
                curve_rows.append(
                    {
                        "PEC": int(row.PEC),
                        "step": int(step_value),
                        "true_freq": float(true_value),
                        "pred_freq": float(pred_value),
                        "observed_sparse_point": int(index in train_indices),
                    }
                )

    stress_df = pd.DataFrame(stress_rows).sort_values("PEC").reset_index(drop=True)
    curve_df = pd.DataFrame(curve_rows).sort_values(["PEC", "step"]).reset_index(drop=True)
    SOURCE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    stress_df.to_csv(SOURCE_DATA_ROOT / "task2_physical_consistency_ladder.csv", index=False)
    curve_df.to_csv(SOURCE_DATA_ROOT / "task2_physical_consistency_curves.csv", index=False)
    return stress_df, curve_df


def build_physical_consistency_figure(stress_df: pd.DataFrame, curve_df: pd.DataFrame) -> None:
    apply_style()
    fig = plt.figure(figsize=(9.2, 8.2))
    gs = GridSpec(
        3,
        4,
        figure=fig,
        width_ratios=[1.88, 1.20, 1.20, 1.20],
        height_ratios=[1.08, 1.08, 1.22],
        wspace=0.64,
        hspace=0.78,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    draw_task1_global_cue(ax_a)
    add_panel_label(ax_a, "a", x=-0.16, y=1.04)

    ax_b = fig.add_subplot(gs[1, 0])
    draw_task1_step_cue(ax_b)
    add_panel_label(ax_b, "b", x=-0.16, y=1.04)

    ax_bridge = fig.add_subplot(gs[2, 0])
    ax_bridge.axis("off")
    add_panel_label(ax_bridge, "c", x=-0.16, y=1.04)
    bridge_text = (
        "Attribution summary\n"
        "RRV index: within-curve coordinate\n"
        "PEC: largest task-level attribution\n\n"
        "Representative case\n"
        "P2; WL 71; retention index 6\n"
        "Reference trends tracked"
    )
    ax_bridge.text(
        0.02,
        0.94,
        bridge_text,
        ha="left",
        va="top",
        fontsize=7.3,
        color=COLORS["ink"],
        linespacing=1.18,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#F7F8FA", edgecolor="#D7DADF"),
    )

    ax_d = fig.add_subplot(gs[0, 1:4])
    add_panel_label(ax_d, "d")
    ax_d.plot(
        stress_df["PEC"],
        stress_df["true_asymmetry"],
        color=COLORS["truth"],
        linewidth=1.6,
        marker="o",
        markersize=3.8,
        label="Complete-curve reference",
    )
    ax_d.plot(
        stress_df["PEC"],
        stress_df["pred_asymmetry"],
        color=COLORS["method"],
        linewidth=1.6,
        marker="s",
        markersize=3.8,
        label="Reconstructed asymmetry",
    )
    ax_d.axhline(0.0, color=COLORS["muted"], linewidth=0.8, linestyle="--")
    ax_d.set_title("Asymmetry across the P/E-cycle sweep", loc="left", fontsize=8.9, pad=5)
    ax_d.set_ylabel("Curve asymmetry index")
    ax_d.set_xlabel("PEC")
    ax_d.set_xticks(stress_df["PEC"])
    ax_d.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax_d.legend(loc="upper left", fontsize=6.8, ncol=2, columnspacing=1.2, handletextpad=0.55)

    ax_e = fig.add_subplot(gs[1, 1:4])
    add_panel_label(ax_e, "e")
    bars = ax_e.bar(
        stress_df["PEC"],
        stress_df["parameter_refinement_norm"],
        width=520,
        color=COLORS["accent_soft"],
        edgecolor=COLORS["accent"],
        linewidth=0.9,
        label=r"Normalized parameter update $\|\Delta z_\tau\|_2 / \|z_\tau^{\mathrm{gen}}\|_2$",
    )
    ax_e.set_ylabel("Normalized update")
    ax_e.set_xlabel("PEC")
    ax_e.set_xticks(stress_df["PEC"])
    ax_e.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax_e.set_title("Parameter refinement across the PEC sweep", loc="left", fontsize=8.9, pad=5)

    ax_e2 = ax_e.twinx()
    ax_e2.plot(
        stress_df["PEC"],
        stress_df["pred_peak_step"],
        color=COLORS["purple"],
        linewidth=1.4,
        marker="o",
        markersize=3.4,
        label="Reconstructed peak step",
    )
    ax_e2.plot(
        stress_df["PEC"],
        stress_df["true_peak_step"],
        color=COLORS["truth"],
        linewidth=1.2,
        marker="D",
        markersize=3.0,
        linestyle="--",
        label="Reference peak index",
    )
    ax_e2.set_ylabel("Peak scan index")
    y_values = np.concatenate([stress_df["true_peak_step"].to_numpy(), stress_df["pred_peak_step"].to_numpy()])
    ax_e2.set_ylim(y_values.min() - 0.6, y_values.max() + 0.6)

    handles_1, labels_1 = ax_e.get_legend_handles_labels()
    handles_2, labels_2 = ax_e2.get_legend_handles_labels()
    ax_e.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left", fontsize=6.5, handletextpad=0.55)

    shared_handles = [
        plt.Line2D([], [], color=COLORS["truth"], linewidth=1.6, label="Complete-curve reference"),
        plt.Line2D([], [], color=COLORS["method"], linewidth=1.6, label="Reconstruction"),
        plt.Line2D([], [], linestyle="none", marker="o", markersize=5.0, color=COLORS["accent"], label="Sparse observations"),
    ]
    fig.legend(
        handles=shared_handles,
        loc="upper center",
        bbox_to_anchor=(0.70, 0.36),
        ncol=3,
        fontsize=6.5,
        columnspacing=1.1,
        handletextpad=0.5,
    )

    for index, pec in enumerate(PHYSICAL_CASE_PECS):
        ax = fig.add_subplot(gs[2, index + 1])
        add_panel_label(ax, chr(ord("f") + index), x=-0.14, y=1.02)
        case_rows = curve_df.loc[curve_df["PEC"] == pec].copy()
        summary_row = stress_df.loc[stress_df["PEC"] == pec].iloc[0]
        ax.plot(case_rows["step"], case_rows["true_freq"], color=COLORS["truth"], linewidth=1.6)
        ax.plot(case_rows["step"], case_rows["pred_freq"], color=COLORS["method"], linewidth=1.6)
        observed = case_rows.loc[case_rows["observed_sparse_point"] == 1]
        ax.scatter(
            observed["step"],
            observed["true_freq"],
            s=16,
            color=COLORS["accent"],
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
        ax.set_title(f"PEC = {pec}", fontsize=8.1, pad=4)
        ax.set_xlabel("RRV scan index")
        ax.set_ylabel("Histogram frequency" if index == 0 else "")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.text(
            0.04,
            0.96,
            rf"$R^2$ = {summary_row['test_r2']:.3f}" "\n"
            rf"$\Delta$peak = {int(summary_row['peak_step_error'])}" "\n"
            rf"$\Delta z$ = {summary_row['parameter_refinement_norm']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.55,
            color=COLORS["ink"],
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D8DCE1", alpha=0.95),
        )

    fig.suptitle(
        "Model attribution and a representative P/E-cycle reconstruction sweep",
        x=0.03,
        y=0.985,
        ha="left",
        fontsize=10.0,
        fontweight="bold",
    )
    save_figure(fig, "fig7_task2_physical_consistency_extremes")
    plt.close(fig)


def build_sparsity_summary_figure() -> None:
    apply_style()
    benchmark_path = resolve_run_dir(3, run_kind="sample") / "sparsity" / "reviewer_depth" / "sparsity_benchmark_summary.csv"
    sparsity_df = pd.read_csv(benchmark_path)
    sparsity_df = sparsity_df.sort_values(["observed_points", "method"]).reset_index(drop=True)
    sparsity_df.to_csv(SOURCE_DATA_ROOT / "task2_extreme_sparsity_model3.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.35), sharex=True)
    metrics = [
        ("task_median_r2", "Median held-out task $R^2$", (-0.10, 1.01)),
        ("frac_test_r2_gt_0_9", "Tasks with test $R^2 > 0.9$", (0.0, 1.02)),
    ]

    for index, (ax, (metric, ylabel, ylim)) in enumerate(zip(axes, metrics)):
        add_panel_label(ax, "a" if index == 0 else "b", x=-0.13, y=1.02)
        for method_name in ("physics_guided_diffusion", "spline", "gaussian"):
            method_df = sparsity_df.loc[sparsity_df["method"] == method_name].sort_values("observed_points")
            ax.plot(
                method_df["observed_points"],
                method_df[metric],
                marker="o" if method_name == "physics_guided_diffusion" else ("s" if method_name == "spline" else "^"),
                markersize=4.0,
                linewidth=1.7,
                color=METHOD_COLORS[method_name],
                label=METHOD_LABELS[method_name],
            )
        ax.set_xlabel("Observed target points per task")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)

    axes[0].legend(loc="lower right", fontsize=6.9)
    fig.suptitle(
        "Observation-budget sweep on the representative TLC state P2 subset",
        x=0.05,
        y=0.99,
        ha="left",
        fontsize=9.4,
        fontweight="bold",
    )
    save_figure(fig, "fig8_task2_extreme_sparsity_curve")
    plt.close(fig)


def build_residual_summary() -> pd.DataFrame:
    rows = []
    final_summary = load_final_summary()
    for model_index in range(2, 9):
        candidate_name = final_summary.loc[final_summary["model_index"] == model_index, "candidate_name"].iloc[0]
        run_dir = RESULT_ROOT / f"model_{model_index}" / "full" / str(candidate_name)
        tasks = load_pickle(run_dir / "target_tasks.pkl")
        step_scaler = load_pickle(run_dir / "step_scaler.pkl")
        params = np.load(run_dir / "target_finetuned_params.npy")

        for task_id, task in enumerate(tasks):
            prediction = reconstruct_curve(task, params[task_id], step_scaler)
            truth = np.asarray(task["freqs"], dtype=np.float32)
            test_indices = np.asarray(task["splits"]["test"], dtype=np.int64)
            error = prediction[test_indices] - truth[test_indices]
            peak_height = max(float(truth.max()), 1.0)
            true_asym, true_peak_index = asymmetry_score(truth)
            pred_asym, pred_peak_index = asymmetry_score(prediction)
            rows.append(
                {
                    "model_index": model_index,
                    "Retention": int(task["Retention"]),
                    "PEC": int(task["PEC"]),
                    "signed_pct_of_peak": float(error.mean() / peak_height),
                    "abs_pct_of_peak": float(np.abs(error).mean() / peak_height),
                    "peak_step_error": int(task["steps"][pred_peak_index] - task["steps"][true_peak_index]),
                    "asymmetry_error": float(pred_asym - true_asym),
                }
            )

    task_df = pd.DataFrame(rows)
    grouped = (
        task_df.groupby(["Retention", "PEC"], sort=True)
        .agg(
            task_count=("model_index", "size"),
            mean_signed_pct_of_peak=("signed_pct_of_peak", "mean"),
            mean_abs_pct_of_peak=("abs_pct_of_peak", "mean"),
            mean_peak_step_error=("peak_step_error", "mean"),
            mean_abs_asymmetry_error=("asymmetry_error", lambda series: float(np.mean(np.abs(series)))),
        )
        .reset_index()
    )
    grouped.to_csv(SOURCE_DATA_ROOT / "task2_residual_heatmap.csv", index=False)
    return grouped


def build_residual_heatmap_figure(residual_df: pd.DataFrame) -> None:
    apply_style()
    signed_matrix = residual_df.pivot(index="Retention", columns="PEC", values="mean_signed_pct_of_peak").sort_index()
    abs_matrix = residual_df.pivot(index="Retention", columns="PEC", values="mean_abs_pct_of_peak").sort_index()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.4, 3.8),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.34},
    )

    signed_values = 100.0 * signed_matrix.to_numpy(dtype=np.float32)
    abs_values = 100.0 * abs_matrix.to_numpy(dtype=np.float32)
    signed_norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.30, vmax=0.30)
    signed_cmap = LinearSegmentedColormap.from_list("signed", ["#B64342", "#F7F7F7", "#0F4D92"])

    ax = axes[0]
    add_panel_label(ax, "a", x=-0.10, y=1.02)
    im = ax.imshow(signed_values, aspect="auto", cmap=signed_cmap, norm=signed_norm)
    ax.set_title("Mean signed residual (% of task peak)", loc="left", fontsize=8.6, pad=6)
    ax.set_xticks(np.arange(signed_matrix.shape[1]))
    ax.set_xticklabels([str(value) for value in signed_matrix.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(signed_matrix.shape[0]))
    ax.set_yticklabels([str(value) for value in signed_matrix.index])
    ax.set_xlabel("PEC")
    ax.set_ylabel("Retention")
    for i in range(signed_values.shape[0]):
        for j in range(signed_values.shape[1]):
            value = signed_values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.0, color=COLORS["ink"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.055)
    cbar.set_label("% of peak")

    ax = axes[1]
    add_panel_label(ax, "b", x=-0.10, y=1.02)
    im = ax.imshow(abs_values, aspect="auto", cmap="YlGnBu", vmin=1.8, vmax=2.7)
    ax.set_title("Mean absolute residual (% of task peak)", loc="left", fontsize=8.6, pad=6)
    ax.set_xticks(np.arange(abs_matrix.shape[1]))
    ax.set_xticklabels([str(value) for value in abs_matrix.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(abs_matrix.shape[0]))
    ax.set_yticklabels([str(value) for value in abs_matrix.index])
    ax.set_xlabel("PEC")
    ax.set_ylabel("")
    for i in range(abs_values.shape[0]):
        for j in range(abs_values.shape[1]):
            value = abs_values[i, j]
            color = "white" if value > 2.35 else COLORS["ink"]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.0, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.055)
    cbar.set_label("% of peak")

    fig.suptitle(
        "Residual structure stays centered and only broadens modestly in the harshest retention corner",
        x=0.03,
        y=1.02,
        ha="left",
        fontsize=10.0,
        fontweight="bold",
    )
    save_figure(fig, "fig9_task2_residual_heatmap")
    plt.close(fig)


def main() -> None:
    stress_df, curve_df = build_physical_consistency_assets()
    build_physical_consistency_figure(stress_df, curve_df)
    build_sparsity_summary_figure()
    residual_df = build_residual_summary()
    build_residual_heatmap_figure(residual_df)


if __name__ == "__main__":
    main()
