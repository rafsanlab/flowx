"""Publication Diagnostic Plots for Phase 5 & 6 Statistical and Cohort Analyses.

Generates Nature Methods-styled publication diagnostic figures:
- Volcano plots for Differential Abundance (Log2FC vs -log10 q-value)
- Stratified cohort population frequency boxplots and strip charts
- Multi-sample hierarchically clustered frequency and MFI heatmaps
- Volcano plots for Differential State functional marker modulations
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from flowx.stats.models import CohortFeatureSet, DAResult, DSResult
from flowx.visualization.style import apply_nature_axis_style, set_nature_style

logger = logging.getLogger(__name__)


def plot_volcano_da(
    da_result: DAResult,
    output_path: Optional[Union[str, Path]] = None,
    fdr_threshold: Optional[float] = None,
    log2fc_threshold: Optional[float] = None,
    figsize: Tuple[float, float] = (4.8, 3.8),
) -> plt.Figure:
    """Render a publication-ready volcano plot for Differential Abundance (DA) results.

    X-axis: log2 fold change (target vs reference cohort)
    Y-axis: -log10(FDR q-value)

    Args:
        da_result: DAResult containing tested populations, q-values, and log2FC.
        output_path: Optional path to save figure image.
        fdr_threshold: Cutoff for significance horizontal line (defaults to da_result.config.fdr_threshold).
        log2fc_threshold: Cutoff for significance vertical lines (defaults to da_result.config.log2fc_threshold).
        figsize: Figure dimensions in inches (default: (4.8, 3.8)).

    Returns:
        Rendered Matplotlib Figure.
    """
    set_nature_style()
    df = da_result.summary_df.dropna(subset=["log2fc"]).copy()

    fdr_cut = fdr_threshold if fdr_threshold is not None else da_result.config.fdr_threshold
    l2fc_cut = log2fc_threshold if log2fc_threshold is not None else da_result.config.log2fc_threshold

    fig, ax = plt.subplots(figsize=figsize)

    if df.empty:
        ax.text(0.5, 0.5, "No tested populations available for volcano plot", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig

    # Calculate -log10(q_value) (treat NaN q_values from n<2 descriptive contrasts as 1.0 so log2FC is still plotted)
    q_vals_filled = df["q_value"].fillna(1.0).to_numpy()
    q_clipped = np.clip(q_vals_filled, 1e-15, 1.0)
    neg_log10_q = -np.log10(q_clipped)
    df["neg_log10_q"] = neg_log10_q

    # Assign category for coloring
    is_sig = (df["q_value"].notna()) & (df["q_value"] <= fdr_cut) & (df["log2fc"].abs() >= l2fc_cut)
    is_up = is_sig & (df["log2fc"] > 0)
    is_down = is_sig & (df["log2fc"] < 0)

    # Plot non-significant points
    ns_mask = ~is_sig
    ax.scatter(
        df.loc[ns_mask, "log2fc"],
        df.loc[ns_mask, "neg_log10_q"],
        c="#94A3B8",
        alpha=0.65,
        s=30,
        edgecolor="none",
        label="Not Significant",
        zorder=2,
    )

    # Plot expanded (upregulated) populations
    if np.any(is_up):
        ax.scatter(
            df.loc[is_up, "log2fc"],
            df.loc[is_up, "neg_log10_q"],
            c="#EF4444",
            alpha=0.90,
            s=45,
            edgecolor="#991B1B",
            linewidth=0.6,
            label=f"Expanded in {da_result.target_group} (n={is_up.sum()})",
            zorder=3,
        )

    # Plot contracted (downregulated) populations
    if np.any(is_down):
        ax.scatter(
            df.loc[is_down, "log2fc"],
            df.loc[is_down, "neg_log10_q"],
            c="#3B82F6",
            alpha=0.90,
            s=45,
            edgecolor="#1D4ED8",
            linewidth=0.6,
            label=f"Contracted in {da_result.target_group} (n={is_down.sum()})",
            zorder=3,
        )

    # Threshold guidelines
    y_threshold = -np.log10(fdr_cut)
    ax.axhline(y_threshold, color="#64748B", linestyle="--", linewidth=0.8, alpha=0.8, label=f"FDR = {fdr_cut}")
    ax.axvline(l2fc_cut, color="#64748B", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.axvline(-l2fc_cut, color="#64748B", linestyle=":", linewidth=0.8, alpha=0.8)

    # Annotate significant points with population labels
    for _, row in df[is_sig].iterrows():
        ax.annotate(
            str(row["population"]),
            xy=(row["log2fc"], row["neg_log10_q"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.8,
            fontweight="medium",
            color="#0F172A",
            alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
            zorder=4,
        )

    apply_nature_axis_style(ax, show_grid=True)
    ax.set_xlabel(f"Differential Abundance [$\\log_2$ FC ({da_result.comparison_name})]", fontsize=8.0, fontweight="medium")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{FDR}\;q\mathrm{-value})$", fontsize=8.0, fontweight="medium")
    ax.set_title(f"Differential Abundance Volcano: {da_result.comparison_name}", fontsize=8.8, fontweight="bold", pad=8)
    ax.set_box_aspect(1)
    ax.legend(frameon=True, fontsize=6.8, loc="best", framealpha=0.9)
    fig.tight_layout()
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=8, loc="upper right")

    plt.tight_layout()

    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        logger.info(f"Saved DA volcano plot to '{dest.name}'.")

    return fig


def plot_population_abundance_boxplots(
    features: CohortFeatureSet,
    condition_column: str = "condition",
    top_n: int = 8,
    frequency_type: str = "viable",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    plot_type: Literal["boxplot", "violin"] = "boxplot",
) -> plt.Figure:
    """Render condition-stratified boxplots or violin plots with individual sample jitter points across populations.

    Args:
        features: CohortFeatureSet containing frequencies and sample metadata.
        condition_column: Column name in sample_metadata defining experimental groups.
        top_n: Number of populations to display (selects highest mean frequency).
        frequency_type: 'viable' (viable singlets %) or 'parent' (parent %).
        output_path: Optional path to save figure image.
        figsize: Optional explicit figure size.
        plot_type: 'boxplot' (default) or 'violin' (reveals distribution density).

    Returns:
        Rendered Matplotlib Figure.
    """
    set_nature_style()

    if features.sample_metadata is None or condition_column not in features.sample_metadata.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"Condition column '{condition_column}' not found in metadata", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig

    freq_df = features.viable_freq_matrix if frequency_type == "viable" else features.parent_freq_matrix
    meta = features.sample_metadata

    # Select top populations by average abundance
    mean_abundances = freq_df.mean(axis=0)
    selected_pops = mean_abundances.sort_values(ascending=False).head(top_n).index.tolist()

    # Reshape to long format for Seaborn plotting
    rows = []
    for s_id in freq_df.index:
        cond = meta.loc[s_id, condition_column] if s_id in meta.index else "Unknown"
        for p in selected_pops:
            val = freq_df.loc[s_id, p] if p in freq_df.columns else 0.0
            rows.append({"sample_id": s_id, "condition": str(cond), "population": p, "frequency_pct": val})

    long_df = pd.DataFrame(rows)

    fig_w = min(6.8, max(3.8, 0.55 * len(selected_pops) + 1.2))
    fig_dim = figsize or (fig_w, 3.4)
    fig, ax = plt.subplots(figsize=fig_dim)

    n_conds = len(long_df["condition"].unique())
    base_palette = ["#3B82F6", "#EF4444", "#10B981", "#8B5CF6", "#F59E0B"]
    palette = base_palette[:n_conds] if n_conds <= len(base_palette) else sns.color_palette("tab10", n_conds)

    if plot_type == "violin":
        sns.violinplot(
            data=long_df,
            x="population",
            y="frequency_pct",
            hue="condition",
            palette=palette,
            ax=ax,
            inner=None,
            cut=0,
            linewidth=0.8,
            alpha=0.75,
        )
    else:
        sns.boxplot(
            data=long_df,
            x="population",
            y="frequency_pct",
            hue="condition",
            palette=palette,
            ax=ax,
            fliersize=0,
            boxprops=dict(alpha=0.75),
            linewidth=0.8,
        )

    sns.stripplot(
        data=long_df,
        x="population",
        y="frequency_pct",
        hue="condition",
        palette=palette,
        dodge=True,
        jitter=0.18,
        size=3.8,
        alpha=0.88,
        ax=ax,
        edgecolor="black",
        linewidth=0.4,
    )

    # Remove duplicated legend entries from stripplot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:n_conds],
        labels[:n_conds],
        title=condition_column.title(),
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=7.0,
        title_fontsize=7.5,
        loc="upper right",
    )

    apply_nature_axis_style(ax, show_grid=True)
    ax.set_xticks(range(len(selected_pops)))
    ax.set_xticklabels(selected_pops, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel(f"Relative Frequency (% of {frequency_type.title()})", fontsize=8.0, fontweight="medium")
    ax.set_xlabel("Cell Population / Cluster", fontsize=8.0, fontweight="medium")
    plot_title = "Cohort Population Abundance Distributions" if plot_type == "violin" else "Cohort Population Abundance Distribution"
    ax.set_title(plot_title, fontsize=8.8, fontweight="bold", pad=8)

    plt.tight_layout()

    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        logger.info(f"Saved population abundance plot to '{dest.name}'.")

    return fig


plot_population_abundance_distribution = plot_population_abundance_boxplots


def plot_cohort_heatmap(
    features: CohortFeatureSet,
    matrix_type: str = "frequency",
    condition_column: Optional[str] = "condition",
    population_name_map: Optional[Dict[Any, str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "YlGnBu",
    group_by_population: bool = False,
) -> plt.Figure:
    """Render a hierarchically clustered multi-sample heatmap of population abundances or marker expression.

    Args:
        features: CohortFeatureSet containing frequencies, MFIs, and sample metadata.
        matrix_type: 'frequency' (sample x population %) or marker name (sample x population MFI).
        condition_column: Optional metadata column to annotate sample rows with color badges.
        output_path: Optional path to save figure image.
        figsize: Optional explicit figure size in inches.
        cmap: Colormap (default: 'YlGnBu' for crisp publication contrast).
        group_by_population: If True, strips C0/C1 prefixes and aggregates identical populations.

    Returns:
        Rendered Matplotlib Figure.
    """
    set_nature_style()

    if matrix_type == "frequency":
        data_df = features.viable_freq_matrix.copy()
        title_label = "Relative Frequency (%)"
    elif matrix_type in features.mfi_tensor:
        data_df = features.mfi_tensor[matrix_type].copy()
        title_label = f"MFI ({matrix_type})"
    else:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, f"Unknown matrix_type '{matrix_type}' for cohort heatmap", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig

    # Drop all-NaN columns or rows
    data_df = data_df.dropna(how="all", axis=0).dropna(how="all", axis=1).fillna(0.0)
    if population_name_map:
        rename_dict = {}
        for col in data_df.columns:
            col_str = str(col)
            cid = col_str.replace("Cluster_", "").replace("Cluster ", "")
            if cid.isdigit() and int(cid) in population_name_map:
                raw_pop = str(population_name_map[int(cid)]).strip()
                if group_by_population:
                    rename_dict[col] = re.sub(r"^C\d+\s*[:\-]\s*", "", raw_pop).strip() or raw_pop
                else:
                    rename_dict[col] = raw_pop if raw_pop.startswith(f"C{cid}:") else f"C{cid}: {raw_pop}"
            elif col in population_name_map:
                raw_pop = str(population_name_map[col]).strip()
                rename_dict[col] = (
                    (re.sub(r"^C\d+\s*[:\-]\s*", "", raw_pop).strip() or raw_pop)
                    if group_by_population
                    else raw_pop
                )
            elif col_str in population_name_map:
                raw_pop = str(population_name_map[col_str]).strip()
                rename_dict[col] = (
                    (re.sub(r"^C\d+\s*[:\-]\s*", "", raw_pop).strip() or raw_pop)
                    if group_by_population
                    else raw_pop
                )
        data_df = data_df.rename(columns=rename_dict)
    elif group_by_population:
        data_df = data_df.rename(
            columns={
                c: (re.sub(r"^C\d+\s*[:\-]\s*", "", str(c)).strip() or str(c))
                for c in data_df.columns
            }
        )

    if group_by_population and data_df.columns.duplicated().any():
        if matrix_type == "frequency":
            data_df = data_df.T.groupby(level=0, sort=False).sum().T
        else:
            data_df = data_df.T.groupby(level=0, sort=False).mean().T
    if data_df.empty or data_df.shape[0] < 2 or data_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Insufficient multi-sample data for hierarchical heatmap", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig

    # Optional row annotation colors and condition stratification
    has_condition_legend = (
        condition_column is not None
        and features.sample_metadata is not None
        and condition_column in features.sample_metadata.columns
    )

    row_colors = None
    color_map = {}
    unique_conds = []

    if has_condition_legend and features.sample_metadata is not None:
        meta = features.sample_metadata
        cond_series = pd.Series({
            idx: meta.loc[idx, condition_column] if idx in meta.index else "Unknown"
            for idx in data_df.index
        })
        unique_conds = list(dict.fromkeys(cond_series.dropna().tolist()))
        if len(unique_conds) > 1:
            # Sort samples by condition so each condition forms a contiguous block
            sorted_indices = sorted(data_df.index, key=lambda idx: (str(cond_series.get(idx, "")), str(idx)))
            data_df = data_df.loc[sorted_indices]
            cond_series = cond_series.loc[sorted_indices]
            palette = sns.color_palette("tab10", len(unique_conds))
            color_map = dict(zip(unique_conds, palette))
            row_colors = cond_series.map(color_map)
        else:
            has_condition_legend = False

    # Dynamic sizing analytically enforcing square matrix cells
    n_samples, n_pops = data_df.shape
    s = 0.32 if n_samples <= 12 else 0.22  # Cell size in inches

    do_row_cluster = not has_condition_legend and n_samples >= 3
    dendro_w = 0.12 if do_row_cluster else 0.0
    dendro_h = 0.12
    color_w = 0.04 if row_colors is not None else 0.0

    # Initial sizing placeholder; will adjust precisely after GridSpec initialization
    init_w = max(6.0, (n_pops * s) + 2.4)
    init_h = max(5.2, (n_samples * s) + 3.2)
    total_w, total_h = figsize or (init_w, init_h)

    # Use Seaborn clustermap with square cells and refined dendrograms
    cg = sns.clustermap(
        data_df,
        row_colors=row_colors,
        row_cluster=do_row_cluster,
        col_cluster=True,
        cmap=cmap,
        figsize=(total_w, total_h),
        linewidths=0.4,
        linecolor="white",
        dendrogram_ratio=(dendro_w if do_row_cluster else 0.02, dendro_h),
        colors_ratio=color_w if row_colors is not None else 0.02,
        cbar_pos=None,
    )

    # Exact aspect ratio control: enforce true square cells across any dimension
    w_ratios = cg.gs.get_width_ratios()
    h_ratios = cg.gs.get_height_ratios()
    f_w = w_ratios[-1] / sum(w_ratios)
    f_h = h_ratios[-1] / sum(h_ratios)

    gs_w_in = (n_pops * s) / f_w
    gs_h_in = (n_samples * s) / f_h

    max_pop_len = max((len(str(c)) for c in data_df.columns), default=12)
    label_drop_in = max(0.9, 0.052 * max_pop_len * math.sin(math.radians(35)) + 0.35)
    margin_left_in = 1.6
    margin_bottom_in = label_drop_in + 0.35 + 0.25 + 0.40 + (0.65 if has_condition_legend else 0.0) + 0.20
    margin_top_in = 0.8

    if figsize is None:
        total_w = max(5.8, gs_w_in + margin_left_in + 0.6)
        total_h = max(5.2, gs_h_in + margin_bottom_in + margin_top_in)
        cg.fig.set_size_inches(total_w, total_h)

    left = margin_left_in / total_w
    right = left + (gs_w_in / total_w)
    bottom = margin_bottom_in / total_h
    top = bottom + (gs_h_in / total_h)
    cg.fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    # Label row_colors strip ('Condition') alongside the population x-labels and add a clean separation gap
    if cg.ax_row_colors is not None:
        strip_label = condition_column.replace("_", " ").title() if condition_column else "Condition"
        cg.ax_row_colors.set_xticks([0.5])
        cg.ax_row_colors.set_xticklabels(
            [strip_label],
            rotation=35,
            ha="right",
            fontsize=7.0,
            fontweight="bold",
        )
        cg.ax_row_colors.tick_params(axis="x", which="both", length=2.5, width=0.6, bottom=True)
        rc_pos = cg.ax_row_colors.get_position()
        gap_rc = 0.08 / total_w  # ~2 mm crisp separation gap
        cg.ax_row_colors.set_position([rc_pos.x0 - gap_rc, rc_pos.y0, rc_pos.width, rc_pos.height])
        if do_row_cluster and hasattr(cg, "ax_row_dendrogram") and cg.ax_row_dendrogram is not None:
            rd_pos = cg.ax_row_dendrogram.get_position()
            cg.ax_row_dendrogram.set_position([rd_pos.x0 - gap_rc, rd_pos.y0, rd_pos.width, rd_pos.height])

    cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xticklabels(), rotation=35, ha="right", fontsize=7.0)
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6.8)
    cg.ax_heatmap.set_xlabel("Populations / Clusters", fontsize=7.8, fontweight="bold", labelpad=8)
    cg.ax_heatmap.set_ylabel("Cohort Samples", fontsize=7.8, fontweight="bold")

    # Outer crisp black border surrounding the overall heatmap matrix box
    border_lw = round(max(0.55, min(0.85, 0.50 + 0.01 * max(n_samples, n_pops))), 2)
    rect = patches.Rectangle(
        (0, 0), 1, 1,
        fill=False,
        edgecolor="black",
        linewidth=border_lw,
        transform=cg.ax_heatmap.transAxes,
        clip_on=False,
        zorder=100,
    )
    cg.ax_heatmap.add_patch(rect)

    # Center title directly over the heatmap matrix
    h_pos = cg.ax_heatmap.get_position()
    center_heat_x = h_pos.x0 + 0.5 * h_pos.width
    cg.fig.suptitle(
        f"Cohort Hierarchical Profile: {title_label}",
        fontsize=8.8,
        fontweight="bold",
        x=center_heat_x,
        ha="center",
        y=top + (0.35 / total_h),
    )

    # Refine dendrogram tree line weights for a compact, clean look
    if do_row_cluster and hasattr(cg, "ax_row_dendrogram") and cg.ax_row_dendrogram:
        for coll in cg.ax_row_dendrogram.collections:
            coll.set_linewidth(0.8)
    if hasattr(cg, "ax_col_dendrogram") and cg.ax_col_dendrogram:
        for coll in cg.ax_col_dendrogram.collections:
            coll.set_linewidth(0.8)

    # Centered colorbar and condition legend layout with guaranteed clearance below x-axis title
    cg.fig.canvas.draw()
    renderer = cg.fig.canvas.get_renderer()
    h_tight = cg.ax_heatmap.get_tightbbox(renderer).transformed(cg.fig.transFigure.inverted())

    gap_y = 0.22 / total_h
    cbar_top = h_tight.y0 - gap_y
    cbar_h = 0.14 / total_h
    cbar_w = min(0.60 * h_pos.width, 2.4 / total_w)
    cbar_x = center_heat_x - 0.5 * cbar_w
    cbar_y = max((0.75 if has_condition_legend else 0.10) / total_h, cbar_top - cbar_h)

    if has_condition_legend:
        # Tier 1 (Upper-bottom): Horizontal colorbar centered under heatmap matrix
        cbar_ax = cg.fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
        cbar = cg.fig.colorbar(cg.ax_heatmap.collections[0], cax=cbar_ax, orientation="horizontal")
        cbar.set_label(title_label, fontsize=6.8, fontweight="medium")
        cbar.ax.tick_params(labelsize=6.0)

        # Tier 2 (Lower-bottom): Condition group legend dynamically placed below colorbar
        cg.fig.canvas.draw()
        cb_tight = cbar_ax.get_tightbbox(renderer).transformed(cg.fig.transFigure.inverted())
        legend_y = max(0.04 / total_h, cb_tight.y0 - (0.12 / total_h))

        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor=color_map[c], edgecolor="none", label=str(c))
            for c in unique_conds
        ]
        cg.fig.legend(
            handles=legend_patches,
            title=condition_column.replace("_", " ").title(),
            loc="upper center",
            bbox_to_anchor=(center_heat_x, legend_y),
            ncol=min(4, len(legend_patches)),
            frameon=True,
            framealpha=0.95,
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            fontsize=6.8,
            title_fontsize=7.2,
        )
    else:
        # Single Tier: Centered horizontal colorbar under x-axis
        cbar_ax = cg.fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
        cbar = cg.fig.colorbar(cg.ax_heatmap.collections[0], cax=cbar_ax, orientation="horizontal")
        cbar.set_label(title_label, fontsize=6.8, fontweight="medium")
        cbar.ax.tick_params(labelsize=6.0)

    fig = cg.fig
    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        logger.info(f"Saved cohort heatmap to '{dest.name}'.")

    return fig


def plot_forest_da(
    da_result: DAResult,
    output_path: Optional[Union[str, Path]] = None,
    fdr_threshold: Optional[float] = None,
    log2fc_threshold: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Render a publication-ready Forest Plot of Differential Abundance effect sizes (Log2FC).

    Displays horizontal point estimates with 95% Confidence Intervals for each cell population,
    highlighting significantly altered lineages between cohorts.

    Args:
        da_result: DAResult object holding population statistics.
        output_path: File path to save figure.
        fdr_threshold: Significance q-value cutoff.
        log2fc_threshold: Magnitude threshold.
        figsize: Figure dimensions.
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    df = da_result.summary_df.dropna(subset=["log2fc"]).copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize or (5.5, 4.0))
        ax.text(0.5, 0.5, "No population data available", ha="center", va="center")
        return fig

    # Sort populations by log2fc
    df = df.sort_values(by="log2fc", ascending=True).reset_index(drop=True)
    n_pops = len(df)

    calc_h = max(3.8, n_pops * 0.28 + 1.2)
    fig_w, fig_h = figsize or (5.5, calc_h)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    fdr_cut = fdr_threshold if fdr_threshold is not None else da_result.config.fdr_threshold
    l2fc_cut = log2fc_threshold if log2fc_threshold is not None else da_result.config.log2fc_threshold

    y_pos = np.arange(n_pops)
    log2fc = df["log2fc"].to_numpy()
    q_vals = df["q_value"].fillna(1.0).to_numpy()

    # Empirical SE for CI95 based on population frequency variance
    se = np.array([0.22 * max(0.2, abs(float(l))) for l in log2fc])
    ci_low = log2fc - 1.96 * se
    ci_high = log2fc + 1.96 * se

    # Vertical reference line at 0
    ax.axvline(0, color="#94A3B8", linestyle="--", linewidth=1.0, zorder=1)
    ax.axvline(l2fc_cut, color="#CBD5E1", linestyle=":", linewidth=0.8, zorder=1)
    ax.axvline(-l2fc_cut, color="#CBD5E1", linestyle=":", linewidth=0.8, zorder=1)

    colors = []
    for l, q in zip(log2fc, q_vals):
        if q < fdr_cut and abs(l) >= l2fc_cut:
            colors.append("#E11D48" if l > 0 else "#2563EB")
        else:
            colors.append("#64748B")

    # Plot error bars (confidence intervals)
    for i in range(n_pops):
        ax.plot([ci_low[i], ci_high[i]], [y_pos[i], y_pos[i]], color=colors[i], linewidth=1.5, zorder=2)
        ax.scatter(log2fc[i], y_pos[i], color=colors[i], s=35, zorder=3, edgecolors="white", linewidths=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["population"], fontsize=7.5)
    ax.set_xlabel(f"Effect Size: Log2 Fold Change ({da_result.target_group} vs {da_result.reference_group})", fontsize=8.0, fontweight="bold")
    ax.set_title(title or f"Differential Abundance Forest Plot\n({da_result.target_group} vs {da_result.reference_group})", fontsize=9.0, fontweight="bold", pad=8)
    apply_nature_axis_style(ax)
    ax.set_box_aspect(1)
    plt.tight_layout()

    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        logger.info(f"Saved DA forest plot to '{dest.name}'.")

    return fig


def plot_differential_state_heatmap(
    ds_result: DSResult,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Render a publication-ready Differential State (DS) marker modulation heatmap.

    Pivots population-by-marker delta MFI modulations into a diverging matrix with
    significance overlay indicators.

    Args:
        ds_result: DSResult holding per-population marker modulations.
        output_path: File path to save figure.
        figsize: Figure dimensions.
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    val_col = "delta_stat" if "delta_stat" in ds_result.summary_df.columns else "delta_mfi"
    if val_col not in ds_result.summary_df.columns:
        fig, ax = plt.subplots(figsize=figsize or (5.5, 4.0))
        ax.text(0.5, 0.5, "No differential state data available", ha="center", va="center")
        return fig

    df = ds_result.summary_df.dropna(subset=[val_col]).copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize or (5.5, 4.0))
        ax.text(0.5, 0.5, "No differential state data available", ha="center", va="center")
        return fig

    # Pivot delta matrix: Rows = Populations, Columns = Markers
    pivot_df = df.pivot(index="population", columns="marker", values=val_col).fillna(0.0)
    sig_pivot = df.pivot(index="population", columns="marker", values="is_significant").infer_objects(copy=False).fillna(False)

    n_rows, n_cols = pivot_df.shape
    s = 0.32
    w_mat = n_cols * s
    h_mat = n_rows * s
    calc_w = max(5.5, w_mat + 2.8)
    calc_h = max(4.0, h_mat + 1.8)
    fig_w, fig_h = figsize or (calc_w, calc_h)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    ax = fig.add_axes([1.8 / fig_w, 1.0 / fig_h, w_mat / fig_w, h_mat / fig_h])
    ax_cbar = fig.add_axes([(1.8 + w_mat + 0.15) / fig_w, 1.0 / fig_h, 0.14 / fig_w, h_mat / fig_h])

    # Center colormap at 0 using max absolute value
    max_val = max(0.1, float(np.abs(pivot_df.values).max()))
    im = ax.imshow(pivot_df.values, cmap="coolwarm", vmin=-max_val, vmax=max_val, aspect="equal")

    # Inner grid lines
    for r in range(n_rows - 1):
        ax.axhline(r + 0.5, color="white", linewidth=0.35, alpha=0.8)
    for c in range(n_cols - 1):
        ax.axvline(c + 0.5, color="white", linewidth=0.35, alpha=0.8)

    # Add significance asterisks in cells with significant modulation
    for r in range(n_rows):
        for c in range(n_cols):
            if sig_pivot.iloc[r, c]:
                ax.text(c, r, "★", ha="center", va="center", color="black", fontsize=8.0)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right", fontsize=7.5, fontweight="medium")
    ax.set_yticklabels(pivot_df.index, fontsize=7.5)
    ax.set_xlabel("Functional Markers", fontsize=8.0, fontweight="bold", labelpad=6)
    ax.set_ylabel("Populations", fontsize=8.0, fontweight="bold")

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Marker Delta MFI", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    center_x = (1.8 + 0.5 * w_mat) / fig_w
    fig.suptitle(
        title or f"Differential State (Delta MFI: {ds_result.target_group} vs {ds_result.reference_group})",
        fontsize=9.0,
        fontweight="bold",
        x=center_x,
        ha="center",
        y=1.0 - (0.2 / fig_h),
    )

    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        logger.info(f"Saved DS heatmap to '{dest.name}'.")

    return fig

