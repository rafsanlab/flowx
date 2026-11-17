"""Batch Preparation and Normalization Diagnostic Plotting Module.

Generates Nature Methods-styled publication diagnostic figures:
- Multi-batch marker density overlays before vs after normalization
- Earth Mover's Distance (EMD / Wasserstein distance) reduction bar charts
- Cohort downsampling and event balance summaries
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from flowx.batch.models import CohortSampleInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from flowx.visualization.style import apply_nature_axis_style, set_nature_style

logger = logging.getLogger(__name__)

# Nature-calibrated batch palette
BATCH_PALETTE = [
    "#2E7D32",  # Deep Forest Green
    "#1565C0",  # Saturated Blue
    "#C62828",  # Crimson Red
    "#6A1B9A",  # Royal Violet
    "#EF6C00",  # Ochre Orange
    "#00838F",  # Teal
    "#455A64",  # Slate
    "#AD1457",  # Deep Rose
]


def plot_batch_density_comparison(
    raw_cohort: Dict[str, pd.DataFrame],
    normalized_cohort: Dict[str, pd.DataFrame],
    sample_info: Dict[str, CohortSampleInfo],
    markers: List[str],
    output_path: Optional[Union[str, Path]] = None,
    controls_only: bool = True,
    max_markers: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Generate multi-panel marker density overlays comparing distributions before vs after batch correction.

    Args:
        raw_cohort: Dictionary mapping sample_id to raw event DataFrame.
        normalized_cohort: Dictionary mapping sample_id to normalized event DataFrame.
        sample_info: Dictionary mapping sample_id to CohortSampleInfo.
        markers: List of marker channel names to visualize.
        output_path: Optional path to save high-resolution figure (PNG/PDF).
        controls_only: Whether to plot technical controls only (cleaner view of technical drift).
        max_markers: Maximum number of marker rows to plot (default 4).
        figsize: Optional explicit figure size (width, height) in inches.

    Returns:
        Rendered Matplotlib Figure.
    """
    set_nature_style()

    # Identify batches and samples
    batch_to_samples: Dict[str, List[str]] = {}
    for s_id, s_info in sample_info.items():
        if controls_only and not s_info.is_control:
            continue
        batch_to_samples.setdefault(s_info.batch_id, []).append(s_id)

    unique_batches = sorted(batch_to_samples.keys())
    active_markers = markers[:max_markers]
    n_rows = len(active_markers)

    if n_rows == 0 or len(unique_batches) < 2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Insufficient batches or markers for comparison", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig

    calc_height = max(2.6, 1.65 * n_rows)
    fig_dim = figsize or (5.8, calc_height)
    fig, axes = plt.subplots(n_rows, 2, figsize=fig_dim, squeeze=False)

    for row_idx, marker in enumerate(active_markers):
        ax_raw = axes[row_idx, 0]
        ax_norm = axes[row_idx, 1]

        # Calculate global bounds for this marker
        all_vals = []
        for s_list in batch_to_samples.values():
            for s in s_list:
                if s in raw_cohort and marker in raw_cohort[s].columns:
                    all_vals.append(raw_cohort[s][marker].to_numpy())
        if all_vals:
            flat_all = np.concatenate(all_vals)
            p1, p99 = np.percentile(flat_all, [0.5, 99.5])
            span = max(1.0, p99 - p1)
            x_grid = np.linspace(p1 - 0.1 * span, p99 + 0.1 * span, 200)
        else:
            x_grid = np.linspace(0, 10, 200)

        # Plot density for each batch with distinct linestyles to prevent overlaid curves from masking each other
        linestyles = ["-", "--", "-.", ":"]
        for b_idx, b_id in enumerate(unique_batches):
            color = BATCH_PALETTE[b_idx % len(BATCH_PALETTE)]
            ls = linestyles[b_idx % len(linestyles)]
            s_list = batch_to_samples[b_id]

            # Raw density
            raw_v_list = [raw_cohort[s][marker].to_numpy() for s in s_list if s in raw_cohort and marker in raw_cohort[s].columns]
            if raw_v_list:
                raw_v = np.concatenate(raw_v_list)
                if len(raw_v) > 20 and np.std(raw_v) > 1e-5:
                    try:
                        kde_raw = gaussian_kde(raw_v, bw_method="scott")
                        ax_raw.plot(x_grid, kde_raw(x_grid), color=color, linestyle=ls, lw=1.8, label=f"Batch {b_id}")
                        ax_raw.fill_between(x_grid, kde_raw(x_grid), color=color, alpha=0.12)
                    except Exception:
                        ax_raw.hist(raw_v, bins=30, density=True, color=color, alpha=0.3, label=f"Batch {b_id}")

            # Normalized density
            norm_v_list = [normalized_cohort[s][marker].to_numpy() for s in s_list if s in normalized_cohort and marker in normalized_cohort[s].columns]
            if norm_v_list:
                norm_v = np.concatenate(norm_v_list)
                if len(norm_v) > 20 and np.std(norm_v) > 1e-5:
                    try:
                        kde_norm = gaussian_kde(norm_v, bw_method="scott")
                        ax_norm.plot(x_grid, kde_norm(x_grid), color=color, linestyle=ls, lw=1.8, label=f"Batch {b_id}")
                        ax_norm.fill_between(x_grid, kde_norm(x_grid), color=color, alpha=0.12)
                    except Exception:
                        ax_norm.hist(norm_v, bins=30, density=True, color=color, alpha=0.3, label=f"Batch {b_id}")

        # Subplot Titles & Formatting
        apply_nature_axis_style(ax_raw, show_grid=True)
        apply_nature_axis_style(ax_norm, show_grid=True)

        ax_raw.set_ylabel(f"{marker}\nDensity", fontsize=7.8, fontweight="medium")
        if row_idx == 0:
            ax_raw.set_title("Before Normalization (Raw Drift)", fontsize=8.8, fontweight="bold", pad=6)
            ax_norm.set_title("After Normalization (Batch Aligned)", fontsize=8.8, fontweight="bold", pad=6)
            ax_raw.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=7.0, loc="upper right")
            ax_norm.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=7.0, loc="upper right")

    plt.tight_layout()
    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        logger.info(f"Saved batch density comparison figure to '{dest.name}'.")

    return fig


def plot_batch_metrics_summary(
    metrics_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    max_markers: int = 15,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of Earth Mover's Distance reduction percentage per marker.

    Args:
        metrics_df: DataFrame output from calculate_batch_invariance_metrics.
        output_path: Optional destination path for figure.
        max_markers: Maximum number of markers to display.
        figsize: Optional explicit figure size.

    Returns:
        Rendered Matplotlib Figure.
    """
    set_nature_style()

    if metrics_df.empty or "emd_reduction_pct" not in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig

    # Group by marker and average across batch pairs
    avg_metrics = metrics_df.groupby("marker")["emd_reduction_pct"].mean().sort_values(ascending=True)
    if len(avg_metrics) > max_markers:
        avg_metrics = avg_metrics.tail(max_markers)

    markers_list = avg_metrics.index.tolist()
    reductions = avg_metrics.to_numpy()

    fig_dim = figsize or (5.4, max(2.6, 0.22 * len(markers_list) + 1.0))
    fig, ax = plt.subplots(figsize=fig_dim)

    # Color bars: Emerald green for reduction > 0, subtle red for < 0
    colors = ["#16A34A" if r >= 0 else "#DC2626" for r in reductions]
    y_pos = np.arange(len(markers_list))

    bars = ax.barh(y_pos, reductions, color=colors, height=0.65, edgecolor="none", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(markers_list, fontsize=7.5)

    # Vertical reference lines
    ax.axvline(0.0, color="#64748B", lw=0.9, ls="-")
    mean_red = float(np.mean(reductions))
    ax.axvline(
        mean_red,
        color="#2563EB",
        lw=1.2,
        ls="--",
        label=f"Cohort Mean ({mean_red:+.1f}%)",
    )

    # Value labels beside bars
    for bar, val in zip(bars, reductions):
        offset = 1.0 if val >= 0 else -1.0
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}%", va="center", ha=ha, fontsize=6.8)

    apply_nature_axis_style(ax, show_grid=True)
    ax.set_xlabel("Earth Mover's Distance (EMD) Reduction (%)", fontsize=7.8, fontweight="medium")
    ax.set_title("Inter-Batch Technical Variation Reduction by Marker", fontsize=8.8, fontweight="bold", pad=8)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=7.0, loc="lower right")

    plt.tight_layout()
    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        logger.info(f"Saved batch metrics summary figure to '{dest.name}'.")

    return fig


def plot_cohort_downsampling_balance(
    sample_info: List[CohortSampleInfo],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot stacked/grouped bar chart illustrating cohort sample event yields and downsampling balance.

    Args:
        sample_info: List of CohortSampleInfo instances.
        output_path: Optional destination path for figure.
        figsize: Optional explicit figure size.

    Returns:
        Rendered Matplotlib Figure.
    """
    set_nature_style()

    if not sample_info:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No sample tracking info provided", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig

    sample_ids = [s.sample_id for s in sample_info]
    raw_counts = []
    clean_counts = []
    down_counts = [s.prepared_event_count for s in sample_info]

    for s in sample_info:
        # Resolve true raw event count
        r_cnt = s.raw_event_count
        if r_cnt == 0 and s.extra_metadata:
            r_cnt = int(s.extra_metadata.get("raw_event_count") or s.extra_metadata.get("raw_events") or s.extra_metadata.get("total_events") or 0)
        raw_counts.append(r_cnt)

        # Resolve true clean event count (never blindly default to raw count)
        c_cnt = s.clean_event_count
        if c_cnt == 0 and s.extra_metadata:
            c_cnt = int(s.extra_metadata.get("clean_event_count") or s.extra_metadata.get("clean_events") or 0)
        if c_cnt == 0 and s.prepared_event_count > 0:
            c_cnt = s.prepared_event_count
        clean_counts.append(c_cnt)

    n_samples = len(sample_ids)
    fig_width = min(9.6, max(5.6, 0.36 * n_samples + 2.4))
    fig_dim = figsize or (fig_width, 3.5)
    fig, ax = plt.subplots(figsize=fig_dim)

    x = np.arange(n_samples)
    width = 0.28

    ax.bar(x - width, raw_counts, width, label="Raw Acquired Events", color="#94A3B8", alpha=0.85)
    ax.bar(x, clean_counts, width, label="Clean Viable Singlets (Step 2)", color="#3B82F6", alpha=0.9)
    ax.bar(x + width, down_counts, width, label="Balanced Downsampled (Step 3)", color="#10B981", alpha=0.95)

    valid_down = [d for d in down_counts if d > 0]
    if valid_down and len(set(valid_down)) == 1:
        ax.axhline(
            valid_down[0],
            color="#EF4444",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            label=f"Target Quota ({valid_down[0]:,} events)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sample_ids, rotation=45, ha="right", fontsize=7.2)
    apply_nature_axis_style(ax, show_grid=True)

    ax.set_ylabel("Event Count", fontsize=7.8, fontweight="medium")
    ax.set_title("Cohort Event Balancing & Preparation Progression", fontsize=8.8, fontweight="bold", pad=8)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=7.2,
    )

    plt.tight_layout()
    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        logger.info(f"Saved cohort downsampling balance figure to '{dest.name}'.")

    return fig


def plot_batch_assessment_summary(
    assessment: Any,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Render a publication-ready diagnostic summary of the batch necessity assessment.

    Includes an executive recommendation badge, drift score metric, and per-marker
    inter-batch Earth Mover's Distance (EMD) bar chart with drift severity thresholds.

    Args:
        assessment: BatchNecessityAssessment instance.
        output_path: Optional destination path for figure export.
        figsize: Optional explicit figure size.

    Returns:
        Rendered Matplotlib Figure.
    """
    set_nature_style()

    color_theme = {
        "NOT_NEEDED": ("#10B981", "NOT NEEDED (Skip Batch Normalization)"),
        "RECOMMENDED_CYTONORM": ("#2563EB", "RECOMMENDED (Anchor-Based CytoNorm)"),
        "CAUTION_ANCHOR_FREE": ("#F59E0B", "CAUTION (Anchor-Free Quantile Only)"),
        "NOT_RECOMMENDED_CONFOUNDED": ("#EF4444", "NOT RECOMMENDED (Confounded Design)"),
    }

    rec = getattr(assessment, "recommendation", "NOT_NEEDED")
    badge_color, badge_text = color_theme.get(rec, ("#64748B", str(rec)))
    drift_score = float(getattr(assessment, "drift_score", 0.0))
    confidence = float(getattr(assessment, "confidence", 1.0)) * 100.0
    marker_metrics = getattr(assessment, "marker_metrics", [])

    fig_dim = figsize or (5.6, 3.8)
    fig, (ax_top, ax_bars) = plt.subplots(
        2, 1, figsize=fig_dim, gridspec_kw={"height_ratios": [1, 3.2]}
    )

    # 1. Executive Banner Panel
    ax_top.axis("off")
    # Banner background box
    rect = plt.Rectangle(
        (0.02, 0.1),
        0.96,
        0.8,
        facecolor=badge_color,
        alpha=0.15,
        edgecolor=badge_color,
        linewidth=1.5,
        linestyle="-",
        transform=ax_top.transAxes,
        zorder=1,
    )
    ax_top.add_patch(rect)

    ax_top.text(
        0.05,
        0.65,
        f"RECOMMENDATION: {badge_text}",
        fontsize=9.2,
        fontweight="bold",
        color=badge_color,
        transform=ax_top.transAxes,
        va="center",
    )
    anchor_str = "Yes" if getattr(assessment, "has_control_anchors", False) else "No"
    batch_cnt = getattr(assessment, "batch_count", 1)
    summary_line = (
        f"Drift Score: {drift_score:.1f}/100.0   |   Confidence: {confidence:.0f}%   |   "
        f"Batches: {batch_cnt}   |   Anchors Present: {anchor_str}"
    )
    ax_top.text(
        0.05,
        0.30,
        summary_line,
        fontsize=7.2,
        color="#334155",
        transform=ax_top.transAxes,
        va="center",
        fontfamily="sans-serif",
    )

    # 2. Per-Marker EMD Drift Bar Chart
    if marker_metrics:
        m_names = [m.marker for m in marker_metrics]
        emds = [m.mean_pairwise_emd for m in marker_metrics]
        drifting = [m.is_drifting for m in marker_metrics]
        y_pos = np.arange(len(m_names))

        bar_colors = ["#EF4444" if d else "#3B82F6" for d in drifting]
        ax_bars.barh(y_pos, emds, color=bar_colors, alpha=0.88, height=0.62)
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(m_names, fontsize=7.5)
        ax_bars.invert_yaxis()  # Top-down order

        # Threshold dashed line
        ax_bars.axvline(
            0.12,
            color="#DC2626",
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            label="Drift Flag Threshold (EMD = 0.12)",
        )

        ax_bars.set_xlabel("Mean Pairwise Earth Mover's Distance (EMD)", fontsize=7.8, fontweight="medium")
        ax_bars.set_title("Marker-Level Inter-Batch Distribution Drift", fontsize=8.8, fontweight="bold", pad=6)
        apply_nature_axis_style(ax_bars, show_grid=True)
        ax_bars.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=7.0, loc="lower right")
    else:
        ax_bars.text(
            0.5,
            0.5,
            "Single batch or homogeneous cohort:\nInter-batch drift calculation bypassed (EMD = 0.0)",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#64748B",
            transform=ax_bars.transAxes,
        )
        ax_bars.set_axis_off()

    plt.tight_layout()

    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        logger.info(f"Saved batch assessment summary figure to '{dest.name}'.")

    return fig

