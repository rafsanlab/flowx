"""Publication-ready visualization routines for spectral flow cytometry.

Generates spectral signature ribbon plots across multi-laser optical detector arrays,
Cosine Spectral Similarity heatmaps, per-event unmixing residual diagnostic distributions,
and multi-panel publication figures in Nature-compliant style.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from flowx.spectral.models import (
    ReferenceSpectralLibrary,
    SpectralMetricsResult,
    UnmixingResult,
)
from flowx.visualization.style import nature_style

logger = logging.getLogger(__name__)


def format_multiline_detector(name: str, max_chars_per_line: int = 6) -> str:
    """Format an optical detector channel name across multiple lines to prevent overlap.

    Handles hyphenated channel names (e.g., 'BUV395-A' -> 'BUV395\\n-A'), spaced names,
    slashed wavelengths, and long identifiers.
    """
    if not name or len(name) <= max_chars_per_line:
        return name
    if " " in name:
        words = name.split(" ")
        lines: List[str] = []
        cur = ""
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_chars_per_line:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return "\n".join(lines)
    if "-" in name:
        parts = name.split("-")
        return "\n-".join(parts)
    if "/" in name:
        return "\n/".join(name.split("/"))
    if "_" in name:
        return "\n_".join(name.split("_"))
    import textwrap
    return "\n".join(textwrap.wrap(name, width=max_chars_per_line))


def get_laser_theme_color(laser_name: str) -> str:
    """Return canonical wavelength-specific color matching the excitation laser name."""
    name_upper = laser_name.upper()
    if "UV" in name_upper or "355" in name_upper:
        return "#7B1FA2"  # Deep Ultraviolet (355nm)
    elif "VIOLET" in name_upper or "405" in name_upper:
        return "#5E35B1"  # Deep Violet (405nm)
    elif "BLUE" in name_upper and "YELLOW" not in name_upper and "488" in name_upper:
        return "#1976D2"  # True Blue (488nm)
    elif "YELLOW" in name_upper or "YG" in name_upper or "561" in name_upper:
        return "#D8A200"  # Rich Golden Yellow-Green (561nm)
    elif "RED" in name_upper or "638" in name_upper or "640" in name_upper:
        return "#D32F2F"  # Laser Red (638/640nm)
    elif "BLUE" in name_upper:
        return "#1976D2"
    return "#616161"  # Neutral Slate


def format_multiline_laser_label(laser_name: str, span_width: int) -> str:
    """Dynamically format laser label to prevent overlapping on narrow or crowded detector bands."""
    if span_width <= 3:
        if "UV" in laser_name:
            return "UV\n355"
        elif "Violet" in laser_name:
            return "V\n405"
        elif "Blue" in laser_name:
            return "B\n488"
        elif "Yellow" in laser_name or "YG" in laser_name:
            return "YG\n561"
        elif "Red" in laser_name:
            return "R\n640"
        return laser_name[:4]
    elif span_width <= 8:
        if " (" in laser_name:
            return laser_name.replace(" (", "\n(")
        return laser_name
    else:
        if len(laser_name) > 13 and " (" in laser_name:
            return laser_name.replace(" (", "\n(")
        return laser_name


def group_detectors_by_laser(detectors: List[str]) -> Dict[str, List[str]]:
    """Categorize optical detector channels into excitation laser groups.

    Recognizes standard Cytek, Sony, and BD detector conventions:
    - UV (355nm): 'UV1', 'UV445', 'BUV'
    - Violet (405nm): 'V1', 'V445', 'V615', 'BV'
    - Blue (488nm): 'B1', 'B525', 'B615', 'BB'
    - Yellow-Green (561nm): 'YG1', 'Y1', 'Y586', 'Y615'
    - Red (637/640nm): 'R1', 'R667', 'R725', 'R780'

    Args:
        detectors: List of optical detector channel names.

    Returns:
        Dictionary mapping laser band name to ordered list of detector channels.
    """
    groups: Dict[str, List[str]] = {
        "UV (355nm)": [],
        "Violet (405nm)": [],
        "Blue (488nm)": [],
        "Yellow-Green (561nm)": [],
        "Red (638/640nm)": [],
        "Other": [],
    }

    for det in detectors:
        d_up = det.upper().strip()
        if d_up.startswith("UV") or "BUV" in d_up or "355" in d_up:
            groups["UV (355nm)"].append(det)
        elif d_up.startswith("V") or "BV" in d_up or "405" in d_up:
            groups["Violet (405nm)"].append(det)
        elif d_up.startswith("B") or "BB" in d_up or "488" in d_up:
            groups["Blue (488nm)"].append(det)
        elif d_up.startswith("YG") or d_up.startswith("Y") or "561" in d_up:
            groups["Yellow-Green (561nm)"].append(det)
        elif d_up.startswith("R") or "640" in d_up or "638" in d_up or "633" in d_up:
            groups["Red (638/640nm)"].append(det)
        else:
            groups["Other"].append(det)

    # Prune empty groups
    return {k: v for k, v in groups.items() if v}


def plot_spectral_ribbons(
    spectral_library: ReferenceSpectralLibrary,
    output_path: Optional[Union[str, Path]] = None,
    max_fluorophores: int = 40,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot multi-laser normalized spectral emission signatures across all optical detectors.

    Calibrated for Nature publication specifications with large text-to-plot ratio,
    clean laser band shading, and strict vertical decoupling to prevent title overlap.

    Args:
        spectral_library: Configured ReferenceSpectralLibrary.
        output_path: Optional file path to save the generated figure.
        max_fluorophores: Maximum number of fluorophores to display simultaneously (default: 40).
        figsize: Optional figure dimensions in inches (width, height).
        ax: Optional matplotlib axes to plot onto.

    Returns:
        Matplotlib Figure object.
    """
    detectors = spectral_library.detectors
    fluorophores = spectral_library.fluorophores[:max_fluorophores] if max_fluorophores else spectral_library.fluorophores
    n_fluoro = len(fluorophores)
    grouped_dets = group_detectors_by_laser(detectors)

    # Order detectors strictly by laser group to guarantee laser shading spans align with detector tick marks
    ordered_detectors: List[str] = []
    for _laser_name, det_list in grouped_dets.items():
        ordered_detectors.extend(det_list)
    if not ordered_detectors or len(ordered_detectors) != len(detectors):
        ordered_detectors = list(detectors)

    # Dynamic legend columns and canvas sizing to prevent text clipping
    legend_cols = 1 if n_fluoro <= 14 else (2 if n_fluoro <= 28 else 3)
    legend_fs = 8.5 if n_fluoro <= 14 else (7.2 if n_fluoro <= 28 else 6.5)

    if figsize is None:
        extra_w = 0.0 if legend_cols == 1 else (2.0 if legend_cols == 2 else 3.8)
        actual_figsize = (8.8 + extra_w, max(4.8, min(6.5, 3.4 + 0.08 * n_fluoro)))
    else:
        actual_figsize = figsize

    with nature_style():
        if ax is None:
            fig, target_ax = plt.subplots(figsize=actual_figsize)
        else:
            fig = ax.figure
            target_ax = ax

        x_indices = np.arange(len(ordered_detectors))
        palette = sns.color_palette("tab20", n_colors=max(len(fluorophores), 1))

        for idx, fluoro in enumerate(fluorophores):
            raw_spec = spectral_library.get_spectrum(fluoro)
            det_to_val = dict(zip(detectors, raw_spec))
            spec = np.array([det_to_val[d] for d in ordered_detectors], dtype=np.float64)
            ref = spectral_library.references.get(fluoro)
            is_af = ref.is_autofluorescence if ref else False
            line_style = "--" if is_af else "-"
            line_width = 2.4 if is_af else 1.8
            alpha = 0.95 if is_af else 0.80

            target_ax.plot(
                x_indices,
                spec,
                label=f"{fluoro} (AF)" if is_af else fluoro,
                color=palette[idx],
                linestyle=line_style,
                linewidth=line_width,
                alpha=alpha,
            )

        # Annotate laser boundaries with dedicated clearance and laser-matched colors
        current_idx = 0
        for laser_name, det_list in grouped_dets.items():
            if not det_list:
                continue
            span_width = len(det_list)
            start_x = current_idx
            end_x = current_idx + span_width
            mid_x = (start_x + end_x - 1) / 2.0
            l_color = get_laser_theme_color(laser_name)
            target_ax.axvspan(
                start_x - 0.5,
                end_x - 0.5,
                color=l_color,
                alpha=0.07,
                zorder=0,
            )
            # Dynamic multiline label preventing inter-laser overlap
            lbl_text = format_multiline_laser_label(laser_name, span_width)
            target_ax.text(
                mid_x,
                1.02,
                lbl_text,
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color=l_color,
                linespacing=0.9,
                transform=target_ax.get_xaxis_transform(),
            )
            if end_x < len(ordered_detectors):
                target_ax.axvline(end_x - 0.5, color="#CCCCCC", linestyle=":", linewidth=0.8)
            current_idx = end_x

        target_ax.set_xticks(x_indices)
        # Simplify detector tick labels if many channels and auto multiline
        step = 1 if len(ordered_detectors) <= 32 else (2 if len(ordered_detectors) <= 64 else 4)
        tick_labels = [format_multiline_detector(det) if (i % step == 0) else "" for i, det in enumerate(ordered_detectors)]
        target_ax.set_xticklabels(tick_labels, rotation=90, fontsize=7.5, va="top", ha="center")

        target_ax.set_xlim(-0.5, len(ordered_detectors) - 0.5)
        target_ax.set_ylim(-0.02, 1.15)
        target_ax.set_xlabel("Optical Detector Channel", fontsize=10.0, fontweight="bold", labelpad=6)
        target_ax.set_ylabel("Normalized Emission Intensity", fontsize=10.0, fontweight="bold", labelpad=6)

        # Ample pad (32 pt) strictly decouples title from laser band labels
        target_ax.set_title(
            f"Spectral Emission Signatures ({len(ordered_detectors)} Detectors, κ={spectral_library.condition_number:.1f})",
            fontsize=11.0,
            fontweight="bold",
            pad=32,
        )
        target_ax.legend(
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            ncol=legend_cols,
            frameon=True,
            framealpha=0.92,
            edgecolor="#DDDDDD",
            fontsize=legend_fs,
            borderaxespad=0.2,
            columnspacing=0.8,
            handletextpad=0.4,
        )
        if ax is None:
            fig.tight_layout()

        if output_path and ax is None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=300, bbox_inches="tight")
            logger.info("Saved publication spectral ribbons plot: %s", out_p.name)

        return fig


def plot_spectral_similarity_heatmap(
    metrics_result: SpectralMetricsResult,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot pairwise Cosine Spectral Similarity Matrix heatmap in publication format.

    Dynamically sizes figure and cells based on panel size to ensure numeric values
    fit cleanly inside each box without overlapping text.

    Args:
        metrics_result: SpectralMetricsResult container.
        output_path: Optional file path to save the generated figure.
        figsize: Optional figure dimensions in inches (dynamically computed if None).
        ax: Optional matplotlib axes.

    Returns:
        Matplotlib Figure object.
    """
    sim_df = metrics_result.similarity_matrix
    n = len(sim_df)

    # Dynamic box and figure sizing to accommodate similarity values comfortably
    if figsize is None:
        cell_size = max(0.42, min(0.68, 8.0 / max(1, n)))
        fig_w = max(6.5, n * cell_size + 3.2)
        fig_h = max(5.5, n * cell_size + 2.4)
        actual_figsize = (fig_w, fig_h)
    else:
        actual_figsize = figsize

    # Dynamic annotation font size that scales with matrix dimension
    if n <= 8:
        annot_fs = 9.5
    elif n <= 14:
        annot_fs = 8.5
    elif n <= 20:
        annot_fs = 7.5
    elif n <= 28:
        annot_fs = 6.8
    elif n <= 40:
        annot_fs = 6.0
    else:
        annot_fs = 5.2

    # Scale tick font size with figure size so labels remain clear on large figures
    if n <= 16:
        tick_fs = 9.0
    elif n <= 28:
        tick_fs = 8.0
    elif n <= 40:
        tick_fs = 7.5
    else:
        tick_fs = 7.0

    show_annot = (n <= 45)
    title_fs = max(11.0, min(14.0, 10.0 + 0.08 * n))

    with nature_style():
        if ax is None:
            fig, target_ax = plt.subplots(figsize=actual_figsize)
        else:
            fig = ax.figure
            target_ax = ax

        sns.heatmap(
            sim_df,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            annot=show_annot,
            fmt=".2f",
            annot_kws={"size": annot_fs, "weight": "bold"},
            cbar_kws={"label": "Cosine Spectral Similarity", "shrink": 0.85},
            ax=target_ax,
            square=True,
        )

        target_ax.set_title(
            f"Spectral Similarity (Complexity Index = {metrics_result.complexity_index:.1f})",
            fontsize=title_fs,
            fontweight="bold",
            pad=14,
        )
        target_ax.set_xticklabels(target_ax.get_xticklabels(), rotation=90, fontsize=tick_fs)
        target_ax.set_yticklabels(target_ax.get_yticklabels(), rotation=0, fontsize=tick_fs)

        if ax is None:
            fig.tight_layout()

        if output_path and ax is None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=300, bbox_inches="tight")
            logger.info("Saved publication spectral similarity heatmap: %s", out_p.name)

        return fig


def plot_deconvolution_goodness_of_fit(
    unmixing_result: UnmixingResult,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (7.6, 3.8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot dedicated deconvolution goodness-of-fit diagnostic panel.

    Renders a dedicated standalone 2-panel figure:
    - Left (Panel A): Residual norm ||y_i - M x_i||_2 distribution histogram with
      median and +3.5 rSD outlier threshold.
    - Right (Panel B): Event Residual Norm vs Total Estimated Fluorophore Signal
      density scatter with running median trendline, evaluating Poisson noise
      scaling and heteroscedasticity.

    Args:
        unmixing_result: UnmixingResult container from unmix_dataframe.
        output_path: Optional file path to save figure.
        figsize: Figure dimensions in inches (default: 10.0 x 4.2).
        ax: Optional single matplotlib axes (renders distribution only if provided).

    Returns:
        Matplotlib Figure object.
    """
    res = unmixing_result.residuals_norm
    med = unmixing_result.median_residual_norm
    mad = float(np.median(np.abs(res - med)))
    rsd = max(1.4826 * mad, 1.0)
    outlier_thresh = med + (3.5 * rsd)
    outliers_pct = (unmixing_result.residual_outliers_count / max(1, len(res))) * 100.0

    with nature_style():
        if ax is not None:
            fig = ax.figure
            ax1 = ax
            ax2 = None
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        p99 = float(np.percentile(res, 99.0))
        bins = np.linspace(0.0, max(p99 * 1.2, outlier_thresh * 1.1), 50)

        # Panel 1: Residual Norm Distribution
        ax1.hist(res, bins=bins, color="#1976D2", alpha=0.75, edgecolor="white", density=True)
        ax1.axvline(med, color="#2E7D32", linestyle="-", linewidth=2.0, label=f"Median: {med:.1f}")
        ax1.axvline(
            outlier_thresh,
            color="#D32F2F",
            linestyle="--",
            linewidth=2.0,
            label=f"Outlier (+3.5 rSD): {outlier_thresh:.1f}",
        )
        ax1.text(
            0.95,
            0.82,
            f"Events: {len(res):,}\nOutliers: {unmixing_result.residual_outliers_count:,} ({outliers_pct:.2f}%)\nSolver: {unmixing_result.method_used.upper()}",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.9),
            fontsize=8.5,
        )
        ax1.set_xlabel("Residual Norm $\\|y_i - \\hat{y}_i\\|_2$", fontsize=9.5, fontweight="bold")
        ax1.set_ylabel("Probability Density", fontsize=9.5, fontweight="bold")
        ax1.set_title("Deconvolution Residual Norm Distribution", fontsize=10.5, fontweight="bold", pad=10)
        ax1.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax1.set_box_aspect(1)

        # Panel 2: Residual Norm vs Total Estimated Signal
        if ax2 is not None:
            # Calculate total positive unmixed abundance per event
            fl_cols = unmixing_result.fluorophore_channels
            if fl_cols and all(c in unmixing_result.unmixed_df.columns for c in fl_cols):
                tot_signal = np.maximum(unmixing_result.unmixed_df[fl_cols].values, 0.0).sum(axis=1)
            else:
                num_cols = [
                    c for c in unmixing_result.unmixed_df.columns
                    if c not in ("Time", "Event #", unmixing_result.residual_channel_name)
                    and np.issubdtype(unmixing_result.unmixed_df[c].dtype, np.number)
                ]
                if num_cols:
                    tot_signal = np.maximum(unmixing_result.unmixed_df[num_cols].values, 0.0).sum(axis=1)
                else:
                    tot_signal = np.zeros(len(res), dtype=float)

            n_pts = len(res)
            sub_n = min(12000, n_pts)
            if n_pts > sub_n:
                rng = np.random.RandomState(42)
                sub_idx = rng.choice(n_pts, size=sub_n, replace=False)
            else:
                sub_idx = np.arange(n_pts)

            x_sub = tot_signal[sub_idx]
            y_sub = res[sub_idx]

            ax2.scatter(x_sub, y_sub, c="#0288D1", alpha=0.35, s=6, edgecolor="none", rasterized=True)

            # Running median trendline across signal bins
            if len(x_sub) > 20:
                p98_x = float(np.percentile(x_sub, 98.0))
                if p98_x > 0:
                    x_bins = np.linspace(0.0, p98_x, 15)
                    bin_centers = []
                    bin_medians = []
                    for i in range(len(x_bins) - 1):
                        m = (x_sub >= x_bins[i]) & (x_sub < x_bins[i + 1])
                        if np.sum(m) > 10:
                            bin_centers.append(0.5 * (x_bins[i] + x_bins[i + 1]))
                            bin_medians.append(float(np.median(y_sub[m])))
                    if bin_centers:
                        ax2.plot(bin_centers, bin_medians, color="#D32F2F", linewidth=2.2, label="Running Median")

            ax2.set_xlabel("Total Estimated Fluorophore Signal (a.u.)", fontsize=9.5, fontweight="bold")
            ax2.set_ylabel("Residual Norm $\\|y_i - \\hat{y}_i\\|_2$", fontsize=9.5, fontweight="bold")
            ax2.set_title("Goodness-of-Fit vs Signal Abundance", fontsize=10.5, fontweight="bold", pad=10)
            handles, labels = ax2.get_legend_handles_labels()
            if handles:
                ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
            ax2.set_box_aspect(1)

        if ax is None:
            fig.tight_layout(w_pad=1.2)

        if output_path and ax is None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=300, bbox_inches="tight")
            logger.info("Saved deconvolution goodness-of-fit plot: %s", out_p.name)

        return fig


def plot_unmixing_residuals_distribution(
    unmixing_result: UnmixingResult,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (7.6, 3.8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Alias for plot_deconvolution_goodness_of_fit for backward compatibility."""
    return plot_deconvolution_goodness_of_fit(
        unmixing_result=unmixing_result,
        output_path=output_path,
        figsize=figsize,
        ax=ax,
    )
