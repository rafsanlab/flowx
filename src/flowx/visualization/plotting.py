"""Flow cytometry visualization and plotting module.

Provides high-quality, publication-ready scatter plots, 2D density plots,
compensation comparison diagnostics, transformation distribution checks,
and gate boundary overlays with statistical annotations.
"""

from __future__ import annotations

import math
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless reproducibility
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import dendrogram, linkage

from flowx.gating.gate import Gate, PolygonGate, RangeGate, RectangleGate
from flowx.visualization.style import (
    DEFAULT_FIGSIZE_DUAL,
    DEFAULT_FIGSIZE_HEATMAP,
    DEFAULT_FIGSIZE_QC,
    DEFAULT_FIGSIZE_SINGLE,
    DEFAULT_FIGSIZE_TRIPLE,
    DEFAULT_FIGSIZE_WIDE,
    apply_nature_axis_style,
    format_multiline_title,
    set_nature_style,
)

# Apply Nature publication style by default
set_nature_style()


def calculate_density_colors(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = 15000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate point density for scatter plot coloring.

    Subsamples data if event count exceeds max_points for speed while maintaining reproducibility.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        max_points: Maximum events to estimate and render.
        random_state: Seed for reproducible subsampling.

    Returns:
        Tuple of (sub_x, sub_y, density_values).
    """
    n = len(x)
    if n == 0:
        return x, y, np.empty(0)

    if n > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        sub_x = x[idx]
        sub_y = y[idx]
    else:
        sub_x = x
        sub_y = y

    # Fast 2D histogram approximation for density
    finite_mask = np.isfinite(sub_x) & np.isfinite(sub_y)
    sub_x = sub_x[finite_mask]
    sub_y = sub_y[finite_mask]

    if len(sub_x) < 2:
        return sub_x, sub_y, np.ones(len(sub_x))

    try:
        # 2D histogram bin density lookup
        nbins = 128
        h, xedges, yedges = np.histogram2d(sub_x, sub_y, bins=nbins)
        x_idx = np.clip(np.digitize(sub_x, xedges) - 1, 0, nbins - 1)
        y_idx = np.clip(np.digitize(sub_y, yedges) - 1, 0, nbins - 1)
        density = h[x_idx, y_idx]
    except Exception:
        density = np.ones(len(sub_x))

    return sub_x, sub_y, density


def plot_scatter_2d(
    df: pd.DataFrame,
    x_channel: str,
    y_channel: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    colormap: str = "turbo",
    max_points: int = 25000,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
    square_plot_area: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generate a 2D density-colored scatter plot for two flow cytometry channels.

    Ensures a standardized 1:1 square plot area by default across samples and channels,
    scaled compactly for publication subplots unless the caller explicitly specifies a custom figsize.

    Args:
        df: Event DataFrame.
        x_channel: Column name for x-axis.
        y_channel: Column name for y-axis.
        x_label: Custom label for x-axis.
        y_label: Custom label for y-axis.
        title: Title of plot.
        colormap: Matplotlib colormap name.
        max_points: Max points to display for responsiveness.
        output_path: Optional file path to export image.
        figsize: Figure width and height in inches. If None, defaults to compact publication subplot square (3.6, 3.6).
        dpi: Resolution for saved image.
        ax: Existing axes to draw onto.
        square_plot_area: If True and figsize not explicitly customized to a rectangle,
            enforces a 1:1 square plot box.

    Returns:
        Tuple of (fig, ax).
    """
    custom_figsize = figsize is not None
    standalone = ax is None
    if standalone:
        actual_figsize = figsize if custom_figsize else DEFAULT_FIGSIZE_SINGLE
        fig, ax = plt.subplots(figsize=actual_figsize, dpi=dpi)
    else:
        fig = ax.figure

    # Enforce square plot area unless caller specified a custom non-square figsize for a standalone plot
    if square_plot_area:
        if not standalone or not custom_figsize or (figsize and figsize[0] == figsize[1]):
            ax.set_box_aspect(1)

    x_vals = df[x_channel].to_numpy()
    y_vals = df[y_channel].to_numpy()

    sub_x, sub_y, density = calculate_density_colors(x_vals, y_vals, max_points=max_points)

    # Sort so dense points plot on top
    sort_idx = np.argsort(density)
    sub_x = sub_x[sort_idx]
    sub_y = sub_y[sort_idx]
    density = density[sort_idx]

    sc = ax.scatter(
        sub_x,
        sub_y,
        c=density,
        cmap=colormap,
        s=3.0,
        alpha=0.7,
        edgecolors="none",
        rasterized=True,
    )

    ax.set_xlabel(x_label or x_channel, fontsize=8.5, fontweight="medium")
    ax.set_ylabel(y_label or y_channel, fontsize=8.5, fontweight="medium")
    if title:
        ax.set_title(format_multiline_title(title, max_chars=42), fontsize=9.0, fontweight="bold", pad=6)

    apply_nature_axis_style(ax)

    # Set axis limits if provided, or guard against extreme digital pulse overflow spikes
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        p99_x = float(np.nanpercentile(x_vals, 99.9))
        max_x = float(np.nanmax(x_vals))
        if max_x > 3.0 * p99_x and p99_x > 0:
            ax.set_xlim(-0.02 * p99_x, 1.05 * p99_x)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        p99_y = float(np.nanpercentile(y_vals, 99.9))
        max_y = float(np.nanmax(y_vals))
        if max_y > 3.0 * p99_y and p99_y > 0:
            ax.set_ylim(-0.02 * p99_y, 1.05 * p99_y)

    # Add count text in corner
    ax.text(
        0.03,
        0.95,
        f"Events: {len(df):,}",
        transform=ax.transAxes,
        fontsize=7.0,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#cccccc", linewidth=0.5),
    )

    if standalone:
        plt.tight_layout()
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    return fig, ax



def plot_histogram_1d(
    df: pd.DataFrame,
    channel: str,
    channel_label: Optional[str] = None,
    title: Optional[str] = None,
    bins: int = 100,
    color: str = "#2b5c8f",
    cutoff: Optional[float] = None,
    cutoff_label: Optional[str] = "Cutoff",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
    square_plot_area: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generate a 1D event distribution histogram for flow cytometry data.

    Ensures a standardized 1:1 square plot area by default across samples and channels,
    scaled compactly for publication subplots unless the caller explicitly specifies a custom figsize.

    Args:
        df: Event DataFrame.
        channel: Column name for channel to histogram.
        channel_label: Custom label for x-axis.
        title: Title of plot.
        bins: Number of histogram bins.
        color: Histogram bar fill color.
        cutoff: Optional vertical cutoff line (e.g. from gating or adaptive threshold).
        cutoff_label: Label for the cutoff line.
        output_path: Optional file path to export image.
        figsize: Figure width and height in inches. If None, defaults to compact publication subplot square (3.6, 3.6).
        dpi: Resolution for saved image.
        ax: Existing axes to draw onto.
        square_plot_area: If True and figsize not explicitly customized to a rectangle,
            enforces a 1:1 square plot box.

    Returns:
        Tuple of (fig, ax).
    """
    custom_figsize = figsize is not None
    standalone = ax is None
    if standalone:
        actual_figsize = figsize if custom_figsize else DEFAULT_FIGSIZE_SINGLE
        fig, ax = plt.subplots(figsize=actual_figsize, dpi=dpi)
    else:
        fig = ax.figure

    # Enforce square plot area unless caller specified a custom non-square figsize for a standalone plot
    if square_plot_area:
        if not standalone or not custom_figsize or (figsize and figsize[0] == figsize[1]):
            ax.set_box_aspect(1)

    vals = df[channel].to_numpy()
    finite_vals = vals[np.isfinite(vals)]

    counts, bins_out, _ = ax.hist(
        finite_vals,
        bins=bins,
        color=color,
        alpha=0.75,
        edgecolor="none",
        density=False,
    )

    if cutoff is not None:
        ax.axvline(
            cutoff,
            color="#d62728",
            linewidth=1.4,
            linestyle="--",
            label=f"{cutoff_label}: {cutoff:.2f}",
        )
        ax.legend(loc="upper right", framealpha=0.85, fontsize=7.5)

    ax.set_xlabel(channel_label or channel, fontsize=8.5, fontweight="medium")
    ax.set_ylabel("Event Count", fontsize=8.5, fontweight="medium")
    if title:
        ax.set_title(format_multiline_title(title, max_chars=42), fontsize=9.0, fontweight="bold", pad=6)

    apply_nature_axis_style(ax)

    ax.text(
        0.03,
        0.95,
        f"Events: {len(finite_vals):,}",
        transform=ax.transAxes,
        fontsize=7.0,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#cccccc", linewidth=0.5),
    )

    if standalone:
        plt.tight_layout()
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    return fig, ax



def plot_gate_overlay(
    df: pd.DataFrame,
    gate: Gate,
    parent_mask: Optional[np.ndarray] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    gate_color: str = "#d62728",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
    square_plot_area: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot parent event scatter with gate boundary overlay and statistics.

    Ensures a standardized 1:1 square plot area by default across samples and channels,
    scaled compactly for publication subplots unless the caller explicitly specifies a custom figsize.

    Args:
        df: DataFrame containing all events.
        gate: Gate instance to visualize.
        parent_mask: Boolean mask indicating which events belong to parent population.
        x_label: Label for x-axis.
        y_label: Label for y-axis.
        title: Plot title.
        gate_color: Color of the gate boundary line.
        output_path: Optional path to save figure.
        figsize: Figure dimensions. If None, defaults to compact publication subplot square (3.6, 3.6).
        dpi: Image DPI.
        ax: Optional matplotlib axes.
        square_plot_area: If True and figsize not explicitly customized to a rectangle,
            enforces a 1:1 square plot box.

    Returns:
        Tuple of (fig, ax).
    """
    custom_figsize = figsize is not None
    standalone = ax is None
    if standalone:
        actual_figsize = figsize if custom_figsize else DEFAULT_FIGSIZE_SINGLE
        fig, ax = plt.subplots(figsize=actual_figsize, dpi=dpi)
    else:
        fig = ax.figure

    # Enforce square plot area unless caller specified a custom non-square figsize for a standalone plot
    if square_plot_area:
        if not standalone or not custom_figsize or (figsize and figsize[0] == figsize[1]):
            ax.set_box_aspect(1)

    parent_df = df if parent_mask is None else df.loc[parent_mask]
    parent_count = len(parent_df)

    # Compute gate mask on parent population
    local_mask = gate.apply(parent_df)
    gate_count = int(np.sum(local_mask))
    pct = (gate_count / parent_count * 100.0) if parent_count > 0 else 0.0

    is_1d_gate = isinstance(gate, RangeGate) or len(gate.channels) == 1

    if is_1d_gate:
        x_ch = gate.channels[0]
        vals = parent_df[x_ch].to_numpy()
        finite_vals = vals[np.isfinite(vals)]

        # Render 1D event distribution histogram of parent population
        counts, bins, _ = ax.hist(
            finite_vals,
            bins=100,
            color="#2b5c8f",
            alpha=0.75,
            edgecolor="none",
            density=False,
        )
        max_count = float(np.max(counts)) if len(counts) > 0 else 1.0

        min_val = getattr(gate, "min_val", None)
        max_val = getattr(gate, "max_val", None)

        cur_xlim = ax.get_xlim()
        shade_x0 = min_val if min_val is not None else cur_xlim[0]
        shade_x1 = max_val if max_val is not None else cur_xlim[1]

        ax.axvspan(shade_x0, shade_x1, color=gate_color, alpha=0.18, label="Gated Region")
        if min_val is not None:
            ax.axvline(min_val, color=gate_color, linewidth=1.4, linestyle="--")
        if max_val is not None:
            ax.axvline(max_val, color=gate_color, linewidth=1.4, linestyle="--")

        ax.set_ylabel(y_label or "Event Count", fontsize=8.5, fontweight="medium")
        ax.set_xlabel(x_label or x_ch, fontsize=8.5, fontweight="medium")

        # Position label inside plot boundaries safely
        span_mid = (shade_x0 + shade_x1) / 2.0
        label_x = float(np.clip(span_mid, cur_xlim[0] + 0.2 * (cur_xlim[1] - cur_xlim[0]), cur_xlim[1] - 0.2 * (cur_xlim[1] - cur_xlim[0])))
        label_y = max_count * 0.78

    else:
        x_ch = gate.channels[0]
        y_ch = gate.channels[1]

        # Draw 2D density-colored scatter plot of parent population
        sub_x, sub_y, density = calculate_density_colors(
            parent_df[x_ch].to_numpy(),
            parent_df[y_ch].to_numpy(),
        )
        sort_idx = np.argsort(density)
        ax.scatter(
            sub_x[sort_idx],
            sub_y[sort_idx],
            c=density[sort_idx],
            cmap="turbo",
            s=3.0,
            alpha=0.7,
            edgecolors="none",
            rasterized=True,
        )

        # Overlay gate geometry
        if isinstance(gate, PolygonGate):
            poly_patch = patches.Polygon(
                gate.vertices,
                closed=True,
                fill=True,
                facecolor=gate_color,
                alpha=0.12,
                edgecolor=gate_color,
                linewidth=1.4,
                linestyle="-",
            )
            ax.add_patch(poly_patch)
            centroid = np.mean(gate.vertices, axis=0)
            label_x, label_y = float(centroid[0]), float(centroid[1])

        elif isinstance(gate, RectangleGate):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x0 = gate.x_min if gate.x_min is not None else xlim[0]
            x1 = gate.x_max if gate.x_max is not None else xlim[1]
            y0 = gate.y_min if gate.y_min is not None else ylim[0]
            y1 = gate.y_max if gate.y_max is not None else ylim[1]

            rect_patch = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=True,
                facecolor=gate_color,
                alpha=0.12,
                edgecolor=gate_color,
                linewidth=1.4,
                linestyle="-",
            )
            ax.add_patch(rect_patch)
            label_x = (x0 + x1) / 2.0
            label_y = (y0 + y1) / 2.0

        else:
            label_x, label_y = 0.5, 0.5

        ax.set_xlabel(x_label or x_ch, fontsize=8.5, fontweight="medium")
        ax.set_ylabel(y_label or y_ch, fontsize=8.5, fontweight="medium")

        # Constrain 2D scatter axes if linear scatter channels have extreme electronic spikes
        if len(parent_df) > 0:
            if x_ch.lower().startswith(("fsc", "ssc")):
                p99_x = float(np.nanpercentile(parent_df[x_ch], 99.9))
                if float(np.nanmax(parent_df[x_ch])) > 3.0 * p99_x and p99_x > 0:
                    ax.set_xlim(-0.02 * p99_x, 1.05 * p99_x)
            if y_ch.lower().startswith(("fsc", "ssc")):
                p99_y = float(np.nanpercentile(parent_df[y_ch], 99.9))
                if float(np.nanmax(parent_df[y_ch])) > 3.0 * p99_y and p99_y > 0:
                    ax.set_ylim(-0.02 * p99_y, 1.05 * p99_y)

    # Annotate Gate statistic
    stat_text = f"{gate.name}\n{gate_count:,} ({pct:.1f}%)"
    ax.annotate(
        stat_text,
        xy=(label_x, label_y),
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="#111111",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.9,
            edgecolor=gate_color,
            linewidth=1.0,
        ),
    )

    default_title = f"{gate.name} Gate on {gate.parent_gate or 'Root'}"
    ax.set_title(format_multiline_title(title or default_title, max_chars=42), fontsize=9.0, fontweight="bold", pad=6)
    apply_nature_axis_style(ax)

    if standalone:
        plt.tight_layout()
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    return fig, ax


def plot_compensation_comparison(
    uncomp_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    x_channel: str,
    y_channel: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    transform: Optional[Any] = None,
    square_plot_area: bool = True,
) -> plt.Figure:
    """Plot side-by-side comparison of uncompensated vs compensated fluorescence.

    Automatically applies Logicle transformation if data is in raw linear scale,
    ensuring populations and negative spillover corrections are clearly visible across decades.
    Ensures standardized 1:1 square plot areas by default for both panels,
    unless the caller explicitly specifies a custom figsize.

    Args:
        uncomp_df: DataFrame of uncompensated raw events.
        comp_df: DataFrame of compensated events.
        x_channel: First fluorescent detector.
        y_channel: Second fluorescent detector.
        x_label: X-axis label.
        y_label: Y-axis label.
        output_path: Optional output file path.
        figsize: Figure size. If None, defaults to publication dual-column (7.2, 3.6) with square panels.
        dpi: Resolution.
        transform: Optional transformation object. If None and max value > 10.0,
            LogicleTransform is automatically applied for diagnostic clarity.
        square_plot_area: If True and figsize not explicitly customized, enforces 1:1 square panels.

    Returns:
        Matplotlib Figure object.
    """
    plot_uncomp = uncomp_df[[x_channel, y_channel]].copy()
    plot_comp = comp_df[[x_channel, y_channel]].copy()

    # If raw linear (> 10.0), apply Logicle transform so compensation effects are visible
    is_raw = float(np.nanmax(plot_uncomp[x_channel])) > 10.0 or float(np.nanmax(plot_uncomp[y_channel])) > 10.0
    scale_note = ""

    if transform is not None:
        plot_uncomp[x_channel] = transform.apply(plot_uncomp[x_channel].to_numpy())
        plot_uncomp[y_channel] = transform.apply(plot_uncomp[y_channel].to_numpy())
        plot_comp[x_channel] = transform.apply(plot_comp[x_channel].to_numpy())
        plot_comp[y_channel] = transform.apply(plot_comp[y_channel].to_numpy())
        scale_note = " (Logicle Scale)"
    elif is_raw:
        from flowx.math.transforms import LogicleTransform
        xform = LogicleTransform()
        plot_uncomp[x_channel] = xform.apply(plot_uncomp[x_channel].to_numpy())
        plot_uncomp[y_channel] = xform.apply(plot_uncomp[y_channel].to_numpy())
        plot_comp[x_channel] = xform.apply(plot_comp[x_channel].to_numpy())
        plot_comp[y_channel] = xform.apply(plot_comp[y_channel].to_numpy())
        scale_note = " (Logicle Scale)"

    custom_figsize = figsize is not None
    actual_figsize = figsize if custom_figsize else DEFAULT_FIGSIZE_DUAL

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=actual_figsize, dpi=dpi)
    if square_plot_area:
        ax1.set_box_aspect(1)
        ax2.set_box_aspect(1)

    xlbl = (x_label or x_channel) + scale_note
    ylbl = (y_label or y_channel) + scale_note

    plot_scatter_2d(
        plot_uncomp,
        x_channel=x_channel,
        y_channel=y_channel,
        x_label=xlbl,
        y_label=ylbl,
        title=f"Uncompensated: {x_label or x_channel} vs {y_label or y_channel}",
        ax=ax1,
        figsize=None,
        square_plot_area=square_plot_area,
    )

    plot_scatter_2d(
        plot_comp,
        x_channel=x_channel,
        y_channel=y_channel,
        x_label=xlbl,
        y_label=ylbl,
        title=format_multiline_title(f"Compensated: {x_label or x_channel} vs {y_label or y_channel}", max_chars=38),
        ax=ax2,
        figsize=None,
        square_plot_area=square_plot_area,
    )

    fig.suptitle(format_multiline_title("Fluorescence Compensation Diagnostic", max_chars=60), fontsize=10.0, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_transformation_comparison(
    raw_vals: np.ndarray,
    transformed_vals: np.ndarray,
    channel_name: str,
    transform_name: str = "Logicle",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    square_plot_area: bool = True,
) -> plt.Figure:
    """Plot side-by-side histograms comparing raw vs transformed scale.

    Ensures standardized 1:1 square plot areas by default for both histogram panels,
    unless the caller explicitly specifies a custom figsize.

    Args:
        raw_vals: 1D array of untransformed values.
        transformed_vals: 1D array of transformed values.
        channel_name: Channel identifier.
        transform_name: Name of transformation (e.g. 'Logicle').
        output_path: Optional export path.
        figsize: Dimensions. If None, defaults to publication dual-column (7.2, 3.6) with square panels.
        dpi: Resolution.
        square_plot_area: If True and figsize not explicitly customized, enforces 1:1 square panels.

    Returns:
        Matplotlib Figure.
    """
    custom_figsize = figsize is not None
    actual_figsize = figsize if custom_figsize else DEFAULT_FIGSIZE_DUAL

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=actual_figsize, dpi=dpi)

    if square_plot_area:
        ax1.set_box_aspect(1)
        ax2.set_box_aspect(1)

    # Raw distribution
    clean_raw = raw_vals[np.isfinite(raw_vals)]
    p1, p99 = np.percentile(clean_raw, [1, 99])
    ax1.hist(clean_raw, bins=120, range=(p1, p99), color="#1f77b4", alpha=0.8, edgecolor="none")
    ax1.set_title(format_multiline_title(f"Raw Linear Scale: {channel_name}", max_chars=38), fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("Raw Intensity (a.u.)", fontsize=8.0)
    ax1.set_ylabel("Count", fontsize=8.0)
    apply_nature_axis_style(ax1)

    # Transformed distribution
    clean_xform = transformed_vals[np.isfinite(transformed_vals)]
    ax2.hist(clean_xform, bins=120, color="#2ca02c", alpha=0.8, edgecolor="none")
    ax2.set_title(format_multiline_title(f"{transform_name} Scale: {channel_name}", max_chars=38), fontsize=8.5, fontweight="bold")
    ax2.set_xlabel(f"{transform_name} Value", fontsize=8.0)
    ax2.set_ylabel("Count", fontsize=8.0)
    apply_nature_axis_style(ax2)

    fig.suptitle(format_multiline_title(f"Scale Transformation Comparison: {channel_name}", max_chars=65), fontsize=10.0, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig



def plot_pipeline_cascade(
    df: pd.DataFrame,
    gates: List[Gate],
    hierarchy_masks: Dict[str, np.ndarray],
    channel_labels: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    square_plot_area: bool = True,
) -> plt.Figure:
    """Plot an end-to-end multi-panel gating cascade summarizing all hierarchy stages.

    Ensures standardized 1:1 square plot areas by default for each cascade panel,
    unless the caller explicitly specifies a custom figsize.

    Args:
        df: Processed DataFrame.
        gates: Ordered list of Gate objects.
        hierarchy_masks: Precomputed boolean masks from GatingHierarchy.evaluate().
        channel_labels: Mapping from channel name to human-readable label.
        output_path: Path to export summary image.
        figsize: Figure dimensions. If None, defaults to (3.6 * n_gates, 3.6) with square panels.
        dpi: Resolution.
        square_plot_area: If True and figsize not explicitly customized, enforces 1:1 square panels.

    Returns:
        Matplotlib Figure object.
    """
    n_gates = len(gates)
    custom_figsize = figsize is not None
    actual_figsize = figsize if custom_figsize else (3.6 * max(n_gates, 1), 3.6)

    fig, axes = plt.subplots(1, n_gates, figsize=actual_figsize, dpi=dpi)
    if n_gates == 1:
        axes = [axes]

    if square_plot_area:
        for ax in axes:
            ax.set_box_aspect(1)

    labels = channel_labels or {}

    for i, gate in enumerate(gates):
        ax = axes[i]
        parent_name = gate.parent_gate if gate.parent_gate else "root"
        parent_mask = hierarchy_masks.get(parent_name, np.ones(len(df), dtype=bool))

        x_ch = gate.channels[0]
        y_ch = gate.channels[1] if len(gate.channels) > 1 else gate.channels[0]

        x_lbl = labels.get(x_ch, x_ch)
        y_lbl = labels.get(y_ch, y_ch)

        plot_gate_overlay(
            df=df,
            gate=gate,
            parent_mask=parent_mask,
            x_label=x_lbl,
            y_label=y_lbl,
            title=f"Step {i+1}: {gate.name}",
            ax=ax,
            figsize=None,
            square_plot_area=square_plot_area,
        )

    fig.suptitle(format_multiline_title("Sequential Flow Cytometry Gating Pipeline", max_chars=60), fontsize=10.5, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_clusters_scatter_2d(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    x_channel: str,
    y_channel: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    max_points: int = 30000,
    random_state: int = 42,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
    legend_mode: str = "centroid",
    square_plot_area: bool = True,
    match_cluster_color: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 2D scatter colored by unsupervised cluster assignments.

    Ensures standardized 1:1 square plot areas by default,
    scaled compactly for publication subplots unless the caller explicitly specifies a custom figsize.

    Args:
        df: Event DataFrame.
        cluster_labels: 1D array of cluster integer labels.
        x_channel: Channel for x-axis.
        y_channel: Channel for y-axis.
        x_label: X-axis label.
        y_label: Y-axis label.
        title: Title of plot.
        max_points: Subsample limit for responsive plotting.
        random_state: Seed.
        output_path: File export path.
        figsize: Figure dimensions. Defaults to compact publication square (3.6, 3.6) or (5.0, 3.6) with external legend.
        dpi: Resolution.
        ax: Existing axes.
        legend_mode: 'centroid' (default, numbered badges at cluster centers),
                     'external' (multi-column legend outside plot), or 'none'.
        square_plot_area: If True and figsize not explicitly customized to a rectangle,
            enforces a 1:1 square plot box.
        match_cluster_color: If True, renders the cluster badge number in the matching cluster color.

    Returns:
        Tuple of (fig, ax).
    """
    custom_figsize = figsize is not None
    standalone = ax is None
    if standalone:
        actual_figsize = figsize if custom_figsize else (DEFAULT_FIGSIZE_WIDE if legend_mode == "external" else DEFAULT_FIGSIZE_SINGLE)
        fig, ax = plt.subplots(figsize=actual_figsize, dpi=dpi)
    else:
        fig = ax.figure

    # Enforce square plot area unless caller specified a custom non-square figsize for a standalone plot
    if square_plot_area:
        if not standalone or not custom_figsize or (figsize and figsize[0] == figsize[1]):
            ax.set_box_aspect(1)


    n = len(df)
    if n > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        sub_x = df[x_channel].iloc[idx].to_numpy()
        sub_y = df[y_channel].iloc[idx].to_numpy()
        sub_c = cluster_labels[idx]
    else:
        sub_x = df[x_channel].to_numpy()
        sub_y = df[y_channel].to_numpy()
        sub_c = cluster_labels

    unique_clusters = np.unique(sub_c)
    cmap = plt.get_cmap("tab20" if len(unique_clusters) > 10 else "tab10", len(unique_clusters))

    for i, cid in enumerate(unique_clusters):
        c_mask = sub_c == cid
        color = cmap(i)
        ax.scatter(
            sub_x[c_mask],
            sub_y[c_mask],
            c=[color],
            s=3.0,
            alpha=0.65,
            edgecolors="none",
            label=f"C{cid} ({np.sum(sub_c == cid):,})",
            rasterized=True,
        )

        # Direct Centroid Badging
        if legend_mode == "centroid" and np.sum(c_mask) > 0:
            med_x = float(np.nanmedian(sub_x[c_mask]))
            med_y = float(np.nanmedian(sub_y[c_mask]))
            text_color = color if match_cluster_color else "black"
            ax.text(
                med_x,
                med_y,
                str(cid),
                fontsize=6.5,
                fontweight="bold",
                ha="center",
                va="center",
                color=text_color,
                bbox=dict(
                    boxstyle="circle,pad=0.18",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.2,
                    alpha=0.92,
                ),
                zorder=10,
            )

    ax.set_xlabel(x_label or x_channel, fontsize=8.5, fontweight="medium")
    ax.set_ylabel(y_label or y_channel, fontsize=8.5, fontweight="medium")
    if title:
        badge_note = " (Centroid Badges: Cluster IDs)" if legend_mode == "centroid" else ""
        ax.set_title(format_multiline_title(f"{title}{badge_note}", max_chars=42), fontsize=8.5, fontweight="bold", pad=6)

    apply_nature_axis_style(ax)

    if legend_mode == "external":
        ncols = 2 if len(unique_clusters) > 12 else 1
        ax.legend(
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            ncols=ncols,
            markerscale=2.0,
            fontsize=6.5,
            framealpha=0.88,
        )
    elif legend_mode == "centroid":
        pass  # Centroid badges replace legend box without obscuring cells

    if standalone:
        plt.tight_layout()
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    return fig, ax


def get_cluster_palette(
    labels: Union[List[Any], np.ndarray, pd.Series],
    cmap_name: str = "tab20",
) -> Dict[str, Tuple[float, float, float, float]]:
    """Generate consistent RGBA colors for clusters/populations across all discovery plots.

    Args:
        labels: Iterable of unique cluster IDs or biological population labels.
        cmap_name: Colormap or Seaborn palette name (default: 'tab20').

    Returns:
        Dictionary mapping string label to RGBA color tuple.
    """
    clean_labels = [str(x) for x in labels]
    # Preserve deterministic ordering (sorting numeric if all digits)
    unique_items = list(dict.fromkeys(clean_labels))
    if all(x.isdigit() for x in unique_items):
        unique_items = sorted(unique_items, key=int)
    elif all(x.startswith("C") and x[1:].isdigit() for x in unique_items):
        unique_items = sorted(unique_items, key=lambda x: int(x[1:]))
    elif all(x.startswith("Cluster ") and x[8:].isdigit() for x in unique_items):
        unique_items = sorted(unique_items, key=lambda x: int(x[8:]))
    else:
        unique_items = sorted(unique_items)

    n = max(1, len(unique_items))
    seaborn_palettes = {
        "deep", "muted", "pastel", "bright", "dark", "colorblind",
        "husl", "hls", "tab10", "tab20", "set1", "set2", "set3",
    }
    if cmap_name.lower() == "tab20" and n > 20:
        # Extend tab20 with tab20b and tab20c so >20 clusters/populations never repeat colors
        base_colors = (
            list(sns.color_palette("tab20", 20))
            + list(sns.color_palette("tab20b", 20))
            + list(sns.color_palette("tab20c", 20))
        )
        if n > len(base_colors):
            base_colors = base_colors + list(sns.color_palette("husl", n - len(base_colors)))
        pal_colors = base_colors[:n]
        return {lbl: (float(c[0]), float(c[1]), float(c[2]), 1.0) for lbl, c in zip(unique_items, pal_colors)}
    if cmap_name.lower() in seaborn_palettes:
        pal_colors = sns.color_palette(cmap_name.lower(), n)
        return {lbl: (float(c[0]), float(c[1]), float(c[2]), 1.0) for lbl, c in zip(unique_items, pal_colors)}

    try:
        cmap = plt.get_cmap(cmap_name)
        if n == 1:
            points = [0.5]
        elif n == 2:
            points = [0.15, 0.85]
        else:
            points = np.linspace(0.08, 0.92, n)
        return {lbl: cmap(val) for lbl, val in zip(unique_items, points)}
    except ValueError:
        pal_colors = sns.color_palette(cmap_name, n)
        return {lbl: (float(c[0]), float(c[1]), float(c[2]), 1.0) for lbl, c in zip(unique_items, pal_colors)}


def plot_cluster_heatmap(
    cluster_stats_df: pd.DataFrame,
    marker_cols: List[str],
    title: str = "Cluster Median Marker Expression",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    ordered_indices: Optional[List[int]] = None,
    cluster_to_macro_group: Optional[Dict[int, str]] = None,
    cluster_to_label: Optional[Dict[int, str]] = None,
    marker_name_map: Optional[Dict[str, str]] = None,
    cmap: str = "YlGnBu",
    cluster_palette: str = "tab20",
    group_by_population: bool = False,
) -> plt.Figure:
    """Plot an organized heatmap of median marker intensities with biological lineage grouping.

    Uses Seaborn clustermap architecture for unified styling with cohort diagnostic heatmaps,
    featuring square cells, mathematically synchronized dendrograms, seamless cluster color badges,
    and a discrete population legend on the right.

    Args:
        cluster_stats_df: DataFrame output from ClusteringResult.cluster_statistics.
        marker_cols: List of marker column names.
        title: Title of heatmap.
        output_path: File path to save figure.
        figsize: Optional explicit figure size in inches.
        dpi: Resolution (default: 300).
        ordered_indices: Optional dendrogram leaf ordering to group related clusters.
        cluster_to_macro_group: Optional dictionary mapping cluster ID to biological lineage.
        cluster_to_label: Optional dictionary mapping cluster ID to detailed phenotype label.
        marker_name_map: Optional mapping from detector name to stain/marker name (e.g. {'PE-TxRed-A': 'CD4'}).
        cmap: Colormap for expression intensity (default: 'YlGnBu').
        cluster_palette: Colormap for cluster identification badges (default: 'tab20').

    Returns:
        Matplotlib Figure object.
    """
    set_nature_style()

    median_cols = [c for c in cluster_stats_df.columns if "Median" in c]
    if not median_cols:
        median_cols = [c for c in marker_cols if c in cluster_stats_df.columns]

    # Reorder clusters by biological similarity if ordered_indices is provided
    if ordered_indices and len(ordered_indices) == len(cluster_stats_df):
        plot_df = cluster_stats_df.iloc[ordered_indices].copy().reset_index(drop=True)
    else:
        plot_df = cluster_stats_df.copy()

    col_labels: List[str] = []
    for col in median_cols:
        c = col.replace(" Median", "")
        marker = marker_name_map.get(c, "") if marker_name_map else ""
        if marker and marker.strip():
            marker_str = marker.strip()
            if f"({c})" in marker_str:
                col_labels.append(marker_str)
            else:
                col_labels.append(f"{marker_str} ({c})")
        else:
            col_labels.append(c)

    # Build concise row labels and resolve biological population names for legend
    cluster_labels: List[str] = []
    row_keys: List[Union[int, str]] = []
    pop_legend_items: List[Tuple[Union[int, str], str]] = []

    if group_by_population:
        pop_groups: Dict[str, List[pd.Series]] = {}
        for idx, row in plot_df.iterrows():
            cid = int(row["Cluster ID"]) if "Cluster ID" in row else int(idx)
            label_text = None
            if cluster_to_label and cid in cluster_to_label:
                label_text = cluster_to_label[cid]
            elif cluster_to_macro_group and cid in cluster_to_macro_group:
                label_text = cluster_to_macro_group[cid]
            if not label_text or label_text == f"Cluster_{cid}":
                label_text = f"Cluster {cid}"
            clean_pop = re.sub(r"^C\d+\s*[:\-]\s*", "", str(label_text)).strip() or str(label_text)
            pop_groups.setdefault(clean_pop, []).append(row)

        agg_rows: List[Dict[str, Any]] = []
        for pop_name, rows in pop_groups.items():
            counts = np.array(
                [max(1.0, float(r.get("Event Count", r.get("Count", 1.0)))) for r in rows],
                dtype=float,
            )
            pct_vals = [
                float(r.get("% Total", r.get("% Parent", r.get("Percentage", 0.0))))
                for r in rows
                if r.get("% Total", r.get("% Parent", r.get("Percentage", None))) is not None
            ]
            total_pct = round(float(np.sum(pct_vals)), 1) if pct_vals else None
            entry: Dict[str, Any] = {"Population": pop_name, "% Total": total_pct}
            for mc in median_cols:
                vals = np.array([float(r[mc]) for r in rows], dtype=float)
                entry[mc] = float(np.average(vals, weights=counts))
            agg_rows.append(entry)

        plot_df = pd.DataFrame(agg_rows)
        for idx, row in plot_df.iterrows():
            pop_name = str(row["Population"])
            row_keys.append(pop_name)
            pct_val = row.get("% Total", None)
            pct_str = f" ({pct_val:.1f}%)" if pct_val is not None else ""
            short_row = pop_name if len(pop_name) <= 26 else (pop_name[:24].rstrip() + "..")
            cluster_labels.append(f"{short_row}{pct_str}")
            pop_legend_items.append((pop_name, f"{pop_name}{pct_str}"))
    else:
        for idx, row in plot_df.iterrows():
            cid = int(row["Cluster ID"]) if "Cluster ID" in row else int(idx)
            row_keys.append(cid)
            pct_val = row.get("% Total", row.get("% Parent", None))
            pct_str = f" ({pct_val}%)" if pct_val is not None else ""
            cluster_labels.append(f"C{cid}{pct_str}")

            label_text = None
            if cluster_to_label and cid in cluster_to_label:
                label_text = cluster_to_label[cid]
            elif cluster_to_macro_group and cid in cluster_to_macro_group:
                label_text = cluster_to_macro_group[cid]

            if not label_text or label_text == f"Cluster_{cid}":
                label_text = f"Cluster {cid}"
            pop_legend_items.append((cid, label_text))

    data_mat = pd.DataFrame(
        plot_df[median_cols].to_numpy(dtype=float),
        index=cluster_labels,
        columns=col_labels,
    )

    n_clusters, n_markers = data_mat.shape

    # Consistent cluster/population color mapping matching UMAP
    all_keys = sorted(list(set(row_keys)), key=lambda x: str(x)) if not group_by_population else list(dict.fromkeys(row_keys))
    cid_color_map = get_cluster_palette(all_keys, cmap_name=cluster_palette)
    row_color_list = [cid_color_map[str(k)] for k in row_keys]
    row_colors = pd.Series(row_color_list, index=data_mat.index)

    do_row_cluster = n_clusters >= 2
    do_col_cluster = n_markers >= 2

    # Dynamic sizing analytically enforcing square matrix cells
    s = 0.35  # Cell size in inches
    dendro_w = 0.12 if do_row_cluster else 0.02
    dendro_h = 0.12 if do_col_cluster else 0.02
    color_w = 0.04

    init_w = max(6.5, (n_markers * s) + 4.8)
    init_h = max(5.0, (n_clusters * s) + 2.6)
    total_w, total_h = figsize or (init_w, init_h)

    cg = sns.clustermap(
        data_mat,
        row_colors=row_colors,
        row_cluster=do_row_cluster,
        col_cluster=do_col_cluster,
        cmap=cmap,
        figsize=(total_w, total_h),
        linewidths=0.4,
        linecolor="white",
        dendrogram_ratio=(dendro_w, dendro_h),
        colors_ratio=color_w,
        cbar_pos=None,
    )

    # Remove white grid lines from the row annotation color strip
    if cg.ax_row_colors is not None and cg.ax_row_colors.collections:
        cg.ax_row_colors.collections[0].set_edgecolor("none")
        cg.ax_row_colors.collections[0].set_linewidth(0)

    # Exact aspect ratio control: enforce true square cells across any dimension
    w_ratios = cg.gs.get_width_ratios()
    h_ratios = cg.gs.get_height_ratios()
    f_w = w_ratios[-1] / sum(w_ratios)
    f_h = h_ratios[-1] / sum(h_ratios)

    gs_w_in = (n_markers * s) / f_w
    gs_h_in = (n_clusters * s) / f_h

    # Calculate dynamic margins based on marker label lengths and legend width
    max_col_len = max((len(c) for c in col_labels), default=10)
    rot_rad = math.radians(45)
    label_drop_in = max(0.9, 0.052 * max_col_len * math.sin(rot_rad) + 0.35)
    margin_left_in = 2.0
    margin_bottom_in = label_drop_in + 0.35 + 0.25 + 0.40 + 0.20
    margin_top_in = 0.8
    margin_right_in = 2.8

    if figsize is None:
        total_w = max(6.8, gs_w_in + margin_left_in + margin_right_in)
        total_h = max(4.6, gs_h_in + margin_bottom_in + margin_top_in)
        cg.fig.set_size_inches(total_w, total_h)

    left = margin_left_in / total_w
    right = left + (gs_w_in / total_w)
    bottom = margin_bottom_in / total_h
    top = bottom + (gs_h_in / total_h)
    cg.fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    # Label the row_colors strip ('Populations') alongside the marker x-labels and add clean separation gap
    def _apply_row_colors_gap(cur_w: float) -> None:
        if cg.ax_row_colors is not None:
            cg.ax_row_colors.set_xticks([0.5])
            cg.ax_row_colors.set_xticklabels(
                ["Populations"],
                rotation=45,
                ha="right",
                fontsize=7.5,
                fontweight="bold",
            )
            cg.ax_row_colors.tick_params(axis="x", which="both", length=2.5, width=0.6, bottom=True)
            rc_pos = cg.ax_row_colors.get_position()
            gap_rc = 0.08 / cur_w
            cg.ax_row_colors.set_position([rc_pos.x0 - gap_rc, rc_pos.y0, rc_pos.width, rc_pos.height])
            if do_row_cluster and hasattr(cg, "ax_row_dendrogram") and cg.ax_row_dendrogram is not None:
                rd_pos = cg.ax_row_dendrogram.get_position()
                cg.ax_row_dendrogram.set_position([rd_pos.x0 - gap_rc, rd_pos.y0, rd_pos.width, rd_pos.height])

    _apply_row_colors_gap(total_w)

    # Inner thin white grid lines separating each cell on boundary edges
    cg.ax_heatmap.set_aspect("equal")
    for r in range(1, n_clusters):
        cg.ax_heatmap.axhline(r, color="white", linewidth=0.4, alpha=0.95)
    for c in range(1, n_markers):
        cg.ax_heatmap.axvline(c, color="white", linewidth=0.4, alpha=0.95)

    # Format ticks and labels
    cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=7.5, fontweight="medium")
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7.0)
    cg.ax_heatmap.set_xlabel("Lineage & Functional Markers", fontsize=7.8, fontweight="bold", labelpad=8)
    cg.ax_heatmap.set_ylabel(
        "Identified Populations" if group_by_population else "Discovered Clusters",
        fontsize=7.8,
        fontweight="bold",
    )

    # Outer crisp black border surrounding the overall heatmap matrix box
    border_lw = round(max(0.55, min(0.85, 0.50 + 0.01 * max(n_clusters, n_markers))), 2)
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

    # Crisp border around the row color strip
    if cg.ax_row_colors is not None:
        rc_rect = patches.Rectangle(
            (0, 0), 1, 1,
            fill=False,
            edgecolor="black",
            linewidth=border_lw,
            transform=cg.ax_row_colors.transAxes,
            clip_on=False,
            zorder=100,
        )
        cg.ax_row_colors.add_patch(rc_rect)

    # Refine dendrogram tree line weights for a compact, clean look
    if do_row_cluster and hasattr(cg, "ax_row_dendrogram") and cg.ax_row_dendrogram:
        for coll in cg.ax_row_dendrogram.collections:
            coll.set_linewidth(0.8)
    if do_col_cluster and hasattr(cg, "ax_col_dendrogram") and cg.ax_col_dendrogram:
        for coll in cg.ax_col_dendrogram.collections:
            coll.set_linewidth(0.8)

    # Center title directly over the heatmap matrix
    h_pos = cg.ax_heatmap.get_position()
    center_heat_x = h_pos.x0 + 0.5 * h_pos.width
    cg.fig.suptitle(
        format_multiline_title(f"{title} (Hierarchically Clustered)", max_chars=55),
        fontsize=9.0,
        fontweight="bold",
        x=center_heat_x,
        ha="center",
        y=top + (0.35 / total_h),
    )

    # Discrete legend on right side matching cluster/population colors
    cid_to_name = {cid: name for cid, name in pop_legend_items}
    legend_handles = []
    if group_by_population:
        unique_legend_keys = list(dict.fromkeys(cid for cid, _ in pop_legend_items))
        for pop_key in unique_legend_keys:
            raw_lbl = str(cid_to_name[pop_key]).strip()
            legend_handles.append(
                patches.Patch(
                    facecolor=cid_color_map[str(pop_key)],
                    edgecolor="none",
                    label=format_multiline_legend_label(raw_lbl, max_chars_per_line=24),
                )
            )
    else:
        unique_legend_cids = sorted(list({int(cid) for cid, _ in pop_legend_items}))
        for cid in unique_legend_cids:
            raw_lbl = str(cid_to_name[cid]).strip()
            prefix = f"C{cid}:"
            if raw_lbl.startswith(prefix):
                raw_lbl = raw_lbl[len(prefix) :].strip()
            legend_handles.append(
                patches.Patch(
                    facecolor=cid_color_map[str(cid)],
                    edgecolor="none",
                    label=f"C{cid}: {format_multiline_legend_label(raw_lbl, max_chars_per_line=22)}",
                )
            )
    leg = cg.ax_heatmap.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        frameon=False,
        title="Populations",
        title_fontsize=7.8,
        fontsize=7.0,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    # Dynamically position legend with guaranteed clearance to the right of y-axis labels
    cg.fig.canvas.draw()
    renderer = cg.fig.canvas.get_renderer()
    y_label_bb = cg.ax_heatmap.yaxis.label.get_window_extent(renderer).transformed(cg.fig.transFigure.inverted())
    h_pos = cg.ax_heatmap.get_position()
    leg_x_target = y_label_bb.x1 + 0.035
    leg.set_bbox_to_anchor((leg_x_target, h_pos.y1), transform=cg.fig.transFigure)

    # Re-draw to assess outer canvas boundary extent
    cg.fig.canvas.draw()
    renderer = cg.fig.canvas.get_renderer()
    h_tight = cg.ax_heatmap.get_tightbbox(renderer).transformed(cg.fig.transFigure.inverted())

    # Dynamically expand canvas width if legend overflows right margin
    if leg is not None:
        leg_bb = leg.get_window_extent(renderer).transformed(cg.fig.transFigure.inverted())
        if leg_bb.x1 > 0.98 and figsize is None:
            overflow_in = (leg_bb.x1 - 0.98) * total_w + 0.35
            total_w += overflow_in
            cg.fig.set_size_inches(total_w, total_h)
            left = margin_left_in / total_w
            right = left + (gs_w_in / total_w)
            cg.fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
            _apply_row_colors_gap(total_w)
            cg.fig.canvas.draw()
            renderer = cg.fig.canvas.get_renderer()
            y_label_bb = cg.ax_heatmap.yaxis.label.get_window_extent(renderer).transformed(cg.fig.transFigure.inverted())
            h_pos = cg.ax_heatmap.get_position()
            leg.set_bbox_to_anchor((y_label_bb.x1 + 0.035, h_pos.y1), transform=cg.fig.transFigure)
            cg.fig.canvas.draw()
            renderer = cg.fig.canvas.get_renderer()
            h_tight = cg.ax_heatmap.get_tightbbox(renderer).transformed(cg.fig.transFigure.inverted())
            h_pos = cg.ax_heatmap.get_position()
            center_heat_x = h_pos.x0 + 0.5 * h_pos.width

    gap_y = 0.22 / total_h
    cbar_top = h_tight.y0 - gap_y
    cbar_h = 0.14 / total_h
    cbar_w = min(0.60 * h_pos.width, 2.4 / total_w)
    cbar_x = center_heat_x - 0.5 * cbar_w
    cbar_y = max(0.08 / total_h, cbar_top - cbar_h)

    cbar_ax = cg.fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    cbar = cg.fig.colorbar(cg.ax_heatmap.collections[0], cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7.0)
    cbar.set_label("Logicle Median Fluorescence Intensity", fontsize=7.5)

    # Ensure fig.axes[0] is always ax_heatmap for backward test compatibility
    heat = cg.ax_heatmap
    if hasattr(cg.fig, "_axstack") and hasattr(cg.fig._axstack, "_axes"):
        cg.fig._axstack._axes = {heat: cg.fig._axstack._axes[heat], **{k: v for k, v in cg.fig._axstack._axes.items() if k is not heat}}

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cg.fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(cg.fig)

    return cg.fig


def plot_population_expression_heatmap(
    cluster_stats_df: pd.DataFrame,
    marker_cols: List[str],
    title: str = "Identified Population Marker Expression",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    cluster_to_macro_group: Optional[Dict[int, str]] = None,
    cluster_to_label: Optional[Dict[int, str]] = None,
    marker_name_map: Optional[Dict[str, str]] = None,
    cmap: str = "YlGnBu",
    cluster_palette: str = "tab20",
) -> plt.Figure:
    """Plot an expression heatmap aggregated by overall identified populations (X, Y, ...) without C0/C1 prefixes."""
    return plot_cluster_heatmap(
        cluster_stats_df=cluster_stats_df,
        marker_cols=marker_cols,
        title=title,
        output_path=output_path,
        figsize=figsize,
        dpi=dpi,
        cluster_to_macro_group=cluster_to_macro_group,
        cluster_to_label=cluster_to_label,
        marker_name_map=marker_name_map,
        cmap=cmap,
        cluster_palette=cluster_palette,
        group_by_population=True,
    )


def plot_clustering_comparison(
    df: pd.DataFrame,
    template_labels: np.ndarray,
    som_labels: Optional[np.ndarray] = None,
    phenograph_labels: Optional[np.ndarray] = None,
    x_channel: str = "",
    y_channel: str = "",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: str = "Comparison: Manual Template Gating vs SOM vs PhenoGraph",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    legend_mode: str = "centroid",
    square_plot_area: bool = True,
) -> plt.Figure:
    """Generate side-by-side 3-panel comparison figure between template gating and clustering.

    Ensures standardized 1:1 square plot areas by default for each comparison panel,
    scaled compactly for publication figures unless the caller explicitly specifies a custom figsize.

    Args:
        df: Event DataFrame.
        template_labels: 1D array of template population names or categories per event.
        som_labels: 1D array of SOM cluster IDs.
        phenograph_labels: 1D array of PhenoGraph cluster IDs.
        x_channel: Channel for x-axis.
        y_channel: Channel for y-axis.
        x_label: X-axis label.
        y_label: Y-axis label.
        title: Title of overarching figure.
        output_path: Path to save figure.
        figsize: Dimensions. If None, defaults to triple-panel publication width (10.8, 3.6) with square panels.
        dpi: Resolution.
        legend_mode: 'centroid' (default, centroid badges) or 'external'.
        square_plot_area: If True and figsize not explicitly customized, enforces 1:1 square panels.

    Returns:
        Matplotlib Figure object.
    """
    if som_labels is None:
        raise ValueError("Must provide som_labels.")
    if phenograph_labels is None:
        raise ValueError("Must provide phenograph_labels.")

    custom_figsize = figsize is not None
    actual_figsize = figsize if custom_figsize else DEFAULT_FIGSIZE_TRIPLE

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=actual_figsize, dpi=dpi)

    if square_plot_area:
        ax1.set_box_aspect(1)
        ax2.set_box_aspect(1)
        ax3.set_box_aspect(1)

    # Panel 1: Template Gating Populations
    unique_temp = np.unique(template_labels)
    cmap_temp = plt.get_cmap("Set1", len(unique_temp))
    for i, pop_name in enumerate(unique_temp):
        mask = template_labels == pop_name
        n_pop = int(np.sum(mask))
        if n_pop == 0:
            continue
        if n_pop > 10000:
            rng = np.random.RandomState(42)
            sub_pop_idx = rng.choice(np.where(mask)[0], size=10000, replace=False)
            x_pts = df.iloc[sub_pop_idx][x_channel].to_numpy()
            y_pts = df.iloc[sub_pop_idx][y_channel].to_numpy()
        else:
            x_pts = df.loc[mask, x_channel].to_numpy()
            y_pts = df.loc[mask, y_channel].to_numpy()

        ax1.scatter(
            x_pts,
            y_pts,
            c=[cmap_temp(i)],
            s=3.0,
            alpha=0.65,
            edgecolors="none",
            label=f"{pop_name} ({n_pop:,})",
            rasterized=True,
        )
    ax1.set_title("1. Default Template Gating", fontsize=8.5, fontweight="bold")
    ax1.set_xlabel(x_label or x_channel, fontsize=8.0)
    ax1.set_ylabel(y_label or y_channel, fontsize=8.0)
    apply_nature_axis_style(ax1)
    ax1.legend(loc="upper right", markerscale=2.0, fontsize=6.5, framealpha=0.85)

    # Panel 2: SOM Clusters
    plot_clusters_scatter_2d(
        df=df,
        cluster_labels=som_labels,
        x_channel=x_channel,
        y_channel=y_channel,
        x_label=x_label,
        y_label=y_label,
        title="2. SOM Unsupervised Clusters",
        ax=ax2,
        legend_mode=legend_mode,
        figsize=None,
        square_plot_area=square_plot_area,
    )

    # Panel 3: PhenoGraph Clusters
    plot_clusters_scatter_2d(
        df=df,
        cluster_labels=phenograph_labels,
        x_channel=x_channel,
        y_channel=y_channel,
        x_label=x_label,
        y_label=y_label,
        title="3. PhenoGraph Community Clusters",
        ax=ax3,
        legend_mode=legend_mode,
        figsize=None,
        square_plot_area=square_plot_area,
    )

    fig.suptitle(format_multiline_title(title, max_chars=65), fontsize=10.0, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_clustering_qc_diagnostic(
    df: pd.DataFrame,
    qc_result: Any,
    fcs_data: Optional[Any] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    square_plot_area: bool = True,
    legend_loc: str = "bottom",
) -> plt.Figure:
    """Plot comprehensive 2-panel diagnostic figure for automated clustering QC.

    Panel A: Scatter Morphology QC (FSC-A vs SSC-A) showing GMM clusters
             (Intact Cells vs Debris and saturation artifacts) with convex hull overlay.
    Panel B: Doublet Discrimination QC (FSC-A vs FSC-H) showing Singlets vs Doublets
             with dynamic singlet envelope overlay.
    Ensures standardized 1:1 square plot areas by default for both panels,
    scaled compactly for publication subplots unless the caller explicitly specifies a custom figsize.

    Args:
        df: Raw events DataFrame.
        qc_result: ClusteringQCResult instance containing masks, gates, and labels.
        fcs_data: Optional FCSData instance for axis label formatting.
        output_path: Optional file path to export PNG.
        figsize: Figure dimensions. If None, defaults to publication dual-column with bottom legend (7.6, 4.4).
        dpi: Export resolution.
        square_plot_area: If True and figsize not explicitly customized, enforces 1:1 square panels.
        legend_loc: Placement of population legends. 'bottom' (default, outside underneath each plot
            to prevent covering cell populations), 'inside' (traditional top/bottom corner),
            or 'none'.

    Returns:
        Matplotlib Figure object.
    """
    custom_figsize = figsize is not None
    actual_figsize = figsize if custom_figsize else (DEFAULT_FIGSIZE_QC if legend_loc == "bottom" else (7.6, 3.8))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=actual_figsize, dpi=dpi)

    if square_plot_area:
        ax1.set_box_aspect(1)
        ax2.set_box_aspect(1)


    fsc_col = qc_result.cells_gate.channels[0]
    ssc_col = qc_result.cells_gate.channels[1]

    # Subsample for responsive, crisp rendering
    n_events = len(df)
    max_pts = 25000
    if n_events > max_pts:
        rng = np.random.RandomState(42)
        plot_idx = rng.choice(n_events, size=max_pts, replace=False)
    else:
        plot_idx = np.arange(n_events)

    df_sub = df.iloc[plot_idx]
    morph_sub = qc_result.morphology_labels[plot_idx]

    # -------------------------------------------------------------
    # Panel A: Scatter Morphology QC
    # -------------------------------------------------------------
    color_map_morph = {
        "Intact Cells": "#1f77b4",          # Classic cytometry blue
        "Debris / Noise": "#b0bec5",        # Light muted gray
        "Saturation Outliers": "#e53935",   # Warning red
    }

    for label_val, color in color_map_morph.items():
        mask_l = morph_sub == label_val
        n_pop = int(np.sum(qc_result.morphology_labels == label_val))
        if np.sum(mask_l) > 0:
            ax1.scatter(
                df_sub.loc[mask_l, fsc_col],
                df_sub.loc[mask_l, ssc_col],
                c=color,
                s=3.0,
                alpha=0.60 if label_val == "Intact Cells" else 0.35,
                edgecolors="none",
                label=f"{label_val} ({n_pop:,} events, {n_pop / n_events * 100:.1f}%)",
                rasterized=True,
            )

    # Overlay Cells Gate Convex Hull polygon
    if qc_result.cells_gate is not None:
        poly_verts = qc_result.cells_gate.vertices
        closed_verts = np.vstack([poly_verts, poly_verts[0]])
        ax1.plot(
            closed_verts[:, 0],
            closed_verts[:, 1],
            color="#0d47a1",
            linewidth=1.4,
            linestyle="-",
            label="Extracted Cells Hull Gate",
        )

    x_label_a = fcs_data.get_channel_label(fsc_col) if fcs_data else fsc_col
    y_label_a = fcs_data.get_channel_label(ssc_col) if fcs_data else ssc_col
    ax1.set_title(
        format_multiline_title(
            f"A. Scatter Morphology QC: Intact Cells vs. Debris\n"
            f"Retained: {qc_result.cell_count:,} / {n_events:,} ({qc_result.cell_count / n_events * 100:.1f}%)",
            max_chars=42,
        ),
        fontsize=8.5,
        fontweight="bold",
    )
    ax1.set_xlabel(x_label_a, fontsize=8.0)
    apply_nature_axis_style(ax1)

    # Bind Panel A axis limits to cytometer dynamic range to prevent extreme outlier compression
    max_fsc = fcs_data.channel_ranges.get(fsc_col, 262144.0) if fcs_data else 262144.0
    max_ssc = fcs_data.channel_ranges.get(ssc_col, 262144.0) if fcs_data else 262144.0
    if max_fsc > 0:
        ax1.set_xlim(-0.02 * max_fsc, 1.05 * max_fsc)
    if max_ssc > 0:
        ax1.set_ylim(-0.02 * max_ssc, 1.05 * max_ssc)

    if legend_loc == "bottom":
        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=1,
            markerscale=2.0,
            fontsize=6.5,
            frameon=True,
            framealpha=0.90,
            edgecolor="#e0e0e0",
            handletextpad=0.4,
        )
    elif legend_loc == "inside":
        ax1.legend(loc="upper right", markerscale=2.0, fontsize=6.5, framealpha=0.90)

    # -------------------------------------------------------------
    # Panel B: Doublet Discrimination QC
    # -------------------------------------------------------------
    sing_fsc_a = qc_result.singlets_gate.channels[0]
    sing_fsc_h = qc_result.singlets_gate.channels[1]

    sing_sub = qc_result.singlet_labels[plot_idx]
    color_map_sing = {
        "Singlets": "#2e7d32",              # Vibrant emerald green
        "Doublets / Aggregates": "#8e24aa", # Purple / magenta
        "Debris / Non-cells": "#cfd8dc",    # Pale gray
    }

    for label_val, color in color_map_sing.items():
        mask_s = sing_sub == label_val
        n_pop_s = int(np.sum(qc_result.singlet_labels == label_val))
        if np.sum(mask_s) > 0:
            ax2.scatter(
                df_sub.loc[mask_s, sing_fsc_a],
                df_sub.loc[mask_s, sing_fsc_h],
                c=color,
                s=3.0,
                alpha=0.65 if label_val == "Singlets" else 0.40,
                edgecolors="none",
                label=f"{label_val} ({n_pop_s:,} events, {n_pop_s / n_events * 100:.1f}%)",
                rasterized=True,
            )

    # Overlay Singlets Gate Convex Hull polygon
    if qc_result.singlets_gate is not None:
        poly_verts_s = qc_result.singlets_gate.vertices
        closed_verts_s = np.vstack([poly_verts_s, poly_verts_s[0]])
        ax2.plot(
            closed_verts_s[:, 0],
            closed_verts_s[:, 1],
            color="#1b5e20",
            linewidth=1.4,
            linestyle="-",
            label="Extracted Singlets Hull Gate",
        )

    x_label_b = fcs_data.get_channel_label(sing_fsc_a) if fcs_data else sing_fsc_a
    y_label_b = fcs_data.get_channel_label(sing_fsc_h) if fcs_data else sing_fsc_h
    sing_pct_cell = (qc_result.singlet_count / qc_result.cell_count * 100.0) if qc_result.cell_count > 0 else 0.0
    ax2.set_title(
        format_multiline_title(
            f"B. Pulse Geometry QC: Singlets vs. Doublets\n"
            f"Singlets: {qc_result.singlet_count:,} ({sing_pct_cell:.1f}% of cells, {qc_result.singlet_count / n_events * 100:.1f}% total)",
            max_chars=42,
        ),
        fontsize=8.5,
        fontweight="bold",
    )
    ax2.set_xlabel(x_label_b, fontsize=8.0)
    ax2.set_ylabel(y_label_b, fontsize=8.0)
    apply_nature_axis_style(ax2)

    # Bind Panel B axis limits to cytometer dynamic range for accurate pulse geometry (FSC-A vs FSC-H)
    max_fsc_a = fcs_data.channel_ranges.get(sing_fsc_a, 262144.0) if fcs_data else 262144.0
    max_fsc_h = fcs_data.channel_ranges.get(sing_fsc_h, 262144.0) if fcs_data else 262144.0
    if max_fsc_a > 0:
        ax2.set_xlim(-0.02 * max_fsc_a, 1.05 * max_fsc_a)
    if max_fsc_h > 0:
        ax2.set_ylim(-0.02 * max_fsc_h, 1.05 * max_fsc_h)

    if legend_loc == "bottom":
        ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=1,
            markerscale=2.0,
            fontsize=6.5,
            frameon=True,
            framealpha=0.90,
            edgecolor="#e0e0e0",
            handletextpad=0.4,
        )
    elif legend_loc == "inside":
        ax2.legend(loc="lower right", markerscale=2.0, fontsize=6.5, framealpha=0.90)

    fig.suptitle(
        format_multiline_title(
            "Automated Clustering Quality Control (QC) Diagnostic",
            max_chars=60,
        ),
        fontsize=10.0,
        fontweight="bold",
        y=0.98 if legend_loc == "bottom" else 1.02,
    )
    if legend_loc == "bottom":
        fig.tight_layout(rect=[0, 0.20, 1, 0.96])
    else:
        plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_spillover_matrix(
    spillover_df: pd.DataFrame,
    title: str = "Fluorescence Spillover Matrix",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    condition_number: Optional[float] = None,
    correlation_with_embedded: Optional[float] = None,
) -> plt.Figure:
    """Plot an annotated publication-quality heatmap of the fluorescence spillover matrix.

    Visualizes cross-detector bleedover percentages, highlighting high off-diagonal spillover
    and reporting the matrix condition number diagnostic.

    Args:
        spillover_df: Square DataFrame of spillover coefficients.
        title: Title of heatmap.
        output_path: Optional filepath to save the plot.
        figsize: Figure dimensions. If None, scales dynamically with channel count.
        dpi: Resolution.
        condition_number: Condition number kappa(S) to display in badge.
        correlation_with_embedded: Pearson correlation r with instrument matrix.

    Returns:
        Matplotlib Figure object.
    """
    set_nature_style()
    n_ch = len(spillover_df)

    raw_labels = [col.replace("Ex ", "").replace("Em ", "").replace("-A", "") for col in spillover_df.columns]
    max_label_len = max((len(lbl) for lbl in raw_labels), default=0)
    wrapped_labels = ["\n".join(textwrap.wrap(lbl, width=14)) if len(lbl) > 13 else lbl for lbl in raw_labels]

    diag_parts = []
    if condition_number is not None:
        diag_parts.append(f"Condition Number κ={condition_number:.2f}")
    if correlation_with_embedded is not None:
        diag_parts.append(f"Instrument Correlation r={correlation_with_embedded:.4f}")

    has_diag = bool(diag_parts)

    base_w = 4.2 + (max_label_len * 0.06) + (n_ch * 0.45)
    base_h = 3.8 + (max_label_len * 0.05) + (n_ch * 0.42)
    if has_diag:
        base_w = max(5.6, base_w)
        base_h = max(5.0, base_h)

    actual_figsize = figsize if figsize is not None else (max(4.5, min(12.0, base_w)), max(4.0, min(11.0, base_h)))
    fig, ax = plt.subplots(figsize=actual_figsize, dpi=dpi)

    # Convert to percentage and create display matrix with diagonal dimmed for contrast
    mat_vals = spillover_df.to_numpy(dtype=np.float64) * 100.0
    display_vals = mat_vals.copy()

    # Mask diagonal for color scaling so off-diagonal spillovers are vividly visible
    off_diag_mask = ~np.eye(n_ch, dtype=bool)
    max_off = float(np.nanmax(display_vals[off_diag_mask])) if np.any(off_diag_mask) else 100.0
    vmax = max(10.0, min(100.0, max_off * 1.05))

    im = ax.imshow(display_vals, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="equal")

    # Ticks and labels (auto-scales with channel count)
    tick_fs = max(6.5, min(8.5, 11.5 - n_ch * 0.35))
    ax.set_xticks(range(n_ch))
    ax.set_yticks(range(n_ch))
    rot = 45 if n_ch <= 10 else 90
    ha = "right" if rot == 45 else "center"
    ax.set_xticklabels(wrapped_labels, rotation=rot, ha=ha, fontsize=tick_fs)
    ax.set_yticklabels(wrapped_labels, fontsize=tick_fs)

    ax.set_xlabel("Detector Channel (Spillover Into)", fontsize=8.5, fontweight="bold", labelpad=6)
    ax.set_ylabel("Stained Fluorophore Channel", fontsize=8.5, fontweight="bold", labelpad=6)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spillover Percentage (%)", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.0)

    # Annotate cells if reasonable size
    if n_ch <= 15:
        cell_fs = max(6.0, min(7.5, 8.5 - n_ch * 0.15))
        for i in range(n_ch):
            for j in range(n_ch):
                val = mat_vals[i, j]
                color = "white" if val > (vmax * 0.6) else "black"
                txt = f"{val:.1f}%" if val >= 0.5 or i == j else ""
                if txt:
                    ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=cell_fs)

    # Title & Diagnostics
    sub_title = "\n(" + " • ".join(diag_parts) + ")" if diag_parts else ""
    wrap_width = max(38, int(actual_figsize[0] * 8.0))
    title_str = format_multiline_title(f"{title}{sub_title}", max_chars=wrap_width)
    ax.set_title(title_str, fontsize=9.0, fontweight="bold", pad=12)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_umap_projection(
    embedding: np.ndarray,
    labels: Union[np.ndarray, List[str], pd.Series],
    title: str = "UMAP 2D Manifold Projection",
    output_path: Optional[Union[str, Path]] = None,
    legend_mode: str = "centroid",
    palette: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    square_plot_area: bool = True,
    match_cluster_color: bool = True,
    show_counts_in_legend: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot UMAP 2D manifold colored by unsupervised cluster or biological lineage assignments.

    Args:
        embedding: (N, 2) NumPy array of UMAP coordinates.
        labels: 1D array or list of categorical or integer cluster/lineage labels.
        title: Title of the plot.
        output_path: Optional path to save image file.
        legend_mode: 'centroid' (badge annotations at population medians),
                     'external' (side legend), 'both', or 'none'.
        palette: Matplotlib colormap name (default: 'tab20' or 'tab10').
        figsize: Optional custom figure dimensions.
        dpi: Export resolution.
        square_plot_area: If True, enforces 1:1 square aspect ratio.
        match_cluster_color: If True, renders the cluster badge number in the matching cluster color.
        show_counts_in_legend: If True, appends (XX.X%, n=...) to external legend entries.

    Returns:
        Tuple of (fig, ax).
    """
    labels_arr = np.asarray(labels)
    raw_unique = list(dict.fromkeys(labels_arr.tolist()))
    str_unique = [str(x) for x in raw_unique]
    if all(s.isdigit() for s in str_unique):
        sorted_pairs = sorted(zip(raw_unique, str_unique), key=lambda p: int(p[1]))
    elif all(s.startswith("C") and s[1:].isdigit() for s in str_unique):
        sorted_pairs = sorted(zip(raw_unique, str_unique), key=lambda p: int(p[1][1:]))
    elif all(s.startswith("Cluster ") and s[8:].isdigit() for s in str_unique):
        sorted_pairs = sorted(zip(raw_unique, str_unique), key=lambda p: int(p[1][8:]))
    else:
        sorted_pairs = sorted(zip(raw_unique, str_unique), key=lambda p: p[1])
    unique_labels = [p[0] for p in sorted_pairs]
    n_labels = len(unique_labels)
    leg_ncol = 1 if n_labels <= 10 else (2 if n_labels <= 20 else 3)

    custom_figsize = figsize is not None
    if custom_figsize:
        actual_figsize = figsize
    elif legend_mode in ("external", "both"):
        actual_figsize = (5.2 + 1.6 * (leg_ncol - 1), 3.8)
    else:
        actual_figsize = (3.8, 3.8)

    fig, ax = plt.subplots(figsize=actual_figsize, dpi=dpi)
    if square_plot_area:
        ax.set_box_aspect(1)

    cmap_name = palette or "tab20"
    color_map = get_cluster_palette(unique_labels, cmap_name=cmap_name)
    n_total = max(1, len(labels_arr))

    for i, lbl in enumerate(unique_labels):
        mask = labels_arr == lbl
        color = color_map.get(str(lbl), (0.2, 0.4, 0.8, 1.0))
        lbl_str = f"Cluster {lbl}" if isinstance(lbl, (int, np.integer)) else str(lbl)
        if show_counts_in_legend and legend_mode in ("external", "both"):
            cnt = int(np.sum(mask))
            pct = 100.0 * cnt / n_total
            leg_label = f"{lbl_str} ({pct:.1f}%, n={cnt:,})"
        else:
            leg_label = lbl_str
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            s=3.0,
            alpha=0.7,
            edgecolors="none",
            label=leg_label,
            rasterized=True,
        )

        if legend_mode in ("centroid", "both") and np.any(mask):
            med_x = float(np.median(embedding[mask, 0]))
            med_y = float(np.median(embedding[mask, 1]))
            if isinstance(lbl, (int, np.integer)) or (isinstance(lbl, str) and lbl.isdigit()):
                badge_text = str(lbl)
            elif isinstance(lbl, str) and lbl.startswith("Cluster ") and lbl[8:].isdigit():
                badge_text = lbl[8:]
            elif isinstance(lbl, str) and lbl.startswith("C") and lbl[1:].isdigit():
                badge_text = lbl[1:]
            else:
                badge_text = str(lbl)[:3].upper()

            text_color = color if match_cluster_color else "black"
            border_color = color if match_cluster_color else "black"
            ax.annotate(
                badge_text,
                xy=(med_x, med_y),
                fontsize=7.5,
                fontweight="bold",
                color=text_color,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="circle,pad=0.25",
                    fc="white",
                    ec=border_color,
                    lw=1.2 if match_cluster_color else 0.8,
                    alpha=0.92,
                ),
                zorder=10,
            )

    ax.set_xlabel("UMAP 1", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=8.5, fontweight="bold")
    ax.set_title(format_multiline_title(title, max_chars=45), fontsize=9.0, fontweight="bold")
    apply_nature_axis_style(ax)

    if legend_mode in ("external", "both"):
        is_pure_clusters = all(
            s.isdigit()
            or (s.startswith("C") and s[1:].isdigit())
            or (s.startswith("Cluster ") and s[8:].isdigit())
            or (s.startswith("Cluster_") and s[8:].isdigit())
            for s in str_unique
        )
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.04, 1.0),
            ncol=leg_ncol,
            columnspacing=1.0,
            frameon=False,
            fontsize=7.0,
            title="Clusters" if is_pure_clusters else "Populations",
            title_fontsize=7.5,
            markerscale=3.0,
        )

    if leg_ncol == 1:
        plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig, ax


def plot_umap_marker_features(
    embedding: np.ndarray,
    df_features: pd.DataFrame,
    markers: List[str],
    marker_name_map: Optional[Dict[str, str]] = None,
    title: str = "UMAP High-Dimensional Marker Expression Maps",
    output_path: Optional[Union[str, Path]] = None,
    max_markers: Optional[int] = None,
    cmap: str = "magma",
    dpi: int = 300,
) -> plt.Figure:
    """Plot multi-panel continuous feature intensity maps across the UMAP manifold.

    Plots cells ordered by expression value so that highly expressed cells are rendered on top,
    preventing positive phenotypes from being obscured by negative clusters.

    Args:
        embedding: (N, 2) array of UMAP coordinates.
        df_features: DataFrame containing marker intensity columns for the embedded events.
        markers: List of marker channel names to visualize.
        marker_name_map: Optional dictionary mapping detector channels to biological marker names.
        title: Overall super-title for the figure.
        output_path: Optional path to save image.
        max_markers: Optional limit on marker panels to plot (default: None, plots all markers).
        cmap: Matplotlib colormap (default: 'magma').
        dpi: Resolution (default: 300).

    Returns:
        Matplotlib Figure object.
    """
    valid_markers = [m for m in markers if m in df_features.columns]
    if max_markers is not None:
        valid_markers = valid_markers[:max_markers]

    n_plots = len(valid_markers)
    if n_plots == 0:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No valid marker features found for UMAP overlay", ha="center", va="center")
        return fig

    # Dynamically arrange subplots in an optimal publication grid
    if n_plots <= 3:
        n_cols = n_plots
    elif n_plots <= 8:
        n_cols = 4 if n_plots in [4, 7, 8] else 3
    elif n_plots <= 15:
        n_cols = 4 if n_plots <= 12 else 5
    else:
        n_cols = 5
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.3 * n_rows), dpi=dpi)
    if n_plots == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    names = marker_name_map or {}

    for i, marker in enumerate(valid_markers):
        ax = axes_flat[i]
        vals = df_features[marker].to_numpy(dtype=np.float64)
        order = np.argsort(vals)

        sc = ax.scatter(
            embedding[order, 0],
            embedding[order, 1],
            c=vals[order],
            cmap=cmap,
            s=2.5,
            alpha=0.85,
            edgecolors="none",
            rasterized=True,
        )
        lbl = names.get(marker, marker)
        ax.set_title(format_multiline_title(lbl, max_chars=25), fontsize=8.0, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=7.0)
        ax.set_ylabel("UMAP 2", fontsize=7.0)
        ax.set_box_aspect(1)
        apply_nature_axis_style(ax)

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6.0)

    # Hide unused axes
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(format_multiline_title(title, max_chars=60), fontsize=10.0, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_population_marker_violins(
    df_events: pd.DataFrame,
    population_labels: Union[np.ndarray, List[str]],
    markers: List[str],
    marker_name_map: Optional[Dict[str, str]] = None,
    title: str = "Phenotypic Marker Expression Distributions",
    output_path: Optional[Union[str, Path]] = None,
    max_markers: int = 6,
    max_populations: int = 8,
    dpi: int = 300,
) -> plt.Figure:
    """Plot violin distributions of key phenotypic markers across discovered clusters or lineages.

    Args:
        df_events: DataFrame of single-cell events.
        population_labels: 1D array/list of population assignments per event.
        markers: List of channel detector names to display.
        marker_name_map: Mapping from channel name to biological marker label.
        title: Super-title.
        output_path: Optional image export path.
        max_markers: Maximum number of marker panels (default: 6).
        max_populations: Maximum populations to display (default: 8).
        dpi: Resolution.

    Returns:
        Matplotlib Figure object.
    """
    import seaborn as sns

    valid_markers = [m for m in markers if m in df_events.columns][:max_markers]
    if not valid_markers:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No valid markers for violin plots", ha="center", va="center")
        return fig

    pop_series = pd.Series(population_labels, index=df_events.index, name="Population")
    top_pops = pop_series.value_counts().index[:max_populations]
    filter_mask = pop_series.isin(top_pops)

    plot_df = df_events.loc[filter_mask, valid_markers].copy()
    plot_df["Population"] = pop_series[filter_mask].astype(str)

    n_markers = len(valid_markers)
    n_cols = min(n_markers, 3)
    n_rows = int(np.ceil(n_markers / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.6 * n_rows), dpi=dpi, sharex=True)
    if n_markers == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    names = marker_name_map or {}

    for i, marker in enumerate(valid_markers):
        ax = axes_flat[i]
        lbl = names.get(marker, marker)
        sns.violinplot(
            data=plot_df,
            x="Population",
            y=marker,
            hue="Population",
            legend=False,
            ax=ax,
            palette="muted",
            inner="quartile",
            density_norm="width",
            cut=0,
        )
        ax.set_title(format_multiline_title(lbl, max_chars=25), fontsize=8.0, fontweight="bold")
        ax.set_ylabel("Intensity", fontsize=7.0)
        ax.set_xlabel("" if i < (n_rows - 1) * n_cols else "Population", fontsize=7.0)
        ax.tick_params(axis="x", rotation=45, labelsize=6.5)
        apply_nature_axis_style(ax)

    for j in range(n_markers, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(format_multiline_title(title, max_chars=60), fontsize=9.5, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_quadrant_gate(
    df: pd.DataFrame,
    x_channel: str,
    y_channel: str,
    x_cutoff: float,
    y_cutoff: float,
    title: str = "Quadrant Gating",
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (6.0, 5.5),
) -> plt.Figure:
    """Plot 2D bivariate density with crosshair quadrant cutoffs and percentage annotations.

    Follows the ISAC standard:
        - Q1 (Upper Left): X- Y+
        - Q2 (Upper Right): X+ Y+ (double positive)
        - Q3 (Lower Left): X- Y- (double negative)
        - Q4 (Lower Right): X+ Y-
    """
    clean_df = df[[x_channel, y_channel]].dropna()
    total = len(clean_df)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    x = clean_df[x_channel].to_numpy()
    y = clean_df[y_channel].to_numpy()

    # Subsample if large for responsive plotting
    if total > 20000:
        sub_idx = np.random.RandomState(42).choice(total, size=20000, replace=False)
        px, py = x[sub_idx], y[sub_idx]
    else:
        px, py = x, y

    # Bivariate scatter with density coloring
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([px, py])
        z = gaussian_kde(xy, bw_method=0.1)(xy)
        idx = z.argsort()
        px, py, z = px[idx], py[idx], z[idx]
        ax.scatter(px, py, c=z, s=3, cmap="viridis", alpha=0.6, rasterized=True)
    except Exception:
        ax.scatter(px, py, s=3, color="#1f77b4", alpha=0.4, rasterized=True)

    # Draw Crosshairs
    ax.axvline(x=x_cutoff, color="#d62728", linestyle="--", lw=1.5, zorder=4)
    ax.axhline(y=y_cutoff, color="#d62728", linestyle="--", lw=1.5, zorder=4)

    # Calculate Quadrant Percentages
    x_pos = x > x_cutoff
    y_pos = y > y_cutoff
    pct_q1 = float(np.mean((~x_pos) & y_pos) * 100) if total > 0 else 0.0
    pct_q2 = float(np.mean(x_pos & y_pos) * 100) if total > 0 else 0.0
    pct_q3 = float(np.mean((~x_pos) & (~y_pos)) * 100) if total > 0 else 0.0
    pct_q4 = float(np.mean(x_pos & (~y_pos)) * 100) if total > 0 else 0.0

    # Annotate 4 Corners
    box_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d62728", alpha=0.85, lw=1)
    ax.text(0.04, 0.94, f"Q1 (UL)\n{pct_q1:.1f}%", transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="top", ha="left", bbox=box_props)
    ax.text(0.96, 0.94, f"Q2 (UR)\n{pct_q2:.1f}%", transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="top", ha="right", bbox=box_props)
    ax.text(0.04, 0.06, f"Q3 (LL)\n{pct_q3:.1f}%", transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left", bbox=box_props)
    ax.text(0.96, 0.06, f"Q4 (LR)\n{pct_q4:.1f}%", transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="bottom", ha="right", bbox=box_props)

    ax.set_xlabel(x_channel, fontsize=10, fontweight="bold")
    ax.set_ylabel(y_channel, fontsize=10, fontweight="bold")
    ax.set_title(format_multiline_title(title, max_chars=40), fontsize=11, fontweight="bold")
    apply_nature_axis_style(ax)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_stratified_umap_density_contours(
    embedding: np.ndarray,
    sample_ids: List[str],
    sample_metadata: pd.DataFrame,
    condition_column: str,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Stratified UMAP Density Contours",
    dpi: int = 300,
) -> plt.Figure:
    """Generate stratified side-by-side 2D density contour maps comparing conditions on UMAP.

    Plots overall cohort events in light gray, overlaid with high-resolution 2D KDE density
    contours for each condition to visually highlight cohort-specific population shifts.

    Args:
        embedding: 2D numpy array [n_events x 2] of UMAP coordinates.
        sample_ids: List of sample IDs corresponding to each event in embedding.
        sample_metadata: Metadata DataFrame mapping sample ID to condition.
        condition_column: Column name in sample_metadata defining experimental groups.
        output_path: Destination image path.
        title: Figure title.
        dpi: Resolution.

    Returns:
        Matplotlib Figure object.
    """
    import seaborn as sns

    conditions = [
        str(sample_metadata.loc[s, condition_column])
        if s in sample_metadata.index and condition_column in sample_metadata.columns
        else "Unknown"
        for s in sample_ids
    ]
    unique_conds = list(dict.fromkeys(conditions))
    if len(unique_conds) < 2:
        unique_conds = list(set(conditions))

    n_conds = len(unique_conds)
    fig, axes = plt.subplots(1, n_conds, figsize=(4.0 * n_conds, 4.0), dpi=dpi, squeeze=False)
    axes_flat = axes.flatten()

    x_min, x_max = float(embedding[:, 0].min() - 0.5), float(embedding[:, 0].max() + 0.5)
    y_min, y_max = float(embedding[:, 1].min() - 0.5), float(embedding[:, 1].max() + 0.5)

    cond_palette = sns.color_palette("tab10", n_conds)

    for idx, cond in enumerate(unique_conds):
        ax = axes_flat[idx]
        mask = [c == cond for c in conditions]
        n_cond_events = sum(mask)

        # Background events (light gray)
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c="#E2E8F0",
            s=1.0,
            alpha=0.25,
            rasterized=True,
            edgecolors="none",
        )

        if n_cond_events >= 20:
            cond_emb = embedding[mask]
            # 2D KDE contour overlay
            sns.kdeplot(
                x=cond_emb[:, 0],
                y=cond_emb[:, 1],
                ax=ax,
                levels=6,
                color=cond_palette[idx % len(cond_palette)],
                fill=True,
                alpha=0.45,
                thresh=0.05,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{cond} (n={n_cond_events:,})", fontsize=10.0, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=8.5, fontweight="medium")
        if idx == 0:
            ax.set_ylabel("UMAP 2", fontsize=8.5, fontweight="medium")
        else:
            ax.set_ylabel("")
        ax.set_box_aspect(1.0)
        apply_nature_axis_style(ax)

    fig.suptitle(format_multiline_title(title, max_chars=60), fontsize=11.0, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def format_multiline_legend_label(text: str, n_count: Optional[int] = None, max_chars_per_line: int = 18) -> str:
    """Format legend labels with clean multiline wrapping to prevent horizontal overlap."""
    clean_text = str(text).strip()
    import re
    tokens = re.split(r'([ _\-/])', clean_text)
    lines = []
    current_line = ""
    for tok in tokens:
        if not tok:
            continue
        if len(current_line) + len(tok) > max_chars_per_line and current_line.strip():
            lines.append(current_line.rstrip())
            current_line = tok.lstrip()
        else:
            current_line += tok
    if current_line:
        lines.append(current_line.rstrip())
    wrapped = "\n".join(lines) if lines else clean_text
    if n_count is not None:
        wrapped += f"\n(n={n_count:,})"
    return wrapped


def plot_cohort_umap_covariates(
    embedding: np.ndarray,
    sample_ids: List[str],
    lineage_or_cluster_labels: Union[np.ndarray, List[str], pd.Series],
    sample_metadata: Optional[pd.DataFrame] = None,
    covariate_columns: Optional[List[str]] = None,
    condition_column: Optional[str] = None,
    batch_column: Optional[str] = None,
    cluster_palette: str = "tab20",
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Cohort UMAP Covariate Analysis",
    dpi: int = 300,
) -> plt.Figure:
    """Generate multi-panel UMAP figures comparing population lineages against cohort covariates.

    Default subplots:
      1. Discovered cell population (if determined) / cluster ID
      2. Sample origin (sample_id, required)
    Additional subplots are included ONLY when explicitly declared:
      - condition_column: experimental condition if declared and present in metadata
      - batch_column: batch identifier if declared and present in metadata
      - covariate_columns: specific metadata columns declared by the user (e.g. cytometer, replicate)

    Args:
        embedding: 2D numpy array [n_events x 2] of UMAP coordinates.
        sample_ids: List of sample IDs corresponding to each event in embedding.
        lineage_or_cluster_labels: Population or cluster assignments for each event.
        sample_metadata: Optional cohort metadata DataFrame keyed by sample_id.
        covariate_columns: Optional explicit list of metadata columns to display as panels.
        condition_column: Optional name of primary condition column.
        batch_column: Optional name of batch column.
        output_path: Destination image path.
        title: Main figure title.
        dpi: Resolution.

    Returns:
        Matplotlib Figure object.
    """
    if embedding is None or len(embedding) == 0:
        fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=dpi)
        ax.text(0.5, 0.5, "No UMAP embedding events available", ha="center", va="center")
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        return fig

    panels: List[Tuple[str, List[str], str, str]] = []
    # (panel_title, label_list, palette_name, legend_title)

    # 1. Biological Populations / Lineages (Default Subplot A)
    pop_labels = [str(x) for x in lineage_or_cluster_labels]
    panels.append(("Discovered Populations", pop_labels, cluster_palette, "Population"))

    # 2. Sample Origin (Default Subplot B - Must)
    samp_labels = [str(s) for s in sample_ids]
    n_samps = len(set(samp_labels))
    samp_palette = "tab20" if n_samps > 10 else "tab10"
    panels.append(("Sample Origin", samp_labels, samp_palette, "Sample ID"))

    # 3. Explicitly declared covariates from metadata (No silent guessing)
    if sample_metadata is not None and isinstance(sample_metadata, pd.DataFrame):
        meta_df = sample_metadata.copy()

        def _lookup(s_id: str, col: str) -> str:
            if s_id in meta_df.index and col in meta_df.columns:
                val = meta_df.loc[s_id, col]
                return str(val) if pd.notna(val) else "Unassigned"
            if "sample_id" in meta_df.columns and col in meta_df.columns:
                match = meta_df[meta_df["sample_id"] == s_id]
                if not match.empty:
                    val = match[col].iloc[0]
                    return str(val) if pd.notna(val) else "Unassigned"
            return "Unassigned"

        declared_cols: List[str] = []
        if condition_column and condition_column in meta_df.columns and condition_column not in declared_cols:
            declared_cols.append(condition_column)

        if batch_column and batch_column in meta_df.columns and batch_column not in declared_cols:
            declared_cols.append(batch_column)

        if covariate_columns:
            for c in covariate_columns:
                if c in meta_df.columns and c not in declared_cols:
                    declared_cols.append(c)

        for col in declared_cols:
            col_labels = [_lookup(s, col) for s in sample_ids]
            u_vals = set(col_labels)
            pal = "Set1" if col == condition_column else ("Dark2" if col == batch_column else "tab10")
            if len(u_vals) > 8:
                pal = "tab20" if len(u_vals) <= 20 else "turbo"
            disp_title = col.replace("_", " ").title()
            panels.append((f"Covariate: {disp_title}", col_labels, pal, disp_title))

    n_panels = len(panels)
    if n_panels == 1:
        nrows, ncols = 1, 1
        figsize = (5.8, 4.8)
        wspace, hspace = 0.45, 0.35
    elif n_panels == 2:
        nrows, ncols = 1, 2
        figsize = (12.2, 4.8)
        wspace, hspace = 0.52, 0.35
    elif n_panels == 3:
        nrows, ncols = 1, 3
        figsize = (17.5, 4.8)
        wspace, hspace = 0.52, 0.35
    elif n_panels == 4:
        nrows, ncols = 2, 2
        figsize = (12.5, 9.6)
        wspace, hspace = 0.48, 0.38
    else:
        ncols = 3
        nrows = int(np.ceil(n_panels / 3))
        figsize = (17.5, 4.8 * nrows)
        wspace, hspace = 0.50, 0.38

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        gridspec_kw={"wspace": wspace, "hspace": hspace},
    )
    axes_flat = axes.flatten()

    x_pad = max(0.5, float((embedding[:, 0].max() - embedding[:, 0].min()) * 0.05))
    y_pad = max(0.5, float((embedding[:, 1].max() - embedding[:, 1].min()) * 0.05))
    x_min, x_max = float(embedding[:, 0].min() - x_pad), float(embedding[:, 0].max() + x_pad)
    y_min, y_max = float(embedding[:, 1].min() - y_pad), float(embedding[:, 1].max() + y_pad)

    for idx, (p_title, p_labels, pal, leg_title) in enumerate(panels):
        ax = axes_flat[idx]
        p_labels_arr = np.asarray(p_labels)
        seen = set()
        u_cats = [c for c in p_labels if not (c in seen or seen.add(c))]
        n_cats = max(1, len(u_cats))
        if idx == 0:
            color_lookup = get_cluster_palette(u_cats, cmap_name=pal)
        else:
            cmap = plt.get_cmap(pal, n_cats)
            color_lookup = {
                cat: (cmap(c_i) if hasattr(cmap, "__call__") else cmap[c_i % len(cmap)])
                for c_i, cat in enumerate(u_cats)
            }

        for cat in u_cats:
            mask = p_labels_arr == cat
            cat_color = color_lookup[cat]
            disp_cat = format_multiline_legend_label(str(cat), n_count=int(mask.sum()), max_chars_per_line=18)
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[cat_color],
                s=2.5,
                alpha=0.7,
                edgecolors="none",
                label=disp_cat,
                rasterized=True,
            )

        ax.set_box_aspect(1.0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("UMAP 1", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("UMAP 2", fontsize=9.5, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=8.5)
        ax.set_title(format_multiline_title(p_title, max_chars=36), fontsize=10.5, fontweight="bold", pad=8)
        apply_nature_axis_style(ax)

        leg_ncol = 1 if n_cats <= 8 else (2 if n_cats <= 16 else 3)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            ncol=leg_ncol,
            columnspacing=0.9,
            frameon=False,
            fontsize=8.0,
            title=leg_title,
            title_fontsize=9.0,
            markerscale=2.5,
            handletextpad=0.5,
            labelspacing=0.55,
        )

    for extra_ax in axes_flat[n_panels:]:
        extra_ax.set_visible(False)

    fig.suptitle(format_multiline_title(title, max_chars=60), fontsize=12.5, fontweight="bold", y=0.99)

    # Dynamic check: verify no legend overlaps neighboring axis
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        if not hasattr(fig.canvas, "get_renderer"):
            FigureCanvasAgg(fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        max_overlap = 0.0
        for r in range(nrows):
            for c in range(ncols - 1):
                idx_curr = r * ncols + c
                idx_next = r * ncols + (c + 1)
                if idx_curr < n_panels and idx_next < n_panels:
                    leg = axes_flat[idx_curr].get_legend()
                    if leg is not None:
                        leg_bb = leg.get_window_extent(renderer)
                        ax_next_bb = axes_flat[idx_next].get_window_extent(renderer)
                        if leg_bb.x1 > (ax_next_bb.x0 - 15.0):
                            overlap = leg_bb.x1 - (ax_next_bb.x0 - 15.0)
                            if overlap > max_overlap:
                                max_overlap = overlap

        if max_overlap > 0:
            fig_w_px = fig.get_size_inches()[0] * dpi
            extra_wspace = (max_overlap / fig_w_px) * 2.5
            fig.subplots_adjust(wspace=wspace + extra_wspace)
    except Exception:
        pass

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_cluster_population_bridge(
    embedding: Optional[np.ndarray],
    cluster_labels: Sequence[Any],
    population_labels: Sequence[Any],
    mapping_df: Optional[pd.DataFrame] = None,
    title: str = "Unsupervised Clusters vs. Annotated Populations",
    output_path: Optional[Union[str, Path]] = None,
    palette: str = "tab20",
    dpi: int = 300,
) -> Figure:
    """Render a paired bridge figure connecting raw technical clusters to annotated populations.

    Panels:
    - Panel A: 2D UMAP colored by raw unsupervised clusters (`C0..Ck`), or cluster size bar if UMAP omitted.
    - Panel B: 2D UMAP colored by annotated biological populations (or lineage composition summary).
    - Panel C (when mapping_df is provided): Compact crosswalk table showing C_i -> Biological Lineage merges.
    """
    set_nature_style()
    cl_arr = np.asarray([str(x) for x in cluster_labels])
    pop_arr = np.asarray([str(x) for x in population_labels])

    has_umap = embedding is not None and len(embedding) == len(cl_arr) and len(cl_arr) > 0
    has_table = mapping_df is not None and not mapping_df.empty

    if has_table:
        fig = plt.figure(figsize=(16.2, 6.0))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.36)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_mid = fig.add_subplot(gs[0, 1])
        ax_right = fig.add_subplot(gs[0, 2])
    else:
        fig, (ax_left, ax_mid) = plt.subplots(1, 2, figsize=(12.0, 5.8))
        ax_right = None
        fig.subplots_adjust(wspace=0.34)

    x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
    if has_umap and embedding is not None:
        x_lo, x_hi = float(np.min(embedding[:, 0])), float(np.max(embedding[:, 0]))
        y_lo, y_hi = float(np.min(embedding[:, 1])), float(np.max(embedding[:, 1]))
        dx = max(1e-6, (x_hi - x_lo) * 0.06)
        dy = max(1e-6, (y_hi - y_lo) * 0.06)
        x_min, x_max = x_lo - dx, x_hi + dx
        y_min, y_max = y_lo - dy, y_hi + dy

    def _scatter_panel(ax: Any, labels_arr: np.ndarray, panel_title: str) -> None:
        uniq = sorted(
            np.unique(labels_arr),
            key=lambda x: (0, int(x[1:])) if (x.startswith("C") and x[1:].isdigit()) else (1, str(x)),
        )
        color_map = get_cluster_palette(uniq, cmap_name=palette)
        for lbl in uniq:
            mask = labels_arr == lbl
            c_col = color_map[str(lbl)]
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[c_col],
                s=5,
                alpha=0.65,
                linewidths=0,
                label=lbl,
                rasterized=True,
            )
            if np.any(mask):
                cx = float(np.median(embedding[mask, 0]))
                cy = float(np.median(embedding[mask, 1]))
                short_txt = lbl if len(lbl) <= 18 else lbl[:16] + ".."
                ax.text(
                    cx,
                    cy,
                    short_txt,
                    fontsize=7.2,
                    fontweight="bold",
                    color=c_col,
                    ha="center",
                    va="center",
                    zorder=5,
                    bbox=dict(
                        boxstyle="round,pad=0.20",
                        facecolor="white",
                        edgecolor=c_col,
                        linewidth=1.1,
                        alpha=0.90,
                    ),
                )
        ax.set_box_aspect(1.0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("UMAP 1", fontsize=8.5, fontweight="bold")
        ax.set_ylabel("UMAP 2", fontsize=8.5, fontweight="bold")
        ax.set_title(panel_title, fontsize=9.8, fontweight="bold", pad=8)
        apply_nature_axis_style(ax)
        if len(uniq) <= 18:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=min(3, max(1, len(uniq))),
                fontsize=6.5,
                frameon=False,
                markerscale=2.0,
            )

    if has_umap:
        _scatter_panel(ax_left, cl_arr, "A. Technical Clusters (Unsupervised)")
        _scatter_panel(ax_mid, pop_arr, "B. Biological Populations (Annotated)")
    else:
        cl_counts = pd.Series(cl_arr).value_counts()
        pop_counts = pd.Series(pop_arr).value_counts()
        ax_left.barh(cl_counts.index.astype(str), cl_counts.values, color="#4C72B0", alpha=0.85)
        ax_left.set_title("A. Technical Clusters (Event Counts)", fontsize=9.8, fontweight="bold")
        ax_left.set_xlabel("Events", fontsize=8.5)
        apply_nature_axis_style(ax_left)

        ax_mid.barh(pop_counts.index.astype(str), pop_counts.values, color="#55A868", alpha=0.85)
        ax_mid.set_title("B. Biological Populations (Event Counts)", fontsize=9.8, fontweight="bold")
        ax_mid.set_xlabel("Events", fontsize=8.5)
        apply_nature_axis_style(ax_mid)

    if ax_right is not None and mapping_df is not None:
        ax_right.axis("off")
        ax_right.set_title("C. Cluster -> Population Crosswalk", fontsize=9.8, fontweight="bold", pad=8)
        show_cols = [c for c in ("Cluster_Label", "Annotated_Population", "Pct_Total_Events") if c in mapping_df.columns]
        if not show_cols:
            show_cols = list(mapping_df.columns)[:3]
        disp_df = mapping_df[show_cols].head(24).copy()
        if "Pct_Total_Events" in disp_df.columns:
            disp_df["Pct_Total_Events"] = disp_df["Pct_Total_Events"].map(lambda v: f"{float(v):.1f}%")
        cell_text = disp_df.astype(str).values.tolist()
        col_labels = [c.replace("_", " ") for c in show_cols]
        tbl = ax_right.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.8)
        tbl.scale(1.0, 1.25)

    fig.suptitle(format_multiline_title(title, max_chars=75), fontsize=12.0, fontweight="bold", y=0.99)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig






