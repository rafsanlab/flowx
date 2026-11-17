"""Reusable Nature-style benchmark comparison visualizations built on core FlowX plots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from flowx.extra.comparisons.metrics import (
    compute_cohort_cleanup_concordance,
    compute_replicate_concordance,
    compute_spillover_concordance,
)
from flowx.gating.gate import Gate
from flowx.math.compensation import CompensationControlResult, CompensationMatrix
from flowx.visualization.comp_plots import plot_spillover_and_diff_matrix
from flowx.visualization.plotting import (
    plot_clusters_scatter_2d,
    plot_gate_overlay,
    plot_scatter_2d,
)
from flowx.visualization.style import apply_nature_axis_style, set_nature_style


def _save_and_maybe_close(
    fig: plt.Figure,
    output_path: Optional[Union[str, Path]],
    dpi: int = 300,
) -> None:
    """Persist figure to disk when ``output_path`` is supplied."""
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")


def plot_cleanup_gates_comparison(
    ref_raw_df: pd.DataFrame,
    flowx_working_df: pd.DataFrame,
    flowx_cells_gate: Gate,
    flowx_singlet_gate: Gate,
    flowx_cells_mask: Optional[np.ndarray] = None,
    ref_singlet_gate: Optional[Gate] = None,
    ref_cells_parent_mask: Optional[np.ndarray] = None,
    ref_cells_gate: Optional[Gate] = None,
    ref_negated_box_vertices: Optional[Sequence[Tuple[float, float]]] = None,
    ref_negated_box_label: str = "Excluded\nDebris Box\n(negated)",
    flowx_cluster_info: Optional[Dict[str, Any]] = None,
    ref_cells_title: str = "A. Published Reference Cells Gate (P1)",
    ref_singlet_title: str = "B. Published Reference Singlets Gate (P2)",
    flowx_cells_title: str = "C. FlowX Automated GMM 'Cells' Gate",
    flowx_singlet_title: str = "D. FlowX Automated Singlet Gate",
    suptitle: Optional[str] = None,
    ref_color: str = "#d62728",
    flowx_color: str = "#2ca02c",
    max_plot_points: int = 25000,
    figsize: Tuple[float, float] = (8.8, 8.2),
    dpi: int = 300,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create a 2x2 comparative clean-up gating figure (Reference vs. FlowX).

    Row 1 displays external published reference gates:
      - Top Left (Panel A): Published P1 / Debris Exclusion on FSC-A vs SSC-A.
      - Top Right (Panel B): Published P2 / Singlets on FSC-A vs FSC-H.
    Row 2 displays FlowX automated gating directly underneath for 1:1 visual comparison:
      - Bottom Left (Panel C): FlowX GMM Debris/Cells Gate & clusters on FSC-A vs SSC-A.
      - Bottom Right (Panel D): FlowX Singlet Gate & doublet discrimination on FSC-A vs FSC-H.

    Args:
        ref_raw_df: Event DataFrame used with the external reference workspace.
        flowx_working_df: Pre-cleaned event DataFrame (e.g., post Time QC) evaluated by FlowX.
        flowx_cells_gate: FlowX Debris/Cells gate instance.
        flowx_singlet_gate: FlowX Singlet gate instance.
        flowx_cells_mask: Boolean mask of events inside ``flowx_cells_gate``.
        ref_singlet_gate: External workspace singlet polygon/gate instance.
        ref_cells_parent_mask: Boolean mask selecting intact reference cells before singlet gating.
        ref_cells_gate: Optional positive polygon/gate for reference intact cells.
        ref_negated_box_vertices: Optional vertex sequence for a negated debris box (FlowJo style).
        ref_negated_box_label: Annotation text inside negated debris box.
        flowx_cluster_info: Optional dictionary containing GMM cluster diagnostics (e.g. ``pred_labels``).
        ref_cells_title: Panel A title.
        ref_singlet_title: Panel B title.
        flowx_cells_title: Panel C title.
        flowx_singlet_title: Panel D title.
        suptitle: Optional figure super-title.
        ref_color: Accent color for external reference gates.
        flowx_color: Accent color for FlowX gates and intact cell clusters.
        max_plot_points: Subsampling ceiling for responsive scatter rendering.
        figsize: Figure dimensions in inches.
        dpi: Resolution in dots per inch.
        output_path: Optional output image path.

    Returns:
        Matplotlib ``Figure`` instance.
    """
    set_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    rng = np.random.RandomState(42)

    # Subsample reference events for scatter
    n_ref = len(ref_raw_df)
    ref_sub_idx = rng.choice(n_ref, size=min(n_ref, max_plot_points), replace=False) if n_ref > max_plot_points else np.arange(n_ref)
    ref_sub_df = ref_raw_df.iloc[ref_sub_idx]

    # Subsample flowx events for scatter
    n_fx = len(flowx_working_df)
    fx_sub_idx = rng.choice(n_fx, size=min(n_fx, max_plot_points), replace=False) if n_fx > max_plot_points else np.arange(n_fx)
    fx_sub_df = flowx_working_df.iloc[fx_sub_idx]

    # -------------------------------------------------------------------------
    # Row 0, Col 0: Panel A - Published Reference Cells Gate (P1 / FSC vs SSC)
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    fsc_col = flowx_cells_gate.channels[0]
    ssc_col = flowx_cells_gate.channels[1]

    ax_a.scatter(
        ref_sub_df[fsc_col],
        ref_sub_df[ssc_col],
        s=1.8,
        c="#4a7ebb",
        alpha=0.32,
        rasterized=True,
        label="Raw Events",
    )
    if ref_cells_gate is not None:
        ref_verts = getattr(ref_cells_gate, "vertices", None)
        if ref_verts is not None:
            v_arr = np.asarray(ref_verts, dtype=np.float64)
            poly_p1 = patches.Polygon(v_arr, closed=True, fill=False, edgecolor=ref_color, linewidth=1.6)
            ax_a.add_patch(poly_p1)
        ref_cells_count = int(np.sum(ref_cells_gate.apply(ref_raw_df)))
        ref_cells_pct = (ref_cells_count / n_ref * 100.0) if n_ref > 0 else 0.0
        ax_a.plot([], [], color=ref_color, linewidth=1.6, label=f"Ref Cells: {ref_cells_count:,} ({ref_cells_pct:.1f}%)")
    elif ref_negated_box_vertices:
        verts_arr = np.asarray(ref_negated_box_vertices, dtype=np.float64)
        debris_patch = patches.Polygon(verts_arr, closed=True, fill=True, facecolor="#7f7f7f", alpha=0.22, edgecolor=ref_color, linewidth=1.5, linestyle="--")
        ax_a.add_patch(debris_patch)
        cx = float(np.mean(verts_arr[:, 0]))
        cy = float(np.mean(verts_arr[:, 1]))
        ax_a.text(cx, cy, ref_negated_box_label, ha="center", va="center", fontsize=7.2, fontweight="bold", color=ref_color, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=ref_color, alpha=0.9))

    def _scatter_upper(series: pd.Series) -> float:
        arr = series.to_numpy(dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 260000.0
        hi = float(np.percentile(arr, 99.7))
        return 260000.0 if hi <= 260000.0 else hi * 1.08

    fsc_a_lim = max(_scatter_upper(ref_sub_df[fsc_col]), _scatter_upper(fx_sub_df[fsc_col]))
    ssc_a_lim = max(_scatter_upper(ref_sub_df[ssc_col]), _scatter_upper(fx_sub_df[ssc_col]))

    ax_a.set_title(ref_cells_title, fontsize=8.8, fontweight="bold", pad=6)
    ax_a.set_xlabel(fsc_col, fontsize=7.8, fontweight="bold")
    ax_a.set_ylabel(ssc_col, fontsize=7.8, fontweight="bold")
    ax_a.set_xlim(0, fsc_a_lim)
    ax_a.set_ylim(0, ssc_a_lim)
    ax_a.legend(loc="upper left", fontsize=7.2, frameon=True)
    apply_nature_axis_style(ax_a)

    # -------------------------------------------------------------------------
    # Row 0, Col 1: Panel B - Published Reference Singlets Gate (P2)
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    fsc_h_col = flowx_singlet_gate.channels[1] if len(flowx_singlet_gate.channels) > 1 else "FSC-H"

    ref_s_x_col = ref_singlet_gate.channels[0] if hasattr(ref_singlet_gate, "channels") and len(ref_singlet_gate.channels) >= 2 else fsc_col
    ref_s_y_col = ref_singlet_gate.channels[1] if hasattr(ref_singlet_gate, "channels") and len(ref_singlet_gate.channels) >= 2 else fsc_h_col

    ref_singlet_eval_df = ref_sub_df
    if ref_cells_parent_mask is not None:
        sub_parent_mask = ref_cells_parent_mask[ref_sub_idx]
        if np.any(sub_parent_mask):
            ref_singlet_eval_df = ref_sub_df.loc[sub_parent_mask]

    ax_b.scatter(
        ref_singlet_eval_df[ref_s_x_col],
        ref_singlet_eval_df[ref_s_y_col],
        s=1.8,
        c="#4a7ebb",
        alpha=0.32,
        rasterized=True,
        label="Parent Events",
    )
    if ref_singlet_gate is not None:
        ref_s_verts = getattr(ref_singlet_gate, "vertices", None)
        if ref_s_verts is not None:
            v_s_arr = np.asarray(ref_s_verts, dtype=np.float64)
            poly_p2 = patches.Polygon(v_s_arr, closed=True, fill=False, edgecolor=ref_color, linewidth=1.6)
            ax_b.add_patch(poly_p2)
        ref_s_count = int(np.sum(ref_singlet_gate.apply(ref_raw_df)))
        ref_s_pct = (ref_s_count / n_ref * 100.0) if n_ref > 0 else 0.0
        ax_b.plot([], [], color=ref_color, linewidth=1.6, label=f"Ref Singlets: {ref_s_count:,} ({ref_s_pct:.1f}%)")

    ax_b.set_title(ref_singlet_title, fontsize=8.8, fontweight="bold", pad=6)
    ax_b.set_xlabel(ref_s_x_col, fontsize=7.8, fontweight="bold")
    ax_b.set_ylabel(ref_s_y_col, fontsize=7.8, fontweight="bold")
    ref_x_lim = _scatter_upper(ref_sub_df[ref_s_x_col])
    ref_y_lim = _scatter_upper(ref_sub_df[ref_s_y_col])
    fsc_h_lim = max(_scatter_upper(ref_sub_df[fsc_h_col]), _scatter_upper(fx_sub_df[fsc_h_col]))
    ax_b.set_xlim(0, ref_x_lim)
    ax_b.set_ylim(0, ref_y_lim)
    ax_b.legend(loc="upper left", fontsize=7.2, frameon=True)
    apply_nature_axis_style(ax_b)

    # -------------------------------------------------------------------------
    # Row 1, Col 0: Panel C - FlowX Automated Cells Gate & Clusters (Below P1)
    # -------------------------------------------------------------------------
    ax_c = axes[1, 0]
    fx_cells_mask_full = flowx_cells_mask if flowx_cells_mask is not None else flowx_cells_gate.apply(flowx_working_df)
    n_fx_cells = int(np.sum(fx_cells_mask_full))
    pct_fx_cells = (n_fx_cells / n_fx * 100.0) if n_fx > 0 else 0.0

    if flowx_cluster_info is not None and "pred_labels" in flowx_cluster_info:
        pred_labels = flowx_cluster_info["pred_labels"]
        sub_pred = pred_labels[fx_sub_idx]
        debris_cids = flowx_cluster_info.get("debris_clusters", [])
        cell_cids = flowx_cluster_info.get("cell_clusters", [])

        # Debris
        deb_mask = np.isin(sub_pred, debris_cids)
        if np.any(deb_mask):
            n_deb_tot = int(np.sum(np.isin(pred_labels, debris_cids)))
            pct_deb_tot = n_deb_tot / n_fx * 100.0
            ax_c.scatter(
                fx_sub_df.loc[deb_mask, fsc_col],
                fx_sub_df.loc[deb_mask, ssc_col],
                s=1.8,
                c="#7f7f7f",
                alpha=0.32,
                rasterized=True,
                label=f"Debris / Low Scatter: {n_deb_tot:,} ({pct_deb_tot:.1f}%)",
            )

        # Margin Saturation
        sat_mask = (sub_pred == -1)
        if np.any(sat_mask):
            n_sat_tot = int(np.sum(pred_labels == -1))
            pct_sat_tot = n_sat_tot / n_fx * 100.0
            ax_c.scatter(
                fx_sub_df.loc[sat_mask, fsc_col],
                fx_sub_df.loc[sat_mask, ssc_col],
                s=1.8,
                c="#e53935",
                alpha=0.45,
                rasterized=True,
                label=f"Margin Saturation: {n_sat_tot:,} ({pct_sat_tot:.1f}%)",
            )

        # Intact Cells
        cell_mask_sub = np.isin(sub_pred, cell_cids) & (~sat_mask) & (~deb_mask)
        if np.any(cell_mask_sub):
            ax_c.scatter(
                fx_sub_df.loc[cell_mask_sub, fsc_col],
                fx_sub_df.loc[cell_mask_sub, ssc_col],
                s=1.8,
                c=flowx_color,
                alpha=0.35,
                rasterized=True,
                label=f"FlowX Intact Cells: {n_fx_cells:,} ({pct_fx_cells:.1f}%)",
            )
    else:
        # Fallback to binary scatter
        sub_cells_mask = fx_cells_mask_full[fx_sub_idx]
        ax_c.scatter(
            fx_sub_df.loc[~sub_cells_mask, fsc_col],
            fx_sub_df.loc[~sub_cells_mask, ssc_col],
            s=1.8,
            c="#7f7f7f",
            alpha=0.32,
            rasterized=True,
            label=f"Excluded Debris: {(n_fx - n_fx_cells):,} ({100.0 - pct_fx_cells:.1f}%)",
        )
        ax_c.scatter(
            fx_sub_df.loc[sub_cells_mask, fsc_col],
            fx_sub_df.loc[sub_cells_mask, ssc_col],
            s=1.8,
            c=flowx_color,
            alpha=0.35,
            rasterized=True,
            label=f"FlowX Intact Cells: {n_fx_cells:,} ({pct_fx_cells:.1f}%)",
        )

    # Overlay FlowX Cells gate polygon if available
    fx_cells_verts = getattr(flowx_cells_gate, "vertices", None)
    if fx_cells_verts is not None:
        v_fx_arr = np.asarray(fx_cells_verts, dtype=np.float64)
        poly_fx_cells = patches.Polygon(v_fx_arr, closed=True, fill=False, edgecolor="#1b5e20", linewidth=1.7, linestyle="-")
        ax_c.add_patch(poly_fx_cells)

    ax_c.set_title(flowx_cells_title, fontsize=8.8, fontweight="bold", pad=6)
    ax_c.set_xlabel(fsc_col, fontsize=7.8, fontweight="bold")
    ax_c.set_ylabel(ssc_col, fontsize=7.8, fontweight="bold")
    ax_c.set_xlim(0, fsc_a_lim)
    ax_c.set_ylim(0, ssc_a_lim)
    ax_c.legend(loc="upper left", fontsize=7.2, frameon=True)
    apply_nature_axis_style(ax_c)

    # -------------------------------------------------------------------------
    # Row 1, Col 1: Panel D - FlowX Singlet Gate & Clusters (Below P2)
    # -------------------------------------------------------------------------
    ax_d = axes[1, 1]
    # Evaluate FlowX singlets on intact cells
    fx_cells_df = flowx_working_df.loc[fx_cells_mask_full]
    fx_singlet_mask_local = flowx_singlet_gate.apply(fx_cells_df)
    n_singlets = int(np.sum(fx_singlet_mask_local))
    pct_singlets_parent = (n_singlets / n_fx_cells * 100.0) if n_fx_cells > 0 else 0.0
    pct_singlets_total = (n_singlets / n_fx * 100.0) if n_fx > 0 else 0.0

    # Subsample parent intact cells for singlet scatter
    n_parent = len(fx_cells_df)
    parent_sub_idx = rng.choice(n_parent, size=min(n_parent, max_plot_points), replace=False) if n_parent > max_plot_points else np.arange(n_parent)
    sub_parent_df = fx_cells_df.iloc[parent_sub_idx]
    sub_is_singlet = fx_singlet_mask_local[parent_sub_idx]

    # Doublets
    if np.any(~sub_is_singlet):
        n_dub = n_parent - n_singlets
        pct_dub = (n_dub / n_parent * 100.0) if n_parent > 0 else 0.0
        ax_d.scatter(
            sub_parent_df.loc[~sub_is_singlet, fsc_col],
            sub_parent_df.loc[~sub_is_singlet, fsc_h_col],
            s=1.8,
            c="#d62728",
            alpha=0.35,
            rasterized=True,
            label=f"Doublets: {n_dub:,} ({pct_dub:.1f}%)",
        )

    # Singlets
    if np.any(sub_is_singlet):
        ax_d.scatter(
            sub_parent_df.loc[sub_is_singlet, fsc_col],
            sub_parent_df.loc[sub_is_singlet, fsc_h_col],
            s=1.8,
            c=flowx_color,
            alpha=0.35,
            rasterized=True,
            label=f"Singlets: {n_singlets:,} ({pct_singlets_parent:.1f}% parent, {pct_singlets_total:.1f}% raw)",
        )

    # Overlay FlowX Singlet Gate Polygon
    fx_singlet_verts = getattr(flowx_singlet_gate, "vertices", None)
    if fx_singlet_verts is not None:
        v_s_arr = np.asarray(fx_singlet_verts, dtype=np.float64)
        poly_fx_singlets = patches.Polygon(v_s_arr, closed=True, fill=False, edgecolor="#1b5e20", linewidth=1.7, linestyle="-")
        ax_d.add_patch(poly_fx_singlets)

    ax_d.set_title(flowx_singlet_title, fontsize=8.8, fontweight="bold", pad=6)
    ax_d.set_xlabel(fsc_col, fontsize=7.8, fontweight="bold")
    ax_d.set_ylabel(fsc_h_col, fontsize=7.8, fontweight="bold")
    ax_d.set_xlim(0, fsc_a_lim)
    ax_d.set_ylim(0, fsc_h_lim)
    ax_d.legend(loc="upper left", fontsize=7.2, frameon=True)
    apply_nature_axis_style(ax_d)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10.2, fontweight="bold", y=0.995)
    plt.tight_layout()
    _save_and_maybe_close(fig, output_path, dpi=dpi)
    return fig


def plot_lineage_cluster_comparison(
    events_df: pd.DataFrame,
    reference_labels: np.ndarray,
    cluster_labels: np.ndarray,
    bivariate_x: str,
    bivariate_y: str,
    bivariate_x_label: Optional[str] = None,
    bivariate_y_label: Optional[str] = None,
    manifold_x: str = "UMAP_1",
    manifold_y: str = "UMAP_2",
    titles: Sequence[str] = (
        "A. Reference Manual Labels",
        "B. FlowX Unsupervised Clusters",
        "C. Reference Lineages on Manifold",
        "D. FlowX Clusters on Manifold",
    ),
    suptitle: Optional[str] = None,
    legend_mode: str = "centroid",
    figsize: Tuple[float, float] = (8.8, 8.4),
    dpi: int = 300,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot a 2x2 event-level comparison between reference manual labels and FlowX clusters.

    Renders both bivariate marker expression space (top row) and 2D UMAP/manifold
    embedding space (bottom row) via :func:`plot_clusters_scatter_2d`.

    Args:
        events_df: Event DataFrame containing marker and manifold coordinate columns.
        reference_labels: Integer array of external manual gate codes aligned to ``events_df``.
        cluster_labels: Integer array of FlowX cluster IDs aligned to ``events_df``.
        bivariate_x: Column name for top-row horizontal axis.
        bivariate_y: Column name for top-row vertical axis.
        bivariate_x_label: Axis label override for ``bivariate_x``.
        bivariate_y_label: Axis label override for ``bivariate_y``.
        manifold_x: Column name for bottom-row horizontal axis (default ``'UMAP_1'``).
        manifold_y: Column name for bottom-row vertical axis (default ``'UMAP_2'``).
        titles: 4-tuple of panel titles ``(A, B, C, D)``.
        suptitle: Optional figure super-title.
        legend_mode: ``'centroid'``, ``'external'``, or ``'none'``.
        figsize: Figure dimensions in inches.
        dpi: Resolution in dots per inch.
        output_path: Optional output image path.

    Returns:
        Matplotlib ``Figure`` instance.
    """
    if len(titles) != 4:
        raise ValueError(f"Expected 4 panel titles, got {len(titles)}.")

    set_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    specs = [
        (axes[0, 0], reference_labels, bivariate_x, bivariate_y, bivariate_x_label or bivariate_x, bivariate_y_label or bivariate_y, titles[0]),
        (axes[0, 1], cluster_labels, bivariate_x, bivariate_y, bivariate_x_label or bivariate_x, bivariate_y_label or bivariate_y, titles[1]),
        (axes[1, 0], reference_labels, manifold_x, manifold_y, "UMAP 1" if manifold_x == "UMAP_1" else manifold_x, "UMAP 2" if manifold_y == "UMAP_2" else manifold_y, titles[2]),
        (axes[1, 1], cluster_labels, manifold_x, manifold_y, "UMAP 1" if manifold_x == "UMAP_1" else manifold_x, "UMAP 2" if manifold_y == "UMAP_2" else manifold_y, titles[3]),
    ]

    for ax, labels, x_col, y_col, x_lbl, y_lbl, title in specs:
        plot_clusters_scatter_2d(
            events_df,
            cluster_labels=labels,
            x_channel=x_col,
            y_channel=y_col,
            x_label=x_lbl,
            y_label=y_lbl,
            title=title,
            legend_mode=legend_mode,
            ax=ax,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=10.2, fontweight="bold", y=0.995)
    plt.tight_layout()
    _save_and_maybe_close(fig, output_path, dpi=dpi)
    return fig


def plot_compensation_concordance(
    calculated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    marker_map: Optional[Dict[str, str]] = None,
    reference_label: str = "External Reference Matrix",
    figsize: Tuple[float, float] = (14.8, 5.6),
    dpi: int = 300,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Compare calculated FlowX spillover matrix against an external instrument/workspace matrix.

    Uses :func:`compute_spillover_concordance` and delegates heatmap rendering to
    :func:`flowx.visualization.comp_plots.plot_spillover_and_diff_matrix`.

    Args:
        calculated_df: FlowX calculated spillover matrix DataFrame.
        reference_df: Reference instrument or workspace spillover matrix DataFrame.
        marker_map: Optional detector-to-marker symbol mapping for axis labels.
        reference_label: Human-readable label shown in Panel B's title.
        figsize: Overall figure dimensions in inches.
        dpi: Resolution in dots per inch.
        output_path: Optional output image path.

    Returns:
        Matplotlib ``Figure`` instance.
    """
    set_nature_style()
    metrics = compute_spillover_concordance(calculated_df, reference_df)
    channels = list(metrics.diff_pct_dataframe.columns)
    calc_aligned = calculated_df.loc[channels, channels]

    comp_res = CompensationControlResult(
        matrix=CompensationMatrix(detectors=channels, matrix=calc_aligned.to_numpy(dtype=np.float64)),
        dataframe=calc_aligned,
        condition_number=metrics.condition_number,
        embedded_correlation=metrics.pearson_r,
        embedded_mae=metrics.off_diag_mae,
        estimation_method_used="hybrid",
        diff_from_embedded_dataframe=metrics.diff_pct_dataframe,
    )
    comp_fig = plot_spillover_and_diff_matrix(
        comp_result=comp_res,
        marker_map=marker_map,
        output_path=None,
        dpi=dpi,
    )
    comp_fig.set_size_inches(*figsize)
    comp_fig.subplots_adjust(left=0.20, right=0.95, top=0.86, bottom=0.28, wspace=0.90)

    if len(comp_fig.axes) >= 2:
        comp_fig.axes[1].set_title(
            f"B. Concordance vs. {reference_label}\n"
            f"(Pearson r = {metrics.pearson_r:.6f} | Off-Diagonal MAE = {metrics.off_diag_mae * 100.0:.3f}%)",
            fontsize=9.0,
            fontweight="bold",
            pad=8,
        )

    _save_and_maybe_close(comp_fig, output_path, dpi=dpi)
    return comp_fig


def plot_grouped_benchmark_bars(
    categories: Sequence[str],
    series: Sequence[Dict[str, Any]],
    ylabel: str = "Proportion (%)",
    title: str = "Benchmark Comparison",
    ylim: Optional[Tuple[float, float]] = None,
    legend_loc: str = "upper right",
    figsize: Tuple[float, float] = (5.0, 3.8),
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a multi-series grouped bar chart comparing literature/reference values vs FlowX.

    Args:
        categories: X-axis group labels.
        series: List of dicts describing bar series, each with keys:
            ``'label'`` (str), ``'values'`` (Sequence[float]), and optional
            ``'errors'`` (Sequence[float]), ``'color'`` (str).
        ylabel: Y-axis label.
        title: Axes title.
        ylim: Optional ``(ymin, ymax)`` axis bounds.
        legend_loc: Matplotlib legend location string.
        figsize: Dimensions used when ``ax`` is None.
        dpi: Resolution in dots per inch.
        ax: Optional pre-existing Matplotlib ``Axes`` for multi-panel composition.
        output_path: Optional output image path.

    Returns:
        Tuple of ``(Figure, Axes)``.
    """
    set_nature_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    n_cats = len(categories)
    n_series = len(series)
    if n_cats == 0 or n_series == 0:
        raise ValueError("Both 'categories' and 'series' must be non-empty.")

    x = np.arange(n_cats, dtype=np.float64)
    total_width = 0.72
    bar_width = total_width / n_series
    default_palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

    max_height = 0.0
    for idx, spec in enumerate(series):
        vals = np.asarray(spec["values"], dtype=np.float64)
        errs = np.asarray(spec.get("errors", np.zeros_like(vals)), dtype=np.float64)
        offset = (idx - (n_series - 1) / 2.0) * bar_width
        color = spec.get("color", default_palette[idx % len(default_palette)])
        has_err = np.any(errs > 0)
        bars = ax.bar(
            x + offset,
            vals,
            bar_width,
            yerr=errs if has_err else None,
            capsize=3 if has_err else 0,
            label=spec["label"],
            color=color,
            alpha=0.85,
        )
        for bar, err in zip(bars, errs):
            h = float(bar.get_height())
            max_height = max(max_height, h + float(err))
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + float(err) + max(1.2, max_height * 0.015),
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0.0, max(10.0, max_height * 1.18))
    ax.set_ylabel(ylabel, fontsize=8.3, fontweight="bold")
    ax.set_title(title, fontsize=8.8, fontweight="bold")
    ax.legend(loc=legend_loc, fontsize=7.1, framealpha=0.9)
    apply_nature_axis_style(ax)

    if standalone:
        plt.tight_layout()
    _save_and_maybe_close(fig, output_path, dpi=dpi)
    return fig, ax


def plot_replicate_scatter_concordance(
    values_x: Sequence[float],
    values_y: Sequence[float],
    point_labels: Optional[Sequence[str]] = None,
    xlabel: str = "Replicate 1 Frequency (%)",
    ylabel: str = "Replicate 2 Frequency (%)",
    title_prefix: str = "Replicate Frequency Concordance",
    point_color: str = "#2b5c8f",
    figsize: Tuple[float, float] = (4.2, 3.8),
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot y = x concordance scatter comparing paired cluster/population frequencies.

    Args:
        values_x: Paired values plotted along x-axis.
        values_y: Paired values plotted along y-axis.
        point_labels: Optional short labels rendered next to each point.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title_prefix: Prefix before computed Pearson ``r`` and max delta summary.
        point_color: Marker fill color.
        figsize: Dimensions used when ``ax`` is None.
        dpi: Resolution in dots per inch.
        ax: Optional pre-existing ``Axes`` for multi-panel composition.
        output_path: Optional output image path.

    Returns:
        Tuple of ``(Figure, Axes)``.
    """
    set_nature_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    stats = compute_replicate_concordance(values_x, values_y)
    vx = np.asarray(values_x, dtype=np.float64)
    vy = np.asarray(values_y, dtype=np.float64)

    ax.scatter(vx, vy, c=point_color, s=55, edgecolor="white", linewidth=0.8, zorder=5)
    lim_max = float(max(np.max(vx), np.max(vy))) + 3.0
    ax.plot([0.0, lim_max], [0.0, lim_max], linestyle="--", color="#555555", linewidth=1.2, label="Identity (y = x)")

    if point_labels is not None:
        for lbl, px, py in zip(point_labels, vx, vy):
            ax.annotate(lbl, (px, py), xytext=(4, -3), textcoords="offset points", fontsize=7.2, fontweight="bold")

    ax.set_xlim(0.0, lim_max)
    ax.set_ylim(0.0, lim_max)
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel, fontsize=8.2, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8.2, fontweight="bold")
    ax.set_title(
        f"{title_prefix}\n(Pearson r = {stats['pearson_r']:.5f} | Max \u0394 = {stats['max_abs_delta']:.2f}%)",
        fontsize=8.8,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=7.2, framealpha=0.9)
    apply_nature_axis_style(ax)

    if standalone:
        plt.tight_layout()
    _save_and_maybe_close(fig, output_path, dpi=dpi)
    return fig, ax


def export_per_sample_cleanup_plots(
    sample_tasks: Sequence[Dict[str, Any]],
    per_sample_dir: Union[str, Path],
    filename_suffix: str = "cleanup_comparison.png",
    dpi: int = 300,
) -> List[Path]:
    """Generate individual 2x2 clean-up comparison PNGs for every sample in a subdirectory.

    Prevents single-page overcrowding on multi-sample cohorts without resorting to
    multi-page PDFs by writing one figure per sample into ``per_sample_dir``.

    Args:
        sample_tasks: Sequence of dicts containing ``'sample_id'`` plus keyword arguments
            accepted by :func:`plot_cleanup_gates_comparison`.
        per_sample_dir: Target subdirectory (e.g. ``plots/per_sample``).
        filename_suffix: Suffix appended as ``<sample_id>_<filename_suffix>``.
        dpi: Output resolution in dots per inch.

    Returns:
        List of generated PNG file paths.
    """
    out_dir = Path(per_sample_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for task in sample_tasks:
        sample_id = str(task["sample_id"])
        kwargs = {k: v for k, v in task.items() if k != "sample_id"}
        safe_id = sample_id.replace("/", "_").replace("\\", "_")
        out_file = out_dir / f"{safe_id}_{filename_suffix}"
        kwargs["dpi"] = dpi
        kwargs["output_path"] = out_file
        fig = plot_cleanup_gates_comparison(**kwargs)
        plt.close(fig)
        saved_paths.append(out_file)

    return saved_paths


def plot_cohort_retention_concordance(
    records_df: pd.DataFrame,
    ref_col: str = "ref_final_pct",
    flowx_col: str = "flowx_final_pct",
    label_col: str = "sample_id",
    condition_col: Optional[str] = "condition",
    xlabel: str = "External Reference Clean-Up Retention (%)",
    ylabel: str = "FlowX Clean-Up Retention (%)",
    title: str = "Cohort-Wide Clean-Up Yield Concordance",
    figsize: Tuple[float, float] = (5.4, 4.6),
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot y = x yield concordance across all samples in a cohort, colored by condition.

    Scales cleanly from single-sample (N = 1) to large multi-condition cohorts (N >= 2).

    Args:
        records_df: Per-sample comparison DataFrame.
        ref_col: Reference yield percentage column.
        flowx_col: FlowX yield percentage column.
        label_col: Column storing sample identifiers for point annotations (shown when ``N <= 16``).
        condition_col: Optional column grouping samples by experimental condition.
        xlabel: Horizontal axis label.
        ylabel: Vertical axis label.
        title: Axes title prefix.
        figsize: Dimensions used when ``ax`` is None.
        dpi: Resolution in dots per inch.
        ax: Optional existing Matplotlib axes.
        output_path: Optional image save path.

    Returns:
        Tuple of ``(Figure, Axes)``.
    """
    set_nature_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    summary = compute_cohort_cleanup_concordance(records_df, flowx_col=flowx_col, ref_col=ref_col)
    df_valid = records_df.dropna(subset=[ref_col, flowx_col]).copy()

    has_cond = condition_col is not None and condition_col in df_valid.columns
    if has_cond:
        groups = list(dict.fromkeys(df_valid[condition_col].astype(str).tolist()))
        palette = sns.color_palette("tab10", max(len(groups), 1))
        for idx, grp in enumerate(groups):
            sub = df_valid[df_valid[condition_col].astype(str) == grp]
            ax.scatter(
                sub[ref_col],
                sub[flowx_col],
                color=palette[idx],
                s=60,
                edgecolor="white",
                linewidth=0.8,
                label=grp,
                zorder=5,
            )
    else:
        ax.scatter(
            df_valid[ref_col],
            df_valid[flowx_col],
            color="#2b5c8f",
            s=60,
            edgecolor="white",
            linewidth=0.8,
            label=f"Samples (n={len(df_valid)})",
            zorder=5,
        )

    all_vals = np.concatenate([df_valid[ref_col].to_numpy(dtype=np.float64), df_valid[flowx_col].to_numpy(dtype=np.float64)])
    val_span = max(10.0, float(np.max(all_vals) - np.min(all_vals)))
    pad = max(5.0, val_span * 0.16)
    vmin = max(0.0, float(np.min(all_vals)) - pad)
    vmax = min(105.0, float(np.max(all_vals)) + pad)

    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="#555555", linewidth=1.2, label="Identity (y = x)")

    if label_col in df_valid.columns and len(df_valid) <= 16:
        span = max(1.0, vmax - vmin)
        placed_boxes: List[Tuple[float, float, float, float]] = [
            ((float(r[ref_col]) - vmin) / span - 0.025, (float(r[flowx_col]) - vmin) / span - 0.025,
             (float(r[ref_col]) - vmin) / span + 0.025, (float(r[flowx_col]) - vmin) / span + 0.025)
            for _, r in df_valid.iterrows()
        ]
        # Candidate offsets in points: (dx_pt, dy_pt, horizontal_alignment)
        candidates = [
            (6, 5, "left"),
            (-6, 5, "right"),
            (6, -10, "left"),
            (-6, -10, "right"),
            (0, 14, "center"),
            (0, -15, "center"),
            (8, 18, "left"),
            (-8, -18, "right"),
        ]
        ordered_rows = df_valid.sort_values(by=[ref_col, flowx_col], ascending=[True, False])
        for _, row in ordered_rows.iterrows():
            px = float(row[ref_col])
            py = float(row[flowx_col])
            text_lbl = str(row[label_col])
            w_est = min(0.45, max(0.10, len(text_lbl) * 0.016))
            h_est = 0.055
            chosen = candidates[0]
            for dx_pt, dy_pt, ha in candidates:
                cx = (px - vmin) / span + dx_pt * 0.004
                cy = (py - vmin) / span + dy_pt * 0.004
                if ha == "left":
                    x0, x1 = cx, cx + w_est
                elif ha == "right":
                    x0, x1 = cx - w_est, cx
                else:
                    x0, x1 = cx - w_est / 2.0, cx + w_est / 2.0
                y0, y1 = cy - h_est / 2.0, cy + h_est / 2.0
                if x0 < 0.01 or x1 > 0.99 or y0 < 0.01 or y1 > 0.96:
                    continue
                if not any(not (x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1) for bx0, by0, bx1, by1 in placed_boxes):
                    chosen = (dx_pt, dy_pt, ha)
                    placed_boxes.append((x0, y0, x1, y1))
                    break
            else:
                dx_pt, dy_pt, ha = chosen
                cx = (px - vmin) / span + dx_pt * 0.004
                cy = (py - vmin) / span + dy_pt * 0.004
                placed_boxes.append((cx - w_est / 2.0, cy - h_est / 2.0, cx + w_est / 2.0, cy + h_est / 2.0))

            dx_pt, dy_pt, ha = chosen
            ax.annotate(
                text_lbl,
                (px, py),
                xytext=(dx_pt, dy_pt),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=6.8,
                arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.6) if abs(dx_pt) > 7 or abs(dy_pt) > 11 else None,
            )

    r_str = f"r = {summary['pearson_r']:.4f} | " if summary["pearson_r"] is not None else ""
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel, fontsize=8.2, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8.2, fontweight="bold")
    ax.set_title(
        f"{title}\n({r_str}n = {summary['n_samples']} | MAE = {summary['mae']:.2f}% | Bias = {summary['mean_bias_pct']:+.2f}%)",
        fontsize=8.6,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=6.9, framealpha=0.9)
    apply_nature_axis_style(ax)

    if standalone:
        plt.tight_layout()
    _save_and_maybe_close(fig, output_path, dpi=dpi)
    return fig, ax


def plot_multi_sample_gate_grid(
    panel_specs: Sequence[Dict[str, Any]],
    max_cols: int = 4,
    suptitle: Optional[str] = None,
    dpi: int = 300,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Arrange sample gating overlays dynamically across an adaptive ``nrows x ncols`` grid.

    Args:
        panel_specs: Sequence of dicts containing keys passed to :func:`plot_gate_overlay`
            (``df``, ``gate``, optional ``parent_mask``, ``x_label``, ``y_label``, ``title``, ``gate_color``).
        max_cols: Maximum subplots per row.
        suptitle: Optional figure super-title.
        dpi: Output resolution in dots per inch.
        output_path: Optional image save path.

    Returns:
        Matplotlib ``Figure`` instance.
    """
    n_panels = len(panel_specs)
    if n_panels == 0:
        raise ValueError("Cannot render multi-sample gate grid with zero panels.")

    set_nature_style()
    ncols = min(max(1, max_cols), n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), dpi=dpi)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, spec in zip(axes_flat, panel_specs):
        plot_gate_overlay(
            df=spec["df"],
            gate=spec["gate"],
            parent_mask=spec.get("parent_mask"),
            x_label=spec.get("x_label"),
            y_label=spec.get("y_label"),
            title=spec.get("title"),
            gate_color=spec.get("gate_color", "#1f77b4"),
            ax=ax,
        )

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10.2, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_and_maybe_close(fig, output_path, dpi=dpi)
    return fig


def plot_cohort_population_heatmap_comparison(
    flowx_freq_df: pd.DataFrame,
    ref_freq_df: Optional[pd.DataFrame] = None,
    sample_conditions: Optional[Dict[str, str]] = None,
    title_left: str = "A. FlowX Population Proportions (%)",
    title_right: str = "B. External Reference Proportions (%)",
    suptitle: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    square: bool = True,
    xtick_rotation: float = 45.0,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot cohort-wide per-sample population frequency heatmaps (FlowX vs. Reference).

    Displays every sample in the cohort sorted by experimental condition along rows
    and biological lineages/populations across columns with square inner boxes and
    45-degree angled label ticks.

    Args:
        flowx_freq_df: DataFrame indexed by ``sample_id`` with population frequency columns.
        ref_freq_df: Optional matching Ground Truth percentage table aligned by sample and column.
        sample_conditions: Optional ``sample_id -> condition`` mapping for row sorting/labeling.
        title_left: Title for FlowX panel.
        title_right: Title for Reference panel (when ``ref_freq_df`` is provided).
        suptitle: Optional figure super-title.
        figsize: Optional overall figure dimensions.
        dpi: Resolution in dots per inch.
        square: If True, forces inner matrix cells to remain 1:1 squares.
        xtick_rotation: Degrees to angle x-axis lineage/marker column labels.
        output_path: Optional image save path.

    Returns:
        Matplotlib ``Figure`` instance.
    """
    set_nature_style()
    ordered_idx = list(flowx_freq_df.index)
    if sample_conditions:
        ordered_idx = sorted(ordered_idx, key=lambda sid: (str(sample_conditions.get(str(sid), "")), str(sid)))
        row_labels = [f"[{sample_conditions.get(str(sid), 'Sample')}] {sid}" for sid in ordered_idx]
    else:
        row_labels = [str(sid) for sid in ordered_idx]

    fx_plot = flowx_freq_df.loc[ordered_idx].copy()
    fx_plot.index = row_labels

    has_ref = ref_freq_df is not None and not ref_freq_df.empty
    n_rows, n_cols = fx_plot.shape
    if figsize is None:
        panel_w = max(4.0, 1.6 + n_cols * 0.95)
        panel_h = max(3.4, 1.8 + n_rows * 0.75)
        figsize = (panel_w * (2 if has_ref else 1) + 0.8, panel_h)

    ncols_fig = 2 if has_ref else 1
    fig, axes = plt.subplots(1, ncols_fig, figsize=figsize, dpi=dpi, sharey=True)
    axes_list = [axes] if ncols_fig == 1 else list(axes)

    vmax = float(np.nanmax(fx_plot.to_numpy(dtype=np.float64)))
    if has_ref:
        common_cols = [c for c in fx_plot.columns if c in ref_freq_df.columns]
        if common_cols:
            fx_plot = fx_plot[common_cols]
            ref_plot = ref_freq_df.loc[ordered_idx, common_cols].copy()
        else:
            ref_plot = ref_freq_df.loc[ordered_idx].copy()
        ref_plot.index = row_labels
        vmax = max(vmax, float(np.nanmax(ref_plot.to_numpy(dtype=np.float64))))

    sns.heatmap(
        fx_plot,
        ax=axes_list[0],
        cmap="YlGnBu",
        vmin=0.0,
        vmax=max(10.0, vmax),
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        square=square,
        cbar=not has_ref,
        cbar_kws={"label": "Frequency (%)"} if not has_ref else None,
        annot_kws={"fontsize": 7.5},
    )
    axes_list[0].set_title(title_left, fontsize=9.0, fontweight="bold", pad=8)
    axes_list[0].set_xticklabels(axes_list[0].get_xticklabels(), rotation=xtick_rotation, ha="right", fontsize=7.8)
    axes_list[0].set_yticklabels(axes_list[0].get_yticklabels(), rotation=0, fontsize=7.6)

    if has_ref:
        sns.heatmap(
            ref_plot,
            ax=axes_list[1],
            cmap="YlGnBu",
            vmin=0.0,
            vmax=max(10.0, vmax),
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            square=square,
            cbar=True,
            cbar_kws={"label": "Frequency (%)"},
            annot_kws={"fontsize": 7.5},
        )
        axes_list[1].set_title(title_right, fontsize=9.0, fontweight="bold", pad=8)
        axes_list[1].set_xticklabels(axes_list[1].get_xticklabels(), rotation=xtick_rotation, ha="right", fontsize=7.8)
        axes_list[1].set_ylabel("")

    if suptitle:
        fig.suptitle(suptitle, fontsize=10.2, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_and_maybe_close(fig, output_path, dpi=dpi)
    return fig

