"""Clean-up Gating diagnostic plotting module for flow cytometry.

Generates publication-quality, Nature-styled multi-panel diagnostic figures
documenting the complete Core Clean-up Gating cascade (Debris, Dual Singlets,
Viability, and Dump Lineage Exclusion) conforming to MIFlowCyt standards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from flowx.gating.cleanup import CleanUpGatingResult
from flowx.gating.gate import PolygonGate, RangeGate
from flowx.io.parser import FCSData
from flowx.visualization.plotting import calculate_density_colors
from flowx.visualization.style import apply_nature_axis_style, set_nature_style

logger = logging.getLogger(__name__)


# Universal color palettes for cluster-aware gating visualization
GREEN_SHADES: List[str] = ["#2ca02c", "#1b9e77", "#31a354", "#006d2c", "#74c476", "#41ab5d"]
RED_SHADES: List[str] = ["#e53935", "#d32f2f", "#c62828", "#ef5350", "#b71c1c", "#ff7043"]


def _get_scatter_limits(fcs_data: FCSData, col: str, data: np.ndarray) -> Optional[Tuple[float, float]]:
    """Determine dynamic axis limits respecting instrument dynamic range ($PnR) while ignoring extreme electrical noise."""
    pnr = fcs_data.channel_ranges.get(col, None)
    if pnr is not None and pnr > 0:
        max_val = float(pnr)
        return (-0.02 * max_val, max_val * 1.05)
    elif len(data) > 0:
        p999 = float(np.percentile(data, 99.9))
        if p999 > 0:
            return (-0.02 * p999, p999 * 1.08)
    return None



def plot_cleanup_diagnostics(
    fcs_data: FCSData,
    cleanup_result: CleanUpGatingResult,
    df_events: Optional[pd.DataFrame] = None,
    df_transformed: Optional[pd.DataFrame] = None,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    max_plot_points: int = 25000,
    show_cluster_shades: bool = True,
) -> Path:
    """Generate publication-ready multi-panel Core Clean-up Gating diagnostic figure.

    Panel A: Scatter Morphology QC (FSC-A vs SSC-A) showing intact cells vs debris with polygon gate.
    Panel B: Forward Scatter Singlets (FSC-A vs FSC-H) showing singlet diagonal envelope.
    Panel C: Side Scatter Singlets (SSC-A vs SSC-H) showing side scatter doublet discrimination.
    Panel D: Viability Dye Gating showing live/dead threshold cutoff line on parent singlets.
    Panel E: Final Clean Target Cells vs Excluded Events overlay on morphological scatter.

    Args:
        fcs_data: Parsed FCSData object.
        cleanup_result: Populated CleanUpGatingResult instance.
        df_events: Linear events DataFrame (defaults to fcs_data.events_df).
        df_transformed: Transformed events DataFrame (defaults to df_events).
        output_path: Optional output image file path (defaults to outputs/cleanup/<sample>_cleanup.png).
        dpi: Output resolution.
        max_plot_points: Maximum events to subsample for responsive high-density rendering.
        show_cluster_shades: If True, renders individual intact cell/singlet clusters in distinctive
            shades of green with numbered centroid badges, and doublets in shades of red with
            numbered centroid badges to distinguish cluster origin. If False, displays standard binary green vs red.

    Returns:
        Resolved Path to saved figure.
    """
    set_nature_style()
    events = df_events if df_events is not None else fcs_data.events_df
    xformed = df_transformed if df_transformed is not None else events

    out_p = Path(output_path).resolve() if output_path else Path(f"outputs/cleanup/{cleanup_result.sample_name}_cleanup.png").resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    n_events = len(events)
    rng = np.random.RandomState(42)
    if n_events > max_plot_points:
        plot_idx = rng.choice(n_events, size=max_plot_points, replace=False)
    else:
        plot_idx = np.arange(n_events)

    events_sub = events.iloc[plot_idx]
    xformed_sub = xformed.iloc[plot_idx]

    # Check if dump channel was active
    has_dump = "Target" in cleanup_result.hierarchy.gates
    has_ssc_singlet = "Singlets_SSC" in cleanup_result.hierarchy.gates
    has_viab = "Viable" in cleanup_result.hierarchy.gates

    n_panels = 5 if not has_dump else 6
    fig = plt.figure(figsize=(15.0 if n_panels <= 5 else 18.0, 3.6), dpi=dpi)

    # -------------------------------------------------------------
    # Panel 1: Debris Exclusion (FSC vs SSC)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(1, n_panels, 1)
    cells_gate: PolygonGate = cleanup_result.hierarchy.get_gate("Cells")
    fsc_col = cells_gate.channels[0]
    ssc_col = cells_gate.channels[1]

    fsc_pts = events_sub[fsc_col].to_numpy(dtype=np.float64)
    ssc_pts = events_sub[ssc_col].to_numpy(dtype=np.float64)

    cluster_info = cleanup_result.cluster_info or getattr(cells_gate, "metadata", {}).get("cluster_info", None)
    sub_pred = None

    if cluster_info is not None and cluster_info.get("method") == "gmm":
        # Automated Clustering Visualization: discrete clusters + Gaussian confidence ellipses
        pred_labels = cluster_info.get("pred_labels")
        cell_clusters = cluster_info.get("cell_clusters", [])
        cluster_names = cluster_info.get("cluster_names", {})
        cluster_colors = cluster_info.get("cluster_colors", {})
        means = cluster_info.get("means", [])
        covariances = cluster_info.get("covariances", [])
        n_comp = cluster_info.get("n_components", len(means))
        debris_clusters = cluster_info.get("debris_clusters", [])

        sub_pred = pred_labels[plot_idx] if pred_labels is not None else np.zeros(len(plot_idx), dtype=int)

        # Universal population grouping: Debris / Low Scatter, Margin Saturation, Intact Cells
        n_cells_total = int(np.sum(cleanup_result.masks["Cells"]))
        pct_cells_total = (n_cells_total / cleanup_result.total_events * 100.0) if cleanup_result.total_events > 0 else 0.0

        # 1. Debris / Low Scatter
        debris_cids = [k for k, name in cluster_names.items() if name == "Debris / Low Scatter"]
        debris_mask_sub = np.isin(sub_pred, debris_cids)
        debris_count = int(np.sum(np.isin(pred_labels, debris_cids))) if pred_labels is not None else int(np.sum(debris_mask_sub))
        debris_pct = (debris_count / cleanup_result.total_events * 100.0) if cleanup_result.total_events > 0 else 0.0

        if np.any(debris_mask_sub):
            ax1.scatter(
                fsc_pts[debris_mask_sub],
                ssc_pts[debris_mask_sub],
                c="#7f7f7f",
                s=2.0,
                alpha=0.35,
                rasterized=True,
                edgecolors="none",
                label=f"Debris / Low Scatter: {debris_count:,} ({debris_pct:.1f}%)",
            )

        # 2. Margin Saturation
        sat_mask_sub = (sub_pred == -1)
        sat_count = int(np.sum(pred_labels == -1)) if pred_labels is not None else int(np.sum(sat_mask_sub))
        sat_pct = (sat_count / cleanup_result.total_events * 100.0) if cleanup_result.total_events > 0 else 0.0

        if np.any(sat_mask_sub):
            ax1.scatter(
                fsc_pts[sat_mask_sub],
                ssc_pts[sat_mask_sub],
                c="#e53935",
                s=2.0,
                alpha=0.45,
                rasterized=True,
                edgecolors="none",
                label=f"Margin Saturation: {sat_count:,} ({sat_pct:.1f}%)",
            )

        # 3. Universal Intact Cells
        cells_mask_sub = np.isin(sub_pred, cell_clusters) & (~sat_mask_sub) & (~debris_mask_sub)
        if np.any(cells_mask_sub):
            if show_cluster_shades and len(cell_clusters) > 1:
                for c_idx, cid in enumerate(cell_clusters):
                    c_mask = (sub_pred == cid) & (~sat_mask_sub) & (~debris_mask_sub)
                    if np.any(c_mask):
                        c_col = GREEN_SHADES[c_idx % len(GREEN_SHADES)]
                        ax1.scatter(
                            fsc_pts[c_mask],
                            ssc_pts[c_mask],
                            c=c_col,
                            s=2.0,
                            alpha=0.60,
                            rasterized=True,
                            edgecolors="none",
                            label=f"Intact Cells: {n_cells_total:,} ({pct_cells_total:.1f}%)" if c_idx == 0 else None,
                        )
            else:
                ax1.scatter(
                    fsc_pts[cells_mask_sub],
                    ssc_pts[cells_mask_sub],
                    c="#2ca02c",
                    s=2.0,
                    alpha=0.60,
                    rasterized=True,
                    edgecolors="none",
                    label=f"Intact Cells: {n_cells_total:,} ({pct_cells_total:.1f}%)",
                )

        # Draw 2-sigma Gaussian confidence ellipses for mixture components
        for k in range(len(means)):
            m = means[k]
            cov = covariances[k]
            is_cell = (k in cell_clusters)
            c_idx = cell_clusters.index(k) if is_cell else 0
            c_color = GREEN_SHADES[c_idx % len(GREEN_SHADES)] if (is_cell and show_cluster_shades) else ("#2ca02c" if is_cell else "#7f7f7f")
            try:
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2.0 * 2.0 * np.sqrt(np.maximum(vals, 1e-9))
                ell = patches.Ellipse(
                    xy=(m[0], m[1]),
                    width=width,
                    height=height,
                    angle=theta,
                    fill=False,
                    edgecolor=c_color,
                    linewidth=1.8 if is_cell else 1.2,
                    linestyle="-" if is_cell else "--",
                    zorder=6,
                )
                ax1.add_patch(ell)
            except Exception as e:
                logger.debug(f"Could not render GMM ellipse for component {k}: {e}")

        # Number badges at component centroids
        if show_cluster_shades and len(means) > 0:
            for k in range(len(means)):
                m = means[k]
                is_cell = (k in cell_clusters)
                c_idx = cell_clusters.index(k) if is_cell else 0
                badge_bg = GREEN_SHADES[c_idx % len(GREEN_SHADES)] if is_cell else ("#7f7f7f" if (k in debris_clusters or k in debris_cids) else "#e53935")
                ax1.text(
                    m[0], m[1], f"{k}",
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="circle,pad=0.22", fc=badge_bg, ec="white", lw=1.0, alpha=0.92),
                    zorder=10,
                )

        # Overlay Cells Gate Convex Hull boundary enclosing intact cell populations
        if cells_gate is not None and len(cells_gate.vertices) >= 3:
            poly_cells = patches.Polygon(
                cells_gate.vertices,
                closed=True,
                fill=False,
                edgecolor="#0d47a1",
                linewidth=1.4,
                linestyle="-",
                label="Extracted Cells Hull Gate",
                zorder=7,
            )
            ax1.add_patch(poly_cells)

        leg = ax1.legend(loc="upper left", fontsize=6.5, framealpha=0.88, handletextpad=0.2, markerscale=2.5)
        leg.get_frame().set_linewidth(0.5)

        n_cells = int(np.sum(cleanup_result.masks["Cells"]))
        pct_cells = (n_cells / cleanup_result.total_events * 100.0) if cleanup_result.total_events > 0 else 0.0
        ax1.set_title(f"A. Debris Exclusion (GMM)\nIntact: {n_cells:,} ({pct_cells:.1f}%)", fontsize=9.5, fontweight="bold")
    else:
        # Traditional manual/adaptive polygon gate visualization
        sub_x1, sub_y1, d_vals1 = calculate_density_colors(fsc_pts, ssc_pts, max_points=max_plot_points)
        ax1.scatter(sub_x1, sub_y1, c=d_vals1, cmap="viridis", s=1.5, alpha=0.6, rasterized=True, edgecolors="none")

        poly_cells = patches.Polygon(
            cells_gate.vertices,
            closed=True,
            fill=False,
            edgecolor="#d9534f",
            linewidth=1.5,
            linestyle="-",
            label=cells_gate.name,
        )
        ax1.add_patch(poly_cells)

        n_cells = int(np.sum(cleanup_result.masks["Cells"]))
        pct_cells = (n_cells / cleanup_result.total_events * 100.0) if cleanup_result.total_events > 0 else 0.0
        ax1.set_title(f"A. Debris Exclusion (Gate)\nIntact: {n_cells:,} ({pct_cells:.1f}%)", fontsize=9.5, fontweight="bold")

    ax1.set_xlabel(fcs_data.get_channel_label(fsc_col), fontsize=8.5)
    ax1.set_ylabel(fcs_data.get_channel_label(ssc_col), fontsize=8.5)
    xlim1 = _get_scatter_limits(fcs_data, fsc_col, fsc_pts)
    ylim1 = _get_scatter_limits(fcs_data, ssc_col, ssc_pts)
    if xlim1:
        ax1.set_xlim(xlim1)
    if ylim1:
        ax1.set_ylim(ylim1)
    ax1.set_box_aspect(1)
    apply_nature_axis_style(ax1)

    # -------------------------------------------------------------
    # Panel 2: FSC Singlets
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(1, n_panels, 2)
    if "Singlets_FSC" in cleanup_result.hierarchy.gates:
        sf_gate: PolygonGate = cleanup_result.hierarchy.get_gate("Singlets_FSC")
        sf_a_col = sf_gate.channels[0]
        sf_h_col = sf_gate.channels[1]

        # If pass-through (no height/width on cytometer)
        if sf_a_col == sf_h_col:
            ax2.text(
                0.5,
                0.5,
                f"Pass-Through Gating\n(No FSC Height/Width in {fcs_data.cytometer or 'FCS'})\nActive Singlets on SSC",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=8.5,
                fontweight="medium",
                color="#495057",
                bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#ced4da", lw=1.2),
            )
            n_sf = int(np.sum(cleanup_result.masks["Singlets_FSC"]))
            pct_sf = (n_sf / n_cells * 100.0) if n_cells > 0 else 0.0
            ax2.set_title(f"B. FSC Singlets (Pass-Through)\nRetained: {n_sf:,} (100.0%)", fontsize=9.5, fontweight="bold")
            ax2.set_xlabel(fcs_data.get_channel_label(sf_a_col), fontsize=8.5)
            ax2.set_ylabel(fcs_data.get_channel_label(sf_h_col), fontsize=8.5)
        else:
            # Discrete cluster visualization: Singlets vs Doublets/Aggregates
            cell_mask_sub = cleanup_result.masks["Cells"][plot_idx]
            n_sf = int(np.sum(cleanup_result.masks["Singlets_FSC"]))
            pct_sf = (n_sf / n_cells * 100.0) if n_cells > 0 else 0.0
            n_doublets = n_cells - n_sf
            pct_doublets = (n_doublets / n_cells * 100.0) if n_cells > 0 else 0.0

            if np.sum(cell_mask_sub) > 0:
                sf_a_all = events_sub[sf_a_col].to_numpy(dtype=np.float64)
                sf_h_all = events_sub[sf_h_col].to_numpy(dtype=np.float64)

                sf_a_pts = sf_a_all[cell_mask_sub]
                sf_h_pts = sf_h_all[cell_mask_sub]
                cell_cids = sub_pred[cell_mask_sub] if sub_pred is not None else None

                fsc_singlet_mask_sub = cleanup_result.masks["Singlets_FSC"][plot_idx]
                is_singlet = fsc_singlet_mask_sub[cell_mask_sub]
                is_doublet = ~is_singlet

                # 1. Doublets / Aggregates (Outliers)
                if np.any(is_doublet):
                    if show_cluster_shades and cell_cids is not None:
                        unique_d_cids = sorted(list(np.unique(cell_cids[is_doublet])))
                        for d_idx, cid in enumerate(unique_d_cids):
                            d_mask = is_doublet & (cell_cids == cid)
                            if np.any(d_mask):
                                d_col = RED_SHADES[d_idx % len(RED_SHADES)]
                                ax2.scatter(
                                    sf_a_pts[d_mask],
                                    sf_h_pts[d_mask],
                                    c=d_col,
                                    s=2.0,
                                    alpha=0.45,
                                    rasterized=True,
                                    edgecolors="none",
                                    label=f"Doublets: {n_doublets:,} ({pct_doublets:.1f}%)" if d_idx == 0 else None,
                                )
                                if np.sum(d_mask) >= 20:
                                    med_x = float(np.median(sf_a_pts[d_mask]))
                                    med_y = float(np.median(sf_h_pts[d_mask]))
                                    ax2.text(
                                        med_x, med_y, f"{cid}",
                                        ha="center", va="center",
                                        fontsize=7.2, fontweight="bold",
                                        color="white",
                                        bbox=dict(boxstyle="circle,pad=0.20", fc=d_col, ec="white", lw=0.9, alpha=0.92),
                                        zorder=10,
                                    )
                    else:
                        ax2.scatter(
                            sf_a_pts[is_doublet],
                            sf_h_pts[is_doublet],
                            c="#e53935",
                            s=2.0,
                            alpha=0.45,
                            rasterized=True,
                            edgecolors="none",
                            label=f"Doublets: {n_doublets:,} ({pct_doublets:.1f}%)",
                        )

                # 2. Singlets
                if np.any(is_singlet):
                    if show_cluster_shades and cell_cids is not None:
                        unique_s_cids = sorted(list(np.unique(cell_cids[is_singlet])))
                        for s_idx, cid in enumerate(unique_s_cids):
                            s_mask = is_singlet & (cell_cids == cid)
                            if np.any(s_mask):
                                s_col = GREEN_SHADES[s_idx % len(GREEN_SHADES)]
                                ax2.scatter(
                                    sf_a_pts[s_mask],
                                    sf_h_pts[s_mask],
                                    c=s_col,
                                    s=2.0,
                                    alpha=0.60,
                                    rasterized=True,
                                    edgecolors="none",
                                    label=f"Singlets: {n_sf:,} ({pct_sf:.1f}%)" if s_idx == 0 else None,
                                )
                                if np.sum(s_mask) >= 20:
                                    med_x = float(np.median(sf_a_pts[s_mask]))
                                    med_y = float(np.median(sf_h_pts[s_mask]))
                                    ax2.text(
                                        med_x, med_y, f"{cid}",
                                        ha="center", va="center",
                                        fontsize=7.2, fontweight="bold",
                                        color="white",
                                        bbox=dict(boxstyle="circle,pad=0.20", fc=s_col, ec="white", lw=0.9, alpha=0.92),
                                        zorder=10,
                                    )
                    else:
                        ax2.scatter(
                            sf_a_pts[is_singlet],
                            sf_h_pts[is_singlet],
                            c="#2ca02c",
                            s=2.0,
                            alpha=0.60,
                            rasterized=True,
                            edgecolors="none",
                            label=f"Singlets: {n_sf:,} ({pct_sf:.1f}%)",
                        )

            poly_sf = patches.Polygon(
                sf_gate.vertices,
                closed=True,
                fill=False,
                edgecolor="#0d47a1",
                linewidth=1.4,
                linestyle="-",
                label="Singlet Gate",
                zorder=7,
            )
            ax2.add_patch(poly_sf)
            leg2 = ax2.legend(loc="upper left", fontsize=6.5, framealpha=0.88, handletextpad=0.2, markerscale=2.5)
            leg2.get_frame().set_linewidth(0.5)

            sf_meta = getattr(sf_gate, "metadata", {})
            title_suffix = " (GMM Ratio)" if sf_meta.get("singlet_method") == "gmm_ratio" else ""
            ax2.set_title(f"B. FSC Singlets{title_suffix}\nSinglets: {n_sf:,} ({pct_sf:.1f}%)", fontsize=9.5, fontweight="bold")
            ax2.set_xlabel(fcs_data.get_channel_label(sf_a_col), fontsize=8.5)
            ax2.set_ylabel(fcs_data.get_channel_label(sf_h_col), fontsize=8.5)
            xlim2 = _get_scatter_limits(fcs_data, sf_a_col, sf_a_pts if "sf_a_pts" in locals() else events_sub[sf_a_col].to_numpy())
            ylim2 = _get_scatter_limits(fcs_data, sf_h_col, sf_h_pts if "sf_h_pts" in locals() else events_sub[sf_h_col].to_numpy())
            if xlim2:
                ax2.set_xlim(xlim2)
            if ylim2:
                ax2.set_ylim(ylim2)
    else:
        ax2.text(
            0.5,
            0.5,
            "FSC Singlet Gating\nNot Active / Skipped",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=8.5,
            fontweight="medium",
            color="#495057",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#ced4da", lw=1.2),
        )
        ax2.set_title("B. FSC Singlets (Skipped)", fontsize=9.5, fontweight="bold")
    ax2.set_box_aspect(1)
    apply_nature_axis_style(ax2)

    # -------------------------------------------------------------
    # Panel 3: SSC Singlets
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(1, n_panels, 3)
    if has_ssc_singlet:
        ss_gate: PolygonGate = cleanup_result.hierarchy.get_gate("Singlets_SSC")
        ss_a_col = ss_gate.channels[0]
        ss_h_col = ss_gate.channels[1]

        parent_key = "Singlets_FSC" if "Singlets_FSC" in cleanup_result.masks else "Cells"
        parent_ssc_mask = cleanup_result.masks[parent_key][plot_idx]
        n_parent_ss = int(np.sum(cleanup_result.masks[parent_key]))
        n_ss = int(np.sum(cleanup_result.masks["Singlets_SSC"]))
        pct_ss = (n_ss / n_parent_ss * 100.0) if n_parent_ss > 0 else 0.0
        n_doublets_ss = n_parent_ss - n_ss
        pct_doublets_ss = (n_doublets_ss / n_parent_ss * 100.0) if n_parent_ss > 0 else 0.0

        if ss_a_col == ss_h_col:
            ax3.text(
                0.5,
                0.5,
                f"Pass-Through Gating\n(No SSC Height/Width in {fcs_data.cytometer or 'FCS'})",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=8.5,
                fontweight="medium",
                color="#495057",
                bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#ced4da", lw=1.2),
            )
            ax3.set_title(f"C. SSC Singlets (Pass-Through)\nRetained: {n_ss:,} (100.0%)", fontsize=9.5, fontweight="bold")
            ax3.set_xlabel(fcs_data.get_channel_label(ss_a_col), fontsize=8.5)
            ax3.set_ylabel(fcs_data.get_channel_label(ss_h_col), fontsize=8.5)
        else:
            if np.sum(parent_ssc_mask) > 0:
                ss_a_all = events_sub[ss_a_col].to_numpy(dtype=np.float64)
                ss_h_all = events_sub[ss_h_col].to_numpy(dtype=np.float64)

                ss_a_pts = ss_a_all[parent_ssc_mask]
                ss_h_pts = ss_h_all[parent_ssc_mask]
                parent_cids = sub_pred[parent_ssc_mask] if sub_pred is not None else None

                ssc_singlet_mask_sub = cleanup_result.masks["Singlets_SSC"][plot_idx]
                is_singlet_ss = ssc_singlet_mask_sub[parent_ssc_mask]
                is_doublet_ss = ~is_singlet_ss

                # 1. Doublets / Aggregates
                if np.any(is_doublet_ss):
                    if show_cluster_shades and parent_cids is not None:
                        unique_d_cids = sorted(list(np.unique(parent_cids[is_doublet_ss])))
                        for d_idx, cid in enumerate(unique_d_cids):
                            d_mask = is_doublet_ss & (parent_cids == cid)
                            if np.any(d_mask):
                                d_col = RED_SHADES[d_idx % len(RED_SHADES)]
                                ax3.scatter(
                                    ss_a_pts[d_mask],
                                    ss_h_pts[d_mask],
                                    c=d_col,
                                    s=2.0,
                                    alpha=0.45,
                                    rasterized=True,
                                    edgecolors="none",
                                    label=f"Doublets: {n_doublets_ss:,} ({pct_doublets_ss:.1f}%)" if d_idx == 0 else None,
                                )
                                if np.sum(d_mask) >= 20:
                                    med_x = float(np.median(ss_a_pts[d_mask]))
                                    med_y = float(np.median(ss_h_pts[d_mask]))
                                    ax3.text(
                                        med_x, med_y, f"{cid}",
                                        ha="center", va="center",
                                        fontsize=7.2, fontweight="bold",
                                        color="white",
                                        bbox=dict(boxstyle="circle,pad=0.20", fc=d_col, ec="white", lw=0.9, alpha=0.92),
                                        zorder=10,
                                    )
                    else:
                        ax3.scatter(
                            ss_a_pts[is_doublet_ss],
                            ss_h_pts[is_doublet_ss],
                            c="#e53935",
                            s=2.0,
                            alpha=0.45,
                            rasterized=True,
                            edgecolors="none",
                            label=f"Doublets: {n_doublets_ss:,} ({pct_doublets_ss:.1f}%)",
                        )

                # 2. Singlets
                if np.any(is_singlet_ss):
                    if show_cluster_shades and parent_cids is not None:
                        unique_s_cids = sorted(list(np.unique(parent_cids[is_singlet_ss])))
                        for s_idx, cid in enumerate(unique_s_cids):
                            s_mask = is_singlet_ss & (parent_cids == cid)
                            if np.any(s_mask):
                                s_col = GREEN_SHADES[s_idx % len(GREEN_SHADES)]
                                ax3.scatter(
                                    ss_a_pts[s_mask],
                                    ss_h_pts[s_mask],
                                    c=s_col,
                                    s=2.0,
                                    alpha=0.60,
                                    rasterized=True,
                                    edgecolors="none",
                                    label=f"Singlets: {n_ss:,} ({pct_ss:.1f}%)" if s_idx == 0 else None,
                                )
                                if np.sum(s_mask) >= 20:
                                    med_x = float(np.median(ss_a_pts[s_mask]))
                                    med_y = float(np.median(ss_h_pts[s_mask]))
                                    ax3.text(
                                        med_x, med_y, f"{cid}",
                                        ha="center", va="center",
                                        fontsize=7.2, fontweight="bold",
                                        color="white",
                                        bbox=dict(boxstyle="circle,pad=0.20", fc=s_col, ec="white", lw=0.9, alpha=0.92),
                                        zorder=10,
                                    )
                    else:
                        ax3.scatter(
                            ss_a_pts[is_singlet_ss],
                            ss_h_pts[is_singlet_ss],
                            c="#2ca02c",
                            s=2.0,
                            alpha=0.60,
                            rasterized=True,
                            edgecolors="none",
                            label=f"Singlets: {n_ss:,} ({pct_ss:.1f}%)",
                        )

            poly_ss = patches.Polygon(
                ss_gate.vertices,
                closed=True,
                fill=False,
                edgecolor="#0d47a1",
                linewidth=1.4,
                linestyle="-",
                label="Singlet Gate",
                zorder=7,
            )
            ax3.add_patch(poly_ss)
            leg3 = ax3.legend(loc="upper left", fontsize=6.5, framealpha=0.88, handletextpad=0.2, markerscale=2.5)
            leg3.get_frame().set_linewidth(0.5)

            ss_meta = getattr(ss_gate, "metadata", {})
            title_suffix = " (GMM Ratio)" if ss_meta.get("singlet_method") == "gmm_ratio" else ""
            ax3.set_title(f"C. SSC Singlets{title_suffix}\nSinglets: {n_ss:,} ({pct_ss:.1f}%)", fontsize=9.5, fontweight="bold")
            ax3.set_xlabel(fcs_data.get_channel_label(ss_a_col), fontsize=8.5)
            ax3.set_ylabel(fcs_data.get_channel_label(ss_h_col), fontsize=8.5)
            xlim3 = _get_scatter_limits(fcs_data, ss_a_col, ss_a_pts if "ss_a_pts" in locals() else events_sub[ss_a_col].to_numpy())
            ylim3 = _get_scatter_limits(fcs_data, ss_h_col, ss_h_pts if "ss_h_pts" in locals() else events_sub[ss_h_col].to_numpy())
            if xlim3:
                ax3.set_xlim(xlim3)
            if ylim3:
                ax3.set_ylim(ylim3)
    else:
        ax3.text(
            0.5,
            0.5,
            "SSC Singlet Gating\nNot Active / Skipped",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=8.5,
            fontweight="medium",
            color="#495057",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#ced4da", lw=1.2),
        )
        ax3.set_title("C. SSC Singlets (Skipped)", fontsize=9.5, fontweight="bold")
    ax3.set_box_aspect(1)
    apply_nature_axis_style(ax3)

    # -------------------------------------------------------------
    # Panel 4: Viability Gating
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(1, n_panels, 4)
    if has_viab:
        vg: RangeGate = cleanup_result.hierarchy.get_gate("Viable")
        v_ch = vg.channels[0]
        v_parent = vg.parent_gate or "Singlets_SSC"
        parent_v_mask = cleanup_result.masks.get(v_parent, cleanup_result.masks["Cells"])[plot_idx]

        v_vals = xformed_sub.loc[parent_v_mask, v_ch].dropna().to_numpy()
        cutoff = vg.max_val if vg.max_val is not None else cleanup_result.cutoffs.get("viability_cutoff", 0.4)

        # Scale safety check: If cutoff was calculated in Logicle unit space (e.g. <= 2.0)
        # but v_vals are in un-transformed linear space (span > 10.0), transform v_vals
        # so histogram and cutoff are in the exact same coordinate system
        if len(v_vals) > 0 and -1.0 <= cutoff <= 2.0:
            v_span = float(np.percentile(v_vals, 99.5) - np.percentile(v_vals, 0.5)) if len(v_vals) >= 10 else float(np.max(v_vals) - np.min(v_vals))
            if v_span > 10.0:
                try:
                    from flowx.math.transforms import LogicleTransform
                    logicle = LogicleTransform()
                    v_vals = logicle.transform(v_vals)
                except Exception:
                    pass

        if len(v_vals) > 10:
            # 1D KDE density plot
            ax4.hist(v_vals, bins=60, density=True, color="#4575b4", alpha=0.45, edgecolor="none")
            try:
                kde = gaussian_kde(v_vals)
                x_eval = np.linspace(float(np.min(v_vals)), float(np.max(v_vals)), 200)
                ax4.plot(x_eval, kde(x_eval), color="#2b5c8f", lw=1.5)
            except Exception:
                pass

        ax4.axvline(cutoff, color="#d9534f", linestyle="--", linewidth=1.5, label=f"Cutoff ({cutoff:.2f})")
        ax4.legend(loc="upper right", fontsize=7.0, framealpha=0.88)

        n_viab = int(np.sum(cleanup_result.masks["Viable"]))
        n_parent_v = int(np.sum(cleanup_result.masks.get(v_parent, cleanup_result.masks["Cells"])))
        pct_viab = (n_viab / n_parent_v * 100.0) if n_parent_v > 0 else 0.0
        ax4.set_title(f"D. Viability Exclusion\nLive: {n_viab:,} ({pct_viab:.1f}%)", fontsize=9.5, fontweight="bold")
        ax4.set_xlabel(f"{fcs_data.get_channel_label(v_ch)} (Transformed)", fontsize=8.5)
        ax4.set_ylabel("Density", fontsize=8.5)
    else:
        ax4.text(
            0.5,
            0.5,
            "Viability Gating\nNot Configured",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=8.5,
            fontweight="medium",
            color="#495057",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#ced4da", lw=1.2),
        )
        ax4.set_title("D. Viability Exclusion\n(Not Configured / Skipped)", fontsize=9.5, fontweight="bold")
    ax4.set_box_aspect(1)
    apply_nature_axis_style(ax4)

    # -------------------------------------------------------------
    # Optional Panel 5: Dump Channel Gating
    # -------------------------------------------------------------
    panel_idx = 5
    if has_dump:
        ax_dump = fig.add_subplot(1, n_panels, panel_idx)
        dg: RangeGate = cleanup_result.hierarchy.get_gate("Target")
        d_ch = dg.channels[0]
        parent_d = dg.parent_gate or "Viable"
        parent_d_mask = cleanup_result.masks.get(parent_d, cleanup_result.masks["Cells"])[plot_idx]

        d_vals = xformed_sub.loc[parent_d_mask, d_ch].dropna().to_numpy()
        d_cutoff = dg.max_val if dg.max_val is not None else 0.35

        # Scale safety check: If d_cutoff is in Logicle unit space but d_vals are linear
        if len(d_vals) > 0 and -1.0 <= d_cutoff <= 2.0:
            d_span = float(np.percentile(d_vals, 99.5) - np.percentile(d_vals, 0.5)) if len(d_vals) >= 10 else float(np.max(d_vals) - np.min(d_vals))
            if d_span > 10.0:
                try:
                    from flowx.math.transforms import LogicleTransform
                    logicle = LogicleTransform()
                    d_vals = logicle.transform(d_vals)
                except Exception:
                    pass

        if len(d_vals) > 10:
            ax_dump.hist(d_vals, bins=60, density=True, color="#74add1", alpha=0.45, edgecolor="none")
            try:
                kde = gaussian_kde(d_vals)
                x_eval = np.linspace(float(np.min(d_vals)), float(np.max(d_vals)), 200)
                ax_dump.plot(x_eval, kde(x_eval), color="#313695", lw=1.5)
            except Exception:
                pass

        ax_dump.axvline(d_cutoff, color="#d9534f", linestyle="--", linewidth=1.5, label="Dump Cutoff")

        n_target = int(np.sum(cleanup_result.masks["Target"]))
        n_parent_d = int(np.sum(cleanup_result.masks.get(parent_d, cleanup_result.masks["Cells"])))
        pct_target = (n_target / n_parent_d * 100.0) if n_parent_d > 0 else 0.0
        ax_dump.set_title(f"E. Dump Lineage Exclusion\nTarget: {n_target:,} ({pct_target:.1f}%)", fontsize=9.5, fontweight="bold")
        ax_dump.set_xlabel(f"{fcs_data.get_channel_label(d_ch)} (Transformed)", fontsize=8.5)
        ax_dump.set_ylabel("Density", fontsize=8.5)
        ax_dump.set_box_aspect(1)
        apply_nature_axis_style(ax_dump)
        panel_idx += 1

    # -------------------------------------------------------------
    # Final Panel: Final Clean Population vs Excluded Events
    # -------------------------------------------------------------
    ax_final = fig.add_subplot(1, n_panels, panel_idx)
    clean_sub_mask = cleanup_result.final_clean_mask[plot_idx]
    excluded_sub_mask = ~clean_sub_mask

    # Background excluded events in light gray
    if np.sum(excluded_sub_mask) > 0:
        ax_final.scatter(
            fsc_pts[excluded_sub_mask],
            ssc_pts[excluded_sub_mask],
            c="#cfd8dc",
            s=1.0,
            alpha=0.35,
            rasterized=True,
            label="Excluded",
            edgecolors="none",
        )

    # Clean viable singlets in bright royal blue / green
    if np.sum(clean_sub_mask) > 0:
        ax_final.scatter(
            fsc_pts[clean_sub_mask],
            ssc_pts[clean_sub_mask],
            c="#1f77b4",
            s=1.5,
            alpha=0.6,
            rasterized=True,
            label="Clean Singlets",
            edgecolors="none",
        )

    title_letter = "F" if has_dump else "E"
    ax_final.set_title(
        f"{title_letter}. Clean Target Population\nClean: {cleanup_result.clean_event_count:,} ({cleanup_result.clean_event_pct:.1f}%)",
        fontsize=9.5,
        fontweight="bold",
    )
    ax_final.set_xlabel(fcs_data.get_channel_label(fsc_col), fontsize=8.5)
    ax_final.set_ylabel(fcs_data.get_channel_label(ssc_col), fontsize=8.5)
    if xlim1:
        ax_final.set_xlim(xlim1)
    if ylim1:
        ax_final.set_ylim(ylim1)
    ax_final.set_box_aspect(1)
    apply_nature_axis_style(ax_final)

    plt.tight_layout()
    fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved clean-up gating diagnostic plot to '{out_p}'")
    return out_p
