"""Flow Cytometrist Compensation Diagnostic & Verification Visualization Suite.

Provides publication-ready visual diagnostics to verify every stage of fluorescence
compensation across single-stained controls and biological samples:
0. Single-Stain Control Scatter, Debris & Singlet Pre-Gating Audit (`plot_control_scatter_singlet_audit_grid`)
1. Single-Stain Control Fluorescence Gating & Carrier Audit Grid (`plot_control_gating_audit_grid`)
2. Post-Compensation Residual Median Alignment & Binned Tracking (`plot_residual_alignment_diagnostics`)
3. Laser-Grouped Spillover Matrix & Instrument Difference Heatmap (`plot_spillover_and_diff_matrix`)
4. Nguyen et al. (2013) Spillover Spreading Matrix Heatmap (`plot_spillover_spreading_matrix`)
5. Biological Sample Top-Spillover Pairs Before/After Gallery (`plot_sample_top_pairs_gallery`)
6. Spillover Estimator Comparison (`plot_estimator_comparison`)
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flowx.math.compensation import CompensationControlResult, CompensationMatrix
from flowx.math.transforms import LogicleTransform
from flowx.visualization.plotting import calculate_density_colors
from flowx.visualization.style import (
    apply_nature_axis_style,
    format_multiline_title,
    set_nature_style,
)

set_nature_style()


def _extract_laser_em_sort_key(ch: str) -> Tuple[int, int, str]:
    """Extract (excitation_nm, emission_nm, channel_str) for logical laser ordering."""
    ex_m = re.search(r"Ex\s*(\d{3})", ch, re.IGNORECASE)
    em_m = re.search(r"Em\s*(\d{3})", ch, re.IGNORECASE)
    ex_val = int(ex_m.group(1)) if ex_m else 999
    em_val = int(em_m.group(1)) if em_m else 999
    return (ex_val, em_val, ch)


def _format_compact_channel_label(
    ch: str,
    marker_map: Optional[Dict[str, str]] = None,
    include_detector: bool = True,
) -> str:
    """Create a compact, readable label combining biological marker and Ex/Em wavelength."""
    short_det = (
        ch.replace("Ex ", "")
        .replace("Em ", "")
        .replace(": ", ":")
        .replace("-A", "")
        .replace("628/594", "628")
    )
    if marker_map and ch in marker_map and marker_map[ch]:
        m_str = marker_map[ch].strip()
        if include_detector:
            return f"{m_str} [{short_det}]"
        return m_str
    return short_det


def _format_multiline_channel_label(
    ch: str,
    marker_map: Optional[Dict[str, str]] = None,
    max_width: int = 14,
    include_detector: bool = True,
) -> str:
    """Format a channel and biological marker into a balanced multi-line label.

    Dynamically wraps biological markers and hardware detector names across multiple lines
    to prevent horizontal collisions on compact x-axes and vertical crowding on y-axes:
    1. If a biological marker is mapped:
       - Wraps marker text across lines respecting word boundaries (<= max_width chars).
       - Places the detector bracket (e.g., '[PE-CF594]') on a separate bottom line.
    2. If no biological marker is mapped:
       - Cleans hardware formatting (-A suffix, redundant prefixes) and wraps detector
         across lines on word boundaries or underscores (<= max_width chars).
    """
    import textwrap

    short_det = (
        ch.replace("Ex ", "")
        .replace("Em ", "")
        .replace(": ", ":")
        .replace("-A", "")
        .replace("628/594", "628")
    )
    m_str = marker_map.get(ch, "").strip() if marker_map else ""
    if m_str:
        m_lines = textwrap.wrap(m_str, width=max_width, break_long_words=False)
        if not m_lines:
            m_lines = [m_str]
        if include_detector:
            det_str = f"[{short_det}]"
            if len(det_str) > max_width:
                d_clean = short_det.replace("_", " ")
                d_wrapped = textwrap.wrap(d_clean, width=max(8, max_width - 2), break_long_words=False)
                det_lines = [f"[{w}]" if idx == 0 else f" {w}" for idx, w in enumerate(d_wrapped)]
                return "\n".join(m_lines + det_lines)
            return "\n".join(m_lines + [det_str])
        return "\n".join(m_lines)
    else:
        det_clean = short_det.replace("_", " ")
        lines = textwrap.wrap(det_clean, width=max_width, break_long_words=False)
        return "\n".join(lines) if lines else short_det


def _get_laser_ordered_indices(
    channels: List[str],
) -> Tuple[List[int], List[str], List[int]]:
    """Order channels by excitation laser and emission wavelength, returning indices and laser boundaries."""
    indexed = [(i, ch, _extract_laser_em_sort_key(ch)) for i, ch in enumerate(channels)]
    indexed.sort(key=lambda x: x[2])
    ordered_idx = [x[0] for x in indexed]
    ordered_ch = [x[1] for x in indexed]

    boundaries: List[int] = []
    prev_ex = None
    for pos, (_, _, (ex_nm, _, _)) in enumerate(indexed):
        if prev_ex is not None and ex_nm != prev_ex:
            boundaries.append(pos)
        prev_ex = ex_nm
    return ordered_idx, ordered_ch, boundaries


def plot_control_scatter_singlet_audit_grid(
    comp_result: CompensationControlResult,
    marker_map: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    ncols: int = 7,
    dpi: int = 300,
) -> plt.Figure:
    """Plot a multi-panel FSC-A vs SSC-A pre-gating audit across all single-stained controls.

    Verifies that sub-cellular debris, detector-saturated events, and bead dimers / cell doublets
    are strictly excluded prior to positive/negative fluorescence population gating:
      - Grey points: Excluded sub-cellular debris / off-scale detector saturation.
      - Amber points: Excluded bead dimers/multimers or cell doublets (FSC-H/FSC-A & SSC-H/SSC-A).
      - Emerald (Beads) / Royal Blue (Cells) points: High-quality monomer beads or intact singlet cells
        retained for spillover calculation.
      - Dashed Gate Overlay: 2D monomer density ellipse (beads) or intact leukocyte morphology boundary (cells).

    Args:
        comp_result: Populated CompensationControlResult containing control_audit_records.
        marker_map: Optional mapping of detector name to biological marker label.
        output_path: Optional file path to save the figure.
        ncols: Number of columns in the grid (default: 7 for 28 controls -> 4x7 grid).
        dpi: Export resolution.

    Returns:
        Matplotlib Figure object.
    """
    set_nature_style()
    records = comp_result.control_audit_records
    channels = list(records.keys())
    _, ordered_ch, _ = _get_laser_ordered_indices(channels)

    n_panels = len(ordered_ch)
    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No Control Audit Records Available", ha="center", va="center")
        return fig

    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.42, nrows * 2.68), dpi=dpi)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, ch in enumerate(ordered_ch):
        ax = axes_flat[idx]
        ax.set_box_aspect(1)
        rec = records[ch]
        st = rec["stats"]
        carrier = st.get("carrier_type", "beads").upper()
        geom = rec.get("gate_geometry") or {}

        raw_fsc = rec.get("raw_fsc_vals")
        raw_ssc = rec.get("raw_ssc_vals")
        deb_mask = rec.get("raw_debris_mask")
        dbl_mask = rec.get("raw_doublet_mask")
        cln_mask = rec.get("raw_clean_mask")

        if raw_fsc is not None and raw_ssc is not None and len(raw_fsc) > 0:
            fsc_k = raw_fsc / 1000.0
            ssc_k = raw_ssc / 1000.0

            if deb_mask is not None and np.any(deb_mask):
                ax.scatter(
                    fsc_k[deb_mask],
                    ssc_k[deb_mask],
                    c="#90a4ae",
                    s=2.0,
                    alpha=0.45,
                    edgecolors="none",
                    label="Debris/Sat",
                    rasterized=True,
                )
            if dbl_mask is not None and np.any(dbl_mask):
                ax.scatter(
                    fsc_k[dbl_mask],
                    ssc_k[dbl_mask],
                    c="#ff8f00",
                    s=2.4,
                    alpha=0.55,
                    edgecolors="none",
                    label="Doublets/Dimers",
                    rasterized=True,
                )
            if cln_mask is not None and np.any(cln_mask):
                clean_color = "#1565c0" if carrier == "CELLS" else "#2e7d32"
                ax.scatter(
                    fsc_k[cln_mask],
                    ssc_k[cln_mask],
                    c=clean_color,
                    s=2.8,
                    alpha=0.68,
                    edgecolors="none",
                    label="Clean Singlets",
                    rasterized=True,
                )

            # Overlay gate boundary
            g_type = geom.get("gate_type", "")
            if g_type == "ellipse":
                cx = float(geom.get("center_fsc", 0.0)) / 1000.0
                cy = float(geom.get("center_ssc", 0.0)) / 1000.0
                rx = float(geom.get("radius_fsc", 10000.0)) / 1000.0
                ry = float(geom.get("radius_ssc", 10000.0)) / 1000.0
                theta = np.linspace(0.0, 2.0 * np.pi, 120)
                ax.plot(
                    cx + rx * np.cos(theta),
                    cy + ry * np.sin(theta),
                    color="#1b5e20",
                    linestyle="--",
                    linewidth=1.25,
                    zorder=8,
                )
                ax.plot(cx, cy, marker="+", color="#1b5e20", markersize=5.5, markeredgewidth=1.2, zorder=9)
            elif g_type == "cell_box":
                f_lo = float(geom.get("fsc_lo", 16000.0)) / 1000.0
                s_lo = float(geom.get("ssc_lo", 7000.0)) / 1000.0
                ax.axvline(f_lo, color="#b71c1c", linestyle="--", linewidth=1.15, alpha=0.85, zorder=8)
                ax.axhline(s_lo, color="#b71c1c", linestyle="--", linewidth=1.15, alpha=0.85, zorder=8)

            x_max = min(265.0, max(120.0, float(np.nanpercentile(fsc_k, 99.2)) * 1.15))
            y_max = min(265.0, max(120.0, float(np.nanpercentile(ssc_k, 99.2)) * 1.18))
            ax.set_xlim(-5.0, x_max)
            ax.set_ylim(-5.0, y_max)
        else:
            ax.text(0.5, 0.5, "Scatter Unavailable", ha="center", va="center", fontsize=7.0)

        lbl = _format_compact_channel_label(ch, marker_map, include_detector=True)
        ax.set_title(lbl, fontsize=7.5, fontweight="bold", pad=4)
        ax.set_xlabel("FSC-A (×10³)", fontsize=6.8)
        if idx % ncols == 0:
            ax.set_ylabel("SSC-A (×10³)", fontsize=6.8)
        apply_nature_axis_style(ax)
        ax.tick_params(labelsize=6.5)

        badge_bg = "#e3f2fd" if carrier == "CELLS" else "#e8f5e9"
        badge_ec = "#1565c0" if carrier == "CELLS" else "#2e7d32"
        mode_tag = "Intact Cells" if carrier == "CELLS" else "Monomer Beads"
        deb_pct = float(st.get("debris_removed_pct", 0.0))
        dbl_pct = float(st.get("doublets_removed_pct", 0.0))
        kept_pct = float(st.get("scatter_singlet_kept_pct", 100.0))
        n_clean = int(st.get("n_clean_events", st.get("n_events", 0)))

        info_txt = (
            f"[{carrier} | {mode_tag}]\n"
            f"Debris: {deb_pct:.1f}% | Dblt: {dbl_pct:.1f}%\n"
            f"Clean: {n_clean:,} ({kept_pct:.1f}%)"
        )
        ax.text(
            0.04,
            0.95,
            info_txt,
            transform=ax.transAxes,
            fontsize=5.9,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=badge_bg,
                edgecolor=badge_ec,
                alpha=0.90,
                linewidth=0.7,
            ),
        )

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        "Single-Stained Compensation Controls: Scatter Morphology, Debris Exclusion & Singlet Pre-Gating Audit",
        fontsize=11.0,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout(pad=0.7, h_pad=1.05, w_pad=0.65)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_control_gating_audit_grid(
    comp_result: CompensationControlResult,
    marker_map: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    ncols: int = 7,
    dpi: int = 300,
) -> plt.Figure:
    """Plot a multi-panel grid auditing population gating, carrier detection, and Time QC across all controls.

    Displays each single-stained control's asinh-transformed primary fluorescence distribution
    with shaded Negative (slate blue) and Positive (emerald/coral) gates, median lines,
    carrier badge ([BEAD] vs [CELL]), Time QC pass rate, and Stain Index / SNR.

    Args:
        comp_result: Populated CompensationControlResult containing control_audit_records.
        marker_map: Optional mapping of detector name to marker label.
        output_path: Optional file path to save the figure.
        ncols: Number of columns in the grid (default: 7 for 28 controls -> 4x7 grid).
        dpi: Export resolution.

    Returns:
        Matplotlib Figure object.
    """
    set_nature_style()
    records = comp_result.control_audit_records
    channels = list(records.keys())
    _, ordered_ch, _ = _get_laser_ordered_indices(channels)

    n_panels = len(ordered_ch)
    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No Control Audit Records Available", ha="center", va="center")
        return fig

    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.42, nrows * 2.68), dpi=dpi)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, ch in enumerate(ordered_ch):
        ax = axes_flat[idx]
        ax.set_box_aspect(1)
        rec = records[ch]
        st = rec["stats"]
        t_vals = rec["target_vals"]
        pos_mask = rec["pos_mask"]
        neg_mask = rec["neg_mask"]
        carrier = st.get("carrier_type", "beads").upper()
        est = "Reg" if "reg" in st.get("estimator_used", "") else "MFI"

        asinh_all = np.arcsinh(np.clip(t_vals / 150.0, -5.0, None))
        asinh_pos = asinh_all[pos_mask]
        asinh_neg = asinh_all[neg_mask]

        bins = np.linspace(
            float(np.nanpercentile(asinh_all, 0.5)) - 0.3,
            float(np.nanpercentile(asinh_all, 99.8)) + 0.3,
            65,
        )
        ax.hist(asinh_all, bins=bins, color="#cfd8dc", alpha=0.55, edgecolor="none")
        if len(asinh_neg) > 0:
            ax.hist(
                asinh_neg,
                bins=bins,
                color="#1f77b4",
                alpha=0.72,
                edgecolor="none",
                label=f"Neg ({len(asinh_neg):,})",
            )
            ax.axvline(float(np.nanmedian(asinh_neg)), color="#0d47a1", linestyle="--", linewidth=1.1)
        if len(asinh_pos) > 0:
            pos_color = "#d62728" if carrier == "CELLS" else "#2ca02c"
            ax.hist(
                asinh_pos,
                bins=bins,
                color=pos_color,
                alpha=0.78,
                edgecolor="none",
                label=f"Pos ({len(asinh_pos):,})",
            )
            ax.axvline(float(np.nanmedian(asinh_pos)), color="#1b5e20" if carrier != "CELLS" else "#8e0000", linestyle="-", linewidth=1.2)

        lbl = _format_compact_channel_label(ch, marker_map, include_detector=True)
        ax.set_title(lbl, fontsize=7.5, fontweight="bold", pad=4)
        ax.set_xlabel("asinh(Intensity / 150)", fontsize=6.8)
        if idx % ncols == 0:
            ax.set_ylabel("Events", fontsize=6.8)
        apply_nature_axis_style(ax)
        ax.tick_params(labelsize=6.5)

        badge_bg = "#fff3e0" if carrier == "CELLS" else "#e8f5e9"
        badge_ec = "#e65100" if carrier == "CELLS" else "#2e7d32"
        info_txt = (
            f"[{carrier} | {est} | Clean: {st.get('scatter_singlet_kept_pct', 100.0):.1f}%]\n"
            f"TimeQC: {st.get('time_qc_passed_pct', 100.0):.1f}% | SNR: {st.get('snr', 0.0):.0f}\n"
            f"ΔMFI: {st.get('delta_mfi', 0.0):,.0f}"
        )
        ax.text(
            0.04,
            0.95,
            info_txt,
            transform=ax.transAxes,
            fontsize=6.0,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=badge_bg,
                edgecolor=badge_ec,
                alpha=0.90,
                linewidth=0.7,
            ),
        )

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        "Single-Stained Compensation Controls: Automated Time QC, Carrier Detection & Population Gating Audit",
        fontsize=11.0,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout(pad=0.7, h_pad=1.05, w_pad=0.65)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_residual_alignment_diagnostics(
    comp_result: CompensationControlResult,
    marker_map: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot Post-Compensation Residual Median Alignment Heatmap and Top 6 Single-Stain Tracking Scatters.

    Panel A (Left): N x N heatmap of post-compensation Residual Z-Scores
        Z[i, j] = (Median(CompPos_j) - Median(CompNeg_j)) / rSD(CompNeg_j),
        proving whether positive and negative medians align across all N x (N-1) channel pairs.
    Panel B (Right, 2x3 grid): Compensated single-stained controls for the 6 highest-spillover
        pairs in Logicle scale, showing Negative population (blue), Positive population (orange),
        and Binned Median Alignment Curve.

    Args:
        comp_result: Populated CompensationControlResult.
        marker_map: Optional detector-to-marker dictionary.
        output_path: Optional path to export image.
        dpi: Resolution.

    Returns:
        Matplotlib Figure object.
    """
    from flowx.io.parser import parse_fcs

    set_nature_style()
    res_z_df = comp_result.residual_z_dataframe
    if res_z_df is None or res_z_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "Residual Z-Score Matrix Not Available", ha="center", va="center")
        return fig

    channels = list(res_z_df.index)
    ordered_idx, ordered_ch, boundaries = _get_laser_ordered_indices(channels)
    z_ordered = res_z_df.iloc[ordered_idx, ordered_idx].to_numpy(dtype=np.float64)
    short_labels = [_format_compact_channel_label(c, marker_map, include_detector=True) for c in ordered_ch]
    n_ch = len(ordered_ch)

    # Dynamic scaling based on channel count (< 10 channels auto-scales down)
    w_heat = max(3.8, min(7.5, 1.8 + n_ch * 0.45))
    w_scatter = max(5.2, min(8.2, 2.2 * 3))
    fig_w = w_heat + w_scatter + 0.6
    fig_h = max(4.2, min(7.6, max(3.5, 1.6 + n_ch * 0.45, 5.0)))

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = gridspec.GridSpec(1, 2, width_ratios=[w_heat, w_scatter], wspace=0.24)

    # Panel A: Residual Z-Score Matrix Heatmap
    ax_heat = fig.add_subplot(gs[0])
    im = ax_heat.imshow(z_ordered, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax_heat.set_xticks(range(n_ch))
    ax_heat.set_yticks(range(n_ch))
    tick_fs = max(5.8, min(8.2, 11.5 - n_ch * 0.45))
    ax_heat.set_xticklabels(short_labels, rotation=90, fontsize=tick_fs)
    ax_heat.set_yticklabels(short_labels, fontsize=tick_fs)
    ax_heat.set_xlabel("Secondary Detector Channel j (Evaluated After Compensation)", fontsize=8.0, fontweight="bold")
    ax_heat.set_ylabel("Primary Stained Control i", fontsize=8.0, fontweight="bold")

    for b in boundaries:
        ax_heat.axhline(b - 0.5, color="#333333", linewidth=0.9, linestyle="--", alpha=0.7)
        ax_heat.axvline(b - 0.5, color="#333333", linewidth=0.9, linestyle="--", alpha=0.7)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.042, pad=0.03)
    cbar.set_label("Post-Comp Residual Z-Score: (Med_Pos - Med_Neg) / rSD_Neg", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.8)

    off_diag = ~np.eye(n_ch, dtype=bool)
    abs_z_off = np.abs(z_ordered[off_diag])
    mean_abs_z = float(np.mean(abs_z_off)) if len(abs_z_off) > 0 else 0.0
    aligned_frac = float(np.mean(abs_z_off <= 0.5) * 100.0) if len(abs_z_off) > 0 else 100.0
    ax_heat.set_title(
        f"A. N×(N-1) Post-Compensation Residual Median Alignment Matrix\n"
        f"(Mean |Z| = {mean_abs_z:.3f} rSD | {aligned_frac:.1f}% of {int(np.sum(off_diag))} pairs within |Z| ≤ 0.5 rSD)",
        fontsize=8.8,
        fontweight="bold",
        pad=8,
    )
    ax_heat.grid(False)

    # Panel B: Top 6 Spillover Control Pairs with Binned Median Tracking Lines
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1], wspace=0.34, hspace=0.38)
    spill_mat = comp_result.dataframe.values.copy()
    np.fill_diagonal(spill_mat, 0.0)

    flat_indices = np.argsort(spill_mat.flatten())[::-1]
    top_pairs: List[Tuple[str, str, float]] = []
    for flat_idx in flat_indices:
        r_idx, c_idx = np.unravel_index(flat_idx, spill_mat.shape)
        val = float(spill_mat[r_idx, c_idx])
        if val <= 0.01:
            break
        top_pairs.append((comp_result.dataframe.index[r_idx], comp_result.dataframe.columns[c_idx], val))
        if len(top_pairs) >= 6:
            break

    # Sort the 6 selected pairs by consistent laser order (primary row, then secondary col)
    ch_order_map = {ch: pos for pos, ch in enumerate(ordered_ch)}
    top_pairs.sort(key=lambda x: (ch_order_map.get(x[0], 999), ch_order_map.get(x[1], 999)))

    xform = LogicleTransform()

    for p_idx, (ch_pri, ch_sec, spill_val) in enumerate(top_pairs):
        ax_sub = fig.add_subplot(gs_right[p_idx // 3, p_idx % 3])
        ax_sub.set_box_aspect(1)

        ctrl_file = comp_result.control_files_mapped.get(ch_pri)
        if not ctrl_file or not Path(ctrl_file).is_file():
            ax_sub.text(0.5, 0.5, "Control Unavailable", ha="center", va="center")
            continue

        ctrl_fcs = parse_fcs(ctrl_file)
        # Apply carrier-aware pre-gating so debris and saturated doublets don't clutter the scatter
        from flowx.math.compensation import gate_control_populations
        clean_ctrl_df, is_pos, is_neg, _ = gate_control_populations(
            df=ctrl_fcs.events_df,
            target_channel=ch_pri,
            max_range=ctrl_fcs.channel_ranges.get(ch_pri, 262144.0),
            all_channels=list(comp_result.dataframe.index),
        )
        comp_df = comp_result.matrix.apply_dataframe(clean_ctrl_df)

        x_lin = comp_df[ch_sec].to_numpy()
        y_lin = comp_df[ch_pri].to_numpy()

        # Adaptive symmetric arcsinh cofactor scaled to population spread so high-SSM tandem
        # channels (e.g. BV786 -> BV421) do not exhibit artificial biexponential valley splitting around 0
        pos_spread_x = float(np.nanpercentile(np.abs(x_lin), 80)) if len(x_lin) > 0 else 400.0
        cofactor_x = max(400.0, pos_spread_x * 0.85)
        cofactor_y = 400.0
        x_sym = np.arcsinh(x_lin / cofactor_x)
        y_sym = np.arcsinh(y_lin / cofactor_y)

        rng = np.random.RandomState(42 + p_idx)
        neg_all_idx = np.where(is_neg & np.isfinite(x_sym) & np.isfinite(y_sym))[0]
        pos_all_idx = np.where(is_pos & np.isfinite(x_sym) & np.isfinite(y_sym))[0]

        neg_idx = rng.choice(neg_all_idx, size=min(len(neg_all_idx), 1500), replace=False) if len(neg_all_idx) > 0 else neg_all_idx
        pos_idx = rng.choice(pos_all_idx, size=min(len(pos_all_idx), 1500), replace=False) if len(pos_all_idx) > 0 else pos_all_idx

        ax_sub.scatter(
            x_sym[neg_idx],
            y_sym[neg_idx],
            c="#1f77b4",
            s=3.5,
            alpha=0.45,
            edgecolors="none",
            label="Neg",
            rasterized=True,
        )
        ax_sub.scatter(
            x_sym[pos_idx],
            y_sym[pos_idx],
            c="#e65100",
            s=3.5,
            alpha=0.55,
            edgecolors="none",
            label="Pos",
            rasterized=True,
        )

        # Compute exact linear medians transformed into symmetric arcsinh coordinates
        neg_x_med = float(np.arcsinh(np.nanmedian(x_lin[neg_all_idx]) / cofactor_x)) if len(neg_all_idx) > 0 else 0.0
        pos_x_med = float(np.arcsinh(np.nanmedian(x_lin[pos_all_idx]) / cofactor_x)) if len(pos_all_idx) > 0 else neg_x_med
        neg_y_med = float(np.arcsinh(np.nanmedian(y_lin[neg_all_idx]) / cofactor_y)) if len(neg_all_idx) > 0 else 0.0
        pos_y_med = float(np.arcsinh(np.nanmedian(y_lin[pos_all_idx]) / cofactor_y)) if len(pos_all_idx) > 0 else 4.0

        # Vertical reference line at negative median
        ax_sub.axvline(neg_x_med, color="#111111", linestyle="--", linewidth=1.1, alpha=0.85)
        # Binned median tracking line connecting Neg centroid to Pos centroid
        ax_sub.plot(
            [neg_x_med, pos_x_med],
            [neg_y_med, pos_y_med],
            color="#d50000",
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            label="Median Track",
            zorder=10,
        )

        z_val = float(comp_result.residual_z_dataframe.loc[ch_pri, ch_sec])
        pct_val = (
            float(comp_result.residual_pct_dataframe.loc[ch_pri, ch_sec])
            if comp_result.residual_pct_dataframe is not None
            else 0.0
        )

        pri_lbl = _format_compact_channel_label(ch_pri, marker_map, include_detector=True)
        sec_lbl = _format_compact_channel_label(ch_sec, marker_map, include_detector=True)
        ax_sub.set_title(
            f"{pri_lbl}\n→ {sec_lbl}",
            fontsize=6.8,
            fontweight="bold",
            pad=4,
        )
        ax_sub.set_xlabel(f"{sec_lbl} (asinh/{cofactor_x:.0f})", fontsize=6.4)
        ax_sub.set_ylabel(f"{pri_lbl} (asinh/{cofactor_y:.0f})", fontsize=6.4)
        x_span = max(1.8, float(np.nanpercentile(np.abs(x_sym[np.concatenate([neg_all_idx, pos_all_idx])]), 98.5)) * 1.25)
        ax_sub.set_xlim(neg_x_med - x_span, neg_x_med + x_span)
        ax_sub.set_ylim(-1.0, max(5.5, pos_y_med + 1.2))
        apply_nature_axis_style(ax_sub)
        ax_sub.tick_params(labelsize=6.5)

        ax_sub.text(
            0.04,
            0.94,
            f"Spill: {spill_val*100:.1f}% | Res Δ: {pct_val:+.2f}%\n|Z|: {abs(z_val):.2f} rSD",
            transform=ax_sub.transAxes,
            fontsize=5.9,
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.88),
        )

    fig.suptitle(
        "Post-Compensation Verification: N×(N-1) Residual Median Alignment & Single-Stain Trajectory Check",
        fontsize=11.0,
        fontweight="bold",
        y=1.01,
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_spillover_and_diff_matrix(
    comp_result: CompensationControlResult,
    marker_map: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot side-by-side Laser-Grouped Active Spillover Matrix (%) and Instrument Difference Matrix.

    Panel A: Calculated Spillover Matrix (%) ordered by excitation laser wavelength with
        detector + biological marker labels, laser group dividers, and condition number kappa(S).
    Panel B: Concordance Difference Matrix (Calculated % - Instrument Embedded %) showing
        agreement with the cytometer's acquisition spillover matrix.

    Args:
        comp_result: Populated CompensationControlResult.
        marker_map: Optional detector-to-marker dictionary.
        output_path: Optional path to export image.
        dpi: Resolution.

    Returns:
        Matplotlib Figure object.
    """
    set_nature_style()
    spill_df = comp_result.dataframe
    diff_df = comp_result.diff_from_embedded_dataframe
    channels = list(spill_df.index)
    ordered_idx, ordered_ch, boundaries = _get_laser_ordered_indices(channels)
    short_labels = [_format_compact_channel_label(c, marker_map, include_detector=True) for c in ordered_ch]
    n_ch = len(ordered_ch)

    has_diff = diff_df is not None and not diff_df.empty
    ncols = 2 if has_diff else 1
    # Dynamic scaling based on channel count (< 10 channels auto-scales down)
    sub_w = max(3.8, min(8.2, 1.6 + n_ch * 0.50))
    sub_h = max(3.8, min(7.6, 1.6 + n_ch * 0.50))
    fig, axes = plt.subplots(1, ncols, figsize=(sub_w * ncols, sub_h), dpi=dpi)
    ax1 = axes[0] if has_diff else axes

    mat_pct = spill_df.iloc[ordered_idx, ordered_idx].to_numpy(dtype=np.float64) * 100.0
    off_diag = ~np.eye(n_ch, dtype=bool)
    max_off = float(np.nanmax(mat_pct[off_diag])) if np.any(off_diag) else 100.0
    vmax = max(15.0, min(100.0, max_off * 1.05))

    im1 = ax1.imshow(mat_pct, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="equal")
    ax1.set_xticks(range(n_ch))
    ax1.set_yticks(range(n_ch))
    tick_fs = max(5.8, min(8.5, 12.0 - n_ch * 0.45))
    annot_fs = max(5.0, min(8.0, 11.0 - n_ch * 0.50))
    ax1.set_xticklabels(short_labels, rotation=90, fontsize=tick_fs)
    ax1.set_yticklabels(short_labels, fontsize=tick_fs)
    ax1.set_xlabel("Secondary Detector Channel (Spillover Into)", fontsize=8.2, fontweight="bold")
    ax1.set_ylabel("Primary Stained Fluorophore Channel", fontsize=8.2, fontweight="bold")

    for b in boundaries:
        ax1.axhline(b - 0.5, color="#222222", linewidth=0.9, linestyle="--", alpha=0.75)
        ax1.axvline(b - 0.5, color="#222222", linewidth=0.9, linestyle="--", alpha=0.75)

    # Annotate spillover cells >= 5%
    for i in range(n_ch):
        for j in range(n_ch):
            if i != j and mat_pct[i, j] >= 5.0:
                val = mat_pct[i, j]
                txt_col = "white" if val > 0.55 * vmax else "black"
                ax1.text(j, i, f"{val:.0f}", ha="center", va="center", color=txt_col, fontsize=annot_fs, fontweight="bold")

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.044, pad=0.03)
    cbar1.set_label("Spillover Coefficient (%)", fontsize=7.8)
    cbar1.ax.tick_params(labelsize=6.8)
    ax1.set_title(
        f"A. Calculated Fluorescence Spillover Matrix (Laser-Ordered)\n"
        f"(Estimator: {comp_result.estimation_method_used.upper()} | Condition Number κ = {comp_result.condition_number:.2f})",
        fontsize=9.0,
        fontweight="bold",
        pad=8,
    )
    ax1.grid(False)

    if has_diff:
        ax2 = axes[1]
        diff_vals = diff_df.iloc[ordered_idx, ordered_idx].to_numpy(dtype=np.float64)
        max_abs_d = max(2.0, min(10.0, float(np.nanmax(np.abs(diff_vals)))))
        im2 = ax2.imshow(diff_vals, cmap="RdBu_r", vmin=-max_abs_d, vmax=max_abs_d, aspect="equal")
        ax2.set_xticks(range(n_ch))
        ax2.set_yticks(range(n_ch))
        ax2.set_xticklabels(short_labels, rotation=90, fontsize=tick_fs)
        ax2.set_yticklabels(short_labels, fontsize=tick_fs)
        ax2.set_xlabel("Secondary Detector Channel", fontsize=8.2, fontweight="bold")
        ax2.set_ylabel("Primary Stained Fluorophore Channel", fontsize=8.2, fontweight="bold")

        for b in boundaries:
            ax2.axhline(b - 0.5, color="#222222", linewidth=0.9, linestyle="--", alpha=0.75)
            ax2.axvline(b - 0.5, color="#222222", linewidth=0.9, linestyle="--", alpha=0.75)

        for i in range(n_ch):
            for j in range(n_ch):
                if i != j and abs(diff_vals[i, j]) >= 1.0:
                    val = diff_vals[i, j]
                    txt_col = "white" if abs(val) > 0.65 * max_abs_d else "black"
                    ax2.text(j, i, f"{val:+.1f}", ha="center", va="center", color=txt_col, fontsize=annot_fs)

        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.044, pad=0.03)
        cbar2.set_label("Difference: Calculated − Embedded (%)", fontsize=7.8)
        cbar2.ax.tick_params(labelsize=6.8)
        r_val = comp_result.embedded_correlation or 0.0
        mae_pct = (comp_result.embedded_mae or 0.0) * 100.0
        ax2.set_title(
            f"B. Concordance vs. Cytometer Embedded $SPILLOVER Matrix\n"
            f"(Pearson r = {r_val:.5f} | Off-Diagonal MAE = {mae_pct:.3f}%)",
            fontsize=9.0,
            fontweight="bold",
            pad=8,
        )
        ax2.grid(False)

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_spillover_spreading_matrix(
    comp_result: CompensationControlResult,
    marker_map: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot the Nguyen et al. (2013) Spillover Spreading Matrix (SSM) ordered by excitation laser.

    Visualizes intrinsic detector resolution loss (photon-counting spread) between every
    primary fluorophore and secondary detector so flow cytometrists can distinguish
    expected population spread from compensation errors.

    Args:
        comp_result: Populated CompensationControlResult with ssm_dataframe.
        marker_map: Optional detector-to-marker dictionary.
        output_path: Optional path to export image.
        dpi: Resolution.

    Returns:
        Matplotlib Figure object.
    """
    set_nature_style()
    ssm_df = comp_result.ssm_dataframe
    if ssm_df is None or ssm_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "Spillover Spreading Matrix Not Available", ha="center", va="center")
        return fig

    channels = list(ssm_df.index)
    ordered_idx, ordered_ch, boundaries = _get_laser_ordered_indices(channels)
    short_labels = [_format_compact_channel_label(c, marker_map, include_detector=True) for c in ordered_ch]
    ssm_vals = ssm_df.iloc[ordered_idx, ordered_idx].to_numpy(dtype=np.float64)
    n_ch = len(ordered_ch)

    # Dynamic scaling based on channel count (< 10 channels auto-scales down)
    fig_w = max(4.0, min(8.6, 1.8 + n_ch * 0.50))
    fig_h = max(3.8, min(7.6, 1.6 + n_ch * 0.50))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    vmax = max(2.0, float(np.nanpercentile(ssm_vals[ssm_vals > 0], 98)) if np.any(ssm_vals > 0) else 5.0)
    im = ax.imshow(ssm_vals, cmap="magma_r", vmin=0.0, vmax=vmax, aspect="equal")

    ax.set_xticks(range(n_ch))
    ax.set_yticks(range(n_ch))
    tick_fs = max(5.8, min(8.5, 12.0 - n_ch * 0.45))
    annot_fs = max(5.0, min(8.0, 11.0 - n_ch * 0.50))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=tick_fs)
    ax.set_yticklabels(short_labels, fontsize=tick_fs)
    ax.set_xlabel("Secondary Detector Channel j (Receives Spread)", fontsize=8.2, fontweight="bold")
    ax.set_ylabel("Primary Fluorophore Channel i (Introduces Spread)", fontsize=8.2, fontweight="bold")

    for b in boundaries:
        ax.axhline(b - 0.5, color="#333333", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.axvline(b - 0.5, color="#333333", linewidth=0.9, linestyle="--", alpha=0.7)

    for i in range(n_ch):
        for j in range(n_ch):
            if i != j and ssm_vals[i, j] >= 0.35 * vmax:
                val = ssm_vals[i, j]
                txt_col = "white" if val > 0.55 * vmax else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=txt_col, fontsize=annot_fs, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
    cbar.set_label("Spillover Spreading Coefficient (SSM)", fontsize=7.8)
    cbar.ax.tick_params(labelsize=6.8)

    mean_ssm = float(np.mean(ssm_vals[~np.eye(n_ch, dtype=bool)]))
    max_ssm = float(np.max(ssm_vals))
    ax.set_title(
        f"Spillover Spreading Matrix (SSM — Nguyen et al., Cytometry Part A 2013)\n"
        f"(Quantifies Post-Compensation Photon Counting Spread | Mean Off-Diag SSM = {mean_ssm:.2f}, Max = {max_ssm:.2f})",
        fontsize=9.0,
        fontweight="bold",
        pad=8,
    )
    ax.grid(False)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def _pregate_biological_sample_for_comp_check(
    uncomp_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    marker_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Gate biological sample to Clean Live Singlets (Lymphocyte/PBMC gate) for compensation inspection.

    Flow cytometrists never inspect marker compensation on ungated whole blood containing
    RBC lysis debris, doublets, high-autofluorescence granulocytes, and dead cells (which
    non-specifically bind tandem dyes). This function isolates intact leukocytes via
    2-component GMM on FSC-A, isolates low-SSC lymphocytes via 2-component GMM on SSC-A,
    gates singlets via FSC-A vs FSC-H, and gates out Zombie+ dead cells.
    """
    from sklearn.mixture import GaussianMixture

    n_total = len(uncomp_df)
    mask = np.ones(n_total, dtype=bool)
    gate_notes: List[str] = []

    # 1. Intact Leukocyte & Low-SSC Lymphocyte gate (excludes RBC lysis debris & granulocytes)
    if "FSC-A" in uncomp_df.columns and "SSC-A" in uncomp_df.columns:
        fsc = uncomp_df["FSC-A"].to_numpy()
        ssc = uncomp_df["SSC-A"].to_numpy()
        fsc_valid = fsc[(fsc > 10000.0) & (fsc < 245000.0) & np.isfinite(fsc)]
        fsc_lo_thresh = 25000.0
        if len(fsc_valid) >= 1000:
            try:
                rng = np.random.RandomState(42)
                sub_f = rng.choice(fsc_valid, size=min(len(fsc_valid), 15000), replace=False).reshape(-1, 1)
                gmm_f = GaussianMixture(n_components=2, random_state=42).fit(sub_f)
                f_means = sorted(gmm_f.means_.flatten())
                if f_means[1] > 75000.0 and (f_means[1] - f_means[0]) > 35000.0:
                    fsc_lo_thresh = max(68000.0, 0.5 * (f_means[0] + f_means[1]))
            except Exception:
                pass

        fsc_ok = (fsc >= fsc_lo_thresh) & (fsc <= min(180000.0, float(np.nanpercentile(fsc, 98.0))))

        # Isolate low-SSC lymphocyte mode from high-SSC granulocytes
        ssc_leuk = ssc[fsc_ok & (ssc > 8000.0) & (ssc < 245000.0) & np.isfinite(ssc)]
        ssc_hi_thresh = 60000.0
        if len(ssc_leuk) >= 1000:
            try:
                rng = np.random.RandomState(42)
                sub_s = rng.choice(ssc_leuk, size=min(len(ssc_leuk), 15000), replace=False).reshape(-1, 1)
                gmm_s = GaussianMixture(n_components=2, random_state=42).fit(sub_s)
                s_means = sorted(gmm_s.means_.flatten())
                if s_means[1] - s_means[0] > 30000.0:
                    ssc_hi_thresh = min(65000.0, max(48000.0, s_means[0] + 0.25 * (s_means[1] - s_means[0])))
            except Exception:
                pass

        ssc_ok = (ssc >= 15000.0) & (ssc <= ssc_hi_thresh)
        if np.sum(fsc_ok & ssc_ok) >= 3000:
            mask &= fsc_ok & ssc_ok
            gate_notes.append("Lymphocyte Scatter")

    # 2. Singlet gate (FSC-A vs FSC-H)
    if "FSC-A" in uncomp_df.columns and "FSC-H" in uncomp_df.columns:
        fsc = uncomp_df["FSC-A"].to_numpy()
        fsc_h = uncomp_df["FSC-H"].to_numpy()
        ratio = fsc_h / np.maximum(fsc, 1.0)
        med_r = float(np.nanmedian(ratio[mask])) if np.any(mask) else 1.0
        mad_r = float(np.nanmedian(np.abs(ratio[mask] - med_r))) if np.any(mask) else 0.04
        sing_ok = np.abs(ratio - med_r) <= 3.0 * max(mad_r, 0.02)
        if np.sum(mask & sing_ok) >= 2500:
            mask &= sing_ok
            gate_notes.append("Singlets")

    # 3. Viability gate (exclude Zombie+ / Dead cells that cause tandem dye non-specific streaks)
    viability_ch = None
    if marker_map:
        for ch, m_name in marker_map.items():
            if any(k in m_name.lower() for k in ("zombie", "viability", "live", "dead", "7aad", "dapi")):
                viability_ch = ch
                break
    if viability_ch and viability_ch in comp_df.columns:
        v_comp = comp_df[viability_ch].to_numpy()
        v_raw = uncomp_df[viability_ch].to_numpy()
        v_cutoff = float(np.nanpercentile(v_comp[mask], 72.0)) if np.any(mask) else 1100.0
        live_ok = (v_comp <= max(750.0, min(v_cutoff, 1200.0))) & (v_raw <= 4500.0)
        if np.sum(mask & live_ok) >= 2000:
            mask &= live_ok
            gate_notes.append("Live (Zombie−)")

    summary_str = " → ".join(gate_notes) if gate_notes else "Time-QC Clean Events"
    return uncomp_df.loc[mask].copy(), comp_df.loc[mask].copy(), summary_str


def _build_adaptive_biex_transform(
    uncomp_vals: np.ndarray,
    comp_vals: np.ndarray,
) -> Tuple[LogicleTransform, List[float], List[str], Tuple[float, float]]:
    """Construct a channel-adaptive LogicleTransform (Parks, Roederer & Moore 2006) with decade ticks.

    Calculates the 1st/2nd percentile of negative compensated values to size the linear width W
    and negative decades A so that high-SSM negative spread is never crushed against the axis wall.
    """
    finite_c = comp_vals[np.isfinite(comp_vals)]
    finite_u = uncomp_vals[np.isfinite(uncomp_vals)]
    valid_parts = [arr for arr in (finite_c, finite_u) if len(arr) > 0]
    all_v = np.concatenate(valid_parts) if valid_parts else np.array([0.0, 262144.0])

    p999 = float(np.nanpercentile(all_v, 99.9)) if len(all_v) > 0 else 262144.0
    param_t = max(262144.0, p999 * 1.1)
    param_m = 4.5

    # Estimate negative spread from 2nd percentile of compensated events
    neg_vals = finite_c[finite_c < 0]
    if len(neg_vals) >= 20:
        r_neg = abs(float(np.nanpercentile(neg_vals, 2.0)))
    else:
        r_neg = 150.0

    # Parks et al. (2006): W = (M - log10(T / r)) / 2, bounded so 2*W < M
    w_calc = (param_m - np.log10(max(param_t / max(r_neg, 50.0), 10.0))) / 2.0
    param_w = float(np.clip(w_calc + 0.15, 0.65, 1.55))
    param_a = float(np.clip(np.log10(max(r_neg, 100.0)) - 1.5, 0.5, 1.3))

    xform = LogicleTransform(param_t=param_t, param_w=param_w, param_m=param_m, param_a=param_a)

    # Compute authentic cytometry decade tick positions without label crowding
    if r_neg > 2500.0:
        raw_ticks = [-10000.0, 0.0, 1000.0, 10000.0, 100000.0]
        tick_labels = ["$-10^4$", "$0$", "$10^3$", "$10^4$", "$10^5$"]
    else:
        raw_ticks = [-1000.0, 0.0, 1000.0, 10000.0, 100000.0]
        tick_labels = ["$-10^3$", "$0$", "$10^3$", "$10^4$", "$10^5$"]

    tick_pos = [float(xform.apply(np.array([tv]))[0]) for tv in raw_ticks]
    # If -10^3 and 0 or 0 and +10^3 are closer than 0.11 in [0,1] space, omit the crowded inner tick label
    if (tick_pos[1] - tick_pos[0]) < 0.11:
        tick_labels[0] = ""
    if (tick_pos[2] - tick_pos[1]) < 0.105:
        tick_labels[2] = ""

    # Compute comfortable axis limits with breathing room so no events touch the frame
    lo_candidates = [tick_pos[0]]
    hi_candidates = [tick_pos[-1]]
    if len(finite_c) > 0:
        t_c = xform.apply(finite_c)
        lo_candidates.append(float(np.nanpercentile(t_c, 0.15)))
        hi_candidates.append(float(np.nanpercentile(t_c, 99.9)))
    if len(finite_u) > 0:
        t_u = xform.apply(finite_u)
        lo_candidates.append(float(np.nanpercentile(t_u, 0.15)))
        hi_candidates.append(float(np.nanpercentile(t_u, 99.9)))
    lo = min(lo_candidates) - 0.035
    hi = max(hi_candidates) + 0.035
    return xform, tick_pos, tick_labels, (max(-0.02, lo), min(1.02, hi))


def plot_sample_top_pairs_gallery(
    uncomp_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    comp_matrix: CompensationMatrix,
    marker_map: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    n_pairs: Optional[int] = None,
    dpi: int = 240,
    reference_control_matrix: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """Plot an auto-scaling Before-vs-After gallery of primary spillover pairs across all stains.

    Automatically scales subplot dimensions, grid columns, and typography to the number of stains:
    - When `n_pairs is None` (default), includes the #1 highest-spillover receiver channel for EVERY
      stain in the panel (e.g., all 28 stains -> 28 [Uncompensated | Compensated] pairs = 56 panels
      arranged in an auto-sized 7x8 grid), sorted from highest spillover % to lowest.
    - Pre-gates biological sample on Live Singlet Lymphocyte cells (and includes Live+Dead singlets
      when inspecting the viability stain itself so Zombie+ events are visible).
    - Applies per-channel adaptive Width-Basis Logicle scaling (Parks et al., 2006) with authentic
      biexponential decade ticks (-10^3, 0, 10^3, 10^4, 10^5), 2D density contours, and orthogonal
      single-positive vs. double-negative median alignment guidelines.

    Args:
        uncomp_df: Uncompensated biological sample events (post-Time QC).
        comp_df: Compensated biological sample events (post-Time QC).
        comp_matrix: Applied CompensationMatrix instance.
        marker_map: Optional detector-to-marker dictionary.
        output_path: Optional path to export image.
        n_pairs: Number of pairs to display. If None (default), automatically plots 1 primary pair
            per stain for all N stains in the panel.
        dpi: Resolution.
        reference_control_matrix: Optional pre-sample-alignment bead control matrix DataFrame so
            pairs remain ranked by raw control spillover and any tandem-aligned coefficients
            are transparently annotated.

    Returns:
        Matplotlib Figure object.
    """
    set_nature_style()

    # 1. Pre-gate biological sample on Live Singlet Lymphocytes (and Singlet Lymphocytes for viability dye)
    gated_uncomp, gated_comp, gate_desc = _pregate_biological_sample_for_comp_check(
        uncomp_df, comp_df, marker_map=marker_map
    )
    singlet_uncomp, singlet_comp, _ = _pregate_biological_sample_for_comp_check(
        uncomp_df, comp_df, marker_map=None
    )

    rank_df = (
        reference_control_matrix
        if reference_control_matrix is not None
        else comp_matrix.dataframe
    )
    spill_mat = rank_df.values.copy()
    np.fill_diagonal(spill_mat, 0.0)

    viability_channels = set()
    if marker_map:
        for ch, m_name in marker_map.items():
            if any(k in m_name.lower() for k in ("zombie", "viability", "live", "dead")):
                viability_channels.add(ch)

    selected_pairs: List[Tuple[str, str, float, float]] = []
    n_detectors = len(comp_matrix.detectors)
    ordered_idx, ordered_ch, _ = _get_laser_ordered_indices(comp_matrix.detectors)
    ch_order_map = {ch: pos for pos, ch in enumerate(ordered_ch)}

    if n_pairs is None or n_pairs >= n_detectors:
        # Select the #1 highest-spillover receiver channel for EVERY stain in consistent laser order
        for r_idx in ordered_idx:
            ch_primary = comp_matrix.detectors[r_idx]
            row_vals = spill_mat[r_idx, :].copy()
            best_c = int(np.argmax(row_vals))
            ctrl_val = float(row_vals[best_c])
            ch_secondary = comp_matrix.detectors[best_c]
            active_val = float(comp_matrix.dataframe.loc[ch_primary, ch_secondary])
            selected_pairs.append((ch_primary, ch_secondary, ctrl_val, active_val))
    else:
        flat_indices = np.argsort(spill_mat.flatten())[::-1]
        seen_undirected = set()
        for flat_idx in flat_indices:
            r_idx, c_idx = np.unravel_index(flat_idx, spill_mat.shape)
            ch_primary = comp_matrix.detectors[r_idx]
            ch_secondary = comp_matrix.detectors[c_idx]
            if ch_primary in viability_channels or ch_secondary in viability_channels:
                continue
            ctrl_val = float(spill_mat[r_idx, c_idx])
            active_val = float(comp_matrix.dataframe.loc[ch_primary, ch_secondary])
            pair_key = tuple(sorted([ch_primary, ch_secondary]))
            if pair_key in seen_undirected:
                continue
            seen_undirected.add(pair_key)
            selected_pairs.append((ch_primary, ch_secondary, ctrl_val, active_val))
            if len(selected_pairs) >= n_pairs:
                break
        selected_pairs.sort(key=lambda x: (ch_order_map.get(x[0], 999), ch_order_map.get(x[1], 999)))

    num_selected = len(selected_pairs)
    ncols = min(7, max(1, num_selected)) if num_selected >= 14 else min(4, max(1, num_selected))
    nrows = math.ceil(num_selected / ncols)
    panel_w, panel_h = 2.55, 2.86
    title_fs, lbl_fs, tick_fs, badge_fs, pt_size = 6.6, 6.3, 5.9, 5.4, 1.8

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), dpi=dpi
    )
    axes_flat = np.atleast_1d(axes).flatten()

    rng = np.random.RandomState(42)
    max_pts = 14000 if num_selected > 12 else 22000
    sub_idx_live = rng.choice(
        len(gated_comp), size=min(len(gated_comp), max_pts), replace=False
    )
    sub_idx_sing = rng.choice(
        len(singlet_comp), size=min(len(singlet_comp), max_pts), replace=False
    )

    for idx in range(nrows * ncols):
        ax_cur = axes_flat[idx]
        if idx >= num_selected:
            ax_cur.axis("off")
            continue

        ch_x, ch_y, ctrl_spill_val, active_spill_val = selected_pairs[idx]
        ax_cur.set_box_aspect(1)

        uses_viability = (ch_x in viability_channels) or (ch_y in viability_channels)
        u_source = singlet_uncomp if uses_viability else gated_uncomp
        c_source = singlet_comp if uses_viability else gated_comp
        s_idx = sub_idx_sing if uses_viability else sub_idx_live

        raw_ux = u_source[ch_x].to_numpy()[s_idx]
        raw_uy = u_source[ch_y].to_numpy()[s_idx]
        raw_cx = c_source[ch_x].to_numpy()[s_idx]
        raw_cy = c_source[ch_y].to_numpy()[s_idx]

        # Build channel-specific adaptive Logicle transforms so negative populations never hit the wall
        xform_x, xticks, xticklabels, xlim = _build_adaptive_biex_transform(raw_ux, raw_cx)
        xform_y, yticks, yticklabels, ylim = _build_adaptive_biex_transform(raw_uy, raw_cy)

        ux = xform_x.apply(raw_ux)
        uy = xform_y.apply(raw_uy)
        cx = xform_x.apply(raw_cx)
        cy = xform_y.apply(raw_cy)

        zero_x = float(xform_x.apply(np.array([0.0]))[0])
        zero_y = float(xform_y.apply(np.array([0.0]))[0])

        lbl_x = _format_compact_channel_label(ch_x, marker_map, include_detector=True)
        lbl_y = _format_compact_channel_label(ch_y, marker_map, include_detector=True)

        # 1. Plot Uncompensated (Before) background layer in soft coral-red + dashed contour
        valid_u = np.isfinite(ux) & np.isfinite(uy)
        ux_v, uy_v = ux[valid_u], uy[valid_u]
        if len(ux_v) > 6000:
            u_sub = rng.choice(len(ux_v), size=6000, replace=False)
            ux_plot, uy_plot = ux_v[u_sub], uy_v[u_sub]
        else:
            ux_plot, uy_plot = ux_v, uy_v

        ax_cur.scatter(
            ux_plot,
            uy_plot,
            c="#ef5350",
            s=1.4,
            alpha=0.26,
            edgecolors="none",
            rasterized=True,
            label="Uncomp",
        )
        try:
            from scipy.ndimage import gaussian_filter
            Hu, xeu, yeu = np.histogram2d(ux_v, uy_v, bins=44, range=[xlim, ylim])
            Hu_s = gaussian_filter(Hu.T, sigma=1.1)
            if np.max(Hu_s) > 8:
                xcu = 0.5 * (xeu[:-1] + xeu[1:])
                ycu = 0.5 * (yeu[:-1] + yeu[1:])
                lv_u = np.percentile(Hu_s[Hu_s > 2], [65, 90])
                if len(np.unique(lv_u)) >= 1:
                    ax_cur.contour(
                        xcu,
                        ycu,
                        Hu_s,
                        levels=np.unique(lv_u),
                        colors="#c62828",
                        linestyles="--",
                        linewidths=0.5,
                        alpha=0.45,
                    )
        except Exception:
            pass

        # 2. Plot Compensated (After) foreground layer in density pseudocolor + solid contour
        sx, sy, dens = calculate_density_colors(cx, cy, max_points=max_pts)
        s_order = np.argsort(dens)
        ax_cur.scatter(
            sx[s_order],
            sy[s_order],
            c=dens[s_order],
            cmap="turbo",
            s=pt_size,
            alpha=0.72,
            edgecolors="none",
            rasterized=True,
            label="Comp",
        )

        try:
            from scipy.ndimage import gaussian_filter
            H, xedges, yedges = np.histogram2d(sx, sy, bins=48, range=[xlim, ylim])
            H_smooth = gaussian_filter(H.T, sigma=1.1)
            if np.max(H_smooth) > 8:
                xc = 0.5 * (xedges[:-1] + xedges[1:])
                yc = 0.5 * (yedges[:-1] + yedges[1:])
                levels = np.percentile(H_smooth[H_smooth > 2], [50, 75, 92])
                if len(np.unique(levels)) >= 2:
                    ax_cur.contour(
                        xc,
                        yc,
                        H_smooth,
                        levels=np.unique(levels),
                        colors="#111111",
                        linewidths=0.42,
                        alpha=0.40,
                    )
        except Exception:
            pass

        # 3. Plot both Uncompensated (Red Dashed) and Compensated (Green Solid) Median Vectors
        u_neg_m = (raw_ux >= float(np.nanpercentile(raw_ux, 8))) & (
            raw_ux <= float(np.nanpercentile(raw_ux, 45))
        )
        u_pos_m = (raw_ux >= float(np.nanpercentile(raw_ux, 90))) & (
            raw_ux <= float(np.nanpercentile(raw_ux, 99.2))
        )
        if np.sum(u_neg_m) >= 30 and np.sum(u_pos_m) >= 25:
            umx_n = float(xform_x.apply(np.array([np.nanmedian(raw_ux[u_neg_m])]))[0])
            umy_n = float(xform_y.apply(np.array([np.nanmedian(raw_uy[u_neg_m])]))[0])
            umx_p = float(xform_x.apply(np.array([np.nanmedian(raw_ux[u_pos_m])]))[0])
            umy_p = float(xform_y.apply(np.array([np.nanmedian(raw_uy[u_pos_m])]))[0])
            ax_cur.plot(
                [umx_n, umx_p],
                [umy_n, umy_p],
                color="white",
                linewidth=2.0,
                zorder=9,
            )
            ax_cur.plot(
                [umx_n, umx_p],
                [umy_n, umy_p],
                color="#d50000",
                linestyle="--",
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#d50000",
                markersize=3.2,
                linewidth=1.3,
                zorder=10,
            )

        y_base_med = float(np.nanmedian(raw_cy))
        y_base_rsd = float(np.nanmedian(np.abs(raw_cy - y_base_med))) * 1.4826
        y_neg_gate = raw_cy <= (y_base_med + 2.5 * max(y_base_rsd, 150.0))
        x_neg_mask = (
            y_neg_gate
            & (raw_cx >= float(np.nanpercentile(raw_cx, 10)))
            & (raw_cx <= float(np.nanpercentile(raw_cx, 55)))
        )
        x_pos_mask = (
            y_neg_gate
            & (raw_cx >= float(np.nanpercentile(raw_cx, 90)))
            & (raw_cx <= float(np.nanpercentile(raw_cx, 99.2)))
        )

        res_badge_txt = f"Uncomp (Red) → Comp (Green)\nn={len(sx):,}"
        if np.sum(x_neg_mask) >= 30 and np.sum(x_pos_mask) >= 25:
            raw_mx_neg = float(np.nanmedian(raw_cx[x_neg_mask]))
            raw_my_neg = float(np.nanmedian(raw_cy[x_neg_mask]))
            raw_mx_pos = float(np.nanmedian(raw_cx[x_pos_mask]))
            raw_my_pos = float(np.nanmedian(raw_cy[x_pos_mask]))
            med_x_neg = float(xform_x.apply(np.array([raw_mx_neg]))[0])
            med_y_neg = float(xform_y.apply(np.array([raw_my_neg]))[0])
            med_x_pos = float(xform_x.apply(np.array([raw_mx_pos]))[0])
            med_y_pos = float(xform_y.apply(np.array([raw_my_pos]))[0])

            dy_val = raw_my_pos - raw_my_neg
            rsd_neg = float(
                np.nanmedian(np.abs(raw_cy[x_neg_mask] - raw_my_neg)) * 1.4826
            )
            z_val = dy_val / max(rsd_neg, 50.0)
            res_badge_txt = (
                f"Uncomp(Red) → Comp(Grn)\n"
                f"Post $\\Delta$MFI: {dy_val:+.0f} | $Z$: {z_val:+.2f}"
            )

            ax_cur.plot(
                [med_x_neg, med_x_pos],
                [med_y_neg, med_y_pos],
                color="white",
                linewidth=2.2,
                zorder=11,
            )
            ax_cur.plot(
                [med_x_neg, med_x_pos],
                [med_y_neg, med_y_pos],
                color="#00c853",
                marker="o",
                markersize=3.5,
                linewidth=1.45,
                zorder=12,
            )

        ax_cur.axhline(zero_y, color="#222222", linestyle="--", linewidth=0.75, alpha=0.75)
        ax_cur.axvline(zero_x, color="#222222", linestyle="--", linewidth=0.75, alpha=0.75)

        ax_cur.set_xlim(xlim)
        ax_cur.set_ylim(ylim)
        ax_cur.set_xticks(xticks)
        ax_cur.set_xticklabels(xticklabels, fontsize=tick_fs)
        ax_cur.set_yticks(yticks)
        ax_cur.set_yticklabels(yticklabels, fontsize=tick_fs)

        ax_cur.set_xlabel(lbl_x, fontsize=lbl_fs, fontweight="medium", labelpad=2)
        ax_cur.set_ylabel(lbl_y, fontsize=lbl_fs, fontweight="medium", labelpad=2)

        if abs(ctrl_spill_val - active_spill_val) > 0.05:
            spill_sub = f"S:{ctrl_spill_val*100:.0f}%→{active_spill_val*100:.1f}%"
        else:
            spill_sub = f"S:{active_spill_val*100:.1f}%"
        ax_cur.set_title(
            f"{lbl_x}\n→ {lbl_y}",
            fontsize=title_fs,
            fontweight="bold",
            pad=3.5,
        )
        ax_cur.text(
            0.03,
            0.95,
            f"{spill_sub} | {res_badge_txt}",
            transform=ax_cur.transAxes,
            fontsize=badge_fs,
            fontweight="bold",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="#e8f5e9",
                edgecolor="#2e7d32",
                alpha=0.88,
                linewidth=0.6,
            ),
        )
        apply_nature_axis_style(ax_cur)

    fig.suptitle(
        f"Complete All-Stains Biological Compensation Verification ({gate_desc}): All {num_selected} Stains Before vs. After\n"
        f"(Laser-Ordered {nrows}×{ncols} Square Grid | Coral + Red Dashed = Uncompensated, Density + Green Solid = Compensated)",
        fontsize=11.0,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout(pad=0.70, h_pad=1.10, w_pad=0.65)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def _resolve_laser_group(ch: str, marker: str = "") -> Tuple[int, str, str, str, str]:
    """Resolve excitation laser identity and color palette for grouping subplots.

    Returns:
        (laser_order_rank, laser_slug, laser_display_name, banner_bg_color, banner_border_color)
    """
    txt = f"{ch} {marker}".upper()
    ex_m = re.search(r"Ex\s*(\d{3})", ch, re.IGNORECASE)
    ex_val = int(ex_m.group(1)) if ex_m else None

    if ex_val is not None:
        if ex_val <= 375:
            return (1, "UV_355nm", "355 nm UV Laser", "#ede7f6", "#512da8")
        elif ex_val <= 425:
            return (2, "Violet_402nm", "402 nm Violet Laser", "#e8eaf6", "#1a237e")
        elif ex_val <= 500:
            return (3, "Blue_488nm", "488 nm Blue Laser", "#e1f5fe", "#0277bd")
        elif ex_val <= 580:
            return (4, "YellowGreen_561nm", "561 nm Yellow-Green Laser", "#fff8e1", "#e65100")
        elif ex_val <= 670:
            return (5, "Red_628nm", "628 nm Red Laser", "#ffebee", "#b71c1c")

    if any(k in txt for k in ["BUV", "UV", "355", "DAPI"]):
        return (1, "UV_355nm", "355 nm UV Laser", "#ede7f6", "#512da8")
    if any(k in txt for k in ["BV", "VIOLET", "PACIFIC", "405", "402"]):
        return (2, "Violet_402nm", "402 nm Violet Laser", "#e8eaf6", "#1a237e")
    if any(k in txt for k in ["BB", "FITC", "PERCP", "488", "BLUE"]):
        return (3, "Blue_488nm", "488 nm Blue Laser", "#e1f5fe", "#0277bd")
    if any(k in txt for k in ["PE", "YG", "561"]):
        return (4, "YellowGreen_561nm", "561 nm Yellow-Green Laser", "#fff8e1", "#e65100")
    if any(k in txt for k in ["APC", "RED", "633", "640", "628"]):
        return (5, "Red_628nm", "628 nm Red Laser", "#ffebee", "#b71c1c")
    return (6, "Other", "Other Lasers", "#f5f5f5", "#424242")


def _render_active_pair_subplot(
    ax: plt.Axes,
    ch_x: str,
    ch_y: str,
    ctrl_s: float,
    act_s: float,
    transform_cache: Dict[Tuple[str, bool], Tuple[Any, Any, Any, Any, np.ndarray, np.ndarray, float]],
    singlet_uncomp: pd.DataFrame,
    gated_uncomp: pd.DataFrame,
    singlet_comp: pd.DataFrame,
    gated_comp: pd.DataFrame,
    sub_idx_sing: np.ndarray,
    sub_idx_live: np.ndarray,
    marker_map: Optional[Dict[str, str]] = None,
    viability_channels: Optional[set] = None,
    max_pts: int = 12000,
    title_fs: float = 6.0,
    lbl_fs: float = 6.0,
    tick_fs: float = 5.6,
    badge_fs: float = 5.2,
) -> None:
    """Render an individual post-compensation verification scatter plot with contours and alignment vector."""
    ax.set_box_aspect(1)
    viab_set = viability_channels or set()
    uses_viab = (ch_x in viab_set) or (ch_y in viab_set)

    def _get_channel_transformed(ch_name: str, is_viab: bool):
        key = (ch_name, is_viab)
        if key not in transform_cache:
            u_src = singlet_uncomp if is_viab else gated_uncomp
            c_src = singlet_comp if is_viab else gated_comp
            s_i = sub_idx_sing if is_viab else sub_idx_live
            r_u = u_src[ch_name].to_numpy()[s_i]
            r_c = c_src[ch_name].to_numpy()[s_i]
            xf, xt, xtl, xl = _build_adaptive_biex_transform(r_u, r_c)
            tc = xf.apply(r_c)
            z0 = float(xf.apply(np.array([0.0]))[0])
            transform_cache[key] = (xf, xt, xtl, xl, r_c, tc, z0)
        return transform_cache[key]

    xf_x, xticks, xticklabels, xlim, raw_cx, cx, zero_x = _get_channel_transformed(ch_x, uses_viab)
    xf_y, yticks, yticklabels, ylim, raw_cy, cy, zero_y = _get_channel_transformed(ch_y, uses_viab)

    sx, sy, dens = calculate_density_colors(cx, cy, max_points=max_pts)
    s_order = np.argsort(dens)
    ax.scatter(
        sx[s_order],
        sy[s_order],
        c=dens[s_order],
        cmap="turbo",
        s=1.5,
        alpha=0.72,
        edgecolors="none",
        rasterized=True,
    )

    try:
        from scipy.ndimage import gaussian_filter
        H, xedges, yedges = np.histogram2d(sx, sy, bins=42, range=[xlim, ylim])
        H_smooth = gaussian_filter(H.T, sigma=1.05)
        if np.max(H_smooth) > 8:
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            levels = np.percentile(H_smooth[H_smooth > 2], [50, 75, 92])
            if len(np.unique(levels)) >= 2:
                ax.contour(
                    xc,
                    yc,
                    H_smooth,
                    levels=np.unique(levels),
                    colors="#111111",
                    linewidths=0.4,
                    alpha=0.4,
                )
    except Exception:
        pass

    y_base_med = float(np.nanmedian(raw_cy))
    y_base_rsd = float(np.nanmedian(np.abs(raw_cy - y_base_med))) * 1.4826
    y_neg_gate = raw_cy <= (y_base_med + 2.5 * max(y_base_rsd, 150.0))
    x_neg_mask = (
        y_neg_gate
        & (raw_cx >= float(np.nanpercentile(raw_cx, 10)))
        & (raw_cx <= float(np.nanpercentile(raw_cx, 55)))
    )
    x_pos_mask = (
        y_neg_gate
        & (raw_cx >= float(np.nanpercentile(raw_cx, 90)))
        & (raw_cx <= float(np.nanpercentile(raw_cx, 99.2)))
    )

    z_val = 0.0
    dy_val = 0.0
    if np.sum(x_neg_mask) >= 25 and np.sum(x_pos_mask) >= 20:
        raw_mx_neg = float(np.nanmedian(raw_cx[x_neg_mask]))
        raw_my_neg = float(np.nanmedian(raw_cy[x_neg_mask]))
        raw_mx_pos = float(np.nanmedian(raw_cx[x_pos_mask]))
        raw_my_pos = float(np.nanmedian(raw_cy[x_pos_mask]))
        med_x_neg = float(xf_x.apply(np.array([raw_mx_neg]))[0])
        med_y_neg = float(xf_y.apply(np.array([raw_my_neg]))[0])
        med_x_pos = float(xf_x.apply(np.array([raw_mx_pos]))[0])
        med_y_pos = float(xf_y.apply(np.array([raw_my_pos]))[0])

        dy_val = raw_my_pos - raw_my_neg
        rsd_neg = float(np.nanmedian(np.abs(raw_cy[x_neg_mask] - raw_my_neg)) * 1.4826)
        z_val = dy_val / max(rsd_neg, 50.0)

        vec_col = "#00c853" if abs(z_val) <= 1.25 else "#ff9100"
        ax.plot(
            [med_x_neg, med_x_pos],
            [med_y_neg, med_y_pos],
            color="white",
            linewidth=1.8,
            zorder=9,
        )
        ax.plot(
            [med_x_neg, med_x_pos],
            [med_y_neg, med_y_pos],
            color=vec_col,
            marker="o",
            markersize=2.8,
            linewidth=1.15,
            zorder=10,
        )

    ax.axhline(zero_y, color="#222222", linestyle="--", linewidth=0.65, alpha=0.7)
    ax.axvline(zero_x, color="#222222", linestyle="--", linewidth=0.65, alpha=0.7)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=tick_fs)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=tick_fs)

    lbl_x = _format_compact_channel_label(ch_x, marker_map, include_detector=True)
    lbl_y = _format_compact_channel_label(ch_y, marker_map, include_detector=True)
    ax.set_xlabel(lbl_x, fontsize=lbl_fs, fontweight="medium", labelpad=1.8)
    ax.set_ylabel(lbl_y, fontsize=lbl_fs, fontweight="medium", labelpad=1.8)

    if ctrl_s >= 1.0:
        tier_edge = "#c62828"
        tier_bg = "#ffebee"
    elif ctrl_s >= 0.50:
        tier_edge = "#ef6c00"
        tier_bg = "#fff3e0"
    else:
        tier_edge = "#1565c0"
        tier_bg = "#e3f2fd"

    if abs(ctrl_s - act_s) > 0.05:
        s_str = f"S:{ctrl_s*100:.0f}%→{act_s*100:.0f}%"
    else:
        s_str = f"S:{act_s*100:.1f}%"

    ax.set_title(
        f"{lbl_x}\n→ {lbl_y}",
        fontsize=title_fs,
        fontweight="bold",
        pad=3.0,
    )
    ax.text(
        0.03,
        0.95,
        f"{s_str} | $\\Delta$MFI:{dy_val:+.0f} | $Z$:{z_val:+.2f}",
        transform=ax.transAxes,
        fontsize=badge_fs,
        fontweight="bold",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.16",
            facecolor=tier_bg,
            edgecolor=tier_edge,
            alpha=0.88,
            linewidth=0.55,
        ),
    )
    apply_nature_axis_style(ax)
    for spine in ax.spines.values():
        spine.set_edgecolor(tier_edge)
        spine.set_linewidth(0.85 if ctrl_s >= 0.50 else 0.55)


def plot_all_active_spillover_pairs_atlas(
    uncomp_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    comp_matrix: CompensationMatrix,
    marker_map: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    min_spillover: float = 0.25,
    dpi: int = 240,
    reference_control_matrix: Optional[pd.DataFrame] = None,
    ncols: int = 8,
    export_laser_sheets: bool = True,
) -> plt.Figure:
    """Plot a Compensated Verification Atlas of ALL active spillover pairs (>= min_spillover).

    Option 1 (Master Figure): Laser-Grouped Sections with visual color banners dividing the 5 lasers
        (355nm UV, 402nm Violet, 488nm Blue, 561nm YG, 628nm Red). Within each laser section, subplots
        are 1:1 square, ordered by primary emission -> secondary emission.
    Option 4 (Laser Sheets): When export_laser_sheets is True and output_path is provided, also exports
        individual per-laser sheets (e.g., comp_07_atlas_laser_1_UV_355nm.png) for easy focused inspection.

    Args:
        uncomp_df: Uncompensated biological sample events (post-Time QC).
        comp_df: Compensated biological sample events (post-Time QC).
        comp_matrix: Applied CompensationMatrix instance.
        marker_map: Optional detector-to-marker dictionary.
        output_path: Optional path to export image.
        min_spillover: Minimum fractional spillover threshold (default: 0.25 for 25%).
        dpi: Resolution.
        reference_control_matrix: Optional pre-sample-alignment bead control matrix DataFrame.
        ncols: Number of columns in master atlas grid (default: 8).
        export_laser_sheets: If True, also exports individual laser figure sheets to output_path directory.

    Returns:
        Matplotlib Figure object of the master laser-grouped atlas.
    """
    set_nature_style()

    gated_uncomp, gated_comp, gate_desc = _pregate_biological_sample_for_comp_check(
        uncomp_df, comp_df, marker_map=marker_map
    )
    singlet_uncomp, singlet_comp, _ = _pregate_biological_sample_for_comp_check(
        uncomp_df, comp_df, marker_map=None
    )

    rank_df = (
        reference_control_matrix
        if reference_control_matrix is not None
        else comp_matrix.dataframe
    )
    spill_mat = rank_df.values.copy()
    np.fill_diagonal(spill_mat, 0.0)

    viability_channels = set()
    if marker_map:
        for ch, m_name in marker_map.items():
            if any(k in m_name.lower() for k in ("zombie", "viability", "live", "dead")):
                viability_channels.add(ch)

    # Auto-adapt threshold if panel has few high-spillover pairs
    eff_thresh = min_spillover
    for cand_t in [min_spillover, 0.15, 0.10, 0.05, 0.01]:
        if np.sum(spill_mat >= cand_t) >= min(12, len(comp_matrix.detectors)):
            eff_thresh = cand_t
            break

    # Order active pairs by excitation laser row (primary i) and column (secondary j) sequence
    ordered_idx, _, _ = _get_laser_ordered_indices(comp_matrix.detectors)
    active_pairs: List[Tuple[str, str, float, float]] = []
    for r_idx in ordered_idx:
        for c_idx in ordered_idx:
            if r_idx == c_idx:
                continue
            ctrl_val = float(spill_mat[r_idx, c_idx])
            if ctrl_val < eff_thresh:
                continue
            ch_primary = comp_matrix.detectors[r_idx]
            ch_secondary = comp_matrix.detectors[c_idx]
            active_val = float(comp_matrix.dataframe.loc[ch_primary, ch_secondary])
            active_pairs.append((ch_primary, ch_secondary, ctrl_val, active_val))

    n_total = len(active_pairs)

    # Graceful handling when no pairs exceed threshold
    if n_total == 0:
        fig = plt.figure(figsize=(7.2, 3.2), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.text(
            0.5,
            0.50,
            f"No Active Spillover Pairs Exceeding Threshold (≥{eff_thresh*100:.0f}%)\n\n"
            f"All {len(comp_matrix.detectors)} detector channels exhibit clean spectral independence.\n"
            f"No significant fluorescence cross-talk requiring active pair audit.",
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="#374151",
            bbox=dict(
                boxstyle="round,pad=0.7",
                facecolor="#f9fafb",
                edgecolor="#d1d5db",
                linewidth=1.2,
            ),
        )
        fig.suptitle(
            f"Compensated Spillover Verification Atlas ({gate_desc})\nAll Channels Spectrally Independent",
            fontsize=11.0,
            fontweight="bold",
            y=0.92,
        )
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
        return fig

    # Group active pairs by primary channel excitation laser
    laser_groups: Dict[Tuple[int, str, str, str, str], List[Tuple[str, str, float, float]]] = {}
    for pair in active_pairs:
        ch_pri = pair[0]
        m_pri = marker_map.get(ch_pri, "") if marker_map else ""
        grp_info = _resolve_laser_group(ch_pri, m_pri)
        if grp_info not in laser_groups:
            laser_groups[grp_info] = []
        laser_groups[grp_info].append(pair)

    sorted_groups = sorted(laser_groups.items(), key=lambda x: x[0][0])
    max_grp_len = max((len(grp_pairs) for _, grp_pairs in sorted_groups), default=1)

    # Dynamically determine columns to eliminate empty gaps and avoid overly wide canvases
    if ncols == 8 or ncols is None:
        if n_total <= 2:
            effective_ncols = max(1, n_total)
        elif n_total <= 4:
            effective_ncols = min(2, max_grp_len) if max_grp_len <= 2 else 2
        elif n_total <= 8:
            effective_ncols = min(3, max(2, max_grp_len))
        elif n_total <= 16:
            effective_ncols = min(4, max(2, max_grp_len))
        elif n_total <= 28:
            effective_ncols = min(5, max(3, max_grp_len))
        else:
            effective_ncols = min(6, max(4, max_grp_len))
    else:
        effective_ncols = max(1, min(ncols, n_total))

    rng = np.random.RandomState(42)
    max_pts = 12000
    sub_idx_live = rng.choice(
        len(gated_comp), size=min(len(gated_comp), max_pts), replace=False
    )
    sub_idx_sing = rng.choice(
        len(singlet_comp), size=min(len(singlet_comp), max_pts), replace=False
    )

    # Pre-cache channel transforms for speed across all pairs
    transform_cache: Dict[Tuple[str, bool], Tuple[Any, Any, Any, Any, np.ndarray, np.ndarray, float]] = {}

    # 1. OPTION 1: Build Master Figure with Laser Sections and Color Banners
    # Calculate row layout: 1 banner row + ceil(len(pairs)/effective_ncols) subplot rows per laser group
    grid_rows_plan: List[Tuple[str, Any]] = []
    height_ratios: List[float] = []

    for grp_info, grp_pairs in sorted_groups:
        grid_rows_plan.append(("banner", grp_info, grp_pairs))
        height_ratios.append(0.20)
        n_sub_rows = math.ceil(len(grp_pairs) / effective_ncols)
        for sub_r in range(n_sub_rows):
            start_i = sub_r * effective_ncols
            end_i = min(len(grp_pairs), start_i + effective_ncols)
            row_pairs = grp_pairs[start_i:end_i]
            grid_rows_plan.append(("subplots", row_pairs, sub_r))
            height_ratios.append(1.0)

    total_rows = len(grid_rows_plan)
    total_sub_rows = sum(1 for r_type, *rest in grid_rows_plan if r_type == "subplots")

    cell_w = max(2.30, min(2.70, 11.0 / effective_ncols))
    cell_h = cell_w * 1.10
    fig_w = max(4.8, min(16.0, effective_ncols * cell_w + 0.8))
    fig_h = max(3.2, (total_sub_rows * cell_h) + (len(sorted_groups) * 0.50) + 0.8)

    left_m = max(0.04, min(0.08, 0.45 / fig_w))
    right_m = 1.0 - left_m
    top_m = 0.95 if total_rows > 3 else 0.90
    bottom_m = 0.035

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = gridspec.GridSpec(
        total_rows,
        effective_ncols,
        height_ratios=height_ratios,
        left=left_m,
        right=right_m,
        top=top_m,
        bottom=bottom_m,
        hspace=0.28,
        wspace=0.22,
    )

    banner_fs = max(7.2, min(9.0, 36.0 / effective_ncols))

    for r_idx, plan_item in enumerate(grid_rows_plan):
        r_type = plan_item[0]
        if r_type == "banner":
            _, grp_info, grp_pairs = plan_item
            laser_order, laser_slug, laser_name, bg_col, bdr_col = grp_info
            ax_banner = fig.add_subplot(gs[r_idx, :])
            ax_banner.axis("off")

            primary_markers = list(dict.fromkeys([
                _format_compact_channel_label(p[0], marker_map, include_detector=False)
                for p in grp_pairs
            ]))
            m_list_str = ", ".join(primary_markers)
            banner_txt = (
                f"  ■  {laser_name} Stains ({len(grp_pairs)} Active Interactions | ≥{eff_thresh*100:.0f}% Spillover)\n"
                f"      Primary Stained Fluorophores: {m_list_str}"
            )
            ax_banner.text(
                0.0,
                0.5,
                banner_txt,
                fontsize=banner_fs,
                fontweight="bold",
                color=bdr_col,
                va="center",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.38,rounding_size=0.15",
                    facecolor=bg_col,
                    edgecolor=bdr_col,
                    linewidth=1.2,
                    alpha=0.95,
                ),
            )
        else:
            _, row_pairs, _ = plan_item
            for c_idx in range(effective_ncols):
                ax_sub = fig.add_subplot(gs[r_idx, c_idx])
                if c_idx < len(row_pairs):
                    ch_x, ch_y, ctrl_s, act_s = row_pairs[c_idx]
                    _render_active_pair_subplot(
                        ax=ax_sub,
                        ch_x=ch_x,
                        ch_y=ch_y,
                        ctrl_s=ctrl_s,
                        act_s=act_s,
                        transform_cache=transform_cache,
                        singlet_uncomp=singlet_uncomp,
                        gated_uncomp=gated_uncomp,
                        singlet_comp=singlet_comp,
                        gated_comp=gated_comp,
                        sub_idx_sing=sub_idx_sing,
                        sub_idx_live=sub_idx_live,
                        marker_map=marker_map,
                        viability_channels=viability_channels,
                        max_pts=max_pts,
                        title_fs=max(5.8, min(7.5, 30.0 / effective_ncols)),
                        lbl_fs=max(5.6, min(7.2, 28.0 / effective_ncols)),
                        tick_fs=max(5.2, min(6.8, 26.0 / effective_ncols)),
                        badge_fs=max(5.0, min(6.5, 24.0 / effective_ncols)),
                    )
                else:
                    ax_sub.axis("off")

    fig.suptitle(
        f"Compensated Spillover Verification Atlas ({gate_desc}): All {n_total} Active Spillover Pairs (≥{eff_thresh*100:.0f}%)\n"
        f"(Organized by Primary Excitation Laser Sections with Banners | Red: S≥100%, Orange: S 50–100%, Blue: S 25–50%)",
        fontsize=max(9.5, min(12.0, 48.0 / effective_ncols)),
        fontweight="bold",
        y=0.992 if total_rows > 3 else 0.985,
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

        # 2. OPTION 4: Export Individual Per-Laser Sheets if requested
        if export_laser_sheets:
            out_dir = out.parent
            for grp_info, grp_pairs in sorted_groups:
                laser_order, laser_slug, laser_name, bg_col, bdr_col = grp_info
                s_ncols = min(4, max(1, len(grp_pairs)))
                s_nrows = math.ceil(len(grp_pairs) / s_ncols)
                s_cell_w = max(2.35, min(2.75, 9.5 / s_ncols))
                s_fig_w = max(4.0, s_ncols * s_cell_w + 0.6)
                s_fig_h = max(3.0, s_nrows * (s_cell_w * 1.10) + 0.8)
                s_fig, s_axes = plt.subplots(
                    s_nrows, s_ncols, figsize=(s_fig_w, s_fig_h), dpi=dpi
                )
                s_axes_flat = np.atleast_1d(s_axes).flatten()

                for p_idx, pair_data in enumerate(grp_pairs):
                    ch_x, ch_y, ctrl_s, act_s = pair_data
                    _render_active_pair_subplot(
                        ax=s_axes_flat[p_idx],
                        ch_x=ch_x,
                        ch_y=ch_y,
                        ctrl_s=ctrl_s,
                        act_s=act_s,
                        transform_cache=transform_cache,
                        singlet_uncomp=singlet_uncomp,
                        gated_uncomp=gated_uncomp,
                        singlet_comp=singlet_comp,
                        gated_comp=gated_comp,
                        sub_idx_sing=sub_idx_sing,
                        sub_idx_live=sub_idx_live,
                        marker_map=marker_map,
                        viability_channels=viability_channels,
                        max_pts=max_pts,
                        title_fs=max(6.0, min(8.0, 32.0 / s_ncols)),
                        lbl_fs=max(5.8, min(7.6, 30.0 / s_ncols)),
                        tick_fs=max(5.4, min(7.0, 28.0 / s_ncols)),
                        badge_fs=max(5.0, min(6.6, 26.0 / s_ncols)),
                    )

                for p_idx in range(len(grp_pairs), len(s_axes_flat)):
                    s_axes_flat[p_idx].axis("off")

                s_fig.suptitle(
                    f"{laser_name} Compensated Spillover Sheet ({gate_desc})\n"
                    f"All {len(grp_pairs)} Active Interactions (≥{eff_thresh*100:.0f}% Spillover | Red: S≥100%, Orange: S 50–100%, Blue: S 25–50%)",
                    fontsize=max(9.0, min(11.5, 42.0 / s_ncols)),
                    fontweight="bold",
                    y=0.995,
                )
                plt.tight_layout(pad=0.70, h_pad=1.05, w_pad=0.65)
                sheet_filename = f"comp_07_atlas_laser_{laser_order}_{laser_slug}.png"
                s_fig.savefig(out_dir / sheet_filename, dpi=dpi, bbox_inches="tight")
                plt.close(s_fig)

        plt.close(fig)

    return fig


def plot_estimator_comparison(
    mfi_result: CompensationControlResult,
    reg_result: CompensationControlResult,
    hybrid_result: CompensationControlResult,
    embedded_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot a 3-panel benchmark comparing Median Delta-MFI, AutoSpill Regression, and Hybrid estimators.

    Panel A: Parity scatter plot of off-diagonal spillover coefficients (%) vs. Embedded Instrument Matrix.
    Panel B: Post-compensation Residual |Z|-Score cumulative distribution (ECDF) across all 756 pairs.
    Panel C: Mean Absolute Error (%) and Residual Alignment summary bar chart.

    Args:
        mfi_result: CompensationControlResult from 'median_delta_mfi'.
        reg_result: CompensationControlResult from 'autospill_regression'.
        hybrid_result: CompensationControlResult from 'hybrid'.
        embedded_df: Optional embedded spillover matrix DataFrame.
        output_path: Optional path to export image.
        dpi: Resolution.

    Returns:
        Matplotlib Figure object.
    """
    set_nature_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.2, 3.9), dpi=dpi)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)

    channels = list(hybrid_result.dataframe.index)
    n_ch = len(channels)
    off_diag = ~np.eye(n_ch, dtype=bool)

    # Panel A: Parity vs Embedded Matrix
    if embedded_df is not None:
        ref_vals = embedded_df.reindex(index=channels, columns=channels).fillna(0.0).values[off_diag] * 100.0
        for res_obj, lbl, col, marker in [
            (mfi_result, f"Median ΔMFI (r={mfi_result.embedded_correlation:.5f})", "#1f77b4", "o"),
            (reg_result, f"AutoSpill Reg (r={reg_result.embedded_correlation:.5f})", "#ff7f0e", "s"),
            (hybrid_result, f"Hybrid (r={hybrid_result.embedded_correlation:.5f})", "#2ca02c", "^"),
        ]:
            calc_vals = res_obj.dataframe.values[off_diag] * 100.0
            ax1.scatter(ref_vals, calc_vals, c=col, s=14, alpha=0.65, marker=marker, label=lbl, edgecolors="none")
        max_v = max(float(np.max(ref_vals)), 100.0)
        ax1.plot([0, max_v], [0, max_v], color="#222222", linestyle="--", linewidth=1.0, label="1:1 Parity")
        ax1.set_xlabel("Instrument Embedded Spillover (%)", fontsize=8.0)
        ax1.set_ylabel("Calculated Spillover (%)", fontsize=8.0)
        ax1.set_title("A. Concordance with Instrument Matrix", fontsize=8.5, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    apply_nature_axis_style(ax1)

    # Panel B: ECDF of Post-Compensation Residual |Z|-Scores
    for res_obj, lbl, col, ls in [
        (mfi_result, "Median ΔMFI", "#1f77b4", "--"),
        (reg_result, "AutoSpill Regression", "#ff7f0e", "-."),
        (hybrid_result, "Hybrid (MFI + AutoSpill)", "#2ca02c", "-"),
    ]:
        z_vals = np.sort(np.abs(res_obj.residual_z_dataframe.values[off_diag]))
        y_ecdf = np.arange(1, len(z_vals) + 1) / len(z_vals) * 100.0
        med_z = float(np.mean(z_vals))
        ax2.plot(z_vals, y_ecdf, color=col, linestyle=ls, linewidth=1.6, label=f"{lbl} (Mean |Z|={med_z:.3f})")

    ax2.axvline(0.5, color="#d62728", linestyle=":", linewidth=1.1, label="Tolerance (|Z| = 0.5 rSD)")
    ax2.set_xlim(0, 1.2)
    ax2.set_ylim(0, 102)
    ax2.set_xlabel("Post-Comp Residual |Z|-Score (rSD units)", fontsize=8.0)
    ax2.set_ylabel("Cumulative % of Channel Pairs", fontsize=8.0)
    ax2.set_title("B. Post-Comp Residual Alignment ECDF", fontsize=8.5, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=6.3, framealpha=0.9)
    apply_nature_axis_style(ax2)

    # Panel C: Summary Metrics Bar Chart
    methods = ["Median\nΔMFI", "AutoSpill\nRegression", "Hybrid\n(Default)"]
    maes = [
        (mfi_result.embedded_mae or 0.0) * 100.0,
        (reg_result.embedded_mae or 0.0) * 100.0,
        (hybrid_result.embedded_mae or 0.0) * 100.0,
    ]
    mean_zs = [
        float(np.mean(np.abs(mfi_result.residual_z_dataframe.values[off_diag]))),
        float(np.mean(np.abs(reg_result.residual_z_dataframe.values[off_diag]))),
        float(np.mean(np.abs(hybrid_result.residual_z_dataframe.values[off_diag]))),
    ]

    x_pos = np.arange(len(methods))
    width = 0.34
    bars1 = ax3.bar(x_pos - width / 2, maes, width, color="#1f77b4", alpha=0.85, label="Embedded MAE (%)")
    bars2 = ax3.bar(x_pos + width / 2, mean_zs, width, color="#2ca02c", alpha=0.85, label="Mean Residual |Z| (rSD)")

    for b in bars1:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2, h + 0.004, f"{h:.3f}%", ha="center", va="bottom", fontsize=6.2)
    for b in bars2:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2, h + 0.004, f"{h:.3f}", ha="center", va="bottom", fontsize=6.2)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, fontsize=7.5)
    ax3.set_ylabel("Error / Residual Magnitude", fontsize=8.0)
    ax3.set_ylim(0, max(max(maes), max(mean_zs)) * 1.28)
    ax3.set_title("C. Estimator Accuracy & Alignment Benchmark", fontsize=8.5, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    apply_nature_axis_style(ax3)

    fig.suptitle(
        "Comparison of Spillover Estimators: Upgraded Median ΔMFI vs. AutoSpill Regression vs. Hybrid",
        fontsize=10.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def _universal_sample_comp_filter(
    uncomp_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    marker_map: Optional[Dict[str, str]] = None,
    channel_ranges: Optional[Dict[str, float]] = None,
    gating_mode: str = "viable",
    viability_channel: Optional[str] = None,
    gate_mask: Optional[np.ndarray] = None,
    min_events_retain: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Universal, cytometer- and sample-agnostic gating filter for compensation verification.

    Filters biological sample events to eliminate non-specific artifacts without assuming
    a specific lineage (works equally across lymphocytes, macrophages, stem cells,
    epithelial lines, and tissue digests across any cytometer brand).

    Args:
        uncomp_df: Uncompensated events DataFrame.
        comp_df: Compensated events DataFrame.
        marker_map: Optional dict mapping channel name to biological marker label.
        channel_ranges: Optional dict mapping channel name to hardware maximum dynamic range.
        gating_mode: Clean-up string controlling which populations are retained:
            - 'viable' (default): Saturation + Debris + Singlets + Viability dye exclusion.
            - 'auto': Saturation + Debris + Singlets, plus Viability exclusion if detected.
            - 'singlet': Saturation + Debris + Singlets (retains both live and dead singlets).
            - 'intact': Saturation + Debris exclusion only (retains doublets and dead cells).
            - 'all' / 'none': Raw ungated events without filtering.
        viability_channel: Optional explicit viability dye channel name.
        gate_mask: Optional boolean array mask of pre-computed viable singlet events.
        min_events_retain: Minimum events required to apply each progressive filter step.
    """
    if gate_mask is not None:
        g = np.asarray(gate_mask, dtype=bool)
        if len(g) == len(uncomp_df) and np.sum(g) >= min_events_retain:
            return uncomp_df.loc[g].copy(), comp_df.loc[g].copy(), "Viable Singlets (Gated)"

    mode = gating_mode.lower().strip() if gating_mode else "viable"
    if mode in ("all", "none", "raw", "ungated"):
        return uncomp_df, comp_df, "All Events (Ungated)"

    n_total = len(uncomp_df)
    if n_total < min_events_retain:
        return uncomp_df, comp_df, "All Events"

    mask = np.ones(n_total, dtype=bool)
    notes: List[str] = []
    cols = list(uncomp_df.columns)

    # 1. Detector Saturation / Margin Ceiling Filter (Universal Bit-Depth Handling)
    fluor_cols = [
        c for c in cols
        if c.lower() not in ("time", "time_qc")
        and not any(k in c.lower() for k in ("fsc", "ssc", "fs-", "ss-", "fsc_", "ssc_"))
    ]
    sat_count = 0
    for c in fluor_cols:
        v = uncomp_df[c].to_numpy()
        if len(v) == 0:
            continue
        p999 = float(np.nanpercentile(v, 99.99)) if np.any(np.isfinite(v)) else 262144.0
        if channel_ranges and c in channel_ranges:
            ceil_val = float(channel_ranges[c])
        elif p999 > 150000.0:
            ceil_val = 262143.0
        elif 40000.0 < p999 <= 70000.0:
            ceil_val = 65535.0
        elif p999 > 800000.0:
            ceil_val = 1048575.0
        else:
            ceil_val = p999

        if ceil_val > 20000.0:
            sat_thresh = ceil_val * 0.99
            is_sat = v >= sat_thresh
            sat_count += int(np.sum(is_sat & mask))
            mask &= ~is_sat

    if sat_count > 0:
        notes.append(f"Saturation Excluded ({sat_count:,})")

    # 2. Universal Debris & Optical Noise Floor Exclusion (Sample-Agnostic)
    fsc_a = next(
        (c for c in cols if "fsc" in c.lower() and ("-a" in c.lower() or "area" in c.lower())),
        None,
    ) or next((c for c in cols if "fsc" in c.lower()), None)
    ssc_a = next(
        (c for c in cols if "ssc" in c.lower() and ("-a" in c.lower() or "area" in c.lower())),
        None,
    ) or next((c for c in cols if "ssc" in c.lower()), None)

    if fsc_a and ssc_a:
        fsc = uncomp_df[fsc_a].to_numpy()
        ssc = uncomp_df[ssc_a].to_numpy()
        f_pos = fsc[fsc > 0]
        f_max = float(np.nanpercentile(f_pos, 99.9)) if len(f_pos) > 0 else 262144.0
        debris_cutoff = max(8000.0, min(25000.0, 0.06 * f_max))
        debris_mask = (fsc >= debris_cutoff) & (ssc >= 500.0)
        if np.sum(debris_mask & mask) >= min_events_retain:
            mask &= debris_mask
            notes.append("Intact Cells")

    if mode == "intact":
        summary = " → ".join(notes) if notes else "Intact Cells"
        return uncomp_df.loc[mask].copy(), comp_df.loc[mask].copy(), summary

    # 3. Universal Singlet Gate (Doublet / Clump Exclusion)
    fsc_h = next(
        (c for c in cols if "fsc" in c.lower() and ("-h" in c.lower() or "height" in c.lower())),
        None,
    )
    if fsc_a and fsc_h:
        fsc = uncomp_df[fsc_a].to_numpy()
        h = uncomp_df[fsc_h].to_numpy()
        valid_sing = (fsc > 5000.0) & (h > 5000.0) & mask
        if np.sum(valid_sing) >= min_events_retain:
            ratio = h / np.maximum(fsc, 1.0)
            med_r = float(np.nanmedian(ratio[valid_sing]))
            mad_r = float(np.nanmedian(np.abs(ratio[valid_sing] - med_r)))
            singlet_ok = np.abs(ratio - med_r) <= 3.5 * max(mad_r, 0.04)
            if np.sum(mask & singlet_ok) >= min_events_retain:
                mask &= singlet_ok
                notes.append("Singlets")

    if mode == "singlet":
        summary = " → ".join(notes) if notes else "Intact Singlets"
        return uncomp_df.loc[mask].copy(), comp_df.loc[mask].copy(), summary

    # 4. Viability Exclusion (Keyword-Driven, Marker-Agnostic)
    viab_ch = viability_channel
    if not viab_ch and marker_map:
        for ch, m in marker_map.items():
            if any(k in m.lower() for k in ("zombie", "viability", "live", "dead", "7aad", "7-aad", "dapi", "propidium", "fvs", "sytox")):
                viab_ch = ch
                break
    if not viab_ch:
        for ch in cols:
            if any(k in ch.lower() for k in ("7aad", "7-aad", "dapi", "zombie", "live", "dead", "viability", "pi-a", "fvs")):
                viab_ch = ch
                break

    if viab_ch and viab_ch in comp_df.columns:
        vc = comp_df[viab_ch].to_numpy()
        valid_v = mask & np.isfinite(vc)
        if np.sum(valid_v) >= min_events_retain:
            v_cut = float(np.nanpercentile(vc[valid_v], 75.0))
            live_ok = vc <= max(800.0, min(v_cut, 2500.0))
            if np.sum(mask & live_ok) >= min_events_retain:
                mask &= live_ok
                notes.append("Viable")
    elif mode == "viable":
        notes.append("Viable (Intact Singlets)")

    if np.sum(mask) < min_events_retain:
        return uncomp_df, comp_df, "All Events (Fallback)"

    summary = " → ".join(notes) if notes else "Viable Singlets"
    return uncomp_df.loc[mask].copy(), comp_df.loc[mask].copy(), summary


def plot_nxn_compensation_matrix(
    uncomp_df: Optional[pd.DataFrame] = None,
    comp_df: Optional[pd.DataFrame] = None,
    comp_matrix: Optional[CompensationMatrix] = None,
    spectral_library: Optional[Any] = None,
    channels: Optional[List[str]] = None,
    detector_map: Optional[Dict[str, str]] = None,
    marker_map: Optional[Dict[str, str]] = None,
    sample_name: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    max_events: int = 5000,
    max_channels: int = 12,
    dpi: int = 240,
    uncomp_color: str = "#d62728",
    comp_color: str = "#1f77b4",
    title: Optional[str] = None,
    lower_triangle_only: bool = True,
    auto_gate_sample: bool = True,
    gating_mode: str = "viable",
    gate_mask: Optional[np.ndarray] = None,
    viability_channel: Optional[str] = None,
    viable_only: bool = True,
    max_chars_per_line: Optional[int] = None,
    channel_ranges: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Plot an N x N compensation or spectral unmixing verification matrix.

    Renders a publication-ready corner matrix across fluorescence or unmixed channels:
    - Eliminates duplicated transposed subplots via lower-triangular corner presentation
      where each unique channel pair (i, j) is displayed exactly once.
    - Diagonal (i, i): 1D Logicle distribution comparing uncompensated / raw peak detector (red)
      vs. compensated / unmixed (blue) populations with a centered channel/marker badge.
    - Lower Triangle (i > j): Bivariate scatter plot of Channel j (X-axis) vs. Channel i (Y-axis)
      with orthogonal zero guidelines.
    - Upper Triangle (i < j): Houses global legend and sample status metadata.
    - Dual Flow Cytometry Compatibility:
      * Conventional Flow: overlays raw uncompensated vs compensated events on matching detectors.
      * Spectral Flow: pairs deconvolved fluorophore abundances with raw peak optical detectors,
        or visualizes the complete P x P unmixed orthogonal manifold.
    - High text-to-plot ratio: Compact subplots with bold, legible typography designed for
      publication composite figures. Dynamically wraps long channel and marker labels into
      multiple centered lines to avoid horizontal overlaps and vertical crowding.
    - Universal Gating: Filters debris, doublets, and detector saturation automatically when
      scatter channels are detected, with user-configurable gating levels (defaults to viable cells).

    Args:
        uncomp_df: Uncompensated or raw detector events DataFrame. If None, derived from comp_df.
        comp_df: Compensated or unmixed events DataFrame. If None, derived via comp_matrix.
        comp_matrix: Optional applied CompensationMatrix to derive comp_df if omitted.
        spectral_library: Optional ReferenceSpectralLibrary for spectral peak channel resolution.
        channels: Optional list of channel/fluorophore names. If None, auto-detected.
        detector_map: Optional mapping from fluorophore name to raw peak optical detector channel.
        marker_map: Optional dict mapping channel name to biological marker label.
        sample_name: Optional sample or control identifier for titling.
        output_path: Optional path to save the resulting figure.
        max_events: Maximum number of events to subsample for responsive rendering.
        max_channels: Maximum number of channels to render in a single corner matrix (default: 12).
            If a panel contains more than max_channels, selects the top channels by variance.
        dpi: Output image resolution.
        uncomp_color: Hex color for uncompensated / raw events (default: '#d62728').
        comp_color: Hex color for compensated / unmixed events (default: '#1f77b4').
        title: Optional figure super-title override.
        lower_triangle_only: If True (default), hides duplicated upper-triangle transposed subplots.
        auto_gate_sample: If True, applies universal sample pre-gating according to gating_mode.
        gating_mode: Pre-gating level ('viable', 'auto', 'singlet', 'intact', or 'all').
        gate_mask: Optional boolean array mask of pre-filtered viable singlet events.
        viability_channel: Optional explicit viability dye channel name.
        viable_only: If True (default), restricts events to viable singlets.
        max_chars_per_line: Optional line wrap width for channel and marker labels.
        channel_ranges: Optional dict of maximum detector dynamic ranges.

    Returns:
        The generated Matplotlib Figure.
    """
    from matplotlib.lines import Line2D

    if comp_df is None:
        if comp_matrix is not None and uncomp_df is not None:
            comp_df = comp_matrix.apply_dataframe(uncomp_df)
        elif uncomp_df is not None:
            comp_df = uncomp_df.copy()
        else:
            raise ValueError("Either comp_df or uncomp_df must be provided.")

    if uncomp_df is None:
        uncomp_df = comp_df.copy()

    # Extract detector map if spectral_library provided
    if spectral_library is not None and detector_map is None and hasattr(spectral_library, "get_peak_detector_map"):
        detector_map = spectral_library.get_peak_detector_map()

    non_fluor_cols = {
        "time", "Time", "TIME",
        "fsc-a", "fsc-h", "fsc-w", "fsc-area", "fsc-height", "fsc-width",
        "ssc-a", "ssc-h", "ssc-w", "ssc-area", "ssc-height", "ssc-width",
        "residual", "residuals", "residual_norm", "outlier", "is_outlier",
    }
    if channels is None:
        if comp_matrix is not None and hasattr(comp_matrix, "detectors"):
            candidate_cols = [
                c for c in comp_matrix.detectors
                if c in comp_df.columns and c.lower() not in non_fluor_cols
            ]
        elif spectral_library is not None and hasattr(spectral_library, "fluorophores"):
            candidate_cols = [
                c for c in spectral_library.fluorophores
                if c in comp_df.columns and c.lower() not in non_fluor_cols
            ]
        else:
            candidate_cols = [
                c for c in comp_df.columns
                if c.lower() not in non_fluor_cols
            ]

        # Cap large panels to max_channels based on variance
        if len(candidate_cols) > max_channels:
            num_cols = [c for c in candidate_cols if np.issubdtype(comp_df[c].dtype, np.number)]
            vars_s = comp_df[num_cols].var().sort_values(ascending=False)
            channels = list(vars_s.index[:max_channels])
        else:
            channels = candidate_cols
    else:
        channels = [c for c in channels if c in comp_df.columns and c.lower() not in non_fluor_cols]

    n_ch = len(channels)
    if n_ch < 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(
            0.5, 0.5,
            f"N x N matrix requires at least 2 channels (found {n_ch})",
            ha="center", va="center", fontsize=10,
        )
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        return fig

    # Ensure uncomp_df and comp_df row counts and indexing match
    if len(uncomp_df) != len(comp_df):
        min_events_aligned = min(len(uncomp_df), len(comp_df))
        uncomp_df = uncomp_df.iloc[:min_events_aligned].reset_index(drop=True)
        comp_df = comp_df.iloc[:min_events_aligned].reset_index(drop=True)
    elif not uncomp_df.index.equals(comp_df.index):
        uncomp_df = uncomp_df.reset_index(drop=True)
        comp_df = comp_df.reset_index(drop=True)

    # Determine mode: paired (conventional or mapped spectral) vs single (unmixed only)
    is_spectral = (
        spectral_library is not None
        or (detector_map is not None)
        or any(c in comp_df.columns and c not in uncomp_df.columns for c in channels)
    )

    u_col_map: Dict[str, Optional[str]] = {}
    is_paired_map: Dict[str, bool] = {}
    for ch in channels:
        if ch in uncomp_df.columns and (not is_spectral or detector_map is None):
            u_col_map[ch] = ch
            is_paired_map[ch] = True
        elif detector_map and ch in detector_map and detector_map[ch] in uncomp_df.columns:
            u_col_map[ch] = detector_map[ch]
            is_paired_map[ch] = True
        else:
            u_col_map[ch] = None
            is_paired_map[ch] = False

    has_paired_overlay = any(is_paired_map.values()) and (is_spectral or not uncomp_df.equals(comp_df))

    # Apply universal sample filtering (debris, saturation, singlet, and viability exclusion)
    if not auto_gate_sample:
        effective_gating_mode = "all"
    elif viable_only or gating_mode == "viable":
        effective_gating_mode = "viable"
    else:
        effective_gating_mode = gating_mode

    gate_note = "Pre-Gated Events"
    if gate_mask is not None:
        g = np.asarray(gate_mask, dtype=bool)
        if len(g) == len(uncomp_df) and np.sum(g) >= 50:
            uncomp_df = uncomp_df.loc[g].copy()
            comp_df = comp_df.loc[g].copy()
            gate_note = "Viable Singlets (Gated)"
        else:
            gate_mask = None

    if gate_mask is None and effective_gating_mode not in ("all", "none", "raw", "ungated"):
        has_scatter_u = any("fsc" in c.lower() for c in uncomp_df.columns)
        has_scatter_c = any("fsc" in c.lower() for c in comp_df.columns)
        if has_scatter_u:
            uncomp_df, comp_df, gate_note = _universal_sample_comp_filter(
                uncomp_df,
                comp_df,
                marker_map=marker_map,
                channel_ranges=channel_ranges,
                gating_mode=effective_gating_mode,
                viability_channel=viability_channel,
            )
        elif has_scatter_c:
            comp_df, uncomp_df, gate_note = _universal_sample_comp_filter(
                comp_df,
                uncomp_df,
                marker_map=marker_map,
                channel_ranges=channel_ranges,
                gating_mode=effective_gating_mode,
                viability_channel=viability_channel,
            )
        else:
            gate_note = "All Events (No Scatter Detectors)"
    elif gate_mask is None:
        gate_note = "All Events (Ungated)"

    # Subsample events if necessary for fast, crisp rendering
    n_events = len(uncomp_df)
    if n_events > max_events:
        rng = np.random.RandomState(42)
        sub_indices = rng.choice(n_events, size=max_events, replace=False)
        u_sub = uncomp_df.iloc[sub_indices]
        c_sub = comp_df.iloc[sub_indices]
    else:
        u_sub = uncomp_df
        c_sub = comp_df

    # Construct channel-adaptive Logicle transforms
    xforms: Dict[str, LogicleTransform] = {}
    ticks_dict: Dict[str, List[float]] = {}
    labels_dict: Dict[str, List[str]] = {}
    lims_dict: Dict[str, Tuple[float, float]] = {}
    zeros_dict: Dict[str, float] = {}
    u_transformed: Dict[str, Optional[np.ndarray]] = {}
    c_transformed: Dict[str, np.ndarray] = {}

    for ch in channels:
        c_vals = c_sub[ch].to_numpy().astype(np.float64)
        u_col = u_col_map.get(ch)
        if has_paired_overlay and u_col and u_col in u_sub.columns and is_paired_map[ch]:
            u_vals = u_sub[u_col].to_numpy().astype(np.float64)
            xform, tick_pos, tick_lbls, lims = _build_adaptive_biex_transform(u_vals, c_vals)
            u_transformed[ch] = xform.apply(u_vals)
        else:
            xform, tick_pos, tick_lbls, lims = _build_adaptive_biex_transform(c_vals, c_vals)
            u_transformed[ch] = None

        c_transformed[ch] = xform.apply(c_vals)
        xforms[ch] = xform
        ticks_dict[ch] = tick_pos
        labels_dict[ch] = tick_lbls
        lims_dict[ch] = lims
        zeros_dict[ch] = float(xform.apply(np.array([0.0]))[0])
        c_transformed[ch] = xform.apply(c_vals)

    # Dynamic multi-line label formatting to prevent axis collisions
    wrap_w = max_chars_per_line if max_chars_per_line is not None else max(10, min(16, int(60 / n_ch)))
    wrapped_labels = {
        ch: _format_multiline_channel_label(ch, marker_map=marker_map, max_width=wrap_w, include_detector=True)
        for ch in channels
    }
    max_x_lines = max((lbl.count("\n") + 1 for lbl in wrapped_labels.values()), default=1)
    max_y_lines = max_x_lines

    # Compact publication sizing with high text-to-plot ratio
    cell_sz = max(1.35, min(2.0, 7.8 / n_ch))
    fig_w = cell_sz * n_ch + 0.35 + (0.12 * max_y_lines)
    fig_h = cell_sz * n_ch + 0.35 + (0.12 * max_x_lines)

    tick_font_sz = max(6.0, min(8.8, 32.0 / n_ch))
    axis_font_sz = max(7.0, min(10.5, 36.0 / n_ch))
    diag_font_sz = max(6.5, min(9.5, 32.0 / n_ch))
    title_font_sz = max(9.5, min(13.5, 48.0 / n_ch))
    pt_sz = max(2.0, min(4.5, 15.0 / n_ch))
    line_w = max(1.0, min(1.5, 5.5 / n_ch))

    bottom_m = max(0.08, min(0.24, (0.35 + 0.16 * max_x_lines) / fig_h))
    left_m = max(0.08, min(0.24, (0.42 + 0.18 * max_y_lines) / fig_w))

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = gridspec.GridSpec(
        nrows=n_ch,
        ncols=n_ch,
        figure=fig,
        hspace=0.08,
        wspace=0.08,
        top=0.92,
        bottom=bottom_m,
        left=left_m,
        right=0.98,
    )

    for i in range(n_ch):
        ch_y = channels[i]
        for j in range(n_ch):
            ch_x = channels[j]
            ax = fig.add_subplot(gs[i, j])

            if lower_triangle_only and j > i:
                # Remove duplicated upper-triangle subplots
                ax.set_visible(False)
                continue

            apply_nature_axis_style(ax, show_grid=False, despine=False)

            if i == j:
                # Diagonal: 1D Distribution Histogram
                ax.axvline(zeros_dict[ch_x], color="#7f8c8d", linestyle="--", linewidth=0.7, alpha=0.55, zorder=1)
                cx = c_transformed[ch_x]
                x_lim = lims_dict[ch_x]

                if has_paired_overlay and u_transformed[ch_x] is not None:
                    ux = u_transformed[ch_x]
                    ax.hist(ux, bins=45, range=x_lim, density=True, color=uncomp_color, alpha=0.25, histtype="stepfilled", zorder=2)
                    ax.hist(ux, bins=45, range=x_lim, density=True, color=uncomp_color, alpha=0.85, histtype="step", linewidth=line_w, zorder=3)

                ax.hist(cx, bins=45, range=x_lim, density=True, color=comp_color, alpha=0.30, histtype="stepfilled", zorder=4)
                ax.hist(cx, bins=45, range=x_lim, density=True, color=comp_color, alpha=0.90, histtype="step", linewidth=line_w, zorder=5)

                m_lbl = wrapped_labels[ch_x]
                ax.text(
                    0.5, 0.74, m_lbl,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=diag_font_sz, fontweight="bold", color="#111827",
                    multialignment="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#d1d5db", alpha=0.90),
                    zorder=10,
                )
                ax.set_xlim(x_lim)
                ax.set_yticks([])
                ax.set_yticklabels([])
            else:
                # Lower triangle (or full off-diagonal): 2D Bivariate Scatter Overlay
                ax.axvline(zeros_dict[ch_x], color="#7f8c8d", linestyle="--", linewidth=0.6, alpha=0.55, zorder=1)
                ax.axhline(zeros_dict[ch_y], color="#7f8c8d", linestyle="--", linewidth=0.6, alpha=0.55, zorder=1)

                cx = c_transformed[ch_x]
                cy = c_transformed[ch_y]

                if has_paired_overlay and u_transformed[ch_x] is not None and u_transformed[ch_y] is not None:
                    ux = u_transformed[ch_x]
                    uy = u_transformed[ch_y]
                    ax.scatter(ux, uy, c=uncomp_color, s=pt_sz, alpha=0.32, edgecolors="none", rasterized=True, zorder=2)

                ax.scatter(cx, cy, c=comp_color, s=pt_sz, alpha=0.45, edgecolors="none", rasterized=True, zorder=3)

                ax.set_xlim(lims_dict[ch_x])
                ax.set_ylim(lims_dict[ch_y])

            # X-axis ticks & labels: bottom-most visible row (row n_ch - 1)
            if i == n_ch - 1:
                ax.set_xticks(ticks_dict[ch_x])
                ax.set_xticklabels(labels_dict[ch_x], fontsize=tick_font_sz)
                ax.set_xlabel(wrapped_labels[ch_x], fontsize=axis_font_sz, fontweight="bold", multialignment="center", labelpad=5)
            else:
                ax.set_xticks(ticks_dict[ch_x])
                ax.set_xticklabels([])
                ax.set_xlabel("")

            # Y-axis ticks & labels: leftmost visible column (column 0)
            if j == 0:
                ax.set_ylabel(wrapped_labels[ch_y], fontsize=axis_font_sz, fontweight="bold", multialignment="center", labelpad=6)
                if i == 0:
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                else:
                    ax.set_yticks(ticks_dict[ch_y])
                    ax.set_yticklabels(labels_dict[ch_y], fontsize=tick_font_sz)
            else:
                ax.set_ylabel("")
                if i != j:
                    ax.set_yticks(ticks_dict[ch_y])
                ax.set_yticklabels([])

            if not lower_triangle_only and i == 0:
                col_label = wrapped_labels[ch_x]
                ax.set_title(col_label, fontsize=axis_font_sz, fontweight="bold", pad=4, multialignment="center")
                ax.set_title(col_label, fontsize=axis_font_sz, fontweight="bold", pad=4)

    if is_spectral:
        label_u = "Raw Peak Detector"
        label_c = "Unmixed (WLS)"
        matrix_type = "Spectral Unmixing Validation Matrix"
    else:
        label_u = "Uncompensated"
        label_c = "Compensated"
        matrix_type = "Compensation Validation Matrix"

    if has_paired_overlay:
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label=label_u,
                   markerfacecolor=uncomp_color, markersize=6.5, alpha=0.85),
            Line2D([0], [0], marker="o", color="w", label=label_c,
                   markerfacecolor=comp_color, markersize=6.5, alpha=0.90),
            Line2D([0], [0], color="#7f8c8d", linestyle="--", linewidth=1.0, label="Zero Guideline"),
        ]
    else:
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="Unmixed Events",
                   markerfacecolor=comp_color, markersize=6.5, alpha=0.90),
            Line2D([0], [0], color="#7f8c8d", linestyle="--", linewidth=1.0, label="Zero Guideline"),
        ]

    if lower_triangle_only and n_ch >= 3:
        # Position legend in open upper-right area
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.96, 0.90),
            ncol=1,
            fontsize=max(6.5, min(9.0, 32.0 / n_ch)),
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#d1d5db",
            framealpha=0.95,
        )
        info_str = f"Status: {gate_note}\nUnique Pairs: {n_ch * (n_ch - 1) // 2}\nEvents Plotted: {len(u_sub):,}"
        fig.text(
            0.72, 0.72,
            info_str,
            ha="center", va="center",
            fontsize=max(6.0, min(8.0, 28.0 / n_ch)),
            color="#374151",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9fafb", edgecolor="#e5e7eb", alpha=0.9),
        )
    else:
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.992),
            ncol=3 if has_paired_overlay else 2,
            fontsize=max(6.0, min(8.5, 30.0 / n_ch)),
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#d1d5db",
            framealpha=0.92,
        )

    fig_title = title
    if fig_title is None:
        name_tag = f" — {sample_name}" if sample_name else ""
        pair_tag = "Unique Pairs" if lower_triangle_only else f"{n_ch}×{n_ch}"
        fig_title = f"{matrix_type} ({pair_tag}){name_tag}"

    fig.suptitle(
        fig_title,
        x=0.09,
        y=0.97 if lower_triangle_only else 0.992,
        ha="left",
        fontsize=title_font_sz,
        fontweight="bold",
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig



