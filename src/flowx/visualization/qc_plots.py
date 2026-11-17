"""Quality Control plotting module for flow cytometry.

Generates Nature-styled diagnostic visualizations for flow rate stability,
fluidic anomalies, signal drift, and manual time gating conforming to MIFlowCyt standards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flowx.io.parser import FCSData
from flowx.qc.models import TimeQCResult
from flowx.visualization.style import apply_nature_axis_style, set_nature_style

logger = logging.getLogger(__name__)


def plot_time_qc_diagnostics(
    fcs_data: FCSData,
    qc_result: TimeQCResult,
    output_path: Union[str, Path],
    dpi: int = 300,
) -> Path:
    """Generate publication-ready 3-panel Time Quality Control diagnostic figure.

    Panel A: Instantaneous flow rate (events/sec) vs Time with anomaly thresholds.
    Panel B: Forward scatter (FSC-A / FS INT) vs Time density showing event stability.
    Panel C: Channel median fluorescence / scatter trajectories over time bins.

    Args:
        fcs_data: FCSData instance.
        qc_result: TimeQCResult instance with computed masks and bin metrics.
        output_path: Target image file path (PNG, PDF, or SVG).
        dpi: Output resolution.

    Returns:
        Path object of saved figure.
    """
    set_nature_style()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Handle skipped Time QC (e.g. pre-gated files with no Time parameter)
    if qc_result.qc_status == "SKIPPED" or not fcs_data.get_time_channel():
        fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=dpi)
        ax.axis("off")
        msg = "\n".join(qc_result.qc_messages) if qc_result.qc_messages else "Time parameter not recorded in file."
        ax.text(
            0.5,
            0.5,
            f"Time Quality Control: Bypassed\n\nSample: {fcs_data.filename}\nTotal Events: {fcs_data.event_count:,} (100% Retained)\n\n{msg}",
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="medium",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.2),
        )
        fig.tight_layout()
        fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_p

    from flowx.qc.automated import extract_time_in_seconds
    time_sec, duration_sec, time_channel = extract_time_in_seconds(fcs_data)
    bin_df = qc_result.bin_stats_df

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=False, gridspec_kw={"hspace": 0.35})

    # -------------------------------------------------------------
    # PANEL A: Flow Rate (events/sec) vs Time
    # -------------------------------------------------------------
    ax_a = axes[0]
    apply_nature_axis_style(ax_a)

    if not bin_df.empty:
        t_mid = (bin_df["time_start_sec"].values + bin_df["time_end_sec"].values) / 2.0
        rates = bin_df["flow_rate_eps"].values
        is_anom = bin_df["flow_rate_anomaly"].values

        # Plot baseline line
        med_rate = float(np.median(rates)) if len(rates) else 0.0
        ax_a.plot(t_mid, rates, color="#1b4965", lw=1.2, label="Flow Rate (eps)", zorder=3)

        # Highlight anomalous bins
        if np.any(is_anom):
            ax_a.scatter(
                t_mid[is_anom],
                rates[is_anom],
                color="#e63946",
                s=20,
                zorder=4,
                label=f"Anomalous Bins ({int(np.sum(is_anom))})",
            )
            for idx in np.where(is_anom)[0]:
                ax_a.axvspan(
                    bin_df.loc[idx, "time_start_sec"],
                    bin_df.loc[idx, "time_end_sec"],
                    color="#e63946",
                    alpha=0.18,
                    zorder=1,
                )

        ax_a.axhline(med_rate, color="#2b2d42", ls="--", lw=1.0, alpha=0.8, label=f"Median ({med_rate:.0f} eps)")

        # Shading for manual exclusions
        if "manual_excluded" in bin_df.columns and np.any(bin_df["manual_excluded"]):
            for idx in np.where(bin_df["manual_excluded"])[0]:
                ax_a.axvspan(
                    bin_df.loc[idx, "time_start_sec"],
                    bin_df.loc[idx, "time_end_sec"],
                    color="#f4a261",
                    alpha=0.25,
                    zorder=2,
                )

    ax_a.set_ylabel("Flow Rate (eps)", fontsize=9, fontweight="bold")
    ax_a.set_title(
        f"A  Flow Rate Stability & Fluidic Quality Control  |  {qc_result.sample_name}",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    ax_a.set_xlabel("Acquisition Time (seconds)", fontsize=9)
    ax_a.legend(loc="upper right", frameon=True, fontsize=8)
    ax_a.grid(True, ls=":", alpha=0.4)

    # -------------------------------------------------------------
    # PANEL B: Forward Scatter vs Time Density
    # -------------------------------------------------------------
    ax_b = axes[1]
    apply_nature_axis_style(ax_b)

    fsc_col = fcs_data.get_fsc_channel()
    fsc_vals = fcs_data.events_df[fsc_col].values
    max_fsc = fcs_data.channel_ranges.get(fsc_col, float(fsc_vals.max() if len(fsc_vals) else 262144.0))

    # Downsample if large for responsive plotting
    n_pts = len(time_sec)
    if n_pts > 40000:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(n_pts, size=40000, replace=False)
        t_sub = time_sec[idx_sub]
        fsc_sub = fsc_vals[idx_sub]
        pass_sub = qc_result.passed_mask[idx_sub]
    else:
        t_sub = time_sec
        fsc_sub = fsc_vals
        pass_sub = qc_result.passed_mask

    # Plot passing events in deep blue, rejected in light coral/grey
    ax_b.scatter(
        t_sub[pass_sub],
        fsc_sub[pass_sub],
        c="#2a9d8f",
        s=1.0,
        alpha=0.4,
        rasterized=True,
        label="Passing Events",
    )
    if np.any(~pass_sub):
        ax_b.scatter(
            t_sub[~pass_sub],
            fsc_sub[~pass_sub],
            c="#e76f51",
            s=2.0,
            alpha=0.6,
            rasterized=True,
            label="Excluded (QC Anomaly / Saturation)",
        )

    ax_b.set_ylabel(f"{fcs_data.get_channel_label(fsc_col)}", fontsize=9, fontweight="bold")
    ax_b.set_title(
        f"B  Scatter Morphology vs. Time  |  FSC Dynamic Range [0 - {max_fsc:,.0f}]",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    ax_b.set_xlabel("Acquisition Time (seconds)", fontsize=9)
    ax_b.set_ylim(-0.02 * max_fsc, 1.05 * max_fsc)
    ax_b.legend(loc="upper right", frameon=True, fontsize=8, markerscale=4)
    ax_b.grid(True, ls=":", alpha=0.4)

    # -------------------------------------------------------------
    # PANEL C: Signal Drift / Running Channel Medians
    # -------------------------------------------------------------
    ax_c = axes[2]
    apply_nature_axis_style(ax_c)

    # Plot normalized median trajectories for evaluated channels
    colors = ["#264653", "#2a9d8f", "#e76f51", "#f4a261", "#457b9d", "#6d597a", "#b56576", "#355070"]
    c_idx = 0

    if qc_result.drift_stats and not bin_df.empty:
        t_mid = (bin_df["time_start_sec"].values + bin_df["time_end_sec"].values) / 2.0
        for ch, stat in qc_result.drift_stats.items():
            # Skip scatter if we have fluorescent channels
            if any(sc in ch.lower() for sc in ["fs", "ss"]) and len(qc_result.drift_stats) > 3:
                continue

            base = stat.baseline_median if abs(stat.baseline_median) > 1e-3 else 1.0
            norm_medians = (stat.bin_medians / base) * 100.0  # Percentage of baseline
            label = fcs_data.detector_to_marker.get(ch, ch)
            col = colors[c_idx % len(colors)]
            c_idx += 1

            ls = "-" if not stat.is_flagged else "--"
            ax_c.plot(t_mid, norm_medians, label=f"{label} (Δz={stat.max_deviation_z:.1f})", color=col, lw=1.2, ls=ls)
            if c_idx >= 7:
                break
        ax_c.axhline(100.0, color="#6c757d", ls=":", lw=1.0)
        ax_c.legend(loc="upper right", frameon=True, fontsize=7.5, ncol=2)
        panel_c_title = "C  Signal Drift Monitoring  |  Channel Median Intensities Across Time Bins"
    else:
        ax_c.text(
            0.5,
            0.5,
            "Signal Drift Monitoring Bypassed (Single-Stain Control Profile Active)\nProtects single-positive control beads from false-positive rejection.",
            ha="center",
            va="center",
            transform=ax_c.transAxes,
            fontsize=8.5,
            color="#495057",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#e9ecef", edgecolor="#ced4da", alpha=0.8),
        )
        panel_c_title = f"C  Signal Drift Monitoring  |  Bypassed for {qc_result.sample_type.replace('_', ' ').title()}"

    ax_c.set_ylabel("Signal Median (% Baseline)", fontsize=9, fontweight="bold")
    ax_c.set_title(
        panel_c_title,
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    ax_c.set_xlabel("Acquisition Time (seconds)", fontsize=9)
    ax_c.set_ylim(60.0, 140.0)
    ax_c.grid(True, ls=":", alpha=0.4)

    # Add summary statistics text box at top
    status_color = "#2a9d8f" if qc_result.qc_status == "PASS" else "#e76f51"
    st_badge = qc_result.sample_type.upper()
    fig.text(
        0.5,
        0.985,
        f"[{st_badge}] Status: {qc_result.qc_status}  |  Events Passing: {qc_result.passed_events:,} / {qc_result.total_events:,} ({qc_result.passed_pct:.1f}%)  |  "
        f"Flow Anomalies: {np.sum(qc_result.flow_rate_anomaly_mask):,}  |  Drift: {np.sum(qc_result.signal_drift_mask):,}  |  "
        f"Saturation: {np.sum(qc_result.margin_saturation_mask):,}  |  Duration: {duration_sec:.1f}s",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=status_color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", lw=0.8),
    )

    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.10, right=0.95, hspace=0.35)
    fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_p
