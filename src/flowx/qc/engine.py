"""Quality Control Engine orchestrating automated and manual time gating.

Integrates flow rate stability, signal drift monitoring, margin saturation trimming,
and manual time gating into a unified, modular pipeline conforming to MIFlowCyt standards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from flowx.io.parser import FCSData
from flowx.qc.automated import (
    compute_time_bins,
    extract_time_in_seconds,
    filter_flow_rate_anomalies,
    filter_margin_events,
    filter_signal_drift,
)
from flowx.qc.manual import apply_manual_time_gate
from flowx.qc.models import TimeQCConfig, TimeQCResult

logger = logging.getLogger(__name__)


def _is_time_channel(name: str) -> bool:
    """Determine whether channel represents acquisition time."""
    clean = name.strip().lower()
    return clean == "time" or clean.startswith("time") or clean in ("time", "time [s]", "time (sec)")


def _is_scatter_channel(name: str) -> bool:
    """Determine whether channel represents morphological scatter (FSC/SSC)."""
    clean = name.strip().lower()
    scatter_stems = ("fsc", "ssc", "fs ", "ss ", "fs.", "ss.", "fs-", "ss-", "fs_", "ss_", "forward", "side")
    return clean.startswith(scatter_stems) or clean in ("fs", "ss", "fsc", "ssc")


def apply_time_qc(
    fcs_data: FCSData,
    config: Optional[TimeQCConfig] = None,
) -> TimeQCResult:
    """Execute end-to-end automated and manual Time Quality Control on an FCS sample.

    Args:
        fcs_data: Parsed FCSData object containing events_df and metadata.
        config: Optional TimeQCConfig configuration override.

    Returns:
        Populated TimeQCResult object containing masks, metrics, and bin statistics.
    """
    cfg = config or TimeQCConfig()
    total_events = fcs_data.event_count

    if total_events == 0:
        empty_mask = np.empty(0, dtype=bool)
        return TimeQCResult(
            sample_name=fcs_data.filename,
            sample_type=cfg.sample_type,
            total_events=0,
            passed_events=0,
            rejected_events=0,
            passed_pct=0.0,
            passed_mask=empty_mask,
            flow_rate_anomaly_mask=empty_mask,
            signal_drift_mask=empty_mask,
            margin_saturation_mask=empty_mask,
            manual_time_mask=empty_mask,
            duration_sec=0.0,
            time_channel="",
            qc_status="FAIL",
            qc_messages=["Sample contains 0 events"],
        )

    # 1. Extract and normalize time parameter
    time_ch = fcs_data.get_time_channel()
    if not time_ch or time_ch not in fcs_data.events_df.columns:
        logger.info(
            f"Sample '{fcs_data.filename}' does not contain an acquisition Time channel (pre-gated or external export). "
            f"Time QC fluidic stability evaluation bypassed."
        )
        empty_mask = np.zeros(event_count, dtype=bool)
        full_mask = np.ones(event_count, dtype=bool)
        return TimeQCResult(
            sample_name=fcs_data.filename,
            sample_type=cfg.sample_type,
            total_events=event_count,
            passed_events=event_count,
            rejected_events=0,
            passed_pct=100.0,
            passed_mask=full_mask,
            flow_rate_anomaly_mask=empty_mask,
            signal_drift_mask=empty_mask,
            margin_saturation_mask=empty_mask,
            manual_time_mask=empty_mask,
            duration_sec=0.0,
            time_channel="",
            qc_status="SKIPPED",
            qc_messages=["Time channel not present in FCS file (pre-gated export). Fluidic QC bypassed."],
        )

    time_sec, duration_sec, time_channel = extract_time_in_seconds(fcs_data)

    # 2. Compute time bins
    bin_indices, bin_starts, bin_ends, bin_counts = compute_time_bins(time_sec, duration_sec, cfg)
    n_bins = len(bin_counts)

    # 3. Automated Flow Rate QC (flowAI / flowCut equivalent)
    fr_mask, fr_bins, fr_stats = filter_flow_rate_anomalies(
        bin_counts=bin_counts,
        bin_starts=bin_starts,
        bin_ends=bin_ends,
        bin_indices=bin_indices,
        config=cfg,
    )

    # Resolve default fluorescence channels for signal drift and margin inspection:
    # Margin saturation and signal drift monitoring target fluorescence channels (flowAI / PeacoQC standard).
    # Morphological scatter channels (FSC/SSC) are handled during Step 2 Core Clean-up (debris & singlets).
    if cfg.channels_to_evaluate:
        eval_channels = [c for c in cfg.channels_to_evaluate if c in fcs_data.pnn_labels]
    else:
        eval_channels = [
            c for c in fcs_data.pnn_labels
            if not _is_time_channel(c) and not _is_scatter_channel(c)
        ]
        # Fallback for scatter-only test datasets
        if not eval_channels:
            eval_channels = [c for c in fcs_data.pnn_labels if not _is_time_channel(c)]

    # 4. Automated Signal Drift QC (PeacoQC / flowAI equivalent)
    if cfg.enable_signal_drift:
        drift_mask, drift_bins, drift_stats = filter_signal_drift(
            events_df=fcs_data.events_df,
            bin_indices=bin_indices,
            n_bins=n_bins,
            channels=eval_channels,
            config=cfg,
        )
    else:
        drift_mask = np.zeros(total_events, dtype=bool)
        drift_bins = np.zeros(n_bins, dtype=bool)
        drift_stats = {}

    # 5. Margin / Dynamic Range Saturation Filtering
    margin_mask = filter_margin_events(
        events_df=fcs_data.events_df,
        channel_ranges=fcs_data.channel_ranges,
        channels=eval_channels,
        config=cfg,
    )

    # 6. Manual Time Gating (Interval bounds and exclusions)
    manual_mask, manual_stats = apply_manual_time_gate(
        time_sec=time_sec,
        config=cfg,
    )

    # 7. Synthesize composite QC outcome
    rejected_mask = fr_mask | drift_mask | margin_mask | manual_mask
    passed_mask = ~rejected_mask

    passed_count = int(np.sum(passed_mask))
    rejected_count = int(np.sum(rejected_mask))
    passed_pct = float((passed_count / total_events) * 100.0) if total_events > 0 else 0.0

    # Build per-bin statistics table
    bin_widths = bin_ends - bin_starts
    bin_widths = np.where(bin_widths > 0, bin_widths, 1.0)
    bin_eps = bin_counts / bin_widths

    bin_manual_excluded = np.zeros(n_bins, dtype=bool)
    if cfg.manual_time_min is not None:
        bin_manual_excluded = bin_manual_excluded | (bin_starts < cfg.manual_time_min)
    if cfg.manual_time_max is not None:
        bin_manual_excluded = bin_manual_excluded | (bin_ends > cfg.manual_time_max)
    if cfg.manual_exclude_intervals:
        for t_start, t_end in cfg.manual_exclude_intervals:
            bin_manual_excluded = bin_manual_excluded | ((bin_starts < t_end) & (bin_ends > t_start))

    bin_composite_anom = fr_bins | drift_bins | bin_manual_excluded

    bin_df = pd.DataFrame(
        {
            "bin_index": np.arange(n_bins),
            "time_start_sec": np.round(bin_starts, 2),
            "time_end_sec": np.round(bin_ends, 2),
            "event_count": bin_counts,
            "flow_rate_eps": np.round(bin_eps, 1),
            "flow_rate_anomaly": fr_bins,
            "signal_drift_anomaly": drift_bins,
            "manual_excluded": bin_manual_excluded,
            "is_anomalous_bin": bin_composite_anom,
        }
    )

    # Determine QC Status and Warnings using configurable thresholds (T12)
    pass_thresh = getattr(cfg, "pass_threshold_pct", 90.0)
    warn_thresh = getattr(cfg, "warn_threshold_pct", 75.0)
    qc_messages: List[str] = []
    if passed_pct >= pass_thresh:
        qc_status = "PASS"
    elif passed_pct >= warn_thresh:
        qc_status = "WARNING"
        qc_messages.append(f"Moderate event exclusion ({100.0 - passed_pct:.1f}% rejected)")
    else:
        qc_status = "FAIL"
        qc_messages.append(f"High event exclusion ({100.0 - passed_pct:.1f}% rejected)")

    fr_anom_pct = (np.sum(fr_mask) / total_events) * 100.0
    if fr_anom_pct > 5.0:
        qc_messages.append(f"Fluidic flow rate instability detected ({fr_anom_pct:.1f}% events in unstable intervals)")

    drift_anom_pct = (np.sum(drift_mask) / total_events) * 100.0
    if drift_anom_pct > 5.0:
        qc_messages.append(f"Channel signal drift detected ({drift_anom_pct:.1f}% events flagged)")

    margin_anom_pct = (np.sum(margin_mask) / total_events) * 100.0
    if margin_anom_pct > 2.0:
        qc_messages.append(f"Boundary / saturation events detected ({margin_anom_pct:.1f}% events)")

    logger.info(
        f"[{fcs_data.filename}] Time QC Completed: {passed_count:,}/{total_events:,} events passed ({passed_pct:.1f}%). "
        f"Flow rate: {fr_stats['median_eps']:.1f} eps (MAD: {fr_stats['mad_eps']:.1f}). Status: {qc_status}"
    )

    return TimeQCResult(
        sample_name=fcs_data.filename,
        sample_type=cfg.sample_type,
        total_events=total_events,
        passed_events=passed_count,
        rejected_events=rejected_count,
        passed_pct=passed_pct,
        passed_mask=passed_mask,
        flow_rate_anomaly_mask=fr_mask,
        signal_drift_mask=drift_mask,
        margin_saturation_mask=margin_mask,
        manual_time_mask=manual_mask,
        duration_sec=duration_sec,
        time_channel=time_channel,
        bin_stats_df=bin_df,
        drift_stats=drift_stats,
        qc_status=qc_status,
        qc_messages=qc_messages,
    )
