"""Automated Quality Control module for flow cytometry.

Implements native Python algorithms mirroring the core methodologies of flowAI,
flowCut, and PeacoQC:
1. Flow rate stability & fluidic anomaly detection (time binning & robust MAD limits).
2. Signal drift & instability detection (channel-wise running medians & baseline shifts).
3. Margin / hardware dynamic range saturation event filtering.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from flowx.io.parser import FCSData
from flowx.qc.models import ChannelDriftStat, TimeQCConfig

logger = logging.getLogger(__name__)


def extract_time_in_seconds(fcs_data: FCSData) -> Tuple[np.ndarray, float, str]:
    """Extract and normalize time parameter values to wall-clock seconds.

    Handles FlowKit preprocessed time scaling, direct acquisition duration keywords
    (@acqtime, @elapsedtime), begin/end clock timestamps ($BTIM / $ETIM), and detects
    non-monotonic time rollovers / concatenation anomalies.

    Args:
        fcs_data: Parsed FCSData object.

    Returns:
        Tuple of (time_in_seconds_array, total_duration_seconds, time_channel_name).
    """
    time_channel = fcs_data.get_time_channel()
    if not time_channel or time_channel not in fcs_data.events_df.columns:
        raise ValueError(f"Sample '{fcs_data.filename}' does not contain a recognized Time channel.")

    raw_time = fcs_data.events_df[time_channel].values.astype(np.float64)
    event_count = len(raw_time)
    if event_count == 0:
        return np.empty(0), 0.0, time_channel

    # Check for non-monotonic time resets / counter rollovers (T11)
    diffs = np.diff(raw_time)
    if np.any(diffs < 0):
        logger.warning(
            f"Sample '{fcs_data.filename}' contains non-monotonic time values (counter rollover or file concatenation). "
            f"Unwrapping time stamps to maintain chronological continuity."
        )
        unwrapped = np.zeros_like(raw_time)
        cum_offset = 0.0
        unwrapped[0] = raw_time[0]
        for i in range(1, len(raw_time)):
            if raw_time[i] < raw_time[i - 1]:
                cum_offset += (raw_time[i - 1] - raw_time[i]) + 0.01
            unwrapped[i] = raw_time[i] + cum_offset
        raw_time = unwrapped

    meta = fcs_data.metadata
    duration_sec = 0.0

    # 1. Check direct acquisition duration keywords
    for key in ["@acqtime", "@elapsedtime", "acqtime", "elapsedtime"]:
        if key in meta and str(meta[key]).strip():
            try:
                val = float(meta[key])
                if val > 0.0:
                    duration_sec = val
                    break
            except ValueError:
                pass

    # 2. Check $BTIM and $ETIM for clock-based duration fallback (T10)
    clock_duration = 0.0
    btim_str = str(meta.get("$btim", meta.get("btim", meta.get("$BTIM", meta.get("BTIM", ""))))).strip()
    etim_str = str(meta.get("$etim", meta.get("etim", meta.get("$ETIM", meta.get("ETIM", ""))))).strip()
    if btim_str and etim_str:
        try:
            def _parse_clock(s: str) -> float:
                parts = s.split(":")
                if len(parts) >= 3:
                    h = float(parts[0])
                    m = float(parts[1])
                    sec = float(parts[2])
                    return h * 3600.0 + m * 60.0 + sec
                return 0.0

            b_sec = _parse_clock(btim_str)
            e_sec = _parse_clock(etim_str)
            if e_sec >= b_sec:
                clock_duration = e_sec - b_sec
            else:
                clock_duration = (e_sec + 86400.0) - b_sec
        except Exception:
            pass

    # 3. Check $TIMESTEP
    timestep = None
    for key in ["$timestep", "timestep", "$TIMESTEP", "TIMESTEP"]:
        if key in meta and str(meta[key]).strip():
            try:
                val = float(meta[key])
                if val > 0.0:
                    timestep = val
                    break
            except ValueError:
                pass

    t_min = float(raw_time.min())
    t_max = float(raw_time.max())
    t_range = t_max - t_min

    # FlowKit scales time only when reading processed events, but raw_events (source="raw")
    # may or may not be scaled depending on cytometer export software.
    # Determine whether raw_time is in clock ticks needing timestep scaling:
    needs_timestep_scaling = False
    if timestep is not None and timestep > 0 and timestep < 1.0:
        if duration_sec > 0.0 and t_range > 0:
            diff_unscaled = abs(t_range - duration_sec)
            diff_scaled = abs(t_range * timestep - duration_sec)
            if diff_scaled < diff_unscaled and (diff_scaled / duration_sec) < 0.35:
                needs_timestep_scaling = True
        elif clock_duration > 0.0 and t_range > 0:
            diff_unscaled = abs(t_range - clock_duration)
            diff_scaled = abs(t_range * timestep - clock_duration)
            if diff_scaled < diff_unscaled and (diff_scaled / clock_duration) < 0.35:
                needs_timestep_scaling = True
        elif t_range > 7200.0 and (t_range * timestep) <= 7200.0:
            needs_timestep_scaling = True

    if duration_sec > 0.0 and t_range > 0:
        # Direct acqtime keyword present (e.g. Beckman / Kaluza @acqtime = 315.8)
        if abs(t_range - duration_sec) / duration_sec < 0.2:
            time_sec = raw_time - t_min
        elif needs_timestep_scaling:
            time_sec = (raw_time - t_min) * timestep
            duration_sec = float(time_sec.max())
        else:
            time_sec = ((raw_time - t_min) / t_range) * duration_sec
    elif clock_duration > 0.0 and t_range > 0:
        if abs(t_range - clock_duration) / clock_duration < 0.2:
            time_sec = raw_time - t_min
            duration_sec = clock_duration
        elif needs_timestep_scaling:
            time_sec = (raw_time - t_min) * timestep
            duration_sec = float(time_sec.max())
        else:
            time_sec = ((raw_time - t_min) / t_range) * clock_duration
            duration_sec = clock_duration
    elif timestep is not None and timestep > 0:
        if needs_timestep_scaling:
            time_sec = (raw_time - t_min) * timestep
            duration_sec = float(time_sec.max())
        else:
            time_sec = raw_time - t_min
            duration_sec = float(time_sec.max())
    elif t_range > 0:
        # Default fallback: if t_range looks like realistic acquisition duration in seconds (1s to 7200s)
        if 1.0 <= t_range <= 7200.0:
            time_sec = raw_time - t_min
            duration_sec = t_range
        else:
            # Assume 100 ticks per second standard FCS fallback
            time_sec = (raw_time - t_min) / 100.0
            duration_sec = float(time_sec.max())
    else:
        time_sec = np.zeros(event_count, dtype=np.float64)
        duration_sec = 1.0

    return time_sec, duration_sec, time_channel


def compute_time_bins(
    time_sec: np.ndarray,
    duration_sec: float,
    config: TimeQCConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Discretize event timestamps into sequential time bins.

    Supports both fixed-interval time binning (with exact last-bin duration accounting)
    and true equal-event binning (flowAI style).

    Args:
        time_sec: 1D array of event timestamps in seconds.
        duration_sec: Total duration in seconds.
        config: TimeQCConfig instance.

    Returns:
        Tuple of (bin_indices_per_event, bin_starts, bin_ends, bin_event_counts).
    """
    event_count = len(time_sec)
    if event_count == 0 or duration_sec <= 0:
        return np.empty(0, dtype=int), np.empty(0), np.empty(0), np.empty(0)

    max_t = max(duration_sec, float(time_sec.max()))

    if config.events_per_bin is not None and config.events_per_bin > 0:
        # True equal-event binning (T7)
        chunk_size = max(10, config.events_per_bin)
        n_bins = max(1, int(np.ceil(event_count / chunk_size)))
        bin_indices = np.minimum(n_bins - 1, np.arange(event_count) // chunk_size)
        bin_starts = np.zeros(n_bins, dtype=np.float64)
        bin_ends = np.zeros(n_bins, dtype=np.float64)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)

        for b in range(n_bins):
            b_mask = bin_indices == b
            if np.any(b_mask):
                bin_starts[b] = float(time_sec[b_mask][0])
                bin_ends[b] = float(time_sec[b_mask][-1])
            else:
                bin_starts[b] = bin_ends[max(0, b - 1)]
                bin_ends[b] = bin_starts[b] + 0.1

        # Ensure non-zero width for rate calculation
        bin_ends = np.maximum(bin_ends, bin_starts + 0.02)
        return bin_indices, bin_starts, bin_ends, bin_counts

    # Fixed-width time binning
    bin_width = max(0.05, config.bin_width_sec)
    n_bins = max(1, int(np.ceil(max_t / bin_width)))
    bin_edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)

    bin_indices = np.digitize(time_sec, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_starts = bin_edges[:-1].copy()
    bin_ends = bin_edges[1:].copy()

    # Exact last-bin duration accounting (T5): clamp end to actual acquisition finish
    if bin_ends[-1] > max_t and max_t > bin_starts[-1]:
        bin_ends[-1] = max_t

    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    return bin_indices, bin_starts, bin_ends, bin_counts


def filter_flow_rate_anomalies(
    bin_counts: np.ndarray,
    bin_starts: np.ndarray,
    bin_ends: np.ndarray,
    bin_indices: np.ndarray,
    config: TimeQCConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Detect fluidic instability, pressure spikes, clogs, and flow rate drops.

    Uses robust non-parametric statistics (Median and Median Absolute Deviation)
    to establish the expected steady-state flow rate envelope, identifying outlier bins.
    Adapts thresholds to slow acquisitions and accounts for partial final bin widths.

    Args:
        bin_counts: Array of event counts per time bin.
        bin_starts: Array of bin start times (sec).
        bin_ends: Array of bin end times (sec).
        bin_indices: Array mapping each event to its corresponding bin index.
        config: TimeQCConfig instance.

    Returns:
        Tuple of (anomalous_events_mask, anomalous_bins_mask, flow_rate_stats).
    """
    n_bins = len(bin_counts)
    bin_widths = bin_ends - bin_starts
    bin_widths = np.where(bin_widths > 0, bin_widths, 1.0)
    flow_rates_eps = bin_counts / bin_widths

    if n_bins < 3:
        empty_mask = np.zeros(len(bin_indices), dtype=bool)
        med_eps = float(np.median(flow_rates_eps) if len(flow_rates_eps) else 0.0)
        fallback_stats = {
            "median_eps": med_eps,
            "mad_eps": 0.0,
            "lower_threshold_eps": 0.0,
            "upper_threshold_eps": float("inf"),
            "anomalous_bin_count": 0,
            "total_bin_count": n_bins,
            "flow_rates_eps": flow_rates_eps,
        }
        return empty_mask, np.zeros(n_bins, dtype=bool), fallback_stats

    # Compute steady-state flow rate statistics
    # If substantial empty/pause bins exist (e.g. paused acquisition or tube swap),
    # estimate the steady-state baseline from active bins (>0 eps) to prevent pauses
    # from dragging the median down to near zero and rejecting genuine events.
    ignore_zero = getattr(config, "ignore_zero_bins_for_baseline", True)
    active_mask = flow_rates_eps > 0.0
    if ignore_zero and np.sum(active_mask) >= 3 and np.sum(~active_mask) > 0.05 * n_bins:
        active_eps = flow_rates_eps[active_mask]
        median_eps = float(np.median(active_eps))
        mad_eps = float(np.median(np.abs(active_eps - median_eps)))
    else:
        median_eps = float(np.median(flow_rates_eps))
        mad_eps = float(np.median(np.abs(flow_rates_eps - median_eps)))

    robust_sigma = 1.4826 * mad_eps

    # Bounds for allowable steady-state flow rate (T4: relative floor for slow acquisitions)
    k = config.flow_rate_mad_factor
    if median_eps < config.flow_rate_min_eps:
        # For low-flow or rare-cell acquisitions, make minimum relative to observed median
        effective_min = max(0.5, 0.2 * median_eps)
    else:
        effective_min = config.flow_rate_min_eps

    lower_threshold = max(effective_min, median_eps - k * robust_sigma)
    upper_threshold = median_eps + k * robust_sigma

    anomalous_bins = (flow_rates_eps < lower_threshold) | (flow_rates_eps > upper_threshold)

    # Flag clogs: bins with flow rate dropping near zero relative to steady state
    clog_threshold = max(1.0, 0.15 * median_eps) if median_eps > 10.0 else 0.5
    clog_bins = flow_rates_eps < clog_threshold
    anomalous_bins = anomalous_bins | clog_bins

    # T5: If the last bin is very narrow (<20% of target bin width), exempt it from false low-rate flagging
    if n_bins >= 2 and bin_widths[-1] < 0.2 * config.bin_width_sec:
        anomalous_bins[-1] = False

    anomalous_events_mask = anomalous_bins[bin_indices]

    stats_dict = {
        "median_eps": median_eps,
        "mad_eps": mad_eps,
        "lower_threshold_eps": lower_threshold,
        "upper_threshold_eps": upper_threshold,
        "anomalous_bin_count": int(np.sum(anomalous_bins)),
        "total_bin_count": n_bins,
        "flow_rates_eps": flow_rates_eps,
    }

    return anomalous_events_mask, anomalous_bins, stats_dict


def filter_signal_drift(
    events_df: pd.DataFrame,
    bin_indices: np.ndarray,
    n_bins: int,
    channels: List[str],
    config: TimeQCConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, ChannelDriftStat]]:
    """Detect fluorescence intensity drifts, laser power fluctuations, and fluidic shifts across time.

    Evaluates running medians across time bins for each channel. Bins displaying
    statistically significant shifts exceeding robust MAD thresholds of the bin medians
    sequence or monotonic trend drift are flagged.

    Args:
        events_df: DataFrame of event channel intensities.
        bin_indices: Array mapping each event to its bin index.
        n_bins: Total number of time bins.
        channels: List of channel names to evaluate.
        config: TimeQCConfig instance.

    Returns:
        Tuple of (anomalous_events_mask, anomalous_bins_mask, drift_stats_dict).
    """
    event_count = len(events_df)
    if event_count == 0 or n_bins < 3 or not channels:
        return np.zeros(event_count, dtype=bool), np.zeros(n_bins, dtype=bool), {}

    bin_flags_per_channel = np.zeros((len(channels), n_bins), dtype=bool)
    drift_stats: Dict[str, ChannelDriftStat] = {}

    for idx, ch in enumerate(channels):
        if ch not in events_df.columns:
            continue

        vals = events_df[ch].values.astype(np.float64)
        baseline_med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - baseline_med)))

        # Calculate median in each bin
        bin_medians = np.zeros(n_bins, dtype=np.float64)
        for b in range(n_bins):
            b_mask = bin_indices == b
            if np.any(b_mask):
                bin_medians[b] = np.median(vals[b_mask])
            else:
                bin_medians[b] = baseline_med

        # T6: Assess variation of the bin medians themselves across time
        seq_med = float(np.median(bin_medians))
        seq_mad = float(np.median(np.abs(bin_medians - seq_med)))
        baseline_sigma = 1.4826 * mad if mad > 1e-4 else 1.0

        # Effective sigma for bin medians: bounds seq_mad so that ultra-steady acquisitions
        # with small seq_mad do not produce false alarms on tiny sub-percent variations.
        bin_sigma = max(
            1.4826 * seq_mad,
            0.20 * baseline_sigma,
            0.05 * max(1.0, abs(seq_med)),
        )

        deviations = np.abs(bin_medians - seq_med)
        z_scores = deviations / bin_sigma
        max_z = float(np.max(z_scores)) if len(z_scores) else 0.0

        # Check for monotonic drift using linear regression slope
        x_norm = np.linspace(0.0, 1.0, n_bins)
        denom_scale = abs(seq_med) if abs(seq_med) > 1.0 else 1.0
        if len(bin_medians) > 2:
            slope, _, _, _, _ = stats.linregress(x_norm, bin_medians / denom_scale)
        else:
            slope = 0.0

        # Flag bins exceeding MAD threshold
        channel_anom_bins = z_scores > config.signal_drift_mad_factor

        # If significant monotonic slope detected (>= 20% shift across acquisition),
        # flag bins that diverge significantly from the sequence median
        slope_thresh = max(0.20, config.signal_drift_slope_threshold)
        if abs(slope) > slope_thresh:
            drift_tail = (deviations > (2.5 * bin_sigma)) & (z_scores > 2.5)
            channel_anom_bins = channel_anom_bins | drift_tail

        bin_flags_per_channel[idx, :] = channel_anom_bins

        is_flagged = bool(np.any(channel_anom_bins) or abs(slope) > slope_thresh)
        drift_stats[ch] = ChannelDriftStat(
            channel=ch,
            baseline_median=baseline_med,
            baseline_mad=mad,
            bin_medians=bin_medians,
            max_deviation_z=max_z,
            monotonic_trend_slope=float(slope),
            is_flagged=is_flagged,
        )

    # Bin is anomalous if at least min_channels_drift_flag channels show drift (default 2 to prevent single-channel noise false positives)
    min_req = min(config.min_channels_drift_flag, len(drift_stats))
    bins_drift_count = np.sum(bin_flags_per_channel, axis=0)
    anomalous_bins = bins_drift_count >= max(1, min_req)
    anomalous_events_mask = anomalous_bins[bin_indices]

    return anomalous_events_mask, anomalous_bins, drift_stats


def filter_margin_events(
    events_df: pd.DataFrame,
    channel_ranges: Dict[str, float],
    channels: List[str],
    config: TimeQCConfig,
) -> np.ndarray:
    """Identify detector saturation (upper range ceiling) and sensor zero-floor events.

    Detects both hardware-specified maximum limits ($PnR / $PnG) and empirical data
    saturation ceilings / negative clamping spikes.

    Args:
        events_df: DataFrame of event channel values.
        channel_ranges: Mapping of channel name to hardware maximum range ($PnR / $PnG).
        channels: Channels to inspect for saturation.
        config: TimeQCConfig instance.

    Returns:
        Boolean array of length len(events_df), True if event is a margin artifact.
    """
    event_count = len(events_df)
    if not config.trim_margins or event_count == 0:
        return np.zeros(event_count, dtype=bool)

    margin_mask = np.zeros(event_count, dtype=bool)

    for ch in channels:
        if ch not in events_df.columns:
            continue

        vals = events_df[ch].values
        pnr = channel_ranges.get(ch, 262144.0)

        # Upper limit saturation: events hitting hardware ceiling or piling up at max
        ch_max = float(np.max(vals)) if len(vals) > 0 else 0.0
        effective_pnr = pnr
        if ch_max > 1.15 * pnr:
            # Data exceeds pnr (common in 32-bit unmixed data with legacy $PnR keywords like 100,000)
            effective_pnr = max(4194304.0, ch_max)

        upper_bound = effective_pnr * (1.0 - config.margin_tolerance)
        upper_sat = vals >= upper_bound

        # Empirical ceiling check: if max value is near effective_pnr and events pile up at max
        if ch_max >= 0.95 * effective_pnr and ch_max > 100.0:
            sat_empirical = vals >= (ch_max * (1.0 - config.margin_tolerance))
            if np.sum(sat_empirical) >= 5:
                upper_sat = upper_sat | sat_empirical

        # Lower boundary clipping (T3):
        # Digital cytometers clamp negative fluctuations at a fixed lower ceiling (e.g. -111 or instrument min).
        # Sensor zero-drop files clamp events at 0.0.
        if np.any(vals < 0.0):
            ch_min = float(np.min(vals))
            # If negative pile-up at the lower clamp limit
            if ch_min < -10.0:
                clamp_floor = vals <= (ch_min + 1.0)
                if np.sum(clamp_floor) >= 5:
                    lower_floor = clamp_floor
                else:
                    lower_floor = vals <= -effective_pnr * (1.0 - config.margin_tolerance)
            else:
                lower_floor = np.zeros(event_count, dtype=bool)
        else:
            # Non-negative dataset: flag sensor zero-dropouts / bottom margins
            lower_floor = vals <= 0.0

        margin_mask = margin_mask | upper_sat | lower_floor

    return margin_mask
