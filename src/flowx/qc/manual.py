"""Manual Quality Control (Time Gating) module for flow cytometry.

Provides user-customizable manual interval time gating to exclude:
1. Initial fluidic turbulence / equilibration periods (e.g. first 2-5 seconds).
2. End-of-tube aspiration / air bubble ingestion (tube runout).
3. Specific user-identified irregular flow interruptions, clogs, or injection perturbations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flowx.qc.models import TimeQCConfig

logger = logging.getLogger(__name__)


def apply_manual_time_gate(
    time_sec: np.ndarray,
    config: TimeQCConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply user-specified manual time intervals to filter event data.

    Args:
        time_sec: 1D array of event timestamps in seconds.
        config: TimeQCConfig containing manual gating parameters.

    Returns:
        Tuple of (excluded_events_mask, manual_gate_stats).
    """
    event_count = len(time_sec)
    if event_count == 0:
        return np.empty(0, dtype=bool), {"excluded_count": 0, "excluded_pct": 0.0}

    excluded_mask = np.zeros(event_count, dtype=bool)
    reasons: List[str] = []

    # 1. Minimum start time bound (discard initial equilibration)
    if config.manual_time_min is not None and config.manual_time_min > 0.0:
        if float(time_sec.max()) > config.manual_time_min:
            start_cut = time_sec < config.manual_time_min
            excluded_mask = excluded_mask | start_cut
            count_start = int(np.sum(start_cut))
            if count_start > 0:
                reasons.append(f"Trimmed initial {count_start:,} events before t={config.manual_time_min:.1f}s")

    # 2. Maximum end time bound (discard tube aspiration / runout)
    if config.manual_time_max is not None:
        end_cut = time_sec > config.manual_time_max
        excluded_mask = excluded_mask | end_cut
        count_end = int(np.sum(end_cut))
        if count_end > 0:
            reasons.append(f"Trimmed final {count_end:,} events after t={config.manual_time_max:.1f}s")

    # 3. Specific excluded intervals [(t_start, t_end), ...]
    if config.manual_exclude_intervals:
        for idx, (t_start, t_end) in enumerate(config.manual_exclude_intervals):
            interval_cut = (time_sec >= t_start) & (time_sec <= t_end)
            excluded_mask = excluded_mask | interval_cut
            count_int = int(np.sum(interval_cut))
            if count_int > 0:
                reasons.append(f"Excluded interval #{idx+1} [{t_start:.1f}s - {t_end:.1f}s]: {count_int:,} events")

    total_excluded = int(np.sum(excluded_mask))
    stats_dict = {
        "manual_time_min": config.manual_time_min,
        "manual_time_max": config.manual_time_max,
        "manual_exclude_intervals": config.manual_exclude_intervals or [],
        "excluded_count": total_excluded,
        "excluded_pct": round((total_excluded / event_count) * 100, 2),
        "notes": "; ".join(reasons) if reasons else "None",
    }

    return excluded_mask, stats_dict
