"""Adaptive morphological gating (Debris exclusion & Singlet discrimination).

Provides instrument-agnostic and sample-agnostic cell gating based on empirical
FSC/SSC density distributions and FSC-A vs FSC-H diagonal singlet discrimination.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from flowx.gating.gate import PolygonGate
from flowx.io.parser import FCSData


def build_adaptive_cell_gate(
    fcs_data: FCSData,
    df_events: pd.DataFrame,
) -> PolygonGate:
    """Construct a data-driven debris exclusion polygon gate for FSC-A vs SSC-A.

    Dynamically calculates bounding vertices based on the instrument's coordinate
    range and the sample's empirical scatter percentiles without tissue assumptions.

    Args:
        fcs_data: Parsed FCS sample data.
        df_events: Raw or transformed events DataFrame.

    Returns:
        PolygonGate configured for 'Cells'.
    """
    fsc_col = fcs_data.get_fsc_channel() if hasattr(fcs_data, "get_fsc_channel") else ("FSC-A" if "FSC-A" in df_events.columns else fcs_data.pnn_labels[0])
    ssc_col = fcs_data.get_ssc_channel() if hasattr(fcs_data, "get_ssc_channel") else ("SSC-A" if "SSC-A" in df_events.columns else (fcs_data.pnn_labels[1] if len(fcs_data.pnn_labels) > 1 else fcs_data.pnn_labels[0]))

    fsc_vals = df_events[fsc_col].to_numpy(dtype=np.float64)
    ssc_vals = df_events[ssc_col].to_numpy(dtype=np.float64)

    # Instrument max range for scatter channels
    fsc_max_scale = fcs_data.channel_ranges.get(fsc_col, float(np.max(fsc_vals)))
    ssc_max_scale = fcs_data.channel_ranges.get(ssc_col, float(np.max(ssc_vals)))

    if fsc_max_scale <= 0:
        fsc_max_scale = 262144.0
    if ssc_max_scale <= 0:
        ssc_max_scale = 262144.0

    # General adaptive scatter gate based on data percentiles
    fsc_min = float(max(fsc_max_scale * 0.18, np.percentile(fsc_vals, 8.0)))
    fsc_max = float(min(fsc_max_scale * 0.95, np.percentile(fsc_vals, 99.0)))
    ssc_min = float(max(ssc_max_scale * 0.05, np.percentile(ssc_vals, 3.0)))
    ssc_max = float(min(ssc_max_scale * 0.90, np.percentile(ssc_vals, 98.0)))

    vertices = [
        [fsc_min, ssc_min],
        [fsc_max, ssc_min * 1.3],
        [fsc_max, ssc_max],
        [fsc_min * 1.1, ssc_max],
        [fsc_min, ssc_max * 0.5],
    ]

    return PolygonGate(
        name="Cells",
        channels=[fsc_col, ssc_col],
        vertices=vertices,
        parent_gate=None,
    )


def build_adaptive_singlet_gate(
    fcs_data: FCSData,
    df_events: pd.DataFrame,
    parent_mask: Optional[np.ndarray] = None,
    tolerance: float = 0.25,
) -> PolygonGate:
    """Construct an adaptive singlet discrimination polygon gate for FSC-A vs FSC-H.

    Calculates the empirical linear pulse area/height ratio slope along the diagonal
    and encloses true single cells within an envelope of width `tolerance`,
    filtering out doublets and aggregates regardless of machine scale.

    Args:
        fcs_data: Parsed FCS sample data.
        df_events: Events DataFrame.
        parent_mask: Optional boolean mask of valid Cells.
        tolerance: Relative boundary width around diagonal (default 0.25 = +/- 25%).

    Returns:
        PolygonGate configured for 'Singlets'.
    """
    fsc_a = fcs_data.get_fsc_channel() if hasattr(fcs_data, "get_fsc_channel") else ("FSC-A" if "FSC-A" in df_events.columns else fcs_data.pnn_labels[0])
    fsc_h = fcs_data.get_fsc_h_channel() if hasattr(fcs_data, "get_fsc_h_channel") else ("FSC-H" if "FSC-H" in df_events.columns else None)

    # Fallback if instrument does not record FSC-H (e.g. FSC-W or Area only)
    if not fsc_h or fsc_h not in df_events.columns:
        # Return wide pass-through polygon
        fsc_max = fcs_data.channel_ranges.get(fsc_a, 262144.0)
        return PolygonGate(
            name="Singlets",
            channels=[fsc_a, fsc_a],
            vertices=[[0.0, 0.0], [fsc_max, 0.0], [fsc_max, fsc_max], [0.0, fsc_max]],
            parent_gate="Cells",
        )

    # Extract valid parent cell events for slope fitting
    if parent_mask is not None and np.sum(parent_mask) > 100:
        sub_a = df_events.loc[parent_mask, fsc_a].to_numpy(dtype=np.float64)
        sub_h = df_events.loc[parent_mask, fsc_h].to_numpy(dtype=np.float64)
    else:
        sub_a = df_events[fsc_a].to_numpy(dtype=np.float64)
        sub_h = df_events[fsc_h].to_numpy(dtype=np.float64)

    # Filter out zeros and extreme noise
    valid_mask = (sub_a > 1000.0) & (sub_h > 1000.0)
    if np.sum(valid_mask) > 50:
        ratios = sub_h[valid_mask] / sub_a[valid_mask]
        # Restrict to realistic ratio domain (0.2 to 2.5)
        clean_ratios = ratios[(ratios >= 0.2) & (ratios <= 2.5)]
        slope = float(np.median(clean_ratios)) if len(clean_ratios) > 50 else 0.85
    else:
        slope = 0.85

    x_min = float(max(10000.0, np.percentile(sub_a, 2.0)))
    x_max = float(min(fcs_data.channel_ranges.get(fsc_a, 262144.0), np.percentile(sub_a, 99.0)))

    # Construct singlet diagonal envelope
    lower_factor = max(0.5, 1.0 - tolerance)
    upper_factor = min(1.6, 1.0 + tolerance)

    y_low_start = float(max(0.0, x_min * slope * lower_factor))
    y_low_end = float(x_max * slope * lower_factor)
    y_high_end = float(x_max * slope * upper_factor)
    y_high_start = float(x_min * slope * upper_factor)

    vertices = [
        [x_min, y_low_start],
        [x_max, y_low_end],
        [x_max, y_high_end],
        [x_min, y_high_start],
    ]

    return PolygonGate(
        name="Singlets",
        channels=[fsc_a, fsc_h],
        vertices=vertices,
        parent_gate="Cells",
    )
