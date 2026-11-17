"""Automated Fluorescence Minus One (FMO) calibration engine.

Provides automated matching, background spillover spread quantification,
and 99th/99.5th percentile positivity threshold determination from FMO controls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from flowx.gating.cleanup import CleanUpGateConfig, run_cleanup_gating
from flowx.gating.gate import GatingHierarchy, RangeGate
from flowx.io.parser import FCSData, parse_fcs
from flowx.math.transforms import apply_transform_to_dataframe, get_optimal_transform


@dataclass
class FMOCalibrationConfig:
    """Configuration for FMO threshold calibration.

    Attributes:
        percentile: Target percentile cutoff for positivity (default: 99.0 for 1% FPR).
        min_events: Minimum clean events required to compute empirical percentile.
        apply_cleanup: Whether to run automated debris and singlet clean-up on FMO files.
        fallback_threshold: Default cutoff if FMO control has insufficient events.
    """

    percentile: float = 99.0
    min_events: int = 50
    apply_cleanup: bool = True
    fallback_threshold: float = 0.50


@dataclass
class FMOCalibrationRecord:
    """Calibration details for a single fluorophore/marker FMO channel."""

    channel: str
    marker: str
    fmo_file: str
    event_count: int
    p95: float
    p99: float
    p999: float
    cutoff: float


@dataclass
class FMOCalibrationResult:
    """Complete FMO calibration result container."""

    channel_cutoffs: Dict[str, float] = field(default_factory=dict)
    marker_cutoffs: Dict[str, float] = field(default_factory=dict)
    records: List[FMOCalibrationRecord] = field(default_factory=list)

    @property
    def summary_df(self) -> pd.DataFrame:
        """Return a structured audit DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=["channel", "marker", "fmo_file", "event_count", "p95", "p99", "p999", "cutoff"])
        return pd.DataFrame([r.__dict__ for r in self.records])

    def get_cutoff(self, channel_or_marker: str, fallback: Optional[float] = None) -> float:
        """Retrieve calibrated positivity cutoff by channel or marker name."""
        if channel_or_marker in self.channel_cutoffs:
            return self.channel_cutoffs[channel_or_marker]
        if channel_or_marker in self.marker_cutoffs:
            return self.marker_cutoffs[channel_or_marker]
        if fallback is not None:
            return fallback
        raise KeyError(f"No FMO calibration cutoff found for '{channel_or_marker}'.")

    def build_positivity_gates(
        self,
        hierarchy: Optional[GatingHierarchy] = None,
        parent_gate: Optional[str] = None,
    ) -> GatingHierarchy:
        """Construct RangeGate positivity gates for all calibrated markers."""
        h = hierarchy if hierarchy is not None else GatingHierarchy()
        parent = parent_gate if (parent_gate and parent_gate in h.gates) else None
        for marker, cutoff in self.marker_cutoffs.items():
            ch = self.channel_cutoffs.get(marker, marker)
            gate_name = f"{marker}+"
            g = RangeGate(name=gate_name, channel=ch, min_val=cutoff, parent_gate=parent)
            h.add_gate(g)
        return h


def match_fmo_file_to_marker(
    filepath_or_name: Union[str, Path],
    available_markers: List[str],
    available_channels: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Infer target marker and channel from an FMO filename.

    Matches patterns like 'FMO80', 'FMO_CD80', 'FMO-CD206', 'FMO11b'.
    """
    stem = Path(filepath_or_name).stem.upper()

    # Direct substring matching
    for m in available_markers:
        m_clean = m.upper().replace("-", "").replace("_", "")
        # Remove 'CD' prefix if needed for fuzzy matching (e.g. FMO80 -> CD80)
        m_core = m_clean.replace("CD", "")
        pattern = rf"FMO[_\-\s]*({re.escape(m_clean)}|{re.escape(m_core)})\b"
        if re.search(pattern, stem) or f"FMO{m_clean}" in stem or f"FMO{m_core}" in stem:
            ch = m
            if available_channels and m not in available_channels:
                # Find corresponding detector channel
                for c in available_channels:
                    if m.lower() in c.lower():
                        ch = c
                        break
            return m, ch

    # Search detector channel names if provided
    if available_channels:
        for c in available_channels:
            c_clean = c.upper().replace("-", "").replace("_", "").replace(" ", "")
            if f"FMO{c_clean}" in stem:
                return c, c

    return None, None


def calibrate_fmo_thresholds(
    fmo_files: List[Union[str, Path]],
    marker_channel_map: Dict[str, str],
    config: Optional[FMOCalibrationConfig] = None,
) -> FMOCalibrationResult:
    """Calibrate positivity thresholds across a set of FMO FCS files.

    Args:
        fmo_files: List of file paths to FMO control FCS files.
        marker_channel_map: Mapping from canonical marker name to detector channel name
                            (e.g. {"CD80": "PE-CF594-A", "CD206": "BV605-A"}).
        config: Calibration configuration options.

    Returns:
        FMOCalibrationResult containing empirical cutoff values and audit records.
    """
    cfg = config if config is not None else FMOCalibrationConfig()
    result = FMOCalibrationResult()

    # Build bidirectional lookup maps
    marker_to_det: Dict[str, str] = {}
    det_to_marker: Dict[str, str] = {}
    for k, v in marker_channel_map.items():
        marker_to_det[k] = v
        marker_to_det[v] = k
        det_to_marker[v] = k
        det_to_marker[k] = v

    candidate_markers = list(marker_channel_map.keys()) + list(marker_channel_map.values())
    candidate_channels = list(marker_channel_map.values()) + list(marker_channel_map.keys())

    for fpath in fmo_files:
        p = Path(fpath)
        if not p.exists():
            continue

        target_marker, target_channel = match_fmo_file_to_marker(
            p.name,
            available_markers=candidate_markers,
            available_channels=candidate_channels,
        )

        if not target_marker:
            continue

        actual_channel = marker_to_det.get(target_marker, target_channel or target_marker)
        canonical_marker = det_to_marker.get(actual_channel, target_marker)

        # Parse and process FMO file
        fcs = parse_fcs(p)
        df = fcs.events_df.copy()

        # Apply compensation if spillover matrix exists
        if fcs.spillover_matrix is not None and fcs.spillover_channels:
            spill_ch = [c for c in fcs.spillover_channels if c in df.columns]
            if len(spill_ch) == fcs.spillover_matrix.shape[0]:
                inv_spill = np.linalg.inv(fcs.spillover_matrix)
                df[spill_ch] = df[spill_ch].values @ inv_spill.T

        # Apply Logicle transformation
        if actual_channel in df.columns:
            xform = get_optimal_transform(df[actual_channel].values)
            df = apply_transform_to_dataframe(df, [actual_channel], {actual_channel: xform})

        # Apply clean-up gating to isolate intact single events
        if cfg.apply_cleanup:
            try:
                c_res = run_cleanup_gating(fcs, df_events=fcs.events_df, df_transformed=df, config=CleanUpGateConfig())
                df = df[c_res.final_clean_mask]
            except Exception:
                pass  # Fallback to all events if clean-up fails

        vals = df[actual_channel].dropna().to_numpy()
        n_events = len(vals)

        if n_events < cfg.min_events:
            p95 = p99 = p999 = cutoff = cfg.fallback_threshold
        else:
            p95 = float(np.percentile(vals, 95.0))
            p99 = float(np.percentile(vals, 99.0))
            p999 = float(np.percentile(vals, 99.9))
            cutoff = float(np.percentile(vals, cfg.percentile))

        rec = FMOCalibrationRecord(
            channel=actual_channel,
            marker=canonical_marker,
            fmo_file=p.name,
            event_count=n_events,
            p95=round(p95, 4),
            p99=round(p99, 4),
            p999=round(p999, 4),
            cutoff=round(cutoff, 4),
        )

        result.records.append(rec)
        result.channel_cutoffs[actual_channel] = rec.cutoff
        result.marker_cutoffs[canonical_marker] = rec.cutoff

    return result
