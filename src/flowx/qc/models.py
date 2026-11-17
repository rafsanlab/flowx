"""Data models and configuration containers for flow cytometry Quality Control (QC).

Defines configuration parameters and result containers for automated flow rate
stability, channel signal drift, margin/saturation filtering, and manual time gating.
Includes tailored presets for biological samples, single-stain compensation controls,
and Fluorescence Minus One (FMO) gating controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class ChannelDriftStat:
    """Statistics for an individual channel across time bins.

    Attributes:
        channel: Name of the channel.
        baseline_median: Median intensity across all events.
        baseline_mad: Median Absolute Deviation across all events.
        bin_medians: Array of median intensities per time bin.
        max_deviation_z: Maximum robust Z-score (deviation / MAD) observed.
        monotonic_trend_slope: Linear trend slope over time.
        is_flagged: True if channel exhibited significant drift or shift.
    """

    channel: str
    baseline_median: float
    baseline_mad: float
    bin_medians: np.ndarray
    max_deviation_z: float
    monotonic_trend_slope: float
    is_flagged: bool = False


@dataclass
class TimeQCConfig:
    """User-customizable configuration parameters for time-based quality control.

    Attributes:
        sample_type: Type of sample being analyzed ('sample', 'control'/'single_stain', or 'fmo').
        bin_width_sec: Target width of each time bin in seconds (default: 0.5s).
        events_per_bin: Optional target number of events per bin (overrides bin_width_sec if set).
        flow_rate_mad_factor: Multiplier for MAD to flag flow rate surges/drops (default: 3.5).
        flow_rate_min_eps: Minimum allowable events-per-second before flagging as a clog (default: 50.0).
        enable_signal_drift: Whether to monitor and filter multi-channel signal drift (default: True).
        signal_drift_mad_factor: Multiplier for MAD to flag abrupt channel intensity shifts (default: 3.5).
        signal_drift_slope_threshold: Absolute slope threshold for monotonic drift detection (default: 0.05).
        min_channels_drift_flag: Number of channels required to drift simultaneously to flag a bin (default: 1).
        trim_margins: If True, exclude events hitting upper saturation or zero floor (default: True).
        margin_tolerance: Fraction from hardware maximum range considered saturated (default: 0.001).
        manual_time_min: Optional minimum time (in seconds) for manual time gating.
        manual_time_max: Optional maximum time (in seconds) for manual time gating.
        manual_exclude_intervals: Optional list of (t_start, t_end) intervals to manually exclude.
        channels_to_evaluate: Optional list of channels to monitor for drift (None = auto-select).
        ignore_zero_bins_for_baseline: If True, exclude idle/empty pause bins when calculating median flow rate (default: True).
    """

    sample_type: str = "sample"
    bin_width_sec: float = 0.5
    events_per_bin: Optional[int] = None
    flow_rate_mad_factor: float = 3.5
    flow_rate_min_eps: float = 50.0
    enable_signal_drift: bool = True
    signal_drift_mad_factor: float = 3.5
    signal_drift_slope_threshold: float = 0.05
    min_channels_drift_flag: int = 1
    trim_margins: bool = True
    margin_tolerance: float = 0.001
    manual_time_min: Optional[float] = None
    manual_time_max: Optional[float] = None
    manual_exclude_intervals: Optional[List[Tuple[float, float]]] = None
    channels_to_evaluate: Optional[List[str]] = None
    pass_threshold_pct: float = 90.0
    warn_threshold_pct: float = 75.0
    ignore_zero_bins_for_baseline: bool = True

    @classmethod
    def for_sample(cls, **overrides: Any) -> TimeQCConfig:
        """Create a standard full QC configuration for heterogeneous biological samples.

        Applies full flow rate stability, multi-channel signal drift monitoring,
        and margin saturation filtering.
        """
        params: Dict[str, Any] = {
            "sample_type": "sample",
            "bin_width_sec": 0.5,
            "flow_rate_mad_factor": 3.5,
            "flow_rate_min_eps": 50.0,
            "enable_signal_drift": True,
            "signal_drift_mad_factor": 3.5,
            "trim_margins": True,
            "margin_tolerance": 0.001,
        }
        params.update(overrides)
        return cls(**params)

    @classmethod
    def for_controls(cls, **overrides: Any) -> TimeQCConfig:
        """Create a tailored QC configuration for single-stained compensation controls (beads or cells).

        Key adaptations:
        - Strict margin saturation trimming: Prevents PMT/detector saturation from skewing spillover slope.
        - Permissive flow rate MAD factor (4.5): Accommodates rapid, short-duration acquisition spikes.
        - Lower minimum flow rate (20.0 eps): Supports lower acquisition flow rates often used with beads.
        - Signal drift monitoring disabled: Avoids false-positive rejection of single-positive bead peaks.
        - Initial fluidic stabilization trimming (default: 1.0s) to discard SIP uptake turbulence.
        """
        params: Dict[str, Any] = {
            "sample_type": "control",
            "bin_width_sec": 0.25,
            "flow_rate_mad_factor": 4.5,
            "flow_rate_min_eps": 20.0,
            "enable_signal_drift": False,  # Bypassed: single-color controls lack multi-channel expression
            "trim_margins": True,           # Strictly required to prevent flattening compensation slope
            "margin_tolerance": 0.001,
            "manual_time_min": None,
        }
        params.update(overrides)
        return cls(**params)

    @classmethod
    def for_fmo(cls, **overrides: Any) -> TimeQCConfig:
        """Create a tailored QC configuration for Fluorescence Minus One (FMO) gating controls.

        Maintains rigorous flow rate and signal drift stability to avoid falsely inflating
        background gating boundaries.
        """
        params: Dict[str, Any] = {
            "sample_type": "fmo",
            "bin_width_sec": 0.5,
            "flow_rate_mad_factor": 3.5,
            "flow_rate_min_eps": 40.0,
            "enable_signal_drift": True,
            "signal_drift_mad_factor": 3.5,
            "trim_margins": True,
            "margin_tolerance": 0.001,
        }
        params.update(overrides)
        return cls(**params)

    @classmethod
    def from_sample_type(cls, sample_type: str, **overrides: Any) -> TimeQCConfig:
        """Resolve the appropriate QC configuration preset from a sample type string."""
        st = sample_type.lower().strip()
        if st in {"control", "controls", "single_stain", "singlestain", "comp_control"}:
            return cls.for_controls(**overrides)
        elif st in {"fmo", "fmo_control"}:
            return cls.for_fmo(**overrides)
        else:
            return cls.for_sample(**overrides)


@dataclass
class TimeQCResult:
    """Container holding results, boolean masks, and diagnostics from Time QC.

    Attributes:
        sample_name: Name of the processed FCS sample.
        sample_type: Type of sample ('sample', 'control', or 'fmo').
        total_events: Total input cell events.
        passed_events: Number of events passing all QC filters.
        rejected_events: Number of events excluded by any QC filter.
        passed_pct: Percentage of events passing QC (0.0 to 100.0).
        passed_mask: Boolean array of length total_events (True = passed).
        flow_rate_anomaly_mask: Boolean array (True = rejected due to flow rate anomaly).
        signal_drift_mask: Boolean array (True = rejected due to signal drift).
        margin_saturation_mask: Boolean array (True = rejected due to margin saturation).
        manual_time_mask: Boolean array (True = rejected due to manual time gating).
        duration_sec: Total acquisition duration in seconds.
        time_channel: Detector name for the time parameter.
        bin_stats_df: DataFrame summarizing metrics for each discrete time bin.
        drift_stats: Dictionary mapping channel name to ChannelDriftStat.
        qc_status: Status label ('PASS', 'WARNING', 'FAIL').
        qc_messages: List of human-readable warnings or findings.
    """

    sample_name: str
    sample_type: str
    total_events: int
    passed_events: int
    rejected_events: int
    passed_pct: float
    passed_mask: np.ndarray
    flow_rate_anomaly_mask: np.ndarray
    signal_drift_mask: np.ndarray
    margin_saturation_mask: np.ndarray
    manual_time_mask: np.ndarray
    duration_sec: float
    time_channel: str
    bin_stats_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    drift_stats: Dict[str, ChannelDriftStat] = field(default_factory=dict)
    qc_status: str = "PASS"
    qc_messages: List[str] = field(default_factory=list)

    def get_clean_dataframe(self, fcs_data: Any) -> pd.DataFrame:
        """Extract clean events DataFrame using passed_mask."""
        if hasattr(fcs_data, "events_df") and isinstance(fcs_data.events_df, pd.DataFrame):
            return fcs_data.events_df[self.passed_mask].copy()
        raise ValueError("Provided fcs_data does not contain a valid events_df DataFrame")

    def evaluate_workflow(
        self,
        fcs_data: Any,
        min_control_events: int = 500,
        min_sample_events: int = 5000,
    ) -> QCWorkflowDecision:
        """Evaluate workflow decision for proceeding downstream with this sample's clean events."""
        return evaluate_qc_workflow_decision(
            fcs_data, self, min_control_events=min_control_events, min_sample_events=min_sample_events
        )

    def to_summary_dict(self, decision: Optional[QCWorkflowDecision] = None) -> Dict[str, Any]:
        """Convert results to a flat dictionary for tabular summary and MIFlowCyt reporting."""
        flow_rate_rej = int(np.sum(self.flow_rate_anomaly_mask))
        drift_rej = int(np.sum(self.signal_drift_mask))
        margin_rej = int(np.sum(self.margin_saturation_mask))
        manual_rej = int(np.sum(self.manual_time_mask))

        action_code = (
            decision.action
            if decision
            else ("PROCEED_CLEAN" if self.qc_status == "PASS" else ("PROCEED_WITH_CAUTION" if self.qc_status == "WARNING" else "PROCEED_MASKED_SALVAGE"))
        )
        recommendation = decision.recommended_action if decision else ""

        return {
            "sample_name": self.sample_name,
            "sample_type": self.sample_type,
            "total_events": self.total_events,
            "passed_events": self.passed_events,
            "rejected_events": self.rejected_events,
            "passed_pct": round(self.passed_pct, 2),
            "flow_rate_anomalies": flow_rate_rej,
            "flow_rate_anomalies_pct": round((flow_rate_rej / self.total_events) * 100, 2) if self.total_events > 0 else 0.0,
            "signal_drift_anomalies": drift_rej,
            "signal_drift_anomalies_pct": round((drift_rej / self.total_events) * 100, 2) if self.total_events > 0 else 0.0,
            "margin_saturation_events": margin_rej,
            "margin_saturation_pct": round((margin_rej / self.total_events) * 100, 2) if self.total_events > 0 else 0.0,
            "manual_time_gated_events": manual_rej,
            "manual_time_gated_pct": round((manual_rej / self.total_events) * 100, 2) if self.total_events > 0 else 0.0,
            "duration_sec": round(self.duration_sec, 2),
            "time_channel": self.time_channel,
            "qc_status": self.qc_status,
            "workflow_action": action_code,
            "recommended_action": recommendation,
            "qc_messages": "; ".join(self.qc_messages) if self.qc_messages else "None",
        }


@dataclass
class QCWorkflowDecision:
    """Actionable workflow decision for proceeding downstream with QC-filtered events.

    Attributes:
        can_proceed: True if clean events meet statistical criteria to continue to downstream steps.
        action: Operational decision code:
            - 'PROCEED_CLEAN': Clean events pass with high fidelity.
            - 'PROCEED_WITH_CAUTION': Moderate event exclusion (WARNING), but sufficient clean events remain.
            - 'PROCEED_MASKED_SALVAGE': File failed QC overall, but sufficient clean events (>min_events)
              remain in stable intervals (e.g. salvaged from fluidic stalls or initial turbulence).
            - 'FALLBACK_INSTRUMENT_MATRIX': Single-stain control failed and remaining clean events are too sparse
              for reliable MFI estimation; recommend using embedded $SPILLOVER / $COMP matrix.
            - 'REJECT_SAMPLE': Biological sample failed and clean events fell below statistical threshold,
              or sample experienced severe multi-channel signal drift.
        clean_event_count: Number of events passing QC.
        clean_events_df: Filtered events DataFrame (fcs_data.events_df[qc_result.passed_mask]) if can_proceed, else None.
        rejection_reason: Explanation if rejected or falling back.
        recommended_action: Human-readable guidance on next operational steps.
    """

    can_proceed: bool
    action: str
    clean_event_count: int
    clean_events_df: Optional[pd.DataFrame] = None
    rejection_reason: Optional[str] = None
    recommended_action: str = ""


def evaluate_qc_workflow_decision(
    fcs_data: Any,
    qc_result: TimeQCResult,
    min_control_events: int = 500,
    min_sample_events: int = 5000,
) -> QCWorkflowDecision:
    """Evaluate whether and how to proceed downstream with QC-filtered events.

    Implements the recommended decision logic:
    - PASS: Proceed with masked clean events.
    - WARNING: Proceed with masked clean events, noting minor/moderate exclusions.
    - FAIL on Control:
        * If passed_events >= min_control_events (e.g. 500):
          PROCEED_MASKED_SALVAGE. The stable interval contains enough beads/cells for MFI estimation,
          and purging the bad events prevents compensation slope flattening or distortion.
        * If passed_events < min_control_events:
          FALLBACK_INSTRUMENT_MATRIX. Clean events too sparse for MFI calculation; fall back to
          embedded cytometer spillover matrix ($SPILLOVER / $COMP).
    - FAIL on Biological Sample:
        * If passed_events >= min_sample_events (e.g. 5,000) and no fatal multi-channel drift:
          PROCEED_MASKED_SALVAGE. Proceed with clean events under caution.
        * If passed_events < min_sample_events:
          REJECT_SAMPLE. Insufficient events for reliable sub-population resolution.
    """
    total = qc_result.total_events
    passed = qc_result.passed_events
    st = qc_result.sample_type.lower().strip()
    is_control = st in {"control", "controls", "single_stain", "singlestain", "comp_control"}

    # Extract clean events if events_df is present
    clean_df: Optional[pd.DataFrame] = None
    if hasattr(fcs_data, "events_df") and isinstance(fcs_data.events_df, pd.DataFrame):
        if len(qc_result.passed_mask) == len(fcs_data.events_df):
            clean_df = fcs_data.events_df[qc_result.passed_mask].copy()

    if qc_result.qc_status == "PASS":
        return QCWorkflowDecision(
            can_proceed=True,
            action="PROCEED_CLEAN",
            clean_event_count=passed,
            clean_events_df=clean_df,
            recommended_action=f"Proceed with {passed:,} clean events (high quality, {qc_result.passed_pct:.1f}% pass rate).",
        )

    if qc_result.qc_status == "WARNING":
        return QCWorkflowDecision(
            can_proceed=True,
            action="PROCEED_WITH_CAUTION",
            clean_event_count=passed,
            clean_events_df=clean_df,
            recommended_action=(
                f"Proceed with {passed:,} clean events ({qc_result.passed_pct:.1f}% pass rate). "
                f"Moderate exclusions ({total - passed:,} events purged); verify remaining count fits target populations."
            ),
        )

    # QC Status is FAIL or ERROR
    if is_control:
        if passed >= min_control_events:
            return QCWorkflowDecision(
                can_proceed=True,
                action="PROCEED_MASKED_SALVAGE",
                clean_event_count=passed,
                clean_events_df=clean_df,
                recommended_action=(
                    f"Proceed with {passed:,} clean events salvaged from stable interval. "
                    f"Sufficient beads ({passed:,} >= {min_control_events}) remain for robust MFI estimation; "
                    f"purged fluidic stall/saturation prevents slope distortion."
                ),
            )
        else:
            return QCWorkflowDecision(
                can_proceed=False,
                action="FALLBACK_INSTRUMENT_MATRIX",
                clean_event_count=passed,
                clean_events_df=clean_df,
                rejection_reason=(
                    f"Clean events after QC ({passed:,}) below minimum threshold ({min_control_events:,}) for MFI estimation."
                ),
                recommended_action=(
                    "Control sample too sparse for reliable spillover calculation. "
                    "Fall back to instrument-embedded $SPILLOVER / $COMP matrix or use matched control from another run."
                ),
            )
    else:
        # Biological sample
        # Check if fatal signal drift occurred (>25% events flagged in multi-channel drift)
        fatal_drift = (np.sum(qc_result.signal_drift_mask) / max(1, total)) > 0.25 if total > 0 else False
        if passed >= min_sample_events and not fatal_drift:
            return QCWorkflowDecision(
                can_proceed=True,
                action="PROCEED_MASKED_SALVAGE",
                clean_event_count=passed,
                clean_events_df=clean_df,
                recommended_action=(
                    f"Proceed with {passed:,} clean events salvaged from stable interval. "
                    f"Substantial events ({total - passed:,}) were excised due to fluidic instability; "
                    f"check diagnostic plot to ensure target cell morphology is preserved."
                ),
            )
        else:
            reason = (
                f"Clean events after QC ({passed:,}) below minimum threshold ({min_sample_events:,})"
                if passed < min_sample_events
                else "Severe multi-channel signal drift (>25% events affected) distorts fluorescence measurements."
            )
            return QCWorkflowDecision(
                can_proceed=False,
                action="REJECT_SAMPLE",
                clean_event_count=passed,
                clean_events_df=clean_df,
                rejection_reason=reason,
                recommended_action=(
                    "Sample quality severely compromised. "
                    "Re-acquire patient sample if possible, or document data exclusion in MIFlowCyt report."
                ),
            )
