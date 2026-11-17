"""Flow cytometry Quality Control (QC) package.

Provides automated fluidic anomaly detection, signal drift monitoring,
saturation/margin filtering, and manual time gating.
"""

from flowx.qc.automated import (
    compute_time_bins,
    extract_time_in_seconds,
    filter_flow_rate_anomalies,
    filter_margin_events,
    filter_signal_drift,
)
from flowx.qc.engine import apply_time_qc
from flowx.qc.manual import apply_manual_time_gate
from flowx.qc.models import (
    ChannelDriftStat,
    QCWorkflowDecision,
    TimeQCConfig,
    TimeQCResult,
    evaluate_qc_workflow_decision,
)
from flowx.qc.runner import TimeQCStepResult, run_cohort_time_qc, run_time_qc_step


def __getattr__(name: str):
    if name == "plot_time_qc_diagnostics":
        from flowx.visualization.qc_plots import plot_time_qc_diagnostics
        return plot_time_qc_diagnostics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TimeQCConfig",
    "TimeQCResult",
    "QCWorkflowDecision",
    "evaluate_qc_workflow_decision",
    "ChannelDriftStat",
    "apply_time_qc",
    "extract_time_in_seconds",
    "compute_time_bins",
    "filter_flow_rate_anomalies",
    "filter_signal_drift",
    "filter_margin_events",
    "apply_manual_time_gate",
    "plot_time_qc_diagnostics",
    "TimeQCStepResult",
    "run_time_qc_step",
    "run_cohort_time_qc",
]
