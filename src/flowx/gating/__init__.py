"""Gating package for flow cytometry population filtering."""

from flowx.gating.adaptive_morphology import (
    build_adaptive_cell_gate,
    build_adaptive_singlet_gate,
)
from flowx.gating.cleanup import (
    CleanUpGateConfig,
    CleanUpGatingResult,
    build_debris_gate,
    build_dump_gate,
    build_singlet_gate,
    build_viability_gate,
    detect_dump_channel,
    detect_viability_channel,
    resolve_scatter_channels,
    run_cleanup_gating,
)
from flowx.gating.density_gating import (
    DensityCutoffResult,
    QuadrantCutoffResult,
    build_adaptive_density_hierarchy,
    find_density_minimum,
    find_quadrant_cutoffs,
)
from flowx.gating.fmo import (
    FMOCalibrationConfig,
    FMOCalibrationRecord,
    FMOCalibrationResult,
    calibrate_fmo_thresholds,
)
from flowx.gating.gate import (
    Gate,
    GatingHierarchy,
    PolygonGate,
    QuadrantGate,
    RangeGate,
    RectangleGate,
    create_gate_from_dict,
)
from flowx.gating.runner import (
    CleanupStepResult,
    resolve_cleanup_channels_from_metadata,
    run_cleanup_step,
    run_cohort_cleanup,
)

__all__ = [
    "Gate",
    "PolygonGate",
    "RectangleGate",
    "RangeGate",
    "QuadrantGate",
    "GatingHierarchy",
    "create_gate_from_dict",
    "DensityCutoffResult",
    "QuadrantCutoffResult",
    "find_density_minimum",
    "find_quadrant_cutoffs",
    "build_adaptive_density_hierarchy",
    "FMOCalibrationConfig",
    "FMOCalibrationRecord",
    "FMOCalibrationResult",
    "calibrate_fmo_thresholds",
    "build_adaptive_cell_gate",
    "build_adaptive_singlet_gate",
    "CleanUpGateConfig",
    "CleanUpGatingResult",
    "detect_viability_channel",
    "detect_dump_channel",
    "resolve_scatter_channels",
    "build_debris_gate",
    "build_singlet_gate",
    "build_viability_gate",
    "build_dump_gate",
    "run_cleanup_gating",
    "CleanupStepResult",
    "resolve_cleanup_channels_from_metadata",
    "run_cleanup_step",
    "run_cohort_cleanup",
]

