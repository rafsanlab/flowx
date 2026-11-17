"""Mathematical operations for flow cytometry data processing."""

from flowx.math.compensation import (
    CompensationControlResult,
    CompensationMatrix,
    CompensationStepResult,
    calculate_spillover_from_controls,
    compensate_events,
    gate_control_populations,
    load_control_manifest,
    match_control_files,
    resolve_compensation_controls,
    run_compensation_step,
)
from flowx.math.transforms import (
    AsinhTransform,
    BaseTransform,
    BiexponentialTransform,
    LogicleTransform,
    LogTransform,
    TransformConfig,
    TransformStepResult,
    apply_transform_to_dataframe,
    create_transform,
    get_optimal_transform,
    resolve_fluorescence_channels,
    run_transformation_step,
)

__all__ = [
    "CompensationMatrix",
    "compensate_events",
    "CompensationControlResult",
    "CompensationStepResult",
    "calculate_spillover_from_controls",
    "resolve_compensation_controls",
    "run_compensation_step",
    "match_control_files",
    "load_control_manifest",
    "gate_control_populations",
    "BaseTransform",
    "LogicleTransform",
    "BiexponentialTransform",
    "AsinhTransform",
    "LogTransform",
    "TransformConfig",
    "TransformStepResult",
    "create_transform",
    "get_optimal_transform",
    "apply_transform_to_dataframe",
    "resolve_fluorescence_channels",
    "run_transformation_step",
]


