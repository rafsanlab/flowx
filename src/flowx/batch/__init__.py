"""FlowX Step 3: Batch Correction & Preparation Package.

Provides high-throughput inter-batch normalization (native CytoNorm, quantile alignment)
and deterministic cohort event downsampling and balancing.
"""

from flowx.batch.assessment import (
    BatchNecessityAssessment,
    MarkerDriftMetric,
    assess_batch_effect_need,
)
from flowx.batch.downsampling import (
    downsample_cohort,
    downsample_dataframe,
    export_downsampled_fcs,
)
from flowx.batch.models import (
    BatchCorrectionConfig,
    BatchPreparationConfig,
    CohortBatchResult,
    CohortSampleInfo,
    DownsampleConfig,
)
from flowx.batch.normalization import (
    CytoNormModel,
    QuantileNormalizer,
    QuantileSplineMapping,
    calculate_batch_invariance_metrics,
    fit_anchor_free_quantile_normalizer,
    fit_cytonorm,
)
from flowx.batch.runner import (
    discover_cohort_samples,
    load_cohort_manifest,
    run_batch_preparation,
)

__all__ = [
    "BatchCorrectionConfig",
    "DownsampleConfig",
    "BatchPreparationConfig",
    "CohortSampleInfo",
    "CohortBatchResult",
    "QuantileSplineMapping",
    "CytoNormModel",
    "QuantileNormalizer",
    "downsample_dataframe",
    "downsample_cohort",
    "export_downsampled_fcs",
    "fit_cytonorm",
    "fit_anchor_free_quantile_normalizer",
    "calculate_batch_invariance_metrics",
    "load_cohort_manifest",
    "discover_cohort_samples",
    "run_batch_preparation",
    "assess_batch_effect_need",
    "BatchNecessityAssessment",
    "MarkerDriftMetric",
]
