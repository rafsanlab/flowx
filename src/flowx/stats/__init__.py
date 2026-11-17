"""FlowX Phase 5: Quantitative Cohort & Statistical Analysis Package.

Provides high-throughput population feature extraction, non-parametric Differential Abundance (DA),
and functional marker Differential State (DS) analysis conforming to MIFlowCyt standards.
"""

from flowx.stats.differential_abundance import run_differential_abundance
from flowx.stats.differential_state import run_differential_state
from flowx.stats.extraction import CohortFeatureExtractor
from flowx.stats.metrics import (
    cohens_d,
    fdr_correction,
    geometric_mean,
    log2_fold_change,
    median_fluorescence_intensity,
    robust_sd,
)
from flowx.stats.models import (
    CohortFeatureSet,
    DAConfig,
    DAResult,
    DSConfig,
    DSResult,
)
from flowx.stats.prism import export_prism_tables, format_prism_grouped_csv

__all__ = [
    "DAConfig",
    "DSConfig",
    "CohortFeatureSet",
    "DAResult",
    "DSResult",
    "CohortFeatureExtractor",
    "run_differential_abundance",
    "run_differential_state",
    "median_fluorescence_intensity",
    "geometric_mean",
    "robust_sd",
    "log2_fold_change",
    "cohens_d",
    "fdr_correction",
    "format_prism_grouped_csv",
    "export_prism_tables",
]
