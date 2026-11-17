"""FlowX Spectral Flow Cytometry module.

Provides full-spectrum optical deconvolution, reference spectral library assembly,
single- and multi-autofluorescence extraction, Cosine Spectral Similarity assessment,
Panel Complexity Index computation, and Weighted Least Squares (WLS) deconvolution.
"""

from __future__ import annotations

from flowx.spectral.autofluorescence import (
    attach_autofluorescence_to_library,
    extract_autofluorescence_spectrum,
    extract_multi_autofluorescence_spectra,
)
from flowx.spectral.library import (
    ReferenceSpectralLibraryBuilder,
    extract_embedded_unmixing_matrix,
    extract_single_stain_spectrum,
    filter_optical_channels,
    is_optical_fluorescence_channel,
)
from flowx.spectral.metrics import (
    compute_panel_complexity_index,
    compute_spectral_similarity_matrix,
    evaluate_spectral_panel,
)
from flowx.spectral.models import (
    ReferenceSpectralLibrary,
    SpectralMetricsResult,
    SpectralReference,
    UnmixingConfig,
    UnmixingResult,
)
from flowx.spectral.runner import (
    SpectralPipelineResult,
    run_spectral_unmixing_pipeline,
)
from flowx.spectral.unmixing import (
    unmix_dataframe,
    unmix_events_ols,
    unmix_events_wls,
)
from flowx.spectral.visualization import (
    plot_deconvolution_goodness_of_fit,
    plot_spectral_ribbons,
    plot_spectral_similarity_heatmap,
    plot_unmixing_residuals_distribution,
)

__all__ = [
    "ReferenceSpectralLibrary",
    "ReferenceSpectralLibraryBuilder",
    "extract_embedded_unmixing_matrix",
    "SpectralMetricsResult",
    "SpectralPipelineResult",
    "SpectralReference",
    "UnmixingConfig",
    "UnmixingResult",
    "attach_autofluorescence_to_library",
    "compute_panel_complexity_index",
    "compute_spectral_similarity_matrix",
    "evaluate_spectral_panel",
    "extract_autofluorescence_spectrum",
    "extract_multi_autofluorescence_spectra",
    "extract_single_stain_spectrum",
    "filter_optical_channels",
    "is_optical_fluorescence_channel",
    "plot_deconvolution_goodness_of_fit",
    "plot_spectral_ribbons",
    "plot_spectral_similarity_heatmap",
    "plot_unmixing_residuals_distribution",
    "run_spectral_unmixing_pipeline",
    "unmix_dataframe",
    "unmix_events_ols",
    "unmix_events_wls",
]

