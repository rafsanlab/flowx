"""FlowX Benchmark Datasets Management Package.

Provides a unified programmatic and CLI interface to list, inspect, verify,
and download conventional and spectral flow cytometry benchmark datasets.
"""

from flowx.datasets.downloader import (
    DatasetVerificationResult,
    download_dataset,
    update_catalog_summaries,
    verify_dataset,
)
from flowx.datasets.registry import (
    BENCHMARK_DATASETS,
    DatasetInfo,
    get_dataset,
    list_datasets,
    resolve_datasets,
)

__all__ = [
    "BENCHMARK_DATASETS",
    "DatasetInfo",
    "DatasetVerificationResult",
    "get_dataset",
    "list_datasets",
    "resolve_datasets",
    "verify_dataset",
    "download_dataset",
    "update_catalog_summaries",
]

