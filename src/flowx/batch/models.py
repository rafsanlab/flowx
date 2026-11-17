"""Data models and configuration dataclasses for FlowX Step 3: Batch Correction & Preparation.

Provides validated configurations and containers for batch normalization (CytoNorm,
quantile alignment) and deterministic cohort downsampling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class DownsampleConfig:
    """Configuration for deterministic downsampling and sample event balancing.

    Attributes:
        enabled: Whether downsampling is active. Default False.
        mode: Subsampling strategy ('target_count', 'min_sample', 'stratified', 'proportional', 'none').
        target_events: Number of events per sample when mode='target_count'. Default 100,000.
        min_event_policy: Policy when sample has fewer events than target:
            - 'cap': Keep all available events with a notice (default).
            - 'exclude': Exclude sample from downstream cohort analyses.
            - 'error': Raise ValueError if sample has insufficient events.
        min_events_threshold: Minimum viable event count required for a sample to participate in
            'min_sample' bottleneck calculation (prevents failed/empty tubes from collapsing the cohort).
            Default 2500.
        fraction: Proportion of events to retain when mode='proportional' (0.0 < fraction <= 1.0).
        stratified_column: Column name holding population labels when mode='stratified'.
        stratified_quota: Target count per subpopulation when mode='stratified'.
        random_seed: Deterministic PRNG seed for reproducible event selection. Default 42.
        preserve_indices: Whether to record 'original_event_index' in the resulting DataFrame. Default True.
        export_fcs: Whether to export downsampled events to individual .fcs files. Default False.
        export_uncompensated: If exporting FCS, whether to export raw uncompensated values. Default False.
    """

    enabled: bool = False
    mode: Literal["min_sample", "target_count", "stratified", "proportional", "none"] = "min_sample"
    target_events: Optional[int] = None
    min_event_policy: Literal["cap", "exclude", "error"] = "cap"
    min_events_threshold: int = 2500
    fraction: float = 0.2
    stratified_column: Optional[str] = None
    stratified_quota: Optional[int] = None
    random_seed: int = 42
    preserve_indices: bool = True
    export_fcs: bool = False
    export_uncompensated: bool = False

    def __post_init__(self) -> None:
        """Validate downsampling parameters."""
        if self.target_events is not None and self.target_events <= 0:
            raise ValueError(f"target_events must be positive, got {self.target_events}")

        if self.target_events is None and self.mode == "target_count":
            self.mode = "min_sample"

        valid_modes = {"target_count", "min_sample", "stratified", "proportional", "none"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid downsample mode '{self.mode}'. Expected one of: {valid_modes}")

        valid_policies = {"cap", "exclude", "error"}
        if self.min_event_policy not in valid_policies:
            raise ValueError(
                f"Invalid min_event_policy '{self.min_event_policy}'. Expected one of: {valid_policies}"
            )

        if self.target_events is not None and self.target_events <= 0:
            raise ValueError(f"target_events must be positive, got {self.target_events}")

        if not (0.0 < self.fraction <= 1.0):
            raise ValueError(f"fraction must be in range (0.0, 1.0], got {self.fraction}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DownsampleConfig:
        """Construct configuration from dictionary."""
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class BatchCorrectionConfig:
    """Configuration for multi-batch technical variation correction.

    Attributes:
        enabled: Whether batch correction is active. Default False.
        method: Normalization methodology:
            - 'cytonorm': FlowSOM-clustered quantile spline warping on control anchors.
            - 'quantile': Anchor-free global or cluster quantile distribution matching.
            - 'combat': Parametric empirical Bayes location-and-scale adjustment.
            - 'none': Bypass batch correction.
        batch_key: Metadata column or dictionary key indicating batch identifier. Default 'batch_id'.
        control_key: Metadata column indicating whether sample is a technical control anchor. Default 'is_control'.
        goal: Target distribution objective:
            - 'median': Median quantile across all control batches (default, highly robust).
            - 'mean': Mean quantile across control batches.
            - 'reference': Match a specific batch specified in reference_batch.
        reference_batch: Identifier of the reference batch if goal='reference'.
        n_clusters: Number of FlowSOM metaclusters for cell subset partitioning. Default 10.
        n_quantiles: Number of quantile bins for spline mapping (default 101, percentiles 0 to 100).
        markers: Fluorescent or mass markers to normalize (auto-detected if None).
        exclude_markers: Markers to exclude from normalization (e.g. morphology or viability dyes).
        min_cluster_events: Minimum events required in a cluster to fit independent splines. Default 20.
        spline_method: Interpolation mode ('pchip' for monotonic cubic Hermite, 'linear' for piecewise). Default 'pchip'.
        save_model: Whether to persist fitted normalizer model to JSON. Default True.
    """

    enabled: bool = False
    method: Literal["cytonorm", "quantile", "combat", "none"] = "cytonorm"
    batch_key: str = "batch_id"
    control_key: str = "is_control"
    goal: Literal["median", "mean", "reference"] = "median"
    reference_batch: Optional[str] = None
    n_clusters: int = 10
    n_quantiles: int = 101
    markers: Optional[List[str]] = None
    exclude_markers: List[str] = field(default_factory=lambda: ["time", "event_length", "fsc-a", "fsc-h", "ssc-a", "ssc-h"])
    min_cluster_events: int = 20
    spline_method: Literal["pchip", "linear"] = "pchip"
    save_model: bool = True

    def __post_init__(self) -> None:
        """Validate batch correction parameters."""
        valid_methods = {"cytonorm", "quantile", "combat", "none"}
        if self.method not in valid_methods:
            raise ValueError(f"Invalid batch correction method '{self.method}'. Expected one of: {valid_methods}")

        valid_goals = {"median", "mean", "reference"}
        if self.goal not in valid_goals:
            raise ValueError(f"Invalid goal '{self.goal}'. Expected one of: {valid_goals}")

        if self.goal == "reference" and not self.reference_batch:
            raise ValueError("reference_batch must be specified when goal='reference'")

        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}")

        if self.n_quantiles < 5:
            raise ValueError(f"n_quantiles must be >= 5, got {self.n_quantiles}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BatchCorrectionConfig:
        """Construct configuration from dictionary."""
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class BatchPreparationConfig:
    """Unified configuration container combining normalization, downsampling, and export options."""

    batch_normalization: BatchCorrectionConfig = field(default_factory=BatchCorrectionConfig)
    downsampling: DownsampleConfig = field(default_factory=DownsampleConfig)
    output_dir: Union[str, Path] = "outputs/batch_prep"
    random_seed: int = 42
    export_pooled_cohort: bool = True
    export_individual_fcs: bool = False
    generate_diagnostics: bool = True
    assess_batch_effect: bool = True
    overwrite_autocheck: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert full configuration to dictionary."""
        return {
            "batch_normalization": self.batch_normalization.to_dict(),
            "downsampling": self.downsampling.to_dict(),
            "output_dir": str(self.output_dir),
            "random_seed": self.random_seed,
            "export_pooled_cohort": self.export_pooled_cohort,
            "export_individual_fcs": self.export_individual_fcs,
            "generate_diagnostics": self.generate_diagnostics,
            "assess_batch_effect": self.assess_batch_effect,
            "overwrite_autocheck": self.overwrite_autocheck,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BatchPreparationConfig:
        """Construct BatchPreparationConfig from dictionary.

        Supports both standalone BatchPreparationConfig dictionaries
        (`batch_normalization`, `downsampling`) and PipelineConfig sections
        (`04_batch_preparation`, `batch_preparation`, `batch`).
        """
        sec = data
        if "04_batch_preparation" in data or "batch_preparation" in data or "batch" in data:
            sec = data.get("04_batch_preparation", data.get("batch_preparation", data.get("batch", {}))) or {}

        raw_overwrite = sec.get("overwrite_autocheck", sec.get("override_autocheck", data.get("overwrite_autocheck", False)))

        if "batch_normalization" in sec or "downsampling" in sec:
            norm_cfg = sec.get("batch_normalization", {})
            down_cfg = sec.get("downsampling", {})
            return cls(
                batch_normalization=BatchCorrectionConfig.from_dict(norm_cfg) if isinstance(norm_cfg, dict) else norm_cfg,
                downsampling=DownsampleConfig.from_dict(down_cfg) if isinstance(down_cfg, dict) else down_cfg,
                output_dir=sec.get("output_dir", data.get("output_dir", "outputs/batch_prep")),
                random_seed=sec.get("random_seed", sec.get("seed", data.get("random_seed", 42))),
                export_pooled_cohort=sec.get("export_pooled_cohort", True),
                export_individual_fcs=sec.get("export_individual_fcs", False),
                generate_diagnostics=sec.get("generate_diagnostics", True),
                assess_batch_effect=sec.get("assess_batch_effect", True),
                overwrite_autocheck=bool(raw_overwrite),
            )

        # Flat PipelineConfig 04_batch_preparation schema
        step_enabled = bool(sec.get("enabled", True))
        raw_norm_method = str(
            sec.get("normalization_method", sec.get("normalization", sec.get("method_normalization", "none")))
        ).lower()
        norm_enabled = step_enabled and raw_norm_method in ("cytonorm", "quantile", "combat", "auto")
        eff_method = "cytonorm" if raw_norm_method == "auto" else (raw_norm_method if raw_norm_method in ("cytonorm", "quantile", "combat", "none") else "none")
        batch_key = str(sec.get("batch_column", sec.get("batch_key", "batch_id")) or "batch_id")

        norm_kwargs: Dict[str, Any] = {
            "enabled": norm_enabled,
            "method": eff_method,
            "batch_key": batch_key,
        }
        for k in ("control_key", "goal", "reference_batch", "n_clusters", "n_quantiles", "markers", "exclude_markers", "min_cluster_events", "spline_method", "save_model"):
            if k in sec and sec[k] is not None:
                norm_kwargs[k] = sec[k]

        down_mode = str(sec.get("mode", sec.get("downsample_mode", "min_sample"))).lower()
        down_enabled = step_enabled and down_mode != "none" and bool(sec.get("downsample_enabled", True))
        down_kwargs: Dict[str, Any] = {
            "enabled": down_enabled,
            "mode": down_mode,
            "target_events": sec.get("target_events", 50000),
            "min_event_policy": sec.get("min_event_policy", "cap"),
            "min_events_threshold": int(sec.get("min_events_threshold", 2500)),
            "random_seed": int(sec.get("random_seed", sec.get("seed", 42))),
        }
        for k in ("fraction", "stratified_column", "stratified_quota", "preserve_indices", "export_fcs", "export_uncompensated"):
            if k in sec and sec[k] is not None:
                down_kwargs[k] = sec[k]

        return cls(
            batch_normalization=BatchCorrectionConfig.from_dict(norm_kwargs),
            downsampling=DownsampleConfig.from_dict(down_kwargs),
            output_dir=sec.get("output_dir", data.get("output_dir", "outputs/batch_prep")),
            random_seed=int(sec.get("random_seed", sec.get("seed", 42))),
            export_pooled_cohort=bool(sec.get("export_pooled_cohort", True)),
            export_individual_fcs=bool(sec.get("export_individual_fcs", False)),
            generate_diagnostics=bool(sec.get("generate_diagnostics", True)),
            assess_batch_effect=bool(sec.get("assess_batch_effect", True)),
            overwrite_autocheck=bool(raw_overwrite),
        )


@dataclass
class CohortSampleInfo:
    """Tracking metadata and event yield progression for an individual sample in a cohort."""

    sample_id: str
    batch_id: str
    is_control: bool = False
    filepath: Optional[str] = None
    raw_event_count: int = 0
    clean_event_count: int = 0
    prepared_event_count: int = 0
    retention_pct_of_raw: float = 0.0
    retention_pct_of_clean: float = 0.0
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sample tracking info to dictionary."""
        return asdict(self)


@dataclass
class CohortBatchResult:
    """Container holding cohort batch preparation results, datasets, and audits."""

    samples: List[CohortSampleInfo]
    prepared_dfs: Dict[str, pd.DataFrame]
    summary_df: pd.DataFrame
    pooled_df: Optional[pd.DataFrame] = None
    normalization_model: Optional[Any] = None
    metrics_df: Optional[pd.DataFrame] = None
    batch_assessment: Optional[Any] = None
    generated_plots: Dict[str, str] = field(default_factory=dict)
    generated_tables: Dict[str, str] = field(default_factory=dict)
    exported_fcs_paths: Dict[str, str] = field(default_factory=dict)
    output_dir: str = ""
    execution_time_seconds: float = 0.0

    @property
    def total_prepared_events(self) -> int:
        """Total number of events retained across the entire prepared cohort."""
        return sum(len(df) for df in self.prepared_dfs.values())

    @property
    def sample_count(self) -> int:
        """Total number of samples in the prepared cohort."""
        return len(self.samples)

