"""Data models and container classes for Phase 5 quantitative cohort and statistical analysis.

Provides structured containers for population feature extraction, differential abundance (DA),
and differential state (DS) testing with full audit serialization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class DAConfig:
    """Configuration for Differential Abundance (DA) statistical hypothesis testing.

    Attributes:
        test_method: Statistical testing framework:
            - 'mann_whitney': Non-parametric rank-sum test for independent cohorts (default).
            - 'wilcoxon_paired': Wilcoxon signed-rank test for paired longitudinal samples.
            - 't_test': Welch's two-sample t-test with unequal variances.
            - 'kruskal': Kruskal-Wallis non-parametric ANOVA for multi-group cohorts (>=3 conditions).
        fdr_threshold: Maximum false discovery rate (q-value) for significance (default: 0.05).
        log2fc_threshold: Absolute log2 fold change threshold for significance (default: 1.0).
        pseudocount: Small percentage pseudocount added before log2 transformation (default: 0.01%).
        min_events: Minimum total events in a population across cohorts to qualify for testing (default: 10).
        frequency_type: Frequency denominator ('viable_singlets' or 'parent').
    """

    test_method: Literal[
        "mann_whitney",
        "wilcoxon_paired",
        "t_test",
        "welch_ttest",
        "kruskal",
        "glm_binomial",
    ] = "mann_whitney"
    fdr_threshold: float = 0.05
    log2fc_threshold: float = 1.0
    pseudocount: float = 0.01
    min_events: int = 10
    frequency_type: Literal["viable_singlets", "parent"] = "viable_singlets"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DAConfig:
        """Construct DAConfig from dictionary."""
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class DSConfig:
    """Configuration for Differential State (DS) functional marker testing.

    Attributes:
        test_method: Statistical testing framework ('mann_whitney', 't_test', 'wilcoxon_paired').
        statistic: Summary statistic to evaluate ('mfi' or 'geomfi').
        fdr_threshold: Maximum false discovery rate (q-value) for significance (default: 0.05).
        min_events_per_cluster: Minimum events required in a population per sample to compute MFI (default: 20).
        functional_markers: Explicit subset of functional/activation markers to test (tests all if None).
    """

    test_method: Literal["mann_whitney", "t_test", "wilcoxon_paired"] = "mann_whitney"
    statistic: Literal["mfi", "geomfi"] = "mfi"
    fdr_threshold: float = 0.05
    min_events_per_cluster: int = 20
    functional_markers: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DSConfig:
        """Construct DSConfig from dictionary."""
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class CohortFeatureSet:
    """Consolidated quantitative feature matrices extracted across a cytometry cohort.

    Attributes:
        sample_ids: Unique identifiers of cohort samples.
        population_names: Discovered cluster names or gated population lineages.
        marker_names: Evaluated fluorescent or mass cytometry detector channels.
        count_matrix: DataFrame of integer event counts [Samples x Populations].
        parent_freq_matrix: DataFrame of percentages relative to immediate parent population.
        viable_freq_matrix: DataFrame of percentages relative to total viable singlets.
        mfi_tensor: Dictionary mapping marker name to DataFrame [Samples x Populations] of MFIs.
        geomfi_tensor: Dictionary mapping marker name to DataFrame of Geometric Mean MFIs.
        rsd_tensor: Dictionary mapping marker name to DataFrame of robust standard deviations.
        sample_metadata: Metadata DataFrame with conditions, batches, patient IDs, etc.
    """

    sample_ids: List[str]
    population_names: List[str]
    marker_names: List[str]
    count_matrix: pd.DataFrame
    parent_freq_matrix: pd.DataFrame
    viable_freq_matrix: pd.DataFrame
    mfi_tensor: Dict[str, pd.DataFrame] = field(default_factory=dict)
    geomfi_tensor: Dict[str, pd.DataFrame] = field(default_factory=dict)
    rsd_tensor: Dict[str, pd.DataFrame] = field(default_factory=dict)
    sample_metadata: Optional[pd.DataFrame] = None

    def get_population_summary(self, population: str) -> pd.DataFrame:
        """Extract multi-sample summary row for a single population across all samples."""
        if population not in self.population_names:
            raise KeyError(f"Population '{population}' not found in feature set.")

        rows = []
        for s in self.sample_ids:
            row: Dict[str, Any] = {
                "sample_id": s,
                "count": self.count_matrix.loc[s, population] if s in self.count_matrix.index else 0,
                "freq_parent_pct": self.parent_freq_matrix.loc[s, population] if s in self.parent_freq_matrix.index else 0.0,
                "freq_viable_pct": self.viable_freq_matrix.loc[s, population] if s in self.viable_freq_matrix.index else 0.0,
            }
            if self.sample_metadata is not None and s in self.sample_metadata.index:
                for col in self.sample_metadata.columns:
                    row[col] = self.sample_metadata.loc[s, col]

            for m in self.marker_names:
                if m in self.mfi_tensor and s in self.mfi_tensor[m].index:
                    row[f"{m}_mfi"] = self.mfi_tensor[m].loc[s, population]

            rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class DAResult:
    """Results from Differential Abundance (DA) hypothesis testing between experimental cohorts.

    Attributes:
        comparison_name: Human-readable comparison description (e.g. 'Treated vs Untreated').
        reference_group: Label of baseline/reference condition.
        target_group: Label of experimental/target condition.
        summary_df: Formatted DataFrame of results per population with p-values, q-values, and log2FC.
        significant_populations: List of population names meeting significance criteria (q < threshold and |log2FC| >= threshold).
        config: DAConfig applied.
    """

    comparison_name: str
    reference_group: str
    target_group: str
    summary_df: pd.DataFrame
    significant_populations: List[str] = field(default_factory=list)
    config: DAConfig = field(default_factory=DAConfig)
    n_ref: int = 0
    n_target: int = 0
    min_achievable_p: Optional[float] = None
    underpowered_warning: Optional[str] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return the complete results summary DataFrame."""
        return self.summary_df.copy()


@dataclass
class DSResult:
    """Results from Differential State (DS) marker expression modulation testing within cell populations.

    Attributes:
        comparison_name: Human-readable comparison description.
        reference_group: Baseline condition label.
        target_group: Experimental condition label.
        summary_df: Formatted DataFrame of results per (population, marker) pair with p-values, q-values, and effect sizes.
        significant_states: List of (population, marker) tuples meeting significance criteria.
        config: DSConfig applied.
        n_ref: Number of reference group biological samples evaluated.
        n_target: Number of target group biological samples evaluated.
        min_achievable_p: Minimum two-sided permutation p-value achievable for non-parametric rank tests.
        underpowered_warning: Explicit diagnostic warning when sample sizes are underpowered for inferential testing.
    """

    comparison_name: str
    reference_group: str
    target_group: str
    summary_df: pd.DataFrame
    significant_states: List[Tuple[str, str]] = field(default_factory=list)
    config: DSConfig = field(default_factory=DSConfig)
    n_ref: int = 0
    n_target: int = 0
    min_achievable_p: Optional[float] = None
    underpowered_warning: Optional[str] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return the complete results summary DataFrame."""
        return self.summary_df.copy()
