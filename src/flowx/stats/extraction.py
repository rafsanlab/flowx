"""Cohort Feature Extraction Engine for Flow Cytometry.

Transforms clustered or gated single-cell event matrices into structured,
sample-by-population quantitative feature tables (counts, parent frequencies,
viable singlet percentages, MFI, GeoMFI, and robust SD).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from flowx.io.logger import get_logger
from flowx.stats.metrics import (
    geometric_mean,
    median_fluorescence_intensity,
    robust_sd,
)
from flowx.stats.models import CohortFeatureSet


class CohortFeatureExtractor:
    """Extracts population frequencies and per-marker intensity statistics across cohort samples."""

    def __init__(
        self,
        population_column: str = "cluster_id",
        markers: Optional[List[str]] = None,
        exclude_markers: Optional[List[str]] = None,
        min_cluster_events: int = 5,
    ) -> None:
        """Initialize the feature extraction engine.

        Args:
            population_column: DataFrame column holding population or cluster labels (e.g. 'cluster_id', 'lineage_id').
            markers: Explicit list of markers to extract stats for (auto-detected if None).
            exclude_markers: Markers to exclude (e.g. morphology or index channels).
            min_cluster_events: Minimum events required to compute valid MFI/GeoMFI (otherwise NaN).
        """
        self.population_column = population_column
        self.markers = markers
        self.exclude_markers = exclude_markers or [
            "time", "event_length", "original_event_index", "sample_id", "batch_id", "cluster_id", "lineage_id",
            "fsc-a", "fsc-h", "fsc-w", "ssc-a", "ssc-h", "ssc-w", "umap_1", "umap_2"
        ]
        self.min_cluster_events = min_cluster_events
        self.logger = get_logger()

    def extract(
        self,
        cohort_dfs: Dict[str, pd.DataFrame],
        sample_metadata: Optional[Union[pd.DataFrame, Dict[str, Dict[str, Any]]]] = None,
        parent_counts: Optional[Dict[str, int]] = None,
        viable_counts: Optional[Dict[str, int]] = None,
    ) -> CohortFeatureSet:
        """Extract multi-sample feature set from a dictionary of event DataFrames.

        Args:
            cohort_dfs: Dictionary mapping sample_id to event DataFrame.
            sample_metadata: Optional DataFrame or dict containing sample conditions/metadata.
            parent_counts: Optional dictionary mapping sample_id to immediate parent event count.
            viable_counts: Optional dictionary mapping sample_id to total clean viable singlets count.

        Returns:
            Populated CohortFeatureSet containing aligned counts, frequencies, and expression tensors.
        """
        sample_ids = sorted(cohort_dfs.keys())
        if not sample_ids:
            raise ValueError("No samples provided for cohort feature extraction.")

        # 1. Identify all unique populations present across the cohort
        all_populations: set = set()
        for s_id, df in cohort_dfs.items():
            if self.population_column in df.columns:
                all_populations.update(df[self.population_column].dropna().unique())

        if not all_populations:
            raise ValueError(
                f"Population column '{self.population_column}' not found in any provided sample DataFrames."
            )

        # Sort populations numerically if possible, otherwise alphabetically
        try:
            sorted_pops = sorted(all_populations, key=lambda x: int(x))
        except (ValueError, TypeError):
            sorted_pops = sorted(all_populations, key=lambda x: str(x))

        population_names = [f"Cluster_{p}" if isinstance(p, (int, np.integer)) else str(p) for p in sorted_pops]
        pop_key_map = dict(zip(population_names, sorted_pops))

        # 2. Resolve marker channels to evaluate
        first_df = cohort_dfs[sample_ids[0]]
        if self.markers is not None:
            active_markers = [m for m in self.markers if m in first_df.columns]
        else:
            active_markers = [
                c for c in first_df.columns
                if c not in self.exclude_markers and not any(c.lower().startswith(ex) for ex in ["time", "event", "orig", "sample", "batch"])
            ]

        # 3. Build Count, Parent Frequency, and Viable Frequency Matrices
        count_data: Dict[str, List[int]] = {p: [] for p in population_names}
        parent_freq_data: Dict[str, List[float]] = {p: [] for p in population_names}
        viable_freq_data: Dict[str, List[float]] = {p: [] for p in population_names}

        # Initialize expression tensors: {marker -> {pop -> [values per sample]}}
        mfi_data: Dict[str, Dict[str, List[float]]] = {m: {p: [] for p in population_names} for m in active_markers}
        geomfi_data: Dict[str, Dict[str, List[float]]] = {m: {p: [] for p in population_names} for m in active_markers}
        rsd_data: Dict[str, Dict[str, List[float]]] = {m: {p: [] for p in population_names} for m in active_markers}

        for s_id in sample_ids:
            df = cohort_dfs[s_id]
            n_events_in_df = len(df)
            n_parent = parent_counts.get(s_id, n_events_in_df) if parent_counts else n_events_in_df
            n_viable = viable_counts.get(s_id, n_events_in_df) if viable_counts else n_events_in_df

            pop_series = df[self.population_column] if self.population_column in df.columns else pd.Series([], dtype=object)

            for p_name in population_names:
                orig_val = pop_key_map[p_name]
                mask = (pop_series == orig_val)
                cnt = int(mask.sum())
                count_data[p_name].append(cnt)

                p_freq = (cnt / n_parent * 100.0) if n_parent > 0 else 0.0
                v_freq = (cnt / n_viable * 100.0) if n_viable > 0 else 0.0
                parent_freq_data[p_name].append(round(p_freq, 4))
                viable_freq_data[p_name].append(round(v_freq, 4))

                sub_df = df[mask]
                for m in active_markers:
                    if cnt >= self.min_cluster_events and m in sub_df.columns:
                        vals = sub_df[m].to_numpy()
                        m_val = median_fluorescence_intensity(vals)
                        geo_val = geometric_mean(vals)
                        rsd_val = robust_sd(vals, method="mad")
                    else:
                        m_val = np.nan
                        geo_val = np.nan
                        rsd_val = np.nan

                    mfi_data[m][p_name].append(m_val)
                    geomfi_data[m][p_name].append(geo_val)
                    rsd_data[m][p_name].append(rsd_val)

        # Convert to DataFrames indexed by sample_ids
        count_df = pd.DataFrame(count_data, index=sample_ids)
        parent_freq_df = pd.DataFrame(parent_freq_data, index=sample_ids)
        viable_freq_df = pd.DataFrame(viable_freq_data, index=sample_ids)

        mfi_tensor = {m: pd.DataFrame(mfi_data[m], index=sample_ids) for m in active_markers}
        geomfi_tensor = {m: pd.DataFrame(geomfi_data[m], index=sample_ids) for m in active_markers}
        rsd_tensor = {m: pd.DataFrame(rsd_data[m], index=sample_ids) for m in active_markers}

        # Normalize sample metadata
        meta_df: Optional[pd.DataFrame] = None
        if sample_metadata is not None:
            if isinstance(sample_metadata, dict):
                meta_df = pd.DataFrame.from_dict(sample_metadata, orient="index")
            elif isinstance(sample_metadata, pd.DataFrame):
                meta_df = sample_metadata.copy()
            if meta_df is not None and not meta_df.index.equals(pd.Index(sample_ids)):
                # Ensure index matches sample_ids
                if "sample_id" in meta_df.columns:
                    meta_df = meta_df.set_index("sample_id")
                meta_df = meta_df.reindex(sample_ids)

        return CohortFeatureSet(
            sample_ids=sample_ids,
            population_names=population_names,
            marker_names=active_markers,
            count_matrix=count_df,
            parent_freq_matrix=parent_freq_df,
            viable_freq_matrix=viable_freq_df,
            mfi_tensor=mfi_tensor,
            geomfi_tensor=geomfi_tensor,
            rsd_tensor=rsd_tensor,
            sample_metadata=meta_df,
        )
