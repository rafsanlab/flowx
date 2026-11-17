"""High-Throughput Inter-Batch Normalization for Flow and Mass Cytometry.

Implements Native Permissive CytoNorm (FlowSOM-clustered quantile spline warping)
and Anchor-Free Quantile Distribution Alignment with MIFlowCyt-compliant audit trails.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.cluster import MiniBatchKMeans

from flowx.batch.models import BatchCorrectionConfig, CohortSampleInfo
from flowx.io.logger import get_logger


@dataclass
class QuantileSplineMapping:
    """Knots and monotonic mapping function for a single (cluster, marker, batch) tuple.

    Attributes:
        cluster_id: Metacluster index (-1 for global/anchor-free).
        marker: Name of normalized fluorescent or mass detector.
        batch_id: Batch identifier.
        input_quantiles: Source batch quantile knots (u coordinates).
        goal_quantiles: Target goal quantile knots (v coordinates).
        method: Interpolation mode ('pchip' or 'linear'). Default 'pchip'.
    """

    cluster_id: int
    marker: str
    batch_id: str
    input_quantiles: List[float]
    goal_quantiles: List[float]
    method: str = "pchip"

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Evaluate monotonic quantile spline mapping on an array of marker values.

        Preserves monotonicity and applies boundary linear extrapolation to prevent
        polynomial runaway outside empirical quantile boundaries.

        Args:
            x: 1D array of input marker intensity values.

        Returns:
            1D array of batch-normalized intensity values.
        """
        u = np.asarray(self.input_quantiles, dtype=float)
        v = np.asarray(self.goal_quantiles, dtype=float)

        if len(u) < 2 or len(v) < 2:
            return x.copy()

        # Deduplicate u coordinates to ensure strict monotonicity
        diffs = np.diff(u)
        keep = np.concatenate(([True], diffs > 1e-7))
        u_clean = u[keep]
        v_clean = v[keep]

        if len(u_clean) < 2:
            return x.copy()

        # Piecewise linear interpolation with linear boundary extrapolation
        y = np.interp(x, u_clean, v_clean)

        # Left boundary linear extrapolation
        mask_left = x < u_clean[0]
        if np.any(mask_left):
            dx_left = u_clean[1] - u_clean[0]
            if dx_left > 1e-7:
                slope_left = (v_clean[1] - v_clean[0]) / dx_left
                y[mask_left] = v_clean[0] + slope_left * (x[mask_left] - u_clean[0])

        # Right boundary linear extrapolation
        mask_right = x > u_clean[-1]
        if np.any(mask_right):
            dx_right = u_clean[-1] - u_clean[-2]
            if dx_right > 1e-7:
                slope_right = (v_clean[-1] - v_clean[-2]) / dx_right
                y[mask_right] = v_clean[-1] + slope_right * (x[mask_right] - u_clean[-1])

        return y

    def to_dict(self) -> Dict[str, Any]:
        """Serialize spline mapping to JSON-compatible dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "marker": self.marker,
            "batch_id": self.batch_id,
            "input_quantiles": [float(val) for val in self.input_quantiles],
            "goal_quantiles": [float(val) for val in self.goal_quantiles],
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuantileSplineMapping:
        """Construct QuantileSplineMapping from dictionary."""
        return cls(
            cluster_id=data["cluster_id"],
            marker=data["marker"],
            batch_id=data["batch_id"],
            input_quantiles=data["input_quantiles"],
            goal_quantiles=data["goal_quantiles"],
            method=data.get("method", "pchip"),
        )


@dataclass
class CytoNormModel:
    """Trained Native CytoNorm normalization model holding metacluster centroids and splines.

    Attributes:
        markers: List of normalized marker channel names.
        batches: List of known batch IDs.
        n_clusters: Number of metaclusters.
        centroids: Centroids matrix of shape (n_clusters, len(markers)).
        splines: Dictionary mapping (cluster_id, marker, batch_id) to QuantileSplineMapping.
        goal: Target objective ('median', 'mean', or 'reference').
        reference_batch: Reference batch ID if goal='reference'.
        created_at: ISO timestamp of model training.
    """

    markers: List[str]
    batches: List[str]
    n_clusters: int
    centroids: np.ndarray
    splines: Dict[Tuple[int, str, str], QuantileSplineMapping]
    goal: str = "median"
    reference_batch: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def transform(self, df: pd.DataFrame, batch_id: str) -> pd.DataFrame:
        """Apply fitted CytoNorm cluster splines to normalize a sample from a given batch.

        Args:
            df: Input event DataFrame.
            batch_id: Batch identifier of the sample.

        Returns:
            Normalized DataFrame with identical shape and non-normalized channels intact.
        """
        logger = get_logger()
        if len(df) == 0:
            return df.copy()

        if batch_id not in self.batches:
            logger.warning(
                f"Sample batch '{batch_id}' not in trained CytoNorm batches: {self.batches}. "
                f"Applying nearest batch '{self.batches[0]}' splines as best approximation."
            )
            active_batch = self.batches[0]
        else:
            active_batch = batch_id

        # Verify required trained markers are present
        missing = [m for m in self.markers if m not in df.columns]
        if missing:
            raise KeyError(
                f"Cannot apply CytoNorm: sample DataFrame is missing {len(missing)} trained markers: {missing}"
            )

        norm_df = df.copy()
        events_x = df[self.markers].to_numpy()

        # 1. Assign each cell to nearest metacluster centroid
        dist_matrix = cdist(events_x, self.centroids, metric="euclidean")
        cluster_assignments = np.argmin(dist_matrix, axis=1)

        # 2. Apply cluster- and marker-specific splines
        for k in range(self.n_clusters):
            k_mask = (cluster_assignments == k)
            if not np.any(k_mask):
                continue

            for m in self.markers:
                mapping = self.splines.get((k, m, active_batch))
                if mapping is not None:
                    raw_vals = norm_df.loc[k_mask, m].to_numpy()
                    norm_df.loc[k_mask, m] = mapping.apply(raw_vals)

        return norm_df

    def save_json(self, filepath: Union[str, Path]) -> Path:
        """Save model centroids, parameters, and spline knots to human-readable JSON."""
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)

        splines_serialized = [s.to_dict() for s in self.splines.values()]
        payload = {
            "algorithm": "CytoNorm",
            "version": "1.0",
            "created_at": self.created_at,
            "goal": self.goal,
            "reference_batch": self.reference_batch,
            "markers": self.markers,
            "batches": self.batches,
            "n_clusters": self.n_clusters,
            "centroids": self.centroids.tolist(),
            "splines": splines_serialized,
        }

        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        get_logger().info(f"Saved CytoNorm model to '{dest.name}' ({len(splines_serialized)} splines).")
        return dest

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> CytoNormModel:
        """Load trained CytoNorm model from JSON file."""
        src = Path(filepath)
        data = json.loads(src.read_text(encoding="utf-8"))

        centroids = np.asarray(data["centroids"], dtype=float)
        splines_dict: Dict[Tuple[int, str, str], QuantileSplineMapping] = {}

        for item in data["splines"]:
            mapping = QuantileSplineMapping.from_dict(item)
            splines_dict[(mapping.cluster_id, mapping.marker, mapping.batch_id)] = mapping

        return cls(
            markers=data["markers"],
            batches=data["batches"],
            n_clusters=data["n_clusters"],
            centroids=centroids,
            splines=splines_dict,
            goal=data.get("goal", "median"),
            reference_batch=data.get("reference_batch"),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


def fit_cytonorm(
    cohort_dfs: Dict[str, pd.DataFrame],
    sample_info: Dict[str, CohortSampleInfo],
    config: Optional[BatchCorrectionConfig] = None,
) -> CytoNormModel:
    """Train Native Permissive CytoNorm model using control anchor samples across batches.

    Args:
        cohort_dfs: Dictionary mapping sample_id to event DataFrame.
        sample_info: Dictionary mapping sample_id to CohortSampleInfo.
        config: BatchCorrectionConfig settings.

    Returns:
        Fitted CytoNormModel ready to normalize experimental cohort samples.
    """
    if config is None:
        config = BatchCorrectionConfig()

    logger = get_logger()

    # Identify control samples and unique batches
    control_samples = [s_id for s_id, s_info in sample_info.items() if s_info.is_control]
    if not control_samples:
        raise ValueError(
            "CytoNorm requires at least one control anchor sample per batch (sample_info.is_control=True). "
            "For cohorts without controls, use method='quantile' (Anchor-Free Quantile Normalization)."
        )

    batch_to_controls: Dict[str, List[str]] = {}
    for s_id in control_samples:
        b_id = sample_info[s_id].batch_id
        batch_to_controls.setdefault(b_id, []).append(s_id)

    unique_batches = sorted(batch_to_controls.keys())
    if len(unique_batches) < 2:
        logger.warning(
            f"CytoNorm called with only {len(unique_batches)} batch ({unique_batches}). "
            "Batch correction is typically applied across 2 or more batches."
        )

    # 1. Resolve markers to normalize
    first_df = cohort_dfs[control_samples[0]]
    if config.markers:
        target_markers = [m for m in config.markers if m in first_df.columns]
    else:
        exclude_set = {ex.lower() for ex in config.exclude_markers}
        target_markers = [
            c for c in first_df.columns
            if c.lower() not in exclude_set
            and not any(c.lower().startswith(ex) for ex in ["time", "event", "orig", "sample", "batch", "fsc", "ssc"])
        ]

    if not target_markers:
        raise ValueError("No valid fluorescent or mass markers identified for CytoNorm normalization.")

    logger.info(
        f"Training Native CytoNorm across {len(unique_batches)} batches on {len(target_markers)} markers "
        f"using {len(control_samples)} control anchor samples."
    )

    # 2. Pool control events and train Metaclusters
    ctrl_dfs_list: List[pd.DataFrame] = []
    for b_id, c_ids in batch_to_controls.items():
        b_dfs = [cohort_dfs[c_id][target_markers] for c_id in c_ids if c_id in cohort_dfs]
        if b_dfs:
            ctrl_dfs_list.append(pd.concat(b_dfs, ignore_index=True))

    pooled_controls = pd.concat(ctrl_dfs_list, ignore_index=True)

    n_clusters = min(config.n_clusters, max(2, len(pooled_controls) // 50))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(512, len(pooled_controls)),
        n_init=3,
    ).fit(pooled_controls[target_markers].to_numpy())
    centroids = kmeans.cluster_centers_

    # Assign cluster labels to each batch control
    batch_control_events: Dict[str, pd.DataFrame] = {}
    batch_cluster_labels: Dict[str, np.ndarray] = {}

    for b_id, c_ids in batch_to_controls.items():
        b_dfs = [cohort_dfs[c_id] for c_id in c_ids if c_id in cohort_dfs]
        merged_b = pd.concat(b_dfs, ignore_index=True)
        batch_control_events[b_id] = merged_b
        preds = kmeans.predict(merged_b[target_markers].to_numpy())
        batch_cluster_labels[b_id] = preds

    # 3. Compute Quantiles and Monotonic Splines
    quantiles_grid = np.linspace(0.0, 1.0, config.n_quantiles)
    spline_mappings: Dict[Tuple[int, str, str], QuantileSplineMapping] = {}

    for k in range(n_clusters):
        for m in target_markers:
            # Extract quantiles per batch
            batch_q: Dict[str, np.ndarray] = {}
            for b_id in unique_batches:
                b_events = batch_control_events[b_id]
                k_mask = (batch_cluster_labels[b_id] == k)
                vals = b_events.loc[k_mask, m].to_numpy()

                if len(vals) >= config.min_cluster_events:
                    batch_q[b_id] = np.percentile(vals, quantiles_grid * 100.0)
                else:
                    # Cluster event count too low in this batch: fallback to batch-level quantiles
                    batch_q[b_id] = np.percentile(b_events[m].to_numpy(), quantiles_grid * 100.0)

            # Determine Goal Quantile
            if config.goal == "reference" and config.reference_batch and config.reference_batch in batch_q:
                goal_q = batch_q[config.reference_batch]
            elif config.goal == "mean":
                goal_q = np.mean(list(batch_q.values()), axis=0)
            else:  # 'median'
                goal_q = np.median(list(batch_q.values()), axis=0)

            # Fit Spline Mapping per batch
            for b_id in unique_batches:
                mapping = QuantileSplineMapping(
                    cluster_id=k,
                    marker=m,
                    batch_id=b_id,
                    input_quantiles=batch_q[b_id].tolist(),
                    goal_quantiles=goal_q.tolist(),
                    method=config.spline_method,
                )
                spline_mappings[(k, m, b_id)] = mapping

    logger.info(
        f"CytoNorm training complete: {len(spline_mappings):,} splines fitted "
        f"across {n_clusters} metaclusters and {len(unique_batches)} batches."
    )

    return CytoNormModel(
        markers=target_markers,
        batches=unique_batches,
        n_clusters=n_clusters,
        centroids=centroids,
        splines=spline_mappings,
        goal=config.goal,
        reference_batch=config.reference_batch,
    )


@dataclass
class QuantileNormalizer:
    """Anchor-Free Quantile Distribution Alignment for multi-batch cohorts without controls.

    Attributes:
        markers: List of normalized marker channel names.
        batches: List of known batch IDs.
        mappings: Dictionary mapping (marker, batch_id) to QuantileSplineMapping.
        goal: Target objective ('median', 'mean', or 'reference').
    """

    markers: List[str]
    batches: List[str]
    mappings: Dict[Tuple[str, str], QuantileSplineMapping]
    goal: str = "median"

    def transform(self, df: pd.DataFrame, batch_id: str) -> pd.DataFrame:
        """Apply anchor-free quantile alignment to a sample from a given batch."""
        if len(df) == 0:
            return df.copy()

        norm_df = df.copy()
        active_batch = batch_id if batch_id in self.batches else (self.batches[0] if self.batches else batch_id)

        for m in self.markers:
            if m not in df.columns:
                continue
            mapping = self.mappings.get((m, active_batch))
            if mapping is not None:
                norm_df[m] = mapping.apply(norm_df[m].to_numpy())

        return norm_df


def fit_anchor_free_quantile_normalizer(
    cohort_dfs: Dict[str, pd.DataFrame],
    sample_info: Dict[str, CohortSampleInfo],
    config: Optional[BatchCorrectionConfig] = None,
) -> QuantileNormalizer:
    """Train Anchor-Free Quantile Normalizer by aligning overall marker distributions across batches.

    Args:
        cohort_dfs: Dictionary mapping sample_id to event DataFrame.
        sample_info: Dictionary mapping sample_id to CohortSampleInfo.
        config: BatchCorrectionConfig settings.

    Returns:
        Fitted QuantileNormalizer instance.
    """
    if config is None:
        config = BatchCorrectionConfig(method="quantile")

    logger = get_logger()

    batch_to_samples: Dict[str, List[str]] = {}
    for s_id, s_info in sample_info.items():
        batch_to_samples.setdefault(s_info.batch_id, []).append(s_id)

    unique_batches = sorted(batch_to_samples.keys())

    # Resolve markers
    first_df = next(iter(cohort_dfs.values()))
    if config.markers:
        target_markers = [m for m in config.markers if m in first_df.columns]
    else:
        exclude_set = {ex.lower() for ex in config.exclude_markers}
        target_markers = [
            c for c in first_df.columns
            if c.lower() not in exclude_set
            and not any(c.lower().startswith(ex) for ex in ["time", "event", "orig", "sample", "batch", "fsc", "ssc"])
        ]

    logger.info(
        f"Training Anchor-Free Quantile Normalization across {len(unique_batches)} batches on {len(target_markers)} markers."
    )

    quantiles_grid = np.linspace(0.0, 1.0, config.n_quantiles)
    mappings: Dict[Tuple[str, str], QuantileSplineMapping] = {}

    for m in target_markers:
        batch_q: Dict[str, np.ndarray] = {}
        for b_id, s_ids in batch_to_samples.items():
            b_dfs = [cohort_dfs[s_id][m] for s_id in s_ids if s_id in cohort_dfs and m in cohort_dfs[s_id].columns]
            if b_dfs:
                vals = pd.concat(b_dfs, ignore_index=True).to_numpy()
                batch_q[b_id] = np.percentile(vals, quantiles_grid * 100.0)
            else:
                batch_q[b_id] = quantiles_grid

        # Goal distribution
        if config.goal == "reference" and config.reference_batch and config.reference_batch in batch_q:
            goal_q = batch_q[config.reference_batch]
        elif config.goal == "mean":
            goal_q = np.mean(list(batch_q.values()), axis=0)
        else:  # 'median'
            goal_q = np.median(list(batch_q.values()), axis=0)

        for b_id in unique_batches:
            mappings[(m, b_id)] = QuantileSplineMapping(
                cluster_id=-1,
                marker=m,
                batch_id=b_id,
                input_quantiles=batch_q[b_id].tolist(),
                goal_quantiles=goal_q.tolist(),
                method=config.spline_method,
            )

    return QuantileNormalizer(
        markers=target_markers,
        batches=unique_batches,
        mappings=mappings,
        goal=config.goal,
    )


def calculate_batch_invariance_metrics(
    raw_cohort: Dict[str, pd.DataFrame],
    normalized_cohort: Dict[str, pd.DataFrame],
    sample_info: Dict[str, CohortSampleInfo],
    markers: Optional[List[str]] = None,
    controls_only: bool = True,
) -> pd.DataFrame:
    """Calculate Earth Mover's Distance (Wasserstein) and Kolmogorov-Smirnov statistics before and after correction.

    Args:
        raw_cohort: Dictionary mapping sample_id to raw event DataFrame.
        normalized_cohort: Dictionary mapping sample_id to normalized event DataFrame.
        sample_info: Dictionary mapping sample_id to CohortSampleInfo.
        markers: Optional list of markers to evaluate (auto-detected if None).
        controls_only: Whether to compute metrics across technical controls (recommended for CytoNorm).

    Returns:
        DataFrame containing pairwise batch distance metrics and percentage reductions.
    """
    # Group samples by batch
    batch_samples: Dict[str, List[str]] = {}
    for s_id, s_info in sample_info.items():
        if controls_only and not s_info.is_control:
            continue
        batch_samples.setdefault(s_info.batch_id, []).append(s_id)

    unique_batches = sorted(batch_samples.keys())
    if len(unique_batches) < 2:
        return pd.DataFrame()

    if markers is None:
        first_df = next(iter(raw_cohort.values()))
        markers = [
            c for c in first_df.columns
            if not any(c.lower().startswith(ex) for ex in ["time", "event", "orig", "sample", "batch", "fsc", "ssc"])
        ]

    rows: List[Dict[str, Any]] = []

    # Iterate over all pairwise batch combinations
    for i in range(len(unique_batches)):
        for j in range(i + 1, len(unique_batches)):
            b1 = unique_batches[i]
            b2 = unique_batches[j]

            s1_list = batch_samples[b1]
            s2_list = batch_samples[b2]

            raw_b1_dfs = [raw_cohort[s] for s in s1_list if s in raw_cohort]
            raw_b2_dfs = [raw_cohort[s] for s in s2_list if s in raw_cohort]
            norm_b1_dfs = [normalized_cohort[s] for s in s1_list if s in normalized_cohort]
            norm_b2_dfs = [normalized_cohort[s] for s in s2_list if s in normalized_cohort]

            if not (raw_b1_dfs and raw_b2_dfs and norm_b1_dfs and norm_b2_dfs):
                continue

            raw_b1 = pd.concat(raw_b1_dfs, ignore_index=True)
            raw_b2 = pd.concat(raw_b2_dfs, ignore_index=True)
            norm_b1 = pd.concat(norm_b1_dfs, ignore_index=True)
            norm_b2 = pd.concat(norm_b2_dfs, ignore_index=True)

            for m in markers:
                if not (m in raw_b1.columns and m in raw_b2.columns and m in norm_b1.columns and m in norm_b2.columns):
                    continue

                v_raw_1 = raw_b1[m].to_numpy()
                v_raw_2 = raw_b2[m].to_numpy()
                v_norm_1 = norm_b1[m].to_numpy()
                v_norm_2 = norm_b2[m].to_numpy()

                raw_emd = float(wasserstein_distance(v_raw_1, v_raw_2))
                norm_emd = float(wasserstein_distance(v_norm_1, v_norm_2))
                emd_reduction = ((raw_emd - norm_emd) / raw_emd * 100.0) if raw_emd > 1e-7 else 0.0

                ks_raw = float(ks_2samp(v_raw_1, v_raw_2, method="asymp").statistic)
                ks_norm = float(ks_2samp(v_norm_1, v_norm_2, method="asymp").statistic)
                ks_reduction = ((ks_raw - ks_norm) / ks_raw * 100.0) if ks_raw > 1e-7 else 0.0

                rows.append({
                    "batch_pair": f"{b1} vs {b2}",
                    "marker": m,
                    "raw_emd": round(raw_emd, 4),
                    "normalized_emd": round(norm_emd, 4),
                    "emd_reduction_pct": round(emd_reduction, 2),
                    "raw_ks_stat": round(ks_raw, 4),
                    "normalized_ks_stat": round(ks_norm, 4),
                    "ks_reduction_pct": round(ks_reduction, 2),
                })

    return pd.DataFrame(rows)

