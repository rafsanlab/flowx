"""Automated clustering-based quality control (QC) module for flow cytometry.

Provides data-driven, unsupervised deconvolution of raw scatter morphology (Intact Cells
vs. Low-Scatter Debris and Saturation Artifacts) and pulse geometry (Singlets vs. Doublets/Aggregates)
using Gaussian Mixture Models (GMM) and geometric convex-hull envelope extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture

from flowx.gating.gate import PolygonGate
from flowx.io.parser import FCSData


@dataclass
class ClusteringQCResult:
    """Container holding results from automated clustering-based quality control.

    Attributes:
        cells_mask: Boolean array indicating events identified as intact cells.
        singlets_mask: Boolean array indicating events identified as single cells.
        cells_gate: PolygonGate dynamically enclosing intact cell events.
        singlets_gate: PolygonGate dynamically enclosing singlet events.
        total_events: Total input events count.
        cell_count: Number of events passing cell clustering QC.
        singlet_count: Number of events passing singlet clustering QC.
        debris_count: Number of events excluded as debris, noise, or saturation.
        doublet_count: Number of intact cells excluded as doublets or aggregates.
        morphology_labels: Array of categorical labels per event ('Intact Cells', 'Debris / Noise', etc.).
        singlet_labels: Array of categorical labels per event ('Singlets', 'Doublets', 'Unclassified').
        summary_stats: DataFrame summarizing event counts and percentages for each QC stage.
    """

    cells_mask: np.ndarray
    singlets_mask: np.ndarray
    cells_gate: PolygonGate
    singlets_gate: PolygonGate
    total_events: int
    cell_count: int
    singlet_count: int
    debris_count: int
    doublet_count: int
    morphology_labels: np.ndarray
    singlet_labels: np.ndarray
    summary_stats: pd.DataFrame


def cluster_morphology_qc(
    fcs_data: FCSData,
    df_events: pd.DataFrame,
    n_components: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, PolygonGate, np.ndarray, Dict[str, Union[int, float]]]:
    """Perform automated clustering-based debris exclusion on scatter morphology (FSC vs SSC).

    Deconvolves the bivariate forward scatter and side scatter distribution using a
    Gaussian Mixture Model (GMM) to differentiate between physiological intact cellular
    populations and sub-cellular debris, sheath fluid fluctuations, and detector saturation.
    Extracts an enclosing PolygonGate via the convex hull of intact cells.

    Args:
        fcs_data: Parsed FCS sample data containing channel scale dynamic ranges.
        df_events: Events DataFrame containing scatter channels.
        n_components: Number of Gaussian mixture components to fit (default: 3).
        random_state: Random seed for reproducible GMM initialization.

    Returns:
        Tuple containing:
            - Boolean mask of intact cell events (length = N).
            - PolygonGate configured for 'Cells'.
            - Categorical label array per event ('Intact Cells', 'Debris / Noise', 'Saturation').
            - Dictionary of morphological QC summary statistics.
    """
    fsc_col = fcs_data.get_fsc_channel() if hasattr(fcs_data, "get_fsc_channel") else ("FSC-A" if "FSC-A" in df_events.columns else fcs_data.pnn_labels[0])
    ssc_col = fcs_data.get_ssc_channel() if hasattr(fcs_data, "get_ssc_channel") else ("SSC-A" if "SSC-A" in df_events.columns else (fcs_data.pnn_labels[1] if len(fcs_data.pnn_labels) > 1 else fcs_data.pnn_labels[0]))

    fsc_vals = df_events[fsc_col].to_numpy(dtype=np.float64)
    ssc_vals = df_events[ssc_col].to_numpy(dtype=np.float64)
    n_events = len(df_events)

    max_fsc = fcs_data.channel_ranges.get(fsc_col, float(np.max(fsc_vals)))
    max_ssc = fcs_data.channel_ranges.get(ssc_col, float(np.max(ssc_vals)))
    if max_fsc <= 0:
        max_fsc = 262144.0
    if max_ssc <= 0:
        max_ssc = 262144.0

    # Normalize scatter features to unit interval [0, 1]
    norm_fsc = fsc_vals / max_fsc
    norm_ssc = ssc_vals / max_ssc
    X_morph = np.column_stack([norm_fsc, norm_ssc])

    # Deterministic subsampling for rapid convergence (< 0.2s for 250k events)
    rng = np.random.RandomState(random_state)
    fit_size = min(50000, n_events)
    if n_events > fit_size:
        train_idx = rng.choice(n_events, size=fit_size, replace=False)
        train_data = X_morph[train_idx]
    else:
        train_data = X_morph

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        max_iter=150,
    )
    gmm.fit(train_data)
    cluster_ids = gmm.predict(X_morph)

    # Biological classification of GMM components:
    # 1. Saturation: cluster centroid pegged at detector ceiling (>= 0.96)
    # 2. Debris: cluster centroid in low-scatter regime (mean FSC < 0.15 and mean SSC < 0.15)
    # 3. Intact Cells: physiological mid-range scatter
    cell_clusters: List[int] = []
    debris_clusters: List[int] = []
    saturation_clusters: List[int] = []

    for k in range(n_components):
        mean_fsc = gmm.means_[k, 0]
        mean_ssc = gmm.means_[k, 1]
        if mean_fsc >= 0.96 or mean_ssc >= 0.96:
            saturation_clusters.append(k)
        elif mean_fsc < 0.10 or (mean_fsc < 0.15 and mean_ssc < 0.18) or (mean_fsc < 0.25 and mean_ssc < 0.15) or mean_ssc < 0.045:
            debris_clusters.append(k)
        else:
            cell_clusters.append(k)

    # Fallback heuristic: ensure at least one cluster is designated as cells
    if not cell_clusters:
        valid_cids = [k for k in range(n_components) if k not in saturation_clusters]
        if valid_cids:
            cell_clusters = [int(valid_cids[np.argmax([gmm.weights_[k] for k in valid_cids])])]
        else:
            cell_clusters = [int(np.argmax(gmm.weights_))]

    # Initial boolean mask for intact cells
    cell_mask = np.isin(cluster_ids, cell_clusters)

    # Hard boundary clipping against zero-noise spikes and saturated edge channels
    cell_mask &= (norm_fsc > 0.08) & (norm_fsc < 0.99) & (norm_ssc < 0.99)

    # Assign event-level descriptive labels
    morph_labels = np.full(n_events, "Debris / Noise", dtype=object)
    for sat_k in saturation_clusters:
        morph_labels[cluster_ids == sat_k] = "Saturation Outliers"
    morph_labels[cell_mask] = "Intact Cells"

    # Construct bounding PolygonGate via convex hull of inlying cellular coordinates
    cell_pts = np.column_stack([fsc_vals[cell_mask], ssc_vals[cell_mask]])
    if len(cell_pts) > 50:
        hull_sample_size = min(10000, len(cell_pts))
        hull_idx = rng.choice(len(cell_pts), size=hull_sample_size, replace=False)
        hull = ConvexHull(cell_pts[hull_idx])
        hull_vertices = cell_pts[hull_idx][hull.vertices].tolist()
    else:
        hull_vertices = [
            [max_fsc * 0.15, max_ssc * 0.05],
            [max_fsc * 0.85, max_ssc * 0.05],
            [max_fsc * 0.85, max_ssc * 0.85],
            [max_fsc * 0.15, max_ssc * 0.85],
        ]

    cells_gate = PolygonGate(
        name="Cells",
        channels=[fsc_col, ssc_col],
        vertices=hull_vertices,
        parent_gate=None,
    )

    stats = {
        "total_events": n_events,
        "cell_count": int(np.sum(cell_mask)),
        "cell_pct": float(np.sum(cell_mask) / n_events * 100.0) if n_events > 0 else 0.0,
        "debris_count": int(n_events - np.sum(cell_mask)),
        "debris_pct": float((n_events - np.sum(cell_mask)) / n_events * 100.0) if n_events > 0 else 0.0,
    }

    return cell_mask, cells_gate, morph_labels, stats


def cluster_singlets_qc(
    fcs_data: FCSData,
    df_events: pd.DataFrame,
    parent_mask: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, PolygonGate, np.ndarray, Dict[str, Union[int, float]]]:
    """Perform automated clustering-based singlet discrimination on pulse Area vs Height (FSC-A vs FSC-H).

    Single spherical cells passing individually through the interrogation zone produce a strictly
    linear area-to-height pulse relationship (FSC-H ~ m * FSC-A). Clumps and doublets have disproportionately
    larger pulse area, resulting in lower height-to-area ratios or off-diagonal residual spread.
    Uses a 2-component Gaussian Mixture Model on pulse ratios to isolate true singlets.

    Args:
        fcs_data: Parsed FCS sample data.
        df_events: Events DataFrame containing FSC-A and FSC-H.
        parent_mask: Optional boolean mask of intact cells from morphological QC.
        random_state: Random seed for reproducible GMM initialization.

    Returns:
        Tuple containing:
            - Boolean mask of singlet events (length = N).
            - PolygonGate configured for 'Singlets'.
            - Categorical label array per event ('Singlets', 'Doublets / Aggregates', 'Non-cellular').
            - Dictionary of singlet QC summary statistics.
    """
    fsc_a_col = fcs_data.get_fsc_channel() if hasattr(fcs_data, "get_fsc_channel") else ("FSC-A" if "FSC-A" in df_events.columns else fcs_data.pnn_labels[0])
    fsc_h_col = fcs_data.get_fsc_h_channel() if hasattr(fcs_data, "get_fsc_h_channel") else ("FSC-H" if "FSC-H" in df_events.columns else None)
    n_events = len(df_events)
    max_fsc = fcs_data.channel_ranges.get(fsc_a_col, 262144.0)

    # Graceful pass-through fallback if instrument does not record pulse height
    if fsc_h_col is None or fsc_h_col not in df_events.columns:
        base_mask = parent_mask if parent_mask is not None else np.ones(n_events, dtype=bool)
        fallback_gate = PolygonGate(
            name="Singlets",
            channels=[fsc_a_col, fsc_a_col],
            vertices=[[0.0, 0.0], [max_fsc, 0.0], [max_fsc, max_fsc], [0.0, max_fsc]],
            parent_gate="Cells",
        )
        labels = np.where(base_mask, "Singlets", "Non-cellular")
        stats = {
            "total_cells": int(np.sum(base_mask)),
            "singlet_count": int(np.sum(base_mask)),
            "singlet_pct_total": float(np.sum(base_mask) / n_events * 100.0) if n_events > 0 else 0.0,
            "singlet_pct_parent": 100.0,
            "doublet_count": 0,
            "doublet_pct": 0.0,
        }
        return base_mask, fallback_gate, labels, stats

    pts_a = df_events[fsc_a_col].to_numpy(dtype=np.float64)
    pts_h = df_events[fsc_h_col].to_numpy(dtype=np.float64)

    if parent_mask is not None and np.sum(parent_mask) > 50:
        sub_indices = np.where(parent_mask)[0]
    else:
        sub_indices = np.arange(n_events)

    sub_a = pts_a[sub_indices]
    sub_h = pts_h[sub_indices]

    # Calculate pulse height-to-area ratio
    valid_pulse = (sub_a > 1000.0) & (sub_h > 1000.0)
    ratios = np.zeros_like(sub_a)
    ratios[valid_pulse] = sub_h[valid_pulse] / sub_a[valid_pulse]

    clean_subset = ratios[valid_pulse & (ratios >= 0.2) & (ratios <= 2.5)]
    med_ratio = float(np.median(clean_subset)) if len(clean_subset) > 50 else 0.85

    # Fit 2-component GMM on 1D pulse ratio to separate high-density Singlet ridge from Doublet tail
    rng = np.random.RandomState(random_state)
    fit_size = min(50000, len(clean_subset))
    if len(clean_subset) > fit_size:
        train_ratios = rng.choice(clean_subset, size=fit_size, replace=False)
    else:
        train_ratios = clean_subset

    gmm_singlets = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=random_state,
        max_iter=150,
    )
    gmm_singlets.fit(train_ratios.reshape(-1, 1))

    # The singlet component is the dominant narrow Gaussian whose mean is closest to med_ratio
    singlet_cid = int(np.argmin(np.abs(gmm_singlets.means_.flatten() - med_ratio)))

    # Compute posterior probability that event belongs to singlet component
    eval_ratios = np.where(valid_pulse, ratios, 0.0)
    posteriors = gmm_singlets.predict_proba(eval_ratios.reshape(-1, 1))
    p_singlet = posteriors[:, singlet_cid]

    # Events with P(Singlet) >= 0.5 and ratio within physiological boundaries
    sub_is_singlet = (p_singlet >= 0.5) & valid_pulse & (ratios >= 0.4) & (ratios <= 1.8)

    full_singlet_mask = np.zeros(n_events, dtype=bool)
    full_singlet_mask[sub_indices[sub_is_singlet]] = True

    # Assign descriptive labels
    singlet_labels = np.full(n_events, "Debris / Non-cells", dtype=object)
    if parent_mask is not None:
        singlet_labels[parent_mask] = "Doublets / Aggregates"
    singlet_labels[full_singlet_mask] = "Singlets"

    # Construct bounding PolygonGate via convex hull of singlet events
    sing_pts = np.column_stack([pts_a[full_singlet_mask], pts_h[full_singlet_mask]])
    if len(sing_pts) > 50:
        hull_sample_size = min(10000, len(sing_pts))
        sh_idx = rng.choice(len(sing_pts), size=hull_sample_size, replace=False)
        hull_s = ConvexHull(sing_pts[sh_idx])
        singlet_vertices = sing_pts[sh_idx][hull_s.vertices].tolist()
    else:
        singlet_vertices = [
            [max_fsc * 0.1, max_fsc * 0.1 * med_ratio * 0.8],
            [max_fsc * 0.9, max_fsc * 0.9 * med_ratio * 0.8],
            [max_fsc * 0.9, max_fsc * 0.9 * med_ratio * 1.2],
            [max_fsc * 0.1, max_fsc * 0.1 * med_ratio * 1.2],
        ]

    singlets_gate = PolygonGate(
        name="Singlets",
        channels=[fsc_a_col, fsc_h_col],
        vertices=singlet_vertices,
        parent_gate="Cells",
    )

    n_cells = len(sub_indices)
    n_singlets = int(np.sum(full_singlet_mask))
    n_doublets = n_cells - n_singlets

    stats = {
        "total_cells": n_cells,
        "singlet_count": n_singlets,
        "singlet_pct_total": float(n_singlets / n_events * 100.0) if n_events > 0 else 0.0,
        "singlet_pct_parent": float(n_singlets / n_cells * 100.0) if n_cells > 0 else 0.0,
        "doublet_count": n_doublets,
        "doublet_pct": float(n_doublets / n_cells * 100.0) if n_cells > 0 else 0.0,
    }

    return full_singlet_mask, singlets_gate, singlet_labels, stats


def run_clustering_qc(
    fcs_data: FCSData,
    df_events: pd.DataFrame,
    n_morph_components: int = 3,
    random_state: int = 42,
) -> ClusteringQCResult:
    """Execute end-to-end automated clustering-based quality control (QC).

    Performs sequential unsupervised deconvolution:
    1. Morphology QC: GMM clustering on FSC-A vs SSC-A to exclude debris and saturation artifacts.
    2. Singlet Discrimination QC: GMM clustering on FSC-A vs FSC-H to exclude doublets and aggregates.

    Produces full boolean masks, empirical PolygonGates, population label assignments, and
    an aggregated summary statistics DataFrame.

    Args:
        fcs_data: Parsed FCS sample data.
        df_events: Raw or transformed events DataFrame.
        n_morph_components: Number of GMM components for morphological scatter (default: 3).
        random_state: Random seed for reproducibility.

    Returns:
        ClusteringQCResult holding masks, gates, event labels, and summary metrics.
    """
    total_n = len(df_events)

    # 1. Morphology QC
    cell_mask, cell_gate, morph_labels, morph_stats = cluster_morphology_qc(
        fcs_data=fcs_data,
        df_events=df_events,
        n_components=n_morph_components,
        random_state=random_state,
    )

    # 2. Singlet Discrimination QC
    singlet_mask, singlet_gate, singlet_labels, sing_stats = cluster_singlets_qc(
        fcs_data=fcs_data,
        df_events=df_events,
        parent_mask=cell_mask,
        random_state=random_state,
    )

    cell_count = int(morph_stats["cell_count"])
    debris_count = int(morph_stats["debris_count"])
    singlet_count = int(sing_stats["singlet_count"])
    doublet_count = int(sing_stats["doublet_count"])

    summary_df = pd.DataFrame([
        {
            "QC Stage": "1. Raw Acquisition",
            "Population": "Total Events",
            "Event Count": total_n,
            "% Total": 100.0,
            "% Parent": 100.0,
            "Methodology": "Flow Cytometer Ingestion",
        },
        {
            "QC Stage": "2. Scatter Debris Exclusion",
            "Population": "Intact Cells",
            "Event Count": cell_count,
            "% Total": round(float(cell_count / total_n * 100.0), 2) if total_n > 0 else 0.0,
            "% Parent": round(float(cell_count / total_n * 100.0), 2) if total_n > 0 else 0.0,
            "Methodology": f"GMM Scatter Clustering (K={n_morph_components})",
        },
        {
            "QC Stage": "2. Scatter Debris Exclusion",
            "Population": "Debris / Saturation",
            "Event Count": debris_count,
            "% Total": round(float(debris_count / total_n * 100.0), 2) if total_n > 0 else 0.0,
            "% Parent": round(float(debris_count / total_n * 100.0), 2) if total_n > 0 else 0.0,
            "Methodology": "Excluded by Scatter GMM",
        },
        {
            "QC Stage": "3. Doublet Discrimination",
            "Population": "Singlets",
            "Event Count": singlet_count,
            "% Total": round(float(singlet_count / total_n * 100.0), 2) if total_n > 0 else 0.0,
            "% Parent": round(float(singlet_count / cell_count * 100.0), 2) if cell_count > 0 else 0.0,
            "Methodology": "GMM Pulse Ratio Clustering (K=2)",
        },
        {
            "QC Stage": "3. Doublet Discrimination",
            "Population": "Doublets / Aggregates",
            "Event Count": doublet_count,
            "% Total": round(float(doublet_count / total_n * 100.0), 2) if total_n > 0 else 0.0,
            "% Parent": round(float(doublet_count / cell_count * 100.0), 2) if cell_count > 0 else 0.0,
            "Methodology": "Excluded by Pulse Ratio GMM",
        },
    ])

    return ClusteringQCResult(
        cells_mask=cell_mask,
        singlets_mask=singlet_mask,
        cells_gate=cell_gate,
        singlets_gate=singlet_gate,
        total_events=total_n,
        cell_count=cell_count,
        singlet_count=singlet_count,
        debris_count=debris_count,
        doublet_count=doublet_count,
        morphology_labels=morph_labels,
        singlet_labels=singlet_labels,
        summary_stats=summary_df,
    )
