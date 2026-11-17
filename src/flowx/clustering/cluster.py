"""Unsupervised clustering module for flow cytometry data.

Provides SOM (Self-Organizing Map + Metaclustering) and PhenoGraph
(Jaccard graph + Louvain community detection) for automated population discovery.
Includes automated biological macro-grouping and dendrogram leaf reordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from minisom import MiniSom
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Container holding unsupervised clustering outputs and population phenotypes.

    Attributes:
        method: Name of the clustering method ('som' or 'phenograph').
        cluster_labels: 1D NumPy array of assigned cluster integer labels per event.
        cluster_count: Total number of discovered clusters.
        markers_used: List of channel detector names used for clustering.
        cluster_statistics: DataFrame with event counts, percentages, and median marker intensities.
        phenotype_labels: Mapping from cluster ID to biological phenotype description.
        macro_groups_df: DataFrame summarizing aggregated statistics per biological macro-group.
        cluster_to_macro_group: Mapping from cluster ID to higher-level lineage name.
        ordered_indices: Cluster row indices reordered by dendrogram leaf order.
        parent_population: Name of the parent population clustered (e.g. 'Viable', 'Singlets', 'All Events').
        parent_count: Number of events in the parent population clustered.
        total_sample_count: Total events in the whole FCS sample.
        full_cluster_labels: 1D array across the full FCS file with -1 for unclustered/gated-out events.
        annotation_audit_df: Per-cluster transparent provenance audit table recording thresholds,
            empirical +/- calls, matched signature rules, and confidence scores.
        marker_name_map: Resolved channel ($PnN) -> biological marker symbol mapping used for annotation.
        marker_thresholds: Computed data-driven positivity cutoffs per clustering channel.
    """

    method: str
    cluster_labels: np.ndarray
    cluster_count: int
    markers_used: List[str]
    cluster_statistics: pd.DataFrame
    phenotype_labels: Dict[int, str] = field(default_factory=dict)
    macro_groups_df: Optional[pd.DataFrame] = None
    cluster_to_macro_group: Dict[int, str] = field(default_factory=dict)
    ordered_indices: List[int] = field(default_factory=list)
    parent_population: str = "All Events"
    parent_count: int = 0
    total_sample_count: int = 0
    full_cluster_labels: Optional[np.ndarray] = None
    k_optimization_scores: Dict[int, float] = field(default_factory=dict)
    k_optimization_metric: str = ""
    annotation_audit_df: Optional[pd.DataFrame] = None
    marker_name_map: Dict[str, str] = field(default_factory=dict)
    marker_thresholds: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_cluster_mask(self, cluster_id: int) -> np.ndarray:
        """Get boolean mask for events belonging to a specific cluster."""
        return self.cluster_labels == cluster_id


def find_optimal_metaclusters(
    weights: np.ndarray,
    k_min: int = 4,
    k_max: int = 20,
    method: str = "consensus",
    consensus_subsample: float = 0.8,
    consensus_iterations: int = 50,
    random_state: int = 42,
) -> Tuple[int, Dict[int, float], str]:
    """Determine optimal number of metaclusters (k) using user-selected algorithmic criteria.

    Supported evaluation algorithms:
    - 'consensus' (Default): Consensus Metaclustering (Monti et al., 2003 / FlowSOM Van Gassen et al., 2015).
      Subsamples SOM codebook nodes multiple times to compute the consensus matrix and minimizes
      the Proportion of Ambiguous Clustering (PAC) for maximal cluster stability.
    - 'davies_bouldin': Minimizes Davies-Bouldin index (ratio of within-cluster spread to separation).
    - 'silhouette': Maximizes mean Silhouette score (cluster cohesion versus separation).
    - 'calinski_harabasz': Maximizes Calinski-Harabasz variance ratio criterion (between/within variance).
    - 'elbow': Finds the knee / inflection point of within-cluster sum of squares (inertia).

    Args:
        weights: 2D array of SOM codebook node weights [n_nodes x n_markers].
        k_min: Minimum metaclusters to evaluate (default: 4).
        k_max: Maximum metaclusters to evaluate (default: 20).
        method: Evaluation algorithm ('consensus', 'davies_bouldin', 'silhouette', 'calinski_harabasz', 'elbow').
        consensus_subsample: Fraction of SOM nodes to subsample per iteration when method='consensus' (default: 0.8).
        consensus_iterations: Number of resampling iterations when method='consensus' (default: 50).
        random_state: Deterministic PRNG seed.

    Returns:
        Tuple of (optimal_k, scores_dict, metric_name).
    """
    scores: Dict[int, float] = {}
    best_k = k_min
    max_k = min(k_max, len(weights) - 1)
    norm_method = (method or "consensus").lower().strip()

    if max_k <= k_min:
        return k_min, {k_min: 1.0}, norm_method

    if norm_method == "davies_bouldin":
        metric_name = "davies_bouldin_index"
        best_score = float("inf")
        for k in range(k_min, max_k + 1):
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = agg.fit_predict(weights)
                if len(set(labels)) > 1:
                    score = float(davies_bouldin_score(weights, labels))
                    scores[k] = score
                    if score < best_score:
                        best_score = score
                        best_k = k
            except Exception:
                continue

    elif norm_method == "silhouette":
        metric_name = "silhouette_score"
        best_score = -float("inf")
        for k in range(k_min, max_k + 1):
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = agg.fit_predict(weights)
                if len(set(labels)) > 1:
                    score = float(silhouette_score(weights, labels))
                    scores[k] = score
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception:
                continue

    elif norm_method == "calinski_harabasz":
        metric_name = "calinski_harabasz_score"
        best_score = -float("inf")
        for k in range(k_min, max_k + 1):
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = agg.fit_predict(weights)
                if len(set(labels)) > 1:
                    score = float(calinski_harabasz_score(weights, labels))
                    scores[k] = score
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception:
                continue

    elif norm_method == "elbow":
        metric_name = "wcss_inertia"
        wcss: Dict[int, float] = {}
        for k in range(k_min, max_k + 1):
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = agg.fit_predict(weights)
                val = 0.0
                for cl in set(labels):
                    pts = weights[labels == cl]
                    centroid = np.mean(pts, axis=0)
                    val += float(np.sum((pts - centroid) ** 2))
                wcss[k] = val
            except Exception:
                continue
        scores = wcss
        k_vals = np.array(list(wcss.keys()))
        w_vals = np.array(list(wcss.values()))
        if len(k_vals) <= 2:
            best_k = int(k_vals[0]) if len(k_vals) > 0 else k_min
        else:
            w_min, w_max = float(np.min(w_vals)), float(np.max(w_vals))
            k_norm = (k_vals - k_vals[0]) / (k_vals[-1] - k_vals[0] + 1e-12)
            w_norm = (w_vals - w_min) / (w_max - w_min + 1e-12)
            # Signed perpendicular distance from diagonal line connecting (0, 1) and (1, 0)
            # A true elbow in a convex decreasing curve bows below the secant line: (1 - (k_norm + w_norm)) > 0
            dists = (1.0 - (k_norm + w_norm)) / np.sqrt(2)
            best_k = int(k_vals[np.argmax(dists)])

    else:  # 'consensus' (Canonical FlowSOM default)
        metric_name = "consensus_pac"
        N = len(weights)
        # Ensure sub_n is strictly greater than max_k to avoid degenerate singleton clustering or crashes
        desired_sub_n = int(round(N * consensus_subsample))
        sub_n = min(N, max(max_k + 2, desired_sub_n))
        if sub_n <= max_k:
            sub_n = N

        rng = np.random.default_rng(random_state)
        # Pre-generate the B subsample partitions once so every candidate k is evaluated on identical resamples (Monti et al., 2003)
        subsamples = [rng.choice(N, size=sub_n, replace=False) for _ in range(consensus_iterations)]

        for k in range(k_min, max_k + 1):
            C = np.zeros((N, N), dtype=int)
            I_mat = np.zeros((N, N), dtype=int)
            for sub_idx in subsamples:
                sub_weights = weights[sub_idx]
                labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(sub_weights)
                same = (labels[:, None] == labels[None, :]).astype(int)
                ix = np.ix_(sub_idx, sub_idx)
                I_mat[ix] += 1
                C[ix] += same

            np.fill_diagonal(I_mat, 0)
            np.fill_diagonal(C, 0)
            mask = np.triu(I_mat > 0, k=1)
            if not np.any(mask):
                scores[k] = 1.0
                continue
            consensus_vals = C[mask] / I_mat[mask]
            # PAC: proportion of consensus values in intermediate / ambiguous range (0.1, 0.9)
            pac = float(np.mean((consensus_vals > 0.1) & (consensus_vals < 0.9)))
            scores[k] = pac

        # Select optimal k: minimize PAC.
        # If multiple k achieve minimal PAC (within tolerance 0.005), select the largest k in that minimal plateau (Senbabaoglu et al., 2014)
        if scores:
            min_pac = min(scores.values())
            candidate_ks = [k for k, p in scores.items() if p <= min_pac + 0.005]
            best_k = max(candidate_ks)
        else:
            best_k = k_min

    return best_k, scores, metric_name


# Declarative biological cell-type signature rules keyed strictly by standardized antigen symbol.
# Rules ONLY trigger if all required positive and negative markers are verified in the active panel.
DEFAULT_CELL_SIGNATURES: List[Dict[str, Any]] = [
    # Fine-grained CD4+ T-cell subsets (requires CCR7 + CD45RA)
    {
        "name": "CD4+ Naive T Cells",
        "macro_group": "CD4+ T-Cell Lineage",
        "required_positive": ["CD3", "CD4", "CCR7", "CD45RA"],
        "required_negative": ["CD8"],
        "priority": 90,
    },
    {
        "name": "CD4+ Central Memory T Cells (TCM)",
        "macro_group": "CD4+ T-Cell Lineage",
        "required_positive": ["CD3", "CD4", "CCR7"],
        "required_negative": ["CD8", "CD45RA"],
        "priority": 90,
    },
    {
        "name": "CD4+ Effector Memory T Cells (TEM)",
        "macro_group": "CD4+ T-Cell Lineage",
        "required_positive": ["CD3", "CD4"],
        "required_negative": ["CD8", "CCR7", "CD45RA"],
        "priority": 88,
    },
    {
        "name": "CD4+ TEMRA T Cells",
        "macro_group": "CD4+ T-Cell Lineage",
        "required_positive": ["CD3", "CD4", "CD45RA"],
        "required_negative": ["CD8", "CCR7"],
        "priority": 88,
    },
    # Fine-grained CD8+ T-cell subsets
    {
        "name": "CD8+ Naive / TEMRA T Cells",
        "macro_group": "CD8+ T-Cell Lineage",
        "required_positive": ["CD3", "CD8", "CD45RA"],
        "required_negative": ["CD4"],
        "priority": 86,
    },
    {
        "name": "CD8+ Central Memory T Cells (TCM)",
        "macro_group": "CD8+ T-Cell Lineage",
        "required_positive": ["CD3", "CD8", "CCR7"],
        "required_negative": ["CD4", "CD45RA"],
        "priority": 86,
    },
    {
        "name": "CD8+ Effector / Memory T Cells",
        "macro_group": "CD8+ T-Cell Lineage",
        "required_positive": ["CD3", "CD8"],
        "required_negative": ["CD4", "CCR7"],
        "priority": 84,
    },
    # Canonical broad T-cell compartments
    {
        "name": "CD4+ T-Cell Lineage",
        "macro_group": "CD4+ T-Cell Lineage",
        "required_positive": ["CD3", "CD4"],
        "required_negative": ["CD8"],
        "priority": 60,
    },
    {
        "name": "CD8+ T-Cell Lineage",
        "macro_group": "CD8+ T-Cell Lineage",
        "required_positive": ["CD3", "CD8"],
        "required_negative": ["CD4"],
        "priority": 60,
    },
    {
        "name": "CD4+ CD8+ DP T Cells",
        "macro_group": "CD4+ CD8+ DP T Cells",
        "required_positive": ["CD3", "CD4", "CD8"],
        "required_negative": [],
        "priority": 62,
    },
    {
        "name": "Immature Thymocytes (CD1a+ CD4+ CD8+)",
        "macro_group": "CD4+ CD8+ DP T Cells",
        "required_positive": ["CD1A", "CD4", "CD8"],
        "required_negative": ["CD3"],
        "priority": 76,
    },
    {
        "name": "CD3+ DN / Other T Cells",
        "macro_group": "CD3+ DN / Other T Cells",
        "required_positive": ["CD3"],
        "required_negative": ["CD4", "CD8"],
        "priority": 55,
    },
    # Myeloid / Macrophage signatures (e.g. THP-1 / monocyte panels)
    {
        "name": "Anti-Inflammatory M2 Macrophages (CD206+)",
        "macro_group": "Anti-Inflammatory M2 Macrophages",
        "required_positive": ["CD33", "CD206"],
        "required_negative": [],
        "priority": 80,
    },
    {
        "name": "Pro-Inflammatory M1 Macrophages (CD80+ CD206-)",
        "macro_group": "Pro-Inflammatory M1 Macrophages",
        "required_positive": ["CD33", "CD80"],
        "required_negative": ["CD206"],
        "priority": 80,
    },
    {
        "name": "Basal Macrophages (CD33+ CD11b+)",
        "macro_group": "Basal / Unpolarized Macrophages",
        "required_positive": ["CD33", "CD11b"],
        "required_negative": [],
        "priority": 50,
    },
    # APC / B-cell / NK non-T signatures
    {
        "name": "Transitional B Cells (CD19+ IgD+ CD24+ CD38+)",
        "macro_group": "B-Cell Lineage",
        "required_positive": ["CD19", "IGD", "CD24", "CD38"],
        "required_negative": ["CD3", "CD27"],
        "exclude_if_positive": ["CD14", "CD33", "CD66B", "CD15", "LY6G", "F4/80", "CD56"],
        "priority": 90,
    },
    {
        "name": "Plasmablasts / Plasma Cells (CD19+ CD27+ CD38+ IgD-)",
        "macro_group": "B-Cell Lineage",
        "required_positive": ["CD19", "CD27", "CD38"],
        "required_negative": ["CD3", "IGD"],
        "exclude_if_positive": ["CD14", "CD33", "CD66B", "CD15", "LY6G", "F4/80", "CD56"],
        "priority": 90,
    },
    {
        "name": "Unswitched Memory B Cells (CD19+ IgD+ CD27+)",
        "macro_group": "B-Cell Lineage",
        "required_positive": ["CD19", "IGD", "CD27"],
        "required_negative": ["CD3"],
        "exclude_if_positive": ["CD14", "CD33", "CD66B", "CD15", "LY6G", "F4/80", "CD56"],
        "priority": 88,
    },
    {
        "name": "Switched Memory B Cells (CD19+ IgD- CD27+)",
        "macro_group": "B-Cell Lineage",
        "required_positive": ["CD19", "CD27"],
        "required_negative": ["CD3", "IGD"],
        "exclude_if_positive": ["CD14", "CD33", "CD66B", "CD15", "LY6G", "F4/80", "CD56"],
        "priority": 86,
    },
    {
        "name": "Naive B Cells (CD19+ IgD+ CD27-)",
        "macro_group": "B-Cell Lineage",
        "required_positive": ["CD19", "IGD"],
        "required_negative": ["CD3", "CD27"],
        "exclude_if_positive": ["CD14", "CD33", "CD66B", "CD15", "LY6G", "F4/80", "CD56"],
        "priority": 85,
    },
    {
        "name": "Double Negative (DN) B Cells (CD19+ IgD- CD27-)",
        "macro_group": "B-Cell Lineage",
        "required_positive": ["CD19"],
        "required_negative": ["CD3", "IGD", "CD27"],
        "exclude_if_positive": ["CD14", "CD33", "CD66B", "CD15", "LY6G", "F4/80", "CD56"],
        "priority": 82,
    },
    {
        "name": "B-Cell Lineage (CD19+)",
        "macro_group": "B-Cell Lineage",
        "required_positive": ["CD19"],
        "required_negative": ["CD3"],
        "exclude_if_positive": ["CD14", "CD33", "CD66B", "CD15", "LY6G", "F4/80", "CD56"],
        "priority": 75,
    },
    {
        "name": "B-Cell Lineage (CD20+)",
        "macro_group": "B-Cell Lineage",
        "required_positive": ["CD20"],
        "required_negative": ["CD3"],
        "exclude_if_positive": ["CD14", "CD33", "CD66B", "CD15", "LY6G", "F4/80"],
        "exclude_if_measured_negative": ["CD19"],
        "priority": 74,
    },
    {
        "name": "Classical Monocytes (CD14+ CD16-)",
        "macro_group": "Monocyte / Myeloid Lineage",
        "required_positive": ["CD14"],
        "required_negative": ["CD3", "CD16"],
        "exclude_if_positive": ["CD66B", "CD15", "CD19"],
        "priority": 78,
    },
    {
        "name": "Intermediate Monocytes (CD14+ CD16+)",
        "macro_group": "Monocyte / Myeloid Lineage",
        "required_positive": ["CD14", "CD16"],
        "required_negative": ["CD3"],
        "exclude_if_positive": ["CD66B", "CD15", "CD19"],
        "priority": 78,
    },
    {
        "name": "Non-Classical Monocytes (CD14- CD16+ HLA-DR+)",
        "macro_group": "Monocyte / Myeloid Lineage",
        "required_positive": ["CD16", "HLA-DR"],
        "required_negative": ["CD3", "CD14"],
        "exclude_if_positive": ["CD66B", "CD15", "CD19", "CD56"],
        "priority": 77,
    },
    {
        "name": "CD14+ Classical Monocytes",
        "macro_group": "Monocyte / Myeloid Lineage",
        "required_positive": ["CD14"],
        "required_negative": ["CD3"],
        "exclude_if_positive": ["CD66B", "CD15"],
        "priority": 75,
    },
    {
        "name": "Neutrophils / Granulocytes (CD66b+ CD15+)",
        "macro_group": "Granulocyte / Neutrophil Lineage",
        "required_positive": ["CD66B", "CD15"],
        "required_negative": [],
        "priority": 85,
    },
    {
        "name": "Neutrophils / Granulocytes (CD66b+)",
        "macro_group": "Granulocyte / Neutrophil Lineage",
        "required_positive": ["CD66B"],
        "required_negative": [],
        "priority": 82,
    },
    {
        "name": "Granulocytes (CD15+ CD16+)",
        "macro_group": "Granulocyte / Neutrophil Lineage",
        "required_positive": ["CD15", "CD16"],
        "required_negative": ["CD14"],
        "priority": 82,
    },
    # Murine Myeloid & Innate Signatures
    {
        "name": "Murine Neutrophils (Ly6G+ CD11b+)",
        "macro_group": "Granulocyte / Neutrophil Lineage",
        "required_positive": ["LY6G", "CD11B"],
        "required_negative": [],
        "priority": 85,
    },
    {
        "name": "Murine Neutrophils (Ly6G+)",
        "macro_group": "Granulocyte / Neutrophil Lineage",
        "required_positive": ["LY6G"],
        "required_negative": [],
        "priority": 82,
    },
    {
        "name": "Murine Macrophages (F4/80+ CD11b+)",
        "macro_group": "Monocyte / Myeloid Lineage",
        "required_positive": ["F4/80", "CD11B"],
        "required_negative": ["LY6G"],
        "priority": 83,
    },
    {
        "name": "Murine Macrophages (F4/80+)",
        "macro_group": "Monocyte / Myeloid Lineage",
        "required_positive": ["F4/80"],
        "required_negative": [],
        "exclude_if_positive": ["LY6G"],
        "priority": 80,
    },
    {
        "name": "Murine Inflammatory Monocytes (Ly6C+ CD11b+)",
        "macro_group": "Monocyte / Myeloid Lineage",
        "required_positive": ["LY6C", "CD11B"],
        "required_negative": [],
        "exclude_if_positive": ["LY6G", "F4/80"],
        "priority": 81,
    },
    {
        "name": "Dendritic Cells (CD11c+ HLA-DR+)",
        "macro_group": "Dendritic Cell Lineage",
        "required_positive": ["CD11C", "HLA-DR"],
        "required_negative": [],
        "exclude_if_positive": ["CD3", "CD14", "CD19", "CD66B"],
        "priority": 79,
    },
    {
        "name": "Myeloid Cells (CD11b+)",
        "macro_group": "Monocyte / Myeloid Lineage",
        "required_positive": ["CD11B"],
        "required_negative": [],
        "exclude_if_positive": ["CD3", "CD19", "LY6G", "CD66B"],
        "priority": 58,
    },
    # Stem / Progenitor / Hematopoietic Organoid Signatures
    {
        "name": "Hematopoietic Stem & Progenitor Cells (CD34+ c-Kit+)",
        "macro_group": "HSPC / Progenitor Lineage",
        "required_positive": ["CD34", "C-KIT"],
        "required_negative": [],
        "priority": 88,
    },
    {
        "name": "CD34+ c-Kit- Endothelial / Stromal Lineage",
        "macro_group": "Endothelial / Stromal Lineage",
        "required_positive": ["CD34"],
        "required_negative": ["C-KIT"],
        "priority": 85,
    },
    {
        "name": "Hematopoietic Stem & Progenitor Cells (CD34+)",
        "macro_group": "HSPC / Progenitor Lineage",
        "required_positive": ["CD34"],
        "required_negative": [],
        "exclude_if_positive": ["CD3", "TCRAB", "CD19", "CD20", "CD56", "CD66B", "LY6G", "CD1A"],
        "priority": 83,
    },
    {
        "name": "Erythroid Progenitors (CD235a+ CD71+)",
        "macro_group": "Erythroid Lineage",
        "required_positive": ["CD235A", "CD71"],
        "required_negative": [],
        "priority": 86,
    },
    {
        "name": "Erythroid Lineage (CD235a+)",
        "macro_group": "Erythroid Lineage",
        "required_positive": ["CD235A"],
        "required_negative": [],
        "priority": 82,
    },
    {
        "name": "Megakaryocytic Lineage (CD41+ CD42b+)",
        "macro_group": "Megakaryocytic Lineage",
        "required_positive": ["CD41", "CD42B"],
        "required_negative": [],
        "priority": 86,
    },
    {
        "name": "Megakaryocytic Progenitors (CD41+)",
        "macro_group": "Megakaryocytic Lineage",
        "required_positive": ["CD41"],
        "required_negative": [],
        "priority": 80,
    },
    {
        "name": "Endothelial / Hemogenic Endothelium (CD31+ CD45-)",
        "macro_group": "Endothelial / Stromal Lineage",
        "required_positive": ["CD31"],
        "required_negative": ["CD45"],
        "priority": 80,
    },
    {
        "name": "HLA-DR+ APC / Monocyte / B-Cell Lineage",
        "macro_group": "HLA-DR+ APC / Monocyte Lineage",
        "required_positive": ["HLA-DR"],
        "required_negative": ["CD3"],
        "exclude_if_positive": ["CD66B", "CD15", "TCRAB"],
        "priority": 55,
    },
    {
        "name": "NK Cells (CD56+)",
        "macro_group": "NK-Cell Lineage",
        "required_positive": ["CD56"],
        "required_negative": ["CD3"],
        "exclude_if_positive": ["CD14", "CD19", "CD66B"],
        "priority": 70,
    },
    {
        "name": "CD3- Non-T Lymphocytes",
        "macro_group": "CD3- Non-T Lymphocytes",
        "required_positive": [],
        "required_negative": ["CD3"],
        "exclude_if_positive": ["CD14", "CD33", "CD66B", "CD15", "LY6G", "F4/80", "CD11B", "CD11C", "CD235A", "CD41", "CD34", "CD1A", "TCRAB"],
        "priority": 30,
    },
]


def _normalize_marker_symbol(sym: str) -> str:
    """Normalize standardized biological antigen aliases for signature matching."""
    import re
    clean = str(sym).strip()
    clean = re.sub(r"\s*\([^)]*\)", "", clean).strip().upper()
    clean = clean.replace("HLADR", "HLA-DR").replace("HLA DR", "HLA-DR")
    clean = clean.replace("LY-6G", "LY6G").replace("LY-6C", "LY6C").replace("F4-80", "F4/80")
    clean = clean.replace("CD66B/CD15", "CD66B").replace("CD197", "CCR7").replace("CD235A/GYPA", "CD235A")
    clean = clean.replace("CD117", "C-KIT").replace("CKIT", "C-KIT")
    return clean


def _normalize_signature_entries(raw: Any, source_desc: str) -> List[Dict[str, Any]]:
    """Normalize list-based or dict-mapped cell signature definitions into canonical rule dicts."""
    if isinstance(raw, dict) and "signatures" in raw:
        raw = raw["signatures"]

    entries: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            if not isinstance(item, dict) or not item.get("name"):
                raise ValueError(f"Invalid signature rule at index {idx} in {source_desc}; expected dict with 'name'.")
            entries.append({
                "name": str(item["name"]),
                "macro_group": str(item.get("macro_group", item["name"])),
                "required_positive": list(item.get("required_positive", item.get("positive", []))),
                "required_negative": list(item.get("required_negative", item.get("negative", []))),
                "exclude_if_positive": list(item.get("exclude_if_positive", [])),
                "exclude_if_measured_negative": list(item.get("exclude_if_measured_negative", [])),
                "priority": int(item.get("priority", 50)),
            })
        return entries

    if isinstance(raw, dict):
        if not raw:
            raise ValueError(f"Empty cell_signatures dictionary in {source_desc}.")
        for name, spec in raw.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Invalid signature spec for '{name}' in {source_desc}; expected mapping.")
            entries.append({
                "name": str(name),
                "macro_group": str(spec.get("macro_group", name)),
                "required_positive": list(spec.get("required_positive", spec.get("positive", []))),
                "required_negative": list(spec.get("required_negative", spec.get("negative", []))),
                "exclude_if_positive": list(spec.get("exclude_if_positive", [])),
                "exclude_if_measured_negative": list(spec.get("exclude_if_measured_negative", [])),
                "priority": int(spec.get("priority", 50)),
            })
        return entries

    raise ValueError(f"Unsupported cell_signatures format in {source_desc}")


def load_cell_signatures(
    signatures_source: Optional[Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Load custom or default cell signature definitions (list or name-keyed mapping)."""
    if signatures_source is None:
        return DEFAULT_CELL_SIGNATURES
    if isinstance(signatures_source, (str, Path)):
        p = Path(signatures_source)
        if not p.is_file():
            raise FileNotFoundError(f"Cell signatures file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            import json
            loaded = json.loads(text)
        else:
            import yaml
            loaded = yaml.safe_load(text)
        return _normalize_signature_entries(loaded, str(p))
    return _normalize_signature_entries(signatures_source, "inline cell_signatures")


def _otsu_1d(vals: np.ndarray, n_bins: int = 64) -> float:
    """Compute 1D Otsu inter-class variance optimal threshold."""
    if len(vals) < 10:
        return float(np.median(vals))
    lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
    if hi - lo < 1e-9:
        return (lo + hi) / 2.0
    hist, bin_edges = np.histogram(np.clip(vals, lo, hi), bins=n_bins, range=(lo, hi))
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w1 = np.cumsum(hist).astype(np.float64)
    w2 = np.cumsum(hist[::-1])[::-1].astype(np.float64)
    m1 = np.cumsum(hist * centers) / np.maximum(w1, 1e-12)
    m2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(w2[::-1], 1e-12))[::-1]
    between_var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    best_idx = int(np.argmax(between_var))
    return float(centers[best_idx])


def compute_marker_positivity_thresholds(
    cluster_stats_df: pd.DataFrame,
    markers: Sequence[str],
    events_df: Optional[pd.DataFrame] = None,
    min_dynamic_range: float = 0.12,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Compute transparent, data-driven positivity cutoffs per marker channel.

    Replaces hardcoded magic thresholds (e.g. 0.35 / 0.40) with empirical thresholds
    derived directly from transformed event distributions and cluster median spans:
    - Bimodal / wide-range markers: 1D Otsu threshold bounded safely inside cluster extremes.
    - Constitutively bright unimodal markers (p05 >= 0.30 & cluster_min >= 0.30): recognizes
      all clusters as positive (+) and identifies bright outliers via empirical Z-score >= 0.5.
    - Low-variance flat markers: flagged as 'uniform' rather than forcing artificial +/- splits.
    """
    thresholds: Dict[str, Dict[str, Any]] = {}
    rng = np.random.RandomState(random_state)

    for m in markers:
        m_col = f"{m} Median" if f"{m} Median" in cluster_stats_df.columns else m
        if m_col not in cluster_stats_df.columns:
            continue
        cm = cluster_stats_df[m_col].to_numpy(dtype=np.float64)
        c_min = float(np.min(cm))
        c_max = float(np.max(cm))
        c_mean = float(np.mean(cm))
        c_std = float(np.std(cm))
        c_span = c_max - c_min

        if events_df is not None and m in events_df.columns and len(events_df) > 20:
            raw_vals = events_df[m].dropna().to_numpy(dtype=np.float64)
            if len(raw_vals) > 25000:
                idx = rng.choice(len(raw_vals), size=25000, replace=False)
                raw_vals = raw_vals[idx]
            p05 = float(np.percentile(raw_vals, 5))
            p95 = float(np.percentile(raw_vals, 95))
            dyn_range = p95 - p05
            otsu_val = _otsu_1d(raw_vals)
        else:
            p05 = c_min
            p95 = c_max
            dyn_range = c_span
            otsu_val = (c_min + c_max) / 2.0

        # Identify marker regime transparently
        if c_max < 0.32 and p95 < 0.45:
            # All cluster centroids are in the negative baseline regime (e.g. CD4 on purified CD8+ T cells)
            cutoff = max(c_max + 0.05, 0.35)
            method = "all_clusters_negative"
        elif c_span < min_dynamic_range and dyn_range < min_dynamic_range:
            # Uniformly expressed across all clusters (check if constitutively bright vs negative)
            is_bright = c_mean >= 0.45 and p05 >= 0.35
            cutoff = c_min - 1e-6 if is_bright else c_max + 0.15
            method = "uniform_constitutive_pos" if is_bright else "uniform_low"
        elif p05 >= 0.50 and c_min >= 0.55 and c_span < 0.22:
            # Constitutively expressed across all cells (e.g. pan-leukocyte CD45 on purified leukocytes);
            # avoid splitting a single high-intensity positive cloud into false negative (-) clusters
            cutoff = max(0.35, c_min - 0.05)
            method = "unimodal_positive_constitutive"
        elif events_df is not None and dyn_range >= 0.20:
            # Clamp Otsu softly within observed cluster bounds when bimodal, with a baseline floor
            lo_bound = c_min + 0.10 * c_span
            hi_bound = c_max - 0.10 * c_span
            cutoff = float(np.clip(otsu_val, lo_bound, hi_bound))
            if p05 < 0.25 and c_max >= 0.35:
                cutoff = max(cutoff, 0.28)
            method = "otsu_bimodal"
        else:
            cutoff = (c_min + c_max) / 2.0
            if p05 < 0.25 and c_max >= 0.35:
                cutoff = max(cutoff, 0.28)
            method = "cluster_midpoint"

        hi_cutoff = max(cutoff + 0.12, c_mean + 0.65 * max(c_std, 1e-6))
        thresholds[m] = {
            "channel": m,
            "pos_cutoff": round(float(cutoff), 4),
            "hi_cutoff": round(float(hi_cutoff), 4),
            "cluster_min": round(c_min, 4),
            "cluster_max": round(c_max, 4),
            "cluster_mean": round(c_mean, 4),
            "cluster_std": round(c_std, 4),
            "method": method,
        }
    return thresholds


def annotate_clusters_transparently(
    cluster_stats_df: pd.DataFrame,
    markers: Sequence[str],
    events_df: Optional[pd.DataFrame] = None,
    marker_name_map: Optional[Dict[str, str]] = None,
    cluster_name_map: Optional[Dict[Any, str]] = None,
    cell_signatures: Optional[Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Tuple[Dict[int, str], Dict[int, str], pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Deterministically annotate clusters without hidden detector or threshold heuristics.

    Args:
        cluster_stats_df: Per-cluster median intensities & counts.
        markers: Clustering channel names ($PnN).
        events_df: Optional pooled transformed event DataFrame for Otsu threshold estimation.
        marker_name_map: Resolved channel -> biological antigen name mapping (e.g. from markers.csv).
        cluster_name_map: Optional explicit user Cluster ID -> Biological Label override.
        cell_signatures: Custom or default declarative cell signature rules.

    Returns:
        Tuple of (phenotype_labels_dict, macro_groups_dict, annotation_audit_df, thresholds_dict).
    """
    median_cols = [c for c in cluster_stats_df.columns if "Median" in c]
    if not median_cols:
        median_cols = [f"{m} Median" for m in markers if f"{m} Median" in cluster_stats_df.columns]
    ch_names = [c.replace(" Median", "") for c in median_cols] if median_cols else list(markers)

    thresholds = compute_marker_positivity_thresholds(
        cluster_stats_df=cluster_stats_df,
        markers=ch_names,
        events_df=events_df,
    )

    # Resolve channel -> display biological symbol
    resolved_map: Dict[str, str] = {}
    biological_detected = False
    for ch in ch_names:
        if marker_name_map and ch in marker_name_map and marker_name_map[ch]:
            sym = str(marker_name_map[ch]).strip()
            resolved_map[ch] = sym
            if sym != ch:
                biological_detected = True
        else:
            resolved_map[ch] = ch

    norm_to_channel: Dict[str, str] = {}
    for ch, sym in resolved_map.items():
        norm_sym = _normalize_marker_symbol(sym)
        if norm_sym not in norm_to_channel:
            norm_to_channel[norm_sym] = ch
        # Check if channel name itself already looks like canonical biological antigen (e.g. legacy tests)
        if norm_sym in {
            "CD3", "CD4", "CD8", "CCR7", "CD45RA", "HLA-DR", "CD19", "CD20", "CD33", "CD11B",
            "CD11C", "CD80", "CD206", "CD14", "CD16", "CD56", "CD66B", "CD15", "LY6G", "LY6C",
            "F4/80", "CD34", "CD235A", "CD71", "CD41", "CD42B", "CD31", "CD45", "DEAD",
        }:
            biological_detected = True

    signatures = load_cell_signatures(cell_signatures)
    # Only consider rules whose required markers were ALL actually measured & mapped
    measured_syms = set(norm_to_channel.keys())

    phenotypes: Dict[int, str] = {}
    macro_groups: Dict[int, str] = {}
    audit_rows: List[Dict[str, Any]] = []

    for _, row in cluster_stats_df.iterrows():
        cid = int(row["Cluster ID"])
        pos_syms: set[str] = set()
        neg_syms: set[str] = set()
        pos_display: List[str] = []
        hi_display: List[str] = []
        neg_display: List[str] = []

        for ch in ch_names:
            m_col = f"{ch} Median" if f"{ch} Median" in row else ch
            if m_col not in row or ch not in thresholds:
                continue
            val = float(row[m_col])
            th = thresholds[ch]
            sym = resolved_map[ch]
            norm_sym = _normalize_marker_symbol(sym)

            # Check dead/viability exclusion only if explicitly mapped as Dead/Viability
            if val >= th["pos_cutoff"]:
                pos_syms.add(norm_sym)
                if val >= th["hi_cutoff"] and (th["cluster_max"] - th["cluster_min"]) >= 0.12:
                    hi_display.append((sym, val - th["cluster_mean"]))
                    pos_display.append(f"{sym}++")
                else:
                    pos_display.append(f"{sym}+")
            else:
                neg_syms.add(norm_sym)
                neg_display.append(f"{sym}-")

        # Determine final annotation cleanly by transparent priority
        user_label = None
        if cluster_name_map and (cid in cluster_name_map or str(cid) in cluster_name_map):
            user_label = cluster_name_map.get(cid) or cluster_name_map.get(str(cid))

        matched_rule = None
        if user_label is None and biological_detected:
            best_score = -1
            for rule in signatures:
                req_pos = {_normalize_marker_symbol(s) for s in rule.get("required_positive", [])}
                req_neg = {_normalize_marker_symbol(s) for s in rule.get("required_negative", [])}
                excl_pos = {_normalize_marker_symbol(s) for s in rule.get("exclude_if_positive", [])}
                excl_if_meas_neg = {_normalize_marker_symbol(s) for s in rule.get("exclude_if_measured_negative", [])}
                if not req_pos and not req_neg:
                    continue
                # Strict guard: rule can only trigger if ALL required positive/negative antigens exist in the panel
                if not (req_pos | req_neg).issubset(measured_syms):
                    continue
                # Disqualify if any measured exclusion marker is positive
                if excl_pos & pos_syms:
                    continue
                # Disqualify if a confirmatory marker is measured in the panel but negative on this cluster
                if (excl_if_meas_neg & measured_syms) - pos_syms:
                    continue
                if req_pos.issubset(pos_syms) and req_neg.issubset(neg_syms):
                    score = int(rule.get("priority", 50)) + len(req_pos) * 5 + len(req_neg) * 3
                    if score > best_score:
                        best_score = score
                        matched_rule = rule

        if user_label:
            final_label = str(user_label)
            macro_label = str(user_label)
            source = "user_cluster_name_map"
            confidence = "1.00 (Explicit Override)"
        elif "DEAD" in pos_syms:
            final_label = "Dead / Non-Viable Events"
            macro_label = "Dead Cells / Debris"
            source = "viability_cutoff"
            confidence = "1.00 (Exclusion Dye Positive)"
        elif matched_rule is not None:
            final_label = str(matched_rule["name"])
            macro_label = str(matched_rule.get("macro_group") or final_label)
            n_constraints = len(matched_rule.get("required_positive", [])) + len(matched_rule.get("required_negative", []))
            source = f"signature_rule:{final_label}"
            confidence = f"1.00 ({n_constraints}/{n_constraints} constraints)"
        else:
            # Strict Zero-Guessing fallback: state exact empirical markers/detectors only
            hi_sorted = [m[0] for m in sorted(hi_display, key=lambda x: x[1], reverse=True)]
            if hi_sorted:
                tag_str = " ".join(f"{m}++" for m in hi_sorted[:3])
            elif pos_display:
                tag_str = " ".join(pos_display[:3])
            else:
                tag_str = "Low/Neg Across Panel"
            final_label = f"C{cid}: {tag_str}"
            macro_label = f"{tag_str} Subset" if pos_display or hi_sorted else f"Cluster {cid} Phenotype"
            source = "empirical_markers" if biological_detected else "empirical_detectors_unmapped"
            confidence = "N/A (Empirical Profile Only)"

        phenotypes[cid] = final_label
        macro_groups[cid] = macro_label
        audit_rows.append({
            "Cluster ID": cid,
            "Assigned Label": final_label,
            "Macro Group": macro_label,
            "Annotation Source": source,
            "Confidence": confidence,
            "Positive Markers": "; ".join(pos_display) if pos_display else "None",
            "Negative Markers": "; ".join(neg_display) if neg_display else "None",
        })

    if not biological_detected and len(cluster_stats_df) > 0:
        logger.warning(
            "Step 05 Annotation: No curated 'markers.csv' or verified FCS $PnS biological names detected. "
            "Clusters were labeled empirically by physical detector channels without guessing cell types."
        )

    audit_df = pd.DataFrame(audit_rows)
    return phenotypes, macro_groups, audit_df, thresholds


def group_clusters_by_expression(
    cluster_stats_df: pd.DataFrame,
    markers: List[str],
    n_macro_groups: int = 5,
    cluster_name_map: Optional[Dict[Any, str]] = None,
    events_df: Optional[pd.DataFrame] = None,
    marker_name_map: Optional[Dict[str, str]] = None,
    cell_signatures: Optional[Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, Dict[int, str], List[int]]:
    """Group clusters into biological macro-lineages and compute dendrogram leaf ordering.

    Uses transparent marker-signature resolution and empirical positivity cutoffs
    (from `annotate_clusters_transparently`) instead of hardcoded detector strings.
    """
    n_clusters = len(cluster_stats_df)
    if n_clusters == 0:
        return pd.DataFrame(), {}, []

    median_cols = [c for c in cluster_stats_df.columns if "Median" in c]
    if not median_cols:
        median_cols = [f"{m} Median" for m in markers if f"{m} Median" in cluster_stats_df.columns]
    if not median_cols:
        median_cols = [m for m in markers if m in cluster_stats_df.columns]

    _, cluster_to_macro, _, _ = annotate_clusters_transparently(
        cluster_stats_df=cluster_stats_df,
        markers=markers,
        events_df=events_df,
        marker_name_map=marker_name_map,
        cluster_name_map=cluster_name_map,
        cell_signatures=cell_signatures,
    )

    # Hierarchical agglomerative linkage ordered cleanly within lineage blocks
    lineage_order = [
        "CD4+ T-Cell Lineage",
        "CD8+ T-Cell Lineage",
        "CD4+ CD8+ DP T Cells",
        "CD3+ DN / Other T Cells",
        "B-Cell Lineage",
        "NK-Cell Lineage",
        "Monocyte / Myeloid Lineage",
        "HLA-DR+ APC / Monocyte Lineage",
        "Anti-Inflammatory M2 Macrophages",
        "Pro-Inflammatory M1 Macrophages",
        "Basal / Unpolarized Macrophages",
        "CD3- Non-T Lymphocytes",
        "Dead Cells / Debris",
    ]

    ordered_indices: List[int] = []
    # Include any custom or empirical macro group names after known order
    seen_groups = list(lineage_order)
    for g in sorted(set(cluster_to_macro.values())):
        if g not in seen_groups:
            seen_groups.append(g)

    for lin in seen_groups:
        matching_rows = cluster_stats_df[
            cluster_stats_df["Cluster ID"].map(lambda c: cluster_to_macro.get(int(c))) == lin
        ]
        if len(matching_rows) == 0:
            continue
        if len(matching_rows) > 2:
            sub_mat = matching_rows[median_cols].to_numpy(dtype=float)
            Z = linkage(sub_mat, method="ward")
            sub_leaves = leaves_list(Z)
            ordered_indices.extend(matching_rows.index[sub_leaves].tolist())
        else:
            ordered_indices.extend(matching_rows.index.tolist())

    remaining = [i for i in range(n_clusters) if i not in ordered_indices]
    ordered_indices.extend(remaining)

    macro_rows: List[Dict[str, Any]] = []
    unique_lineages = [lin for lin in seen_groups if lin in cluster_to_macro.values()]

    for gid, lineage in enumerate(unique_lineages, start=1):
        cids = [cid for cid, lin in cluster_to_macro.items() if lin == lineage]
        subset = cluster_stats_df[cluster_stats_df["Cluster ID"].isin(cids)]
        total_cells = int(subset["Count"].sum())
        total_pct = round(float(subset["% Total"].sum()), 2)
        parent_pct = round(float(subset["% Parent"].sum()), 2) if "% Parent" in subset.columns else total_pct

        row: Dict[str, Any] = {
            "Macro Group ID": f"Group {gid}",
            "Lineage Name": lineage,
            "Clusters": ", ".join(f"C{cid}" for cid in sorted(cids)),
            "Cluster Count": len(cids),
            "Total Cells": total_cells,
            "% Parent": parent_pct,
            "% Total Events": total_pct,
        }
        for col in median_cols:
            clean_name = col.replace(" Median", "")
            weighted_val = float((subset[col] * subset["Count"]).sum() / total_cells) if total_cells > 0 else 0.0
            row[f"{clean_name} Mean"] = round(weighted_val, 3)
        macro_rows.append(row)

    macro_groups_df = pd.DataFrame(macro_rows).sort_values(by="Total Cells", ascending=False).reset_index(drop=True)
    for idx in range(len(macro_groups_df)):
        macro_groups_df.loc[idx, "Macro Group ID"] = f"Group {idx + 1}"

    return macro_groups_df, cluster_to_macro, ordered_indices


def _annotate_and_group_clusters(
    stats_df: pd.DataFrame,
    valid_markers: List[str],
    df: Optional[pd.DataFrame] = None,
    cluster_name_map: Optional[Dict[Any, str]] = None,
    marker_name_map: Optional[Dict[str, str]] = None,
    cell_signatures: Optional[Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Tuple[Dict[int, str], pd.DataFrame, Dict[int, str], List[int], pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Shared transparent annotation, empirical thresholding, and macro-grouping pipeline."""
    phenotypes, _, audit_df, thresholds = annotate_clusters_transparently(
        cluster_stats_df=stats_df,
        markers=valid_markers,
        events_df=df,
        marker_name_map=marker_name_map,
        cluster_name_map=cluster_name_map,
        cell_signatures=cell_signatures,
    )
    macro_df, cluster_to_macro, ordered_idx = group_clusters_by_expression(
        cluster_stats_df=stats_df,
        markers=valid_markers,
        n_macro_groups=5,
        cluster_name_map=cluster_name_map,
        events_df=df,
        marker_name_map=marker_name_map,
        cell_signatures=cell_signatures,
    )
    return phenotypes, macro_df, cluster_to_macro, ordered_idx, audit_df, thresholds


def run_som(
    df: pd.DataFrame,
    markers: List[str],
    x_dim: int = 10,
    y_dim: int = 10,
    n_metaclusters: int = 8,
    auto_k: bool = False,
    auto_k_method: str = "consensus",
    k_min: int = 4,
    k_max: int = 20,
    consensus_subsample: float = 0.8,
    consensus_iterations: int = 50,
    random_state: int = 42,
    subsample_train: int = 50000,
    total_sample_events: Optional[int] = None,
    parent_population: str = "All Events",
    cluster_name_map: Optional[Dict[Any, str]] = None,
    marker_name_map: Optional[Dict[str, str]] = None,
    cell_signatures: Optional[Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]] = None,
) -> ClusteringResult:
    """Execute SOM algorithm: Self-Organizing Map + Consensus Metaclustering.

    Trains a 2D toroidal SOM grid (via MiniSom) on cell marker expression, then
    groups the SOM codebook nodes into higher-level metaclusters using hierarchical clustering.
    """
    valid_markers = [m for m in markers if m in df.columns]
    if len(valid_markers) < 1:
        raise ValueError(f"SOM requires at least 1 valid marker, found: {valid_markers}")

    if len(valid_markers) == 1:
        col_vec = df[valid_markers[0]].to_numpy(dtype=np.float64)
        data_matrix = np.column_stack([col_vec, col_vec])
    else:
        data_matrix = df[valid_markers].to_numpy(dtype=np.float64)
    n_events = len(data_matrix)
    n_features = data_matrix.shape[1]

    # Subsample for training SOM weights deterministically
    if n_events > subsample_train:
        rng = np.random.RandomState(random_state)
        train_idx = rng.choice(n_events, size=subsample_train, replace=False)
        train_data = data_matrix[train_idx]
    else:
        train_data = data_matrix

    # Step 1: Train Self-Organizing Map
    som = MiniSom(
        x=x_dim,
        y=y_dim,
        input_len=n_features,
        sigma=1.2,
        learning_rate=0.5,
        random_seed=random_state,
    )
    if len(train_data) > 0:
        som.random_weights_init(train_data)
    som.train_batch(train_data, num_iteration=1200)

    # Step 2: Map all events to BMU nodes first so metaclustering runs only on populated SOM nodes
    weights = som.get_weights().reshape((x_dim * y_dim, n_features))
    all_bmu_indices = np.empty(n_events, dtype=int)
    batch_size = 20000
    for i in range(0, n_events, batch_size):
        chunk = data_matrix[i : i + batch_size]
        dists = np.linalg.norm(chunk[:, np.newaxis, :] - weights[np.newaxis, :, :], axis=2)
        all_bmu_indices[i : i + batch_size] = np.argmin(dists, axis=1)

    populated_nodes = np.unique(all_bmu_indices)
    active_weights = weights[populated_nodes]

    # Step 3: Metaclustering on populated SOM codebook weights (prevents empty metaclusters)
    k_scores: Dict[int, float] = {}
    metric_name: str = ""
    if len(populated_nodes) <= 1:
        n_metaclusters = 1
        event_clusters = np.zeros(n_events, dtype=int)
    else:
        if auto_k:
            eff_k_max = min(k_max, len(populated_nodes) - 1)
            eff_k_min = min(k_min, max(2, eff_k_max))
            n_metaclusters, k_scores, metric_name = find_optimal_metaclusters(
                active_weights,
                k_min=eff_k_min,
                k_max=eff_k_max,
                method=auto_k_method,
                consensus_subsample=consensus_subsample,
                consensus_iterations=consensus_iterations,
                random_state=random_state,
            )
        n_metaclusters = min(n_metaclusters, len(populated_nodes))
        agg = AgglomerativeClustering(n_clusters=n_metaclusters, linkage="ward")
        active_metaclusters = agg.fit_predict(active_weights)
        node_metaclusters = np.zeros(len(weights), dtype=int)
        node_metaclusters[populated_nodes] = active_metaclusters
        event_clusters = node_metaclusters[all_bmu_indices]

    # Compute cluster statistics & transparent biological annotations
    stats_df = _compute_cluster_statistics(
        df=df,
        labels=event_clusters,
        markers=valid_markers,
        total_sample_events=total_sample_events,
    )
    phenotypes, macro_df, cluster_to_macro, ordered_idx, audit_df, thresholds = _annotate_and_group_clusters(
        stats_df=stats_df,
        valid_markers=valid_markers,
        df=df,
        cluster_name_map=cluster_name_map,
        marker_name_map=marker_name_map,
        cell_signatures=cell_signatures,
    )

    return ClusteringResult(
        method="som",
        cluster_labels=event_clusters,
        cluster_count=n_metaclusters,
        markers_used=valid_markers,
        cluster_statistics=stats_df,
        phenotype_labels=phenotypes,
        macro_groups_df=macro_df,
        cluster_to_macro_group=cluster_to_macro,
        ordered_indices=ordered_idx,
        parent_population=parent_population,
        parent_count=n_events,
        total_sample_count=total_sample_events or n_events,
        k_optimization_scores=k_scores,
        k_optimization_metric=metric_name,
        annotation_audit_df=audit_df,
        marker_name_map=dict(marker_name_map or {}),
        marker_thresholds=thresholds,
    )


def run_phenograph(
    df: pd.DataFrame,
    markers: List[str],
    k: int = 30,
    subsample: int = 25000,
    random_state: int = 42,
    total_sample_events: Optional[int] = None,
    parent_population: str = "All Events",
    cluster_name_map: Optional[Dict[Any, str]] = None,
    marker_name_map: Optional[Dict[str, str]] = None,
    cell_signatures: Optional[Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]] = None,
) -> ClusteringResult:
    """Execute PhenoGraph algorithm: k-NN graph construction + Louvain community detection."""
    try:
        import phenograph
    except ImportError as exc:
        raise ImportError(
            "PhenoGraph is an optional unsupervised clustering engine and is not installed.\n"
            "To enable PhenoGraph, install it separately using:\n"
            "    pip install PhenoGraph\n"
            "(Note: PhenoGraph uses community detection libraries licensed under GPL-3.0/GPL-2.0)."
        ) from exc

    valid_markers = [m for m in markers if m in df.columns]
    if len(valid_markers) < 2:
        raise ValueError(f"PhenoGraph requires at least 2 valid markers, found: {valid_markers}")

    data_matrix = df[valid_markers].to_numpy(dtype=np.float64)
    n_events = len(data_matrix)

    if n_events > subsample:
        rng = np.random.RandomState(random_state)
        sub_idx = rng.choice(n_events, size=subsample, replace=False)
        sub_data = data_matrix[sub_idx]

        sub_labels, _, _ = phenograph.cluster(sub_data, k=k, seed=random_state)

        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn.fit(sub_data, sub_labels)
        all_labels = knn.predict(data_matrix)
    else:
        all_labels, _, _ = phenograph.cluster(data_matrix, k=k)

    if np.any(all_labels < 0):
        all_labels[all_labels < 0] = 0

    unique_clusters = np.unique(all_labels)
    mapping = {old: new for new, old in enumerate(unique_clusters)}
    final_labels = np.array([mapping[l] for l in all_labels], dtype=int)
    cluster_count = len(unique_clusters)

    stats_df = _compute_cluster_statistics(
        df=df,
        labels=final_labels,
        markers=valid_markers,
        total_sample_events=total_sample_events,
    )
    phenotypes, macro_df, cluster_to_macro, ordered_idx, audit_df, thresholds = _annotate_and_group_clusters(
        stats_df=stats_df,
        valid_markers=valid_markers,
        df=df,
        cluster_name_map=cluster_name_map,
        marker_name_map=marker_name_map,
        cell_signatures=cell_signatures,
    )

    return ClusteringResult(
        method="phenograph",
        cluster_labels=final_labels,
        cluster_count=cluster_count,
        markers_used=valid_markers,
        cluster_statistics=stats_df,
        phenotype_labels=phenotypes,
        macro_groups_df=macro_df,
        cluster_to_macro_group=cluster_to_macro,
        ordered_indices=ordered_idx,
        parent_population=parent_population,
        parent_count=n_events,
        total_sample_count=total_sample_events or n_events,
        annotation_audit_df=audit_df,
        marker_name_map=dict(marker_name_map or {}),
        marker_thresholds=thresholds,
    )


def _compute_cluster_statistics(
    df: pd.DataFrame,
    labels: np.ndarray,
    markers: List[str],
    total_sample_events: Optional[int] = None,
) -> pd.DataFrame:
    """Compute cell counts, percentages, and median intensities per cluster."""
    parent_events = len(labels)
    sample_total = total_sample_events if total_sample_events is not None else parent_events
    unique_clusters = np.unique(labels)
    rows: List[Dict[str, Any]] = []

    for cid in unique_clusters:
        if cid < 0:
            continue
        mask = labels == cid
        count = int(np.sum(mask))
        pct_parent = (count / parent_events * 100.0) if parent_events > 0 else 0.0
        pct_total = (count / sample_total * 100.0) if sample_total > 0 else 0.0

        sub_df = df.loc[mask]
        row: Dict[str, Any] = {
            "Cluster ID": int(cid),
            "Count": count,
            "% Parent": round(pct_parent, 2),
            "% Total": round(pct_total, 2),
        }

        for m in markers:
            if m in sub_df.columns and count > 0:
                row[f"{m} Median"] = round(float(sub_df[m].median()), 3)
            else:
                row[f"{m} Median"] = 0.0

        rows.append(row)

    res_df = pd.DataFrame(rows)
    return res_df.sort_values(by="Count", ascending=False).reset_index(drop=True)


def _infer_phenotypes(
    stats_df: pd.DataFrame,
    markers: List[str],
    marker_name_map: Optional[Dict[str, str]] = None,
    events_df: Optional[pd.DataFrame] = None,
    cluster_name_map: Optional[Dict[Any, str]] = None,
    cell_signatures: Optional[Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[int, str]:
    """Infer biological labels transparently using empirical cutoffs and marker signatures."""
    phenotypes, _, _, _ = annotate_clusters_transparently(
        cluster_stats_df=stats_df,
        markers=markers,
        events_df=events_df,
        marker_name_map=marker_name_map,
        cluster_name_map=cluster_name_map,
        cell_signatures=cell_signatures,
    )
    return phenotypes
