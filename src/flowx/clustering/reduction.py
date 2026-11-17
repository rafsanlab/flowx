"""Dimensionality reduction module for high-dimensional flow cytometry.

Provides non-linear manifold embeddings (UMAP, t-SNE) for projecting
high-dimensional fluorescence and mass cytometry parameter spaces into
intuitive 2D coordinate embeddings for discovery and phenotypic visualization.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


logger = logging.getLogger("flowx.clustering.reduction")


def compute_umap_embedding(
    df: pd.DataFrame,
    markers: List[str],
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.2,
    metric: str = "euclidean",
    random_state: int = 42,
    subsample: Optional[int] = 25000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute UMAP 2D manifold embedding across high-dimensional flow cytometry markers.

    Projects high-dimensional single-cell events onto a non-linear low-dimensional
    manifold preserving both local cluster structures and global lineage relationships.

    Args:
        df: Input DataFrame containing compensated and transformed single-cell events.
        markers: List of channel detector names to include in the manifold embedding.
        n_components: Target dimensionality (default: 2 for 2D visualization).
        n_neighbors: UMAP neighborhood size balancing local vs global structure (default: 15).
        min_dist: Effective minimum distance between points in the embedding (default: 0.2).
        metric: Metric for computing high-dimensional distance (default: 'euclidean').
        random_state: Random seed for reproducible manifold projection.
        subsample: Optional maximum number of events to sample. For large FCS files (>100k events),
                   subsampling 20,000–30,000 events preserves all major and rare phenotypes while
                   executing in seconds. Set to None to embed all events.

    Returns:
        Tuple of:
            - embedding: 2D NumPy array of shape (N_sampled, n_components) with UMAP coordinates.
            - sampled_indices: 1D NumPy array of integer indices in `df` corresponding to the sampled events.

    Raises:
        ImportError: If the `umap-learn` package is not installed.
        ValueError: If fewer than 2 valid markers are provided or data contains insufficient events.
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP dimensionality reduction requires the 'umap-learn' package.\n"
            "Please install it using: pip install umap-learn"
        ) from exc

    valid_markers = [m for m in markers if m in df.columns]
    if len(valid_markers) < 1:
        raise ValueError(
            f"UMAP requires at least 1 valid marker column, but found {len(valid_markers)}: {valid_markers}"
        )

    n_events = len(df)
    if n_events == 0:
        raise ValueError("Cannot compute UMAP on an empty DataFrame.")

    # Determine subset of events to project
    if subsample is not None and n_events > subsample:
        rng = np.random.RandomState(random_state)
        sampled_indices = np.sort(rng.choice(n_events, size=subsample, replace=False))
        data_matrix = df.iloc[sampled_indices][valid_markers].to_numpy(dtype=np.float64)
    else:
        sampled_indices = np.arange(n_events)
        data_matrix = df[valid_markers].to_numpy(dtype=np.float64)

    if data_matrix.shape[1] == 1:
        rng_1d = np.random.RandomState(random_state)
        jitter = rng_1d.normal(0.0, 1e-3, size=(len(data_matrix), 1))
        data_matrix = np.hstack([data_matrix, data_matrix + jitter])

    # Impute or clean non-finite values if any
    if not np.all(np.isfinite(data_matrix)):
        data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

    logger.info(
        f"Computing UMAP embedding on {len(data_matrix):,} events across {len(valid_markers)} markers "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist}, seed={random_state})..."
    )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        n_jobs=-1 if random_state is None else 1,
    )
    embedding = reducer.fit_transform(data_matrix)

    return embedding, sampled_indices


def compute_tsne_embedding(
    df: pd.DataFrame,
    markers: List[str],
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
    subsample: Optional[int] = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute t-SNE 2D manifold embedding across high-dimensional flow cytometry markers.

    Provides complementary t-Distributed Stochastic Neighbor Embedding for detailed
    local neighborhood structure preservation.

    Args:
        df: Input DataFrame with transformed single-cell events.
        markers: List of channel detector names.
        n_components: Target dimensionality (default: 2).
        perplexity: Effective number of nearest neighbors (default: 30.0).
        random_state: Random seed for reproducibility.
        subsample: Maximum number of events to sample (default: 10,000 for responsive compute).

    Returns:
        Tuple of (embedding 2D array, sampled indices array).
    """
    from sklearn.manifold import TSNE

    valid_markers = [m for m in markers if m in df.columns]
    if len(valid_markers) < 2:
        raise ValueError(f"t-SNE requires at least 2 valid markers, found: {valid_markers}")

    n_events = len(df)
    if n_events == 0:
        raise ValueError("Cannot compute t-SNE on an empty DataFrame.")

    if subsample is not None and n_events > subsample:
        rng = np.random.RandomState(random_state)
        sampled_indices = np.sort(rng.choice(n_events, size=subsample, replace=False))
        data_matrix = df.iloc[sampled_indices][valid_markers].to_numpy(dtype=np.float64)
    else:
        sampled_indices = np.arange(n_events)
        data_matrix = df[valid_markers].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(data_matrix)):
        data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, max(5.0, len(data_matrix) / 4.0)),
        random_state=random_state,
        n_jobs=-1,
    )
    embedding = tsne.fit_transform(data_matrix)

    return embedding, sampled_indices
