"""Unit tests for UMAP, t-SNE, and high-dimensional single-cell visualization."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flowx.clustering.reduction import compute_umap_embedding, compute_tsne_embedding
from flowx.visualization.plotting import (
    plot_umap_projection,
    plot_umap_marker_features,
    plot_population_marker_violins,
)


@pytest.fixture
def mock_flow_df():
    """Create synthetic multi-parametric flow cytometry DataFrame."""
    rng = np.random.RandomState(42)
    n = 1000
    markers = [f"Marker_{i}" for i in range(10)]
    data = rng.randn(n, len(markers))
    return pd.DataFrame(data, columns=markers)


def test_compute_umap_embedding(mock_flow_df):
    """Verify UMAP embedding dimensionality and subsampling."""
    markers = list(mock_flow_df.columns)
    emb, sampled_idx = compute_umap_embedding(
        df=mock_flow_df,
        markers=markers,
        n_components=2,
        random_state=42,
        subsample=200,
    )
    assert emb.shape == (200, 2)
    assert len(sampled_idx) == 200
    assert np.all(np.isfinite(emb))


def test_compute_tsne_embedding(mock_flow_df):
    """Verify t-SNE embedding dimensionality and subsampling."""
    markers = list(mock_flow_df.columns)
    emb, sampled_idx = compute_tsne_embedding(
        df=mock_flow_df,
        markers=markers,
        n_components=2,
        random_state=42,
        subsample=150,
    )
    assert emb.shape == (150, 2)
    assert len(sampled_idx) == 150
    assert np.all(np.isfinite(emb))


def test_plot_umap_projection(tmp_path, mock_flow_df):
    """Verify UMAP 2D projection plotting with centroid and external legend modes."""
    markers = list(mock_flow_df.columns)
    emb, sampled_idx = compute_umap_embedding(
        df=mock_flow_df,
        markers=markers,
        subsample=100,
        random_state=42,
    )
    labels = np.array([f"Pop_{i % 4}" for i in range(len(emb))])

    out1 = tmp_path / "umap_centroid.png"
    fig1, ax1 = plot_umap_projection(
        embedding=emb,
        labels=labels,
        title="Test UMAP Centroid",
        legend_mode="centroid",
        output_path=out1,
    )
    assert out1.is_file()
    assert out1.stat().st_size > 1000

    out2 = tmp_path / "umap_external.png"
    fig2, ax2 = plot_umap_projection(
        embedding=emb,
        labels=labels,
        title="Test UMAP External",
        legend_mode="external",
        output_path=out2,
    )
    assert out2.is_file()
    assert out2.stat().st_size > 1000


def test_plot_umap_marker_features(tmp_path, mock_flow_df):
    """Verify multi-panel continuous marker feature heatmaps on UMAP."""
    markers = list(mock_flow_df.columns)[:6]
    emb, sampled_idx = compute_umap_embedding(
        df=mock_flow_df,
        markers=markers,
        subsample=100,
        random_state=42,
    )
    features_df = mock_flow_df.iloc[sampled_idx].reset_index(drop=True)

    out = tmp_path / "umap_features.png"
    fig = plot_umap_marker_features(
        embedding=emb,
        df_features=features_df,
        markers=markers,
        output_path=out,
        max_markers=6,
    )
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_plot_population_marker_violins(tmp_path, mock_flow_df):
    """Verify population marker distribution violins."""
    markers = list(mock_flow_df.columns)[:4]
    labels = np.array([f"Lineage_{i % 3}" for i in range(len(mock_flow_df))])

    out = tmp_path / "violins.png"
    fig = plot_population_marker_violins(
        df_events=mock_flow_df,
        population_labels=labels,
        markers=markers,
        output_path=out,
        max_markers=4,
    )
    assert out.is_file()
    assert out.stat().st_size > 1000
