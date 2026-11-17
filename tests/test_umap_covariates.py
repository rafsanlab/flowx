"""Unit tests for multi-panel UMAP covariate plotting."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from flowx.visualization.plotting import format_multiline_legend_label, plot_cohort_umap_covariates


@pytest.fixture
def mock_umap_data():
    """Create synthetic UMAP coordinates, sample IDs, and cluster labels."""
    np.random.seed(42)
    n_events = 200
    embedding = np.random.randn(n_events, 2)
    sample_ids = ["Sample_A"] * 100 + ["Sample_B"] * 100
    labels = ["CD4+ T Cells"] * 80 + ["CD8+ T Cells"] * 70 + ["Monocytes"] * 50
    return embedding, sample_ids, labels


@pytest.fixture
def mock_sample_metadata():
    """Create sample metadata DataFrame."""
    return pd.DataFrame({
        "sample_id": ["Sample_A", "Sample_B"],
        "condition": ["Control", "Treated"],
        "batch_id": ["Batch_1", "Batch_2"],
        "donor": ["D1", "D2"],
    }).set_index("sample_id")


def test_plot_cohort_umap_covariates_no_metadata(mock_umap_data, tmp_path):
    """Test generating UMAP covariate plot without metadata (defaults to 2 panels)."""
    embedding, sample_ids, labels = mock_umap_data
    out_path = tmp_path / "test_umap_covariates.png"

    fig = plot_cohort_umap_covariates(
        embedding=embedding,
        sample_ids=sample_ids,
        lineage_or_cluster_labels=labels,
        sample_metadata=None,
        output_path=out_path,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    # Exactly 2 panels: Populations and Sample Origin
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_cohort_umap_covariates_undeclared_metadata_defaults_to_two(mock_umap_data, mock_sample_metadata, tmp_path):
    """Verify that when metadata is present but NO condition/batch/covariates are declared, strictly 2 default subplots are rendered."""
    embedding, sample_ids, labels = mock_umap_data
    out_path = tmp_path / "test_umap_undeclared.png"

    fig = plot_cohort_umap_covariates(
        embedding=embedding,
        sample_ids=sample_ids,
        lineage_or_cluster_labels=labels,
        sample_metadata=mock_sample_metadata,
        condition_column=None,
        batch_column=None,
        covariate_columns=None,
        output_path=out_path,
    )

    assert out_path.exists()
    # Must strictly be 2 panels (Populations and Sample Origin) - no silent guessing
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_cohort_umap_covariates_declared_condition(mock_umap_data, mock_sample_metadata, tmp_path):
    """Verify condition is rendered when explicitly declared."""
    embedding, sample_ids, labels = mock_umap_data
    out_path = tmp_path / "test_umap_cond.png"

    fig = plot_cohort_umap_covariates(
        embedding=embedding,
        sample_ids=sample_ids,
        lineage_or_cluster_labels=labels,
        sample_metadata=mock_sample_metadata,
        condition_column="condition",
        output_path=out_path,
    )

    assert out_path.exists()
    # 3 panels: Populations, Sample Origin, Condition
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_cohort_umap_covariates_with_declared_metadata(mock_umap_data, mock_sample_metadata, tmp_path):
    """Test generating UMAP covariate plot with explicitly declared condition and batch_id (4 panels)."""
    embedding, sample_ids, labels = mock_umap_data
    out_path = tmp_path / "test_umap_covariates_meta.png"

    fig = plot_cohort_umap_covariates(
        embedding=embedding,
        sample_ids=sample_ids,
        lineage_or_cluster_labels=labels,
        sample_metadata=mock_sample_metadata,
        condition_column="condition",
        batch_column="batch_id",
        output_path=out_path,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    # 4 panels: Populations, Sample Origin, Condition, Batch ID
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_cohort_umap_covariates_custom_columns(mock_umap_data, mock_sample_metadata, tmp_path):
    """Test specifying custom covariate columns (e.g. donor only)."""
    embedding, sample_ids, labels = mock_umap_data
    out_path = tmp_path / "test_umap_covariates_custom.png"

    fig = plot_cohort_umap_covariates(
        embedding=embedding,
        sample_ids=sample_ids,
        lineage_or_cluster_labels=labels,
        sample_metadata=mock_sample_metadata,
        covariate_columns=["donor"],
        output_path=out_path,
    )

    assert out_path.exists()
    # 3 panels: Populations, Sample Origin, Donor
    assert len(fig.axes) == 3
    plt.close(fig)


def test_multiline_legend_wrapping_and_no_overlap(mock_umap_data, mock_sample_metadata, tmp_path):
    """Verify that long legend labels wrap into multiline text and do not collide with neighboring subplots."""
    embedding, _, _ = mock_umap_data
    long_pop = ["Anti-Inflammatory M2 Macrophages"] * 100 + ["Pro-Inflammatory M1 Macrophages"] * 100
    long_samp = ["Specimen_001_01gx 11b"] * 100 + ["Specimen_001_01gx 206"] * 100

    wrapped = format_multiline_legend_label("Anti-Inflammatory M2 Macrophages", n_count=8516, max_chars_per_line=18)
    assert "\n" in wrapped
    assert "(n=8,516)" in wrapped

    out_path = tmp_path / "test_overlap.png"
    fig = plot_cohort_umap_covariates(
        embedding=embedding,
        sample_ids=long_samp,
        lineage_or_cluster_labels=long_pop,
        sample_metadata=mock_sample_metadata,
        condition_column="condition",
        batch_column="batch_id",
        output_path=out_path,
    )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for row in range(2):
        leg = fig.axes[row * 2].get_legend()
        next_ax = fig.axes[row * 2 + 1]
        leg_bb = leg.get_window_extent(r)
        next_ax_bb = next_ax.get_window_extent(r)
        # Verify no horizontal collision
        assert leg_bb.x1 < next_ax_bb.x0, f"Legend in row {row} collides with next subplot: {leg_bb.x1} >= {next_ax_bb.x0}"

    plt.close(fig)


def test_plot_cohort_umap_covariates_empty_embedding(tmp_path):
    """Test handling of empty embedding gracefully."""
    out_path = tmp_path / "test_empty.png"
    fig = plot_cohort_umap_covariates(
        embedding=np.empty((0, 2)),
        sample_ids=[],
        lineage_or_cluster_labels=[],
        output_path=out_path,
    )
    assert out_path.exists()
    plt.close(fig)
