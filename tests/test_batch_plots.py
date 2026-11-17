"""Unit tests for batch diagnostics and visualization."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from flowx.batch.models import CohortSampleInfo
from flowx.visualization.batch_plots import (
    plot_batch_density_comparison,
    plot_batch_metrics_summary,
    plot_cohort_downsampling_balance,
)


@pytest.fixture
def mock_cohort_data():
    """Generate synthetic multi-batch cohort DataFrames."""
    rng = np.random.default_rng(42)
    s1 = pd.DataFrame({"CD4": rng.normal(10, 2, 500), "CD8": rng.normal(8, 1.5, 500)})
    s2 = pd.DataFrame({"CD4": rng.normal(12, 2, 500), "CD8": rng.normal(9, 1.5, 500)})

    raw = {"S1": s1, "S2": s2}
    norm = {"S1": s1, "S2": s1}  # S2 normalized to S1 distribution

    info = {
        "S1": CohortSampleInfo(
            sample_id="S1", batch_id="B1", is_control=True, raw_event_count=1000, clean_event_count=800, prepared_event_count=500
        ),
        "S2": CohortSampleInfo(
            sample_id="S2", batch_id="B2", is_control=True, raw_event_count=1500, clean_event_count=1200, prepared_event_count=500
        ),
    }
    return raw, norm, info


def test_plot_batch_density_comparison(mock_cohort_data, tmp_path: Path):
    """Test generating multi-batch density comparison figure."""
    raw, norm, info = mock_cohort_data
    out_path = tmp_path / "density_test.png"

    fig = plot_batch_density_comparison(
        raw_cohort=raw,
        normalized_cohort=norm,
        sample_info=info,
        markers=["CD4", "CD8"],
        output_path=out_path,
        controls_only=True,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    plt.close(fig)


def test_plot_batch_metrics_summary(tmp_path: Path):
    """Test generating EMD reduction summary bar chart."""
    metrics_df = pd.DataFrame([
        {"marker": "CD4", "emd_reduction_pct": 45.2},
        {"marker": "CD8", "emd_reduction_pct": 32.8},
        {"marker": "CD3", "emd_reduction_pct": -5.0},
    ])
    out_path = tmp_path / "metrics_test.png"

    fig = plot_batch_metrics_summary(metrics_df, output_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    plt.close(fig)


def test_plot_cohort_downsampling_balance(mock_cohort_data, tmp_path: Path):
    """Test generating cohort event yield balance chart."""
    _, _, info = mock_cohort_data
    out_path = tmp_path / "downsample_balance_test.png"

    fig = plot_cohort_downsampling_balance(list(info.values()), output_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    plt.close(fig)

