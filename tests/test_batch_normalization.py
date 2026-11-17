"""Unit and integration tests for native CytoNorm and anchor-free quantile normalization."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from flowx.batch.models import BatchCorrectionConfig, CohortSampleInfo
from flowx.batch.normalization import (
    CytoNormModel,
    QuantileNormalizer,
    QuantileSplineMapping,
    calculate_batch_invariance_metrics,
    fit_anchor_free_quantile_normalizer,
    fit_cytonorm,
)
from flowx.io.parser import parse_fcs


def test_quantile_spline_mapping_linear_and_boundary():
    """Verify monotonic spline mapping and slope extrapolation at boundaries."""
    # Maps [0, 5, 10] to [10, 20, 30] (slope 2.0)
    mapping = QuantileSplineMapping(
        cluster_id=0,
        marker="CD4",
        batch_id="Batch1",
        input_quantiles=[0.0, 5.0, 10.0],
        goal_quantiles=[10.0, 20.0, 30.0],
    )

    x = np.array([-2.0, 0.0, 5.0, 10.0, 15.0])
    y = mapping.apply(x)

    # Within knots
    assert np.isclose(y[1], 10.0)
    assert np.isclose(y[2], 20.0)
    assert np.isclose(y[3], 30.0)

    # Left extrapolation: 10 + 2.0 * (-2 - 0) = 6.0
    assert np.isclose(y[0], 6.0)

    # Right extrapolation: 30 + 2.0 * (15 - 10) = 40.0
    assert np.isclose(y[4], 40.0)


def test_cytonorm_model_save_load(tmp_path: Path):
    """Verify CytoNormModel JSON serialization and deserialization."""
    mapping = QuantileSplineMapping(
        cluster_id=0,
        marker="CD4",
        batch_id="Batch1",
        input_quantiles=[1.0, 2.0, 3.0],
        goal_quantiles=[2.0, 4.0, 6.0],
    )
    model = CytoNormModel(
        markers=["CD4"],
        batches=["Batch1"],
        n_clusters=1,
        centroids=np.array([[2.0]]),
        splines={(0, "CD4", "Batch1"): mapping},
        goal="median",
    )

    dest = tmp_path / "cytonorm_model.json"
    model.save_json(dest)
    assert dest.exists()

    loaded = CytoNormModel.load_json(dest)
    assert loaded.markers == ["CD4"]
    assert loaded.batches == ["Batch1"]
    assert loaded.n_clusters == 1
    assert (0, "CD4", "Batch1") in loaded.splines

    # Test transform
    df = pd.DataFrame({"CD4": [1.0, 2.0, 3.0]})
    res = loaded.transform(df, "Batch1")
    np.testing.assert_allclose(res["CD4"], [2.0, 4.0, 6.0])


def test_fit_cytonorm_on_benchmark_data(synthetic_cytonorm_cohort_dir: Path):
    """Test fitting CytoNorm on benchmark cohort and verify EMD reduction."""
    files = sorted(synthetic_cytonorm_cohort_dir.glob("*.fcs"))
    assert len(files) == 6, "Expected 6 benchmark files"

    cohort_dfs = {}
    sample_info = {}

    for f in files:
        fcs = parse_fcs(f)
        s_id = f.stem
        # Extract batch ID (e.g. PTLG021)
        batch = next(b for b in ["PTLG021", "PTLG028", "PTLG034"] if b in s_id)
        is_ctrl = "Control_1" in s_id

        # Arcsinh transform CyTOF channels (cofactor 5)
        markers = [
            c for c in fcs.pnn_labels
            if not any(c.lower().startswith(x) for x in ["time", "event", "pd102", "pd104", "pd105", "pd106", "pd108"])
        ]
        df_t = fcs.events_df.copy()
        df_t[markers] = np.arcsinh(df_t[markers].to_numpy() / 5.0)

        cohort_dfs[s_id] = df_t
        sample_info[s_id] = CohortSampleInfo(
            sample_id=s_id,
            batch_id=batch,
            is_control=is_ctrl,
            raw_event_count=fcs.event_count,
            clean_event_count=fcs.event_count,
        )

    config = BatchCorrectionConfig(
        enabled=True,
        method="cytonorm",
        goal="median",
        markers=markers,
        n_clusters=8,
        n_quantiles=50,
    )

    model = fit_cytonorm(cohort_dfs, sample_info, config)
    assert model.n_clusters <= 8
    assert len(model.batches) == 3

    # Normalize control samples
    norm_cohort = {}
    for s_id, df in cohort_dfs.items():
        norm_cohort[s_id] = model.transform(df, sample_info[s_id].batch_id)

    # Compute batch invariance metrics on controls
    metrics_df = calculate_batch_invariance_metrics(
        raw_cohort=cohort_dfs,
        normalized_cohort=norm_cohort,
        sample_info=sample_info,
        markers=model.markers,
        controls_only=True,
    )

    assert not metrics_df.empty
    mean_reduction = metrics_df["emd_reduction_pct"].mean()
    positive_fraction = (metrics_df["emd_reduction_pct"] > 0).mean()
    # Confirm significant technical variation reduction (> 25% mean reduction and >70% markers reduced)
    assert mean_reduction > 25.0, f"Expected >25% EMD reduction, got {mean_reduction:.2f}%"
    assert positive_fraction >= 0.70, f"Expected >=70% markers improved, got {positive_fraction * 100:.1f}%"


def test_anchor_free_quantile_normalizer():
    """Verify Anchor-Free Quantile Normalizer aligns synthetic batch distributions."""
    rng = np.random.default_rng(42)
    # Batch 1 centered at 10.0, Batch 2 shifted to 15.0 (technical shift)
    b1_df = pd.DataFrame({"Marker1": rng.normal(10.0, 2.0, 3000)})
    b2_df = pd.DataFrame({"Marker1": rng.normal(15.0, 2.0, 3000)})

    cohort = {"Sample_B1": b1_df, "Sample_B2": b2_df}
    sample_info = {
        "Sample_B1": CohortSampleInfo(sample_id="Sample_B1", batch_id="B1", is_control=False),
        "Sample_B2": CohortSampleInfo(sample_id="Sample_B2", batch_id="B2", is_control=False),
    }

    norm = fit_anchor_free_quantile_normalizer(
        cohort, sample_info, BatchCorrectionConfig(method="quantile", goal="median")
    )

    b1_norm = norm.transform(b1_df, "B1")
    b2_norm = norm.transform(b2_df, "B2")

    med1 = np.median(b1_norm["Marker1"])
    med2 = np.median(b2_norm["Marker1"])

    # Both should be aligned to ~12.5 (median of 10.0 and 15.0)
    assert np.isclose(med1, 12.5, atol=0.2)
    assert np.isclose(med2, 12.5, atol=0.2)
    assert np.isclose(med1, med2, atol=0.1)


def test_fit_cytonorm_missing_controls_raises():
    """Assert ValueError when CytoNorm is attempted without designated control samples."""
    cohort = {"S1": pd.DataFrame({"M": [1.0, 2.0]})}
    sample_info = {"S1": CohortSampleInfo(sample_id="S1", batch_id="B1", is_control=False)}

    with pytest.raises(ValueError, match="CytoNorm requires at least one control anchor"):
        fit_cytonorm(cohort, sample_info, BatchCorrectionConfig())
