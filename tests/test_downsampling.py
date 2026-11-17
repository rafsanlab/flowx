"""Unit and integration tests for deterministic downsampling and cohort balancing."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from flowx.batch.downsampling import (
    downsample_cohort,
    downsample_dataframe,
    export_downsampled_fcs,
)
from flowx.batch.models import DownsampleConfig
from flowx.io.parser import parse_fcs


@pytest.fixture
def synthetic_events_df() -> pd.DataFrame:
    """Generate a reproducible synthetic flow cytometry event DataFrame."""
    rng = np.random.default_rng(123)
    n = 10000
    df = pd.DataFrame({
        "FSC-A": rng.normal(100000, 20000, n),
        "SSC-A": rng.normal(50000, 15000, n),
        "CD4": rng.exponential(1.5, n),
        "CD8": rng.exponential(1.2, n),
        "gate_label": rng.choice(["T_cells", "B_cells", "Monocytes"], n, p=[0.6, 0.25, 0.15]),
    })
    return df


def test_downsample_config_validation():
    """Verify validation of DownsampleConfig parameters."""
    cfg = DownsampleConfig(mode="target_count", target_events=5000)
    assert cfg.target_events == 5000

    with pytest.raises(ValueError, match="Invalid downsample mode"):
        DownsampleConfig(mode="invalid_mode")  # type: ignore

    with pytest.raises(ValueError, match="target_events must be positive"):
        DownsampleConfig(target_events=0)

    with pytest.raises(ValueError, match="fraction must be in range"):
        DownsampleConfig(fraction=1.5)


def test_downsample_target_count(synthetic_events_df: pd.DataFrame):
    """Test standard target count downsampling and index preservation."""
    cfg = DownsampleConfig(enabled=True, mode="target_count", target_events=2000, random_seed=42)
    sampled = downsample_dataframe(synthetic_events_df, cfg)

    assert len(sampled) == 2000
    assert "original_event_index" in sampled.columns
    # Check that original_event_index accurately references synthetic_events_df
    idx = sampled["original_event_index"].iloc[0]
    assert np.isclose(sampled["FSC-A"].iloc[0], synthetic_events_df["FSC-A"].iloc[idx])


def test_downsample_reproducibility(synthetic_events_df: pd.DataFrame):
    """Ensure identical random seeds yield byte-for-byte identical event samples."""
    cfg1 = DownsampleConfig(enabled=True, mode="target_count", target_events=1500, random_seed=999)
    cfg2 = DownsampleConfig(enabled=True, mode="target_count", target_events=1500, random_seed=999)

    s1 = downsample_dataframe(synthetic_events_df, cfg1)
    s2 = downsample_dataframe(synthetic_events_df, cfg2)

    pd.testing.assert_frame_equal(s1, s2)


def test_downsample_undersized_policies(synthetic_events_df: pd.DataFrame):
    """Test behavior when sample contains fewer events than target quota."""
    small_df = synthetic_events_df.iloc[:500].copy()

    # Policy 1: 'cap' (keep all)
    cfg_cap = DownsampleConfig(enabled=True, target_events=2000, min_event_policy="cap")
    res_cap = downsample_dataframe(small_df, cfg_cap)
    assert len(res_cap) == 500

    # Policy 2: 'exclude' (empty)
    cfg_exc = DownsampleConfig(enabled=True, target_events=2000, min_event_policy="exclude")
    res_exc = downsample_dataframe(small_df, cfg_exc)
    assert len(res_exc) == 0

    # Policy 3: 'error' (raise exception)
    cfg_err = DownsampleConfig(enabled=True, target_events=2000, min_event_policy="error")
    with pytest.raises(ValueError, match="fewer than target quota"):
        downsample_dataframe(small_df, cfg_err)


def test_downsample_proportional(synthetic_events_df: pd.DataFrame):
    """Verify fractional proportional downsampling."""
    cfg = DownsampleConfig(enabled=True, mode="proportional", fraction=0.35, random_seed=42)
    res = downsample_dataframe(synthetic_events_df, cfg)
    assert len(res) == 3500


def test_downsample_stratified(synthetic_events_df: pd.DataFrame):
    """Verify stratified downsampling across biological populations."""
    cfg = DownsampleConfig(
        enabled=True,
        mode="stratified",
        stratified_column="gate_label",
        stratified_quota=300,
        random_seed=42,
    )
    res = downsample_dataframe(synthetic_events_df, cfg)
    counts = res["gate_label"].value_counts()
    for cat in ["T_cells", "B_cells", "Monocytes"]:
        assert counts[cat] == 300
    assert len(res) == 900


def test_downsample_cohort_min_sample():
    """Verify cohort auto-balancing downsampling to the minimum sample count and low-event floor bypass."""
    rng = np.random.default_rng(42)
    cohort = {
        "Sample_A": pd.DataFrame({"marker": rng.normal(0, 1, 5000)}),
        "Sample_B": pd.DataFrame({"marker": rng.normal(0, 1, 3000)}),
        "Sample_C": pd.DataFrame({"marker": rng.normal(0, 1, 8000)}),
    }

    cfg = DownsampleConfig(enabled=True, mode="min_sample", random_seed=42)
    sampled_cohort, summary = downsample_cohort(cohort, cfg)

    assert len(sampled_cohort) == 3
    for s_name, s_df in sampled_cohort.items():
        assert len(s_df) == 3000, f"Sample {s_name} was not balanced to min size 3000"

    assert len(summary) == 3
    assert set(summary["sample_id"]) == {"Sample_A", "Sample_B", "Sample_C"}

    # Also verify low-event (< 2,500) bottleneck protection floor
    cohort_with_tiny = {
        "Sample_A": pd.DataFrame({"marker": rng.normal(0, 1, 5000)}),
        "Sample_Low": pd.DataFrame({"marker": rng.normal(0, 1, 2000)}),
        "Sample_C": pd.DataFrame({"marker": rng.normal(0, 1, 8000)}),
    }
    sampled_tiny, _ = downsample_cohort(cohort_with_tiny, cfg)
    assert len(sampled_tiny["Sample_A"]) == 5000
    assert len(sampled_tiny["Sample_C"]) == 5000
    assert len(sampled_tiny["Sample_Low"]) == 2000


def test_export_downsampled_fcs(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Test exporting downsampled events to standard FCS and re-parsing."""
    fcs = parse_fcs(tqc_sample_fcs_path)

    cfg = DownsampleConfig(enabled=True, target_events=250, random_seed=42)
    down_df = downsample_dataframe(fcs.events_df, cfg)
    assert len(down_df) == 250

    out_file = tmp_path / "downsampled_test.fcs"
    export_downsampled_fcs(down_df, fcs_source=fcs, output_path=out_file)

    assert out_file.exists()
    reparsed = parse_fcs(out_file)
    assert reparsed.event_count == 250
    assert reparsed.channel_count == fcs.channel_count

