"""Unit tests for automated FMO threshold calibration engine."""

from pathlib import Path

from flowx.gating.fmo import (
    FMOCalibrationConfig,
    calibrate_fmo_thresholds,
    match_fmo_file_to_marker,
)


def test_match_fmo_file_to_marker():
    """Verify filename parser maps various FMO naming patterns to target markers."""
    markers = ["CD80", "CD206", "CD11b", "CD33", "IFNAR1"]
    channels = ["PE-CF594-A", "BV605-A", "HBB515-A", "PE-A", "PE-A"]

    m1, _ = match_fmo_file_to_marker("Specimen_001_01gx FMO80.fcs", markers, channels)
    assert m1 == "CD80"

    m2, _ = match_fmo_file_to_marker("FMO206_rep1.fcs", markers, channels)
    assert m2 == "CD206"

    m3, _ = match_fmo_file_to_marker("FMO_CD11b.fcs", markers, channels)
    assert m3 == "CD11b"

    m4, _ = match_fmo_file_to_marker("Unstained_Control.fcs", markers, channels)
    assert m4 is None


def test_fmo_calibration_on_synthetic_suite(synthetic_fmo_suite: Path):
    """Verify FMO calibration executes on FMO controls and computes empirical 99th-percentile cutoffs."""
    fmo_files = sorted(synthetic_fmo_suite.glob("*FMO*.fcs"))
    assert len(fmo_files) == 4

    marker_map = {
        "CD80": "PE-CF594-A",
        "CD206": "BV605-A",
        "CD33": "PE-A",
        "CD11b": "Horizon Brilliant Blue 515-A",
    }

    res = calibrate_fmo_thresholds(
        fmo_files=fmo_files,
        marker_channel_map=marker_map,
        config=FMOCalibrationConfig(percentile=99.0),
    )

    assert len(res.records) >= 3
    assert "CD80" in res.marker_cutoffs
    assert "CD206" in res.marker_cutoffs

    cd80_cut = res.get_cutoff("CD80")
    cd206_cut = res.get_cutoff("CD206")

    # Cutoffs should be positive numbers reflecting empirical 99th percentile
    assert 0.35 <= cd80_cut <= 0.95
    assert 0.35 <= cd206_cut <= 0.95

    # Check summary dataframe
    df = res.summary_df
    assert len(df) == len(res.records)
    assert "cutoff" in df.columns
    assert "p99" in df.columns

    # Test building positivity gates
    h = res.build_positivity_gates(parent_gate="Viable")
    assert "CD80+" in h.order
    assert "CD206+" in h.order
