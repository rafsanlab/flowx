"""Unit tests for FCS parser module."""

from pathlib import Path
import numpy as np
import pytest

from flowx.io.parser import FCSData, parse_fcs


def test_parse_valid_fcs(cytotrol_fcs_path: Path):
    """Verify parsing a valid FCS file extracts events, channels, and metadata."""
    data = parse_fcs(cytotrol_fcs_path)
    assert isinstance(data, FCSData)
    assert data.event_count > 0
    assert data.channel_count == 12
    assert len(data.pnn_labels) == 12
    assert "FSC-A" in data.pnn_labels
    assert "SSC-A" in data.pnn_labels
    assert data.events_df.shape == (data.event_count, 12)


def test_channel_resolution(cytotrol_fcs_path: Path):
    """Verify channel resolving works for detectors and markers."""
    data = parse_fcs(cytotrol_fcs_path)
    # By detector
    assert data.resolve_channel_name("FSC-A") == "FSC-A"
    # By marker
    assert data.resolve_channel_name("CD3") == "V450-A"
    assert data.resolve_channel_name("CD4") == "B710-A"
    assert data.resolve_channel_name("CD8") == "R780-A"

    with pytest.raises(KeyError):
        data.resolve_channel_name("NonExistentMarker_XYZ")


def test_spillover_extraction(cytotrol_fcs_path: Path):
    """Verify spillover extraction from FCS header text."""
    data = parse_fcs(cytotrol_fcs_path)
    assert data.spillover_matrix is not None
    assert len(data.spillover_channels) == 7
    assert data.spillover_matrix.shape == (7, 7)
    # Check diagonal is 1.0 (self-spillover normalized)
    diag = np.diag(data.spillover_matrix.values)
    assert np.allclose(diag, 1.0)


def test_parse_nonexistent_file():
    """Verify robust error handling for missing files."""
    with pytest.raises(FileNotFoundError):
        parse_fcs("does_not_exist.fcs")


def test_read_only_data_guarantee(cytotrol_fcs_path: Path):
    """Verify raw events array is independent and raw file is never mutated."""
    data = parse_fcs(cytotrol_fcs_path)
    original_val = data.raw_events[0, 0]
    data.raw_events[0, 0] = 999999.0

    # Re-reading should yield original untouched value
    data2 = parse_fcs(cytotrol_fcs_path)
    assert data2.raw_events[0, 0] == original_val


def test_instrument_metadata_and_range(cytotrol_fcs_path: Path):
    """Verify instrument scale range and alias detection."""
    data = parse_fcs(cytotrol_fcs_path)
    max_range = data.detect_max_range()
    assert max_range == 262144.0
    assert len(data.channel_ranges) > 0

    # Alias matching
    assert data.find_channel_by_alias(["cd3", "cd3e"]) == "V450-A"
    assert data.find_channel_by_alias(["cd4"]) == "B710-A"
    assert data.find_channel_by_alias(["cd8"]) == "R780-A"
    assert data.find_channel_by_alias(["nonexistent_alias_xyz"]) is None


def test_custom_marker_mapping(cytotrol_fcs_path: Path):
    """Verify user-provided manual marker mapping overrides."""
    custom_map = {"CustomCD3": "V450-A", "CustomMarker": "R780-A"}
    data = parse_fcs(cytotrol_fcs_path, marker_mapping=custom_map)
    assert data.resolve_channel_name("CustomCD3") == "V450-A"
    assert data.resolve_channel_name("CustomMarker") == "R780-A"


def test_variant3_case30_guardrails(navios_suite_dir: Path):
    """Verify Variant 3 (Case30 dot notation and 20-bit range) guardrails."""
    case30_path = navios_suite_dir / "Case30_A.fcs"
    data = parse_fcs(case30_path)
    assert data.get_fsc_channel() == "FS.INT"
    assert data.get_ssc_channel() == "SS.INT"
    assert data.get_fsc_h_channel() is None
    assert data.resolve_channel_name("FSC-A") == "FS.INT"
    assert data.resolve_channel_name("SSC-A") == "SS.INT"
    assert data.resolve_channel_name("CD14") == "CD14.FITC"
    assert data.resolve_channel_name("CD45") == "CD45.KO"
    assert data.resolve_channel_name("HLA-DR") == "HLA.PB"
    assert data.detect_max_range() > 1000000.0


def test_variant1_case1_guardrails(navios_suite_dir: Path):
    """Verify Variant 1 (Navios raw with separate detectors and stain names) guardrails."""
    case1_path = navios_suite_dir / "Case1_A.fcs"
    data = parse_fcs(case1_path)
    assert data.get_fsc_channel() == "FS INT"
    assert data.get_ssc_channel() == "SS INT"
    assert data.resolve_channel_name("FSC-A") == "FS INT"
    assert data.resolve_channel_name("SSC-A") == "SS INT"
    assert data.resolve_channel_name("CD14") == "FL1 INT"
    assert data.resolve_channel_name("CD45") == "FL10 INT"
    assert data.resolve_channel_name("HLA-DR") == "FL9 INT"


def test_channel_ranges_gain_effective_pnr(navios_suite_dir: Path):
    """Verify channel ranges account for channel gain ($PnG) as effective_pnr = pnr / gain."""
    case1_path = navios_suite_dir / "Case1_A.fcs"
    data = parse_fcs(case1_path)
    # FS INT has $PnR = 1,048,576 and $PnG = 5.0 -> effective range = 209,715.2
    assert "FS INT" in data.channel_ranges
    assert data.channel_ranges["FS INT"] == pytest.approx(209715.2, rel=1e-3)

    # SS INT has $PnR = 1,048,576 and $PnG = 5.0 -> effective range = 209,715.2
    assert "SS INT" in data.channel_ranges
    assert data.channel_ranges["SS INT"] == pytest.approx(209715.2, rel=1e-3)

    # FL1 INT has $PnR = 1,048,576 and $PnG = 1.0 -> effective range = 1,048,576.0
    assert "FL1 INT" in data.channel_ranges
    assert data.channel_ranges["FL1 INT"] == pytest.approx(1048576.0, rel=1e-3)


def test_write_fcs_gain_sanitization(tmp_path: Path, navios_suite_dir: Path):
    """Verify write_fcs sanitizes $PnG tags to 1.0 and prevents double gain division upon re-reading."""
    case8_path = navios_suite_dir / "Case8_A.fcs"
    raw_data = parse_fcs(case8_path)
    raw_fs_max = raw_data.events_df["FS INT"].max()
    raw_ss_max = raw_data.events_df["SS INT"].max()

    out_file = tmp_path / "test_sanitized_case8.fcs"
    raw_data.export_fcs(out_file)

    reloaded_data = parse_fcs(out_file)
    assert np.isclose(reloaded_data.events_df["FS INT"].max(), raw_fs_max)
    assert np.isclose(reloaded_data.events_df["SS INT"].max(), raw_ss_max)
    assert reloaded_data.channel_ranges["FS INT"] == pytest.approx(raw_data.channel_ranges["FS INT"], rel=1e-3)
    assert reloaded_data.channel_ranges["SS INT"] == pytest.approx(raw_data.channel_ranges["SS INT"], rel=1e-3)
