"""Unit tests for Flow Cytometry Panel Harmonization and Channel Resolution Audit."""

from pathlib import Path
import pandas as pd
import pytest

from flowx.io.config import PipelineConfig
from flowx.io.parser import parse_fcs
from flowx.panels.harmonization import (
    CanonicalChannelSpec,
    HarmonizationResult,
    PanelHarmonizer,
    load_panel_harmonizer,
)


def test_load_panel_harmonizer_yaml(panel_harmonization_yaml_path: Path):
    """Verify loading panel harmonization specification from YAML file."""
    harmonizer = load_panel_harmonizer(panel_harmonization_yaml_path)

    assert isinstance(harmonizer, PanelHarmonizer)
    assert len(harmonizer.channels) >= 12
    assert "CD14" in harmonizer.channels
    assert "CD19" in harmonizer.channels
    assert "HLA-DR" in harmonizer.channels
    assert "FSC-A" in harmonizer.channels
    assert "SSC-A" in harmonizer.channels

    cd14_spec = harmonizer.channels["CD14"]
    assert "FL1 INT" in cd14_spec.aliases
    assert "CD14.FITC" in cd14_spec.aliases
    assert cd14_spec.required is True
    assert cd14_spec.fluorophore == "FITC"


def test_channel_resolution_audit_logging(navios_suite_dir: Path):
    """Verify resolve_channel_name tracks audit trail, caching, and DataFrame export."""
    data = parse_fcs(navios_suite_dir / "Case1_A.fcs")
    assert len(data.resolution_audit_log) == 0

    # Query multiple channels
    det_cd14 = data.resolve_channel_name("CD14")
    assert det_cd14 == "FL1 INT"
    assert "CD14" in data.channel_resolutions
    assert len(data.resolution_audit_log) == 1

    # Second query for same channel should hit cache
    det_cd14_cached = data.resolve_channel_name("CD14")
    assert det_cd14_cached == "FL1 INT"

    # Query scatter and marker with punctuation
    det_fsc = data.resolve_channel_name("FSC-A")
    det_hla = data.resolve_channel_name("HLA-DR")
    assert det_fsc == "FS INT"
    assert det_hla == "FL9 INT"

    # Verify audit DataFrame
    audit_df = data.get_resolution_audit()
    assert isinstance(audit_df, pd.DataFrame)
    assert len(audit_df) >= 3
    assert set(audit_df["query"]).issuperset({"CD14", "FSC-A", "HLA-DR"})
    assert "resolved_detector" in audit_df.columns
    assert "strategy" in audit_df.columns
    assert "timestamp" in audit_df.columns


def test_export_resolution_audit(tmp_path: Path, navios_suite_dir: Path):
    """Verify exporting resolution audit log to CSV file."""
    data = parse_fcs(navios_suite_dir / "Case1_A.fcs")
    data.resolve_channel_name("CD14")
    data.resolve_channel_name("CD34")

    out_csv = tmp_path / "resolution_audit.csv"
    exported_path = data.export_resolution_audit(out_csv)
    assert exported_path.is_file()

    read_back = pd.read_csv(exported_path)
    assert len(read_back) == 2
    assert "CD14" in read_back["query"].values
    assert "CD34" in read_back["query"].values


def test_harmonize_case1_variant1(navios_suite_dir: Path, panel_harmonization_yaml_path: Path):
    """Verify harmonizing Variant 1 (Navios raw FL1 INT..FL10 INT) to canonical markers."""
    harmonizer = load_panel_harmonizer(panel_harmonization_yaml_path)
    s1 = parse_fcs(navios_suite_dir / "Case1_A.fcs")
    res = harmonizer.harmonize_sample(s1)

    assert isinstance(res, HarmonizationResult)
    assert res.success is True
    assert res.channel_mapping["CD14"] == "FL1 INT"
    assert res.channel_mapping["CD19"] == "FL2 INT"
    assert res.channel_mapping["CD13"] == "FL3 INT"
    assert res.channel_mapping["CD33"] == "FL4 INT"
    assert res.channel_mapping["CD34"] == "FL5 INT"
    assert res.channel_mapping["CD117"] == "FL6 INT"
    assert res.channel_mapping["CD7"] == "FL7 INT"
    assert res.channel_mapping["CD16"] == "FL8 INT"
    assert res.channel_mapping["HLA-DR"] == "FL9 INT"
    assert res.channel_mapping["CD45"] == "FL10 INT"
    assert res.channel_mapping["FSC-A"] == "FS INT"
    assert res.channel_mapping["SSC-A"] == "SS INT"

    # Check harmonized DataFrame contains canonical column names
    cols = res.harmonized_df.columns
    assert "CD14" in cols
    assert "CD19" in cols
    assert "CD34" in cols
    assert "HLA-DR" in cols
    assert "CD45" in cols
    assert "FSC-A" in cols
    assert "SSC-A" in cols


def test_harmonize_case30_variant3(navios_suite_dir: Path, panel_harmonization_yaml_path: Path):
    """Verify harmonizing Variant 3 (flowCore dot notation CD14.FITC..) to canonical markers."""
    harmonizer = load_panel_harmonizer(panel_harmonization_yaml_path)
    s30 = parse_fcs(navios_suite_dir / "Case30_A.fcs")
    res = harmonizer.harmonize_sample(s30)

    assert isinstance(res, HarmonizationResult)
    assert res.success is True
    assert res.channel_mapping["CD14"] == "CD14.FITC"
    assert res.channel_mapping["CD19"] == "CD19.PE"
    assert res.channel_mapping["CD13"] == "CD13.ECD"
    assert res.channel_mapping["CD33"] == "CD33.PC5.5"
    assert res.channel_mapping["CD34"] == "CD34.PC7"
    assert res.channel_mapping["CD117"] == "CD117.APC"
    assert res.channel_mapping["CD7"] == "CD7.APC700"
    assert res.channel_mapping["CD16"] == "CD16.APC750"
    assert res.channel_mapping["HLA-DR"] == "HLA.PB"
    assert res.channel_mapping["CD45"] == "CD45.KO"
    assert res.channel_mapping["FSC-A"] == "FS.INT"
    assert res.channel_mapping["SSC-A"] == "SS.INT"

    # Check harmonized DataFrame contains canonical column names
    cols = res.harmonized_df.columns
    assert "CD14" in cols
    assert "CD19" in cols
    assert "CD34" in cols
    assert "HLA-DR" in cols
    assert "CD45" in cols


def test_cross_sample_harmonization_concordance(navios_suite_dir: Path, panel_harmonization_yaml_path: Path):
    """Verify Case1 and Case30 harmonize into identical column structures."""
    harmonizer = load_panel_harmonizer(panel_harmonization_yaml_path)
    s1 = parse_fcs(navios_suite_dir / "Case1_A.fcs")
    s30 = parse_fcs(navios_suite_dir / "Case30_A.fcs")

    res1 = harmonizer.harmonize_sample(s1)
    res30 = harmonizer.harmonize_sample(s30)

    # Core canonical markers must match exactly in both dataframes
    canonical_markers = ["FSC-A", "SSC-A", "CD14", "CD19", "CD13", "CD33", "CD34", "CD117", "CD7", "CD16", "HLA-DR", "CD45"]
    for m in canonical_markers:
        assert m in res1.harmonized_df.columns
        assert m in res30.harmonized_df.columns

    # Cohort report concordance check
    cohort_report = harmonizer.harmonize_cohort([s1, s30])
    assert cohort_report.total_samples == 2
    assert cohort_report.is_fully_concordant is True
    assert cohort_report.concordance_matrix.loc["Case1_A.fcs", "CD14"] == "FL1 INT"
    assert cohort_report.concordance_matrix.loc["Case30_A.fcs", "CD14"] == "CD14.FITC"


def test_cohort_report_export(tmp_path: Path, navios_suite_dir: Path, panel_harmonization_yaml_path: Path):
    """Verify exporting cohort concordance report to CSV matrix and Markdown."""
    harmonizer = load_panel_harmonizer(panel_harmonization_yaml_path)
    s1 = parse_fcs(navios_suite_dir / "Case1_A.fcs")
    s30 = parse_fcs(navios_suite_dir / "Case30_A.fcs")

    report = harmonizer.harmonize_cohort([s1, s30])
    csv_path, md_path = report.export(tmp_path)

    assert csv_path.is_file()
    assert md_path.is_file()
    md_text = md_path.read_text(encoding="utf-8")
    assert "Cohort Panel Harmonization Concordance Report" in md_text
    assert "Case1_A.fcs" in md_text
    assert "Case30_A.fcs" in md_text


def test_fcs_data_harmonize_convenience_method(navios_suite_dir: Path, panel_harmonization_yaml_path: Path):
    """Verify fcs_data.harmonize() and fcs_data.get_harmonized_df() methods."""
    s1 = parse_fcs(navios_suite_dir / "Case1_A.fcs")
    harm_df = s1.get_harmonized_df(panel_harmonization_yaml_path)

    assert "CD14" in harm_df.columns
    assert "CD19" in harm_df.columns
    assert "HLA-DR" in harm_df.columns
    assert "harmonized_channels" in s1.metadata


def test_missing_required_channel_policy(navios_suite_dir: Path):
    """Verify on_missing='error' raises ValueError for required missing channels."""
    incomplete_specs = {
        "NonExistentAntigen_XYZ": CanonicalChannelSpec(
            canonical_name="NonExistentAntigen_XYZ",
            aliases=["XYZ-A", "FL99-A"],
            required=True,
        )
    }
    strict_harmonizer = PanelHarmonizer(incomplete_specs, on_missing="error")

    s1 = parse_fcs(navios_suite_dir / "Case1_A.fcs")
    with pytest.raises(ValueError, match="required canonical channels missing"):
        strict_harmonizer.harmonize_sample(s1)

    # With on_missing='warn', it should not raise but record missing
    lenient_harmonizer = PanelHarmonizer(incomplete_specs, on_missing="warn")
    res = lenient_harmonizer.harmonize_sample(s1)
    assert res.success is False
    assert "NonExistentAntigen_XYZ" in res.missing_channels


def test_pipeline_config_harmonization_parsing():
    """Verify PipelineConfig parses harmonization_file correctly."""
    data = {
        "fcs_file": "sample.fcs",
        "harmonization": "panel_harmonization.yaml",
    }
    cfg = PipelineConfig.from_dict(data)
    assert cfg.harmonization_file == "panel_harmonization.yaml"
    assert cfg.enable_harmonization is True

    # Nested input format
    nested_data = {
        "input": {
            "fcs_file": "sample.fcs",
            "harmonization_file": "panel_harmonization.yaml",
        }
    }
    cfg2 = PipelineConfig.from_dict(nested_data)
    assert cfg2.harmonization_file == "panel_harmonization.yaml"
    assert cfg2.enable_harmonization is True
