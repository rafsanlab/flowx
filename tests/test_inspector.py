"""Unit tests for the FCS inspection and quality control module."""

from pathlib import Path
import pandas as pd

from flowx.io.inspector import (
    FCSBatchSummary,
    FCSFileInfo,
    export_inspection_markdown,
    export_inspection_suite,
    export_qc_file_lists,
    find_fcs_files,
    inspect_fcs_directory,
    inspect_fcs_file,
    inspect_fcs_files,
    load_population_map,
    read_fcs_header_and_text,
)

def test_fast_header_and_text_reader(navios_suite_dir: Path):
    """Verify sub-millisecond header and TEXT segment extraction."""
    raw_case1_a = navios_suite_dir / "Case1_A.fcs"
    version, meta, text_start, text_end, data_start, data_end = read_fcs_header_and_text(raw_case1_a)
    assert version.startswith("FCS")
    assert text_start > 0
    assert text_end > text_start
    assert "$TOT" in meta or "tot" in [k.lower() for k in meta.keys()]
    assert "$PAR" in meta or "par" in [k.lower() for k in meta.keys()]


def test_population_map_loading(population_map_json_path: Path):
    """Verify decoupled and scalable population map loading from dict and JSON."""
    # From direct dict
    d_map, pat = load_population_map({"A": "Total cells", "B": "Other"})
    assert d_map["A"] == "Total cells"
    assert pat is None

    # From JSON config file
    f_map, f_pat = load_population_map(population_map_json_path)
    assert f_map["A"] == "Full cell population (Total)"
    assert f_map["O"] == "T lymphocytes"
    assert f_pat is not None

    # From None (generic mode)
    empty_map, no_pat = load_population_map(None)
    assert empty_map == {}
    assert no_pat is None


def test_inspect_single_fcs_file(navios_suite_dir: Path, population_map_json_path: Path):
    """Verify detailed single-file inspection adhering to MIFlowCyt standards."""
    raw_case1_a = navios_suite_dir / "Case1_A.fcs"
    # 1. Generic inspection (no dataset-specific assumptions)
    generic_info = inspect_fcs_file(raw_case1_a)
    assert isinstance(generic_info, FCSFileInfo)
    assert generic_info.filename == "Case1_A.fcs"
    assert generic_info.vendor == "Beckman Coulter"
    assert generic_info.cytometer_model == "Navios"
    assert generic_info.event_count == 1200
    assert generic_info.channel_count == 15
    assert len(generic_info.scatter_channels) == 4
    assert len(generic_info.fluorescent_channels) == 10
    assert generic_info.has_spillover is True
    assert generic_info.spillover_channel_count == 10
    assert "CD14" in generic_info.markers
    assert "CD19" in generic_info.markers
    assert "CD45" in generic_info.markers
    assert generic_info.qc_status == "PASS"

    # 2. Configured population inspection
    mapped_info = inspect_fcs_file(raw_case1_a, population_map=population_map_json_path)
    assert mapped_info.population_class == "A"
    assert "Full cell population" in mapped_info.population_name


def test_inspect_explicit_files_list(navios_suite_dir: Path, population_map_json_path: Path):
    """Verify inspecting an explicit list of filenames."""
    file_list = [navios_suite_dir / "Case1_A.fcs", navios_suite_dir / "Case10_B.fcs"]
    results, summary = inspect_fcs_files(file_list, population_map=population_map_json_path)

    assert len(results) == 2
    assert summary.total_files == 2
    assert summary.total_events == results[0].event_count + results[1].event_count
    assert "Beckman Coulter" in summary.unique_vendors
    assert "A" in summary.population_counts
    assert "B" in summary.population_counts


def test_multi_vendor_recognition(navios_suite_dir: Path):
    """Verify auto-detection of different cytometer manufacturers and models."""
    # 1. Beckman Coulter Navios
    bc_info = inspect_fcs_file(navios_suite_dir / "Case1_A.fcs")
    assert bc_info.vendor == "Beckman Coulter"
    assert "Navios" in bc_info.cytometer_model

    # 2. BD Biosciences LSRFortessa
    fortessa_info = inspect_fcs_file(navios_suite_dir / "BD_Fortessa_sample.fcs")
    assert fortessa_info.vendor == "BD Biosciences"
    assert "LSRFortessa" in fortessa_info.cytometer_model

    # 3. BD Biosciences LSRII
    lsrii_info = inspect_fcs_file(navios_suite_dir / "BD_LSRII_sample.fcs")
    assert lsrii_info.vendor == "BD Biosciences"
    assert "LSRII" in lsrii_info.cytometer_model


def test_panel_variants_and_qc_discrepancy(navios_suite_dir: Path, population_map_json_path: Path):
    """Verify detection of altered channel signatures and missing compensation matrices."""
    # Case10_B has dots in detector names and lacks embedded spillover
    b_info = inspect_fcs_file(navios_suite_dir / "Case10_B.fcs", population_map=population_map_json_path)
    assert "Variant-2" in b_info.panel_signature
    assert b_info.population_class == "B"
    assert b_info.has_spillover is False
    assert b_info.qc_status == "WARNING"
    assert any("spillover" in w.lower() for w in b_info.qc_warnings)

    # Case30_A has clean stain names as detectors and lacks embedded spillover
    case30_info = inspect_fcs_file(navios_suite_dir / "Case30_A.fcs", population_map=population_map_json_path)
    assert "Variant-3" in case30_info.panel_signature
    assert case30_info.has_spillover is False
    assert case30_info.qc_status == "WARNING"


def test_batch_directory_inspection(navios_suite_dir: Path, population_map_json_path: Path):
    """Verify batch inspection of multiple files and aggregate summary generation."""
    results, summary = inspect_fcs_directory(
        navios_suite_dir, recursive=False, pattern="Case1_*.fcs", population_map=population_map_json_path
    )

    assert len(results) == 7
    assert isinstance(summary, FCSBatchSummary)
    assert summary.total_files == 7
    assert summary.total_events > 0
    assert summary.min_events > 0
    assert summary.max_events >= summary.min_events
    assert "Beckman Coulter" in summary.unique_vendors
    assert "A" in summary.population_counts
    assert "O" in summary.population_counts
    assert "P" in summary.population_counts


def test_export_inspection_markdown(tmp_path: Path, navios_suite_dir: Path, population_map_json_path: Path):
    """Verify programmatic Markdown report generation directly from code."""
    file_list = [navios_suite_dir / "Case1_A.fcs", navios_suite_dir / "Case10_B.fcs"]
    results, summary = inspect_fcs_files(file_list, population_map=population_map_json_path)

    out_md = tmp_path / "test_report.md"
    saved_path = export_inspection_markdown(summary, results, out_md, generated_by="FlowX Inspector Test Engine")

    assert saved_path.is_file()
    content = saved_path.read_text(encoding="utf-8")
    assert "FlowX Inspector Test Engine" in content
    assert "MIFlowCyt" in content
    assert "Case1_A.fcs" in content
    assert "Case10_B.fcs" in content
    assert "Downstream Preprocessing Recommendations" in content


def test_export_inspection_suite(tmp_path: Path, navios_suite_dir: Path, population_map_json_path: Path):
    """Verify complete suite export creating summary CSV, channel CSV, and report in output dir."""
    file_list = [navios_suite_dir / "Case1_A.fcs", navios_suite_dir / "Case10_B.fcs"]
    results, summary = inspect_fcs_files(file_list, population_map=population_map_json_path)

    suite_dir = tmp_path / "flowcyt_inspection"
    paths = export_inspection_suite(results, summary, output_dir=suite_dir)

    assert paths["summary_csv"].is_file()
    assert paths["channels_csv"].is_file()
    assert paths["report_md"].is_file()
    assert paths["files_pass"].is_file()
    assert paths["files_flagged"].is_file()

    # Check summary CSV
    df = pd.read_csv(paths["summary_csv"])
    assert len(df) == 2
    assert "filename" in df.columns
    assert "vendor" in df.columns

    # Check channels CSV
    ch_df = pd.read_csv(paths["channels_csv"])
    assert len(ch_df) == 30  # 15 channels * 2 files
    assert "pnn_detector" in ch_df.columns


def test_export_qc_file_lists(tmp_path: Path, navios_suite_dir: Path, population_map_json_path: Path):
    """Verify export of pass and flagged file lists for downstream handoff."""
    file_list = [navios_suite_dir / "Case1_A.fcs", navios_suite_dir / "Case30_A.fcs"]
    results, _ = inspect_fcs_files(file_list, population_map=population_map_json_path)

    pass_p, flag_p = export_qc_file_lists(results, tmp_path)
    assert pass_p.is_file()
    assert flag_p.is_file()

    pass_content = pass_p.read_text(encoding="utf-8")
    flag_content = flag_p.read_text(encoding="utf-8")

    assert "Case1_A.fcs" in pass_content
    assert "Case30_A.fcs" in flag_content


def test_find_fcs_files_nested_subdirectories(tmp_path: Path):
    """Verify finding FCS files across multiple nested subdirectories, uppercase extensions, and depth limits."""
    p1 = tmp_path / "patient_01"
    p2_deep = tmp_path / "patient_02" / "run_date_2026"
    hidden = tmp_path / ".hidden_dir"

    p1.mkdir(parents=True)
    p2_deep.mkdir(parents=True)
    hidden.mkdir(parents=True)

    (tmp_path / "root_sample.fcs").write_text("dummy")
    (p1 / "tube_01.fcs").write_text("dummy")
    (p1 / "tube_02.FCS").write_text("dummy")
    (p2_deep / "deep_sample.fcs3").write_text("dummy")
    (hidden / "ignored.fcs").write_text("dummy")

    # 1. Recursive scan (default): finds all 4 valid files across subdirectories, ignores hidden
    all_files = find_fcs_files(tmp_path, recursive=True)
    assert len(all_files) == 4
    filenames = {p.name for p in all_files}
    assert "root_sample.fcs" in filenames
    assert "tube_01.fcs" in filenames
    assert "tube_02.FCS" in filenames  # Case insensitivity
    assert "deep_sample.fcs3" in filenames
    assert "ignored.fcs" not in filenames  # Hidden folder skipped

    # 2. Non-recursive scan: only finds root file
    root_only = find_fcs_files(tmp_path, recursive=False)
    assert len(root_only) == 1
    assert root_only[0].name == "root_sample.fcs"

    # 3. Max depth = 1: finds root and patient_01, but skips depth 2 (patient_02/run_date_2026)
    depth_1_files = find_fcs_files(tmp_path, recursive=True, max_depth=1)
    assert len(depth_1_files) == 3
    depth_1_names = {p.name for p in depth_1_files}
    assert "deep_sample.fcs3" not in depth_1_names


def test_cli_recursive_options_parser(navios_suite_dir: Path):
    """Verify CLI argument parser supports -r, --recursive, --no-recursive, and --max-depth."""
    from flowx.cli import build_parser

    parser = build_parser()

    # Default is recursive=True
    args_default = parser.parse_args(["inspect", str(navios_suite_dir)])
    assert args_default.recursive is True
    assert args_default.max_depth is None

    # Explicit -r / --recursive flag
    args_r = parser.parse_args(["inspect", str(navios_suite_dir), "-r", "--max-depth", "3"])
    assert args_r.recursive is True
    assert args_r.max_depth == 3

    # Explicit --no-recursive
    args_no_r = parser.parse_args(["inspect", str(navios_suite_dir), "--no-recursive"])
    assert args_no_r.recursive is False
