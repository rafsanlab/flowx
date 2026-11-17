"""Integration tests for FlowCytometryPipeline."""

import json
from pathlib import Path

from flowx.pipeline import FlowCytometryPipeline, resolve_cohort_metadata, resolve_cohort_samples


def test_pipeline_execution(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify full pipeline runs end-to-end, generates all artifacts, and validates reproducibility."""
    pipeline = FlowCytometryPipeline(output_dir=tmp_path, random_seed=42, enable_discovery=True)
    result = pipeline.run(tqc_sample_fcs_path)

    assert result.event_count > 0
    assert result.gating_mode == "som"
    assert result.qc_method == "clustering"
    assert result.reproducibility_verified is True
    assert Path(result.workspace_json_path).is_file()
    assert Path(result.workspace_xml_path).is_file()
    assert Path(result.statistics_csv_path).is_file()
    assert Path(result.report_md_path).is_file()
    assert Path(result.report_md_path).stat().st_size > 0

    # Verify key plots were created
    assert "step1_raw_morphology" in result.generated_plots
    assert "step1_clustering_qc_diagnostic" in result.generated_plots
    assert "step2_compensation_diagnostic" in result.generated_plots
    assert "step3_transformation_diagnostic" in result.generated_plots
    assert "step4_cleanup_diagnostics" in result.generated_plots
    assert "step5_som_clusters_scatter" in result.generated_plots
    assert "step5_som_marker_heatmap" in result.generated_plots
    assert result.cleanup_result is not None

    for plot_path in result.generated_plots.values():
        assert Path(plot_path).is_file()
        assert Path(plot_path).stat().st_size > 0


def test_pipeline_template_gating_mode(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify pipeline runs with explicit template gating mode and generates cascade overview."""
    pipeline = FlowCytometryPipeline(
        output_dir=tmp_path,
        gating_mode="template",
        qc_method="template",
        random_seed=42,
    )
    result = pipeline.run(tqc_sample_fcs_path)
    assert result.gating_mode == "template"
    assert result.qc_method == "template"
    assert "step4_cascade_overview" in result.generated_plots
    assert Path(result.generated_plots["step4_cascade_overview"]).is_file()


def test_consecutive_runs_prevent_overwrite(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify running the same sample multiple times preserves both runs and updates latest pointer."""
    pipeline = FlowCytometryPipeline(output_dir=tmp_path, timestamp_dir=True, random_seed=42)

    # First run
    run1 = pipeline.run(tqc_sample_fcs_path)
    # Second run immediately after
    run2 = pipeline.run(tqc_sample_fcs_path)

    # Assert distinct directories
    assert run1.run_dir != run2.run_dir
    assert Path(run1.run_dir).is_dir()
    assert Path(run2.run_dir).is_dir()

    # Both reports must exist independently
    assert Path(run1.report_md_path).is_file()
    assert Path(run2.report_md_path).is_file()

    # Manifest must record both runs
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.is_file()
    history = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(history) == 2


def test_overwrite_mode(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify timestamp_dir=False with overwrite=True writes directly to output_dir."""
    pipeline = FlowCytometryPipeline(
        output_dir=tmp_path, timestamp_dir=False, overwrite=True, random_seed=42
    )
    result = pipeline.run(tqc_sample_fcs_path)
    assert result.run_dir == str(tmp_path)
    assert Path(result.report_md_path).is_file()


def test_pipeline_logging(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify pipeline execution generates a structured, debug-friendly pipeline.log file in output folder."""
    pipeline = FlowCytometryPipeline(output_dir=tmp_path, random_seed=42, log_level="DEBUG")
    result = pipeline.run(tqc_sample_fcs_path)

    # 1. PipelineResult contains log_file_path
    assert result.log_file_path is not None
    log_path = Path(result.log_file_path)
    assert log_path.is_file()
    assert log_path.name == "pipeline.log"
    assert log_path.parent == Path(result.run_dir)

    # 2. Log contents verify step execution, timings, and debug diagnostics
    log_content = log_path.read_text(encoding="utf-8")
    assert ">>> Starting Step 1: Parse FCS File & Extract Metadata..." in log_content
    assert "<<< Completed Step 1: Parse FCS File & Extract Metadata (duration:" in log_content
    assert ">>> Starting Step 2: Fluorescence Spillover Compensation..." in log_content
    assert ">>> Starting Step 3: Scale Transformation..." in log_content
    assert ">>> Starting Step 4: Gating Hierarchy Construction & Evaluation..." in log_content
    assert ">>> Starting Step 5: Population Statistics..." in log_content
    assert ">>> Starting Step 6: Workspace Session Persistence..." in log_content
    assert ">>> Starting Step 7: Reproducibility Verification..." in log_content
    assert ">>> Starting Step 8: Markdown Report Generation..." in log_content

    # Check high-precision timestamp format and log level
    assert "DEBUG" in log_content
    assert "INFO" in log_content

    # Verify manifest references the log file and execution duration
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.is_file()
    history = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "log_file" in history[-1]
    assert history[-1]["log_file"] == str(log_path)
    assert "execution_time_seconds" in history[-1]
    assert "execution_duration_formatted" in history[-1]
    assert history[-1]["execution_time_seconds"] > 0


def test_pipeline_execution_timing(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify execution timer is recorded in log, report, manifest, and PipelineResult."""
    pipeline = FlowCytometryPipeline(output_dir=tmp_path, random_seed=42)
    result = pipeline.run(tqc_sample_fcs_path)

    # 1. PipelineResult holds execution timer
    assert result.execution_time_seconds > 0.0
    assert result.execution_duration_formatted.endswith("s")

    # 2. Markdown report includes execution duration in summary and footer
    report_text = Path(result.report_md_path).read_text(encoding="utf-8")
    assert "- **Total Execution Time:**" in report_text
    assert "*Pipeline execution completed in" in report_text

    # 3. Log file captures initiation and completion timings
    log_text = Path(result.log_file_path).read_text(encoding="utf-8")
    assert "Initiated flow cytometry pipeline run for sample" in log_text
    assert "All pipeline tasks completed successfully in" in log_text


def test_pipeline_time_qc_integration(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify flowx pipeline automatically runs Step 1 Time QC and produces diagnostic outputs."""
    pipeline = FlowCytometryPipeline(
        output_dir=tmp_path,
        random_seed=42,
        enable_time_qc=True,
        enable_umap=False,
    )
    result = pipeline.run(tqc_sample_fcs_path)

    assert result.time_qc_result is not None
    assert result.time_qc_result.passed_pct > 85.0
    assert "step1_time_qc" in result.generated_plots
    assert Path(result.generated_plots["step1_time_qc"]).is_file()
    assert (Path(result.run_dir) / "01_time_qc" / "tables" / "time_qc_sample_summary.csv").is_file()
    assert result.reproducibility_verified is True


def test_pipeline_time_qc_on_case1(tmp_path: Path, navios_suite_dir: Path):
    """Verify flowx pipeline runs Step 1 Time QC on Case1 Navios file with 'TIME' channel (T2)."""
    case1_path = navios_suite_dir / "Case1_A.fcs"
    pipeline = FlowCytometryPipeline(
        output_dir=tmp_path,
        random_seed=42,
        enable_time_qc=True,
        enable_umap=False,
        gating_mode="template",
        qc_method="template",
    )
    result = pipeline.run(case1_path)

    assert result.time_qc_result is not None
    assert result.time_qc_result.time_channel == "TIME"
    assert result.time_qc_result.duration_sec > 300.0
    assert (Path(result.run_dir) / "01_time_qc" / "tables" / "time_qc_sample_summary.csv").is_file()


def test_resolve_cohort_samples_and_metadata(synthetic_conventional_dataset_dir: Path):
    """Verify universal cohort sample and metadata resolution on schema-compliant dataset."""
    files, name = resolve_cohort_samples(synthetic_conventional_dataset_dir)
    assert len(files) == 4
    assert all(f.suffix.lower() == ".fcs" for f in files)
    assert name == "synth_pbmc_cohort"
    # Assert control files in raw/controls/ were excluded
    assert all("compensation" not in f.name.lower() for f in files)
    assert all("unstained" not in f.name.lower() for f in files)
    meta = resolve_cohort_metadata(files, dataset_dir=synthetic_conventional_dataset_dir)
    assert len(meta) == 4
    assert "condition" in meta.columns
    assert set(meta["condition"]) == {"Healthy_Control", "Treated"}
