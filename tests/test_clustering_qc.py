"""Unit tests for automated clustering-based quality control (QC) module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from flowx.clustering.qc_clustering import (
    ClusteringQCResult,
    cluster_morphology_qc,
    cluster_singlets_qc,
    run_clustering_qc,
)
from flowx.gating.gate import PolygonGate
from flowx.io.config import PipelineConfig
from flowx.io.parser import parse_fcs
from flowx.pipeline import FlowCytometryPipeline


def test_cluster_morphology_qc(tqc_sample_fcs_path: Path):
    """Verify automated GMM scatter morphology clustering identifies cells and builds a valid gate."""
    fcs = parse_fcs(tqc_sample_fcs_path)
    cell_mask, cell_gate, labels, stats = cluster_morphology_qc(fcs, fcs.events_df, n_components=3, random_state=42)

    assert isinstance(cell_gate, PolygonGate)
    assert cell_gate.name == "Cells"
    assert len(cell_gate.channels) == 2
    assert len(cell_gate.vertices) >= 3

    assert len(cell_mask) == fcs.event_count
    assert len(labels) == fcs.event_count

    # In PBMC sample, intact cells must represent a dominant fraction
    assert stats["cell_count"] > 0.60 * fcs.event_count
    assert stats["debris_count"] >= 0
    assert "Intact Cells" in np.unique(labels)


def test_cluster_singlets_qc(tqc_sample_fcs_path: Path):
    """Verify automated pulse geometry clustering identifies singlets and excludes doublets."""
    fcs = parse_fcs(tqc_sample_fcs_path)
    cell_mask, _, _, _ = cluster_morphology_qc(fcs, fcs.events_df, random_state=42)

    singlet_mask, singlet_gate, labels, stats = cluster_singlets_qc(
        fcs, fcs.events_df, parent_mask=cell_mask, random_state=42
    )

    assert isinstance(singlet_gate, PolygonGate)
    assert singlet_gate.name == "Singlets"
    assert singlet_gate.parent_gate == "Cells"
    assert len(singlet_gate.vertices) >= 3

    assert len(singlet_mask) == fcs.event_count
    assert stats["singlet_count"] > 0
    assert stats["singlet_pct_parent"] > 60.0
    assert "Singlets" in np.unique(labels)


def test_run_clustering_qc(tqc_sample_fcs_path: Path):
    """Verify end-to-end run_clustering_qc returns structured results and summary metrics."""
    fcs = parse_fcs(tqc_sample_fcs_path)
    qc_res = run_clustering_qc(fcs, fcs.events_df, random_state=42)

    assert isinstance(qc_res, ClusteringQCResult)
    assert qc_res.total_events == fcs.event_count
    assert qc_res.cell_count > 0
    assert qc_res.singlet_count > 0
    assert qc_res.debris_count >= 0
    assert qc_res.doublet_count >= 0

    assert isinstance(qc_res.summary_stats, pd.DataFrame)
    assert "QC Stage" in qc_res.summary_stats.columns
    assert "Population" in qc_res.summary_stats.columns
    assert "Event Count" in qc_res.summary_stats.columns
    assert len(qc_res.summary_stats) == 5


def test_pipeline_clustering_qc_integration(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify FlowCytometryPipeline runs with qc_method='clustering', produces QC plot, and updates report."""
    pipeline = FlowCytometryPipeline(
        output_dir=tmp_path,
        qc_method="clustering",
        gating_mode="template",
        random_seed=42,
    )
    res = pipeline.run(tqc_sample_fcs_path)

    assert res.qc_method == "clustering"
    assert res.clustering_qc_result is not None
    assert res.reproducibility_verified is True

    # Check that diagnostic plot was created
    assert "step1_clustering_qc_diagnostic" in res.generated_plots
    qc_plot_path = Path(res.generated_plots["step1_clustering_qc_diagnostic"])
    assert qc_plot_path.is_file()
    assert qc_plot_path.stat().st_size > 0

    # Check report content
    report_text = Path(res.report_md_path).read_text(encoding="utf-8")
    assert "Automated Clustering-Based Quality Control (QC)" in report_text
    assert "**QC Methodology:** `Automated Clustering (GMM)`" in report_text
    assert "clustering_qc_diagnostic.png" in report_text


def test_config_qc_method_validation():
    """Verify PipelineConfig validates and propagates qc_method."""
    cfg = PipelineConfig(qc_method="clustering")
    cfg.validate()
    assert cfg.qc_method == "clustering"

    with pytest.raises(ValueError, match="Invalid qc_method"):
        invalid_cfg = PipelineConfig(qc_method="unsupported_method")
        invalid_cfg.validate()

    pipeline, _ = cfg.to_pipeline()
    assert pipeline.qc_method == "clustering"
