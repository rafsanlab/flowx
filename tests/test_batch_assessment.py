"""Unit tests for FlowX Batch Necessity & Technical Drift Assessment Engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from flowx.batch.assessment import (
    BatchNecessityAssessment,
    assess_batch_effect_need,
)
from flowx.batch.models import CohortSampleInfo
from flowx.cli import main
from flowx.visualization.batch_plots import plot_batch_assessment_summary


def test_assess_batch_single_batch():
    """Test assessment when all samples come from a single batch."""
    rng = np.random.default_rng(42)
    dfs = {
        "Sample_1": pd.DataFrame({"CD3": rng.normal(10, 2, 500), "CD4": rng.normal(8, 2, 500)}),
        "Sample_2": pd.DataFrame({"CD3": rng.normal(10, 2, 500), "CD4": rng.normal(8, 2, 500)}),
    }
    samples = [
        CohortSampleInfo(sample_id="Sample_1", batch_id="Batch_1"),
        CohortSampleInfo(sample_id="Sample_2", batch_id="Batch_1"),
    ]

    res = assess_batch_effect_need(dfs, samples)
    assert res.recommendation == "NOT_NEEDED"
    assert res.drift_score == 0.0
    assert res.confidence == 1.0
    assert res.is_normalization_recommended is False
    assert "Single experimental batch detected" in res.rationale[0]


def test_assess_batch_confounded_condition():
    """Test assessment when batches are 100% confounded with biological condition."""
    rng = np.random.default_rng(42)
    dfs = {
        "Sample_1": pd.DataFrame({"CD3": rng.normal(10, 2, 500), "CD19": rng.normal(5, 1, 500)}),
        "Sample_2": pd.DataFrame({"CD3": rng.normal(15, 2, 500), "CD19": rng.normal(9, 1, 500)}),
    }
    samples = [
        CohortSampleInfo(sample_id="Sample_1", batch_id="Batch_A", extra_metadata={"condition": "Healthy"}),
        CohortSampleInfo(sample_id="Sample_2", batch_id="Batch_B", extra_metadata={"condition": "Leukemia"}),
    ]

    res = assess_batch_effect_need(dfs, samples, condition_key="condition")
    assert res.recommendation == "NOT_RECOMMENDED_CONFOUNDED"
    assert res.confounded_with_condition is True
    assert res.is_normalization_recommended is False
    assert "confounded" in res.condition_confounding_note.lower()


def test_assess_batch_confounded_multibatch_pure_conditions():
    """Test confounding when one condition spans multiple batches and another is isolated in its own batch."""
    rng = np.random.default_rng(42)
    dfs = {
        "ALS_1": pd.DataFrame({"CD3": rng.normal(14, 2, 300)}),
        "HC_1": pd.DataFrame({"CD3": rng.normal(10, 2, 300)}),
        "HC_2": pd.DataFrame({"CD3": rng.normal(10, 2, 300)}),
    }
    samples = [
        CohortSampleInfo(sample_id="ALS_1", batch_id="2024-05-16", extra_metadata={"condition": "ALS_Patient"}),
        CohortSampleInfo(sample_id="HC_1", batch_id="2024-04-08", extra_metadata={"condition": "Healthy_Control"}),
        CohortSampleInfo(sample_id="HC_2", batch_id="2023-11-27", extra_metadata={"condition": "Healthy_Control"}),
    ]

    res = assess_batch_effect_need(dfs, samples, condition_key="condition")
    assert res.recommendation == "NOT_RECOMMENDED_CONFOUNDED"
    assert res.confounded_with_condition is True
    assert res.is_normalization_recommended is False


def test_assess_batch_with_control_anchors_drifting():
    """Test assessment when multi-batch controls show substantial instrument drift."""
    rng = np.random.default_rng(42)
    # Simulate batch shift of +5.0 MFI on CD3
    dfs = {
        "Ctrl_B1": pd.DataFrame({"CD3": rng.normal(10.0, 1.5, 600), "CD8": rng.normal(8.0, 1.5, 600)}),
        "Ctrl_B2": pd.DataFrame({"CD3": rng.normal(16.0, 1.5, 600), "CD8": rng.normal(13.0, 1.5, 600)}),
        "Pat_B1": pd.DataFrame({"CD3": rng.normal(10.5, 1.5, 600), "CD8": rng.normal(8.2, 1.5, 600)}),
        "Pat_B2": pd.DataFrame({"CD3": rng.normal(15.8, 1.5, 600), "CD8": rng.normal(12.8, 1.5, 600)}),
    }
    samples = [
        CohortSampleInfo(sample_id="Ctrl_B1", batch_id="B1", is_control=True),
        CohortSampleInfo(sample_id="Ctrl_B2", batch_id="B2", is_control=True),
        CohortSampleInfo(sample_id="Pat_B1", batch_id="B1", is_control=False),
        CohortSampleInfo(sample_id="Pat_B2", batch_id="B2", is_control=False),
    ]

    res = assess_batch_effect_need(dfs, samples)
    assert res.recommendation == "RECOMMENDED_CYTONORM"
    assert res.has_control_anchors is True
    assert res.drift_score > 25.0
    assert res.is_normalization_recommended is True
    df_metrics = res.to_dataframe()
    assert len(df_metrics) == 2
    assert any(df_metrics["is_drifting"])


def test_assess_batch_minimal_drift():
    """Test assessment when multi-batch technical replicates have minimal baseline drift."""
    rng = np.random.default_rng(42)
    # Negligible drift (<2%)
    dfs = {
        "Rep_D1": pd.DataFrame({"CD3": rng.normal(10.0, 1.5, 800), "CD4": rng.normal(8.0, 1.2, 800)}),
        "Rep_D2": pd.DataFrame({"CD3": rng.normal(10.05, 1.5, 800), "CD4": rng.normal(8.02, 1.2, 800)}),
    }
    samples = [
        CohortSampleInfo(sample_id="Rep_D1", batch_id="Day1", is_control=False),
        CohortSampleInfo(sample_id="Rep_D2", batch_id="Day2", is_control=False),
    ]

    res = assess_batch_effect_need(dfs, samples)
    assert res.recommendation == "NOT_NEEDED"
    assert res.drift_score < 18.0
    assert res.is_normalization_recommended is False


def test_assess_batch_anchor_free_caution():
    """Test assessment when multi-batch drift is high but no control anchors are provided."""
    rng = np.random.default_rng(42)
    # Significant drift without control anchors
    dfs = {
        "Samp_B1": pd.DataFrame({"CD3": rng.normal(10.0, 1.5, 600), "CD8": rng.normal(7.0, 1.5, 600)}),
        "Samp_B2": pd.DataFrame({"CD3": rng.normal(17.0, 1.5, 600), "CD8": rng.normal(14.0, 1.5, 600)}),
    }
    samples = [
        CohortSampleInfo(sample_id="Samp_B1", batch_id="B1", is_control=False),
        CohortSampleInfo(sample_id="Samp_B2", batch_id="B2", is_control=False),
    ]

    res = assess_batch_effect_need(dfs, samples)
    assert res.recommendation == "CAUTION_ANCHOR_FREE"
    assert res.has_control_anchors is False
    assert res.drift_score > 25.0
    assert res.is_normalization_recommended is True


def test_assessment_markdown_and_plot(tmp_path: Path):
    """Test report generation and diagnostic plotting for batch assessment."""
    rng = np.random.default_rng(42)
    dfs = {
        "Ctrl_1": pd.DataFrame({"CD3": rng.normal(10.0, 1.5, 400), "CD4": rng.normal(8.0, 1.2, 400)}),
        "Ctrl_2": pd.DataFrame({"CD3": rng.normal(15.0, 1.5, 400), "CD4": rng.normal(12.0, 1.2, 400)}),
    }
    samples = [
        CohortSampleInfo(sample_id="Ctrl_1", batch_id="Batch1", is_control=True),
        CohortSampleInfo(sample_id="Ctrl_2", batch_id="Batch2", is_control=True),
    ]

    res = assess_batch_effect_need(dfs, samples)
    md = res.to_markdown()
    assert "# FlowX Step 3 Batch Effect Necessity Assessment" in md
    assert "RECOMMENDED" in md

    out_png = tmp_path / "assessment_test.png"
    fig = plot_batch_assessment_summary(res, output_path=out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 1000


def test_cli_batch_check_manifest(tmp_path: Path, cytotrol_fcs_path: Path):
    """Test CLI batch-check command with a synthesized manifest."""
    manifest_csv = tmp_path / "cohort_manifest.csv"
    manifest_csv.write_text(
        f"filepath,sample_id,batch_id,is_control\n{cytotrol_fcs_path},Sample1,Batch1,False\n{cytotrol_fcs_path},Sample2,Batch2,False\n",
        encoding="utf-8",
    )

    exit_code = main(["batch-check", "--manifest", str(manifest_csv)])
    assert exit_code == 0
