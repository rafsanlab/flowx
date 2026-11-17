"""Unit and integration tests for Cohort Batch Runner and CLI commands."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from flowx.batch.models import (
    BatchCorrectionConfig,
    BatchPreparationConfig,
    CohortSampleInfo,
    DownsampleConfig,
)
from flowx.batch.runner import (
    discover_cohort_samples,
    load_cohort_manifest,
    run_batch_preparation,
)
from flowx.cli import main


def test_load_cohort_manifest_csv(tmp_path: Path):
    """Test parsing cohort manifest CSV into CohortSampleInfo instances."""
    csv_file = tmp_path / "manifest.csv"
    csv_file.write_text(
        "filepath,sample_id,batch_id,is_control\n"
        "data/sample1.fcs,Sample1,BatchA,True\n"
        "data/sample2.fcs,Sample2,BatchB,False\n",
        encoding="utf-8",
    )

    samples = load_cohort_manifest(csv_file)
    assert len(samples) == 2
    assert samples[0].sample_id == "Sample1"
    assert samples[0].batch_id == "BatchA"
    assert samples[0].is_control is True
    assert samples[1].is_control is False


def test_discover_cohort_samples(tmp_path: Path):
    """Test auto-discovery of FCS files defaults to uniform Batch_1 without arbitrary filename guessing."""
    f1 = tmp_path / "PTLG021_Control_1.fcs"
    f2 = tmp_path / "PTLG021_Sample_2.fcs"
    f1.write_bytes(b"dummy")
    f2.write_bytes(b"dummy")

    samples = discover_cohort_samples(tmp_path)
    assert len(samples) == 2
    s_map = {s.sample_id: s for s in samples}

    # Verify no arbitrary filename token guessing
    assert s_map["PTLG021_Control_1"].batch_id == "Batch_1"
    assert s_map["PTLG021_Control_1"].is_control is False
    assert s_map["PTLG021_Sample_2"].batch_id == "Batch_1"
    assert s_map["PTLG021_Sample_2"].is_control is False


def test_run_batch_preparation_in_memory(tmp_path: Path):
    """Test in-memory chained handover execution of normalization and downsampling."""
    rng = np.random.default_rng(42)

    # Simulate 2 batches with 2 samples each
    b1_ctrl = pd.DataFrame({"CD4": rng.normal(10, 2, 2000), "CD8": rng.normal(8, 1.5, 2000)})
    b1_exp = pd.DataFrame({"CD4": rng.normal(10, 2, 3000), "CD8": rng.normal(8, 1.5, 3000)})

    b2_ctrl = pd.DataFrame({"CD4": rng.normal(14, 2, 2000), "CD8": rng.normal(11, 1.5, 2000)})
    b2_exp = pd.DataFrame({"CD4": rng.normal(14, 2, 4000), "CD8": rng.normal(11, 1.5, 4000)})

    cohort_dfs = {
        "B1_Ctrl": b1_ctrl,
        "B1_Exp": b1_exp,
        "B2_Ctrl": b2_ctrl,
        "B2_Exp": b2_exp,
    }

    sample_info = [
        CohortSampleInfo(sample_id="B1_Ctrl", batch_id="B1", is_control=True),
        CohortSampleInfo(sample_id="B1_Exp", batch_id="B1", is_control=False),
        CohortSampleInfo(sample_id="B2_Ctrl", batch_id="B2", is_control=True),
        CohortSampleInfo(sample_id="B2_Exp", batch_id="B2", is_control=False),
    ]

    cfg = BatchPreparationConfig(
        batch_normalization=BatchCorrectionConfig(
            enabled=True,
            method="cytonorm",
            goal="median",
            n_clusters=4,
            n_quantiles=25,
        ),
        downsampling=DownsampleConfig(
            enabled=True,
            mode="target_count",
            target_events=1000,
            random_seed=42,
        ),
        output_dir=tmp_path / "batch_out",
        export_pooled_cohort=True,
        generate_diagnostics=True,
    )

    result = run_batch_preparation(
        cohort_dfs=cohort_dfs,
        sample_info=sample_info,
        config=cfg,
    )

    assert result.sample_count == 4
    assert result.total_prepared_events == 4000  # 4 samples * 1000 events
    for df in result.prepared_dfs.values():
        assert len(df) == 1000

    assert result.pooled_df is not None
    assert len(result.pooled_df) == 4000
    assert "sample_id" in result.pooled_df.columns
    assert "batch_id" in result.pooled_df.columns

    # Verify generated plots and tables
    assert Path(result.output_dir).exists()
    assert (tmp_path / "batch_out" / "tables" / "cohort_batch_summary.csv").exists()
    assert (tmp_path / "batch_out" / "plots" / "step3_cohort_downsampling_balance.png").exists()


def test_cli_batch_prepare_help(capsys):
    """Test CLI batch-prepare subcommand execution with --help."""
    with pytest.raises(SystemExit) as exc_info:
        main(["batch-prepare", "--help"])
    assert exc_info.value.code == 0


def test_cli_batch_prepare_standalone(tmp_path: Path, synthetic_cytonorm_cohort_dir: Path):
    """Test running flowx batch-prepare CLI command on benchmark directory."""
    out_dir = tmp_path / "cli_batch_out"
    args = [
        "batch-prepare",
        str(synthetic_cytonorm_cohort_dir),
        "--manifest", str(synthetic_cytonorm_cohort_dir / "cohort_manifest.csv"),
        "--downsample", "200",
        "--method", "cytonorm",
        "-o", str(out_dir),
    ]

    ret = main(args)
    assert ret == 0
    assert (out_dir / "tables" / "cohort_batch_summary.csv").exists()
    assert (out_dir / "plots" / "step3_cohort_downsampling_balance.png").exists()


def test_cli_batch_prepare_with_config(tmp_path: Path, synthetic_cytonorm_cohort_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Test running flowx batch-prepare CLI command using a YAML configuration."""
    cfg_file = synthetic_cytonorm_cohort_dir / "batch_prep_cytonorm.yaml"
    out_dir = tmp_path / "cli_config_out"
    args = [
        "batch-prepare",
        "-c", str(cfg_file),
        "-o", str(out_dir),
    ]
    monkeypatch.setattr("sys.argv", ["flowx"] + args)

    ret = main(args)
    assert ret == 0
    assert (out_dir / "tables" / "cohort_batch_summary.csv").exists()


def test_cytonorm_missing_marker_raises_keyerror(tmp_path: Path):
    """Test that applying CytoNorm to a DataFrame missing trained markers raises a descriptive KeyError."""
    from flowx.batch.normalization import fit_cytonorm

    rng = np.random.default_rng(42)
    b1_ctrl = pd.DataFrame({"CD4": rng.normal(10, 2, 500), "CD8": rng.normal(8, 1.5, 500)})
    b2_ctrl = pd.DataFrame({"CD4": rng.normal(14, 2, 500), "CD8": rng.normal(11, 1.5, 500)})

    cohort_dfs = {"B1_Ctrl": b1_ctrl, "B2_Ctrl": b2_ctrl}
    sample_info = {
        "B1_Ctrl": CohortSampleInfo(sample_id="B1_Ctrl", batch_id="B1", is_control=True),
        "B2_Ctrl": CohortSampleInfo(sample_id="B2_Ctrl", batch_id="B2", is_control=True),
    }

    cfg = BatchCorrectionConfig(enabled=True, method="cytonorm", n_clusters=2, n_quantiles=10)
    model = fit_cytonorm(cohort_dfs, sample_info, cfg)

    # DataFrame missing 'CD8'
    incomplete_df = pd.DataFrame({"CD4": rng.normal(10, 2, 100)})
    with pytest.raises(KeyError, match="missing 1 trained markers"):
        model.transform(incomplete_df, "B1")


def test_run_batch_preparation_existing_metadata_columns(tmp_path: Path):
    """Test that pooling handles DataFrames that already contain sample_id and batch_id columns."""
    rng = np.random.default_rng(42)
    df1 = pd.DataFrame({
        "sample_id": ["old_s1"] * 100,
        "batch_id": ["old_b1"] * 100,
        "CD4": rng.normal(10, 2, 100),
    })

    cohort_dfs = {"Sample_A": df1}
    sample_info = [CohortSampleInfo(sample_id="Sample_A", batch_id="New_Batch_1")]

    cfg = BatchPreparationConfig(
        batch_normalization=BatchCorrectionConfig(enabled=False, method="none"),
        downsampling=DownsampleConfig(enabled=True, target_events=50),
        output_dir=tmp_path / "metadata_out",
        export_pooled_cohort=True,
    )

    result = run_batch_preparation(
        cohort_dfs=cohort_dfs,
        sample_info=sample_info,
        config=cfg,
    )

    assert result.pooled_df is not None
    assert len(result.pooled_df) == 50
    assert result.pooled_df["sample_id"].iloc[0] == "Sample_A"
    assert result.pooled_df["batch_id"].iloc[0] == "New_Batch_1"


def test_run_batch_preparation_integration_multiday_quantile(tmp_path: Path):
    """Verify full end-to-end multi-day batch normalization and downsampling workflow."""
    rng = np.random.default_rng(42)
    df_day1 = pd.DataFrame({
        "FSC-A": rng.normal(50000, 5000, 200),
        "SSC-A": rng.normal(30000, 3000, 200),
        "CD4": rng.normal(15, 2, 200),
    })
    df_day2 = pd.DataFrame({
        "FSC-A": rng.normal(53000, 5000, 250),
        "SSC-A": rng.normal(32000, 3000, 250),
        "CD4": rng.normal(18, 2, 250),
    })

    cohort_dfs = {"Sample_D1": df_day1, "Sample_D2": df_day2}
    sample_info = [
        CohortSampleInfo(sample_id="Sample_D1", batch_id="Day1", is_control=False),
        CohortSampleInfo(sample_id="Sample_D2", batch_id="Day2", is_control=False),
    ]

    out_dir = tmp_path / "batch_out"
    batch_cfg = BatchPreparationConfig(
        downsampling=DownsampleConfig(enabled=True, mode="target_count", target_events=100, random_seed=42),
        batch_normalization=BatchCorrectionConfig(enabled=True, method="quantile", goal="median"),
        export_individual_fcs=False,
        export_pooled_cohort=True,
    )

    result = run_batch_preparation(
        cohort_dfs=cohort_dfs,
        sample_info=sample_info,
        config=batch_cfg,
        output_dir=out_dir,
    )

    assert result.pooled_df is not None
    assert len(result.pooled_df) == 200  # 100 per sample
    assert (out_dir / "tables/downsampling_summary.csv").is_file()
    assert (out_dir / "tables/batch_invariance_metrics.csv").is_file()
    assert (out_dir / "plots/step3_batch_density_comparison.png").is_file()
    assert (out_dir / "plots/step3_batch_metrics_reduction.png").is_file()
    assert (out_dir / "plots/step3_cohort_downsampling_balance.png").is_file()

