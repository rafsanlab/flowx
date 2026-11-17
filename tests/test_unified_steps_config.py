"""Tests for unified step runners, config builders, and CLI --config parity."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import flowkit as fk
import numpy as np
import pandas as pd
import pytest
import yaml

from flowx.batch.models import BatchPreparationConfig
from flowx.cli import build_parser, main as cli_main
from flowx.gating.cleanup import CleanUpGateConfig
from flowx.gating.runner import run_cleanup_step
from flowx.io.config import PipelineConfig
from flowx.io.parser import FCSData, parse_fcs
from flowx.math.compensation import CompensationConfig, run_compensation_step
from flowx.math.transforms import TransformConfig, run_transformation_step
from flowx.qc.models import TimeQCConfig
from flowx.qc.runner import run_time_qc_step


def _make_synthetic_df(n_events: int = 800, seed: int = 42, shift: float = 0.0) -> pd.DataFrame:
    """Create a synthetic flow cytometry DataFrame with scatter, time, and fluorescence columns."""
    rng = np.random.default_rng(seed)
    fsc_a = rng.normal(loc=110000.0, scale=15000.0, size=n_events).clip(20000.0, 240000.0)
    fsc_h = (fsc_a * rng.normal(loc=0.95, scale=0.04, size=n_events)).clip(15000.0, 240000.0)
    ssc_a = rng.normal(loc=55000.0, scale=12000.0, size=n_events).clip(5000.0, 220000.0)
    ssc_h = (ssc_a * rng.normal(loc=0.95, scale=0.04, size=n_events)).clip(4000.0, 220000.0)
    time_vals = np.linspace(0.0, 30.0, n_events)

    # Bimodal/skewed fluorescence distributions + viability low negative & small dead peak
    cd3 = np.concatenate([
        rng.normal(250.0 + shift, 80.0, n_events // 2),
        rng.normal(12000.0 + shift, 2500.0, n_events - n_events // 2),
    ])
    cd4 = np.concatenate([
        rng.normal(300.0 + shift, 90.0, n_events // 2),
        rng.normal(9000.0 + shift, 1800.0, n_events - n_events // 2),
    ])
    viability = rng.normal(180.0, 60.0, n_events)
    rng.shuffle(cd3)
    rng.shuffle(cd4)

    return pd.DataFrame(
        {
            "FSC-A": fsc_a,
            "FSC-H": fsc_h,
            "SSC-A": ssc_a,
            "SSC-H": ssc_h,
            "FITC-A": cd3,
            "PE-A": cd4,
            "Zombie-A": viability,
            "Time": time_vals,
        }
    )


def _make_synthetic_fcs_data(n_events: int = 800, seed: int = 42, with_spill: bool = True) -> FCSData:
    """Construct an in-memory FCSData instance for unit testing step runners."""
    df = _make_synthetic_df(n_events=n_events, seed=seed)
    pnn = list(df.columns)
    pns = ["", "", "", "", "CD3", "CD4", "LiveDead", "Time"]
    fl_cols = ["FITC-A", "PE-A", "Zombie-A"]
    spill_df = None
    if with_spill:
        spill_df = pd.DataFrame(
            [
                [1.0, 0.08, 0.0],
                [0.03, 1.0, 0.0],
                [0.0, 0.01, 1.0],
            ],
            index=fl_cols,
            columns=fl_cols,
        )
    return FCSData(
        filepath="synthetic_sample.fcs",
        filename="synthetic_sample.fcs",
        event_count=len(df),
        channel_count=len(pnn),
        pnn_labels=pnn,
        pns_labels=pns,
        detector_to_marker={"FITC-A": "CD3", "PE-A": "CD4", "Zombie-A": "LiveDead"},
        marker_to_detector={"CD3": "FITC-A", "CD4": "PE-A", "LiveDead": "Zombie-A"},
        metadata={"cyt": "LSRFortessa", "timestep": "0.01"},
        raw_events=df.to_numpy(),
        events_df=df,
        spillover_matrix=spill_df,
        spillover_channels=fl_cols,
        cytometer="LSRFortessa",
        channel_ranges={c: 262144.0 for c in pnn},
    )


def _write_synthetic_fcs(path: Path, n_events: int = 600, seed: int = 42, shift: float = 0.0) -> Path:
    """Write a valid synthetic .fcs file to disk via FlowKit Sample."""
    df = _make_synthetic_df(n_events=n_events, seed=seed, shift=shift)
    sample = fk.Sample(df, sample_id=path.stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.export(str(path.name), source="raw", directory=str(path.parent))
    return path


def test_pipeline_config_step_builders() -> None:
    """Verify PipelineConfig maps numbered/canonical sections into step-specific dataclasses."""
    raw: Dict[str, object] = {
        "input": {"fcs_file": "sample.fcs", "technology": "conventional"},
        "output": {"output_dir": "outputs/test_run"},
        "01_time_qc": {
            "enabled": True,
            "bin_width": 0.5,
            "enable_signal_drift": False,
            "trim_margins": False,
            "flow_rate_mad_factor": 4.0,
        },
        "02_compensation": {
            "enabled": True,
            "source": "embedded",
            "estimation_method": "autospill_regression",
            "apply_time_qc_to_controls": False,
            "voltage_mismatch_policy": "fallback_embedded",
            "voltage_tolerance_v": 10.0,
        },
        "03_cleanup_gating": {
            "enabled": True,
            "debris_method": "gmm",
            "debris_n_components": 3,
            "fsc_channel": "FSC-A",
            "ssc_channel": "SSC-A",
            "singlet_mode": "fsc_only",
            "singlet_tolerance": 0.35,
            "viability_channel": "Zombie-A",
            "viability_cutoff": 2.2,
            "dump_channel": "PE-A",
            "dump_cutoff": 3.1,
        },
        "04_transformation": {
            "method": "arcsinh",
            "cofactor": 150.0,
            "channels": ["FITC-A", "PE-A"],
        },
        "04_batch_preparation": {
            "enabled": True,
            "downsampling": {"enabled": True, "mode": "target_count", "target_events": 1200},
            "batch_normalization": {"enabled": True, "method": "quantile", "goal": "median", "n_quantiles": 51},
        },
    }

    cfg = PipelineConfig.from_dict(raw)

    tqc = cfg.to_time_qc_config()
    assert isinstance(tqc, TimeQCConfig)
    assert tqc.bin_width_sec == 0.5
    assert tqc.enable_signal_drift is False
    assert tqc.trim_margins is False
    assert tqc.flow_rate_mad_factor == 4.0

    comp = cfg.to_compensation_config()
    assert isinstance(comp, CompensationConfig)
    assert comp.estimation_method == "autospill_regression"
    assert comp.apply_time_qc_to_controls is False
    assert comp.voltage_mismatch_policy == "fallback_embedded"
    assert comp.voltage_tolerance_v == 10.0

    xform = cfg.to_transform_config()
    assert isinstance(xform, TransformConfig)
    assert xform.method == "arcsinh"
    assert xform.cofactor == 150.0
    assert xform.channels == ["FITC-A", "PE-A"]

    cleanup = cfg.to_cleanup_config()
    assert isinstance(cleanup, CleanUpGateConfig)
    assert cleanup.debris_n_components == 3
    assert cleanup.singlet_mode == "fsc_only"
    assert cleanup.singlet_tolerance == 0.35
    assert cleanup.viability_channel == "Zombie-A"
    assert cleanup.viability_cutoff == 2.2
    assert cleanup.dump_channel == "PE-A"
    assert cleanup.dump_cutoff == 3.1

    batch_cfg = cfg.to_batch_preparation_config()
    assert isinstance(batch_cfg, BatchPreparationConfig)
    assert batch_cfg.downsampling.target_events == 1200
    assert batch_cfg.batch_normalization.method == "quantile"
    assert batch_cfg.batch_normalization.n_quantiles == 51

    pipe, fcs_target = cfg.to_pipeline()
    assert fcs_target == "sample.fcs"
    assert pipe.transform_config.method == "arcsinh"
    assert pipe.time_qc_config.bin_width_sec == 0.5
    assert pipe.comp_config.estimation_method == "autospill_regression"
    assert pipe.spectral_config is None  # Must stay None unless spectral is indicated


def test_batch_preparation_config_supports_both_schemas() -> None:
    """Ensure BatchPreparationConfig.from_dict parses both standalone and 04_batch_preparation schemas."""
    pipe_schema = {
        "output": {"output_dir": "outputs/demo"},
        "04_batch_preparation": {
            "enabled": True,
            "assess_batch_effect": False,
            "batch_normalization": {"enabled": True, "method": "cytonorm", "n_clusters": 8},
            "downsampling": {"enabled": False, "target_events": 5000},
        },
    }
    cfg1 = BatchPreparationConfig.from_dict(pipe_schema)
    assert cfg1.assess_batch_effect is False
    assert cfg1.batch_normalization.n_clusters == 8
    assert cfg1.downsampling.enabled is False

    standalone_schema = {
        "batch_normalization": {"enabled": True, "method": "combat"},
        "downsampling": {"enabled": True, "mode": "min_sample"},
    }
    cfg2 = BatchPreparationConfig.from_dict(standalone_schema)
    assert cfg2.batch_normalization.method == "combat"
    assert cfg2.downsampling.mode == "min_sample"


def test_independent_step_runners_chain_cleanly(tmp_path: Path) -> None:
    """Run Steps 1 -> 2 -> 3 -> 4 (cleanup) sequentially via the canonical step runners."""
    fcs = _make_synthetic_fcs_data(n_events=600, with_spill=True)

    tqc_res = run_time_qc_step(
        fcs_data=fcs,
        config=TimeQCConfig(bin_width_sec=1.0, enable_signal_drift=True),
        generate_plots=False,
    )
    assert len(tqc_res.clean_df) > 0

    comp_res = run_compensation_step(
        fcs_data=fcs,
        working_df=tqc_res.clean_df,
        enable_compensation=True,
        compensation_source="embedded",
        technology="conventional",
        comp_config=CompensationConfig(),
        generate_plots=False,
        generate_nxn=False,
    )
    assert comp_res.comp_source_used == "embedded"
    assert len(comp_res.compensated_df) == len(tqc_res.clean_df)

    xform_res = run_transformation_step(
        fcs_data=fcs,
        df_events=comp_res.compensated_df,
        config=TransformConfig(method="arcsinh", cofactor=150.0, channels=["FITC-A", "PE-A", "Zombie-A"]),
        generate_plot=False,
    )
    assert set(["FITC-A", "PE-A", "Zombie-A"]).issubset(set(xform_res.channels_transformed))

    cleanup_res = run_cleanup_step(
        fcs_data=fcs,
        df_events=comp_res.compensated_df,
        df_transformed=xform_res.transformed_df,
        config=CleanUpGateConfig(viability_channel="Zombie-A", debris_n_components=2),
        output_plot_path=tmp_path / "cleanup.png",
        generate_plots=False,
        export_clean_fcs=True,
        clean_fcs_path=tmp_path / "clean_out.fcs",
    )
    assert cleanup_res.cleanup_result.total_events == len(tqc_res.clean_df)
    assert cleanup_res.clean_fcs_path is not None and Path(cleanup_res.clean_fcs_path).is_file()


def test_cli_subcommands_accept_config_flags() -> None:
    """Verify parser supports --config across all preprocessing step subcommands without conflicts."""
    parser = build_parser()
    for argv in [
        ["time-qc", "--config", "cfg.yaml", "--no-plot"],
        ["compensate", "--config", "cfg.yaml", "--no-plots"],
        ["unmix", "--config", "cfg.yaml", "--method", "ols"],
        ["transform", "--config", "cfg.yaml", "--cofactor", "200"],
        ["cleanup", "--config", "cfg.yaml", "--no-plot"],
        ["batch-prepare", "--config", "cfg.yaml", "--no-plot"],
        ["batch-check", "--config", "cfg.yaml"],
    ]:
        parsed = parser.parse_args(argv)
        assert parsed.config == "cfg.yaml"


def test_cli_step_subcommands_execute_from_config(tmp_path: Path) -> None:
    """Execute time-qc, transform, cleanup, and batch-check via --config on synthetic FCS files."""
    fcs_a = _write_synthetic_fcs(tmp_path / "samples" / "sample_a.fcs", n_events=500, seed=11)
    fcs_b = _write_synthetic_fcs(tmp_path / "samples" / "sample_b.fcs", n_events=500, seed=22, shift=400.0)

    meta_csv = tmp_path / "samples.csv"
    pd.DataFrame(
        [
            {"sample_id": "sample_a", "filename": fcs_a.name, "filepath": str(fcs_a), "condition": "Ctrl", "batch_id": "B1", "is_control": False},
            {"sample_id": "sample_b", "filename": fcs_b.name, "filepath": str(fcs_b), "condition": "Case", "batch_id": "B2", "is_control": False},
        ]
    ).to_csv(meta_csv, index=False)

    out_dir = tmp_path / "out"
    cfg_path = tmp_path / "step_cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "input": {
                    "fcs_file": str(fcs_a),
                    "metadata_file": str(meta_csv),
                },
                "output": {"output_dir": str(out_dir)},
                "01_time_qc": {"enabled": True, "bin_width": 5.0},
                "03_cleanup_gating": {"debris_n_components": 2, "viability_channel": "Zombie-A"},
                "04_transformation": {"method": "arcsinh", "cofactor": 150.0, "channels": ["FITC-A", "PE-A"]},
                "05_population_discovery": {"condition_column": "condition"},
            }
        ),
        encoding="utf-8",
    )

    assert cli_main(["time-qc", "--config", str(cfg_path), "--no-plot"]) == 0
    assert (out_dir / "01_time_qc").is_dir()

    assert cli_main(["transform", "--config", str(cfg_path)]) == 0
    assert (out_dir / "03_transformation" / f"{fcs_a.stem}_transformed.csv").is_file()

    assert cli_main(["cleanup", "--config", str(cfg_path), "--no-plot", "--no-compensation"]) == 0
    assert (out_dir / "03_cleanup_gating").is_dir()

    assert cli_main(["batch-check", "--config", str(cfg_path)]) == 0
    assert (out_dir / "04_batch_check" / "batch_necessity_report.md").is_file()


def test_metadata_auto_channel_and_marker_resolution(tmp_path: Path) -> None:
    """Role-based and 'auto' channel/marker resolution from markers.csv should expand cleanly and respect excludes."""
    from flowx.io.metadata import (
        resolve_marker_metadata,
        resolve_scatter_and_role_channels,
        select_markers_from_metadata,
    )
    from flowx.gating.runner import resolve_cleanup_channels_from_metadata
    from flowx.math.transforms import resolve_fluorescence_channels

    markers_csv = tmp_path / "markers.csv"
    pd.DataFrame(
        [
            {"channel_name": "FSC-A", "marker_name": "FSC-A", "fluorophore": "Scatter", "marker_class": "scatter", "cell_type_association": "All"},
            {"channel_name": "FSC-H", "marker_name": "FSC-H", "fluorophore": "Scatter", "marker_class": "scatter", "cell_type_association": "All"},
            {"channel_name": "SSC-A", "marker_name": "SSC-A", "fluorophore": "Scatter", "marker_class": "scatter", "cell_type_association": "All"},
            {"channel_name": "SSC-H", "marker_name": "SSC-H", "fluorophore": "Scatter", "marker_class": "scatter", "cell_type_association": "All"},
            {"channel_name": "Time", "marker_name": "Time", "fluorophore": "Time", "marker_class": "time", "cell_type_association": "All"},
            {"channel_name": "FITC-A", "marker_name": "CD3", "fluorophore": "FITC", "marker_class": "lineage", "cell_type_association": "T cells"},
            {"channel_name": "PE-A", "marker_name": "CD19", "fluorophore": "PE", "marker_class": "lineage", "cell_type_association": "B cells"},
            {"channel_name": "APC-A", "marker_name": "CD69", "fluorophore": "APC", "marker_class": "functional", "cell_type_association": "Activated"},
            {"channel_name": "BV421-A", "marker_name": "HLA-DR", "fluorophore": "BV421", "marker_class": "lineage;functional", "cell_type_association": "APCs"},
            {"channel_name": "Zombie-A", "marker_name": "Viability", "fluorophore": "Zombie", "marker_class": "viability", "cell_type_association": "Dead"},
            {"channel_name": "BV510-A", "marker_name": "Dump", "fluorophore": "BV510", "marker_class": "dump", "cell_type_association": "Exclude"},
        ]
    ).to_csv(markers_csv, index=False)

    fcs = _make_synthetic_fcs_data(n_events=100)
    cols = ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "Time", "FITC-A", "PE-A", "APC-A", "BV421-A", "Zombie-A", "BV510-A"]
    res = resolve_marker_metadata(
        markers_file=markers_csv,
        fcs_pnn=cols,
        fcs_pns=[""] * len(cols),
    )

    roles_map = resolve_scatter_and_role_channels(res, cols)
    assert roles_map["fsc_channel"] == "FSC-A"
    assert roles_map["viability_channel"] == "Zombie-A"
    assert roles_map["dump_channel"] == "BV510-A"

    clean_cfg = resolve_cleanup_channels_from_metadata(
        CleanUpGateConfig(
            fsc_channel="auto",
            ssc_channel="auto",
            fsc_height_channel="auto",
            ssc_height_channel="auto",
            viability_channel="auto",
            dump_channel="auto",
        ),
        cols,
        marker_metadata=res,
    )
    assert clean_cfg.viability_channel == "Zombie-A"
    assert clean_cfg.dump_channel == "BV510-A"
    assert clean_cfg.dump_enabled is True

    # Role-based clustering marker selection (only 'lineage' matches FITC-A, PE-A, BV421-A; include CD69, exclude CD19)
    selected = select_markers_from_metadata(
        res,
        available_columns=cols,
        explicit_markers="auto",
        marker_roles=["lineage"],
        include_markers=["CD69"],
        exclude_markers=["CD19"],
    )
    assert selected == ["FITC-A", "BV421-A", "APC-A"]

    # Transform resolution supports "auto" and excludes requested channel
    fcs_full = _make_synthetic_fcs_data(n_events=100)
    fcs_full.pnn_labels = ["FSC-A", "SSC-A", "FITC-A", "PE-A", "Zombie-A", "Time"]
    xform_chs = resolve_fluorescence_channels(
        fcs_full,
        explicit_channels="auto",
        exclude_channels=["PE-A"],
        marker_metadata=res,
    )
    assert "FITC-A" in xform_chs
    assert "PE-A" not in xform_chs


def test_config_auto_selectors_and_stats_granularity() -> None:
    """Verify PipelineConfig parses auto selectors, marker_roles, and stats_granularity."""
    raw = {
        "04_transformation": {"method": "logicle", "channels": "auto", "exclude_channels": ["Time"]},
        "05_population_discovery": {
            "enabled": True,
            "lineage_markers": "auto",
            "marker_roles": ["lineage", "functional"],
            "include_markers": ["CD3"],
            "exclude_markers": ["CD19"],
        },
        "06_statistical_analysis": {
            "stats_granularity": "both",
        },
    }
    cfg = PipelineConfig.from_dict(raw)
    cfg.validate()
    assert cfg.transform_channels == "auto"
    assert cfg.transform_exclude_channels == ["Time"]
    assert cfg.lineage_markers == "auto"
    assert cfg.marker_roles == ["lineage", "functional"]
    assert cfg.include_markers == ["CD3"]
    assert cfg.exclude_markers == ["CD19"]
    assert cfg.stats_granularity == "both"

    bad = PipelineConfig(stats_granularity="invalid_level")
    with pytest.raises(ValueError, match="stats_granularity"):
        bad.validate()


def test_cohort_step5_folder_split_stats_granularity_and_resolved_config(tmp_path: Path) -> None:
    """Verify Step 5 01_clusters/02_populations split, stats_granularity='both', and resolved_config.yaml."""
    from flowx.clustering.cohort_runner import run_cohort_population_analysis
    from flowx.pipeline import FlowCytometryPipeline

    rng = np.random.default_rng(7)
    cohort_dfs = {}
    meta_rows = []
    for idx, (sid, cond) in enumerate([("C1", "Ctrl"), ("C2", "Ctrl"), ("T1", "Case"), ("T2", "Case")]):
        shift = 0.0 if cond == "Ctrl" else 1.5
        df = pd.DataFrame(
            {
                "FITC-A": np.concatenate([rng.normal(1.0, 0.2, 80), rng.normal(4.0 + shift, 0.2, 80)]),
                "PE-A": np.concatenate([rng.normal(4.0, 0.2, 80), rng.normal(1.0, 0.2, 80)]),
                "APC-A": rng.normal(2.0 + (0.4 if cond == "Case" else 0.0), 0.3, 160),
            }
        )
        cohort_dfs[sid] = df
        meta_rows.append({"sample_id": sid, "condition": cond, "batch_id": f"B{(idx % 2) + 1}"})

    meta_df = pd.DataFrame(meta_rows)
    out_dir = tmp_path / "cohort_both"
    res = run_cohort_population_analysis(
        cohort_dfs=cohort_dfs,
        sample_metadata=meta_df,
        markers=["FITC-A", "PE-A", "APC-A"],
        n_clusters=3,
        auto_k=False,
        compute_umap=True,
        umap_events_per_sample=60,
        condition_column="condition",
        reference_group="Ctrl",
        target_group="Case",
        stats_granularity="both",
        cluster_name_map={"0": "T Cells", "1": "T Cells", "2": "Myeloid"},
        output_dir=out_dir,
        random_seed=42,
        generate_plots=True,
    )

    assert res.stats_granularity == "both"
    assert (out_dir / "05_population_discovery/plots/clusters/cluster_expression_heatmap.png").is_file()
    assert (out_dir / "05_population_discovery/plots/clusters/cluster_frequency_clustermap.png").is_file()
    assert (out_dir / "05_population_discovery/plots/clusters/cluster_qc_metrics.png").is_file()
    assert (out_dir / "05_population_discovery/plots/clusters/umap_clusters.png").is_file()
    assert (out_dir / "05_population_discovery/plots/populations/population_expression_heatmap.png").is_file()
    assert (out_dir / "05_population_discovery/plots/populations/population_frequency_clustermap.png").is_file()
    assert (out_dir / "05_population_discovery/plots/populations/umap_populations.png").is_file()
    assert (out_dir / "05_population_discovery/plots/cluster_to_population_bridge.png").is_file()
    assert (out_dir / "05_population_discovery/tables/cluster_to_population_map.csv").is_file()

    # Step 6 population-level AND cluster-level deliverables
    assert (out_dir / "06_statistical_analysis/tables/differential_abundance_summary.csv").is_file()
    assert (out_dir / "06_statistical_analysis/tables/clusters/differential_abundance_summary.csv").is_file()
    assert (out_dir / "06_statistical_analysis/plots/clusters/differential_abundance_forest.png").is_file()
    assert (out_dir / "06_statistical_analysis/plots/clusters/differential_abundance_volcano.png").is_file()

    # Run-level resolved_config.yaml audit writer
    fcs_a = _write_synthetic_fcs(tmp_path / "single.fcs", n_events=250, seed=5)
    pipe_out = tmp_path / "pipe_audit"
    pipe = FlowCytometryPipeline(
        output_dir=pipe_out,
        timestamp_dir=False,
        enable_discovery=False,
        lineage_markers="auto",
        stats_granularity="populations",
    )
    pipe_res = pipe.run(fcs_a)
    assert (pipe_out / "resolved_config.yaml").is_file() or (pipe_out / "resolved_config.json").is_file()
    assert "resolved_config" in pipe_res.generated_tables


def test_valid_utility_marker_classes_excluded(tmp_path: Path) -> None:
    """Utility roles (unused, index, qc, autofluorescence) validate and stay excluded from fluorescence/clustering."""
    from flowx.io.metadata import (
        VALID_MARKER_CLASSES,
        resolve_marker_metadata,
        select_markers_from_metadata,
    )
    from flowx.math.transforms import resolve_fluorescence_channels

    for role in ("unused", "index", "qc", "autofluorescence"):
        assert role in VALID_MARKER_CLASSES

    markers_csv = tmp_path / "markers.csv"
    markers_csv.write_text(
        "\n".join(
            [
                "channel_name,marker_name,fluorophore,clone_or_catalog,marker_class,cell_type_association,provenance_source",
                "FSC-A,FSC-A,Scatter,NA,scatter,All Events,FCS $P1N",
                "Time,Time,Time,NA,time,All Events,FCS $P2N",
                "AF-A,AF-A,Autofluorescence,NA,autofluorescence,Background,FCS $P3N",
                "PeacoQC,PeacoQC,QC,NA,qc,Quality Flag,PeacoQC",
                "Original_ID,Original_ID,Index,NA,index,Event Index,Indexed",
                "BUV737-A,Empty,BUV737,NA,unused,Unstained Channel,Panel",
                "FITC-A,CD3,FITC,NA,lineage,T Cells,FCS $P7S",
                "PE-A,PD1,PE,NA,functional,Activated T Cells,FCS $P8S",
            ]
        ),
        encoding="utf-8",
    )

    mres = resolve_marker_metadata(markers_file=markers_csv)
    cols = ["FSC-A", "Time", "AF-A", "PeacoQC", "Original_ID", "BUV737-A", "FITC-A", "PE-A"]
    df = pd.DataFrame(100.0, index=[0, 1, 2], columns=cols)

    fl = resolve_fluorescence_channels(df, marker_metadata=mres)
    assert fl == ["FITC-A", "PE-A"]

    lin = select_markers_from_metadata(mres, cols, explicit_markers="auto", marker_roles=["lineage"])
    assert lin == ["FITC-A"]


def test_config_roundtrip_and_step_forwarding(tmp_path: Path) -> None:
    """PipelineConfig preserves files_list, 04_transformation, comp tuning, stats methods, and export gates across round-trips."""
    raw = {
        "input": {
            "fcs_file": str(tmp_path),
            "files_list": [str(tmp_path / "s1.fcs"), str(tmp_path / "s2.fcs")],
            "technology": "spectral",
        },
        "analysis": {
            "enable_qc": True,
            "enable_discovery": True,
            "numbered_steps": True,
            "seed": 17,
        },
        "02_compensation": {
            "enabled": True,
            "source": "controls",
            "min_control_events": 250,
            "pmt_gain_exponent_alpha": 6.5,
            "sample_nxn_gating_mode": "scatter",
            "unmixing_method": "ols",
            "autofluorescence_mode": "multi",
        },
        "04_transformation": {
            "enabled": True,
            "method": "arcsinh",
            "cofactor": 2000.0,
            "channels": "auto",
            "exclude_channels": ["AF-A"],
        },
        "05_population_discovery": {
            "enabled": True,
            "lineage_markers": "auto",
            "marker_roles": ["lineage"],
            "stratified_density_contours": False,
            "cluster_palette": "viridis",
        },
        "06_statistical_analysis": {
            "enabled": True,
            "da_test_method": "welch_ttest",
            "ds_test_method": "t_test",
            "fdr_threshold": 0.1,
            "log2fc_threshold": 0.5,
            "generate_abundance_boxplots": False,
            "generate_volcano_plot": False,
            "generate_forest_plot": True,
            "generate_ds_heatmap": True,
        },
        "exports": {
            "export_augmented_fcs": False,
            "export_flowjo_wsp": False,
            "export_gating_ml": True,
            "generate_miflowcyt_report": False,
        },
    }
    cfg = PipelineConfig.from_dict(raw)
    cfg.validate()

    rt = PipelineConfig.from_dict(cfg.to_dict())
    assert rt.files_list == raw["input"]["files_list"]
    assert rt.technology == "spectral"
    assert rt.transform_type == "arcsinh"
    assert rt.da_test_method == "welch_ttest"
    assert rt.ds_test_method == "t_test"
    assert rt.fdr_threshold == pytest.approx(0.1)
    assert rt.log2fc_threshold == pytest.approx(0.5)
    assert rt.stratified_density_contours is False
    assert rt.export_flowjo_wsp is False
    assert rt.export_gating_ml is True
    assert rt.generate_miflowcyt_report is False

    comp_cfg = rt.to_compensation_config()
    assert comp_cfg.min_control_events == 250
    assert comp_cfg.pmt_gain_exponent_alpha == pytest.approx(6.5)
    assert comp_cfg.sample_nxn_gating_mode == "scatter"

    unmix_cfg = rt.to_unmixing_config()
    assert unmix_cfg.method == "ols"
    assert unmix_cfg.autofluorescence_mode == "multi"

    pipe, input_fcs = rt.to_pipeline()
    assert pipe.da_test_method == "welch_ttest"
    assert pipe.ds_test_method == "t_test"
    assert pipe.fdr_threshold == pytest.approx(0.1)
    assert pipe.export_flowjo_wsp is False
    assert pipe.export_gating_ml is True
    assert pipe.generate_miflowcyt_report is False
    assert str(tmp_path) == input_fcs


def test_cli_adjustable_step_flags_parsing() -> None:
    """Verify newly wired adjustable parameters are accepted by CLI subcommands."""
    parser = build_parser()

    ns_pipe = parser.parse_args(
        [
            "pipeline",
            "sample.fcs",
            "--technology",
            "spectral",
            "--no-time-drift",
            "--no-time-margins",
            "--voltage-mismatch-policy",
            "strict",
            "--unmix-method",
            "ols",
            "--af-mode",
            "multi",
            "--transform-channels",
            "auto",
            "--debris-n-components",
            "2",
            "--batch-goal",
            "mean",
            "--no-batch-check",
            "--da-method",
            "welch_ttest",
            "--ds-method",
            "t_test",
            "--fdr-threshold",
            "0.1",
            "--log2fc-threshold",
            "0.75",
            "--no-flowjo-wsp",
            "--no-gating-ml",
            "--no-miflowcyt-report",
        ]
    )
    assert ns_pipe.technology == "spectral"
    assert ns_pipe.no_time_drift is True
    assert ns_pipe.no_time_margins is True
    assert ns_pipe.voltage_mismatch_policy == "strict"
    assert ns_pipe.unmix_method == "ols"
    assert ns_pipe.af_mode == "multi"
    assert ns_pipe.debris_n_components == 2
    assert ns_pipe.batch_goal == "mean"
    assert ns_pipe.no_batch_check is True
    assert ns_pipe.da_test_method == "welch_ttest"
    assert ns_pipe.ds_test_method == "t_test"
    assert ns_pipe.export_flowjo_wsp is False
    assert ns_pipe.export_gating_ml is False
    assert ns_pipe.generate_miflowcyt_report is False

    ns_xform = parser.parse_args(["transform", "a.fcs", "--channel", "FITC-A", "--exclude-channels", "AF-A", "--cofactor", "2000"])
    assert ns_xform.channel == "FITC-A"
    assert ns_xform.exclude_channels == "AF-A"
    assert ns_xform.cofactor == pytest.approx(2000.0)

    ns_clean = parser.parse_args(["cleanup", "a.fcs", "--debris-n-components", "2", "--dump-channel", "APC-Cy7-A", "--dump-cutoff", "1.5"])
    assert ns_clean.debris_n_components == 2
    assert ns_clean.dump_channel == "APC-Cy7-A"
    assert ns_clean.dump_cutoff == pytest.approx(1.5)


def test_config_run_all_twenty_datasets_valid_and_parity(synthetic_conventional_dataset_dir: Path) -> None:
    """Verify dataset YAML configs load, validate, round-trip, and build FlowCytometryPipeline."""
    root_dir = synthetic_conventional_dataset_dir.parent.parent.parent
    cfg_root = root_dir / "configs" / "config_run"
    yaml_files = sorted(list((cfg_root / "conventional").glob("*.yaml")) + list((cfg_root / "spectral").glob("*.yaml")))
    assert len(yaml_files) >= 1

    for yml in yaml_files:
        cfg = PipelineConfig.load(yml)
        cfg.validate()
        rt = PipelineConfig.from_dict(cfg.to_dict())
        rt.validate()
        pipe, _ = rt.to_pipeline()
        assert pipe.enable_qc is True
        assert pipe.enable_discovery is True
        assert Path(cfg.metadata_file).is_file()
        assert Path(cfg.markers_file).is_file()


def test_cohort_runner_custom_stats_and_export_toggles(tmp_path: Path) -> None:
    """Verify cohort runner respects da_test_method, ds_test_method, and FlowJo/Gating-ML/MIFlowCyt toggles."""
    from flowx.clustering.cohort_runner import run_cohort_population_analysis

    dfs = {}
    meta_rows = []
    for i, (cond, shift) in enumerate([("Ctrl", 0.0), ("Ctrl", 50.0), ("Case", 2000.0), ("Case", 2200.0)]):
        sid = f"S{i+1}"
        df = _make_synthetic_df(n_events=300, seed=100 + i, shift=shift)
        df["APC-A"] = df["FITC-A"] * 0.6 + 200.0
        dfs[sid] = df
        meta_rows.append({"sample_id": sid, "condition": cond, "batch_id": "B1", "is_control": False})
    meta_df = pd.DataFrame(meta_rows)

    out_dir = tmp_path / "cohort_custom"
    res = run_cohort_population_analysis(
        cohort_dfs=dfs,
        sample_metadata=meta_df,
        markers=["FITC-A", "PE-A", "APC-A"],
        n_clusters=3,
        auto_k=False,
        compute_umap=False,
        condition_column="condition",
        reference_group="Ctrl",
        target_group="Case",
        da_test_method="welch_ttest",
        ds_test_method="t_test",
        fdr_threshold=0.25,
        log2fc_threshold=0.1,
        stratified_density_contours=False,
        generate_volcano_plot=False,
        generate_forest_plot=False,
        generate_abundance_boxplots=False,
        generate_ds_heatmap=False,
        export_prism=False,
        export_augmented_fcs=False,
        export_flowjo_wsp=False,
        export_gating_ml=False,
        generate_miflowcyt_report=False,
        output_dir=out_dir,
        random_seed=42,
        generate_plots=True,
    )
    assert res.da_result is not None
    assert res.ds_result is not None
    assert not (out_dir / "07_exports" / "flowjo_workspace.wsp").exists()
    assert not (out_dir / "07_exports" / "gating_ml_v2.xml").exists()
    assert not (out_dir / "07_exports" / "MIFlowCyt_Cohort_Report.md").exists()


def test_cmd_pipeline_files_list_cohort_uses_unified_batch_prep(tmp_path: Path) -> None:
    """When config.files_list has >1 FCS files, CLI pipeline routes through unified _run_cohort & 04_batch_preparation."""
    f1 = _write_synthetic_fcs(tmp_path / "S1.fcs", n_events=260, seed=11)
    f2 = _write_synthetic_fcs(tmp_path / "S2.fcs", n_events=260, seed=12)
    meta_csv = tmp_path / "samples.csv"
    pd.DataFrame(
        [
            {"sample_id": "S1", "filename": "S1.fcs", "condition": "Ctrl", "batch_id": "B1", "is_control": "false"},
            {"sample_id": "S2", "filename": "S2.fcs", "condition": "Case", "batch_id": "B1", "is_control": "false"},
        ]
    ).to_csv(meta_csv, index=False)

    out_dir = tmp_path / "out_fl"
    cfg_path = tmp_path / "pipe_fl.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "input": {
                    "files_list": [str(f1), str(f2)],
                    "metadata_file": str(meta_csv),
                },
                "analysis": {"enable_qc": True, "enable_discovery": False, "numbered_steps": True},
                "01_time_qc": {"enabled": False},
                "02_compensation": {"enabled": False, "source": "none"},
                "03_cleanup_gating": {"enabled": True, "enable_viability": False},
                "04_batch_preparation": {"enabled": True, "mode": "min_sample", "target_events": 120},
                "05_population_discovery": {"enabled": False},
                "output": {"output_dir": str(out_dir), "timestamp_dir": False, "overwrite": True},
            }
        ),
        encoding="utf-8",
    )

    rc = cli_main(["pipeline", "--config", str(cfg_path)])
    assert rc == 0
    # Unified _run_cohort creates 04_batch_preparation and config.yaml
    assert (out_dir / "04_batch_preparation").is_dir()
    assert (out_dir / "config.yaml").is_file()


def test_overwrite_disables_timestamp_and_cleans_stale_files(tmp_path: Path) -> None:
    """Verify overwrite=True overrides timestamp_dir=True and removes stale files from output_dir."""
    from flowx.cli import main as cli_main
    from flowx.pipeline import FlowCytometryPipeline

    fcs_a = _write_synthetic_fcs(tmp_path / "sample_ovw.fcs", n_events=220, seed=11)
    out_dir = tmp_path / "ovw_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stale_file = out_dir / "stale_artifact.txt"
    stale_file.write_text("old", encoding="utf-8")

    pipe = FlowCytometryPipeline(
        output_dir=out_dir,
        timestamp_dir=True,
        overwrite=True,
        enable_qc=False,
        enable_compensation=False,
        enable_discovery=False,
    )
    res = pipe.run(fcs_a)
    assert Path(res.run_dir).resolve() == out_dir.resolve()
    assert not stale_file.exists()

    # Also verify CLI --overwrite overrides default timestamp_dir=True
    cli_out = tmp_path / "cli_ovw_out"
    cli_out.mkdir(parents=True, exist_ok=True)
    cli_stale = cli_out / "old_run_leftover.txt"
    cli_stale.write_text("stale", encoding="utf-8")

    rc = cli_main([
        "pipeline",
        str(fcs_a),
        "-o",
        str(cli_out),
        "--overwrite",
        "--no-discovery",
    ])
    assert rc == 0
    assert not cli_stale.exists()
    assert (cli_out / "pipeline.log").is_file()


def test_config_roundtrip_stats_marker_roles(tmp_path: Path) -> None:
    """Verify 06_statistical_analysis.marker_roles parses, serializes, and maps to FlowCytometryPipeline."""
    raw_cfg = {
        "05_population_discovery": {
            "marker_roles": ["lineage"],
        },
        "06_statistical_analysis": {
            "stats_granularity": "populations",
            "marker_roles": ["functional"],
        },
    }
    cfg = PipelineConfig.from_dict(raw_cfg)
    assert cfg.marker_roles == ["lineage"]
    assert cfg.stats_marker_roles == ["functional"]
    assert cfg.to_dict()["06_statistical_analysis"]["marker_roles"] == ["functional"]

    pipe, _ = cfg.to_pipeline()
    assert pipe.stats_marker_roles == ["functional"]




