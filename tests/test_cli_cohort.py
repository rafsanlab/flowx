"""Unit tests for FlowX CLI Cohort Population Discovery and Statistical Analysis commands."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from flowx.cli import build_parser, cmd_cohort, main


def test_cli_parser_cohort_arguments():
    """Verify that cohort subcommand and all Phase 4 & 5 arguments are parsed correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "cohort",
        "sample1.fcs",
        "sample2.fcs",
        "--metadata-file", "meta.csv",
        "--markers", "CD3,CD4,CD8",
        "--method", "som",
        "-k", "8",
        "--auto-k",
        "--k-min", "3",
        "--k-max", "12",
        "--umap",
        "--umap-events", "500",
        "--condition-column", "treatment",
        "--reference-group", "vehicle",
        "--target-group", "drug_A",
        "--paired-column", "patient_id",
        "--cluster-name-map", '{"0": "T_Cells"}',
        "--export-augmented-fcs",
        "--export-prism",
        "-o", "outputs/test_cohort",
        "--seed", "123",
    ])

    assert args.command == "cohort"
    assert args.targets == ["sample1.fcs", "sample2.fcs"]
    assert args.metadata_file == "meta.csv"
    assert args.markers == "CD3,CD4,CD8"
    assert args.clustering_method == "som"
    assert args.n_clusters == 8
    assert args.auto_k is True
    assert args.k_min == 3
    assert args.k_max == 12
    assert args.compute_umap is True
    assert args.umap_events_per_sample == 500
    assert args.condition_column == "treatment"
    assert args.reference_group == "vehicle"
    assert args.target_group == "drug_A"
    assert args.paired_column == "patient_id"
    assert args.cluster_name_map == '{"0": "T_Cells"}'
    assert args.export_augmented_fcs is True
    assert args.export_prism is True
    assert args.output_dir == "outputs/test_cohort"
    assert args.seed == 123


def test_cmd_cohort_execution_with_csv(tmp_path: Path):
    """Test full cmd_cohort execution with synthetic CSV samples and metadata."""
    rng = np.random.default_rng(42)

    # Create 4 synthetic samples
    sample_files = []
    for s_name in ["C1", "C2", "T1", "T2"]:
        shift = 3.0 if s_name.startswith("T") else 0.0
        c1 = np.column_stack([rng.normal(10, 1, 100), rng.normal(5, 1, 100)])
        c2 = np.column_stack([rng.normal(2, 1, 100), rng.normal(8 + shift, 1, 100)])
        df = pd.DataFrame(np.vstack([c1, c2]), columns=["CD3", "CD4"])
        fp = tmp_path / f"{s_name}.csv"
        df.to_csv(fp, index=False)
        sample_files.append(str(fp))

    meta_df = pd.DataFrame({
        "sample_id": ["C1", "C2", "T1", "T2"],
        "condition": ["Control", "Control", "Treated", "Treated"],
    })
    meta_path = tmp_path / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    out_dir = tmp_path / "cohort_out"

    parser = build_parser()
    cli_args = parser.parse_args([
        "cohort",
        *sample_files,
        "--metadata-file", str(meta_path),
        "--markers", "CD3,CD4",
        "-k", "4",
        "--condition-column", "condition",
        "--reference-group", "Control",
        "--target-group", "Treated",
        "--umap-events", "100",
        "-o", str(out_dir),
        "--seed", "42",
    ])

    ret = cmd_cohort(cli_args)
    assert ret == 0

    # Verify structured deliverables
    assert (out_dir / "05_population_discovery/tables/population_counts.csv").is_file()
    assert (out_dir / "05_population_discovery/tables/population_frequencies_viable.csv").is_file()
    assert (out_dir / "05_population_discovery/tables/cluster_to_population_map.csv").is_file()
    assert (out_dir / "05_population_discovery/plots/populations/population_frequency_clustermap.png").is_file()
    assert (out_dir / "05_population_discovery/plots/cluster_to_population_bridge.png").is_file()
    assert (out_dir / "06_statistical_analysis/tables/differential_abundance_summary.csv").is_file()
    assert (out_dir / "exports/workspaces/flowjo_workspace.wsp").is_file()
    assert (out_dir / "exports/workspaces/gating_ml_2.0.xml").is_file()
    assert (out_dir / "exports/reports/miflowcyt_analysis_report.md").is_file()


def test_cmd_cohort_cluster_name_map(tmp_path: Path):
    """Test cmd_cohort with cluster name map file."""
    rng = np.random.default_rng(99)
    sample_files = []
    for s_name in ["S1", "S2"]:
        data = rng.normal(5, 1, (120, 2))
        df = pd.DataFrame(data, columns=["Marker1", "Marker2"])
        fp = tmp_path / f"{s_name}.csv"
        df.to_csv(fp, index=False)
        sample_files.append(str(fp))

    map_file = tmp_path / "name_map.json"
    map_file.write_text(json.dumps({"0": "Lineage_Alpha", "1": "Lineage_Beta"}), encoding="utf-8")

    out_dir = tmp_path / "cohort_named_out"
    parser = build_parser()
    cli_args = parser.parse_args([
        "population",  # testing alias
        *sample_files,
        "--markers", "Marker1,Marker2",
        "-k", "4",
        "--cluster-name-map", str(map_file),
        "-o", str(out_dir),
    ])

    ret = cmd_cohort(cli_args)
    assert ret == 0

    freq_df = pd.read_csv(out_dir / "05_population_discovery/tables/population_frequencies_viable.csv", index_col=0)
    assert "Lineage_Alpha" in freq_df.columns


def test_cli_parser_rf7_flags_and_config_precedence():
    """Verify RF-7 CLI flags on pipeline/cohort subcommands and None defaults preserving YAML config."""
    parser = build_parser()

    pipe_args = parser.parse_args([
        "pipeline",
        "sample.fcs",
        "--markers-file", "meta/markers.csv",
        "--cell-signatures", "meta/sigs.json",
        "--cluster-name-map", '{"0": "CD4_T"}',
        "--clustering-method", "phenograph",
        "--no-auto-k",
        "--auto-k-method", "silhouette",
        "--k-min", "3",
        "--k-max", "9",
        "--consensus-subsample", "8000",
        "--consensus-iterations", "15",
        "--umap-events-per-sample", "750",
        "--reference-group", "Ref",
        "--target-group", "Tgt",
        "--paired-column", "donor",
    ])
    assert pipe_args.markers_file == "meta/markers.csv"
    assert pipe_args.cell_signatures == "meta/sigs.json"
    assert pipe_args.cluster_name_map == '{"0": "CD4_T"}'
    assert pipe_args.clustering_method == "phenograph"
    assert pipe_args.auto_k is False
    assert pipe_args.auto_k_method == "silhouette"
    assert pipe_args.k_min == 3
    assert pipe_args.k_max == 9
    assert pipe_args.consensus_subsample == 8000
    assert pipe_args.consensus_iterations == 15
    assert pipe_args.umap_events_per_sample == 750
    assert pipe_args.reference_group == "Ref"
    assert pipe_args.target_group == "Tgt"
    assert pipe_args.paired_column == "donor"

    cohort_cfg_only = parser.parse_args(["cohort", "a.fcs", "b.fcs", "--config", "cfg.yaml"])
    for attr in (
        "clustering_method",
        "n_clusters",
        "auto_k",
        "auto_k_method",
        "k_min",
        "k_max",
        "compute_umap",
        "umap_events_per_sample",
        "condition_column",
        "export_augmented_fcs",
        "export_prism",
        "output_dir",
        "seed",
        "markers_file",
        "cell_signatures",
    ):
        assert getattr(cohort_cfg_only, attr) is None

    cohort_explicit = parser.parse_args([
        "cohort",
        "a.fcs",
        "b.fcs",
        "--markers-metadata", "meta/markers.csv",
        "--cell-signatures", "meta/sigs.yaml",
        "--no-auto-k",
        "--auto-k-method", "consensus",
        "--consensus-subsample", "9000",
        "--consensus-iterations", "20",
    ])
    assert cohort_explicit.markers_file == "meta/markers.csv"
    assert cohort_explicit.cell_signatures == "meta/sigs.yaml"
    assert cohort_explicit.auto_k is False
    assert cohort_explicit.auto_k_method == "consensus"
    assert cohort_explicit.consensus_subsample == 9000
    assert cohort_explicit.consensus_iterations == 20


def test_cmd_cohort_markers_and_config_n_clusters_null(tmp_path: Path):
    """Test cmd_cohort with YAML config n_clusters=null, markers.csv, and custom cell_signatures."""
    import yaml

    rng = np.random.default_rng(7)
    sample_files = []
    for idx, s_name in enumerate(["C1", "C2", "T1", "T2"]):
        shift = 3.5 if s_name.startswith("T") else 0.0
        n1 = 60 + idx * 12
        n2 = 160 - n1
        c1 = np.column_stack([rng.normal(9.0, 0.6, n1), rng.normal(9.0, 0.6, n1)])
        c2 = np.column_stack([rng.normal(2.0, 0.6, n2), rng.normal(2.0 + shift * 0.1, 0.6, n2)])
        df = pd.DataFrame(np.vstack([c1, c2]), columns=["B710-A", "R660-A"])
        fp = tmp_path / f"{s_name}.csv"
        df.to_csv(fp, index=False)
        sample_files.append(str(fp))

    samples_csv = tmp_path / "samples.csv"
    pd.DataFrame({
        "sample_id": ["C1", "C2", "T1", "T2"],
        "condition": ["Control", "Control", "Treated", "Treated"],
    }).to_csv(samples_csv, index=False)

    markers_csv = tmp_path / "markers.csv"
    pd.DataFrame({
        "channel_name": ["B710-A", "R660-A"],
        "marker_name": ["CD3", "CD4"],
        "fluorophore": ["PerCP-Cy5.5", "APC"],
        "clone_or_catalog": ["OKT3", "SK3"],
        "marker_class": ["lineage", "lineage"],
        "cell_type_association": ["T Cells", "Helper T Cells"],
        "provenance_source": ["unit_test", "unit_test"],
    }).to_csv(markers_csv, index=False)

    sigs_json = tmp_path / "sigs.json"
    sigs_json.write_text(
        json.dumps({"Custom_CD4_Helper": {"positive": ["CD3", "CD4"], "negative": []}}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "cohort_cfg_out"
    cfg_path = tmp_path / "cohort.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "input": {"metadata_file": str(samples_csv), "markers_file": str(markers_csv)},
            "05_population_discovery": {
                "clustering_method": "som",
                "n_clusters": None,
                "auto_k": True,
                "auto_k_method": "davies_bouldin",
                "k_min": 2,
                "k_max": 4,
                "umap_events_per_sample": 80,
            },
            "06_statistical_analysis": {
                "condition_column": "condition",
                "reference_group": "Control",
                "target_group": "Treated",
            },
            "output": {"output_dir": str(out_dir)},
        }),
        encoding="utf-8",
    )

    parser = build_parser()
    cli_args = parser.parse_args([
        "cohort",
        *sample_files,
        "--config", str(cfg_path),
        "--cell-signatures", str(sigs_json),
        "--seed", "7",
    ])
    ret = cmd_cohort(cli_args)
    assert ret == 0

    audit_path = out_dir / "05_population_discovery/tables/cluster_annotation_audit.csv"
    thresh_path = out_dir / "05_population_discovery/tables/cluster_marker_thresholds.csv"
    assert audit_path.is_file()
    assert thresh_path.is_file()
    audit_df = pd.read_csv(audit_path)
    assert not audit_df.empty
    assert "Custom_CD4_Helper" in audit_df["Assigned Label"].tolist()
    assert (out_dir / "06_statistical_analysis/tables/differential_abundance_summary.csv").is_file()

