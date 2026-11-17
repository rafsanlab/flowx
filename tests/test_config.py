"""Unit tests for pipeline configuration loading, validation, and parameter files."""

import json
from pathlib import Path

import pytest
import yaml

from flowx.io.config import PipelineConfig
from flowx.pipeline import FlowCytometryPipeline


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.fcs_file is None
    assert cfg.output_dir == "outputs"
    assert cfg.gating_mode == "som"
    assert cfg.cluster_parent == "viable"
    assert cfg.qc_method == "clustering"
    assert cfg.adaptive_gating is False
    assert cfg.legend_mode == "centroid"
    assert cfg.seed == 42
    assert cfg.timestamp_dir is True
    assert cfg.overwrite is False
    assert cfg.workspace_file is None
    assert cfg.marker_mapping is None
    assert cfg.debris_method == "gmm"
    assert cfg.singlet_mode == "dual"
    assert cfg.singlet_method == "gmm_ratio"
    assert cfg.singlet_tolerance == 0.25
    assert cfg.viability_cutoff is None
    assert cfg.enable_viability is True
    assert cfg.export_clean_fcs is False


def test_pipeline_config_from_flat_dict():
    data = {
        "fcs_file": "data/sample.fcs",
        "output_dir": "outputs/custom",
        "gating_mode": "compare",
        "cluster_parent": "singlets",
        "adaptive_gating": True,
        "legend_mode": "external",
        "seed": 99,
        "no_timestamp": True,
        "overwrite": True,
        "marker_mapping": {"CD3": "FL1-A"},
    }
    cfg = PipelineConfig.from_dict(data)
    assert cfg.fcs_file == "data/sample.fcs"
    assert cfg.output_dir == "outputs/custom"
    assert cfg.gating_mode == "compare"
    assert cfg.cluster_parent == "singlets"
    assert cfg.adaptive_gating is True
    assert cfg.legend_mode == "external"
    assert cfg.seed == 99
    assert cfg.timestamp_dir is False
    assert cfg.overwrite is True
    assert cfg.marker_mapping == {"CD3": "FL1-A"}


def test_pipeline_config_from_nested_dict():
    data = {
        "input": {
            "fcs_file": "data/my_tube.fcs",
            "marker_mapping": {"CD4": "FL2-A"},
        },
        "analysis": {
            "gating_mode": "som",
            "cluster_parent": "cells",
            "adaptive_gating": False,
            "seed": 123,
            "workspace_file": "workspace.json",
        },
        "visualization": {
            "legend_mode": "external",
        },
        "output": {
            "output_dir": "outputs/run_a",
            "timestamp_dir": False,
            "overwrite": True,
        },
    }
    cfg = PipelineConfig.from_dict(data)
    assert cfg.fcs_file == "data/my_tube.fcs"
    assert cfg.marker_mapping == {"CD4": "FL2-A"}
    assert cfg.gating_mode == "som"
    assert cfg.cluster_parent == "cells"
    assert cfg.adaptive_gating is False
    assert cfg.seed == 123
    assert cfg.workspace_file == "workspace.json"
    assert cfg.legend_mode == "external"
    assert cfg.output_dir == "outputs/run_a"
    assert cfg.timestamp_dir is False
    assert cfg.overwrite is True


def test_pipeline_config_validation():
    PipelineConfig(gating_mode="som").validate()

    # flowsom is rejected (flowsom backward-compatibility removed, strictly use som)
    with pytest.raises(ValueError, match="Invalid gating_mode"):
        PipelineConfig(gating_mode="flowsom").validate()

    with pytest.raises(ValueError, match="Invalid gating_mode"):
        PipelineConfig(gating_mode="invalid_mode").validate()

    with pytest.raises(ValueError, match="Invalid legend_mode"):
        PipelineConfig(legend_mode="floating").validate()

    with pytest.raises(ValueError, match="Invalid debris_method"):
        PipelineConfig(debris_method="unsupported_method").validate()

    with pytest.raises(ValueError, match="Invalid singlet_mode"):
        PipelineConfig(singlet_mode="triple").validate()

    with pytest.raises(ValueError, match="Invalid singlet_method"):
        PipelineConfig(singlet_method="unknown_ratio").validate()


def test_pipeline_config_cleanup_nested_and_flat():
    nested_data = {
        "cleanup": {
            "debris_method": "manual",
            "singlet_mode": "single",
            "singlet_method": "envelope",
            "singlet_tolerance": 0.35,
            "viability_cutoff": 0.42,
            "enable_viability": False,
            "export_clean_fcs": True,
        }
    }
    cfg = PipelineConfig.from_dict(nested_data)
    assert cfg.debris_method == "manual"
    assert cfg.singlet_mode == "single"
    assert cfg.singlet_method == "envelope"
    assert cfg.singlet_tolerance == 0.35
    assert cfg.viability_cutoff == 0.42
    assert cfg.enable_viability is False
    assert cfg.export_clean_fcs is True

    # Test conversion to pipeline forwards cleanup parameters
    pipeline, _ = cfg.to_pipeline()
    assert pipeline.debris_method == "manual"
    assert pipeline.singlet_mode == "single"
    assert pipeline.singlet_method == "envelope"
    assert pipeline.singlet_tolerance == 0.35
    assert pipeline.viability_cutoff == 0.42
    assert pipeline.enable_viability is False
    assert pipeline.export_clean_fcs is True


def test_cli_cleanup_and_pipeline_parsing():
    from flowx.cli import build_parser

    parser = build_parser()

    # Test cleanup subcommand parsing with new options
    args_cleanup = parser.parse_args([
        "cleanup",
        "data/sample.fcs",
        "--debris-method", "gmm",
        "--singlet-mode", "single",
        "--singlet-method", "gmm_ratio",
        "--fsc-channel", "FS INT",
        "--ssc-channel", "SS INT",
        "--fsc-height", "FS PEAK",
        "--ssc-height", "SS PEAK",
        "--viability-cutoff", "0.45",
        "--export-clean-fcs",
    ])
    assert args_cleanup.debris_method == "gmm"
    assert args_cleanup.singlet_mode == "single"
    assert args_cleanup.singlet_method == "gmm_ratio"
    assert args_cleanup.fsc_channel == "FS INT"
    assert args_cleanup.ssc_channel == "SS INT"
    assert args_cleanup.fsc_height_channel == "FS PEAK"
    assert args_cleanup.ssc_height_channel == "SS PEAK"
    assert args_cleanup.viability_cutoff == 0.45
    assert args_cleanup.export_clean_fcs is True

    # Test pipeline subcommand parsing with cleanup options
    args_pipe = parser.parse_args([
        "pipeline",
        "data/sample.fcs",
        "--debris-method", "gmm",
        "--singlet-mode", "one_time",
        "--singlet-method", "envelope",
        "--viability-cutoff", "0.52",
        "--no-viability",
        "--export-clean-fcs",
    ])
    assert args_pipe.debris_method == "gmm"
    assert args_pipe.singlet_mode == "one_time"
    assert args_pipe.singlet_method == "envelope"
    assert args_pipe.viability_cutoff == 0.52
    assert args_pipe.no_viability is True
    assert args_pipe.export_clean_fcs is True


def test_yaml_and_json_roundtrip(tmp_path: Path):
    cfg = PipelineConfig(
        fcs_file="data/test.fcs",
        output_dir=str(tmp_path / "out"),
        gating_mode="phenograph",
        adaptive_gating=True,
        legend_mode="centroid",
        seed=7,
        timestamp_dir=False,
        overwrite=True,
    )

    # Test YAML
    yaml_file = tmp_path / "test_config.yaml"
    cfg.save(yaml_file)
    loaded_yaml = PipelineConfig.load(yaml_file)
    assert loaded_yaml.fcs_file == cfg.fcs_file
    assert loaded_yaml.gating_mode == "phenograph"
    assert loaded_yaml.adaptive_gating is True

    # Test JSON
    json_file = tmp_path / "test_config.json"
    cfg.save(json_file)
    loaded_json = PipelineConfig.load(json_file)
    assert loaded_json.fcs_file == cfg.fcs_file
    assert loaded_json.gating_mode == "phenograph"
    assert loaded_json.seed == 7


def test_pipeline_from_config(tmp_path: Path):
    cfg_data = {
        "input": {"fcs_file": "data/sample.fcs"},
        "analysis": {"gating_mode": "compare", "cluster_parent": "singlets", "adaptive_gating": True, "seed": 42},
        "output": {"output_dir": str(tmp_path), "timestamp_dir": False, "overwrite": True},
    }
    cfg_path = tmp_path / "test.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f)

    pipeline, fcs_file = FlowCytometryPipeline.from_config(cfg_path)
    assert fcs_file == "data/sample.fcs"
    assert pipeline.gating_mode == "compare"
    assert pipeline.cluster_parent == "singlets"
    assert pipeline.adaptive_gating is True
    assert pipeline.random_seed == 42
    assert pipeline.timestamp_dir is False
    assert pipeline.overwrite is True


def test_generate_template():
    yaml_str = PipelineConfig.generate_template("yaml")
    assert "fcs_file:" in yaml_str
    assert "gating_mode:" in yaml_str
    assert "adaptive_gating:" in yaml_str

    json_str = PipelineConfig.generate_template("json")
    parsed = json.loads(json_str)
    assert "input" in parsed
    assert "analysis" in parsed


def test_config_file_copy_in_output_directory(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify that when a config file is used, a copy is saved to the output run directory."""
    fcs_file = str(tqc_sample_fcs_path)
    out_dir = tmp_path / "yaml_run_out"
    cfg_file = tmp_path / "custom_experiment_config.yaml"

    cfg_data = {
        "input": {"fcs_file": fcs_file},
        "analysis": {"gating_mode": "template", "adaptive_gating": False, "seed": 42},
        "output": {"output_dir": str(out_dir), "timestamp_dir": False, "overwrite": True, "numbered_steps": False},
    }
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f)

    cfg = PipelineConfig.load(cfg_file)
    res = cfg.run()

    assert res.config_file_path is not None
    assert Path(res.config_file_path).is_file()
    # Check that standardized config.yaml and resolved_config.yaml copies exist
    assert (out_dir / "config.yaml").is_file()
    assert (out_dir / "resolved_config.yaml").is_file()

    # Verify content matches
    content = (out_dir / "config.yaml").read_text(encoding="utf-8")
    assert "template" in content

    # Verify report.md contains config file mention
    report_text = Path(res.report_md_path).read_text(encoding="utf-8")
    assert "config.yaml" in report_text

    # Verify manifest records config_file
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest[-1]["config_file"] == res.config_file_path


def test_config_json_copy_in_output_directory(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify that JSON config files are also copied into the output directory."""
    fcs_file = str(tqc_sample_fcs_path)
    out_dir = tmp_path / "json_run_out"
    cfg_file = tmp_path / "params.json"

    cfg_data = {
        "input": {"fcs_file": fcs_file},
        "analysis": {"gating_mode": "template", "adaptive_gating": False, "seed": 42},
        "output": {"output_dir": str(out_dir), "timestamp_dir": False, "overwrite": True, "numbered_steps": False},
    }
    cfg_file.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")

    pipeline, target_fcs = FlowCytometryPipeline.from_config(cfg_file)
    res = pipeline.run(target_fcs)

    assert res.config_file_path is not None
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "resolved_config.yaml").is_file()


def test_pipeline_run_without_config_does_not_copy_config(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify that running without a config file leaves config_file_path as None."""
    fcs_file = str(tqc_sample_fcs_path)
    pipeline = FlowCytometryPipeline(output_dir=tmp_path, timestamp_dir=False, overwrite=True)
    res = pipeline.run(fcs_file)
    assert res.config_file_path is None
    assert not (tmp_path / "config.yaml").exists()
    assert not (tmp_path / "config.json").exists()


def test_pipeline_config_population_and_stats_sections():
    """Verify loading Step 05-07 configurations from hierarchical dictionary."""
    data = {
        "output": {"output_dir": "outputs/my_run"},
        "05_population_discovery": {
            "clustering_method": "som",
            "n_clusters": 12,
            "auto_k": True,
            "k_min": 5,
            "k_max": 18,
            "compute_umap": True,
            "umap_events_per_sample": 1500,
            "cluster_name_map": {"0": "CD4+ T", "1": "B Cells"},
            "export_prism": True,
        },
        "06_statistical_analysis": {
            "condition_column": "treatment",
            "reference_group": "Vehicle",
            "target_group": "DrugA",
            "paired_column": "donor_id",
            "fdr_threshold": 0.01,
            "log2fc_threshold": 1.5,
            "generate_forest_plot": True,
            "generate_ds_heatmap": True,
        },
        "07_workspaces_reporting": {
            "export_augmented_fcs": True,
            "export_flowjo_wsp": True,
            "export_gating_ml": True,
            "generate_miflowcyt_report": True,
        },
    }
    cfg = PipelineConfig.from_dict(data)
    assert cfg.output_dir == "outputs/my_run"
    assert cfg.clustering_method == "som"
    assert cfg.n_clusters == 12
    assert cfg.auto_k is True
    assert cfg.k_min == 5
    assert cfg.k_max == 18
    assert cfg.compute_umap is True
    assert cfg.umap_events_per_sample == 1500
    assert cfg.cluster_name_map == {"0": "CD4+ T", "1": "B Cells"}
    assert cfg.condition_column == "treatment"
    assert cfg.reference_group == "Vehicle"
    assert cfg.target_group == "DrugA"
    assert cfg.paired_column == "donor_id"
    assert cfg.fdr_threshold == 0.01
    assert cfg.log2fc_threshold == 1.5
    assert cfg.generate_forest_plot is True
    assert cfg.generate_ds_heatmap is True
    assert cfg.export_augmented_fcs is True
    assert cfg.export_flowjo_wsp is True
    assert cfg.generate_miflowcyt_report is True


def test_pipeline_config_all_numbered_steps():
    """Verify loading complete 01-07 numbered pipeline steps configuration."""
    data = {
        "01_time_qc": {"enabled": True, "bin_width": 2.5},
        "02_compensation": {
            "source": "controls",
            "controls_dir": "data/controls",
            "manifest": "configs/comp_manifest.csv",
            "unstained_control": "data/unstained.fcs",
        },
        "03_cleanup_gating": {
            "debris_method": "adaptive",
            "singlet_mode": "dual",
            "singlet_tolerance": 0.20,
            "enable_viability": True,
            "viability_cutoff": 450.0,
            "export_clean_fcs": True,
        },
        "04_batch_preparation": {
            "mode": "target_count",
            "target_events": 100000,
        },
        "05_population_discovery": {
            "clustering_method": "som",
            "n_clusters": 8,
            "auto_k": False,
        },
        "06_statistical_analysis": {
            "condition_column": "group",
            "reference_group": "A",
            "target_group": "B",
        },
        "exports": {
            "export_augmented_fcs": True,
            "generate_miflowcyt_report": True,
        },
    }
    cfg = PipelineConfig.from_dict(data)
    assert cfg.enable_time_qc is True
    assert cfg.time_qc_bin_width == 2.5
    assert cfg.compensation_source == "controls"
    assert cfg.comp_controls_dir == "data/controls"
    assert cfg.comp_manifest == "configs/comp_manifest.csv"
    assert cfg.unstained_control == "data/unstained.fcs"
    assert cfg.debris_method == "adaptive"
    assert cfg.singlet_tolerance == 0.20
    assert cfg.viability_cutoff == 450.0
    assert cfg.export_clean_fcs is True
    assert cfg.batch_preparation["target_events"] == 100000
    assert cfg.n_clusters == 8
    assert cfg.condition_column == "group"
    assert cfg.generate_miflowcyt_report is True

    # Check to_dict serialization includes exports section
    d = cfg.to_dict(nested=True)
    assert "01_time_qc" in d
    assert "02_compensation" in d
    assert "03_cleanup_gating" in d
    assert "04_batch_preparation" in d
    assert "05_population_discovery" in d
    assert "06_statistical_analysis" in d
    assert "exports" in d


def test_pipeline_config_metadata_and_step_enablement():
    """Verify metadata_file parsing and individual step enablement toggles."""
    data = {
        "input": {
            "fcs_file": "data/sample.fcs",
            "metadata_file": "metadata/study_cohort.csv",
        },
        "01_time_qc": {"enabled": False},
        "02_compensation": {"enabled": False, "source": "none"},
        "03_cleanup_gating": {"enabled": False},
        "04_batch_preparation": {"enabled": False, "target_events": 50000},
        "05_population_discovery": {"enabled": False, "n_clusters": 12},
        "06_statistical_analysis": {
            "enabled": False,
            "metadata_file": "metadata/study_cohort.csv",
            "condition_column": "treatment",
        },
        "exports": {"enabled": False},
    }
    cfg = PipelineConfig.from_dict(data)
    assert cfg.metadata_file == "metadata/study_cohort.csv"
    assert cfg.enable_time_qc is False
    assert cfg.enable_compensation is False
    assert cfg.enable_cleanup_gating is False
    assert cfg.enable_batch_preparation is False
    assert cfg.enable_population_discovery is False
    assert cfg.enable_statistical_analysis is False
    assert cfg.enable_workspaces_reporting is False

    # Check to_dict round-trip
    d = cfg.to_dict(nested=True)
    assert d["input"]["metadata_file"] == "metadata/study_cohort.csv"
    assert d["01_time_qc"]["enabled"] is False
    assert d["02_compensation"]["enabled"] is False
    assert d["03_cleanup_gating"]["enabled"] is False
    assert d["04_batch_preparation"]["enabled"] is False
    assert d["05_population_discovery"]["enabled"] is False
    assert d["06_statistical_analysis"]["enabled"] is False
    assert d["exports"]["enabled"] is False


def test_pipeline_config_markers_and_transparent_clustering_roundtrip(tmp_path: Path):
    """Verify markers_file, cell_signatures, auto_k_method, consensus params, and n_clusters=None round-trip."""
    sigs = {
        "Classical_Monocytes": {"positive": ["CD14", "CD33"], "negative": ["CD3"]},
    }
    data = {
        "input": {
            "fcs_file": None,
            "metadata_file": "metadata/samples.csv",
            "markers_file": "metadata/markers.csv",
        },
        "05_population_discovery": {
            "clustering_method": "som",
            "n_clusters": None,
            "auto_k": True,
            "auto_k_method": "consensus",
            "k_min": 2,
            "k_max": 12,
            "consensus_subsample": 12000,
            "consensus_iterations": 25,
            "cell_signatures": sigs,
        },
        "06_statistical_analysis": {
            "condition_column": "condition",
            "reference_group": "Soft_Hydrogel",
            "target_group": "Stiff_Hydrogel",
            "paired_column": "donor_id",
        },
        "output": {"output_dir": str(tmp_path), "timestamp_dir": False},
    }
    cfg = PipelineConfig.from_dict(data)
    assert cfg.markers_file == "metadata/markers.csv"
    assert cfg.n_clusters is None
    assert cfg.auto_k is True
    assert cfg.auto_k_method == "consensus"
    assert cfg.consensus_subsample == 12000
    assert cfg.consensus_iterations == 25
    assert cfg.cell_signatures == sigs
    assert cfg.reference_group == "Soft_Hydrogel"
    assert cfg.target_group == "Stiff_Hydrogel"
    assert cfg.paired_column == "donor_id"

    nested = cfg.to_dict(nested=True)
    assert nested["input"]["markers_file"] == "metadata/markers.csv"
    assert nested["05_population_discovery"]["n_clusters"] is None
    assert nested["05_population_discovery"]["auto_k_method"] == "consensus"
    assert nested["05_population_discovery"]["consensus_subsample"] == 12000
    assert nested["05_population_discovery"]["consensus_iterations"] == 25
    assert nested["05_population_discovery"]["cell_signatures"] == sigs

    pipeline, fcs_file = cfg.to_pipeline()
    assert fcs_file is None
    assert pipeline.markers_file == Path("metadata/markers.csv")
    assert pipeline.cell_signatures == sigs
    assert pipeline.n_clusters is None
    assert pipeline.auto_k is True
    assert pipeline.auto_k_method == "consensus"
    assert pipeline.consensus_subsample == 12000
    assert pipeline.consensus_iterations == 25
    assert pipeline.reference_group == "Soft_Hydrogel"
    assert pipeline.target_group == "Stiff_Hydrogel"

    yaml_tmpl = PipelineConfig.generate_template("yaml")
    assert "markers_file:" in yaml_tmpl
    assert "auto_k_method:" in yaml_tmpl
    assert "cell_signatures:" in yaml_tmpl


