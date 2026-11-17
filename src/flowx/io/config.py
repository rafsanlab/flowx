"""Configuration management and parameter file parsing for flow cytometry pipelines.

Supports both YAML (.yaml, .yml) and JSON (.json) configuration files, with support
for flat dictionaries or hierarchically grouped sections (input, analysis, visualization, output).
Enables programmatic execution, CLI overrides, and template generation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml

    HAS_YAML = True
except ImportError:  # pragma: no cover
    HAS_YAML = False

from flowx.io.paths import sanitize_path


VALID_GATING_MODES = {"template", "som", "phenograph", "compare"}
VALID_QC_MODES = {"template", "clustering"}
VALID_LEGEND_MODES = {"centroid", "external"}
VALID_TRANSFORM_TYPES = {"logicle", "biexponential", "biex", "wsp_biex", "asinh", "arcsinh", "log"}
VALID_DEBRIS_METHODS = {"gmm", "adaptive", "manual"}
VALID_SINGLET_MODES = {"dual", "single", "one_time", "fsc_only", "ssc_only", "none"}
VALID_SINGLET_METHODS = {"gmm_ratio", "envelope"}
VALID_TECHNOLOGIES = {"auto", "conventional", "spectral"}
VALID_STATS_GRANULARITY = {"populations", "clusters", "both"}


@dataclass
class PipelineConfig:
    """Configuration container holding all pipeline execution parameters.

    Attributes:
        fcs_file: Path to raw input .fcs file (optional in config if supplied via CLI/API).
        output_dir: Base directory for storing runs, plots, tables, and workspaces.
        gating_mode: Analysis mode: 'som' (default), 'template', 'phenograph', or 'compare'.
        qc_method: Quality control mode: 'clustering' (default GMM) or 'template'.
        adaptive_gating: If True, uses openCyto/flowDensity local KDE minimum valley-seeking.
        cluster_parent: Parent population for unsupervised clustering ('viable', 'singlets', 'cells', 'all').
        legend_mode: Legend style: 'centroid' (direct badge at cluster center) or 'external'.
        seed: Random seed for reproducible clustering and subsampling.
        timestamp_dir: If True, creates timestamped subfolder (sample_YYYYMMDD_HHMMSS).
        overwrite: If True, allows writing directly into output_dir without timestamping.
        workspace_file: Optional path to custom workspace.json or workspace.xml.
        marker_mapping: Optional dict mapping canonical marker names to detector names.
        comp_controls_dir: Directory containing single-stained compensation control FCS files.
        comp_manifest: Path to CSV or YAML manifest mapping detectors to control FCS files.
        unstained_control: Optional path to universal unstained / negative control FCS file.
        compensation_source: Source of spillover matrix ('auto', 'controls', 'embedded', 'none').
        transform_type: Scale transformation method ('logicle' [default], 'biexponential', 'asinh', 'log').
    """

    fcs_file: Optional[str] = None
    output_dir: str = "outputs"
    gating_mode: str = "som"
    cluster_parent: str = "viable"
    qc_method: str = "clustering"
    adaptive_gating: bool = False
    legend_mode: str = "centroid"
    seed: int = 42
    timestamp_dir: bool = True
    overwrite: bool = False
    workspace_file: Optional[str] = None
    marker_mapping: Optional[Dict[str, str]] = None
    comp_controls_dir: Optional[str] = None
    comp_manifest: Optional[str] = None
    unstained_control: Optional[str] = None
    compensation_source: str = "auto"
    comp_channel_mapping: Optional[Dict[str, str]] = None
    config_file_path: Optional[str] = None
    log_level: str = "INFO"
    transform_type: str = "logicle"
    enable_umap: bool = True
    umap_subsample: int = 20000
    enable_time_qc: bool = True
    time_qc_bin_width: Optional[float] = None
    harmonization_file: Optional[str] = None
    enable_harmonization: bool = False
    harmonization_config: Optional[Dict[str, Any]] = None
    debris_method: str = "gmm"
    debris_n_components: int = 2
    debris_sd_scale: float = 1.0
    debris_min_probability: Optional[float] = None
    fsc_channel: Optional[str] = None
    ssc_channel: Optional[str] = None
    fsc_height_channel: Optional[str] = None
    ssc_height_channel: Optional[str] = None
    singlet_mode: str = "dual"
    singlet_method: str = "gmm_ratio"
    singlet_tolerance: float = 0.25
    viability_cutoff: Optional[float] = None
    viability_channel: Optional[str] = None
    viability_looseness: float = 1.0
    enable_viability: bool = True
    dump_channel: Optional[str] = None
    dump_cutoff: Optional[float] = None
    dump_looseness: float = 1.0
    dump_enabled: bool = False
    export_clean_fcs: bool = False
    metadata_file: Optional[str] = None
    markers_file: Optional[str] = None
    technology: str = "auto"
    data_dir: Optional[str] = None
    files_list: Optional[Union[str, List[str]]] = None
    transform_cofactor: float = 5.0
    transform_channels: Optional[Union[str, List[str]]] = None
    transform_exclude_channels: Optional[List[str]] = None

    # Step Enablement Flags
    enable_qc: bool = True
    enable_discovery: bool = False
    numbered_steps: bool = True
    enable_compensation: bool = True
    enable_cleanup_gating: bool = True
    enable_transformation: bool = True
    enable_batch_preparation: bool = True
    enable_population_discovery: bool = False
    enable_statistical_analysis: bool = True
    enable_workspaces_reporting: bool = True

    # Step 01: Time Quality Control
    time_qc: Optional[Dict[str, Any]] = None

    # Step 02: Spectral Compensation
    compensation: Optional[Dict[str, Any]] = None

    # Step 03: Core Clean-up Gating
    cleanup_gating: Optional[Dict[str, Any]] = None

    # Step 04: Scale Transformation & Batch Preparation
    transformation: Optional[Dict[str, Any]] = None
    batch_preparation: Optional[Dict[str, Any]] = None
    batch_column: Optional[str] = None
    overwrite_autocheck: bool = False

    # Step 05: Population Discovery
    clustering_method: str = "som"
    n_clusters: Optional[int] = 10
    auto_k: bool = True
    auto_k_method: str = "consensus"
    k_min: int = 4
    k_max: int = 20
    consensus_subsample: float = 0.8
    consensus_iterations: int = 50
    compute_umap: bool = True
    umap_events_per_sample: int = 1000
    stratified_density_contours: bool = True
    umap_covariates: Optional[List[str]] = None
    cluster_palette: str = "tab20"
    export_prism: bool = True
    cluster_name_map: Optional[Dict[str, str]] = None
    cell_signatures: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = None
    lineage_markers: Optional[Union[str, List[str]]] = None
    marker_roles: Optional[List[str]] = None
    include_markers: Optional[List[str]] = None
    exclude_markers: Optional[List[str]] = None
    population_discovery: Optional[Dict[str, Any]] = None

    # Step 06: Statistical Analysis
    condition_column: Optional[str] = None
    reference_group: Optional[str] = None
    target_group: Optional[str] = None
    paired_column: Optional[str] = None
    da_test_method: str = "auto"
    ds_test_method: str = "auto"
    stats_granularity: str = "populations"
    stats_marker_roles: Optional[List[str]] = None
    fdr_threshold: float = 0.05
    log2fc_threshold: float = 1.0
    generate_forest_plot: bool = True
    generate_ds_heatmap: bool = True
    generate_abundance_boxplots: bool = True
    generate_volcano_plot: bool = True
    statistical_analysis: Optional[Dict[str, Any]] = None

    # Deliverable Exports & Reporting (Run-level metadata deliverables)
    export_augmented_fcs: bool = True
    export_flowjo_wsp: bool = True
    export_gating_ml: bool = True
    generate_miflowcyt_report: bool = True
    exports: Optional[Dict[str, Any]] = None
    workspaces_reporting: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        """Validate parameter values, raising ValueError for invalid settings."""
        if self.gating_mode.lower() not in VALID_GATING_MODES:
            raise ValueError(
                f"Invalid gating_mode '{self.gating_mode}'. Allowed: {sorted(VALID_GATING_MODES)}"
            )
        if not self.cluster_parent or not isinstance(self.cluster_parent, str):
            raise ValueError("cluster_parent must be a non-empty string (e.g. 'viable', 'singlets', 'cells', 'all').")
        if self.qc_method.lower() not in VALID_QC_MODES:
            raise ValueError(
                f"Invalid qc_method '{self.qc_method}'. Allowed: {sorted(VALID_QC_MODES)}"
            )
        if self.legend_mode.lower() not in VALID_LEGEND_MODES:
            raise ValueError(
                f"Invalid legend_mode '{self.legend_mode}'. Allowed: {sorted(VALID_LEGEND_MODES)}"
            )
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                f"Invalid log_level '{self.log_level}'. Allowed: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
        if self.transform_type.lower() not in VALID_TRANSFORM_TYPES:
            raise ValueError(
                f"Invalid transform_type '{self.transform_type}'. Allowed: {sorted(VALID_TRANSFORM_TYPES)}"
            )
        if self.debris_method.lower() not in VALID_DEBRIS_METHODS:
            raise ValueError(
                f"Invalid debris_method '{self.debris_method}'. Allowed: {sorted(VALID_DEBRIS_METHODS)}"
            )
        if self.singlet_mode.lower() not in VALID_SINGLET_MODES:
            raise ValueError(
                f"Invalid singlet_mode '{self.singlet_mode}'. Allowed: {sorted(VALID_SINGLET_MODES)}"
            )
        if self.singlet_method.lower() not in VALID_SINGLET_METHODS:
            raise ValueError(
                f"Invalid singlet_method '{self.singlet_method}'. Allowed: {sorted(VALID_SINGLET_METHODS)}"
            )
        if str(self.stats_granularity or "populations").lower() not in VALID_STATS_GRANULARITY:
            raise ValueError(
                f"Invalid stats_granularity '{self.stats_granularity}'. Allowed: {sorted(VALID_STATS_GRANULARITY)}"
            )
        if self.technology.lower() not in VALID_TECHNOLOGIES:
            raise ValueError(
                f"Invalid technology '{self.technology}'. Allowed: {sorted(VALID_TECHNOLOGIES)}"
            )


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        """Parse configuration from a dictionary.

        Accepts either a flat dictionary or a structured hierarchical dictionary
        with sections: 'input', 'analysis', 'visualization', 'output'.
        """
        flat: Dict[str, Any] = {}

        # 1. Unpack nested sections if present
        if any(k in data for k in ("input", "analysis", "visualization", "output", "outputs", "cleanup")):
            # Section: input
            inp = data.get("input", {})
            if isinstance(inp, dict):
                if "fcs_file" in inp:
                    flat["fcs_file"] = inp["fcs_file"]
                elif "filepath" in inp:
                    flat["fcs_file"] = inp["filepath"]
                if "marker_mapping" in inp:
                    flat["marker_mapping"] = inp["marker_mapping"]
                if "comp_manifest" in inp:
                    flat["comp_manifest"] = inp["comp_manifest"]
                elif "control_manifest" in inp:
                    flat["comp_manifest"] = inp["control_manifest"]
                if "comp_controls_dir" in inp:
                    flat["comp_controls_dir"] = inp["comp_controls_dir"]
                elif "compensation_controls" in inp:
                    flat["comp_controls_dir"] = inp["compensation_controls"]
                if "unstained_control" in inp:
                    flat["unstained_control"] = inp["unstained_control"]
                if "compensation_source" in inp:
                    flat["compensation_source"] = inp["compensation_source"]
                elif "comp_source" in inp:
                    flat["compensation_source"] = inp["comp_source"]
                if "comp_channel_mapping" in inp:
                    flat["comp_channel_mapping"] = inp["comp_channel_mapping"]
                if "harmonization_file" in inp:
                    flat["harmonization_file"] = inp["harmonization_file"]
                    flat["enable_harmonization"] = True
                elif "harmonization" in inp:
                    val = inp["harmonization"]
                    if isinstance(val, str):
                        flat["harmonization_file"] = val
                        flat["enable_harmonization"] = True
                    elif isinstance(val, bool):
                        flat["enable_harmonization"] = val
                    elif isinstance(val, dict):
                        flat["harmonization_config"] = val
                        flat["enable_harmonization"] = True
                if "enable_harmonization" in inp:
                    flat["enable_harmonization"] = bool(inp["enable_harmonization"])
                if "metadata_file" in inp:
                    flat["metadata_file"] = str(inp["metadata_file"]) if inp["metadata_file"] else None
                elif "sample_metadata" in inp:
                    flat["metadata_file"] = str(inp["sample_metadata"]) if inp["sample_metadata"] else None
                elif "metadata" in inp:
                    flat["metadata_file"] = str(inp["metadata"]) if inp["metadata"] else None
                if "markers_file" in inp:
                    flat["markers_file"] = str(inp["markers_file"]) if inp["markers_file"] else None
                elif "marker_metadata" in inp:
                    flat["markers_file"] = str(inp["marker_metadata"]) if inp["marker_metadata"] else None
                elif "markers_metadata" in inp:
                    flat["markers_file"] = str(inp["markers_metadata"]) if inp["markers_metadata"] else None
                if "technology" in inp:
                    flat["technology"] = str(inp["technology"]).lower()
                if "data_dir" in inp:
                    flat["data_dir"] = str(inp["data_dir"])
                elif "samples_dir" in inp:
                    flat["data_dir"] = str(inp["samples_dir"])
                if "files_list" in inp:
                    raw_fl = inp["files_list"]
                    if isinstance(raw_fl, (list, tuple)):
                        flat["files_list"] = [str(p) for p in raw_fl if p]
                    elif raw_fl:
                        flat["files_list"] = str(raw_fl)

            # Section: analysis
            ana = data.get("analysis", {})
            if isinstance(ana, dict):
                if "gating_mode" in ana:
                    flat["gating_mode"] = ana["gating_mode"]
                if "cluster_parent" in ana:
                    flat["cluster_parent"] = str(ana["cluster_parent"])
                if "qc_method" in ana:
                    flat["qc_method"] = ana["qc_method"]
                if "adaptive_gating" in ana:
                    flat["adaptive_gating"] = bool(ana["adaptive_gating"])
                if "seed" in ana:
                    flat["seed"] = int(ana["seed"])
                if "workspace_file" in ana:
                    flat["workspace_file"] = ana["workspace_file"]
                elif "workspace" in ana:
                    flat["workspace_file"] = ana["workspace"]
                if "log_level" in ana:
                    flat["log_level"] = ana["log_level"]
                if "transform_type" in ana:
                    flat["transform_type"] = ana["transform_type"]
                elif "transform" in ana:
                    flat["transform_type"] = ana["transform"]
                elif "xform" in ana:
                    flat["transform_type"] = ana["xform"]
                if "enable_umap" in ana:
                    flat["enable_umap"] = bool(ana["enable_umap"])
                elif "umap" in ana:
                    flat["enable_umap"] = bool(ana["umap"])
                if "umap_subsample" in ana:
                    flat["umap_subsample"] = int(ana["umap_subsample"])
                if "enable_time_qc" in ana:
                    flat["enable_time_qc"] = bool(ana["enable_time_qc"])
                elif "time_qc" in ana:
                    flat["enable_time_qc"] = bool(ana["time_qc"])
                if "time_qc_bin_width" in ana:
                    flat["time_qc_bin_width"] = float(ana["time_qc_bin_width"])
                if "enable_qc" in ana:
                    flat["enable_qc"] = bool(ana["enable_qc"])
                elif "qc" in ana:
                    flat["enable_qc"] = bool(ana["qc"])
                if "enable_discovery" in ana:
                    flat["enable_discovery"] = bool(ana["enable_discovery"])
                    flat["enable_population_discovery"] = bool(ana["enable_discovery"])
                elif "discovery" in ana:
                    flat["enable_discovery"] = bool(ana["discovery"])
                    flat["enable_population_discovery"] = bool(ana["discovery"])
                if "numbered_steps" in ana:
                    flat["numbered_steps"] = bool(ana["numbered_steps"])

            # Section: visualization
            vis = data.get("visualization", {})
            if isinstance(vis, dict):
                if "legend_mode" in vis:
                    flat["legend_mode"] = vis["legend_mode"]

            # Section: cleanup
            cln = data.get("cleanup", {})
            if isinstance(cln, dict):
                if "debris_method" in cln:
                    flat["debris_method"] = str(cln["debris_method"])
                if "debris_sd_scale" in cln and cln["debris_sd_scale"] is not None:
                    flat["debris_sd_scale"] = float(cln["debris_sd_scale"])
                if "debris_min_probability" in cln:
                    flat["debris_min_probability"] = float(cln["debris_min_probability"]) if cln["debris_min_probability"] is not None else None
                if "singlet_mode" in cln:
                    flat["singlet_mode"] = str(cln["singlet_mode"])
                if "singlet_method" in cln:
                    flat["singlet_method"] = str(cln["singlet_method"])
                if "singlet_tolerance" in cln:
                    flat["singlet_tolerance"] = float(cln["singlet_tolerance"])
                if "viability_cutoff" in cln:
                    flat["viability_cutoff"] = float(cln["viability_cutoff"]) if cln["viability_cutoff"] is not None else None
                if "viability_looseness" in cln and cln["viability_looseness"] is not None:
                    flat["viability_looseness"] = float(cln["viability_looseness"])
                if "dump_looseness" in cln and cln["dump_looseness"] is not None:
                    flat["dump_looseness"] = float(cln["dump_looseness"])
                if "enable_viability" in cln:
                    flat["enable_viability"] = bool(cln["enable_viability"])
                elif "viability_enabled" in cln:
                    flat["enable_viability"] = bool(cln["viability_enabled"])
                if "export_clean_fcs" in cln:
                    flat["export_clean_fcs"] = bool(cln["export_clean_fcs"])

            # Section: output / outputs
            out = data.get("outputs", data.get("output", {}))
            if isinstance(out, dict):
                if "output_dir" in out:
                    flat["output_dir"] = out["output_dir"]
                if "timestamp_dir" in out:
                    flat["timestamp_dir"] = bool(out["timestamp_dir"])
                elif "no_timestamp" in out:
                    flat["timestamp_dir"] = not bool(out["no_timestamp"])
                if "overwrite" in out:
                    flat["overwrite"] = bool(out["overwrite"])

        # Section: 01_time_qc (or time_qc)
        tqc = data.get("01_time_qc", data.get("time_qc", {}))
        if isinstance(tqc, dict) and tqc:
            flat["time_qc"] = tqc
            if "enabled" in tqc:
                flat["enable_time_qc"] = bool(tqc["enabled"])
            elif "enable_qc" in tqc:
                flat["enable_time_qc"] = bool(tqc["enable_qc"])
            if "bin_width" in tqc:
                flat["time_qc_bin_width"] = float(tqc["bin_width"]) if tqc["bin_width"] is not None else None

        # Section: 02_compensation (or compensation)
        cmp = data.get("02_compensation", data.get("compensation", {}))
        if isinstance(cmp, dict) and cmp:
            flat["compensation"] = cmp
            if "enabled" in cmp:
                flat["enable_compensation"] = bool(cmp["enabled"])
            if "method" in cmp:
                comp_method = str(cmp["method"]).lower()
                if comp_method in ("unmix", "unmixing", "spectral") and flat.get("technology", "auto") == "auto":
                    flat["technology"] = "spectral"
            if "source" in cmp:
                flat["compensation_source"] = str(cmp["source"])
            elif "compensation_source" in cmp:
                flat["compensation_source"] = str(cmp["compensation_source"])
            if "controls_dir" in cmp and cmp["controls_dir"] is not None:
                flat["comp_controls_dir"] = str(cmp["controls_dir"])
            elif "comp_controls_dir" in cmp and cmp["comp_controls_dir"] is not None:
                flat["comp_controls_dir"] = str(cmp["comp_controls_dir"])
            if "manifest" in cmp and cmp["manifest"] is not None:
                flat["comp_manifest"] = str(cmp["manifest"])
            elif "comp_manifest" in cmp and cmp["comp_manifest"] is not None:
                flat["comp_manifest"] = str(cmp["comp_manifest"])
            if "unstained_control" in cmp and cmp["unstained_control"] is not None:
                flat["unstained_control"] = str(cmp["unstained_control"])
            if cmp.get("channel_mapping") is not None:
                flat["comp_channel_mapping"] = dict(cmp["channel_mapping"])
            elif cmp.get("comp_channel_mapping") is not None:
                flat["comp_channel_mapping"] = dict(cmp["comp_channel_mapping"])

        # Section: 03_cleanup_gating (or cleanup_gating / cleanup)
        cln_sec = data.get("03_cleanup_gating", data.get("cleanup_gating", {}))
        if isinstance(cln_sec, dict) and cln_sec:
            flat["cleanup_gating"] = cln_sec
            if "enabled" in cln_sec:
                flat["enable_cleanup_gating"] = bool(cln_sec["enabled"])
            if "debris_method" in cln_sec:
                flat["debris_method"] = str(cln_sec["debris_method"])
            if "debris_n_components" in cln_sec and cln_sec["debris_n_components"] is not None:
                flat["debris_n_components"] = int(cln_sec["debris_n_components"])
            if "debris_sd_scale" in cln_sec and cln_sec["debris_sd_scale"] is not None:
                flat["debris_sd_scale"] = float(cln_sec["debris_sd_scale"])
            if "debris_min_probability" in cln_sec:
                flat["debris_min_probability"] = float(cln_sec["debris_min_probability"]) if cln_sec["debris_min_probability"] is not None else None
            if "singlet_mode" in cln_sec:
                flat["singlet_mode"] = str(cln_sec["singlet_mode"])
            if "singlet_method" in cln_sec:
                flat["singlet_method"] = str(cln_sec["singlet_method"])
            if "singlet_tolerance" in cln_sec:
                flat["singlet_tolerance"] = float(cln_sec["singlet_tolerance"])
            if "viability_cutoff" in cln_sec:
                flat["viability_cutoff"] = float(cln_sec["viability_cutoff"]) if cln_sec["viability_cutoff"] is not None else None
            if "viability_looseness" in cln_sec and cln_sec["viability_looseness"] is not None:
                flat["viability_looseness"] = float(cln_sec["viability_looseness"])
            if "viability_channel" in cln_sec and cln_sec["viability_channel"] is not None:
                flat["viability_channel"] = str(cln_sec["viability_channel"])
            if "enable_viability" in cln_sec:
                flat["enable_viability"] = bool(cln_sec["enable_viability"])
            elif "viability_enabled" in cln_sec:
                flat["enable_viability"] = bool(cln_sec["viability_enabled"])
            if "fsc_channel" in cln_sec and cln_sec["fsc_channel"] is not None:
                flat["fsc_channel"] = str(cln_sec["fsc_channel"])
            if "ssc_channel" in cln_sec and cln_sec["ssc_channel"] is not None:
                flat["ssc_channel"] = str(cln_sec["ssc_channel"])
            if "fsc_height_channel" in cln_sec and cln_sec["fsc_height_channel"] is not None:
                flat["fsc_height_channel"] = str(cln_sec["fsc_height_channel"])
            elif "fsc_h_channel" in cln_sec and cln_sec["fsc_h_channel"] is not None:
                flat["fsc_height_channel"] = str(cln_sec["fsc_h_channel"])
            if "ssc_height_channel" in cln_sec and cln_sec["ssc_height_channel"] is not None:
                flat["ssc_height_channel"] = str(cln_sec["ssc_height_channel"])
            elif "ssc_h_channel" in cln_sec and cln_sec["ssc_h_channel"] is not None:
                flat["ssc_height_channel"] = str(cln_sec["ssc_h_channel"])
            if "dump_channel" in cln_sec and cln_sec["dump_channel"] is not None:
                flat["dump_channel"] = str(cln_sec["dump_channel"])
            if "dump_cutoff" in cln_sec and cln_sec["dump_cutoff"] is not None:
                flat["dump_cutoff"] = float(cln_sec["dump_cutoff"])
            if "dump_looseness" in cln_sec and cln_sec["dump_looseness"] is not None:
                flat["dump_looseness"] = float(cln_sec["dump_looseness"])
            if "dump_enabled" in cln_sec:
                flat["dump_enabled"] = bool(cln_sec["dump_enabled"])
            if "export_clean_fcs" in cln_sec:
                flat["export_clean_fcs"] = bool(cln_sec["export_clean_fcs"])

            # Support nested sub-dictionaries inside 03_cleanup_gating (spectral config schema)
            deb_sub = cln_sec.get("debris_removal")
            if isinstance(deb_sub, dict):
                if "method" in deb_sub:
                    flat["debris_method"] = str(deb_sub["method"])
                if "n_components" in deb_sub and deb_sub["n_components"] is not None:
                    flat["debris_n_components"] = int(deb_sub["n_components"])
                if "sd_scale" in deb_sub and deb_sub["sd_scale"] is not None:
                    flat["debris_sd_scale"] = float(deb_sub["sd_scale"])
                elif "debris_sd_scale" in deb_sub and deb_sub["debris_sd_scale"] is not None:
                    flat["debris_sd_scale"] = float(deb_sub["debris_sd_scale"])
                if "min_probability" in deb_sub:
                    flat["debris_min_probability"] = float(deb_sub["min_probability"]) if deb_sub["min_probability"] is not None else None
                elif "debris_min_probability" in deb_sub:
                    flat["debris_min_probability"] = float(deb_sub["debris_min_probability"]) if deb_sub["debris_min_probability"] is not None else None
            sng_sub = cln_sec.get("singlet_gating")
            if isinstance(sng_sub, dict):
                if "enabled" in sng_sub and not bool(sng_sub["enabled"]):
                    flat["singlet_mode"] = "none"
                elif "mode" in sng_sub:
                    flat["singlet_mode"] = str(sng_sub["mode"])
                if "method" in sng_sub and str(sng_sub["method"]).lower() in VALID_SINGLET_METHODS:
                    flat["singlet_method"] = str(sng_sub["method"])
                if "tolerance" in sng_sub and sng_sub["tolerance"] is not None:
                    flat["singlet_tolerance"] = float(sng_sub["tolerance"])
            via_sub = cln_sec.get("viability_gating")
            if isinstance(via_sub, dict):
                if "enabled" in via_sub:
                    flat["enable_viability"] = bool(via_sub["enabled"])
                if "channel" in via_sub and via_sub["channel"] is not None:
                    flat["viability_channel"] = str(via_sub["channel"])
                if "cutoff" in via_sub and via_sub["cutoff"] is not None:
                    flat["viability_cutoff"] = float(via_sub["cutoff"])
                if "looseness" in via_sub and via_sub["looseness"] is not None:
                    flat["viability_looseness"] = float(via_sub["looseness"])
                elif "viability_looseness" in via_sub and via_sub["viability_looseness"] is not None:
                    flat["viability_looseness"] = float(via_sub["viability_looseness"])
            dmp_sub = cln_sec.get("dump_gating")
            if isinstance(dmp_sub, dict):
                if "enabled" in dmp_sub:
                    flat["dump_enabled"] = bool(dmp_sub["enabled"])
                if "channel" in dmp_sub and dmp_sub["channel"] is not None:
                    flat["dump_channel"] = str(dmp_sub["channel"])
                if "cutoff" in dmp_sub and dmp_sub["cutoff"] is not None:
                    flat["dump_cutoff"] = float(dmp_sub["cutoff"])
                if "looseness" in dmp_sub and dmp_sub["looseness"] is not None:
                    flat["dump_looseness"] = float(dmp_sub["looseness"])
                elif "dump_looseness" in dmp_sub and dmp_sub["dump_looseness"] is not None:
                    flat["dump_looseness"] = float(dmp_sub["dump_looseness"])

        # Section: 04_transformation (or transformation)
        xform_sec = data.get("04_transformation", data.get("transformation", {}))
        if isinstance(xform_sec, dict) and xform_sec:
            flat["transformation"] = xform_sec
            if "enabled" in xform_sec:
                flat["enable_transformation"] = bool(xform_sec["enabled"])
            if "method" in xform_sec and xform_sec["method"]:
                flat["transform_type"] = str(xform_sec["method"]).lower()
            elif "transform_type" in xform_sec and xform_sec["transform_type"]:
                flat["transform_type"] = str(xform_sec["transform_type"]).lower()
            if "cofactor" in xform_sec and xform_sec["cofactor"] is not None:
                flat["transform_cofactor"] = float(xform_sec["cofactor"])
            if "channels" in xform_sec and xform_sec["channels"] is not None:
                raw_xf_ch = xform_sec["channels"]
                if isinstance(raw_xf_ch, str):
                    flat["transform_channels"] = raw_xf_ch
                elif isinstance(raw_xf_ch, list):
                    flat["transform_channels"] = [str(c) for c in raw_xf_ch if c]
            if "exclude_channels" in xform_sec and isinstance(xform_sec["exclude_channels"], list):
                flat["transform_exclude_channels"] = [str(c) for c in xform_sec["exclude_channels"] if c]

        # Section: 04_batch_preparation (or batch_preparation / batch)
        btch = data.get("04_batch_preparation", data.get("batch_preparation", data.get("batch", {})))
        if isinstance(btch, dict) and btch:
            flat["batch_preparation"] = btch
            if "enabled" in btch:
                flat["enable_batch_preparation"] = bool(btch["enabled"])
            if "batch_column" in btch:
                flat["batch_column"] = str(btch["batch_column"]) if btch["batch_column"] else None
            elif "batch_key" in btch:
                flat["batch_column"] = str(btch["batch_key"]) if btch["batch_key"] else None
            if "overwrite_autocheck" in btch:
                flat["overwrite_autocheck"] = bool(btch["overwrite_autocheck"])
            elif "override_autocheck" in btch:
                flat["overwrite_autocheck"] = bool(btch["override_autocheck"])

        # Section: population_discovery (or 05_population_discovery)
        pop = data.get("population_discovery", data.get("05_population_discovery", {}))
        if isinstance(pop, dict) and pop:
            flat["population_discovery"] = pop
            if "enabled" in pop:
                flat["enable_population_discovery"] = bool(pop["enabled"])
                flat["enable_discovery"] = bool(pop["enabled"])
            elif "enable_discovery" in pop:
                flat["enable_population_discovery"] = bool(pop["enable_discovery"])
                flat["enable_discovery"] = bool(pop["enable_discovery"])
            if "clustering_method" in pop:
                cm_val = str(pop["clustering_method"]).lower()
                flat["clustering_method"] = "som" if cm_val in ("flow_som", "flowsom") else cm_val
            elif "method" in pop:
                cm_val = str(pop["method"]).lower()
                flat["clustering_method"] = "som" if cm_val in ("flow_som", "flowsom") else cm_val
            if "n_clusters" in pop:
                flat["n_clusters"] = int(pop["n_clusters"]) if pop["n_clusters"] is not None else None
            elif "clusters" in pop:
                flat["n_clusters"] = int(pop["clusters"]) if pop["clusters"] is not None else None
            if "auto_k" in pop:
                flat["auto_k"] = bool(pop["auto_k"])
            if "auto_k_method" in pop:
                flat["auto_k_method"] = str(pop["auto_k_method"]).lower()
            elif "k_method" in pop:
                flat["auto_k_method"] = str(pop["k_method"]).lower()
            if "k_min" in pop:
                flat["k_min"] = int(pop["k_min"])
            if "k_max" in pop:
                flat["k_max"] = int(pop["k_max"])
            if "consensus_subsample" in pop:
                flat["consensus_subsample"] = float(pop["consensus_subsample"])
            if "consensus_iterations" in pop:
                flat["consensus_iterations"] = int(pop["consensus_iterations"])
            if "compute_umap" in pop:
                flat["compute_umap"] = bool(pop["compute_umap"])
            elif "umap" in pop:
                flat["compute_umap"] = bool(pop["umap"])
            if "umap_events_per_sample" in pop:
                flat["umap_events_per_sample"] = int(pop["umap_events_per_sample"])
            if "stratified_density_contours" in pop:
                flat["stratified_density_contours"] = bool(pop["stratified_density_contours"])
            if "umap_covariates" in pop:
                covs = pop["umap_covariates"]
                if isinstance(covs, list):
                    flat["umap_covariates"] = [str(c) for c in covs]
                elif isinstance(covs, str):
                    flat["umap_covariates"] = [str(c).strip() for c in covs.split(",") if str(c).strip()]
            if "export_prism" in pop:
                flat["export_prism"] = bool(pop["export_prism"])
            if "cluster_palette" in pop:
                flat["cluster_palette"] = str(pop["cluster_palette"])
            elif "palette" in pop:
                flat["cluster_palette"] = str(pop["palette"])
            if pop.get("cluster_name_map") is not None:
                flat["cluster_name_map"] = dict(pop["cluster_name_map"])
            elif pop.get("name_map") is not None:
                flat["cluster_name_map"] = dict(pop["name_map"])
            if pop.get("cell_signatures") is not None:
                flat["cell_signatures"] = pop["cell_signatures"]
            elif pop.get("signatures_file") is not None:
                flat["cell_signatures"] = pop["signatures_file"]
            if "lineage_markers" in pop and pop["lineage_markers"] is not None:
                raw_lm = pop["lineage_markers"]
                if isinstance(raw_lm, str):
                    flat["lineage_markers"] = raw_lm
                elif isinstance(raw_lm, list):
                    flat["lineage_markers"] = [str(m) for m in raw_lm if m]
            if "marker_roles" in pop and pop["marker_roles"] is not None:
                raw_roles = pop["marker_roles"]
                if isinstance(raw_roles, str):
                    flat["marker_roles"] = [r.strip() for r in raw_roles.split(",") if r.strip()]
                elif isinstance(raw_roles, list):
                    flat["marker_roles"] = [str(r).strip() for r in raw_roles if str(r).strip()]
            if "include_markers" in pop and isinstance(pop["include_markers"], list):
                flat["include_markers"] = [str(m) for m in pop["include_markers"] if m]
            if "exclude_markers" in pop and isinstance(pop["exclude_markers"], list):
                flat["exclude_markers"] = [str(m) for m in pop["exclude_markers"] if m]

            # Support nested clustering / dim_reduction sub-dictionaries (spectral config schema)
            clust_sub = pop.get("clustering")
            if isinstance(clust_sub, dict):
                if "method" in clust_sub and clust_sub["method"]:
                    cm_val = str(clust_sub["method"]).lower()
                    flat["clustering_method"] = "som" if cm_val in ("flow_som", "flowsom") else cm_val
                if "k" in clust_sub and clust_sub["k"] is not None:
                    flat["n_clusters"] = int(clust_sub["k"])
                    if "auto_k" not in clust_sub and "auto_k" not in pop:
                        flat["auto_k"] = False
                elif "n_clusters" in clust_sub and clust_sub["n_clusters"] is not None:
                    flat["n_clusters"] = int(clust_sub["n_clusters"])
                    if "auto_k" not in clust_sub and "auto_k" not in pop:
                        flat["auto_k"] = False
                if "auto_k" in clust_sub:
                    flat["auto_k"] = bool(clust_sub["auto_k"])
                if "k_min" in clust_sub and clust_sub["k_min"] is not None:
                    flat["k_min"] = int(clust_sub["k_min"])
                if "k_max" in clust_sub and clust_sub["k_max"] is not None:
                    flat["k_max"] = int(clust_sub["k_max"])
            dim_sub = pop.get("dim_reduction")
            if isinstance(dim_sub, dict):
                if "enabled" in dim_sub:
                    flat["compute_umap"] = bool(dim_sub["enabled"])
                elif "method" in dim_sub:
                    flat["compute_umap"] = str(dim_sub["method"]).lower() in ("umap", "tsne", "true")

        # Section: statistical_analysis (or 06_statistical_analysis)
        stats = data.get("statistical_analysis", data.get("06_statistical_analysis", {}))
        if isinstance(stats, dict) and stats:
            flat["statistical_analysis"] = stats
            if "enabled" in stats:
                flat["enable_statistical_analysis"] = bool(stats["enabled"])
            if "metadata_file" in stats:
                flat["metadata_file"] = str(stats["metadata_file"]) if stats["metadata_file"] else None
            elif "sample_metadata" in stats:
                flat["metadata_file"] = str(stats["sample_metadata"]) if stats["sample_metadata"] else None
            elif "metadata" in stats:
                flat["metadata_file"] = str(stats["metadata"]) if stats["metadata"] else None
            if "condition_column" in stats:
                flat["condition_column"] = str(stats["condition_column"]) if stats["condition_column"] else None
            elif "condition" in stats:
                flat["condition_column"] = str(stats["condition"]) if stats["condition"] else None
            if "reference_group" in stats:
                flat["reference_group"] = str(stats["reference_group"]) if stats["reference_group"] else None
            elif "reference" in stats:
                flat["reference_group"] = str(stats["reference"]) if stats["reference"] else None
            if "target_group" in stats:
                flat["target_group"] = str(stats["target_group"]) if stats["target_group"] else None
            elif "target" in stats:
                flat["target_group"] = str(stats["target"]) if stats["target"] else None
            if "paired_column" in stats:
                flat["paired_column"] = str(stats["paired_column"]) if stats["paired_column"] else None
            elif "paired" in stats:
                flat["paired_column"] = str(stats["paired"]) if stats["paired"] else None
            if "da_test_method" in stats:
                flat["da_test_method"] = str(stats["da_test_method"])
            if "ds_test_method" in stats:
                flat["ds_test_method"] = str(stats["ds_test_method"])
            if "stats_granularity" in stats and stats["stats_granularity"]:
                flat["stats_granularity"] = str(stats["stats_granularity"]).lower()
            elif "stats_level" in stats and stats["stats_level"]:
                flat["stats_granularity"] = str(stats["stats_level"]).lower()
            if "marker_roles" in stats and stats["marker_roles"] is not None:
                raw_s_roles = stats["marker_roles"]
                if isinstance(raw_s_roles, str):
                    flat["stats_marker_roles"] = [r.strip() for r in raw_s_roles.split(",") if r.strip()]
                elif isinstance(raw_s_roles, (list, tuple)):
                    flat["stats_marker_roles"] = [str(r).strip() for r in raw_s_roles if str(r).strip()]
            if "fdr_threshold" in stats:
                flat["fdr_threshold"] = float(stats["fdr_threshold"])
            if "log2fc_threshold" in stats:
                flat["log2fc_threshold"] = float(stats["log2fc_threshold"])
            if "generate_forest_plot" in stats:
                flat["generate_forest_plot"] = bool(stats["generate_forest_plot"])
            if "generate_ds_heatmap" in stats:
                flat["generate_ds_heatmap"] = bool(stats["generate_ds_heatmap"])
            if "generate_abundance_boxplots" in stats:
                flat["generate_abundance_boxplots"] = bool(stats["generate_abundance_boxplots"])
            if "generate_volcano_plot" in stats:
                flat["generate_volcano_plot"] = bool(stats["generate_volcano_plot"])

        # Section: exports (or workspaces_reporting / 07_workspaces_reporting)
        rep = data.get("exports", data.get("workspaces_reporting", data.get("07_workspaces_reporting", {})))
        if isinstance(rep, dict) and rep:
            flat["exports"] = rep
            flat["workspaces_reporting"] = rep
            if "enabled" in rep:
                flat["enable_workspaces_reporting"] = bool(rep["enabled"])
            if "export_augmented_fcs" in rep:
                flat["export_augmented_fcs"] = bool(rep["export_augmented_fcs"])
            elif "augmented_fcs" in rep:
                flat["export_augmented_fcs"] = bool(rep["augmented_fcs"])
            if "export_flowjo_wsp" in rep:
                flat["export_flowjo_wsp"] = bool(rep["export_flowjo_wsp"])
            elif "flowjo_wsp" in rep:
                flat["export_flowjo_wsp"] = bool(rep["flowjo_wsp"])
            if "export_gating_ml" in rep:
                flat["export_gating_ml"] = bool(rep["export_gating_ml"])
            elif "gating_ml" in rep:
                flat["export_gating_ml"] = bool(rep["gating_ml"])
            if "generate_miflowcyt_report" in rep:
                flat["generate_miflowcyt_report"] = bool(rep["generate_miflowcyt_report"])
            elif "miflowcyt_report" in rep:
                flat["generate_miflowcyt_report"] = bool(rep["miflowcyt_report"])

        # 2. Overlay any top-level keys
        for key, val in data.items():
            if key in (
                "input",
                "analysis",
                "visualization",
                "output",
                "outputs",
                "cleanup",
                "01_time_qc",
                "time_qc",
                "02_compensation",
                "compensation",
                "03_cleanup_gating",
                "cleanup_gating",
                "04_transformation",
                "transformation",
                "04_batch_preparation",
                "batch_preparation",
                "batch",
                "population_discovery",
                "05_population_discovery",
                "statistical_analysis",
                "06_statistical_analysis",
                "workspaces_reporting",
                "07_workspaces_reporting",
                "exports",
            ):
                continue
            if key == "no_timestamp":
                flat["timestamp_dir"] = not bool(val)
            elif key in ("workspace", "workspace_path"):
                flat["workspace_file"] = val
            elif key in ("time_qc", "enable_time_qc"):
                flat["enable_time_qc"] = bool(val)
            elif key == "time_qc_bin_width":
                flat["time_qc_bin_width"] = float(val) if val is not None else None
            elif key in ("metadata_file", "sample_metadata", "metadata"):
                flat["metadata_file"] = str(val) if val else None
            elif key in ("markers_file", "marker_metadata", "markers_metadata"):
                flat["markers_file"] = str(val) if val else None
            elif key in ("cell_signatures", "signatures_file"):
                flat["cell_signatures"] = val
            elif key in ("harmonization", "harmonization_file"):
                if isinstance(val, str):
                    flat["harmonization_file"] = val
                    flat["enable_harmonization"] = True
                elif isinstance(val, bool):
                    flat["enable_harmonization"] = val
                elif isinstance(val, dict):
                    flat["harmonization_config"] = val
                    flat["enable_harmonization"] = True
            elif key in cls.__dataclass_fields__:
                flat[key] = val

        config = cls(**flat)
        config.validate()
        return config

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> PipelineConfig:
        """Load configuration from a YAML or JSON file.

        Args:
            filepath: Path to .yaml, .yml, or .json file.

        Returns:
            Validated PipelineConfig instance.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        ext = path.suffix.lower()
        content = path.read_text(encoding="utf-8")

        if ext in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError(
                    "PyYAML is required to parse YAML config files. Install via 'pip install pyyaml' "
                    "or provide a .json config file instead."
                )
            data = yaml.safe_load(content) or {}
        elif ext == ".json":
            data = json.loads(content)
        else:
            # Fallback: try JSON first, then YAML
            try:
                data = json.loads(content)
            except Exception:
                if HAS_YAML:
                    data = yaml.safe_load(content) or {}
                else:
                    raise ValueError(f"Unsupported config extension '{ext}'. Use .yaml or .json.")

        if not isinstance(data, dict):
            raise ValueError(f"Config content in {path} must be a dictionary/mapping, got {type(data).__name__}.")

        config = cls.from_dict(data)
        config.config_file_path = sanitize_path(path)
        return config

    def to_dict(self, nested: bool = True) -> Dict[str, Any]:
        """Convert configuration to dictionary, optionally hierarchically structured."""
        if not nested:
            return asdict(self)

        return {
            "input": {
                "fcs_file": self.fcs_file,
                "data_dir": self.data_dir,
                "files_list": self.files_list,
                "technology": self.technology,
                "metadata_file": self.metadata_file,
                "markers_file": self.markers_file,
                "marker_mapping": self.marker_mapping,
                "harmonization_file": self.harmonization_file,
                "enable_harmonization": self.enable_harmonization,
            },
            "analysis": {
                "gating_mode": self.gating_mode,
                "cluster_parent": self.cluster_parent,
                "qc_method": self.qc_method,
                "adaptive_gating": self.adaptive_gating,
                "seed": self.seed,
                "workspace_file": self.workspace_file,
                "log_level": self.log_level,
                "transform_type": self.transform_type,
                "enable_umap": self.enable_umap,
                "umap_subsample": self.umap_subsample,
            },
            "visualization": {
                "legend_mode": self.legend_mode,
            },
            "output": {
                "output_dir": self.output_dir,
                "timestamp_dir": self.timestamp_dir,
                "overwrite": self.overwrite,
            },
            "cleanup": {
                "debris_method": self.debris_method,
                "debris_sd_scale": self.debris_sd_scale,
                "debris_min_probability": self.debris_min_probability,
                "singlet_mode": self.singlet_mode,
                "singlet_method": self.singlet_method,
                "singlet_tolerance": self.singlet_tolerance,
                "viability_cutoff": self.viability_cutoff,
                "viability_looseness": self.viability_looseness,
                "enable_viability": self.enable_viability,
                "dump_looseness": self.dump_looseness,
                "export_clean_fcs": self.export_clean_fcs,
            },
            "01_time_qc": {
                **(self.time_qc or {}),
                "enabled": self.enable_time_qc,
                "bin_width": self.time_qc_bin_width,
            },
            "02_compensation": {
                **(self.compensation or {}),
                "enabled": self.enable_compensation,
                "source": self.compensation_source,
                "controls_dir": self.comp_controls_dir,
                "manifest": self.comp_manifest,
                "unstained_control": self.unstained_control,
            },
            "03_cleanup_gating": {
                "enabled": self.enable_cleanup_gating,
                "debris_method": self.debris_method,
                "debris_n_components": self.debris_n_components,
                "debris_sd_scale": self.debris_sd_scale,
                "debris_min_probability": self.debris_min_probability,
                "fsc_channel": self.fsc_channel,
                "ssc_channel": self.ssc_channel,
                "fsc_height_channel": self.fsc_height_channel,
                "ssc_height_channel": self.ssc_height_channel,
                "singlet_mode": self.singlet_mode,
                "singlet_method": self.singlet_method,
                "singlet_tolerance": self.singlet_tolerance,
                "viability_channel": self.viability_channel,
                "viability_cutoff": self.viability_cutoff,
                "viability_looseness": self.viability_looseness,
                "enable_viability": self.enable_viability,
                "dump_enabled": self.dump_enabled,
                "dump_channel": self.dump_channel,
                "dump_cutoff": self.dump_cutoff,
                "dump_looseness": self.dump_looseness,
                "export_clean_fcs": self.export_clean_fcs,
            },
            "04_transformation": {
                **(self.transformation or {}),
                "enabled": self.enable_transformation,
                "method": self.transform_type,
                "cofactor": self.transform_cofactor,
                "channels": self.transform_channels,
                "exclude_channels": self.transform_exclude_channels,
            },
            "04_batch_preparation": {
                "enabled": self.enable_batch_preparation,
                "batch_column": self.batch_column,
                "assess_batch_effect": True,
                "overwrite_autocheck": self.overwrite_autocheck,
                **(self.batch_preparation or {
                    "target_events": 50000,
                    "mode": "min_sample",
                }),
            },
            "05_population_discovery": {
                "enabled": self.enable_population_discovery,
                "clustering_method": self.clustering_method,
                "n_clusters": self.n_clusters,
                "auto_k": self.auto_k,
                "auto_k_method": self.auto_k_method,
                "k_min": self.k_min,
                "k_max": self.k_max,
                "consensus_subsample": self.consensus_subsample,
                "consensus_iterations": self.consensus_iterations,
                "compute_umap": self.compute_umap,
                "umap_events_per_sample": self.umap_events_per_sample,
                "stratified_density_contours": self.stratified_density_contours,
                "umap_covariates": self.umap_covariates,
                "cluster_palette": self.cluster_palette,
                "export_prism": self.export_prism,
                "cluster_name_map": self.cluster_name_map,
                "cell_signatures": self.cell_signatures,
                "lineage_markers": self.lineage_markers,
                "marker_roles": self.marker_roles,
                "include_markers": self.include_markers,
                "exclude_markers": self.exclude_markers,
            },
            "06_statistical_analysis": {
                "enabled": self.enable_statistical_analysis,
                "metadata_file": self.metadata_file,
                "condition_column": self.condition_column,
                "reference_group": self.reference_group,
                "target_group": self.target_group,
                "paired_column": self.paired_column,
                "da_test_method": self.da_test_method,
                "ds_test_method": self.ds_test_method,
                "stats_granularity": self.stats_granularity,
                "marker_roles": self.stats_marker_roles,
                "fdr_threshold": self.fdr_threshold,
                "log2fc_threshold": self.log2fc_threshold,
                "generate_forest_plot": self.generate_forest_plot,
                "generate_ds_heatmap": self.generate_ds_heatmap,
                "generate_abundance_boxplots": self.generate_abundance_boxplots,
                "generate_volcano_plot": self.generate_volcano_plot,
            },
            "exports": {
                "enabled": self.enable_workspaces_reporting,
                "export_augmented_fcs": self.export_augmented_fcs,
                "export_flowjo_wsp": self.export_flowjo_wsp,
                "export_gating_ml": self.export_gating_ml,
                "generate_miflowcyt_report": self.generate_miflowcyt_report,
            },
        }

    def save(self, filepath: Union[str, Path], nested: bool = True) -> Path:
        """Save configuration to a YAML or JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()
        data = self.to_dict(nested=nested)

        if ext in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError("PyYAML is required to serialize YAML files. Use .json instead.")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
        elif ext == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config extension '{ext}'. Use .yaml or .json.")

        return path

    @staticmethod
    def generate_template(format: str = "yaml") -> str:
        """Generate a commented starter template configuration string."""
        fmt = format.lower()
        if fmt in ("yaml", "yml"):
            return """# ==============================================================================
# Flow Cytometry Pipeline Configuration Template
# ==============================================================================

input:
  # Path to raw input .fcs file (can also be passed as CLI argument)
  fcs_file: "data/conventional/tqc_2020_basic_tutorial/raw/LD1_NS+NS_A01_exp.fcs"

  # Path to curated cohort samples.csv (auto-discovers metadata/samples.csv if omitted)
  metadata_file: null

  # Path to curated antibody panel markers.csv (auto-discovers metadata/markers.csv if omitted)
  markers_file: null
  
  # Optional custom marker-to-detector mappings (e.g. CD3: "Ax700-A")
  marker_mapping: null

analysis:
  # Gating methodology:
  # - 'som': Unsupervised Self-Organizing Map clustering (default)
  # - 'template': Standard hierarchical gating
  # - 'phenograph': Graph-based community detection clustering
  # - 'compare': All three methodologies computed and cross-compared
  gating_mode: "som"
  
  # Quality Control (QC) methodology for debris & doublet exclusion:
  # - 'clustering': Automated Gaussian Mixture Model deconvolution (default)
  # - 'template': Standard polygon templates
  qc_method: "clustering"
  
  # Data-driven adaptive density valley-seeking (openCyto / flowDensity KDE)
  adaptive_gating: true
  
  # Random seed for reproducible clustering and sampling
  seed: 42
  
  # Optional path to existing workspace session (.json or .xml) to replay
  workspace_file: null

visualization:
  # Legend styling: 'centroid' (direct badges at cluster centers) or 'external' (side legend)
  legend_mode: "centroid"

output:
  # Base directory for plots, tables, and workspace manifests
  output_dir: "outputs/experiment_run"
  
  # If true, creates a timestamped run subdirectory (e.g. sample_YYYYMMDD_HHMMSS)
  timestamp_dir: false
  
  # If true, allows writing directly to output_dir without timestamping
  overwrite: true

# Modular Step Pipeline Configurations
01_time_qc:
  enabled: true
  bin_width: null

02_compensation:
  enabled: true
  source: "auto"

03_cleanup_gating:
  enabled: true
  debris_method: "gmm"
  singlet_mode: "dual"
  enable_viability: true
  export_clean_fcs: false

04_batch_preparation:
  enabled: true
  target_events: 50000
  mode: "min_sample"

05_population_discovery:
  enabled: true
  clustering_method: "som"
  n_clusters: 10
  auto_k: true
  auto_k_method: "consensus"
  consensus_subsample: 0.8
  consensus_iterations: 50
  compute_umap: true
  export_prism: true
  cluster_name_map: null
  cell_signatures: null

06_statistical_analysis:
  enabled: true
  metadata_file: null
  condition_column: "condition"
  reference_group: null
  target_group: null

exports:
  enabled: true
  export_augmented_fcs: true
  export_flowjo_wsp: true
  export_gating_ml: true
  generate_miflowcyt_report: true
"""
        elif fmt == "json":
            tmpl = {
                "input": {
                    "fcs_file": "data/conventional/tqc_2020_basic_tutorial/raw/LD1_NS+NS_A01_exp.fcs",
                    "marker_mapping": None,
                },
                "analysis": {
                    "gating_mode": "som",
                    "qc_method": "clustering",
                    "adaptive_gating": True,
                    "seed": 42,
                    "workspace_file": None,
                },
                "visualization": {
                    "legend_mode": "centroid",
                },
                "output": {
                    "output_dir": "outputs/experiment_run",
                    "timestamp_dir": False,
                    "overwrite": True,
                },
            }
            return json.dumps(tmpl, indent=2)
        else:
            raise ValueError(f"Unknown template format '{format}'. Use 'yaml' or 'json'.")

    def to_time_qc_config(self) -> Any:
        """Build a step-level TimeQCConfig from this PipelineConfig."""
        from flowx.qc.models import TimeQCConfig

        tqc_sec = self.time_qc or {}
        kwargs: Dict[str, Any] = {}
        if self.time_qc_bin_width is not None:
            kwargs["bin_width_sec"] = float(self.time_qc_bin_width)
        elif "bin_width_sec" in tqc_sec and tqc_sec["bin_width_sec"] is not None:
            kwargs["bin_width_sec"] = float(tqc_sec["bin_width_sec"])
        for k in (
            "sample_type",
            "events_per_bin",
            "flow_rate_mad_factor",
            "flow_rate_min_eps",
            "enable_signal_drift",
            "signal_drift_mad_factor",
            "signal_drift_slope_threshold",
            "min_channels_drift_flag",
            "trim_margins",
            "margin_tolerance",
            "manual_time_min",
            "manual_time_max",
            "manual_exclude_intervals",
            "channels_to_evaluate",
            "pass_threshold_pct",
            "warn_threshold_pct",
            "ignore_zero_bins_for_baseline",
        ):
            if k in tqc_sec and tqc_sec[k] is not None:
                kwargs[k] = tqc_sec[k]
        if "num_bins" in tqc_sec and tqc_sec["num_bins"] is not None and "events_per_bin" not in kwargs:
            kwargs["events_per_bin"] = None  # keep default bin_width_sec unless explicit
        sample_type = str(kwargs.pop("sample_type", "sample"))
        if sample_type == "control":
            return TimeQCConfig.for_controls(**kwargs)
        if sample_type == "fmo":
            return TimeQCConfig.for_fmo(**kwargs)
        return TimeQCConfig.for_sample(**kwargs)

    def to_compensation_config(self) -> Any:
        """Build a step-level CompensationConfig from this PipelineConfig."""
        from flowx.math.compensation import CompensationConfig

        cmp_sec = self.compensation or {}
        kwargs: Dict[str, Any] = {
            "source": self.compensation_source,
            "apply_time_qc_to_sample": self.enable_time_qc,
            "apply_time_qc_to_controls": self.enable_time_qc,
            "random_seed": self.seed,
        }
        est = cmp_sec.get("estimation_method") or cmp_sec.get("algorithm")
        if est:
            kwargs["estimation_method"] = str(est)
        elif "method" in cmp_sec and str(cmp_sec["method"]).lower() in (
            "hybrid",
            "median_delta_mfi",
            "autospill_regression",
        ):
            kwargs["estimation_method"] = str(cmp_sec["method"]).lower()
        for k in (
            "apply_time_qc_to_controls",
            "apply_time_qc_to_sample",
            "auto_scatter_singlet_gate",
            "control_carrier_type",
            "negative_reference",
            "clamp_negative",
            "autospill_iterations",
            "fallback_policy",
            "manual_overrides",
            "max_condition_number_warn",
            "residual_z_threshold",
            "compute_ssm",
            "align_sample_tandem_mismatch",
            "min_control_events",
            "sample_nxn_gating_mode",
            "voltage_mismatch_policy",
            "voltage_tolerance_v",
            "pmt_gain_exponent_alpha",
        ):
            if k in cmp_sec and cmp_sec[k] is not None:
                kwargs[k] = cmp_sec[k]
        return CompensationConfig(**kwargs)

    def to_unmixing_config(self) -> Any:
        """Build a step-level spectral UnmixingConfig from this PipelineConfig."""
        from flowx.spectral.models import UnmixingConfig

        cmp_sec = self.compensation or {}
        kwargs: Dict[str, Any] = {"random_seed": self.seed}
        solver = cmp_sec.get("solver") or cmp_sec.get("unmixing_method")
        if not solver and str(cmp_sec.get("method", "")).lower() in ("wls", "ols", "nnls"):
            solver = str(cmp_sec["method"]).lower()
        if solver:
            kwargs["method"] = str(solver).lower()
        af_mode = cmp_sec.get("autofluorescence_mode") or cmp_sec.get("af_mode")
        if af_mode:
            kwargs["autofluorescence_mode"] = str(af_mode).lower()
        for k in (
            "alpha_poisson",
            "electronic_noise",
            "allow_negative_slack",
            "negative_slack_floor",
            "compute_residuals",
            "residual_outlier_sigma",
            "multi_af_components",
            "min_control_events",
        ):
            if k in cmp_sec and cmp_sec[k] is not None:
                kwargs[k] = cmp_sec[k]
        return UnmixingConfig(**kwargs)

    def to_transform_config(self) -> Any:
        """Build a step-level TransformConfig from this PipelineConfig."""
        from flowx.math.transforms import TransformConfig

        xf_sec = self.transformation or {}
        xf_channels: Any = None
        if isinstance(self.transform_channels, str):
            xf_channels = self.transform_channels
        elif isinstance(self.transform_channels, list):
            xf_channels = list(self.transform_channels)
        excl_chs = self.transform_exclude_channels
        if excl_chs is None and isinstance(xf_sec.get("exclude_channels"), list):
            excl_chs = [str(c) for c in xf_sec.get("exclude_channels", []) if c]
        return TransformConfig(
            enabled=self.enable_transformation,
            method=str(self.transform_type or "logicle").lower(),
            cofactor=float(self.transform_cofactor),
            channels=xf_channels,
            exclude_channels=list(excl_chs) if excl_chs else None,
            param_w=float(xf_sec.get("param_w", 0.5)),
            param_a=float(xf_sec.get("param_a", 0.0)),
            normalized=bool(xf_sec.get("normalized", True)),
        )

    def to_cleanup_config(self) -> Any:
        """Build a step-level CleanUpGateConfig from this PipelineConfig."""
        from flowx.gating.cleanup import CleanUpGateConfig

        cln_sec = self.cleanup_gating or {}
        kwargs: Dict[str, Any] = {
            "debris_method": self.debris_method,
            "debris_n_components": self.debris_n_components,
            "debris_sd_scale": self.debris_sd_scale,
            "debris_min_probability": self.debris_min_probability,
            "fsc_channel": self.fsc_channel,
            "ssc_channel": self.ssc_channel,
            "fsc_height_channel": self.fsc_height_channel,
            "ssc_height_channel": self.ssc_height_channel,
            "singlet_mode": self.singlet_mode,
            "singlet_method": self.singlet_method,
            "singlet_tolerance": self.singlet_tolerance,
            "viability_enabled": self.enable_viability,
            "viability_channel": self.viability_channel,
            "viability_cutoff": self.viability_cutoff,
            "viability_looseness": self.viability_looseness,
            "dump_enabled": self.dump_enabled,
            "dump_channel": self.dump_channel,
            "dump_cutoff": self.dump_cutoff,
            "dump_looseness": self.dump_looseness,
        }
        for k in (
            "debris_vertices",
            "singlet_fsc_enabled",
            "singlet_ssc_enabled",
            "viability_gate_range",
            "dump_gate_range",
            "min_events_debris",
            "min_events_singlet",
            "min_events_viability",
            "show_cluster_shades",
        ):
            if k in cln_sec and cln_sec[k] is not None:
                kwargs[k] = cln_sec[k]
        return CleanUpGateConfig(**kwargs)

    def to_batch_preparation_config(self, output_dir: Optional[Union[str, Path]] = None) -> Any:
        """Build a step-level BatchPreparationConfig from this PipelineConfig."""
        from flowx.batch.models import BatchPreparationConfig

        bp_sec = dict(self.batch_preparation or {})
        if "enabled" not in bp_sec:
            bp_sec["enabled"] = self.enable_batch_preparation
        if "assess_batch_effect" not in bp_sec:
            bp_sec["assess_batch_effect"] = True
        if "overwrite_autocheck" not in bp_sec and "override_autocheck" not in bp_sec:
            bp_sec["overwrite_autocheck"] = self.overwrite_autocheck
        if self.batch_column and "batch_column" not in bp_sec and "batch_key" not in bp_sec:
            bp_sec["batch_column"] = self.batch_column
        if "random_seed" not in bp_sec and "seed" not in bp_sec:
            bp_sec["random_seed"] = self.seed
        if output_dir is not None:
            bp_sec["output_dir"] = str(output_dir)
        elif "output_dir" not in bp_sec:
            bp_sec["output_dir"] = str(Path(self.output_dir) / "04_batch_preparation")
        return BatchPreparationConfig.from_dict(bp_sec)

    def to_pipeline(self) -> Tuple[Any, Optional[str]]:
        """Instantiate a FlowCytometryPipeline and return (pipeline, fcs_file).

        Returns:
            Tuple containing (FlowCytometryPipeline instance, resolved fcs_filepath string or None).
        """
        # Lazy import to avoid circular dependency
        from flowx.pipeline import FlowCytometryPipeline
        from flowx.workspace.session import WorkspaceSession

        custom_hierarchy = None
        if self.workspace_file:
            wsp_path = Path(self.workspace_file)
            if wsp_path.is_file():
                if wsp_path.suffix.lower() == ".json":
                    session = WorkspaceSession.load_json(wsp_path)
                else:
                    session = WorkspaceSession.load_xml(wsp_path)
                custom_hierarchy = session.hierarchy

        pipeline = FlowCytometryPipeline(
            output_dir=self.output_dir,
            random_seed=self.seed,
            timestamp_dir=self.timestamp_dir,
            overwrite=self.overwrite,
            gating_mode=self.gating_mode,
            cluster_parent=self.cluster_parent,
            qc_method=self.qc_method,
            legend_mode=self.legend_mode,
            adaptive_gating=self.adaptive_gating,
            custom_hierarchy=custom_hierarchy,
            marker_mapping=self.marker_mapping,
            comp_controls_dir=self.comp_controls_dir,
            comp_manifest=self.comp_manifest,
            unstained_control=self.unstained_control,
            compensation_source=self.compensation_source,
            comp_channel_mapping=self.comp_channel_mapping,
            config_file=self.config_file_path,
            config_source=self,
            log_level=self.log_level,
            transform_type=self.transform_type,
            enable_umap=self.enable_umap,
            umap_subsample=self.umap_subsample,
            enable_time_qc=self.enable_time_qc,
            harmonization_file=self.harmonization_file,
            enable_harmonization=self.enable_harmonization,
            harmonization_config=self.harmonization_config,
            debris_method=self.debris_method,
            debris_n_components=self.debris_n_components,
            debris_sd_scale=self.debris_sd_scale,
            debris_min_probability=self.debris_min_probability,
            singlet_mode=self.singlet_mode,
            singlet_method=self.singlet_method,
            singlet_tolerance=self.singlet_tolerance,
            viability_cutoff=self.viability_cutoff,
            viability_channel=self.viability_channel,
            viability_looseness=self.viability_looseness,
            enable_viability=self.enable_viability,
            dump_channel=self.dump_channel,
            dump_looseness=self.dump_looseness,
            dump_enabled=self.dump_enabled,
            export_clean_fcs=self.export_clean_fcs,
            enable_qc=self.enable_qc,
            enable_discovery=self.enable_discovery,
            numbered_steps=self.numbered_steps,
            enable_compensation=self.enable_compensation,
            enable_cleanup_gating=self.enable_cleanup_gating,
            enable_transformation=self.enable_transformation,
            transform_cofactor=self.transform_cofactor,
            transform_config=self.to_transform_config(),
            time_qc_config=self.to_time_qc_config(),
            comp_config=self.to_compensation_config(),
            spectral_config=(
                self.to_unmixing_config()
                if (
                    self.technology == "spectral"
                    or self.compensation_source == "spectral"
                    or (
                        self.compensation
                        and (
                            str(self.compensation.get("method", "")).lower() in ("unmix", "unmixing", "spectral", "wls", "ols", "nnls")
                            or any(k in self.compensation for k in ("solver", "unmixing_method", "af_mode", "autofluorescence_mode", "alpha_poisson"))
                        )
                    )
                )
                else None
            ),
            cleanup_config=self.to_cleanup_config(),
            technology=self.technology,
            metadata_file=self.metadata_file,
            markers_file=self.markers_file,
            cell_signatures=self.cell_signatures,
            lineage_markers=self.lineage_markers,
            marker_roles=self.marker_roles,
            stats_marker_roles=self.stats_marker_roles,
            include_markers=self.include_markers,
            exclude_markers=self.exclude_markers,
            data_dir=self.data_dir,
            files_list=self.files_list,
            clustering_method=self.clustering_method,
            n_clusters=self.n_clusters,
            auto_k=self.auto_k,
            auto_k_method=self.auto_k_method,
            k_min=self.k_min,
            k_max=self.k_max,
            consensus_subsample=self.consensus_subsample,
            consensus_iterations=self.consensus_iterations,
            umap_events_per_sample=self.umap_events_per_sample,
            condition_column=self.condition_column,
            batch_column=self.batch_column,
            umap_covariates=self.umap_covariates,
            reference_group=self.reference_group,
            target_group=self.target_group,
            paired_column=self.paired_column,
            da_test_method=self.da_test_method,
            ds_test_method=self.ds_test_method,
            stats_granularity=self.stats_granularity,
            fdr_threshold=self.fdr_threshold,
            log2fc_threshold=self.log2fc_threshold,
            stratified_density_contours=self.stratified_density_contours,
            generate_forest_plot=self.generate_forest_plot,
            generate_ds_heatmap=self.generate_ds_heatmap,
            generate_abundance_boxplots=self.generate_abundance_boxplots,
            generate_volcano_plot=self.generate_volcano_plot,
            cluster_palette=self.cluster_palette,
            export_prism=self.export_prism,
            export_augmented_fcs=self.export_augmented_fcs,
            export_flowjo_wsp=self.export_flowjo_wsp,
            export_gating_ml=self.export_gating_ml,
            generate_miflowcyt_report=self.generate_miflowcyt_report,
            cluster_name_map=self.cluster_name_map,
            batch_preparation=self.batch_preparation,
            enable_batch_preparation=self.enable_batch_preparation,
            overwrite_autocheck=self.overwrite_autocheck,
        )
        target_fcs = self.fcs_file or self.data_dir or self.files_list
        return pipeline, target_fcs

    def run(self, fcs_override: Optional[Union[str, Path]] = None) -> Any:
        """Run the pipeline end-to-end using this configuration.

        Args:
            fcs_override: Optional path to FCS file overriding self.fcs_file.

        Returns:
            PipelineResult instance.
        """
        target_fcs = fcs_override or self.fcs_file or self.data_dir or self.files_list
        if not target_fcs:
            raise ValueError("No input FCS file or cohort directory specified in configuration or arguments.")

        pipeline, _ = self.to_pipeline()
        return pipeline.run(target_fcs)
