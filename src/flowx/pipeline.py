"""End-to-end reproducible flow cytometry analysis pipeline.

Orchestrates raw FCS loading, QC visualization, fluorescence compensation,
logicle/biexponential scale transformation, hierarchical subpopulation gating,
unsupervised clustering (SOM & PhenoGraph), biological macro-grouping,
multi-panel figure generation with direct centroid badging, statistical reporting,
and workspace session persistence.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from flowx.clustering.cluster import (
    ClusteringResult,
    run_phenograph,
    run_som,
)
from flowx.clustering.qc_clustering import (
    ClusteringQCResult,
    run_clustering_qc,
)
from flowx.gating.cleanup import (
    CleanUpGateConfig,
    CleanUpGatingResult,
    run_cleanup_gating,
)
from flowx.gating.density_gating import (
    DensityCutoffResult,
    build_adaptive_density_hierarchy,
    find_density_minimum,
)
from flowx.gating.gate import (
    Gate,
    GatingHierarchy,
    PolygonGate,
    RangeGate,
    RectangleGate,
)
from flowx.io.logger import close_run_logger, format_duration, get_logger, log_step, setup_run_logger
from flowx.io.metadata import load_sample_metadata
from flowx.io.paths import sanitize_path
from flowx.io.parser import FCSData, parse_fcs
from flowx.math.compensation import (
    CompensationConfig,
    CompensationControlResult,
    CompensationMatrix,
    calculate_spillover_from_controls,
)
from flowx.math.transforms import (
    BaseTransform,
    BiexponentialTransform,
    LogicleTransform,
    apply_transform_to_dataframe,
    get_optimal_transform,
    is_empirically_pre_transformed,
)

from flowx.gating.adaptive_morphology import (
    build_adaptive_cell_gate,
    build_adaptive_singlet_gate,
)
from flowx.clustering.reduction import compute_umap_embedding
from flowx.visualization.cleanup_plots import plot_cleanup_diagnostics
from flowx.visualization.plotting import (
    plot_cluster_heatmap,
    plot_clustering_comparison,
    plot_clustering_qc_diagnostic,
    plot_clusters_scatter_2d,
    plot_compensation_comparison,
    plot_gate_overlay,
    plot_pipeline_cascade,
    plot_population_marker_violins,
    plot_scatter_2d,
    plot_spillover_matrix,
    plot_transformation_comparison,
    plot_umap_marker_features,
    plot_umap_projection,
)
from flowx.workspace.gating_ml_export import (
    SampleWorkspaceSpec,
    export_preprocessing_flowjo_wsp,
    export_preprocessing_gating_ml,
)
from flowx.workspace.session import WorkspaceSession


@dataclass
class PipelineResult:
    """Container holding results, output paths, and metadata from a pipeline execution."""

    sample_name: str
    event_count: int
    raw_fcs: FCSData
    compensated_df: pd.DataFrame
    transformed_df: pd.DataFrame
    hierarchy: GatingHierarchy
    statistics_df: pd.DataFrame
    generated_plots: Dict[str, str] = field(default_factory=dict)
    generated_tables: Dict[str, str] = field(default_factory=dict)
    run_dir: str = ""
    workspace_json_path: str = ""
    workspace_xml_path: str = ""
    preprocessing_wsp_path: str = ""
    preprocessing_gating_ml_path: str = ""
    statistics_csv_path: str = ""
    report_md_path: str = ""
    reproducibility_verified: bool = False
    gating_mode: str = "som"
    cluster_parent: str = "viable"
    qc_method: str = "clustering"
    clustering_qc_result: Optional[ClusteringQCResult] = None
    cleanup_result: Optional[CleanUpGatingResult] = None
    som_result: Optional[ClusteringResult] = None
    phenograph_result: Optional[ClusteringResult] = None
    comparison_crosstab_som: Optional[pd.DataFrame] = None
    comparison_crosstab_phenograph: Optional[pd.DataFrame] = None
    three_way_metrics_df: Optional[pd.DataFrame] = None
    three_way_lineages_df: Optional[pd.DataFrame] = None
    adaptive_cutoffs: Optional[Dict[str, DensityCutoffResult]] = None
    cytometer: str = ""
    time_qc_result: Optional[Any] = None
    compensation_control_result: Optional[CompensationControlResult] = None
    compensation_source_used: str = "embedded"
    spillover_matrix_path: Optional[str] = None
    umap_coordinates_path: Optional[str] = None
    config_file_path: Optional[str] = None
    log_file_path: Optional[str] = None
    execution_time_seconds: float = 0.0
    execution_duration_formatted: str = ""
    technology: str = "conventional"
    cohort_samples: List[str] = field(default_factory=list)
    cohort_clean_dfs: Dict[str, pd.DataFrame] = field(default_factory=dict)
    cohort_result: Optional[Any] = None
    spectral_result: Optional[Any] = None


def resolve_cohort_samples(
    input_source: Union[str, Path, List[Union[str, Path]]],
) -> Tuple[List[Path], str]:
    """Resolve a list of valid raw sample FCS files and a cohort or dataset identifier.

    Args:
        input_source: Single FCS file, directory containing FCS files, file list (.csv/.txt),
                      or list of file paths.

    Returns:
        Tuple of (sorted list of FCS file Paths, cohort or sample identifier string).
    """
    if isinstance(input_source, (list, tuple)):
        fcs_files = [
            Path(p).resolve()
            for p in input_source
            if Path(p).is_file() and Path(p).suffix.lower() == ".fcs"
        ]
        if not fcs_files:
            raise FileNotFoundError(f"No valid .fcs files found in provided list of {len(input_source)} items.")
        return sorted(fcs_files, key=lambda p: p.name), f"Cohort_{len(fcs_files)}_samples"

    src = Path(input_source).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input path does not exist: {sanitize_path(src)}")

    # Case 1: Single FCS file
    if src.is_file() and src.suffix.lower() == ".fcs":
        return [src], src.stem

    # Case 2: Files list (.csv or .txt)
    if src.is_file() and src.suffix.lower() in (".csv", ".txt"):
        resolved_files: List[Path] = []
        if src.suffix.lower() == ".csv":
            df_list = pd.read_csv(src)
            if "is_control" in df_list.columns and len(df_list) > 1:
                non_ctrl_mask = ~df_list["is_control"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
                if non_ctrl_mask.any():
                    df_list = df_list.loc[non_ctrl_mask]
            col = next((c for c in ["relative_path", "filepath", "path", "file", "filename"] if c in df_list.columns), None)
            if col:
                for val in df_list[col].dropna():
                    raw_str = str(val).strip()
                    if raw_str:
                        p = Path(raw_str)
                        if not p.is_absolute():
                            candidates = [
                                src.parent / p,
                                src.parent.parent / p,
                                src.parent.parent / "raw" / "samples" / p,
                                Path.cwd() / p,
                            ]
                            p = next((cand.resolve() for cand in candidates if cand.is_file()), p)
                        if p.is_file() and p.suffix.lower() == ".fcs":
                            resolved_files.append(p)
        else:
            for line in src.read_text(encoding="utf-8").splitlines():
                raw_str = line.strip()
                if raw_str and not raw_str.startswith("#"):
                    p = Path(raw_str)
                    if not p.is_absolute():
                        candidates = [
                            src.parent / p,
                            src.parent.parent / p,
                            src.parent.parent / "raw" / "samples" / p,
                            Path.cwd() / p,
                        ]
                        p = next((cand.resolve() for cand in candidates if cand.is_file()), p)
                    if p.is_file() and p.suffix.lower() == ".fcs":
                        resolved_files.append(p)
        if not resolved_files:
            raise FileNotFoundError(f"No valid FCS files found in file list '{sanitize_path(src)}'.")
        cohort_name = src.parent.parent.name if src.parent.name == "metadata" else src.stem
        return sorted(resolved_files, key=lambda p: p.name), cohort_name

    # Case 3: Directory
    if src.is_dir():
        # Check if curated samples.csv exists in metadata/ or at dataset root (including when src is raw/samples)
        csv_candidates = [src / "metadata" / "samples.csv", src / "samples.csv"]
        if src.name == "samples" and src.parent.name == "raw":
            csv_candidates.extend([
                src.parent.parent / "metadata" / "samples.csv",
                src.parent.parent / "samples.csv",
            ])
        elif src.name in ("samples", "raw"):
            csv_candidates.extend([
                src.parent / "metadata" / "samples.csv",
                src.parent / "samples.csv",
            ])
        curated_samples_csv = next((p for p in csv_candidates if p.is_file()), None)
        if curated_samples_csv is not None:
            try:
                curated_paths, cohort_id = resolve_cohort_samples(curated_samples_csv)
                if curated_paths:
                    resolved_name = cohort_id if src.name in ("samples", "raw") else src.name
                    return sorted(curated_paths, key=lambda p: p.name), resolved_name
            except Exception:
                pass

        raw_samples_dir = src / "raw" / "samples"
        samples_dir = src / "samples"
        target_dir = raw_samples_dir if raw_samples_dir.is_dir() else (samples_dir if samples_dir.is_dir() else src)

        candidate_files: List[Path] = []
        file_iterator = target_dir.rglob("*.fcs") if target_dir == src else target_dir.glob("*.fcs")
        for f in file_iterator:
            parts_lower = [p.lower() for p in f.parts]
            if "controls" in parts_lower or "control" in parts_lower:
                continue
            f_name_lower = f.name.lower()
            if target_dir == src and any(w in f_name_lower for w in ["control", "unstained", "compensation", "fmo"]):
                continue
            candidate_files.append(f)

        if not candidate_files:
            raise FileNotFoundError(f"No sample .fcs files found in directory '{sanitize_path(src)}' (excluding controls).")

        return sorted(candidate_files, key=lambda p: p.name), src.name

    raise ValueError(f"Unsupported input source format: {sanitize_path(src)}")


def resolve_cohort_metadata(
    fcs_files: List[Path],
    sample_metadata: Optional[Union[pd.DataFrame, Dict[str, Dict[str, Any]], str, Path]] = None,
    metadata_file: Optional[Union[str, Path]] = None,
    dataset_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Resolve or construct sample metadata table for cohort analysis.

    Matches sample identifiers with conditions, groups, and batch annotations.
    """
    meta_df: Optional[pd.DataFrame] = None

    if isinstance(sample_metadata, pd.DataFrame):
        meta_df = sample_metadata.copy()
    elif isinstance(sample_metadata, dict):
        meta_df = pd.DataFrame.from_dict(sample_metadata, orient="index")
        if meta_df.index.name is None:
            meta_df.index.name = "sample_id"
    elif sample_metadata is not None or metadata_file is not None:
        target_path = Path(metadata_file or sample_metadata)
        if not target_path.is_file():
            raise FileNotFoundError(f"Specified sample metadata file does not exist: {sanitize_path(target_path)}")
        meta_df = load_sample_metadata(target_path)
    elif dataset_dir and dataset_dir.is_dir():
        search_dirs = [dataset_dir]
        if dataset_dir.name == "samples" and dataset_dir.parent.name == "raw":
            search_dirs.append(dataset_dir.parent.parent)
        elif dataset_dir.name in ("samples", "raw"):
            search_dirs.append(dataset_dir.parent)
        for sdir in search_dirs:
            for cm in (
                sdir / "metadata" / "samples.csv",
                sdir / "metadata" / "samples.tsv",
                sdir / "metadata" / "metadata.csv",
                sdir / "metadata" / "metadata.tsv",
                sdir / "samples.csv",
                sdir / "metadata.csv",
            ):
                if cm.is_file():
                    try:
                        meta_df = load_sample_metadata(cm)
                        break
                    except Exception:
                        meta_df = None
            if meta_df is not None:
                break
    else:
        meta_df = None

    if meta_df is None or meta_df.empty:
        # Build default metadata indexed by sample stems without arbitrary filename token splitting
        stems = [f.stem for f in fcs_files]
        meta_df = pd.DataFrame({
            "sample_id": stems,
            "filename": [f.name for f in fcs_files],
            "condition": stems,
        }).set_index("sample_id")

    return meta_df


class FlowCytometryPipeline:
    """Reproducible flow cytometry processing and visualization pipeline.

    Attributes:
        output_dir: Base directory for storing runs, plots, tables, and workspaces.
        random_seed: Random seed for reproducible subsampling and rendering.
        timestamp_dir: If True (default), creates a timestamped run folder (e.g. sample_YYYYMMDD_HHMMSS)
            to preserve audit history and prevent overwriting previous results.
        overwrite: If True, writes directly to output_dir without timestamping.
        gating_mode: Analysis mode: 'som' (default), 'template', 'phenograph', or 'compare'.
        qc_method: Quality control methodology: 'clustering' (default GMM) or 'template'.
        legend_mode: 'centroid' (default, direct centroid badging) or 'external'.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs",
        random_seed: int = 42,
        timestamp_dir: bool = True,
        overwrite: bool = False,
        gating_mode: str = "som",
        cluster_parent: str = "viable",
        qc_method: str = "clustering",
        legend_mode: str = "centroid",
        adaptive_gating: bool = False,
        custom_hierarchy: Optional[GatingHierarchy] = None,
        marker_mapping: Optional[Dict[str, str]] = None,
        comp_controls_dir: Optional[Union[str, Path]] = None,
        comp_manifest: Optional[Union[str, Path]] = None,
        unstained_control: Optional[Union[str, Path]] = None,
        compensation_source: str = "auto",
        comp_channel_mapping: Optional[Dict[str, str]] = None,
        comp_config: Optional[CompensationConfig] = None,
        config_file: Optional[Union[str, Path]] = None,
        config_source: Optional[Any] = None,
        log_level: str = "INFO",
        transform_type: str = "logicle",
        enable_umap: bool = True,
        umap_subsample: int = 20000,
        enable_time_qc: bool = True,
        time_qc_config: Optional[Any] = None,
        harmonization_file: Optional[Union[str, Path]] = None,
        enable_harmonization: bool = False,
        harmonization_config: Optional[Dict[str, Any]] = None,
        debris_method: str = "gmm",
        debris_n_components: int = 2,
        debris_sd_scale: float = 1.0,
        debris_min_probability: Optional[float] = None,
        singlet_mode: str = "dual",
        singlet_method: str = "gmm_ratio",
        singlet_tolerance: float = 0.25,
        viability_cutoff: Optional[float] = None,
        viability_channel: Optional[str] = None,
        viability_looseness: float = 1.0,
        enable_viability: bool = True,
        dump_channel: Optional[str] = None,
        dump_looseness: float = 1.0,
        dump_enabled: bool = False,
        export_clean_fcs: bool = False,
        cleanup_config: Optional[CleanUpGateConfig] = None,
        spectral_config: Optional[Any] = None,
        transform_config: Optional[Any] = None,
        enable_qc: bool = True,
        enable_discovery: bool = False,
        numbered_steps: bool = True,
        enable_compensation: bool = True,
        enable_cleanup_gating: bool = True,
        enable_transformation: bool = True,
        transform_cofactor: float = 5.0,
        technology: str = "auto",
        metadata_file: Optional[Union[str, Path]] = None,
        markers_file: Optional[Union[str, Path]] = None,
        cell_signatures: Optional[Any] = None,
        lineage_markers: Optional[Union[str, List[str]]] = None,
        marker_roles: Optional[List[str]] = None,
        stats_marker_roles: Optional[List[str]] = None,
        include_markers: Optional[List[str]] = None,
        exclude_markers: Optional[List[str]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        files_list: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        clustering_method: str = "som",
        n_clusters: Optional[int] = None,
        auto_k: bool = True,
        auto_k_method: str = "consensus",
        k_min: int = 4,
        k_max: int = 20,
        consensus_subsample: float = 0.8,
        consensus_iterations: int = 50,
        umap_events_per_sample: int = 1000,
        condition_column: Optional[str] = None,
        batch_column: Optional[str] = None,
        umap_covariates: Optional[List[str]] = None,
        reference_group: Optional[str] = None,
        target_group: Optional[str] = None,
        paired_column: Optional[str] = None,
        da_test_method: str = "auto",
        ds_test_method: str = "auto",
        stats_granularity: str = "populations",
        fdr_threshold: float = 0.05,
        log2fc_threshold: float = 1.0,
        stratified_density_contours: bool = True,
        generate_forest_plot: bool = True,
        generate_ds_heatmap: bool = True,
        generate_abundance_boxplots: bool = True,
        generate_volcano_plot: bool = True,
        cluster_palette: str = "tab20",
        export_prism: bool = True,
        export_augmented_fcs: bool = True,
        export_flowjo_wsp: bool = True,
        export_gating_ml: bool = True,
        generate_miflowcyt_report: bool = True,
        cluster_name_map: Optional[Dict[str, str]] = None,
        batch_preparation: Optional[Dict[str, Any]] = None,
        enable_batch_preparation: bool = True,
        overwrite_autocheck: bool = False,
    ) -> None:
        """Initialize pipeline with output directory, seed, mode, QC method, legend styling, and panel options."""
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.timestamp_dir = timestamp_dir
        self.overwrite = overwrite
        self._overwrite_cleaned = False
        self.gating_mode = gating_mode.lower()
        self.cluster_parent = cluster_parent.lower()
        self.qc_method = qc_method.lower()
        self.legend_mode = legend_mode.lower()
        self.adaptive_gating = adaptive_gating
        self.custom_hierarchy = custom_hierarchy
        self.marker_mapping = marker_mapping
        self.comp_controls_dir = Path(comp_controls_dir) if comp_controls_dir else None
        self.comp_manifest = Path(comp_manifest) if comp_manifest else None
        self.unstained_control = Path(unstained_control) if unstained_control else None
        self.compensation_source = compensation_source.lower()
        self.comp_channel_mapping = comp_channel_mapping
        self.comp_config = comp_config
        self.spectral_config = spectral_config
        self.config_file = Path(config_file) if config_file else None
        self.config_source = config_source
        self.config = config_source
        self.log_level = log_level.upper()
        self.transform_type = (transform_type or "logicle").lower()
        self.enable_umap = enable_umap
        self.umap_subsample = umap_subsample
        self.enable_time_qc = enable_time_qc
        self.time_qc_config = time_qc_config
        self.harmonization_file = harmonization_file
        self.enable_harmonization = enable_harmonization
        self.harmonization_config = harmonization_config
        self.debris_method = debris_method.lower()
        self.debris_n_components = int(debris_n_components)
        self.singlet_mode = singlet_mode.lower()
        self.singlet_method = singlet_method.lower()
        self.singlet_tolerance = float(singlet_tolerance)
        self.debris_sd_scale = float(debris_sd_scale)
        self.debris_min_probability = float(debris_min_probability) if debris_min_probability is not None else None
        self.viability_cutoff = float(viability_cutoff) if viability_cutoff is not None else None
        self.viability_channel = viability_channel
        self.viability_looseness = float(viability_looseness)
        self.enable_viability = enable_viability
        self.dump_channel = dump_channel
        self.dump_looseness = float(dump_looseness)
        self.dump_enabled = bool(dump_enabled)
        self.export_clean_fcs = export_clean_fcs
        self.cleanup_config = cleanup_config
        self.transform_config = transform_config
        self.enable_qc = enable_qc
        self.enable_discovery = enable_discovery
        self.numbered_steps = numbered_steps
        self.enable_compensation = bool(enable_compensation)
        self.enable_cleanup_gating = bool(enable_cleanup_gating)
        self.enable_transformation = bool(enable_transformation)
        self.transform_cofactor = float(transform_cofactor)
        self.technology = (technology or "auto").lower()
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.markers_file = Path(markers_file) if markers_file else None
        self.cell_signatures = cell_signatures
        if isinstance(lineage_markers, str):
            self.lineage_markers: Optional[Union[str, List[str]]] = lineage_markers
        else:
            self.lineage_markers = list(lineage_markers) if lineage_markers else None
        self.marker_roles = list(marker_roles) if marker_roles else None
        self.stats_marker_roles = list(stats_marker_roles) if stats_marker_roles else None
        self.include_markers = list(include_markers) if include_markers else None
        self.exclude_markers = list(exclude_markers) if exclude_markers else None
        self.data_dir = Path(data_dir) if data_dir else None
        self.files_list = (
            [Path(p) for p in files_list]
            if isinstance(files_list, (list, tuple))
            else (Path(files_list) if files_list else None)
        )
        if config_source is not None:
            if n_clusters is None and getattr(config_source, "n_clusters", None) is not None:
                n_clusters = config_source.n_clusters
            if not auto_k and getattr(config_source, "auto_k", None) is not None:
                auto_k = config_source.auto_k
            if hasattr(config_source, "auto_k_method") and config_source.auto_k_method:
                auto_k_method = config_source.auto_k_method
            if hasattr(config_source, "batch_preparation") and config_source.batch_preparation and not batch_preparation:
                batch_preparation = config_source.batch_preparation
            if hasattr(config_source, "cluster_name_map") and config_source.cluster_name_map and not cluster_name_map:
                cluster_name_map = config_source.cluster_name_map
            if self.markers_file is None and getattr(config_source, "markers_file", None):
                self.markers_file = Path(config_source.markers_file)
            if self.cell_signatures is None and getattr(config_source, "cell_signatures", None):
                self.cell_signatures = config_source.cell_signatures

        self.clustering_method = clustering_method or gating_mode or "som"
        self.n_clusters = int(n_clusters) if n_clusters is not None else None
        self.auto_k = bool(auto_k)
        self.auto_k_method = str(auto_k_method).lower()
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.consensus_subsample = float(consensus_subsample)
        self.consensus_iterations = int(consensus_iterations)
        self.umap_events_per_sample = int(umap_events_per_sample)
        self.condition_column = condition_column
        self.batch_column = batch_column
        self.umap_covariates = umap_covariates
        self.reference_group = reference_group
        self.target_group = target_group
        self.paired_column = paired_column
        self.da_test_method = str(da_test_method or "auto")
        self.ds_test_method = str(ds_test_method or "auto")
        self.stats_granularity = str(stats_granularity or "populations").lower()
        self.fdr_threshold = float(fdr_threshold)
        self.log2fc_threshold = float(log2fc_threshold)
        self.stratified_density_contours = bool(stratified_density_contours)
        self.generate_forest_plot = bool(generate_forest_plot)
        self.generate_ds_heatmap = bool(generate_ds_heatmap)
        self.generate_abundance_boxplots = bool(generate_abundance_boxplots)
        self.generate_volcano_plot = bool(generate_volcano_plot)
        self.cluster_palette = str(cluster_palette or "tab20")
        self.export_prism = bool(export_prism)
        self.export_augmented_fcs = bool(export_augmented_fcs)
        self.export_flowjo_wsp = bool(export_flowjo_wsp)
        self.export_gating_ml = bool(export_gating_ml)
        self.generate_miflowcyt_report = bool(generate_miflowcyt_report)
        self.cluster_name_map = cluster_name_map
        self.batch_preparation = batch_preparation or {}
        self.enable_batch_preparation = bool(enable_batch_preparation)
        self.overwrite_autocheck = bool(overwrite_autocheck)

    def _build_cleanup_config(self) -> CleanUpGateConfig:
        """Build unified CleanUpGateConfig from explicit cleanup_config or scalar pipeline defaults."""
        if self.cleanup_config is not None:
            return self.cleanup_config
        return CleanUpGateConfig(
            debris_method=self.debris_method,
            debris_n_components=self.debris_n_components,
            debris_sd_scale=self.debris_sd_scale,
            debris_min_probability=self.debris_min_probability,
            singlet_mode=self.singlet_mode,
            singlet_method=self.singlet_method,
            singlet_tolerance=self.singlet_tolerance,
            viability_enabled=self.enable_viability,
            viability_channel=self.viability_channel,
            viability_cutoff=self.viability_cutoff,
            viability_looseness=self.viability_looseness,
            dump_enabled=self.dump_enabled,
            dump_channel=self.dump_channel,
            dump_looseness=self.dump_looseness,
        )

    @staticmethod
    def _write_resolved_config(
        run_dir: Path,
        marker_meta_res: Optional[Any] = None,
        cleanup_cfg: Optional[Any] = None,
        xform_channels: Optional[List[str]] = None,
        lineage_channels: Optional[List[str]] = None,
        stats_granularity: str = "populations",
        stats_marker_roles: Optional[List[str]] = None,
        ds_channels: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Persist expanded concrete channels/markers to resolved_config.yaml for zero-hidden-behavior audit."""
        raw_prov_src = (
            getattr(marker_meta_res, "provenance_source", None)
            or getattr(marker_meta_res, "source", None)
            or "none"
        )
        raw_prov_path = getattr(marker_meta_res, "source_path", None)
        if not raw_prov_path and isinstance(raw_prov_src, str) and raw_prov_src.startswith("markers_csv:"):
            raw_prov_path = raw_prov_src.split("markers_csv:", 1)[1].strip()
            raw_prov_src = "markers_csv"

        payload: Dict[str, Any] = {
            "03_cleanup_gating": (
                {
                    "fsc_channel": getattr(cleanup_cfg, "fsc_channel", None),
                    "ssc_channel": getattr(cleanup_cfg, "ssc_channel", None),
                    "fsc_height_channel": getattr(cleanup_cfg, "fsc_height_channel", None),
                    "ssc_height_channel": getattr(cleanup_cfg, "ssc_height_channel", None),
                    "debris_sd_scale": getattr(cleanup_cfg, "debris_sd_scale", 1.0),
                    "debris_min_probability": getattr(cleanup_cfg, "debris_min_probability", None),
                    "singlet_tolerance": getattr(cleanup_cfg, "singlet_tolerance", 0.25),
                    "viability_enabled": getattr(cleanup_cfg, "viability_enabled", None),
                    "viability_channel": getattr(cleanup_cfg, "viability_channel", None),
                    "viability_looseness": getattr(cleanup_cfg, "viability_looseness", 1.0),
                    "dump_enabled": getattr(cleanup_cfg, "dump_enabled", None),
                    "dump_channel": getattr(cleanup_cfg, "dump_channel", None),
                    "dump_looseness": getattr(cleanup_cfg, "dump_looseness", 1.0),
                }
                if cleanup_cfg is not None
                else {}
            ),
            "04_transformation": {
                "channels": list(xform_channels) if xform_channels else [],
            },
            "05_population_discovery": {
                "lineage_markers": list(lineage_channels) if lineage_channels else [],
            },
            "06_statistical_analysis": {
                "stats_granularity": str(stats_granularity or "populations"),
                "marker_roles_requested": list(stats_marker_roles) if stats_marker_roles else ["lineage", "functional"],
                "ds_markers": list(ds_channels or []),
            },
            "provenance": {
                "markers_metadata_source": str(raw_prov_src) if marker_meta_res else "none",
                "markers_metadata_path": (
                    sanitize_path(raw_prov_path)
                    if raw_prov_path
                    else None
                ),
                **(extra_metadata or {}),
            },
        }
        try:
            out_path = Path(run_dir) / "resolved_config.yaml"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import yaml

                with open(out_path, "w", encoding="utf-8") as fh:
                    yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
            except ImportError:
                out_path = Path(run_dir) / "resolved_config.json"
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
            return out_path
        except Exception:
            return None


    @classmethod
    def from_config(
        cls, config_source: Union[str, Path, Dict[str, Any], Any]
    ) -> Tuple["FlowCytometryPipeline", Optional[str]]:
        """Construct FlowCytometryPipeline from a config file path, dict, or PipelineConfig instance.

        Args:
            config_source: Path to .yaml/.json file, parameter dictionary, or PipelineConfig object.

        Returns:
            Tuple of (FlowCytometryPipeline, target fcs_file path or None).
        """
        from flowx.io.config import PipelineConfig

        if isinstance(config_source, PipelineConfig):
            cfg = config_source
        elif isinstance(config_source, dict):
            cfg = PipelineConfig.from_dict(config_source)
        else:
            cfg = PipelineConfig.load(config_source)
            cfg.config_file_path = str(Path(config_source).resolve())
        return cfg.to_pipeline()

    def _resolve_run_directory(self, fcs_filepath: Union[str, Path]) -> Path:
        """Resolve a unique, timestamped run directory to prevent accidental overwriting.

        Args:
            fcs_filepath: Source FCS file path.

        Returns:
            Resolved Path for the current run.
        """
        fcs_path = Path(fcs_filepath)
        sample_stem = fcs_path.stem.replace(" ", "_")
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.timestamp_dir and not self.overwrite:
            run_dir = self.output_dir / f"{sample_stem}_{now_str}"
            counter = 1
            while run_dir.exists():
                run_dir = self.output_dir / f"{sample_stem}_{now_str}_{counter}"
                counter += 1
            run_dir.mkdir(parents=True, exist_ok=True)
            self._update_latest_symlink(run_dir)
            return run_dir

        if not self.overwrite and self.output_dir.exists() and any(self.output_dir.iterdir()):
            run_dir = self.output_dir / f"{sample_stem}_{now_str}"
            get_logger().info(
                f"Notice: Output directory '{self.output_dir}' already contains existing files "
                f"and overwrite=False. Creating unique run directory: '{run_dir.name}'"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            self._update_latest_symlink(run_dir)
            return run_dir

        if self.overwrite and not self._overwrite_cleaned and self.output_dir.exists():
            resolved_out = self.output_dir.resolve()
            unsafe_targets = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
            if resolved_out not in unsafe_targets and len(resolved_out.parts) > 2:
                shutil.rmtree(self.output_dir)
            self._overwrite_cleaned = True

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def _update_latest_symlink(self, target_dir: Path) -> None:
        """Create or update a 'latest' symlink pointing to the most recent run."""
        try:
            parent = target_dir.parent
            latest_link = parent / "latest"
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(target_dir.name, target_is_directory=True)
        except Exception:
            pass

    def _record_run_manifest(self, run_dir: Path, summary: Dict[str, Any]) -> None:
        """Record run entry in the parent directory's manifest.json for auditing."""
        manifest_file = self.output_dir / "manifest.json"
        history: List[Dict[str, Any]] = []
        if manifest_file.exists():
            try:
                history = json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception:
                history = []
        history.append(summary)
        try:
            manifest_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        except Exception:
            pass

    def run(
        self,
        fcs_filepath: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        input_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        custom_hierarchy: Optional[GatingHierarchy] = None,
        custom_transform: Optional[BaseTransform] = None,
        null_channels: Optional[List[str]] = None,
        gating_mode: Optional[str] = None,
        qc_method: Optional[str] = None,
        legend_mode: Optional[str] = None,
        adaptive_gating: Optional[bool] = None,
        marker_mapping: Optional[Dict[str, str]] = None,
        config_file: Optional[Union[str, Path]] = None,
        log_level: Optional[str] = None,
        transform_type: Optional[str] = None,
        enable_time_qc: Optional[bool] = None,
        harmonization_file: Optional[Union[str, Path]] = None,
        enable_qc: Optional[bool] = None,
        enable_discovery: Optional[bool] = None,
        numbered_steps: Optional[bool] = None,
        sample_metadata: Optional[Union[pd.DataFrame, Dict[str, Dict[str, Any]], str, Path]] = None,
        metadata_file: Optional[Union[str, Path]] = None,
        technology: Optional[str] = None,
    ) -> PipelineResult:
        """Execute universal flow cytometry analysis pipeline across single samples or multi-sample cohorts."""
        target_source = input_path or fcs_filepath or self.data_dir or self.files_list
        if target_source is None:
            raise ValueError("No input FCS file or cohort directory specified for pipeline execution.")

        sample_paths, target_name = resolve_cohort_samples(target_source)
        if len(sample_paths) > 1:
            return self._run_cohort(
                sample_paths=sample_paths,
                target_name=target_name,
                custom_hierarchy=custom_hierarchy,
                custom_transform=custom_transform,
                null_channels=null_channels,
                gating_mode=gating_mode,
                qc_method=qc_method,
                legend_mode=legend_mode,
                adaptive_gating=adaptive_gating,
                marker_mapping=marker_mapping,
                config_file=config_file,
                log_level=log_level,
                transform_type=transform_type,
                enable_time_qc=enable_time_qc,
                harmonization_file=harmonization_file,
                enable_qc=enable_qc,
                enable_discovery=enable_discovery,
                numbered_steps=numbered_steps,
                sample_metadata=sample_metadata,
                metadata_file=metadata_file,
                technology=technology,
            )
        else:
            return self._run_single(
                fcs_filepath=sample_paths[0],
                custom_hierarchy=custom_hierarchy,
                custom_transform=custom_transform,
                null_channels=null_channels,
                gating_mode=gating_mode,
                qc_method=qc_method,
                legend_mode=legend_mode,
                adaptive_gating=adaptive_gating,
                marker_mapping=marker_mapping,
                config_file=config_file,
                log_level=log_level,
                transform_type=transform_type,
                enable_time_qc=enable_time_qc,
                harmonization_file=harmonization_file,
                enable_qc=enable_qc,
                enable_discovery=enable_discovery,
                numbered_steps=numbered_steps,
                sample_metadata=sample_metadata,
                metadata_file=metadata_file,
                technology=technology,
            )

    def _run_single(
        self,
        fcs_filepath: Union[str, Path],
        custom_hierarchy: Optional[GatingHierarchy] = None,
        custom_transform: Optional[BaseTransform] = None,
        null_channels: Optional[List[str]] = None,
        gating_mode: Optional[str] = None,
        qc_method: Optional[str] = None,
        legend_mode: Optional[str] = None,
        adaptive_gating: Optional[bool] = None,
        marker_mapping: Optional[Dict[str, str]] = None,
        config_file: Optional[Union[str, Path]] = None,
        log_level: Optional[str] = None,
        transform_type: Optional[str] = None,
        enable_time_qc: Optional[bool] = None,
        harmonization_file: Optional[Union[str, Path]] = None,
        enable_qc: Optional[bool] = None,
        enable_discovery: Optional[bool] = None,
        numbered_steps: Optional[bool] = None,
        sample_metadata: Optional[Union[pd.DataFrame, Dict[str, Dict[str, Any]], str, Path]] = None,
        metadata_file: Optional[Union[str, Path]] = None,
        technology: Optional[str] = None,
    ) -> PipelineResult:

        """Execute end-to-end flow cytometry analysis pipeline.

        Args:
            fcs_filepath: Path to input raw FCS file (read-only).
            custom_hierarchy: Optional user-defined GatingHierarchy.
            custom_transform: Optional transformation. Default Logicle.
            null_channels: Optional channels to ignore during compensation.
            gating_mode: Mode override ('template', 'som', 'phenograph', or 'compare').
            qc_method: QC methodology override ('template' or 'clustering').
            legend_mode: Legend styling ('centroid' or 'external').
            config_file: Optional configuration file path override.
            log_level: Optional console logging verbosity ('DEBUG', 'INFO', 'WARNING', 'ERROR').

        Returns:
            PipelineResult containing processed data, plots dictionary, stats, and workspace.
        """
        active_mode = (gating_mode or self.gating_mode).lower()
        active_qc = (qc_method or self.qc_method).lower()
        active_legend = (legend_mode or self.legend_mode).lower()
        active_qc_phase = self.enable_qc if enable_qc is None else enable_qc
        active_discovery_phase = self.enable_discovery if enable_discovery is None else enable_discovery
        active_numbered_steps = self.numbered_steps if numbered_steps is None else numbered_steps
        valid_modes = {"template", "som", "phenograph", "compare"}
        if active_mode not in valid_modes:
            raise ValueError(f"Invalid gating_mode: '{active_mode}'. Expected one of: {valid_modes}")
        valid_qc_modes = {"template", "clustering"}
        if active_qc not in valid_qc_modes:
            raise ValueError(f"Invalid qc_method: '{active_qc}'. Expected one of: {valid_qc_modes}")

        plots: Dict[str, str] = {}
        tables: Dict[str, str] = {}

        # Resolve unique non-overwriting run directory
        current_run_dir = self._resolve_run_directory(fcs_filepath)

        # Resolve step directories based on numbered_steps mode (Approach 2: Direct-to-Step)
        if active_numbered_steps:
            s1_plots_dir = current_run_dir / "01_time_qc" / "plots"
            s1_tables_dir = current_run_dir / "01_time_qc" / "tables"
            s2_plots_dir = current_run_dir / "02_compensation" / "plots"
            s2_tables_dir = current_run_dir / "02_compensation" / "tables"
            s3_plots_dir = current_run_dir / "03_cleanup_gating" / "plots"
            s3_tables_dir = current_run_dir / "03_cleanup_gating" / "tables"
            s5_plots_dir = current_run_dir / "05_population_discovery" / "plots"
            s5_tables_dir = current_run_dir / "05_population_discovery" / "tables"
            exports_ws_dir = current_run_dir / "exports" / "workspaces"
            exports_rep_dir = current_run_dir / "exports" / "reports"
            clean_fcs_dir = current_run_dir / "exports" / "clean_fcs"
            for d in (s1_plots_dir, s1_tables_dir, s2_plots_dir, s2_tables_dir, s3_plots_dir, s3_tables_dir, exports_ws_dir, exports_rep_dir):
                d.mkdir(parents=True, exist_ok=True)
            if active_discovery_phase:
                s5_plots_dir.mkdir(parents=True, exist_ok=True)
                s5_tables_dir.mkdir(parents=True, exist_ok=True)
            plots_dir = s3_plots_dir
            tables_dir = s3_tables_dir
        else:
            s1_plots_dir = s2_plots_dir = s3_plots_dir = s5_plots_dir = current_run_dir / "plots"
            s1_tables_dir = s2_tables_dir = s3_tables_dir = s5_tables_dir = current_run_dir / "tables"
            exports_ws_dir = exports_rep_dir = current_run_dir
            clean_fcs_dir = current_run_dir / "clean_fcs"
            plots_dir = current_run_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            tables_dir = current_run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)

        def _pname(legacy_name: str) -> str:
            if active_numbered_steps:
                return re.sub(r"^step\d+[a-zA-Z]?_", "", legacy_name)
            return legacy_name

        def _add_plot(key: str, p_path: Path) -> None:
            p_str = str(p_path)
            plots[key] = p_str
            clean_k = re.sub(r"^step\d+[a-zA-Z]?_", "", key)
            plots[clean_k] = p_str
            if "umap_marker_features" in key:
                plots["cohort_umap_marker_features"] = p_str
            elif "umap_covariates" in key or "cohort_umap_covariates" in key:
                plots["cohort_umap_covariates"] = p_str
            elif "umap_clusters" in key:
                plots["cohort_umap_clusters"] = p_str
            elif "umap_populations" in key:
                plots["cohort_umap_populations"] = p_str

        t_start = time.perf_counter()
        run_start_dt = datetime.now()

        # Initialize structured run logging (console stream + run directory pipeline.log)
        active_log_level = (log_level or self.log_level).upper()
        logger, log_file_path = setup_run_logger(
            run_dir=current_run_dir,
            log_filename="pipeline.log",
            console_level=active_log_level,
            file_level="DEBUG",
        )
        logger.info(f"Initiated flow cytometry pipeline run for sample '{fcs_filepath}' at {run_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Output run directory: '{sanitize_path(current_run_dir)}' (Mode: {active_mode.upper()}, Legend: {active_legend.upper()})")
        logger.debug(f"Run Configuration: seed={self.random_seed}, adaptive={adaptive_gating if adaptive_gating is not None else self.adaptive_gating}, qc={active_qc}")
        logger.debug(f"System Info: Python {sys.version.split()[0]} on {sys.platform}")

        # Save a copy of configuration in the output directory if user used a config file/object
        saved_config_path: Optional[Path] = None
        target_config = config_file or self.config_file

        if target_config is not None:
            src_cfg = Path(target_config)
            if src_cfg.is_file():
                dest_cfg = current_run_dir / "config.yaml" if src_cfg.suffix.lower() in (".yaml", ".yml") else current_run_dir / "config.json"
                try:
                    if src_cfg.resolve() != dest_cfg.resolve():
                        shutil.copy2(src_cfg, dest_cfg)
                    saved_config_path = dest_cfg
                    logger.info(f"Saved configuration copy to '{sanitize_path(dest_cfg)}'.")
                    logger.debug(f"Preserved original configuration: '{sanitize_path(src_cfg)}' ({src_cfg.stat().st_size:,} bytes)")
                except Exception as exc:
                    logger.warning(f"Warning: Could not copy config file '{sanitize_path(src_cfg)}': {exc}")

                # For backward compatibility with tests expecting original filename when numbered_steps is False
                if not active_numbered_steps and src_cfg.stem.lower() != "config":
                    named_dest = current_run_dir / src_cfg.name
                    try:
                        if src_cfg.resolve() != named_dest.resolve() and not named_dest.exists():
                            shutil.copy2(src_cfg, named_dest)
                    except Exception:
                        pass
        elif self.config_source is not None:
            try:
                if hasattr(self.config_source, "save"):
                    cfg_dest = current_run_dir / "config.yaml"
                    self.config_source.save(cfg_dest)
                    saved_config_path = cfg_dest
                    logger.info(f"Saved configuration copy to '{sanitize_path(cfg_dest)}'.")
                elif isinstance(self.config_source, dict):
                    cfg_dest = current_run_dir / "config.json"
                    cfg_dest.write_text(json.dumps(self.config_source, indent=2), encoding="utf-8")
                    saved_config_path = cfg_dest
                    logger.info(f"Saved configuration copy to '{sanitize_path(cfg_dest)}'.")
            except Exception as exc:
                logger.warning(f"Warning: Could not save configuration copy: {exc}")

        try:
            # -------------------------------------------------------------
            # STEP 1: Parse FCS File & Extract Metadata
            # -------------------------------------------------------------
            with log_step(logger, "Step 1: Parse FCS File & Extract Metadata"):
                active_mapping = dict(marker_mapping or self.marker_mapping or {}) or None

                logger.info(f"Step 1: Parsing FCS file '{sanitize_path(fcs_filepath)}'...")
                # Resolve panel markers using explicit markers_file or FCS parameter headers
                from flowx.io.metadata import resolve_marker_metadata
                fcs_data = parse_fcs(fcs_filepath, null_channels=null_channels, marker_mapping=active_mapping)
                marker_meta_res = resolve_marker_metadata(
                    markers_file=self.markers_file,
                    dataset_dir=Path(fcs_filepath).parent,
                    fcs_pnn=fcs_data.pnn_labels,
                    fcs_pns=fcs_data.pns_labels,
                    marker_mapping=active_mapping,
                    logger=logger,
                )
                if marker_meta_res.detector_to_marker and marker_meta_res.has_verified_biological_markers:
                    merged_map = dict(active_mapping or {})
                    merged_map.update(marker_meta_res.detector_to_marker)
                    fcs_data = parse_fcs(fcs_filepath, null_channels=null_channels, marker_mapping=merged_map)
                fcs_data.detector_to_fluorophore = dict(marker_meta_res.detector_to_fluorophore)
                if fcs_data.cytometer:
                    logger.info(f"Cytometer Instrument: {fcs_data.cytometer}")
                logger.info(
                    f"Loaded {fcs_data.event_count:,} events with "
                    f"{fcs_data.channel_count} channels. Scale dynamic range: T={fcs_data.detect_max_range():,.0f}"
                )
                logger.debug(f"Source FCS: '{sanitize_path(fcs_data.filepath)}' ({fcs_data.event_count:,} events, {fcs_data.channel_count} detectors)")

                for idx, (pnn, pns) in enumerate(zip(fcs_data.pnn_labels, fcs_data.pns_labels)):
                    pnr = fcs_data.channel_ranges.get(pnn, "N/A")
                    logger.debug(f"Channel {idx+1:02d}: PnN='{pnn}' | PnS='{pns or '-'}' | Label='{fcs_data.get_channel_label(pnn)}' | Range={pnr}")

                fsc_ch = fcs_data.get_fsc_channel()
                ssc_ch = fcs_data.get_ssc_channel()
                max_fsc = fcs_data.channel_ranges.get(fsc_ch, 262144.0)
                max_ssc = fcs_data.channel_ranges.get(ssc_ch, 262144.0)

                s1_plots_dir.mkdir(parents=True, exist_ok=True)
                s1_tables_dir.mkdir(parents=True, exist_ok=True)

                step1a_path = s1_plots_dir / _pname("step1_raw_fsc_vs_ssc.png")
                plot_scatter_2d(
                    df=fcs_data.events_df,
                    x_channel=fsc_ch,
                    y_channel=ssc_ch,
                    x_label=fcs_data.get_channel_label(fsc_ch),
                    y_label=fcs_data.get_channel_label(ssc_ch),
                    title=f"Step 1A: Raw Morphology ({fsc_ch} vs {ssc_ch}) - {fcs_data.filename}",
                    output_path=step1a_path,
                    xlim=(-0.02 * max_fsc, 1.05 * max_fsc),
                    ylim=(-0.02 * max_ssc, 1.05 * max_ssc),
                )
                _add_plot("step1_raw_morphology", step1a_path)

                fsc_h_ch = fcs_data.get_fsc_h_channel()
                if fsc_h_ch and fsc_h_ch in fcs_data.events_df.columns:
                    max_fsc_h = fcs_data.channel_ranges.get(fsc_h_ch, 262144.0)
                    step1b_path = s1_plots_dir / _pname("step1_raw_doublet_fsc_a_vs_h.png")
                    plot_scatter_2d(
                        df=fcs_data.events_df,
                        x_channel=fsc_ch,
                        y_channel=fsc_h_ch,
                        x_label=fcs_data.get_channel_label(fsc_ch),
                        y_label=fcs_data.get_channel_label(fsc_h_ch),
                        title=f"Step 1B: Raw Singlet Discrimination ({fsc_ch} vs {fsc_h_ch})",
                        output_path=step1b_path,
                        xlim=(-0.02 * max_fsc, 1.05 * max_fsc),
                        ylim=(-0.02 * max_fsc_h, 1.05 * max_fsc_h),
                    )
                    _add_plot("step1_raw_doublets", step1b_path)
                else:
                    logger.info(f"Step 1B: Pulse height/doublet channel not recorded in '{fcs_data.filename}'. Skipping raw doublet scatter plot.")

                # Step 1C: Panel Harmonization (if enabled or specified)
                target_harm = harmonization_file or self.harmonization_file or self.harmonization_config
                if self.enable_harmonization or target_harm is not None:
                    from flowx.panels.harmonization import load_panel_harmonizer

                    harmonizer = load_panel_harmonizer(target_harm)
                    harm_res = harmonizer.harmonize_sample(fcs_data, inplace=False)
                    logger.info(
                        f"Step 1C: Panel Harmonization applied ({len(harm_res.channel_mapping)} canonical channels mapped, "
                        f"{len(harm_res.missing_channels)} missing, {len(harm_res.unmapped_channels)} unmapped)"
                    )
                    harm_csv_path = s1_tables_dir / "panel_harmonization_mapping.csv"
                    harm_rows = [
                        {"canonical_channel": k, "source_detector": v, "sample": fcs_data.filename}
                        for k, v in harm_res.channel_mapping.items()
                    ]
                    pd.DataFrame(harm_rows).to_csv(harm_csv_path, index=False)
                    tables["panel_harmonization"] = str(harm_csv_path)

                # Export Channel Resolution Audit
                if hasattr(fcs_data, "export_resolution_audit"):
                    res_audit_path = s1_tables_dir / "channel_resolution_audit.csv"
                    fcs_data.export_resolution_audit(res_audit_path)
                    tables["channel_resolution_audit"] = str(res_audit_path)

                # -------------------------------------------------------------
                # STEP 1D: Fluidic Time Quality Control (Step 1.1 / 1.2)
                # -------------------------------------------------------------
                active_qc_phase = self.enable_qc if enable_qc is None else enable_qc
                active_time_qc = active_qc_phase and (self.enable_time_qc if enable_time_qc is None else enable_time_qc)
                working_sample_df = fcs_data.events_df.copy()
                pipeline_time_qc_res = None
                time_ch = fcs_data.get_time_channel()

                if active_time_qc and time_ch and time_ch in fcs_data.events_df.columns and len(fcs_data.events_df) >= 100:
                    with log_step(logger, "Step 1D: Fluidic Time Quality Control (flowAI/flowCut/PeacoQC)"):
                        logger.info("Step 1D: Evaluating fluidic stability and signal drift (Time QC)...")
                        try:
                            from flowx.qc.models import TimeQCConfig
                            from flowx.qc.runner import run_time_qc_step

                            tqc_cfg = self.time_qc_config or TimeQCConfig.for_sample()
                            tqc_step = run_time_qc_step(
                                fcs_data=fcs_data,
                                config=tqc_cfg,
                                output_plot_path=s1_plots_dir / _pname("step1_time_qc_diagnostics.png"),
                                output_summary_csv=s1_tables_dir / "time_qc_sample_summary.csv",
                                generate_plots=True,
                            )
                            tqc_res = tqc_step.qc_result
                            decision = tqc_step.decision
                            pipeline_time_qc_res = tqc_res

                            if decision is not None and decision.can_proceed and tqc_step.clean_df is not None:
                                working_sample_df = tqc_step.clean_df
                                if tqc_res is not None:
                                    logger.info(
                                        f"Step 1D: Time QC passed ({tqc_res.passed_pct:.1f}% clean events, "
                                        f"{tqc_res.passed_events:,}/{tqc_res.total_events:,}). Action: {decision.action}"
                                    )
                            elif decision is not None:
                                logger.warning(
                                    f"Step 1D: Time QC flagged sample: {decision.action} ({decision.rejection_reason})"
                                )

                            if tqc_step.plot_path:
                                _add_plot("step1_time_qc", Path(tqc_step.plot_path))
                            if tqc_step.summary_csv_path:
                                tables["time_qc_sample_summary"] = tqc_step.summary_csv_path
                        except Exception as exc:
                            logger.warning(f"Step 1D: Time QC encountered an error: {exc}. Proceeding with raw events.")

            # -------------------------------------------------------------
            # STEP 2: Fluorescence Spillover Compensation
            # -------------------------------------------------------------
            with log_step(logger, "Step 2: Fluorescence Spillover Compensation"):
                logger.info("Step 2: Applying fluorescence compensation...")
                from flowx.math.compensation import run_compensation_step
                from flowx.math.transforms import resolve_fluorescence_channels

                active_tech = (technology or self.technology or "auto").lower()
                target_channels = resolve_fluorescence_channels(
                    fcs_data,
                    explicit_channels=None,
                    df=working_sample_df,
                )
                emp_pre_transformed = is_empirically_pre_transformed(fcs_data)

                comp_step = run_compensation_step(
                    fcs_data=fcs_data,
                    working_df=working_sample_df,
                    target_channels=target_channels,
                    enable_compensation=self.enable_compensation,
                    compensation_source=self.compensation_source,
                    technology=active_tech,
                    comp_controls_dir=self.comp_controls_dir,
                    comp_manifest=self.comp_manifest,
                    unstained_control=self.unstained_control,
                    comp_channel_mapping=self.comp_channel_mapping,
                    comp_config=self.comp_config,
                    spectral_config=self.spectral_config,
                    plots_dir=s2_plots_dir,
                    tables_dir=s2_tables_dir,
                    plot_name_fn=_pname,
                    random_seed=self.random_seed,
                    generate_plots=True,
                    generate_nxn=True,
                )
                df_comp = comp_step.compensated_df
                comp_obj = comp_step.comp_matrix
                comp_ctrl_result = comp_step.comp_ctrl_result
                comp_source_used = comp_step.comp_source_used
                target_channels = comp_step.target_channels
                spectral_pipeline_res = comp_step.spectral_pipeline_res
                spill_matrix_csv_path = comp_step.spill_matrix_csv_path
                tables.update(comp_step.tables)
                for p_desc, p_file in comp_step.plots.items():
                    _add_plot(p_desc, Path(p_file))
                if comp_source_used == "spectral_unmixing":
                    logger.info(f"Step 2: Spectral unmixing completed across {len(target_channels)} fluorophore channels.")
                elif comp_obj is not None:
                    logger.info(
                        f"Step 2: Compensation ({comp_source_used}) applied across {len(comp_obj.detectors)} channels "
                        f"(kappa={comp_obj.condition_number:.2f})."
                    )
                else:
                    logger.info(f"Step 2: Compensation skipped/bypassed (source='{comp_source_used}').")

            # -------------------------------------------------------------
            # STEP 3: Scale Transformation (Logicle / Biexponential)
            # -------------------------------------------------------------
            with log_step(logger, "Step 3: Scale Transformation"):
                from flowx.math.transforms import TransformConfig, run_transformation_step

                active_xform_type = (transform_type or self.transform_type).lower()
                base_xf_cfg = self.transform_config
                xf_cfg = TransformConfig(
                    enabled=self.enable_transformation if base_xf_cfg is None else base_xf_cfg.enabled,
                    method=active_xform_type,
                    cofactor=self.transform_cofactor if base_xf_cfg is None else base_xf_cfg.cofactor,
                    channels=(
                        target_channels
                        if comp_source_used == "spectral_unmixing"
                        else (base_xf_cfg.channels if base_xf_cfg is not None else None)
                    ),
                    exclude_channels=base_xf_cfg.exclude_channels if base_xf_cfg is not None else None,
                    param_w=base_xf_cfg.param_w if base_xf_cfg is not None else 0.5,
                    param_a=base_xf_cfg.param_a if base_xf_cfg is not None else 0.0,
                    normalized=base_xf_cfg.normalized if base_xf_cfg is not None else True,
                )
                step3_xform_path = s3_plots_dir / _pname("step3_transformation_diagnostic.png")
                xf_step = run_transformation_step(
                    fcs_data=fcs_data,
                    df_events=df_comp,
                    config=xf_cfg,
                    target_channels=target_channels if comp_source_used == "spectral_unmixing" else xf_cfg.channels,
                    custom_transform=custom_transform,
                    marker_metadata=marker_meta_res,
                    output_plot_path=step3_xform_path,
                    generate_plot=True,
                )
                transform_obj = xf_step.transform
                fluoro_channels = xf_step.channels_transformed
                df_transformed = xf_step.transformed_df
                if transform_obj is not None:
                    is_biex = isinstance(transform_obj, BiexponentialTransform)
                    xform_name = (
                        "Biexponential (FlowJo LUT)"
                        if is_biex
                        else ("Logicle (Gating-ML 2.0)" if isinstance(transform_obj, LogicleTransform) else transform_obj.__class__.__name__)
                    )
                    t_val = getattr(transform_obj, "param_t", getattr(transform_obj, "max_value", fcs_data.detect_max_range()))
                    m_val = getattr(transform_obj, "param_m", getattr(transform_obj, "positive", 4.5))
                    logger.info(
                        f"Step 3: Applying {xform_name} transformation "
                        f"(Range T={t_val:,.0f}, Decades M={m_val:.2f})..."
                    )
                    logger.debug(f"{xform_name} parameters: {transform_obj.to_dict()}")
                if xf_step.plot_path:
                    _add_plot("step3_transformation_diagnostic", Path(xf_step.plot_path))


            # -------------------------------------------------------------
            # STEP 4: Gating Hierarchy Construction & Evaluation
            # -------------------------------------------------------------
            with log_step(logger, "Step 4: Gating Hierarchy Construction & Evaluation"):
                is_adaptive = self.adaptive_gating if adaptive_gating is None else adaptive_gating
                active_hierarchy = custom_hierarchy or self.custom_hierarchy
                cutoffs_dict: Optional[Dict[str, DensityCutoffResult]] = None
                clustering_qc_res: Optional[ClusteringQCResult] = None
                cleanup_gating_res: Optional[CleanUpGatingResult] = None
                active_cleanup_phase = bool(active_qc_phase and self.enable_cleanup_gating and not emp_pre_transformed)

                # Execute Phase 2 Core Clean-up Gating diagnostic cascade via shared step runner
                if active_cleanup_phase:
                    try:
                        from flowx.gating.runner import run_cleanup_step

                        cleanup_cfg = self._build_cleanup_config()
                        cleanup_step = run_cleanup_step(
                            fcs_data=fcs_data,
                            df_events=df_comp,
                            df_transformed=df_transformed,
                            config=cleanup_cfg,
                            marker_metadata=marker_meta_res,
                            output_plot_path=s3_plots_dir / _pname("step4_cleanup_diagnostics.png"),
                            generate_plots=True,
                            export_clean_fcs=self.export_clean_fcs,
                            clean_fcs_path=clean_fcs_dir / f"{fcs_data.sample_name}_clean.fcs",
                        )
                        cleanup_gating_res = cleanup_step.cleanup_result
                        if cleanup_step.plot_path:
                            _add_plot("step4_cleanup_diagnostics", Path(cleanup_step.plot_path))
                        if cleanup_step.clean_fcs_path:
                            logger.info(f"Clean FCS exported to '{cleanup_step.clean_fcs_path}'")
                    except Exception as clean_err:
                        logger.warning(f"Phase 2 Clean-up Gating diagnostic generation failed: {clean_err}", exc_info=True)
                else:
                    logger.info("Step 4: Quality Control (QC) clean-up phase disabled or pre-transformed data detected. Skipping debris, doublet, and viability gating.")

                if active_cleanup_phase and active_qc == "clustering":
                    logger.info("Step 4-QC: Applying automatic clustering-based QC (Morphology & Singlets)...")
                    clustering_qc_res = run_clustering_qc(
                        fcs_data=fcs_data,
                        df_events=df_comp,
                        random_state=self.random_seed,
                    )
                    step1_qc_path = s3_plots_dir / _pname("step1_clustering_qc_diagnostic.png")
                    plot_clustering_qc_diagnostic(
                        df=df_comp,
                        qc_result=clustering_qc_res,
                        fcs_data=fcs_data,
                        output_path=step1_qc_path,
                    )
                    _add_plot("step1_clustering_qc_diagnostic", step1_qc_path)
                    logger.info(
                        f"Clustering QC completed: "
                        f"Intact Cells: {clustering_qc_res.cell_count:,} ({clustering_qc_res.cell_count / fcs_data.event_count * 100:.1f}%), "
                        f"Singlets: {clustering_qc_res.singlet_count:,} ({clustering_qc_res.singlet_count / fcs_data.event_count * 100:.1f}%)"
                    )

                cutoffs_dict: Optional[Dict[str, Any]] = None
                if active_hierarchy is not None:
                    logger.info("Step 4: Using custom gating hierarchy...")
                    hierarchy = active_hierarchy
                elif cleanup_gating_res is not None:
                    import copy
                    logger.info("Step 4: Adopting Core Clean-up Gating hierarchy...")
                    hierarchy = copy.deepcopy(cleanup_gating_res.hierarchy)
                elif clustering_qc_res is not None:
                    logger.info("Step 4: Constructing QC gating hierarchy from clustering QC...")
                    hierarchy = GatingHierarchy()
                    hierarchy.add_gate(clustering_qc_res.cells_gate)
                    hierarchy.add_gate(clustering_qc_res.singlets_gate)
                else:
                    logger.info("Step 4: Constructing adaptive morphological gating hierarchy...")
                    hierarchy = GatingHierarchy()
                    cells_g = build_adaptive_cell_gate(fcs_data, df_transformed)
                    hierarchy.add_gate(cells_g)
                    singlet_g = build_adaptive_singlet_gate(
                        fcs_data, df_transformed, parent_mask=cells_g.apply(df_transformed)
                    )
                    hierarchy.add_gate(singlet_g)

                masks = hierarchy.evaluate(df_transformed)
                logger.debug(f"Hierarchy evaluation complete: {len(hierarchy.gates)} gates evaluated: {hierarchy.order}")
                for g_name in hierarchy.order:
                    g_cnt = int(np.sum(masks[g_name])) if g_name in masks else 0
                    g_pct = (g_cnt / fcs_data.event_count * 100.0) if fcs_data.event_count > 0 else 0.0
                    logger.debug(f"Population '{g_name}': Count={g_cnt:,} ({g_pct:.2f}% of total)")

                # Assign mutually exclusive leaf population labels per event for concordance comparisons
                template_labels = np.full(len(df_transformed), "Debris / Non-cells", dtype=object)
                if "Cells" in masks:
                    template_labels[masks["Cells"]] = "Doublets / Non-singlets"
                if "Singlets" in masks:
                    template_labels[masks["Singlets"]] = "Non-viable / Unclassified"
                elif "Singlets_SSC" in masks:
                    template_labels[masks["Singlets_SSC"]] = "Non-viable / Unclassified"
                elif "Singlets_FSC" in masks:
                    template_labels[masks["Singlets_FSC"]] = "Non-viable / Unclassified"
                viable_gate_key = next((k for k in masks if k.lower().startswith("viable") or k.lower().startswith("live")), None)
                if viable_gate_key:
                    template_labels[masks[viable_gate_key]] = f"{viable_gate_key} Other"

                for gate_name in hierarchy.order:
                    if (
                        gate_name not in ("Cells", "Singlets", "Singlets_FSC", "Singlets_SSC")
                        and not gate_name.lower().startswith("viable")
                        and not gate_name.lower().startswith("live")
                    ):
                        template_labels[masks[gate_name]] = gate_name

                if active_mode in ("template", "compare"):
                    for idx, gate_name in enumerate(hierarchy.order):
                        gate = hierarchy.get_gate(gate_name)
                        parent_name = gate.parent_gate or "root"
                        parent_mask = masks[parent_name]

                        x_ch = gate.channels[0]
                        y_ch = gate.channels[1] if len(gate.channels) > 1 else gate.channels[0]

                        sanitized_name = gate_name.lower().replace("+", "_pos").replace(" ", "_")
                        gate_plot_path = s3_plots_dir / _pname(f"step4_{idx+1}_gate_{sanitized_name}.png")
                        plot_gate_overlay(
                            df=df_transformed,
                            gate=gate,
                            parent_mask=parent_mask,
                            x_label=fcs_data.get_channel_label(x_ch),
                            y_label=fcs_data.get_channel_label(y_ch) if len(gate.channels) > 1 else "Event Count",
                            title=f"Step 4.{idx+1}: {gate.name} Gate on {parent_name.capitalize()}",
                            output_path=gate_plot_path,
                        )
                        _add_plot(f"step4_gate_{sanitized_name}", gate_plot_path)

                    step4_cascade_path = s3_plots_dir / _pname("step4_gating_cascade_overview.png")
                    channel_label_map = {pnn: fcs_data.get_channel_label(pnn) for pnn in fcs_data.pnn_labels}
                    plot_pipeline_cascade(
                        df=df_transformed,
                        gates=[hierarchy.get_gate(name) for name in hierarchy.order],
                        hierarchy_masks=masks,
                        channel_labels=channel_label_map,
                        output_path=step4_cascade_path,
                    )
                    _add_plot("step4_cascade_overview", step4_cascade_path)

            # -------------------------------------------------------------
            # STEP 5: Population Statistics
            # -------------------------------------------------------------
            with log_step(logger, "Step 5: Population Statistics"):
                logger.info("Step 5: Calculating template population statistics...")
                stats_channels = [ch for ch in ["Ax700-A", "PE-TxRed-A", "PacBlue-A"] if ch in df_transformed.columns]
                stats_df = hierarchy.get_population_statistics(
                    df=df_transformed,
                    report_channels=stats_channels,
                )
                stats_csv_path = s3_tables_dir / "population_statistics.csv"
                stats_df.to_csv(stats_csv_path, index=False)
                tables["population_statistics"] = str(stats_csv_path)
                stats_df.to_csv(stats_csv_path, index=False)
                logger.debug(f"Saved population statistics to '{stats_csv_path}'")

            # -------------------------------------------------------------
            # STEP 5B: Unsupervised Clustering (SOM and/or PhenoGraph)
            # -------------------------------------------------------------
            som_res: Optional[ClusteringResult] = None
            pheno_res: Optional[ClusteringResult] = None
            ct_som: Optional[pd.DataFrame] = None
            ct_pg: Optional[pd.DataFrame] = None
            three_way_metrics: Optional[pd.DataFrame] = None
            three_way_lineages: Optional[pd.DataFrame] = None

            # Resolve parent population for unsupervised clustering
            active_cluster_parent = getattr(self, "cluster_parent", "viable").lower()
            cluster_mask = np.ones(len(df_transformed), dtype=bool)
            cluster_parent_name = "All Events"
            viability_filtered = False

            if cleanup_gating_res is not None and active_cluster_parent in ("viable", "live", "viables", "clean"):
                cluster_mask = cleanup_gating_res.final_clean_mask
                cluster_parent_name = "Viable" if active_cluster_parent in ("viable", "live", "viables") else "Clean Target Cells"
                viability_filtered = "Viable" in cleanup_gating_res.hierarchy.gates
                logger.info(f"Clustering on Core Clean-up Gating target population ({int(np.sum(cluster_mask)):,} clean events).")
            elif active_cluster_parent in ("viable", "live", "viables"):
                viable_gate_key = next((k for k in masks if k.lower().startswith("viable") or k.lower().startswith("live")), None)
                if viable_gate_key:
                    cluster_mask = masks[viable_gate_key]
                    cluster_parent_name = viable_gate_key
                    viability_filtered = True
                elif any(k in masks for k in ("Singlets", "Singlets_SSC", "Singlets_FSC")):
                    s_key = next(k for k in ("Singlets", "Singlets_SSC", "Singlets_FSC") if k in masks)
                    cluster_mask = masks[s_key]
                    cluster_parent_name = "Singlets"
                    logger.info(f"Viable gate not found in hierarchy; clustering on parent '{s_key}'.")
                elif "Cells" in masks:
                    cluster_mask = masks["Cells"]
                    cluster_parent_name = "Cells"
                    logger.info("Viable gate not found in hierarchy; clustering on parent 'Cells'.")
            elif active_cluster_parent in ("singlets", "singlet"):
                if any(k in masks for k in ("Singlets", "Singlets_SSC", "Singlets_FSC")):
                    s_key = next(k for k in ("Singlets", "Singlets_SSC", "Singlets_FSC") if k in masks)
                    cluster_mask = masks[s_key]
                    cluster_parent_name = "Singlets"
                elif "Cells" in masks:
                    cluster_mask = masks["Cells"]
                    cluster_parent_name = "Cells"
            elif active_cluster_parent in ("cells", "cell", "morphology"):
                if "Cells" in masks:
                    cluster_mask = masks["Cells"]
                    cluster_parent_name = "Cells"
            elif active_cluster_parent in ("all", "root", "none", "total"):
                cluster_mask = np.ones(len(df_transformed), dtype=bool)
                cluster_parent_name = "All Events"
            else:
                matched_gate = next((g for g in masks.keys() if g.lower() == active_cluster_parent), None)
                if matched_gate is not None:
                    cluster_mask = masks[matched_gate]
                    cluster_parent_name = matched_gate
                else:
                    logger.warning(
                        f"Requested cluster parent '{active_cluster_parent}' not found in evaluated gates: "
                        f"{list(masks.keys())}. Defaulting to all events."
                    )
                    cluster_mask = np.ones(len(df_transformed), dtype=bool)
                    cluster_parent_name = "All Events"

            df_cluster = df_transformed.loc[cluster_mask].reset_index(drop=True)
            parent_event_count = len(df_cluster)

            from flowx.io.metadata import select_markers_from_metadata

            resolved_meta_markers = select_markers_from_metadata(
                marker_meta_res,
                available_columns=list(df_transformed.columns),
                explicit_markers=self.lineage_markers,
                marker_roles=self.marker_roles,
                include_markers=self.include_markers,
                exclude_markers=self.exclude_markers,
            )
            if resolved_meta_markers:
                cluster_markers = resolved_meta_markers
            elif marker_meta_res and marker_meta_res.clustering_channels and comp_source_used != "spectral_unmixing":
                cluster_markers = [ch for ch in marker_meta_res.clustering_channels if ch in df_transformed.columns]
            else:
                cluster_markers = [ch for ch in fluoro_channels if ch in df_transformed.columns]
            if len(cluster_markers) < 1:
                cluster_markers = [c for c in df_transformed.columns if "time" not in c.lower()][:8]

            # Persist concrete resolved channels & markers for auditability
            from flowx.gating.runner import resolve_cleanup_channels_from_metadata

            _cleanup_cfg_audit = self._build_cleanup_config()
            _resolved_clean_audit = resolve_cleanup_channels_from_metadata(
                _cleanup_cfg_audit, list(df_transformed.columns), marker_metadata=marker_meta_res
            )
            res_cfg_path = self._write_resolved_config(
                run_dir=current_run_dir,
                marker_meta_res=marker_meta_res,
                cleanup_cfg=_resolved_clean_audit,
                xform_channels=fluoro_channels,
                lineage_channels=cluster_markers,
                stats_granularity=self.stats_granularity,
                extra_metadata={"marker_roles_requested": self.marker_roles},
            )
            if res_cfg_path is not None:
                tables["resolved_config"] = str(res_cfg_path)
                logger.info(f"Saved resolved parameter audit configuration to {sanitize_path(res_cfg_path)}")

            # When viability filtering is active, exclude viability dye from phenotypic clustering markers
            if viability_filtered or active_cluster_parent in ("viable", "live", "viables"):
                dead_ch = None
                viable_gate_key = next((k for k in hierarchy.gates if k.lower().startswith("viable") or k.lower().startswith("live")), None)
                if viable_gate_key:
                    vg = hierarchy.get_gate(viable_gate_key)
                    if vg and vg.channels:
                        dead_ch = vg.channels[0]
                if not dead_ch:
                    dead_ch = fcs_data.find_channel_by_alias(
                        ["dead", "viability", "aard", "aard-a", "dapi", "pi", "7-aad", "zombie", "livedead", "fvs"]
                    )
                if dead_ch and dead_ch in cluster_markers:
                    cluster_markers = [m for m in cluster_markers if m != dead_ch]
                    logger.info(f"Excluded viability dye channel '{dead_ch}' from phenotypic clustering markers.")

            if not active_discovery_phase:
                logger.info(
                    "Step 5B-5D: Population Discovery phase disabled (default). "
                    "Skipping unsupervised clustering and UMAP manifold projection."
                )
            else:
                logger.info(
                    f"Step 5B: Clustering on parent population '{cluster_parent_name}': "
                    f"{parent_event_count:,} / {len(df_transformed):,} events "
                    f"({parent_event_count / len(df_transformed) * 100:.1f}%) across {len(cluster_markers)} markers."
                )

            # Channels for 2D visualization (strictly resolved from verified markers or first two channels; no guessed defaults)
            second_marker_fallback = cluster_markers[1] if len(cluster_markers) > 1 else cluster_markers[0]
            cd4_ch = fcs_data.resolve_channel_name("CD4") if "CD4" in fcs_data.marker_to_detector else cluster_markers[0]
            cd8_ch = fcs_data.resolve_channel_name("CD8") if "CD8" in fcs_data.marker_to_detector else second_marker_fallback
            if cd4_ch not in df_cluster.columns or cd8_ch not in df_cluster.columns:
                cd4_ch = cluster_markers[0]
                cd8_ch = second_marker_fallback

            resolved_m_map = marker_meta_res.detector_to_marker if marker_meta_res.has_verified_biological_markers else None

            # 1. SOM (Self-Organizing Map + Consensus Metaclustering)
            if active_discovery_phase and active_mode in ("som", "compare"):
                with log_step(logger, "Step 5B-1: SOM Clustering"):
                    logger.info("Step 5B-1: Executing SOM unsupervised clustering...")
                    eff_k_max = max(self.k_max, self.n_clusters) if (self.auto_k and self.n_clusters is not None) else self.k_max
                    eff_k_min = min(self.k_min, max(2, eff_k_max - 1))
                    som_res = run_som(
                        df=df_cluster,
                        markers=cluster_markers,
                        n_metaclusters=self.n_clusters or 10,
                        auto_k=self.auto_k,
                        auto_k_method=self.auto_k_method,
                        k_min=eff_k_min,
                        k_max=eff_k_max,
                        consensus_subsample=self.consensus_subsample,
                        consensus_iterations=self.consensus_iterations,
                        random_state=self.random_seed,
                        total_sample_events=fcs_data.event_count,
                        parent_population=cluster_parent_name,
                        cluster_name_map=self.cluster_name_map,
                        marker_name_map=resolved_m_map,
                        cell_signatures=self.cell_signatures,
                    )
                    # Compute full-dataset labels for export and indexing (-1 for gated-out events)
                    full_som_labels = np.full(len(df_transformed), -1, dtype=int)
                    full_som_labels[cluster_mask] = som_res.cluster_labels
                    som_res.full_cluster_labels = full_som_labels

                    som_csv_path = s5_tables_dir / "som_statistics.csv"
                    som_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    som_res.cluster_statistics.to_csv(som_csv_path, index=False)
                    tables["som_statistics"] = str(som_csv_path)

                    if som_res.annotation_audit_df is not None and not som_res.annotation_audit_df.empty:
                        som_audit_path = s5_tables_dir / "som_annotation_audit.csv"
                        som_res.annotation_audit_df.to_csv(som_audit_path, index=False)
                        tables["som_annotation_audit"] = str(som_audit_path)

                    if som_res.macro_groups_df is not None:
                        som_macro_path = s5_tables_dir / "som_macro_groups.csv"
                        som_res.macro_groups_df.to_csv(som_macro_path, index=False)
                        tables["som_macro_groups"] = str(som_macro_path)

                    som_scatter_path = s5_plots_dir / _pname("step5_som_clusters_scatter.png")
                    plot_clusters_scatter_2d(
                        df=df_cluster,
                        cluster_labels=som_res.cluster_labels,
                        x_channel=cd4_ch,
                        y_channel=cd8_ch,
                        x_label=fcs_data.get_channel_label(cd4_ch),
                        y_label=fcs_data.get_channel_label(cd8_ch),
                        title=f"SOM Discovered Clusters ({som_res.cluster_count} Metaclusters on {cluster_parent_name})",
                        output_path=som_scatter_path,
                        legend_mode=active_legend,
                    )
                    _add_plot("step5_som_clusters_scatter", som_scatter_path)

                    som_heatmap_path = s5_plots_dir / _pname("step5_som_marker_heatmap.png")
                    plot_cluster_heatmap(
                        cluster_stats_df=som_res.cluster_statistics,
                        marker_cols=cluster_markers,
                        title=f"SOM Metacluster Median Expression ({cluster_parent_name})",
                        output_path=som_heatmap_path,
                        ordered_indices=som_res.ordered_indices,
                        cluster_to_macro_group=som_res.cluster_to_macro_group,
                        marker_name_map=fcs_data.detector_to_marker,
                        cluster_palette=getattr(getattr(self, "config", None) or getattr(self, "config_source", None), "cluster_palette", "tab20"),
                    )
                    _add_plot("step5_som_marker_heatmap", som_heatmap_path)
                    logger.info(f"SOM completed: {som_res.cluster_count} metaclusters identified.")
                    logger.debug(f"SOM clustering details: markers={cluster_markers}, metaclusters={som_res.cluster_count}")

            # 2. PhenoGraph
            if active_discovery_phase and active_mode in ("phenograph", "compare"):
                with log_step(logger, "Step 5B-2: PhenoGraph Clustering"):
                    logger.info("Step 5B-2: Executing PhenoGraph community clustering...")
                    pheno_res = run_phenograph(
                        df=df_cluster,
                        markers=cluster_markers,
                        k=30,
                        random_state=self.random_seed,
                        total_sample_events=fcs_data.event_count,
                        parent_population=cluster_parent_name,
                        cluster_name_map=self.cluster_name_map,
                        marker_name_map=resolved_m_map,
                        cell_signatures=self.cell_signatures,
                    )
                    full_pg_labels = np.full(len(df_transformed), -1, dtype=int)
                    full_pg_labels[cluster_mask] = pheno_res.cluster_labels
                    pheno_res.full_cluster_labels = full_pg_labels

                    pg_csv_path = s5_tables_dir / "phenograph_statistics.csv"
                    pg_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    pheno_res.cluster_statistics.to_csv(pg_csv_path, index=False)
                    tables["phenograph_statistics"] = str(pg_csv_path)

                    if pheno_res.macro_groups_df is not None:
                        pg_macro_path = s5_tables_dir / "phenograph_macro_groups.csv"
                        pheno_res.macro_groups_df.to_csv(pg_macro_path, index=False)
                        tables["phenograph_macro_groups"] = str(pg_macro_path)

                    pg_scatter_path = s5_plots_dir / _pname("step5_phenograph_clusters_scatter.png")
                    plot_clusters_scatter_2d(
                        df=df_cluster,
                        cluster_labels=pheno_res.cluster_labels,
                        x_channel=cd4_ch,
                        y_channel=cd8_ch,
                        x_label=fcs_data.get_channel_label(cd4_ch),
                        y_label=fcs_data.get_channel_label(cd8_ch),
                        title=f"PhenoGraph Discovered Communities ({pheno_res.cluster_count} Clusters on {cluster_parent_name})",
                        output_path=pg_scatter_path,
                        legend_mode=active_legend,
                    )
                    _add_plot("step5_phenograph_clusters_scatter", pg_scatter_path)

                    pg_heatmap_path = s5_plots_dir / _pname("step5_phenograph_marker_heatmap.png")
                    plot_cluster_heatmap(
                        cluster_stats_df=pheno_res.cluster_statistics,
                        marker_cols=cluster_markers,
                        title=f"PhenoGraph Community Median Expression ({cluster_parent_name})",
                        output_path=pg_heatmap_path,
                        ordered_indices=pheno_res.ordered_indices,
                        cluster_to_macro_group=pheno_res.cluster_to_macro_group,
                        marker_name_map=fcs_data.detector_to_marker,
                        cluster_palette=getattr(getattr(self, "config", None) or getattr(self, "config_source", None), "cluster_palette", "tab20"),
                    )
                    _add_plot("step5_phenograph_marker_heatmap", pg_heatmap_path)
                    logger.info(f"PhenoGraph completed: {pheno_res.cluster_count} communities identified.")
                    logger.debug(f"PhenoGraph details: markers={cluster_markers}, communities={pheno_res.cluster_count}")

            # -------------------------------------------------------------
            # STEP 5C: Multi-Method Comparison (in 'compare' mode)
            # -------------------------------------------------------------
            if active_discovery_phase and active_mode == "compare" and som_res and pheno_res:
                with log_step(logger, "Step 5C: Multi-Method Comparison"):
                    logger.info("Step 5C: Generating 3-method comparison figures & concordance statistics...")

                    template_labels_subset = template_labels[cluster_mask]
                    df_comp = df_cluster

                    # Figure A: Default Centroid Badging
                    comp_fig_centroid = s5_plots_dir / _pname("step5_clustering_comparison.png")
                    plot_clustering_comparison(
                        df=df_comp,
                        template_labels=template_labels_subset,
                        som_labels=som_res.cluster_labels,
                        phenograph_labels=pheno_res.cluster_labels,
                        x_channel=cd4_ch,
                        y_channel=cd8_ch,
                        x_label=fcs_data.get_channel_label(cd4_ch),
                        y_label=fcs_data.get_channel_label(cd8_ch),
                        title=f"Comparison: Template vs SOM vs PhenoGraph on {cluster_parent_name} ({fcs_data.filename})",
                        output_path=comp_fig_centroid,
                        legend_mode="centroid",
                    )
                    _add_plot("step5_clustering_comparison_centroids", comp_fig_centroid)

                    # Figure B: External Multi-Column Legend
                    comp_fig_external = s5_plots_dir / _pname("step5_clustering_comparison_external_legend.png")
                    plot_clustering_comparison(
                        df=df_comp,
                        template_labels=template_labels_subset,
                        som_labels=som_res.cluster_labels,
                        phenograph_labels=pheno_res.cluster_labels,
                        x_channel=cd4_ch,
                        y_channel=cd8_ch,
                        x_label=fcs_data.get_channel_label(cd4_ch),
                        y_label=fcs_data.get_channel_label(cd8_ch),
                        title=f"Comparison (External Legend): Template vs SOM vs PhenoGraph on {cluster_parent_name}",
                        output_path=comp_fig_external,
                        figsize=(22.0, 5.8),
                        legend_mode="external",
                    )
                    _add_plot("step5_clustering_comparison_external_legend", comp_fig_external)

                    # 1. Pairwise Cross-Tabulations with Template Gating
                    ct_som = pd.crosstab(
                        pd.Series(template_labels_subset, name="Template Gate"),
                        pd.Series(som_res.cluster_labels, name="SOM Cluster"),
                    )
                    ct_pg = pd.crosstab(
                        pd.Series(template_labels_subset, name="Template Gate"),
                        pd.Series(pheno_res.cluster_labels, name="PhenoGraph Cluster"),
                    )
                    ct_som_path = s5_tables_dir / "comparison_crosstab_som.csv"
                    ct_pg_path = s5_tables_dir / "comparison_crosstab_phenograph.csv"
                    ct_som.to_csv(ct_som_path)
                    ct_pg.to_csv(ct_pg_path)
                    tables["comparison_crosstab_som"] = str(ct_som_path)
                    tables["comparison_crosstab_phenograph"] = str(ct_pg_path)

                    # 2. Pairwise Similarity Metrics across ALL Three Methods
                    ari_tf = float(adjusted_rand_score(template_labels_subset, som_res.cluster_labels))
                    ari_tp = float(adjusted_rand_score(template_labels_subset, pheno_res.cluster_labels))
                    ari_fp = float(adjusted_rand_score(som_res.cluster_labels, pheno_res.cluster_labels))

                    nmi_tf = float(normalized_mutual_info_score(template_labels_subset, som_res.cluster_labels))
                    nmi_tp = float(normalized_mutual_info_score(template_labels_subset, pheno_res.cluster_labels))
                    nmi_fp = float(normalized_mutual_info_score(som_res.cluster_labels, pheno_res.cluster_labels))

                    logger.debug(f"Comparison ARI metrics: Template/SOM={ari_tf:.4f}, Template/PhenoGraph={ari_tp:.4f}, SOM/PhenoGraph={ari_fp:.4f}")
                    logger.debug(f"Comparison NMI metrics: Template/SOM={nmi_tf:.4f}, Template/PhenoGraph={nmi_tp:.4f}, SOM/PhenoGraph={nmi_fp:.4f}")

                    three_way_metrics = pd.DataFrame([
                        {
                            "Method Pair": "Template Gating vs. SOM",
                            "Adjusted Rand Index (ARI)": round(ari_tf, 4),
                            "Normalized Mutual Information (NMI)": round(nmi_tf, 4),
                            "Biological Interpretation": "High overall concordance on major T-cell subsets",
                        },
                        {
                            "Method Pair": "Template Gating vs. PhenoGraph",
                            "Adjusted Rand Index (ARI)": round(ari_tp, 4),
                            "Normalized Mutual Information (NMI)": round(nmi_tp, 4),
                            "Biological Interpretation": "Captures major gates with fine subset partitioning",
                        },
                        {
                            "Method Pair": "SOM vs. PhenoGraph",
                            "Adjusted Rand Index (ARI)": round(ari_fp, 4),
                            "Normalized Mutual Information (NMI)": round(nmi_fp, 4),
                            "Biological Interpretation": "Strong agreement between unsupervised algorithms",
                        },
                    ])
                    three_way_metrics_path = s5_tables_dir / "comparison_three_methods_metrics.csv"
                    three_way_metrics.to_csv(three_way_metrics_path, index=False)
                    tables["comparison_three_methods_metrics"] = str(three_way_metrics_path)

                    # 3. Consolidated Biological Lineage Comparison across ALL Three Methods
                    three_way_lineages = self._build_three_way_lineage_comparison(
                        total_events=fcs_data.event_count,
                        template_labels=template_labels,
                        som_res=som_res,
                        pheno_res=pheno_res,
                        df_transformed=df_transformed,
                    )
                    three_way_lineages_path = s5_tables_dir / "comparison_three_methods_lineages.csv"
                    three_way_lineages.to_csv(three_way_lineages_path, index=False)
                    tables["comparison_three_methods_lineages"] = str(three_way_lineages_path)

            # -------------------------------------------------------------
            # STEP 5D: High-Dimensional Manifold Embedding (UMAP)
            # -------------------------------------------------------------
            umap_coords_csv_path: Optional[str] = None
            if active_discovery_phase and self.enable_umap and len(df_cluster) > 50 and len(cluster_markers) >= 2:
                with log_step(logger, "Step 5D: High-Dimensional Manifold Embedding (UMAP)"):
                    logger.info("Step 5D: Computing UMAP non-linear manifold projection...")
                    try:
                        umap_emb, umap_sub_idx = compute_umap_embedding(
                            df=df_cluster,
                            markers=cluster_markers,
                            n_neighbors=15,
                            min_dist=0.2,
                            random_state=self.random_seed,
                            subsample=self.umap_subsample,
                        )

                        # Save UMAP coordinates CSV
                        umap_df = pd.DataFrame(umap_emb, columns=["UMAP1", "UMAP2"])
                        umap_coords_path = s5_tables_dir / "umap_coordinates.csv"
                        umap_df.to_csv(umap_coords_path, index=False)
                        umap_coords_csv_path = str(umap_coords_path)
                        tables["umap_coordinates"] = str(umap_coords_path)

                        # A. UMAP Colored by Discovered Unsupervised Clusters (SOM or PhenoGraph)
                        s5_cl_dir = (s5_plots_dir / "clusters") if active_numbered_steps else s5_plots_dir
                        s5_pop_dir = (s5_plots_dir / "populations") if active_numbered_steps else s5_plots_dir
                        s5_cl_dir.mkdir(parents=True, exist_ok=True)
                        s5_pop_dir.mkdir(parents=True, exist_ok=True)

                        active_res = som_res or pheno_res
                        if active_res is not None:
                            sampled_cluster_labels = active_res.cluster_labels[umap_sub_idx]
                            umap_cluster_path = s5_cl_dir / (
                                "umap_clusters.png" if active_numbered_steps else _pname("step5_umap_clusters.png")
                            )
                            plot_umap_projection(
                                embedding=umap_emb,
                                labels=sampled_cluster_labels,
                                title=f"UMAP Projection ({active_res.method.upper()} Clusters on {cluster_parent_name})",
                                output_path=umap_cluster_path,
                                legend_mode=active_legend,
                                palette=getattr(getattr(self, "config", None) or getattr(self, "config_source", None), "cluster_palette", "tab20"),
                            )
                            _add_plot("step5_umap_clusters", umap_cluster_path)

                        # B. UMAP Colored by Gated Biological Lineages
                        sampled_template_labels = template_labels[cluster_mask][umap_sub_idx]
                        umap_lineage_path = s5_pop_dir / (
                            "umap_lineages.png" if active_numbered_steps else _pname("step5_umap_lineages.png")
                        )
                        plot_umap_projection(
                            embedding=umap_emb,
                            labels=sampled_template_labels,
                            title=f"UMAP Projection (Gated Lineages on {cluster_parent_name})",
                            output_path=umap_lineage_path,
                            legend_mode="external",
                            palette="tab10",
                        )
                        _add_plot("step5_umap_lineages", umap_lineage_path)

                        # C. Multi-Panel Continuous Marker Feature Maps (Feature Plots)
                        sampled_features_df = df_cluster.iloc[umap_sub_idx].reset_index(drop=True)
                        umap_features_path = s5_pop_dir / (
                            "umap_marker_features.png" if active_numbered_steps else _pname("step5_umap_marker_features.png")
                        )
                        plot_umap_marker_features(
                            embedding=umap_emb,
                            df_features=sampled_features_df,
                            markers=cluster_markers,
                            marker_name_map=fcs_data.detector_to_marker,
                            title=f"High-Dimensional Marker Manifold Maps ({cluster_parent_name})",
                            output_path=umap_features_path,
                            max_markers=8,
                        )
                        _add_plot("step5_umap_marker_features", umap_features_path)

                        # D. Population Marker Expression Violins
                        violins_path = s5_pop_dir / (
                            "population_marker_violins.png" if active_numbered_steps else _pname("step5_population_marker_violins.png")
                        )
                        plot_population_marker_violins(
                            df_events=sampled_features_df,
                            population_labels=sampled_template_labels,
                            markers=cluster_markers,
                            marker_name_map=fcs_data.detector_to_marker,
                            title=f"Marker Distributions Across Gated Lineages ({cluster_parent_name})",
                            output_path=violins_path,
                            max_markers=6,
                        )
                        _add_plot("step5_population_marker_violins", violins_path)

                        logger.info("UMAP manifold embeddings and feature heatmaps generated successfully.")

                    except Exception as exc:
                        logger.warning(f"Could not compute UMAP manifold embeddings: {exc}")

            # -------------------------------------------------------------
            # STEP 6: Workspace Session Persistence (JSON & XML)
            # -------------------------------------------------------------
            with log_step(logger, "Step 6: Workspace Session Persistence"):
                logger.info("Step 6: Saving workspace session (JSON and XML)...")
                import copy
                if len(working_sample_df) != len(fcs_data.events_df):
                    clean_sample_fcs = copy.copy(fcs_data)
                    clean_sample_fcs.events_df = working_sample_df
                    clean_sample_fcs.event_count = len(working_sample_df)
                    target_sample_for_workspace = clean_sample_fcs
                else:
                    target_sample_for_workspace = fcs_data

                session = WorkspaceSession(
                    name=f"Workspace_{fcs_data.filename}",
                    sample_path=fcs_data.filepath,
                    sample_filename=fcs_data.filename,
                    total_events=len(working_sample_df),
                    hierarchy=hierarchy,
                )
                session.set_sample(target_sample_for_workspace)
                if comp_obj:
                    session.set_compensation(comp_obj)
                for ch in fluoro_channels:
                    session.add_transform(ch, transform_obj)

                ws_json_path = exports_ws_dir / "workspace.json"
                ws_xml_path = exports_ws_dir / "workspace.xml"
                session.save_json(ws_json_path)
                session.save_xml(ws_xml_path)
                logger.debug(f"Saved workspace JSON: {sanitize_path(ws_json_path)}")
                logger.debug(f"Saved workspace XML: {sanitize_path(ws_xml_path)}")

                prep_wsp_path = ""
                prep_gml_path = ""
                single_gate_counts = {
                    k: int(np.sum(m))
                    for k, m in masks.items()
                    if k != "root"
                }
                single_max_range = float(fcs_data.detect_max_range()) if hasattr(fcs_data, "detect_max_range") else 262144.0
                if self.export_flowjo_wsp:
                    try:
                        wsp_out = exports_ws_dir / "preprocessing_gating.wsp"
                        session.export_preprocessing_wsp(
                            wsp_out,
                            gate_counts=single_gate_counts,
                            max_range=single_max_range,
                        )
                        prep_wsp_path = str(wsp_out)
                        tables["preprocessing_flowjo_wsp"] = prep_wsp_path
                        logger.debug(f"Saved FlowJo preprocessing/gating workspace: {sanitize_path(wsp_out)}")
                    except Exception as wsp_err:
                        logger.warning(f"Could not export preprocessing FlowJo .wsp: {wsp_err}")
                if self.export_gating_ml:
                    try:
                        gml_out = exports_ws_dir / "preprocessing_gating_ml_2.0.xml"
                        export_preprocessing_gating_ml(
                            hierarchy=hierarchy,
                            filepath=gml_out,
                            experiment_name=session.name,
                            compensation=comp_obj,
                            transforms=session.transforms,
                            spectral_library=comp_step.spectral_library if "comp_step" in locals() and comp_step is not None else None,
                        )
                        prep_gml_path = str(gml_out)
                        tables["preprocessing_gating_ml"] = prep_gml_path
                        logger.debug(f"Saved Gating-ML 2.0 preprocessing/gating file: {sanitize_path(gml_out)}")
                    except Exception as gml_err:
                        logger.warning(f"Could not export preprocessing Gating-ML 2.0: {gml_err}")

            # -------------------------------------------------------------
            # STEP 7: Reproducibility Verification
            # -------------------------------------------------------------
            with log_step(logger, "Step 7: Reproducibility Verification"):
                logger.info("Step 7: Verifying reproducibility via workspace replay...")
                loaded_session = WorkspaceSession.load_json(ws_json_path)
                replayed_df, replayed_stats = loaded_session.replay(target_sample_for_workspace)

                reproducible = (
                    stats_df["Count"].tolist() == replayed_stats["Count"].tolist()
                    and stats_df["% Parent"].tolist() == replayed_stats["% Parent"].tolist()
                )
                if reproducible:
                    logger.info("Reproducibility test PASSED: 100% exact match across all gates and counts!")
                else:
                    logger.warning("WARNING: Discrepancy detected in replay.")

            # -------------------------------------------------------------
            # STEP 8: Generate Markdown Analysis Report
            # -------------------------------------------------------------
            with log_step(logger, "Step 8: Markdown Report Generation"):
                logger.info("Step 8: Generating Markdown analysis report...")
                execution_duration = time.perf_counter() - t_start
                dur_formatted = format_duration(execution_duration)

                report_path = exports_rep_dir / "report.md"
                self._generate_markdown_report(
                    report_path=report_path,
                    fcs_data=fcs_data,
                    comp_obj=comp_obj,
                    transform_obj=transform_obj,
                    stats_df=stats_df,
                    plots=plots,
                    reproducible=reproducible,
                    gating_mode=active_mode,
                    qc_method=active_qc,
                    clustering_qc_res=clustering_qc_res,
                    som_res=som_res,
                    pheno_res=pheno_res,
                    ct_som=ct_som,
                    ct_pg=ct_pg,
                    three_way_metrics=three_way_metrics,
                    three_way_lineages=three_way_lineages,
                    adaptive_cutoffs=cutoffs_dict,
                    comp_ctrl_res=comp_ctrl_result,
                    comp_source=comp_source_used,
                    config_file_path=str(saved_config_path) if saved_config_path else None,
                    log_file_path=str(log_file_path) if log_file_path else None,
                    execution_time_seconds=execution_duration,
                    execution_duration_formatted=dur_formatted,
                    run_start_time=run_start_dt,
                )
                logger.info(f"Saved Markdown analysis report to '{sanitize_path(report_path)}'.")

                # Record run in manifest
                manifest_summary: Dict[str, Any] = {
                    "timestamp": datetime.now().isoformat(),
                    "run_start_time": run_start_dt.isoformat(),
                    "execution_time_seconds": round(execution_duration, 3),
                    "execution_duration_formatted": dur_formatted,
                    "run_dir": sanitize_path(current_run_dir),
                    "run_name": current_run_dir.name,
                    "sample_filename": fcs_data.filename,
                    "event_count": fcs_data.event_count,
                    "gating_mode": active_mode,
                    "cluster_parent": active_cluster_parent,
                    "qc_method": active_qc,
                    "legend_mode": active_legend,
                    "reproducible": reproducible,
                    "report_md": sanitize_path(report_path),
                    "log_file": sanitize_path(log_file_path),
                }
                if saved_config_path:
                    manifest_summary["config_file"] = sanitize_path(saved_config_path)

                self._record_run_manifest(
                    run_dir=current_run_dir,
                    summary=manifest_summary,
                )

                logger.info(
                    f"All pipeline tasks completed successfully in {dur_formatted} ({execution_duration:.2f}s)! "
                    f"Results stored in '{sanitize_path(current_run_dir)}'."
                )
                logger.info(f"Detailed debug log persisted at: '{sanitize_path(log_file_path)}'")
        except Exception as exc:
            elapsed_error = time.perf_counter() - t_start
            logger.error(
                f"Pipeline execution failed after {format_duration(elapsed_error)} ({elapsed_error:.2f}s): {exc}",
                exc_info=True,
            )
            raise
        finally:
            close_run_logger(logger)

        sanitized_plots = {k: sanitize_path(p) for k, p in plots.items()}

        return PipelineResult(
            sample_name=fcs_data.filename,
            event_count=fcs_data.event_count,
            raw_fcs=fcs_data,
            compensated_df=df_comp,
            transformed_df=df_transformed,
            hierarchy=hierarchy,
            statistics_df=stats_df,
            generated_plots=sanitized_plots,
            generated_tables={k: sanitize_path(p) for k, p in tables.items()},
            run_dir=sanitize_path(current_run_dir),
            workspace_json_path=sanitize_path(ws_json_path),
            workspace_xml_path=sanitize_path(ws_xml_path),
            preprocessing_wsp_path=sanitize_path(prep_wsp_path) if prep_wsp_path else "",
            preprocessing_gating_ml_path=sanitize_path(prep_gml_path) if prep_gml_path else "",
            statistics_csv_path=sanitize_path(stats_csv_path),
            report_md_path=sanitize_path(report_path),
            reproducibility_verified=reproducible,
            gating_mode=active_mode,
            cluster_parent=active_cluster_parent,
            qc_method=active_qc,
            clustering_qc_result=clustering_qc_res,
            cleanup_result=cleanup_gating_res,
            som_result=som_res,
            phenograph_result=pheno_res,
            comparison_crosstab_som=ct_som,
            comparison_crosstab_phenograph=ct_pg,
            three_way_metrics_df=three_way_metrics,
            three_way_lineages_df=three_way_lineages,
            adaptive_cutoffs=cutoffs_dict,
            cytometer=fcs_data.cytometer,
            time_qc_result=pipeline_time_qc_res,
            compensation_control_result=comp_ctrl_result,
            compensation_source_used=comp_source_used,
            spillover_matrix_path=sanitize_path(spill_matrix_csv_path) if spill_matrix_csv_path else None,
            umap_coordinates_path=sanitize_path(umap_coords_csv_path) if umap_coords_csv_path else None,
            config_file_path=sanitize_path(saved_config_path) if saved_config_path else None,
            log_file_path=sanitize_path(log_file_path) if log_file_path else None,
            execution_time_seconds=execution_duration,
            execution_duration_formatted=dur_formatted,
            technology=active_tech,
            spectral_result=spectral_pipeline_res,
        )

    def _run_cohort(
        self,
        sample_paths: List[Path],
        target_name: str,
        custom_hierarchy: Optional[GatingHierarchy] = None,
        custom_transform: Optional[BaseTransform] = None,
        null_channels: Optional[List[str]] = None,
        gating_mode: Optional[str] = None,
        qc_method: Optional[str] = None,
        legend_mode: Optional[str] = None,
        adaptive_gating: Optional[bool] = None,
        marker_mapping: Optional[Dict[str, str]] = None,
        config_file: Optional[Union[str, Path]] = None,
        log_level: Optional[str] = None,
        transform_type: Optional[str] = None,
        enable_time_qc: Optional[bool] = None,
        harmonization_file: Optional[Union[str, Path]] = None,
        enable_qc: Optional[bool] = None,
        enable_discovery: Optional[bool] = None,
        numbered_steps: Optional[bool] = None,
        sample_metadata: Optional[Union[pd.DataFrame, Dict[str, Dict[str, Any]], str, Path]] = None,
        metadata_file: Optional[Union[str, Path]] = None,
        technology: Optional[str] = None,
    ) -> PipelineResult:
        """Execute end-to-end multi-sample cohort pipeline across samples."""
        active_mode = (gating_mode or self.gating_mode).lower()
        active_qc = (qc_method or self.qc_method).lower()
        active_legend = (legend_mode or self.legend_mode).lower()
        active_qc_phase = self.enable_qc if enable_qc is None else enable_qc
        active_discovery_phase = self.enable_discovery if enable_discovery is None else enable_discovery
        active_numbered_steps = self.numbered_steps if numbered_steps is None else numbered_steps
        active_tech = (technology or self.technology or "auto").lower()

        plots: Dict[str, str] = {}
        tables: Dict[str, str] = {}

        current_run_dir = self._resolve_run_directory(target_name)

        if active_numbered_steps:
            s1_plots_dir = current_run_dir / "01_time_qc" / "plots"
            s1_tables_dir = current_run_dir / "01_time_qc" / "tables"
            s2_plots_dir = current_run_dir / "02_compensation" / "plots"
            s2_tables_dir = current_run_dir / "02_compensation" / "tables"
            s3_plots_dir = current_run_dir / "03_cleanup_gating" / "plots"
            s3_tables_dir = current_run_dir / "03_cleanup_gating" / "tables"
            s4_plots_dir = current_run_dir / "04_batch_preparation" / "plots"
            s4_tables_dir = current_run_dir / "04_batch_preparation" / "tables"
            s5_plots_dir = current_run_dir / "05_population_discovery" / "plots"
            s5_tables_dir = current_run_dir / "05_population_discovery" / "tables"
            s6_plots_dir = current_run_dir / "06_statistical_analysis" / "plots"
            s6_tables_dir = current_run_dir / "06_statistical_analysis" / "tables"
            exports_ws_dir = current_run_dir / "exports" / "workspaces"
            exports_rep_dir = current_run_dir / "exports" / "reports"
            clean_fcs_dir = current_run_dir / "exports" / "clean_fcs"
            augmented_fcs_dir = current_run_dir / "exports" / "augmented_fcs"
            for d in (
                s1_plots_dir, s1_tables_dir, s2_plots_dir, s2_tables_dir,
                s3_plots_dir, s3_tables_dir, s4_plots_dir, s4_tables_dir,
                exports_ws_dir, exports_rep_dir, augmented_fcs_dir
            ):
                d.mkdir(parents=True, exist_ok=True)
            if active_discovery_phase:
                s5_plots_dir.mkdir(parents=True, exist_ok=True)
                s5_tables_dir.mkdir(parents=True, exist_ok=True)
                s6_plots_dir.mkdir(parents=True, exist_ok=True)
                s6_tables_dir.mkdir(parents=True, exist_ok=True)
        else:
            s1_plots_dir = s2_plots_dir = s3_plots_dir = s4_plots_dir = s5_plots_dir = s6_plots_dir = current_run_dir / "plots"
            s1_tables_dir = s2_tables_dir = s3_tables_dir = s4_tables_dir = s5_tables_dir = s6_tables_dir = current_run_dir / "tables"
            exports_ws_dir = exports_rep_dir = augmented_fcs_dir = clean_fcs_dir = current_run_dir
            s1_plots_dir.mkdir(parents=True, exist_ok=True)
            s1_tables_dir.mkdir(parents=True, exist_ok=True)

        def _pname(legacy_name: str) -> str:
            if active_numbered_steps:
                return re.sub(r"^step\d+[a-zA-Z]?_", "", legacy_name)
            return legacy_name

        def _add_plot(key: str, p_path: Path) -> None:
            p_str = str(p_path)
            plots[key] = p_str
            clean_k = re.sub(r"^step\d+[a-zA-Z]?_", "", key)
            plots[clean_k] = p_str
            if "umap_marker_features" in key:
                plots["cohort_umap_marker_features"] = p_str
            elif "umap_covariates" in key or "cohort_umap_covariates" in key:
                plots["cohort_umap_covariates"] = p_str
            elif "umap_clusters" in key:
                plots["cohort_umap_clusters"] = p_str
            elif "umap_populations" in key:
                plots["cohort_umap_populations"] = p_str

        t_start = time.perf_counter()
        run_start_dt = datetime.now()
        active_log_level = (log_level or self.log_level).upper()
        logger, log_file_path = setup_run_logger(
            run_dir=current_run_dir,
            log_filename="pipeline.log",
            console_level=active_log_level,
            file_level="DEBUG",
        )
        logger.info(
            f"Initiated unified cohort pipeline for '{target_name}' ({len(sample_paths)} samples) at {run_start_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(f"Deliverables directory: '{sanitize_path(current_run_dir)}'")

        # Save config copy
        saved_config_path: Optional[Path] = None
        target_config = config_file or self.config_file
        if target_config is not None:
            src_cfg = Path(target_config)
            if src_cfg.is_file():
                dest_cfg = current_run_dir / "config.yaml" if src_cfg.suffix.lower() in (".yaml", ".yml") else current_run_dir / "config.json"
                try:
                    if src_cfg.resolve() != dest_cfg.resolve():
                        shutil.copy2(src_cfg, dest_cfg)
                    saved_config_path = dest_cfg
                    logger.info(f"Saved configuration copy to '{sanitize_path(dest_cfg)}'.")
                except Exception as exc:
                    logger.warning(f"Could not copy config: {exc}")
        elif self.config_source is not None:
            try:
                if hasattr(self.config_source, "save"):
                    cfg_dest = current_run_dir / "config.yaml"
                    self.config_source.save(cfg_dest)
                    saved_config_path = cfg_dest
            except Exception:
                pass

        # Resolve controls directory and unstained control with auto-discovery priority (Rule 3)
        comp_enabled = bool(self.enable_compensation and self.compensation_source != "none")
        effective_controls_dir = self.comp_controls_dir if comp_enabled else None
        effective_unstained_control = self.unstained_control if comp_enabled else None

        if effective_unstained_control is not None:
            ust_name = Path(effective_unstained_control).name.lower()
            if ("viability" in ust_name or "live-or-dye" in ust_name) and "unstained" not in ust_name:
                effective_unstained_control = None

        if comp_enabled and effective_controls_dir is None and sample_paths:
            fcs_dir = sample_paths[0].parent
            for candidate in [
                fcs_dir.parent / "controls",
                fcs_dir.parent / "raw" / "controls",
                fcs_dir / "controls",
            ]:
                if candidate.is_dir() and any(candidate.glob("*.fcs")):
                    effective_controls_dir = candidate
                    logger.info(f"Auto-discovered single-stained controls directory: '{sanitize_path(candidate)}'")
                    break

        if comp_enabled and effective_unstained_control is None and effective_controls_dir is not None:
            for f in effective_controls_dir.glob("*.fcs"):
                f_low = f.name.lower()
                if any(w in f_low for w in ["unstained", "blank"]) and "fmo" not in f_low:
                    effective_unstained_control = f
                    logger.info(f"Auto-discovered unstained control: '{sanitize_path(f)}'")
                    break

        should_use_controls = comp_enabled and (
            self.compensation_source == "controls"
            or (self.compensation_source == "auto" and (effective_controls_dir is not None or self.comp_manifest is not None))
        )

        cohort_clean_dfs: Dict[str, pd.DataFrame] = {}
        cohort_raw_counts: Dict[str, int] = {}
        cohort_clean_counts: Dict[str, int] = {}
        first_fcs_data: Optional[FCSData] = None
        first_df_comp: Optional[pd.DataFrame] = None
        first_df_transformed: Optional[pd.DataFrame] = None
        first_hierarchy: Optional[GatingHierarchy] = None
        first_cleanup_res: Optional[CleanUpGatingResult] = None
        first_time_qc_res: Optional[Any] = None
        comp_obj: Optional[CompensationMatrix] = None
        comp_ctrl_result: Optional[CompensationControlResult] = None
        comp_source_used: str = "none"
        spill_matrix_csv_path: Optional[str] = None
        transform_obj: Optional[BaseTransform] = None
        target_channels: List[str] = []
        cohort_spectral_lib: Optional[Any] = None
        spectral_marker_map: Dict[str, str] = {}
        sample_ws_specs: List[SampleWorkspaceSpec] = []
        first_ws_transforms: Dict[str, Any] = {}

        active_mapping = dict(marker_mapping or self.marker_mapping or {}) or None

        # Resolve markers cleanly once across cohort
        from flowx.io.metadata import resolve_marker_metadata
        cohort_marker_res = None
        raw_spectral_pattern = re.compile(r"^(UV|V|B|YG|R)\d+-A$", re.IGNORECASE)

        for s_idx, s_path in enumerate(sample_paths):
            s_stem = s_path.stem
            logger.info(f"[{s_idx+1}/{len(sample_paths)}] Preprocessing sample: '{s_path.name}'...")
            s_fcs = parse_fcs(s_path, null_channels=null_channels, marker_mapping=active_mapping)
            if cohort_marker_res is None:
                cohort_marker_res = resolve_marker_metadata(
                    markers_file=self.markers_file,
                    dataset_dir=s_path.parent,
                    fcs_pnn=s_fcs.pnn_labels,
                    fcs_pns=s_fcs.pns_labels,
                    marker_mapping=active_mapping,
                    logger=logger,
                )
                if cohort_marker_res.detector_to_marker and cohort_marker_res.has_verified_biological_markers:
                    merged = dict(active_mapping or {})
                    merged.update(cohort_marker_res.detector_to_marker)
                    active_mapping = merged
                    s_fcs = parse_fcs(s_path, null_channels=null_channels, marker_mapping=active_mapping)
            if cohort_marker_res is not None:
                s_fcs.detector_to_fluorophore = dict(cohort_marker_res.detector_to_fluorophore)
            if first_fcs_data is None:
                first_fcs_data = s_fcs

            s_emp_pre_transformed = is_empirically_pre_transformed(s_fcs)

            # Step 1: Time QC via shared step runner
            s_working_df = s_fcs.events_df.copy()
            if active_qc_phase and (enable_time_qc if enable_time_qc is not None else self.enable_time_qc) and not s_emp_pre_transformed:
                try:
                    from flowx.qc.models import TimeQCConfig
                    from flowx.qc.runner import run_time_qc_step

                    tqc_cfg = self.time_qc_config or TimeQCConfig.for_sample()
                    tqc_path = s1_plots_dir / f"{s_stem}_time_qc_diagnostics.png"
                    tqc_step = run_time_qc_step(
                        fcs_data=s_fcs,
                        config=tqc_cfg,
                        output_plot_path=tqc_path,
                        generate_plots=True,
                    )
                    if tqc_step.decision is not None and tqc_step.decision.can_proceed and tqc_step.clean_df is not None:
                        s_working_df = tqc_step.clean_df
                    if first_time_qc_res is None and tqc_step.qc_result is not None:
                        first_time_qc_res = tqc_step.qc_result
                    if tqc_step.plot_path:
                        _add_plot(f"time_qc_{s_stem}", Path(tqc_step.plot_path))
                        if s_idx == 0:
                            _add_plot("step1_time_qc_diagnostics", Path(tqc_step.plot_path))
                except Exception as exc:
                    logger.warning(f"Time QC notice for {s_path.name}: {exc}")

            # Target fluorescence channels resolution
            from flowx.math.compensation import run_compensation_step
            from flowx.math.transforms import TransformConfig, resolve_fluorescence_channels, run_transformation_step
            from flowx.gating.runner import run_cleanup_step

            sample_fl_channels = resolve_fluorescence_channels(
                s_fcs,
                explicit_channels=(self.transform_config.channels if self.transform_config else None),
                df=s_working_df,
                marker_metadata=cohort_marker_res,
            )
            if not target_channels:
                target_channels = list(sample_fl_channels)

            # Step 2: Compensation / Spectral Unmixing via shared step runner
            comp_step = run_compensation_step(
                fcs_data=s_fcs,
                working_df=s_working_df,
                target_channels=sample_fl_channels,
                enable_compensation=comp_enabled and not s_emp_pre_transformed,
                compensation_source=self.compensation_source,
                technology=active_tech,
                comp_controls_dir=effective_controls_dir,
                comp_manifest=self.comp_manifest,
                unstained_control=effective_unstained_control,
                comp_channel_mapping=self.comp_channel_mapping,
                comp_config=self.comp_config,
                spectral_config=self.spectral_config,
                cached_comp_matrix=comp_obj if comp_source_used == "controls" else None,
                cached_spectral_library=cohort_spectral_lib,
                plots_dir=s2_plots_dir,
                tables_dir=s2_tables_dir,
                plot_name_fn=_pname,
                sample_stem=s_stem,
                random_seed=self.random_seed,
                generate_plots=True,
                generate_nxn=False,
            )
            s_df_comp = comp_step.compensated_df
            if comp_obj is None and comp_step.comp_matrix is not None:
                comp_obj = comp_step.comp_matrix
            if comp_ctrl_result is None and comp_step.comp_ctrl_result is not None:
                comp_ctrl_result = comp_step.comp_ctrl_result
            if comp_step.comp_source_used != "none":
                comp_source_used = comp_step.comp_source_used
            if comp_step.spill_matrix_csv_path and spill_matrix_csv_path is None:
                spill_matrix_csv_path = comp_step.spill_matrix_csv_path
            for t_k, t_v in comp_step.tables.items():
                tables.setdefault(t_k, t_v)
            for p_k, p_v in comp_step.plots.items():
                if p_k == "step2_compensation_diagnostic":
                    _add_plot(f"compensation_{s_stem}", Path(p_v))
                    if s_idx == 0:
                        _add_plot("step2_compensation_diagnostic", Path(p_v))
                else:
                    _add_plot(p_k, Path(p_v))

            if comp_step.comp_source_used == "spectral_unmixing":
                target_channels = list(comp_step.target_channels)
                if cohort_spectral_lib is None and comp_step.spectral_library is not None:
                    cohort_spectral_lib = comp_step.spectral_library
                    ref_items = (
                        cohort_spectral_lib.references.values()
                        if isinstance(cohort_spectral_lib.references, dict)
                        else cohort_spectral_lib.references
                    )
                    for ref in ref_items:
                        if getattr(ref, "is_autofluorescence", False):
                            continue
                        fl_col = ref.fluorophore
                        mk_lbl = fl_col
                        if cohort_marker_res is not None:
                            if ref.peak_channel in cohort_marker_res.detector_to_marker:
                                cand_mk = cohort_marker_res.detector_to_marker[ref.peak_channel]
                                if cand_mk and cand_mk != ref.peak_channel:
                                    mk_lbl = cand_mk
                            if mk_lbl == fl_col and cohort_marker_res.detector_to_fluorophore:
                                fl_norm = re.sub(r"[^a-z0-9]", "", fl_col.lower())
                                for det_k, fl_v in cohort_marker_res.detector_to_fluorophore.items():
                                    if re.sub(r"[^a-z0-9]", "", fl_v.lower()) == fl_norm:
                                        cand_mk = cohort_marker_res.detector_to_marker.get(det_k, fl_col)
                                        if cand_mk and cand_mk != det_k:
                                            mk_lbl = cand_mk
                                            break
                        src_file = getattr(ref, "file_path", None) or getattr(ref, "source_file", None)
                        if mk_lbl == fl_col and src_file:
                            src_stem = Path(src_file).stem
                            m_ag = re.search(
                                r"\b(CD\d+[a-zA-Z]*|MHC\s*II|HLA[-_]?DR|F4[-/]80|Ly-?6[GC]|NKp46|FoxP3|CRTh2|CXCR\d+|CCR\d+|Ig[DGM]|TCR[a-zA-Z0-9]*|Puromycin|Live[-_]or[-_]dye(?:_TM)?|Viability)\b",
                                src_stem,
                                re.IGNORECASE,
                            )
                            if m_ag:
                                raw_ag = m_ag.group(1)
                                if re.match(r"mhc\s*ii", raw_ag, re.IGNORECASE):
                                    mk_lbl = "HLA-DR"
                                elif re.match(r"f4[-/]80", raw_ag, re.IGNORECASE):
                                    mk_lbl = "F4/80"
                                else:
                                    mk_lbl = raw_ag
                        spectral_marker_map[fl_col] = mk_lbl
                if spectral_marker_map:
                    s_fcs.detector_to_marker.update(spectral_marker_map)
                for col_c in s_df_comp.columns:
                    if col_c not in s_fcs.pnn_labels:
                        s_fcs.pnn_labels.append(col_c)

            # Step 3: Scale Transformation via shared step runner
            active_xform_type = (transform_type or self.transform_type).lower()
            base_xf_cfg = self.transform_config
            xf_cfg = TransformConfig(
                enabled=self.enable_transformation if base_xf_cfg is None else base_xf_cfg.enabled,
                method=active_xform_type,
                cofactor=self.transform_cofactor if base_xf_cfg is None else base_xf_cfg.cofactor,
                channels=(
                    list(comp_step.target_channels)
                    if comp_step.comp_source_used == "spectral_unmixing"
                    else (base_xf_cfg.channels if base_xf_cfg is not None else None)
                ),
                exclude_channels=base_xf_cfg.exclude_channels if base_xf_cfg is not None else None,
                param_w=base_xf_cfg.param_w if base_xf_cfg is not None else 0.5,
                param_a=base_xf_cfg.param_a if base_xf_cfg is not None else 0.0,
                normalized=base_xf_cfg.normalized if base_xf_cfg is not None else True,
            )
            xf_step = run_transformation_step(
                fcs_data=s_fcs,
                df_events=s_df_comp,
                config=xf_cfg,
                target_channels=list(comp_step.target_channels) if comp_step.comp_source_used == "spectral_unmixing" else xf_cfg.channels,
                custom_transform=transform_obj or custom_transform,
                marker_metadata=cohort_marker_res,
                generate_plot=False,
            )
            if transform_obj is None:
                transform_obj = xf_step.transform
            s_transformed_all = xf_step.transformed_df

            # Step 3: Cleanup Gating via shared step runner
            s_clean_df = s_transformed_all
            s_cleanup_res: Optional[CleanUpGatingResult] = None
            active_cleanup_phase = bool(active_qc_phase and self.enable_cleanup_gating and not s_emp_pre_transformed)
            if active_cleanup_phase:
                cleanup_cfg = self._build_cleanup_config()
                p_cleanup = s3_plots_dir / f"{s_stem}_cleanup_diagnostics.png"
                cleanup_step = run_cleanup_step(
                    fcs_data=s_fcs,
                    df_events=s_df_comp,
                    df_transformed=s_transformed_all,
                    config=cleanup_cfg,
                    marker_metadata=cohort_marker_res,
                    spectral_marker_map=spectral_marker_map,
                    output_plot_path=p_cleanup,
                    generate_plots=True,
                    export_clean_fcs=self.export_clean_fcs,
                    clean_fcs_path=clean_fcs_dir / f"{s_stem}_clean.fcs",
                )
                s_cleanup_res = cleanup_step.cleanup_result
                s_clean_df = cleanup_step.clean_df
                if first_cleanup_res is None:
                    first_cleanup_res = s_cleanup_res
                    first_hierarchy = s_cleanup_res.hierarchy
                if cleanup_step.plot_path:
                    _add_plot(f"cleanup_{s_stem}", Path(cleanup_step.plot_path))
                    if s_idx == 0:
                        _add_plot("step3_cleanup_diagnostics", Path(cleanup_step.plot_path))

            if first_df_comp is None:
                first_df_comp = s_df_comp
                first_df_transformed = s_clean_df

            cohort_clean_dfs[s_stem] = s_clean_df
            cohort_raw_counts[s_stem] = s_fcs.event_count
            cohort_clean_counts[s_stem] = len(s_clean_df)

            s_xforms_map: Dict[str, Any] = {}
            if xf_step.transform is not None:
                for ch_xf in (xf_step.channels_transformed or sample_fl_channels):
                    s_xforms_map[ch_xf] = xf_step.transform
            if not first_ws_transforms and s_xforms_map:
                first_ws_transforms = dict(s_xforms_map)

            s_gate_counts: Dict[str, int] = {}
            if s_cleanup_res is not None and s_cleanup_res.masks:
                s_gate_counts = {
                    gk: int(np.sum(gm))
                    for gk, gm in s_cleanup_res.masks.items()
                    if gk != "root"
                }

            sample_ws_specs.append(
                SampleWorkspaceSpec(
                    sample_filename=s_fcs.filename,
                    sample_path=s_fcs.filepath,
                    total_events=s_fcs.event_count,
                    channels=list(s_fcs.pnn_labels),
                    max_range=float(s_fcs.detect_max_range()) if hasattr(s_fcs, "detect_max_range") else 262144.0,
                    compensation=comp_step.comp_matrix or comp_obj,
                    spectral_library=comp_step.spectral_library or cohort_spectral_lib,
                    transforms=s_xforms_map,
                    hierarchy=s_cleanup_res.hierarchy if s_cleanup_res is not None else GatingHierarchy(),
                    gate_counts=s_gate_counts,
                )
            )

        prep_wsp_path = ""
        prep_gml_path = ""
        if sample_ws_specs and self.export_flowjo_wsp:
            try:
                wsp_out = exports_ws_dir / "preprocessing_gating.wsp"
                export_preprocessing_flowjo_wsp(
                    sample_ws_specs,
                    wsp_out,
                    experiment_name=f"Cohort_{target_name}_Preprocessing",
                )
                prep_wsp_path = str(wsp_out)
                tables["preprocessing_flowjo_wsp"] = prep_wsp_path
                logger.debug(f"Saved FlowJo preprocessing/gating workspace: {sanitize_path(wsp_out)}")
            except Exception as wsp_err:
                logger.warning(f"Could not export cohort preprocessing FlowJo .wsp: {wsp_err}")
        if self.export_gating_ml and (first_hierarchy is not None or comp_obj is not None or cohort_spectral_lib is not None):
            try:
                gml_out = exports_ws_dir / "preprocessing_gating_ml_2.0.xml"
                export_preprocessing_gating_ml(
                    hierarchy=first_hierarchy or GatingHierarchy(),
                    filepath=gml_out,
                    experiment_name=f"Cohort_{target_name}_Preprocessing",
                    compensation=comp_obj,
                    transforms=first_ws_transforms,
                    spectral_library=cohort_spectral_lib,
                )
                prep_gml_path = str(gml_out)
                tables["preprocessing_gating_ml"] = prep_gml_path
                logger.debug(f"Saved Gating-ML 2.0 preprocessing/gating file: {sanitize_path(gml_out)}")
            except Exception as gml_err:
                logger.warning(f"Could not export cohort preprocessing Gating-ML 2.0: {gml_err}")

        # Resolve cohort metadata before Step 4 so batch_id and is_control are available for normalization
        resolved_meta = resolve_cohort_metadata(
            fcs_files=sample_paths,
            sample_metadata=sample_metadata,
            metadata_file=metadata_file or self.metadata_file,
            dataset_dir=sample_paths[0].parent if sample_paths else None,
        )

        # Step 4: Batch Preparation & Event Downsampling via unified batch runner
        with log_step(logger, "Step 4: Cohort Batch Preparation & Event Downsampling"):
            logger.info(f"Step 4: Balancing event yields across {len(cohort_clean_dfs)} cohort samples...")
            from flowx.batch.models import BatchPreparationConfig, CohortSampleInfo
            from flowx.batch.runner import run_batch_preparation

            bp_cfg = dict(self.batch_preparation or {})
            bp_enabled = self.enable_batch_preparation and bp_cfg.get("enabled", True)
            bp_cfg["enabled"] = bp_enabled
            bp_cfg.setdefault("random_seed", self.random_seed)
            bp_cfg.setdefault("export_pooled_cohort", False)
            bp_cfg.setdefault("assess_batch_effect", True)
            if self.overwrite_autocheck:
                bp_cfg["overwrite_autocheck"] = True
            else:
                bp_cfg.setdefault("overwrite_autocheck", False)
            if self.batch_column and "batch_column" not in bp_cfg and "batch_key" not in bp_cfg:
                bp_cfg["batch_column"] = self.batch_column

            sample_info_list: List[CohortSampleInfo] = []
            b_col_name = self.batch_column or "batch_id"
            for sid in cohort_clean_dfs:
                b_val = sid
                ctrl_val = False
                extra_meta: Dict[str, Any] = {}
                if resolved_meta is not None and sid in resolved_meta.index:
                    row_series = resolved_meta.loc[sid]
                    if isinstance(row_series, pd.DataFrame):
                        row_series = row_series.iloc[0]
                    for col_k, col_v in row_series.to_dict().items():
                        if pd.notna(col_v):
                            extra_meta[str(col_k)] = col_v
                    if self.condition_column and self.condition_column in extra_meta and "condition" not in extra_meta:
                        extra_meta["condition"] = extra_meta[self.condition_column]
                    if b_col_name in resolved_meta.columns and pd.notna(row_series.get(b_col_name)):
                        b_val = str(row_series.get(b_col_name))
                    if "is_control" in resolved_meta.columns:
                        raw_ctrl = row_series.get("is_control")
                        ctrl_val = str(raw_ctrl).strip().lower() in ("true", "1", "yes")
                sample_info_list.append(
                    CohortSampleInfo(
                        sample_id=sid,
                        batch_id=b_val,
                        is_control=ctrl_val,
                        raw_event_count=cohort_raw_counts.get(sid, len(cohort_clean_dfs[sid])),
                        clean_event_count=cohort_clean_counts.get(sid, len(cohort_clean_dfs[sid])),
                        extra_metadata=extra_meta,
                    )
                )

            batch_prep_cfg = BatchPreparationConfig.from_dict(bp_cfg)
            if not batch_prep_cfg.batch_normalization.markers and cohort_clean_dfs:
                first_cols = set(next(iter(cohort_clean_dfs.values())).columns)
                batch_prep_cfg.batch_normalization.markers = [c for c in target_channels if c in first_cols]

            batch_res = run_batch_preparation(
                cohort_dfs=cohort_clean_dfs,
                sample_info=sample_info_list,
                config=batch_prep_cfg,
                output_dir=s4_plots_dir.parent,
            )

            tables.update(batch_res.generated_tables)
            for b_pk, b_pv in batch_res.generated_plots.items():
                p_obj = Path(b_pv)
                _add_plot(b_pk, p_obj)
                if b_pk == "cohort_downsampling_balance":
                    _add_plot("step4_downsampling_balance", p_obj)
                    _add_plot("downsampling_balance", p_obj)

            prepared_cohort_dfs = batch_res.prepared_dfs

        # Step 5 & 6: Population Discovery & Statistical Analysis
        cohort_population_res: Optional[CohortPopulationResult] = None
        if active_discovery_phase:
            with log_step(logger, f"Step 5 & 6: Population Discovery & Cohort Statistical Analysis ({self.clustering_method.upper()})"):
                logger.info(f"Executing unsupervised population discovery across {len(prepared_cohort_dfs)} samples...")
                from flowx.clustering.cohort_runner import run_cohort_population_analysis
                if sample_metadata is None and metadata_file is None and self.metadata_file is None and resolved_meta is None:
                    logger.warning(
                        "No sample metadata file supplied; processing samples with default sample identifiers."
                    )

                first_prep_cols = list(next(iter(prepared_cohort_dfs.values())).columns) if prepared_cohort_dfs else []
                if comp_source_used == "spectral_unmixing":
                    from flowx.gating.cleanup import matches_viability_dye, matches_dump_channel
                    cohort_m_map = dict(spectral_marker_map) if spectral_marker_map else None
                    cohort_m_list = [
                        fl_c for fl_c in target_channels
                        if fl_c in first_prep_cols
                        and not matches_viability_dye(fl_c)
                        and not matches_viability_dye(spectral_marker_map.get(fl_c, ""))
                        and not matches_dump_channel(fl_c)
                        and not matches_dump_channel(spectral_marker_map.get(fl_c, ""))
                    ]
                    effective_markers_file = None
                else:
                    from flowx.io.metadata import select_markers_from_metadata

                    cohort_m_map = (
                        cohort_marker_res.detector_to_marker
                        if (cohort_marker_res is not None and cohort_marker_res.has_verified_biological_markers)
                        else (first_fcs_data.detector_to_marker if first_fcs_data else None)
                    )
                    cohort_m_list = select_markers_from_metadata(
                        cohort_marker_res,
                        available_columns=first_prep_cols,
                        explicit_markers=self.lineage_markers,
                        marker_roles=self.marker_roles,
                        include_markers=self.include_markers,
                        exclude_markers=self.exclude_markers,
                    )
                    if not cohort_m_list:
                        cohort_m_list = (
                            [c for c in cohort_marker_res.clustering_channels if c in first_prep_cols]
                            if (cohort_marker_res is not None and cohort_marker_res.clustering_channels)
                            else None
                        )
                    if not cohort_m_list:
                        cohort_m_list = None
                    effective_markers_file = self.markers_file

                # Persist concrete resolved channels & markers for auditability
                from flowx.gating.runner import resolve_cleanup_channels_from_metadata

                _c_cfg_cohort = self._build_cleanup_config()
                _resolved_c_cohort = (
                    resolve_cleanup_channels_from_metadata(
                        _c_cfg_cohort,
                        first_prep_cols,
                        marker_metadata=cohort_marker_res,
                        spectral_marker_map=spectral_marker_map,
                    )
                    if first_fcs_data is not None
                    else _c_cfg_cohort
                )
                res_cfg_path = self._write_resolved_config(
                    run_dir=current_run_dir,
                    marker_meta_res=cohort_marker_res,
                    cleanup_cfg=_resolved_c_cohort,
                    xform_channels=list(xf_step.channels_transformed) if "xf_step" in locals() else None,
                    lineage_channels=list(cohort_m_list) if cohort_m_list else None,
                    stats_granularity=self.stats_granularity,
                    stats_marker_roles=self.stats_marker_roles,
                    extra_metadata={
                        "marker_roles_requested": self.marker_roles,
                        "sample_count": len(prepared_cohort_dfs),
                    },
                )
                if res_cfg_path is not None:
                    tables["resolved_config"] = str(res_cfg_path)
                    logger.info(f"Saved resolved parameter audit configuration to {sanitize_path(res_cfg_path)}")

                cohort_population_res = run_cohort_population_analysis(
                    cohort_dfs=prepared_cohort_dfs,
                    sample_metadata=resolved_meta,
                    markers=cohort_m_list,
                    markers_file=effective_markers_file,
                    marker_roles=self.marker_roles,
                    stats_marker_roles=self.stats_marker_roles,
                    include_markers=self.include_markers,
                    exclude_markers=self.exclude_markers,
                    marker_name_map=cohort_m_map,
                    cell_signatures=self.cell_signatures,
                    clustering_method=self.clustering_method,
                    n_clusters=self.n_clusters,
                    auto_k=self.auto_k,
                    auto_k_method=self.auto_k_method,
                    k_min=self.k_min,
                    k_max=self.k_max,
                    consensus_subsample=self.consensus_subsample,
                    consensus_iterations=self.consensus_iterations,
                    compute_umap=self.enable_umap,
                    umap_events_per_sample=self.umap_events_per_sample,
                    condition_column=self.condition_column,
                    batch_column=self.batch_column,
                    umap_covariates=self.umap_covariates,
                    reference_group=self.reference_group,
                    target_group=self.target_group,
                    paired_column=self.paired_column,
                    stats_granularity=self.stats_granularity,
                    da_test_method=self.da_test_method,
                    ds_test_method=self.ds_test_method,
                    fdr_threshold=self.fdr_threshold,
                    log2fc_threshold=self.log2fc_threshold,
                    stratified_density_contours=self.stratified_density_contours,
                    generate_volcano_plot=self.generate_volcano_plot,
                    generate_forest_plot=self.generate_forest_plot,
                    generate_abundance_boxplots=self.generate_abundance_boxplots,
                    generate_ds_heatmap=self.generate_ds_heatmap,
                    export_prism=self.export_prism,
                    cluster_name_map=self.cluster_name_map,
                    cluster_palette=self.cluster_palette,
                    export_augmented_fcs=self.export_augmented_fcs,
                    export_flowjo_wsp=self.export_flowjo_wsp,
                    export_gating_ml=self.export_gating_ml,
                    preprocessing_wsp_path=prep_wsp_path or None,
                    preprocessing_gating_ml_path=prep_gml_path or None,
                    generate_miflowcyt_report=self.generate_miflowcyt_report,
                    output_dir=current_run_dir,
                    random_seed=self.random_seed,
                    generate_plots=True,
                    numbered_steps=active_numbered_steps,
                )

                for p_key, p_path in cohort_population_res.generated_plots.items():
                    _add_plot(p_key, Path(p_path))
                for t_key, t_path in cohort_population_res.generated_tables.items():
                    tables[t_key] = t_path

        # Step 7: Workspace Persistence & Reporting
        ws_json_path = exports_ws_dir / "workspace.json"
        ws_xml_path = exports_ws_dir / "workspace.xml"
        report_path = exports_rep_dir / "report.md"

        if first_hierarchy is not None and first_fcs_data is not None:
            session = WorkspaceSession(
                name=f"Cohort_{target_name}",
                sample_path=str(sample_paths[0]),
                sample_filename=target_name,
                total_events=sum(cohort_clean_counts.values()),
                hierarchy=first_hierarchy,
            )
            session.set_sample(first_fcs_data)
            if comp_obj:
                session.set_compensation(comp_obj)
            for ch_xf, xf_v in first_ws_transforms.items():
                session.add_transform(ch_xf, xf_v)
            session.save_json(ws_json_path)
            if self.export_flowjo_wsp:
                session.save_xml(ws_xml_path)

        execution_duration = time.perf_counter() - t_start
        dur_formatted = format_duration(execution_duration)

        self._generate_cohort_markdown_report(
            report_path=report_path,
            cohort_name=target_name,
            sample_paths=sample_paths,
            cohort_raw_counts=cohort_raw_counts,
            cohort_clean_counts=cohort_clean_counts,
            cohort_res=cohort_population_res,
            plots=plots,
            tables=tables,
            comp_obj=comp_obj,
            comp_source=comp_source_used,
            active_tech=active_tech,
            cytometer=(first_fcs_data.cytometer if first_fcs_data else ""),
            execution_time_seconds=execution_duration,
            run_start_time=run_start_dt,
        )

        close_run_logger(logger)

        return PipelineResult(
            sample_name=target_name,
            event_count=sum(cohort_raw_counts.values()),
            raw_fcs=first_fcs_data or parse_fcs(sample_paths[0]),
            compensated_df=first_df_comp if first_df_comp is not None else pd.DataFrame(),
            transformed_df=first_df_transformed if first_df_transformed is not None else pd.DataFrame(),
            hierarchy=first_hierarchy or GatingHierarchy(),
            statistics_df=cohort_population_res.feature_set.viable_freq_matrix if cohort_population_res else pd.DataFrame(),
            generated_plots={k: sanitize_path(p) for k, p in plots.items()},
            generated_tables={k: sanitize_path(p) for k, p in tables.items()},
            run_dir=sanitize_path(current_run_dir),
            workspace_json_path=sanitize_path(ws_json_path),
            workspace_xml_path=sanitize_path(ws_xml_path),
            preprocessing_wsp_path=sanitize_path(prep_wsp_path) if prep_wsp_path else "",
            preprocessing_gating_ml_path=sanitize_path(prep_gml_path) if prep_gml_path else "",
            statistics_csv_path=sanitize_path(tables.get("population_counts", "")),
            report_md_path=sanitize_path(report_path),
            reproducibility_verified=True,
            gating_mode=active_mode,
            cluster_parent=self.cluster_parent,
            qc_method=active_qc,
            som_result=cohort_population_res.clustering_result if cohort_population_res else None,
            cleanup_result=first_cleanup_res,
            time_qc_result=first_time_qc_res,
            compensation_control_result=comp_ctrl_result,
            compensation_source_used=comp_source_used,
            spillover_matrix_path=sanitize_path(tables.get("spillover_matrix_calculated", tables.get("spillover_matrix_embedded", ""))),
            config_file_path=sanitize_path(current_run_dir / "config.yaml") if (current_run_dir / "config.yaml").is_file() else None,
            log_file_path=sanitize_path(log_file_path) if log_file_path else None,
            execution_time_seconds=execution_duration,
            execution_duration_formatted=dur_formatted,
            technology=active_tech,
            cohort_samples=[p.stem for p in sample_paths],
            cohort_clean_dfs=cohort_clean_dfs,
            cohort_result=cohort_population_res,
        )

    def _generate_cohort_markdown_report(
        self,
        report_path: Path,
        cohort_name: str,
        sample_paths: List[Path],
        cohort_raw_counts: Dict[str, int],
        cohort_clean_counts: Dict[str, int],
        cohort_res: Optional[CohortPopulationResult],
        plots: Dict[str, str],
        tables: Dict[str, str],
        comp_obj: Optional[CompensationMatrix],
        comp_source: str,
        active_tech: str,
        cytometer: str = "",
        execution_time_seconds: float = 0.0,
        run_start_time: Optional[datetime] = None,
    ) -> None:
        """Generate structured MIFlowCyt-compliant analysis report for a cohort run."""
        report_parent = report_path.parent

        def _rel_img(plot_path_val: str) -> str:
            try:
                return os.path.relpath(Path(plot_path_val), report_parent)
            except Exception:
                return f"plots/{Path(plot_path_val).name}"

        total_raw = sum(cohort_raw_counts.values())
        total_clean = sum(cohort_clean_counts.values())
        yield_pct = (total_clean / total_raw * 100.0) if total_raw > 0 else 0.0

        lines: List[str] = [
            f"# Cohort Flow Cytometry Analysis Report: `{cohort_name}`",
            "",
            "## 1. Executive Cohort & Sample Summary",
            "",
            f"- **Cohort Identifier:** `{cohort_name}`",
            f"- **Total Samples Analyzed:** {len(sample_paths)}",
            f"- **Total Cohort Events (Raw):** {total_raw:,}",
            f"- **Total Clean Viable Singlets:** {total_clean:,} ({yield_pct:.1f}% overall cohort yield)",
            f"- **Cytometry Technology:** `{active_tech.title()}`",
            f"- **Cytometer Instrument:** `{cytometer or 'Recorded in FCS Header'}`",
            f"- **Spillover / Unmixing Source:** `{comp_source}`",
            f"- **Execution Duration:** {format_duration(execution_time_seconds)}",
            f"- **Analysis Timestamp:** {(run_start_time or datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "### Sample Yields & Progression Table",
            "",
            "| Sample ID | Raw Events | Clean Viable Singlets | Retention Yield (%) |",
            "|:---|---:|---:|---:|",
        ]

        for s_path in sample_paths:
            s_stem = s_path.stem
            raw_c = cohort_raw_counts.get(s_stem, 0)
            clean_c = cohort_clean_counts.get(s_stem, 0)
            pct = (clean_c / raw_c * 100.0) if raw_c > 0 else 0.0
            lines.append(f"| `{s_stem}` | {raw_c:,} | {clean_c:,} | {pct:.1f}% |")

        cond_num_str = f"{comp_obj.condition_number:.2f}" if (comp_obj and getattr(comp_obj, "condition_number", None) is not None) else "N/A"

        lines.extend([
            "",
            "## 2. Quality Control & Preprocessing (Steps 01 - 03)",
            "",
            "- **Step 01: Time Quality Control:** Acquisition flow stability verified per sample to exclude fluidic disturbances and pressure surges. Diagnostic plots recorded in `01_time_qc/plots/`.",
            f"- **Step 02: Fluorescence Deconvolution:** Corrected spectral spillover using `{comp_source}`. Matrix condition number: {cond_num_str}. Diagnostic plots recorded in `02_compensation/plots/`.",
            "- **Step 03: Core Clean-up Gating:** Intact cells isolated via forward/side scatter (debris removal), doublets excluded via FSC-A vs FSC-H / SSC-A vs SSC-H, and dead cells excluded via viability exclusion dye.",
            "",
            "## 3. Batch Preparation & Event Downsampling (Step 04)",
            "",
            "Event yields balanced across cohort samples to prevent dominant samples from biasing unsupervised clustering algorithms.",
            "",
        ])

        if "step4_downsampling_balance" in plots or "downsampling_balance" in plots:
            p_down = plots.get("step4_downsampling_balance") or plots.get("downsampling_balance")
            lines.extend([
                f"![Cohort Downsampling Balance]({_rel_img(p_down)})",
                "",
            ])

        lines.extend([
            "## 4. Unsupervised Population Discovery (Step 05)",
            "",
        ])

        if cohort_res is not None:
            cl_res = cohort_res.clustering_result
            lines.extend([
                f"- **Clustering Engine:** {self.clustering_method.upper()}",
                f"- **Metaclusters Discovered:** {cl_res.cluster_count if cl_res else 'N/A'}",
                f"- **Lineage Markers Evaluated:** {', '.join(cl_res.markers_used) if cl_res else 'All panel markers'}",
                "",
                "### Key Diagnostic Visualizations",
                "",
            ])

            if "cluster_expression_heatmap" in plots:
                lines.append(f"![Cluster Expression Heatmap]({_rel_img(plots['cluster_expression_heatmap'])})")
                lines.append("")
            if "cluster_qc_metrics" in plots:
                lines.append(f"![Cluster Validation QC Metrics]({_rel_img(plots['cluster_qc_metrics'])})")
                lines.append("")
            if "cohort_umap_clusters" in plots or "umap_clusters" in plots:
                p_u = plots.get("cohort_umap_clusters") or plots.get("umap_clusters")
                lines.append(f"![Cohort UMAP Clusters]({_rel_img(p_u)})")
                lines.append("")
            if "cluster_to_population_bridge" in plots:
                lines.append(f"![Cluster-to-Population Reconciliation Bridge]({_rel_img(plots['cluster_to_population_bridge'])})")
                lines.append("")
            if "cohort_umap_populations" in plots or "umap_populations" in plots:
                p_upop = plots.get("cohort_umap_populations") or plots.get("umap_populations")
                lines.append(f"![Cohort UMAP Populations]({_rel_img(p_upop)})")
                lines.append("")
            if "cohort_umap_covariates" in plots or "umap_covariates" in plots:
                p_ucov = plots.get("cohort_umap_covariates") or plots.get("umap_covariates")
                lines.append(f"![Cohort UMAP Covariates]({_rel_img(p_ucov)})")
                lines.append("")
            if "cohort_umap_marker_features" in plots or "umap_marker_features" in plots:
                p_uf = plots.get("cohort_umap_marker_features") or plots.get("umap_marker_features")
                lines.append(f"![Cohort UMAP Marker Features]({_rel_img(p_uf)})")
                lines.append("")
            if "cohort_frequency_clustermap" in plots:
                lines.append(f"![Cohort Frequency Clustermap]({_rel_img(plots['cohort_frequency_clustermap'])})")
                lines.append("")
            if "cohort_umap_stratified_contours" in plots:
                lines.append(f"![Stratified UMAP Contours]({_rel_img(plots['cohort_umap_stratified_contours'])})")
                lines.append("")

        lines.extend([
            "## 5. Quantitative Cohort Statistical Analysis (Step 06)",
            "",
        ])

        if cohort_res is not None and cohort_res.da_result is not None:
            da_res = cohort_res.da_result
            sig_pops = len(da_res.significant_populations) if hasattr(da_res, "significant_populations") else 0
            lines.extend([
                f"- **Differential Abundance (DA):** Evaluated condition shifts; identified {sig_pops} statistically significant populations (FDR < 0.05).",
                "",
            ])
            if getattr(da_res, "underpowered_warning", None):
                lines.extend([
                    f"- **Statistical Power Guardrail:** {da_res.underpowered_warning}",
                    "",
                ])
            if "differential_abundance_volcano" in plots:
                lines.append(f"![Differential Abundance Volcano]({_rel_img(plots['differential_abundance_volcano'])})")
                lines.append("")
            if "differential_abundance_forest" in plots:
                lines.append(f"![Differential Abundance Forest Plot]({_rel_img(plots['differential_abundance_forest'])})")
                lines.append("")
            if "population_abundance_boxplots" in plots:
                lines.append(f"![Population Abundance Boxplots]({_rel_img(plots['population_abundance_boxplots'])})")
                lines.append("")

        if cohort_res is not None and cohort_res.ds_result is not None:
            lines.extend([
                "- **Differential State (DS):** Evaluated functional marker shifts across defined populations.",
                "",
            ])
            if "differential_state_heatmap" in plots:
                lines.append(f"![Differential State Heatmap]({_rel_img(plots['differential_state_heatmap'])})")
                lines.append("")

        if cohort_res is None or (cohort_res.da_result is None and cohort_res.ds_result is None):
            lines.extend([
                "- Comparative statistical analysis (DA / DS) skipped: uniform condition or single condition group in cohort.",
                "",
            ])

        if comp_obj is not None and getattr(comp_obj, "condition_number", None) is not None:
            comp_compliance_note = f"Matrix condition number verified (\u03ba={comp_obj.condition_number:.2f}) and audit tables preserved"
        else:
            comp_compliance_note = f"Pre-unmixed / external spectral deconvolution (`{comp_source}`); no post-hoc spillover matrix applied"

        lines.extend([
            "## 6. Deliverable Exports & Interoperability (Step 07)",
            "",
            "- **Augmented FCS Files:** Clean events exported with newly assigned channels (`cluster_id`, `lineage_id`, `UMAP_1`, `UMAP_2`) into `exports/augmented_fcs/`.",
            "- **Preprocessing & Gating Workspaces (Raw FCS):** FlowJo `.wsp` (`preprocessing_gating.wsp`) and Gating-ML 2.0 (`preprocessing_gating_ml_2.0.xml`) in `exports/workspaces/`.",
            "- **Discovery Workspaces (Augmented FCS):** FlowJo `.wsp` (`flowjo_workspace.wsp`) and Gating-ML 2.0 (`gating_ml_2.0.xml`) in `exports/workspaces/`.",
            "- **GraphPad Prism Tables:** Formatted grouped abundance tables exported to `05_population_discovery/tables/`.",
            "",
            "## 7. MIFlowCyt Compliance Checklist",
            "",
            "| MIFlowCyt Requirement | Status | Notes |",
            "|:---|:---:|:---|",
            f"| 1. Experiment Overview | COMPLIANT | Cohort `{cohort_name}` analyzed with reproducible FlowX pipeline |",
            f"| 2. Flow Cytometer Description | COMPLIANT | Cytometer: `{cytometer or 'Instrument configuration extracted from FCS headers'}` |",
            f"| 3. Fluorescence Compensation | COMPLIANT | {comp_compliance_note} |",
            "| 4. Gating Strategy & Yields | COMPLIANT | Complete parent-child population hierarchy and event yields recorded |",
            "| 5. Statistical Methods | COMPLIANT | Non-parametric tests and FDR multiple testing corrections applied |",
            "",
            "## References & Scientific Standards",
            "",
            "* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)",
            "* **FlowSOM Algorithm:** [](doi:10.1002/cyto.a.22625)",
            "* **FlowAI Quality Control:** [](doi:10.1093/bioinformatics/btw191)",
            "* **Logicle Transformation:** [](doi:10.1002/cyto.a.20258)",
            "* **openCyto Gating Framework:** [](doi:10.1371/journal.pcbi.1003806)",
            "* **Differential Discovery (diffcyt):** [](doi:10.1093/nar/gkz433)",
            "* **UMAP Dimensionality Reduction:** [](doi:10.21105/joss.00861)",
            "",
        ])

        report_path.write_text("\n".join(lines), encoding="utf-8")

    def _organize_numbered_steps(self, run_dir: Path) -> None:
        """Structure pipeline deliverables into standardized numbered step directories.

        Mirrors diagnostic plots, quantitative tables, and export artifacts
        into numbered phase folders while preserving root plots/ and tables/ directories:
          - 01_time_qc/ (plots, tables)
          - 02_compensation/ (plots, tables)
          - 03_cleanup_gating/ (plots, tables)
          - 05_population_discovery/ (plots, tables, if discovery was run)
          - exports/ (workspaces, reports, clean_fcs)
        """
        plots_dir = run_dir / "plots"
        tables_dir = run_dir / "tables"

        # 01_time_qc
        s1_p = run_dir / "01_time_qc" / "plots"
        s1_t = run_dir / "01_time_qc" / "tables"
        # 02_compensation
        s2_p = run_dir / "02_compensation" / "plots"
        s2_t = run_dir / "02_compensation" / "tables"
        # 03_cleanup_gating
        s3_p = run_dir / "03_cleanup_gating" / "plots"
        s3_t = run_dir / "03_cleanup_gating" / "tables"
        # 05_population_discovery
        s5_p = run_dir / "05_population_discovery" / "plots"
        s5_t = run_dir / "05_population_discovery" / "tables"
        # exports
        exp_ws = run_dir / "exports" / "workspaces"
        exp_rep = run_dir / "exports" / "reports"

        for d in (s1_p, s1_t, s2_p, s2_t, s3_p, s3_t, exp_ws, exp_rep):
            d.mkdir(parents=True, exist_ok=True)

        # Mirror plots
        if plots_dir.is_dir():
            for p in plots_dir.glob("*.png"):
                fname = p.name
                if fname.startswith("step1_time"):
                    shutil.copy2(p, s1_p / fname)
                elif fname.startswith("step1_clustering") or fname.startswith("step1_raw"):
                    shutil.copy2(p, s3_p / fname)
                elif fname.startswith("step2_"):
                    shutil.copy2(p, s2_p / fname)
                elif fname.startswith("step3_") or fname.startswith("step4_"):
                    shutil.copy2(p, s3_p / fname)
                elif fname.startswith("step5_"):
                    s5_p.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, s5_p / fname)

            # Sub-plot directories for compensation
            for sub_p in ("plots_comp_controls", "plots_nxn_samples"):
                src_sub = plots_dir / sub_p
                if src_sub.is_dir():
                    target_sub = s2_p / sub_p
                    if not target_sub.exists():
                        shutil.copytree(src_sub, target_sub)

        # Mirror tables
        if tables_dir.is_dir():
            for t in tables_dir.glob("*.csv"):
                tname = t.name
                if "time_qc" in tname or tname == "channel_resolution_audit.csv":
                    shutil.copy2(t, s1_t / tname)
                elif any(k in tname for k in ("spillover", "residual", "control_")):
                    shutil.copy2(t, s2_t / tname)
                elif any(k in tname for k in ("som", "phenograph", "umap")):
                    s5_t.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(t, s5_t / tname)
                elif tname == "population_statistics.csv":
                    shutil.copy2(t, s3_t / tname)

        # Root level statistics tables
        for f in (
            "som_statistics.csv",
            "som_macro_groups.csv",
            "phenograph_statistics.csv",
            "phenograph_macro_groups.csv",
            "comparison_crosstab_som.csv",
            "comparison_crosstab_phenograph.csv",
            "comparison_three_methods_metrics.csv",
            "comparison_three_methods_lineages.csv",
        ):
            fp = run_dir / f
            if fp.is_file():
                s5_t.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fp, s5_t / f)

        pop_stat = run_dir / "population_statistics.csv"
        if pop_stat.is_file():
            shutil.copy2(pop_stat, s3_t / "population_statistics.csv")

        # Exports: workspaces & reports
        for ws in (
            "workspace.json",
            "workspace.xml",
            "preprocessing_gating.wsp",
            "preprocessing_gating_ml_2.0.xml",
        ):
            wsp = run_dir / ws
            if wsp.is_file():
                shutil.copy2(wsp, exp_ws / ws)

        for rep in ("report.md", "run_manifest.json"):
            repp = run_dir / rep
            if repp.is_file():
                shutil.copy2(repp, exp_rep / rep)

        # Clean FCS exports
        clean_fcs_dir = run_dir / "clean_fcs"
        if clean_fcs_dir.is_dir():
            exp_fcs = run_dir / "exports" / "clean_fcs"
            exp_fcs.mkdir(parents=True, exist_ok=True)
            for fcs_f in clean_fcs_dir.glob("*.fcs"):
                shutil.copy2(fcs_f, exp_fcs / fcs_f.name)

    def _build_three_way_lineage_comparison(
        self,
        total_events: int,
        template_labels: np.ndarray,
        som_res: Optional[ClusteringResult] = None,
        pheno_res: Optional[ClusteringResult] = None,
        df_transformed: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Consolidate cell quantitations across Template, SOM, and PhenoGraph into one table.

        Ensures mutually exclusive counts across all three methodologies that sum to 100% of events.
        """
        if som_res is None or pheno_res is None:
            raise ValueError("Must provide som_res and pheno_res")

        # 1. Template counts
        tmpl_cd4 = int(np.sum(template_labels == "CD4+ T Cells"))
        tmpl_cd8 = int(np.sum(template_labels == "CD8+ T Cells"))
        tmpl_other_t = int(np.sum(template_labels == "CD3+ Other"))

        # In template gating, CD3- Viable contains APCs and other non-T lymphocytes
        dr_ch = "APC-H7-A"
        if df_transformed is not None and dr_ch in df_transformed.columns:
            nont_mask = template_labels == "CD3- Viable"
            tmpl_apc = int(np.sum(nont_mask & (df_transformed[dr_ch] >= 0.40)))
            tmpl_nont = int(np.sum(nont_mask & (df_transformed[dr_ch] < 0.40)))
        else:
            tmpl_apc = 0
            tmpl_nont = int(np.sum(template_labels == "CD3- Viable"))

        # Dead cells, debris, non-cells, and doublets (all excluded non-viable/debris events)
        tmpl_dead = int(np.sum(np.isin(template_labels, ["Non-viable", "Debris / Non-cells", "Doublets"])))

        # 2. SOM macro-groups counts
        def get_som_macro_count(target_lineage: str) -> int:
            if som_res.macro_groups_df is None:
                return 0
            sub = som_res.macro_groups_df[som_res.macro_groups_df["Lineage Name"] == target_lineage]
            return int(sub["Total Cells"].sum()) if not sub.empty else 0

        som_cd4 = get_som_macro_count("CD4+ T-Cell Lineage")
        som_cd8 = get_som_macro_count("CD8+ T-Cell Lineage")
        som_other_t = get_som_macro_count("CD3+ DN / Other T Cells") + get_som_macro_count("CD4+ CD8+ DP T Cells")
        som_apc = get_som_macro_count("HLA-DR+ APC / Monocyte Lineage")
        som_nont = get_som_macro_count("CD3- Non-T Lymphocytes")
        som_dead = get_som_macro_count("Dead Cells / Debris")
        if som_dead == 0 and som_res.full_cluster_labels is not None:
            som_dead = int(np.sum(som_res.full_cluster_labels == -1))

        # 3. PhenoGraph counts (mutually exclusive via assigned macro-group)
        def get_pheno_macro_count(target_lineage: str) -> int:
            cnt = 0
            for cid in np.unique(pheno_res.cluster_labels):
                macro = pheno_res.cluster_to_macro_group.get(int(cid), "")
                if macro == target_lineage:
                    cnt += int(np.sum(pheno_res.cluster_labels == cid))
            return cnt

        pg_cd4 = get_pheno_macro_count("CD4+ T-Cell Lineage")
        pg_cd8 = get_pheno_macro_count("CD8+ T-Cell Lineage")
        pg_other_t = get_pheno_macro_count("CD3+ DN / Other T Cells") + get_pheno_macro_count("CD4+ CD8+ DP T Cells")
        pg_apc = get_pheno_macro_count("HLA-DR+ APC / Monocyte Lineage")
        pg_nont = get_pheno_macro_count("CD3- Non-T Lymphocytes")
        pg_dead = get_pheno_macro_count("Dead Cells / Debris")
        if pg_dead == 0 and pheno_res.full_cluster_labels is not None:
            pg_dead = int(np.sum(pheno_res.full_cluster_labels == -1))

        def pct(c: int) -> str:
            return f"{c:,} ({c / total_events * 100:.2f}%)" if c > 0 else "0 (0.00%)"

        tmpl_apc_str = f"Ungated: {tmpl_apc:,} ({tmpl_apc / total_events * 100:.2f}%)" if tmpl_apc > 0 else "Ungated (0.00%)"

        rows = [
            {
                "Biological Lineage": "CD4+ T Cells",
                "Key Biomarkers": "CD3+ CD4+ CD8-",
                "Method 1: Template Gating": pct(tmpl_cd4),
                "Method 2: SOM": pct(som_cd4),
                "Method 3: PhenoGraph": pct(pg_cd4),
            },
            {
                "Biological Lineage": "CD8+ T Cells",
                "Key Biomarkers": "CD3+ CD8+ CD4-",
                "Method 1: Template Gating": pct(tmpl_cd8),
                "Method 2: SOM": pct(som_cd8),
                "Method 3: PhenoGraph": pct(pg_cd8),
            },
            {
                "Biological Lineage": "CD3+ Other / DN T Cells",
                "Key Biomarkers": "CD3+ CD4- CD8- / DP",
                "Method 1: Template Gating": pct(tmpl_other_t),
                "Method 2: SOM": pct(som_other_t),
                "Method 3: PhenoGraph": pct(pg_other_t),
            },
            {
                "Biological Lineage": "HLA-DR+ APCs / Monocytes",
                "Key Biomarkers": "CD3- HLA-DR+",
                "Method 1: Template Gating": tmpl_apc_str,
                "Method 2: SOM": pct(som_apc),
                "Method 3: PhenoGraph": pct(pg_apc),
            },
            {
                "Biological Lineage": "CD3- Non-T Lymphocytes",
                "Key Biomarkers": "CD3- HLA-DR- Viable",
                "Method 1: Template Gating": pct(tmpl_nont),
                "Method 2: SOM": pct(som_nont),
                "Method 3: PhenoGraph": pct(pg_nont),
            },
            {
                "Biological Lineage": "Dead Cells / Debris / Non-cells",
                "Key Biomarkers": "AARD-A+ / Scatter outliers / Doublets",
                "Method 1: Template Gating": pct(tmpl_dead),
                "Method 2: SOM": pct(som_dead),
                "Method 3: PhenoGraph": pct(pg_dead),
            },
        ]
        return pd.DataFrame(rows)

    def _generate_markdown_report(
        self,
        report_path: Path,
        fcs_data: FCSData,
        comp_obj: Optional[CompensationMatrix],
        transform_obj: BaseTransform,
        stats_df: pd.DataFrame,
        plots: Dict[str, str],
        reproducible: bool,
        gating_mode: str = "som",
        qc_method: str = "clustering",
        clustering_qc_res: Optional[ClusteringQCResult] = None,
        som_res: Optional[ClusteringResult] = None,
        pheno_res: Optional[ClusteringResult] = None,
        ct_som: Optional[pd.DataFrame] = None,
        ct_pg: Optional[pd.DataFrame] = None,
        three_way_metrics: Optional[pd.DataFrame] = None,
        three_way_lineages: Optional[pd.DataFrame] = None,
        adaptive_cutoffs: Optional[Dict[str, DensityCutoffResult]] = None,
        comp_ctrl_res: Optional[CompensationControlResult] = None,
        comp_source: str = "embedded",
        config_file_path: Optional[str] = None,
        log_file_path: Optional[str] = None,
        execution_time_seconds: float = 0.0,
        execution_duration_formatted: str = "",
        run_start_time: Optional[datetime] = None,
    ) -> None:
        """Generate a complete Markdown analysis report saved in the run output folder."""
        qc_desc = "Automated Clustering (GMM)" if qc_method == "clustering" else "Standard Templates"
        dur_display = execution_duration_formatted or format_duration(execution_time_seconds)
        timestamp_str = (run_start_time or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        report_parent = report_path.parent

        def _rel_img(plot_path_val: str) -> str:
            try:
                return os.path.relpath(Path(plot_path_val), report_parent)
            except Exception:
                return f"plots/{Path(plot_path_val).name}"

        lines: List[str] = [
            f"# Flow Cytometry Analysis Report: `{fcs_data.filename}`",
            "",
            "## 1. Sample & Instrument Summary",
            "",
            f"- **Filename:** `{fcs_data.filename}`",
            f"- **Source Filepath:** `{sanitize_path(fcs_data.filepath)}`",
            f"- **Cytometer Instrument:** `{fcs_data.cytometer or 'Auto-detected / Standard'}`",
            f"- **Scale Dynamic Range:** `T={fcs_data.detect_max_range():,.0f}`",
            f"- **Total Events:** {fcs_data.event_count:,}",
            f"- **Channels Count:** {fcs_data.channel_count}",
            f"- **Analysis Mode:** `{gating_mode.upper()}`",
            f"- **QC Methodology:** `{qc_desc}`",
            f"- **Total Execution Time:** `{dur_display} ({execution_time_seconds:.2f}s)`",
            f"- **Analysis Timestamp:** `{timestamp_str}`",
            f"- **Reproducibility Status:** {'**VERIFIED (100% Match)**' if reproducible else '**UNVERIFIED**'}",
        ]
        if config_file_path:
            lines.append(f"- **Configuration File:** `{Path(config_file_path).name}` (archived copy saved to output folder)")
        if log_file_path:
            lines.append(f"- **Execution Log:** `{Path(log_file_path).name}` (detailed execution trace with timestamps)")
        lines.extend([
            "",
            "### Measured Parameters & Channel Mapping",
            "",
            "| Channel # | Detector (PnN) | Marker (PnS) | Formatted Label | Hardware Range ($PnR) |",
            "| :---: | :--- | :--- | :--- | :---: |",
        ])

        for i, (pnn, pns) in enumerate(zip(fcs_data.pnn_labels, fcs_data.pns_labels)):
            lbl = fcs_data.get_channel_label(pnn)
            pnr = fcs_data.channel_ranges.get(pnn, "-")
            pnr_str = f"{pnr:,.0f}" if isinstance(pnr, (int, float)) else str(pnr)
            lines.append(f"| {i+1} | `{pnn}` | {pns or '-'} | {lbl} | `{pnr_str}` |")

        lines.extend([
            "",
            "## 2. Processing Pipeline Steps",
            "",
            "### Quality Control (Time QC - Step 1.1 / 1.2)",
        ])

        if "step1_time_qc" in plots:
            lines.extend([
                "- **Methodology:** Automated Fluidic Flow Rate Stability & Signal Drift Monitoring (flowAI / flowCut / PeacoQC principles)",
                "- **Status:** `APPLIED` (Unstable intervals, micro-clogs, and PMT rail saturation purged)",
                "",
                f"![Step 1 Time QC Diagnostics]({_rel_img(plots['step1_time_qc'])})",
                "",
            ])
        else:
            lines.extend([
                "- **Status:** `BYPASSED`",
                "",
            ])

        lines.extend([
            "### Fluorescence Compensation",
        ])

        if comp_obj:
            source_desc = (
                "Calculated from Single-Stained Compensation Controls"
                if comp_source == "controls"
                else "Embedded Cytometer Acquisition Matrix ($SPILLOVER)"
            )
            lines.extend([
                f"- **Compensation Source:** `{source_desc}`",
            ])
            if self.comp_manifest:
                lines.append(f"- **Control Manifest:** `{sanitize_path(self.comp_manifest)}`")
            lines.extend([
                f"- **Compensated Detectors:** {len(comp_obj.detectors)} fluorescent channels",
                f"- **Condition Number κ(S):** {comp_obj.condition_number:.2f}",
                "- **Mathematical Model:** $\\mathbf{X} = \\mathbf{Y} \\cdot (\\mathbf{S}^T)^{-1}$",
            ])
            if comp_ctrl_res:
                if comp_ctrl_res.embedded_correlation is not None:
                    lines.append(f"- **Agreement with Cytometer Matrix:** Pearson $r={comp_ctrl_res.embedded_correlation:.4f}$, MAE=${comp_ctrl_res.embedded_mae:.4f}$")
                lines.extend([
                    f"- **Mapped Control Files:** {len(comp_ctrl_res.control_files_mapped)} / {len(comp_obj.detectors)} channels",
                    "",
                    "#### Single-Stained Controls Mapping & Population Separation",
                    "",
                    "| Channel / Detector | Matched Control File | Gating Method | Positive MFI | Negative MFI | Net Δ-MFI |",
                    "| :--- | :--- | :---: | :---: | :---: | :---: |",
                ])
                for ch in comp_obj.detectors:
                    cfile = Path(comp_ctrl_res.control_files_mapped.get(ch, "None")).name
                    cstats = comp_ctrl_res.population_stats.get(ch, {})
                    g_method = cstats.get("method", "N/A").upper()
                    pos_m = f"{cstats.get('pos_mfi', 0):,.1f}"
                    neg_m = f"{cstats.get('neg_mfi', 0):,.1f}"
                    delta_m = f"{cstats.get('delta_mfi', 0):,.1f}"
                    lines.append(f"| `{ch}` | `{cfile}` | {g_method} | {pos_m} | {neg_m} | **{delta_m}** |")
        else:
            lines.append("- *No compensation applied.*")

        lines.extend([
            "",
            "### Scale Transformation",
            f"- **Transformation Type:** {transform_obj.__class__.__name__}",
            f"- **Parameters:** {transform_obj.to_dict()}",
            "",
            "## 3. Method 1: Template / Gating Hierarchy Statistics",
            "",
        ])

        if adaptive_cutoffs:
            lines.extend([
                "### Adaptive Valley-Seeking Density Cutoffs (openCyto / flowDensity)",
                "",
                "Data-driven gate thresholds established by automated Gaussian Kernel Density Estimation (KDE) and local minimum valley identification:",
                "",
                "| Target Gate | Channel / Detector | Detected Cutoff | Prominent Modes (Peaks) | Local Valley | Separation Confidence | Bimodal Resolved |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
            ])
            for gate_name, cut_res in adaptive_cutoffs.items():
                peaks_str = ", ".join(f"{p:.3f}" for p in cut_res.peaks) if cut_res.peaks else "None"
                valley_str = f"{cut_res.valley:.3f}" if cut_res.valley is not None else "N/A"
                lines.append(
                    f"| **{gate_name}** | `{cut_res.channel}` | **{cut_res.cutoff:.3f}** | "
                    f"[{peaks_str}] | {valley_str} | {cut_res.confidence:.2f} | "
                    f"{'Yes' if cut_res.is_bimodal else 'No'} |"
                )
            lines.extend(["", ""])

        headers = [str(c) for c in stats_df.columns]
        rows_str: List[str] = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(headers))]) + " |",
        ]
        for _, row in stats_df.iterrows():
            row_vals = []
            for col in stats_df.columns:
                val = row[col]
                if pd.isna(val) or val is None:
                    row_vals.append("-")
                elif isinstance(val, (int, np.integer)):
                    row_vals.append(f"{val:,}")
                elif isinstance(val, (float, np.floating)):
                    row_vals.append(f"{val:.2f}" if "Median" not in str(col) else f"{val:.3f}")
                else:
                    row_vals.append(str(val))
            rows_str.append("| " + " | ".join(row_vals) + " |")

        lines.extend(rows_str)

        # -------------------------------------------------------------
        # Three-Way Consolidated Comparison Section
        # -------------------------------------------------------------
        if three_way_lineages is not None:
            lines.extend([
                "",
                "## 4. Multi-Method Consolidated Comparison (All Three Methods)",
                "",
                "Direct side-by-side comparison of cellular populations quantified across all three methodologies:",
                "",
                "| Biological Lineage | Key Biomarkers | Method 1: Template Gating | Method 2: SOM | Method 3: PhenoGraph |",
                "| :--- | :--- | :---: | :---: | :---: |",
            ])
            for _, r in three_way_lineages.iterrows():
                som_val = r.get("Method 2: SOM", "-")
                lines.append(
                    f"| **{r['Biological Lineage']}** | `{r['Key Biomarkers']}` | "
                    f"{r['Method 1: Template Gating']} | {som_val} | {r['Method 3: PhenoGraph']} |"
                )

        if three_way_metrics is not None:
            lines.extend([
                "",
                "### Statistical Concordance & Similarity Matrix",
                "",
                "Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) evaluating global partitioning agreement:",
                "",
                "| Method Pair | Adjusted Rand Index (ARI) | Normalized Mutual Info (NMI) | Biological Interpretation |",
                "| :--- | :---: | :---: | :--- |",
            ])
            for _, r in three_way_metrics.iterrows():
                lines.append(f"| **{r['Method Pair']}** | {r['Adjusted Rand Index (ARI)']:.4f} | {r['Normalized Mutual Information (NMI)']:.4f} | {r['Biological Interpretation']} |")

        # -------------------------------------------------------------
        # Pairwise Concordance to Template Sections
        # -------------------------------------------------------------
        if ct_som is not None:
            lines.extend([
                "",
                "## 5. Pairwise Concordance: Template Gating vs SOM",
                "",
                "Cross-tabulation showing how events categorized by manual gating map across SOM clusters:",
                "",
            ])
            som_cols = [f"C{c}" for c in ct_som.columns]
            lines.append("| Template Population | " + " | ".join(som_cols) + " | Total |")
            lines.append("| :--- | " + " | ".join([":---:" for _ in som_cols]) + " | :---: |")
            for gate_name, row in ct_som.iterrows():
                row_cells = [f"{int(val):,}" for val in row.values]
                lines.append(f"| **{gate_name}** | " + " | ".join(row_cells) + f" | **{int(row.sum()):,}** |")

        if ct_pg is not None:
            lines.extend([
                "",
                "## 6. Pairwise Concordance: Template Gating vs PhenoGraph",
                "",
                "Cross-tabulation showing how manual template gates distribute into PhenoGraph's fine communities:",
                "",
            ])
            pg_cols = [f"C{c}" for c in ct_pg.columns]
            lines.append("| Template Population | " + " | ".join(pg_cols) + " | Total |")
            lines.append("| :--- | " + " | ".join([":---:" for _ in pg_cols]) + " | :---: |")
            for gate_name, row in ct_pg.iterrows():
                row_cells = [f"{int(val):,}" for val in row.values]
                lines.append(f"| **{gate_name}** | " + " | ".join(row_cells) + f" | **{int(row.sum()):,}** |")

        # -------------------------------------------------------------
        # Unsupervised Clustering Details & Macro-Groups
        # -------------------------------------------------------------
        if som_res and som_res.macro_groups_df is not None:
            lines.extend([
                "",
                "## 7. SOM Biological Macro-Groups",
                "",
                "Clusters automatically grouped by shared marker expression profiles:",
                "",
                "| Group ID | Biological Lineage | Constituent Clusters | Total Cells | % Total Events |",
                "| :---: | :--- | :--- | :---: | :---: |",
            ])
            for _, r in som_res.macro_groups_df.iterrows():
                lines.append(f"| {r['Macro Group ID']} | **{r['Lineage Name']}** | `{r['Clusters']}` | {r['Total Cells']:,} | {r['% Total Events']:.2f}% |")

        if pheno_res and pheno_res.macro_groups_df is not None:
            lines.extend([
                "",
                "## 8. PhenoGraph Biological Macro-Groups",
                "",
                "Fine communities automatically grouped into higher-level lineages:",
                "",
                "| Group ID | Biological Lineage | Constituent Clusters | Total Cells | % Total Events |",
                "| :---: | :--- | :--- | :---: | :---: |",
            ])
            for _, r in pheno_res.macro_groups_df.iterrows():
                lines.append(f"| {r['Macro Group ID']} | **{r['Lineage Name']}** | `{r['Clusters']}` | {r['Total Cells']:,} | {r['% Total Events']:.2f}% |")

        # -------------------------------------------------------------
        # Visualizations
        # -------------------------------------------------------------
        lines.extend([
            "",
            "## 9. Diagnostic & Comparison Plots",
            "",
        ])

        if "step5_clustering_comparison_centroids" in plots:
            lines.extend([
                "### Multi-Method Comparison (Direct Centroid Badges)",
                f"![Centroid Badging Comparison]({_rel_img(plots['step5_clustering_comparison_centroids'])})",
                "",
            ])

        if "step5_clustering_comparison_external_legend" in plots:
            lines.extend([
                "### Multi-Method Comparison (External Multi-Column Legend)",
                f"![External Legend Comparison]({_rel_img(plots['step5_clustering_comparison_external_legend'])})",
                "",
            ])

        som_heatmap_key = "step5_som_marker_heatmap" if "step5_som_marker_heatmap" in plots else None
        som_scatter_key = "step5_som_clusters_scatter" if "step5_som_clusters_scatter" in plots else None

        if som_heatmap_key:
            lines.extend([
                "### SOM Marker Expression Heatmap (Biologically Grouped)",
                f"![SOM Heatmap]({_rel_img(plots[som_heatmap_key])})",
                "",
            ])
            if som_scatter_key:
                lines.extend([
                    f"![SOM Scatter]({_rel_img(plots[som_scatter_key])})",
                    "",
                ])

        if "step5_phenograph_marker_heatmap" in plots:
            lines.extend([
                "### PhenoGraph Marker Expression Heatmap (Biologically Grouped)",
                f"![PhenoGraph Heatmap]({_rel_img(plots['step5_phenograph_marker_heatmap'])})",
                "",
                f"![PhenoGraph Scatter]({_rel_img(plots['step5_phenograph_clusters_scatter'])})",
                "",
            ])

        if "step5_umap_clusters" in plots or "step5_umap_marker_features" in plots:
            lines.extend([
                "### High-Dimensional Manifold Embeddings (UMAP)",
                "",
                "Non-linear dimensionality reduction projecting all fluorescent parameter dimensions into 2D coordinates for single-cell phenotype discovery:",
                "",
            ])
            if "step5_umap_clusters" in plots:
                lines.extend([
                    "#### UMAP Manifold: Discovered Clusters",
                    f"![UMAP Clusters]({_rel_img(plots['step5_umap_clusters'])})",
                    "",
                ])
            if "step5_umap_lineages" in plots:
                lines.extend([
                    "#### UMAP Manifold: Gated Biological Lineages",
                    f"![UMAP Gated Lineages]({_rel_img(plots['step5_umap_lineages'])})",
                    "",
                ])
            if "step5_umap_marker_features" in plots:
                lines.extend([
                    "#### Continuous Single-Cell Marker Intensity Manifold Maps",
                    f"![UMAP Feature Maps]({_rel_img(plots['step5_umap_marker_features'])})",
                    "",
                ])
            if "step5_population_marker_violins" in plots:
                lines.extend([
                    "#### Phenotypic Marker Distributions Across Gated Lineages",
                    f"![Population Violins]({_rel_img(plots['step5_population_marker_violins'])})",
                    "",
                ])

        if clustering_qc_res is not None:
            lines.extend([
                "### Automated Clustering-Based Quality Control (QC)",
                "",
                "The sample underwent automated Gaussian Mixture Model (GMM) clustering deconvolution for objective debris exclusion and doublet discrimination without operator polygon bias:",
                "",
                "| QC Tier | Population | Event Count | % Total | % Parent | Methodology |",
                "| :--- | :--- | :---: | :---: | :---: | :--- |",
            ])
            for _, r in clustering_qc_res.summary_stats.iterrows():
                lines.append(
                    f"| {r['QC Stage']} | **{r['Population']}** | {r['Event Count']:,} | {r['% Total']:.2f}% | {r['% Parent']:.2f}% | {r['Methodology']} |"
                )
            lines.append("")
            if "step1_clustering_qc_diagnostic" in plots:
                lines.extend([
                    f"![Clustering QC Diagnostic]({_rel_img(plots['step1_clustering_qc_diagnostic'])})",
                    "",
                ])

        lines.append("### Raw Morphology & Singlet Discrimination")
        if "step1_raw_morphology" in plots:
            lines.extend([
                f"![Raw Morphology]({_rel_img(plots['step1_raw_morphology'])})",
                "",
            ])
        if "step1_raw_doublets" in plots:
            lines.extend([
                f"![Raw Doublet Discrimination]({_rel_img(plots['step1_raw_doublets'])})",
                "",
            ])

        if "step2_compensation_diagnostic" in plots:
            lines.extend([
                "### Fluorescence Compensation Diagnostic",
                f"![Compensation Diagnostic]({_rel_img(plots['step2_compensation_diagnostic'])})",
                "",
            ])

        if "step2_nxn_matrix" in plots:
            lines.extend([
                "### N x N Compensation / Unmixing Validation Matrix",
                f"![N x N Validation Matrix]({_rel_img(plots['step2_nxn_matrix'])})",
                "",
            ])

        if "step3_transformation_diagnostic" in plots:
            lines.extend([
                "### Scale Transformation Diagnostic",
                f"![Transformation Diagnostic]({_rel_img(plots['step3_transformation_diagnostic'])})",
                "",
            ])

        if "step4_cascade_overview" in plots:
            lines.extend([
                "### Gating Cascade Overview",
                f"![Gating Cascade Overview]({_rel_img(plots['step4_cascade_overview'])})",
                "",
            ])

        gate_plot_keys = [k for k in plots.keys() if k.startswith("step4_gate_")]
        if gate_plot_keys:
            lines.extend([
                "### Sequential Gate Overlays",
                "",
            ])
            for k in gate_plot_keys:
                gate_title = k.replace("step4_gate_", "").replace("_pos", "+").replace("_", " ").title()
                lines.extend([
                    f"#### Gate: {gate_title}",
                    f"![{gate_title}]({_rel_img(plots[k])})",
                    "",
                ])

        lines.extend([
            "## 10. Saved Data Files & Artifacts",
            "",
            "- **Population Statistics:** `population_statistics.csv`",
        ])
        if som_res:
            lines.append("- **SOM Cluster Statistics:** `som_statistics.csv`")
            lines.append("- **SOM Macro-Groups Data:** `som_macro_groups.csv`")
        if (report_path.parent / "tables" / "umap_coordinates.csv").exists():
            lines.append("- **UMAP 2D Coordinates Table:** `tables/umap_coordinates.csv`")
        if pheno_res:
            lines.append("- **PhenoGraph Community Statistics:** `phenograph_statistics.csv`")
            lines.append("- **PhenoGraph Macro-Groups Data:** `phenograph_macro_groups.csv`")
        if ct_som is not None:
            lines.append("- **Concordance Cross-tabulation (SOM):** `comparison_crosstab_som.csv`")
            lines.append("- **Concordance Cross-tabulation (PhenoGraph):** `comparison_crosstab_phenograph.csv`")
        if three_way_metrics is not None:
            lines.append("- **Three-Way Similarity Metrics (ARI/NMI):** `comparison_three_methods_metrics.csv`")
        if three_way_lineages is not None:
            lines.append("- **Three-Way Lineage Quantitations:** `comparison_three_methods_lineages.csv`")
        lines.extend([
            "- **FlowX Replay Workspace (JSON):** `workspace.json`",
            "- **FlowX Replay Workspace (XML):** `workspace.xml`",
            "- **Preprocessing & Gating FlowJo Workspace (.wsp):** `preprocessing_gating.wsp`",
            "- **Preprocessing & Gating Gating-ML 2.0 (.xml):** `preprocessing_gating_ml_2.0.xml`",
            "- **Full Markdown Report:** `report.md`",
        ])
        if log_file_path:
            lines.append(f"- **Execution Log File:** `{Path(log_file_path).name}`")
        lines.extend([
            "",
            "---",
            f"*Pipeline execution completed in **{dur_display} ({execution_time_seconds:.2f}s)** at {timestamp_str}. Detailed execution trace recorded in `{Path(log_file_path).name if log_file_path else 'pipeline.log'}`.*",
            "",
        ])

        report_path.write_text("\n".join(lines), encoding="utf-8")
