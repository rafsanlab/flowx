"""Command-Line Interface (CLI) for Flow Cytometry Analysis.

Provides commands to inspect FCS files, plot scatter figures, apply compensation
and transformations, execute the end-to-end analysis pipeline, and replay saved workspaces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path for direct script execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import time

import numpy as np
import pandas as pd

from flowx.io.logger import format_duration
from flowx.io.paths import sanitize_path
from flowx.io.parser import parse_fcs
from flowx.math.compensation import (
    CompensationMatrix,
    calculate_spillover_from_controls,
)
from flowx.math.transforms import (
    BaseTransform,
    BiexponentialTransform,
    LogicleTransform,
    apply_transform_to_dataframe,
    get_optimal_transform,
)
from flowx.io.inspector import (
    export_channel_details_csv,
    export_inspection_csv,
    export_inspection_markdown,
    export_inspection_suite,
    export_qc_file_lists,
    find_fcs_files,
    inspect_fcs_directory,
    inspect_fcs_file,
    inspect_fcs_files,
    load_population_map,
)

from flowx.batch import (
    BatchCorrectionConfig,
    BatchPreparationConfig,
    CohortSampleInfo,
    DownsampleConfig,
    assess_batch_effect_need,
    discover_cohort_samples,
    load_cohort_manifest,
    run_batch_preparation,
)
from flowx.visualization.batch_plots import plot_batch_assessment_summary
from flowx.pipeline import FlowCytometryPipeline
from flowx.gating.cleanup import (
    CleanUpGateConfig,
    CleanUpGatingResult,
    run_cleanup_gating,
)
from flowx.gating.runner import run_cohort_cleanup
from flowx.visualization.cleanup_plots import plot_cleanup_diagnostics
from flowx.ui.console import (
    render_artifacts_panel,
    render_banner,
    render_cli_home,
    render_parser_help,
    render_cleanup_summary,
    render_dataset_card,
    render_datasets_table,
    render_fcs_batch_summary,
    render_fcs_info,
    render_fcs_inspection_table,
    COLOR_TINT_1,
    get_console,
    maybe_progress,
    progress_enabled,
    render_success_footer,
)
from flowx.datasets import (
    BENCHMARK_DATASETS,
    DatasetInfo,
    download_dataset,
    get_dataset,
    list_datasets,
    resolve_datasets,
    update_catalog_summaries,
    verify_dataset,
)

from flowx.visualization.plotting import (
    plot_compensation_comparison,
    plot_scatter_2d,
    plot_spillover_matrix,
    plot_transformation_comparison,
)
from flowx.workspace.session import WorkspaceSession


class FlowXArgumentParser(argparse.ArgumentParser):
    """ArgumentParser subclass that renders --help with the FlowX Rich colour palette."""

    def print_help(self, file: Optional[Any] = None) -> None:
        render_parser_help(self, file=file)


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the CLI argument parser."""
    from flowx import __version__

    parser = FlowXArgumentParser(
        prog="flowx",
        description="FlowX: Computational Flow Cytometry for Automated Analysis.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Sub-commands",
        parser_class=FlowXArgumentParser,
    )

    # Command 0: inspect
    inspect_p = subparsers.add_parser(
        "inspect",
        help="Inspect FCS files, detect instruments & channels, and export summary CSV & report.",
    )
    inspect_p.add_argument(
        "targets",
        type=str,
        nargs="*",
        default=[],
        help="Path(s) to raw .fcs file(s) or directory containing .fcs files.",
    )
    inspect_p.add_argument(
        "--files-list",
        "--file-list",
        dest="files_list",
        type=str,
        default=None,
        help="Path to text file (one filepath per line) or CSV file listing FCS files to inspect.",
    )
    inspect_p.add_argument(
        "-o",
        "--output-dir",
        "--output",
        dest="output_dir",
        type=str,
        default="outputs/flowcyt_inspection",
        help="Directory to save inspection summary, channel details, and report (default: outputs/flowcyt_inspection).",
    )
    inspect_p.add_argument(
        "-m",
        "--population-map",
        "--pop-map",
        dest="population_map",
        type=str,
        default=None,
        help="Optional path to JSON/YAML population mapping configuration (e.g. configs/population_maps/flowcyt_benchmark.json).",
    )
    inspect_p.add_argument(
        "--channels-csv",
        dest="channels_csv",
        type=str,
        default=None,
        help="Custom filename/path for detailed per-channel configuration CSV (default: channel_details.csv in output dir).",
    )
    inspect_p.add_argument(
        "--no-markdown",
        action="store_true",
        help="Disable automatic generation of inspection_report.md Markdown audit.",
    )
    inspect_p.add_argument(
        "--pattern",
        type=str,
        default="*.fcs",
        help="File pattern for directory search (default: *.fcs).",
    )
    inspect_p.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Recursively scan all subdirectories for FCS files (default: True).",
    )
    inspect_p.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive directory scanning; only inspect top-level target directory.",
    )
    inspect_p.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum subdirectory depth for recursive scanning (e.g. --max-depth 2).",
    )
    inspect_p.add_argument(
        "--event-stats",
        action="store_true",
        help="Calculate deep event-level statistics (negative event rate, detector saturation).",
    )
    inspect_p.add_argument(
        "-H",
        "--harmonize",
        type=str,
        default=None,
        help="Optional path to panel harmonization specification YAML/JSON (e.g. configs/panel_harmonization.yaml).",
    )
    inspect_p.add_argument(
        "--max-rows",
        type=int,
        default=25,
        help="Maximum sample rows to show in console table (default: 25).",
    )


    # Command 1: info
    info_p = subparsers.add_parser("info", help="Inspect FCS file header, channels, and metadata.")
    info_p.add_argument("fcs_file", type=str, help="Path to raw .fcs file.")


    # Command 2: plot
    plot_p = subparsers.add_parser("plot", help="Generate a 2D scatter/density plot.")
    plot_p.add_argument("fcs_file", type=str, help="Path to raw .fcs file.")
    plot_p.add_argument("--x", required=True, type=str, help="X-axis channel (detector or marker).")
    plot_p.add_argument("--y", required=True, type=str, help="Y-axis channel (detector or marker).")
    plot_p.add_argument("--output", type=str, default="scatter_plot.png", help="Output image file path.")
    plot_p.add_argument("--title", type=str, default=None, help="Custom plot title.")

    # Command 2.5: time-qc (Phase 1 Automated & Manual Time QC)
    tqc_p = subparsers.add_parser(
        "time-qc",
        aliases=["qc"],
        help="Run Phase 1 Time QC (flow-rate anomalies, signal drift, margin trimming) and export clean FCS/plots.",
    )
    tqc_p.add_argument(
        "target",
        type=str,
        nargs="?",
        default=None,
        help="Path to raw .fcs file or directory containing .fcs files.",
    )
    tqc_p.add_argument(
        "--files-list",
        "--file-list",
        dest="files_list",
        type=str,
        default=None,
        help="Path to text/CSV file listing .fcs paths to process.",
    )
    tqc_p.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        default=None,
        help="Path to YAML/JSON pipeline config file (reads 01_time_qc and input sections).",
    )
    tqc_p.add_argument(
        "--bin-width",
        dest="bin_width",
        type=float,
        default=None,
        help="Time bin width in seconds (default: 1.0 or from config).",
    )
    tqc_p.add_argument(
        "--no-drift",
        dest="no_drift",
        action="store_true",
        help="Disable fluorescence signal drift detection across time bins.",
    )
    tqc_p.add_argument(
        "--no-margins",
        dest="no_margins",
        action="store_true",
        help="Disable start/end acquisition margin trimming.",
    )
    tqc_p.add_argument(
        "--min-sample-events",
        dest="min_sample_events",
        type=int,
        default=5000,
        help="Minimum retained events warning threshold for sample tubes (default: 5000).",
    )
    tqc_p.add_argument(
        "--min-control-events",
        dest="min_control_events",
        type=int,
        default=500,
        help="Minimum retained events warning threshold for control tubes (default: 500).",
    )
    tqc_p.add_argument(
        "--export-clean-fcs",
        dest="export_clean_fcs",
        action="store_true",
        help="Export Time-QC filtered events to new .fcs files.",
    )
    tqc_p.add_argument(
        "--no-plot",
        dest="no_plot",
        action="store_true",
        help="Skip rendering diagnostic Time QC plots.",
    )
    tqc_p.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Target output directory (default: outputs/time_qc or <config.output_dir>/01_time_qc).",
    )

    # Command 3: compensate
    comp_p = subparsers.add_parser("compensate", help="Apply Time-QC integrated fluorescence compensation and generate diagnostic suite.")
    comp_p.add_argument("fcs_file", type=str, nargs="?", default=None, help="Path to raw .fcs file or dataset directory containing controls and sample.")
    comp_p.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        default=None,
        help="Path to YAML/JSON pipeline config file (reads 02_compensation and input sections).",
    )
    comp_p.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Target output directory for full compensation suite (e.g. outputs/<dataset_name>_compensation).",
    )
    comp_p.add_argument(
        "--method",
        dest="estimation_method",
        type=str,
        default="hybrid",
        choices=["hybrid", "median_delta_mfi", "autospill_regression"],
        help="Spillover estimation algorithm: 'hybrid' (default), 'median_delta_mfi', or 'autospill_regression'.",
    )
    comp_p.add_argument(
        "--no-time-qc",
        dest="no_time_qc",
        action="store_true",
        help="Disable automatic Time QC filtering on controls and biological sample.",
    )
    comp_p.add_argument(
        "--override",
        dest="manual_overrides",
        type=str,
        action="append",
        default=None,
        help="Manual spillover override in 'Primary->Secondary=Value' format (can be specified multiple times, e.g. --override 'Ex 355: Em 660-A->Ex 628/594: Em 670-A=0.71').",
    )
    comp_p.add_argument(
        "--matrix-csv",
        dest="matrix_csv",
        type=str,
        default=None,
        help="Load a manually edited spillover matrix CSV file directly instead of recalculating from controls.",
    )
    comp_p.add_argument(
        "--comp-controls-dir",
        "--compensation-controls",
        dest="comp_controls_dir",
        type=str,
        default=None,
        help="Directory containing single-stained compensation control FCS files.",
    )
    comp_p.add_argument(
        "--unstained-control",
        dest="unstained_control",
        type=str,
        default=None,
        help="Optional path to separate unstained / negative control FCS file.",
    )
    comp_p.add_argument(
        "--comp-manifest",
        "--control-manifest",
        dest="comp_manifest",
        type=str,
        default=None,
        help="Path to CSV or YAML manifest mapping detector channels to control FCS files.",
    )
    comp_p.add_argument(
        "--qc-manifest",
        "--qc-summary",
        dest="qc_manifest",
        type=str,
        default=None,
        help="Path to existing Time QC summary CSV (qc_cohort_summary.csv) to reuse pre-computed QC decisions and skip cohort time binning.",
    )
    comp_p.add_argument(
        "--output-matrix",
        dest="output_matrix",
        type=str,
        default=None,
        help="Optional path to save calculated spillover matrix CSV.",
    )
    comp_p.add_argument(
        "--voltage-policy",
        dest="voltage_policy",
        type=str,
        default="warn",
        choices=["warn", "fallback_embedded", "scale_power_law", "strict_error"],
        help="PMT detector voltage mismatch policy: 'warn' (default), 'fallback_embedded', 'scale_power_law', or 'strict_error'.",
    )
    comp_p.add_argument(
        "--voltage-tolerance",
        dest="voltage_tolerance",
        type=float,
        default=5.0,
        help="Maximum allowable PMT voltage drift in Volts before triggering discrepancy action (default: 5.0 V).",
    )
    comp_p.add_argument(
        "--no-plots",
        dest="no_plots",
        action="store_true",
        help="Skip rendering verification plots for fast headless/batch execution.",
    )
    comp_p.add_argument("--output-plot", type=str, default="outputs/compensation_diagnostic.png", help="Path to save diagnostic plot.")

    # Command 3b: unmix (Full Spectrum Flow Cytometry Deconvolution)
    unmix_p = subparsers.add_parser(
        "unmix",
        help="Perform full-spectrum optical deconvolution (WLS/OLS), autofluorescence extraction, and diagnostic reporting.",
    )
    unmix_p.add_argument(
        "--sample",
        "-s",
        type=str,
        required=False,
        default=None,
        help="Path to raw spectral FCS sample file (or omit when supplied via --config).",
    )
    unmix_p.add_argument(
        "--config",
        dest="config",
        type=str,
        default=None,
        help="Path to YAML/JSON pipeline config file (reads spectral_unmixing and input sections).",
    )
    unmix_p.add_argument(
        "--controls-dir",
        "-c",
        type=str,
        default=None,
        help="Directory containing single-stained compensation/reference controls.",
    )
    unmix_p.add_argument(
        "--unstained",
        "-u",
        type=str,
        default=None,
        help="Path to unstained biological control FCS file for autofluorescence extraction.",
    )
    unmix_p.add_argument(
        "--method",
        type=str,
        default="wls",
        choices=["wls", "ols", "nnls"],
        help="Deconvolution solver: 'wls' (default: Weighted Least Squares with Poisson noise), 'ols', or 'nnls'.",
    )
    unmix_p.add_argument(
        "--af-mode",
        dest="af_mode",
        type=str,
        default="single",
        choices=["single", "multi", "none"],
        help="Autofluorescence extraction mode: 'single' (default baseline), 'multi' (subpopulation GMM), or 'none'.",
    )
    unmix_p.add_argument(
        "--multi-af-components",
        type=int,
        default=2,
        help="Number of distinct cellular AF subpopulations if af-mode='multi' (default: 2).",
    )
    unmix_p.add_argument(
        "--alpha-poisson",
        type=float,
        default=1.0,
        help="Poisson variance scaling parameter for WLS (default: 1.0).",
    )
    unmix_p.add_argument(
        "--electronic-noise",
        type=float,
        default=50.0,
        help="Baseline electronic read noise variance floor for WLS (default: 50.0).",
    )
    unmix_p.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/spectral_unmixing",
        help="Target output directory for unmixed FCS, diagnostic plots, and report.",
    )

    # Command 4: transform
    xform_p = subparsers.add_parser("transform", help="Apply Logicle, Biexponential, or Arcsinh transformation and plot diagnostic.")
    xform_p.add_argument("fcs_file", type=str, nargs="?", default=None, help="Path to raw .fcs file (or omit when supplied via --config).")
    xform_p.add_argument("-c", "--config", dest="config", type=str, default=None, help="Path to YAML/JSON pipeline config file (reads 04_transformation and input sections).")
    xform_p.add_argument("--channel", type=str, required=False, default=None, help="Fluorescence channel to transform (optional; defaults to config channels or all fluorescence channels).")
    xform_p.add_argument(
        "--transform",
        "--type",
        dest="transform_type",
        type=str,
        default=None,
        choices=["logicle", "biexponential", "biex", "asinh", "arcsinh", "log"],
        help="Transformation scale: 'logicle' (default), 'biexponential', 'asinh'/'arcsinh', or 'log'.",
    )
    xform_p.add_argument("--cofactor", type=float, default=None, help="Arcsinh cofactor (default: 5.0 or from config).")
    xform_p.add_argument(
        "--exclude-channels",
        dest="exclude_channels",
        type=str,
        default=None,
        help="Comma-separated channel names to exclude from transformation.",
    )
    xform_p.add_argument(
        "--markers-file",
        "--markers-metadata",
        dest="markers_file",
        type=str,
        default=None,
        help="Path to curated panel markers.csv for excluding scatter/time/utility roles.",
    )
    xform_p.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=None, help="Optional output directory to write transformed CSV and diagnostic plot(s).")
    xform_p.add_argument("--output-plot", type=str, default="outputs/transform_diagnostic.png", help="Path to save diagnostic plot.")

    # Command 4b: cleanup (Phase 2 Core Clean-up Gating)
    cleanup_p = subparsers.add_parser(
        "cleanup",
        help="Run Phase 2 Core Clean-up Gating (debris removal, dual singlets, viability, and dump gating).",
    )
    cleanup_p.add_argument(
        "target",
        type=str,
        nargs="?",
        default=None,
        help="Path to .fcs file, or directory containing .fcs files.",
    )
    cleanup_p.add_argument(
        "--files-list",
        "--file-list",
        dest="files_list",
        type=str,
        default=None,
        help="Path to text or CSV file listing FCS files to process in cohort mode.",
    )
    cleanup_p.add_argument(
        "--debris-method",
        dest="debris_method",
        type=str,
        default="gmm",
        choices=["gmm", "adaptive", "manual"],
        help="Debris exclusion gating algorithm: 'gmm' (default: clustering) or 'adaptive'.",
    )
    cleanup_p.add_argument(
        "--debris-n-components",
        dest="debris_n_components",
        type=int,
        default=3,
        help="Number of mixture components for GMM scatter clustering (default: 3).",
    )
    cleanup_p.add_argument(
        "--debris-sd-scale",
        dest="debris_sd_scale",
        type=float,
        default=1.0,
        help="Debris boundary looseness multiplier (default: 1.0; >1.0 loosens boundary retention, <1.0 tightens).",
    )
    cleanup_p.add_argument(
        "--debris-min-probability",
        dest="debris_min_probability",
        type=float,
        default=None,
        help="Optional minimum GMM posterior probability across cell clusters (e.g. 0.15-0.30 loosens boundary retention).",
    )
    cleanup_p.add_argument(
        "--singlet-mode",
        dest="singlet_mode",
        type=str,
        default="dual",
        choices=["dual", "single", "one_time", "fsc_only", "ssc_only", "none"],
        help="Singlet gating mode: 'dual' (default: FSC then SSC), 'single'/'one_time' (single active gate), 'fsc_only', 'ssc_only', or 'none'.",
    )
    cleanup_p.add_argument(
        "--singlet-method",
        dest="singlet_method",
        type=str,
        default="gmm_ratio",
        choices=["gmm_ratio", "envelope"],
        help="Singlet discrimination algorithm: 'gmm_ratio' (default: dynamic pulse-ratio clustering) or 'envelope'.",
    )
    cleanup_p.add_argument(
        "--singlet-tolerance",
        dest="singlet_tolerance",
        type=float,
        default=0.25,
        help="Relative diagonal envelope width for doublet exclusion (default: 0.25 = +/- 25%%).",
    )
    cleanup_p.add_argument(
        "--no-fsc-singlets",
        dest="no_fsc_singlets",
        action="store_true",
        help="Disable forward scatter (FSC-A vs FSC-H) singlet gating.",
    )
    cleanup_p.add_argument(
        "--no-ssc-singlets",
        dest="no_ssc_singlets",
        action="store_true",
        help="Disable side scatter (SSC-A vs SSC-H) singlet gating.",
    )
    cleanup_p.add_argument(
        "--fsc-channel",
        dest="fsc_channel",
        type=str,
        default=None,
        help="Explicit detector or marker name for FSC channel (e.g. 'FSC-A', 'FS INT').",
    )
    cleanup_p.add_argument(
        "--ssc-channel",
        dest="ssc_channel",
        type=str,
        default=None,
        help="Explicit detector or marker name for SSC channel (e.g. 'SSC-A', 'SS INT').",
    )
    cleanup_p.add_argument(
        "--fsc-height",
        "--fsc-height-channel",
        dest="fsc_height_channel",
        type=str,
        default=None,
        help="Explicit detector or marker name for FSC height channel (e.g. 'FSC-H', 'FS PEAK').",
    )
    cleanup_p.add_argument(
        "--ssc-height",
        "--ssc-height-channel",
        dest="ssc_height_channel",
        type=str,
        default=None,
        help="Explicit detector or marker name for SSC height channel (e.g. 'SSC-H', 'SS PEAK').",
    )
    cleanup_p.add_argument(
        "--viability-channel",
        dest="viability_channel",
        type=str,
        default=None,
        help="Explicit detector or marker name for viability dye (auto-detected if omitted).",
    )
    cleanup_p.add_argument(
        "--viability-cutoff",
        dest="viability_cutoff",
        type=float,
        default=None,
        help="Explicit upper threshold cutoff for live cells in transformed space (auto-detected if omitted).",
    )
    cleanup_p.add_argument(
        "--viability-looseness",
        dest="viability_looseness",
        type=float,
        default=1.0,
        help="Viability automated cutoff looseness multiplier (default: 1.0; >1.0 loosens, <1.0 tightens).",
    )
    cleanup_p.add_argument(
        "--no-viability",
        dest="no_viability",
        action="store_true",
        help="Disable viability dye exclusion gating.",
    )
    cleanup_p.add_argument(
        "--dump-channel",
        dest="dump_channel",
        type=str,
        default=None,
        help="Explicit detector or marker name for dump / lineage exclusion channel.",
    )
    cleanup_p.add_argument(
        "--dump-cutoff",
        dest="dump_cutoff",
        type=float,
        default=None,
        help="Explicit upper threshold cutoff for dump channel in transformed space.",
    )
    cleanup_p.add_argument(
        "--dump-looseness",
        dest="dump_looseness",
        type=float,
        default=1.0,
        help="Dump automated cutoff looseness multiplier (default: 1.0; >1.0 loosens, <1.0 tightens).",
    )
    cleanup_p.add_argument(
        "--compensation-matrix",
        dest="compensation_matrix",
        type=str,
        default=None,
        help="Path to external calculated spillover matrix CSV (overrides embedded matrix).",
    )
    cleanup_p.add_argument(
        "--no-compensation",
        dest="no_compensation",
        action="store_true",
        help="Disable applying compensation matrix during clean-up.",
    )
    cleanup_p.add_argument(
        "--export-clean-fcs",
        dest="export_clean_fcs",
        action="store_true",
        help="Export clean, viable, single events to standardized .fcs file(s).",
    )
    cleanup_p.add_argument(
        "--export-uncompensated",
        dest="export_uncompensated",
        action="store_true",
        help="Export uncompensated raw event values rather than compensated values.",
    )
    cleanup_p.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="outputs/cleanup",
        help="Directory to save summary tables, reports, plots, and clean FCS files.",
    )
    cleanup_p.add_argument(
        "--output-plot",
        dest="output_plot",
        type=str,
        default=None,
        help="Explicit destination path for diagnostic plot (single file mode).",
    )
    cleanup_p.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        default=None,
        help="Path to YAML/JSON pipeline config file (reads 03_cleanup_gating, 04_transformation, and input sections).",
    )
    cleanup_p.add_argument(
        "--markers-file",
        "--markers-metadata",
        dest="markers_file",
        type=str,
        default=None,
        help="Path to curated panel markers.csv for role-based viability/dump/scatter resolution.",
    )
    cleanup_p.add_argument(
        "--transform",
        "--transform-type",
        dest="transform_type",
        type=str,
        default=None,
        choices=["logicle", "biexponential", "biex", "asinh", "arcsinh", "log"],
        help="Pre-gating scale transformation method (default: config or logicle).",
    )
    cleanup_p.add_argument(
        "--cofactor",
        dest="cofactor",
        type=float,
        default=None,
        help="Arcsinh cofactor for pre-gating transformation (default: config or 5.0).",
    )
    cleanup_p.add_argument(
        "--no-plot",
        dest="no_plot",
        action="store_true",
        help="Skip rendering diagnostic plots for fast headless execution.",
    )

    # Command 4.5: batch-prepare
    batch_p = subparsers.add_parser(
        "batch-prepare",
        aliases=["batch", "normalize"],
        help="Run Step 3 Batch Preparation (CytoNorm normalization, cohort downsampling, and balancing).",
    )
    batch_p.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        default=None,
        help="Path to YAML or JSON configuration file specifying batch preparation parameters.",
    )
    batch_p.add_argument(
        "targets",
        type=str,
        nargs="*",
        default=[],
        help="Path(s) to FCS files or directory containing cohort FCS files.",
    )
    batch_p.add_argument(
        "--manifest",
        dest="manifest",
        type=str,
        default=None,
        help="Path to cohort manifest CSV/JSON specifying sample_id, batch_id, is_control, and filepath.",
    )
    batch_p.add_argument(
        "--files-list",
        "--file-list",
        dest="files_list",
        type=str,
        default=None,
        help="Path to text or CSV file listing cohort FCS files.",
    )
    batch_p.add_argument(
        "--method",
        dest="method",
        type=str,
        default="cytonorm",
        choices=["cytonorm", "quantile", "none"],
        help="Batch normalization methodology: 'cytonorm' (FlowSOM-clustered splines on controls) or 'quantile' (anchor-free).",
    )
    batch_p.add_argument(
        "--no-normalize",
        dest="no_normalize",
        action="store_true",
        help="Disable batch normalization (run downsampling/balancing only).",
    )
    batch_p.add_argument(
        "--goal",
        dest="goal",
        type=str,
        default="median",
        choices=["median", "mean", "reference"],
        help="Target distribution goal for quantile normalization (default: 'median').",
    )
    batch_p.add_argument(
        "--reference-batch",
        dest="reference_batch",
        type=str,
        default=None,
        help="Identifier of the reference batch when --goal reference is used.",
    )
    batch_p.add_argument(
        "--clusters",
        dest="clusters",
        type=int,
        default=10,
        help="Number of FlowSOM metaclusters for cell subset partitioning (default: 10).",
    )
    batch_p.add_argument(
        "--quantiles",
        dest="quantiles",
        type=int,
        default=101,
        help="Number of quantile bins for spline mapping (default: 101).",
    )
    batch_p.add_argument(
        "--downsample",
        dest="downsample",
        type=int,
        default=50000,
        help="Target event quota per sample (default: 50,000 events).",
    )
    batch_p.add_argument(
        "--downsample-mode",
        dest="downsample_mode",
        type=str,
        default="target_count",
        choices=["target_count", "min_sample", "stratified", "proportional", "none"],
        help="Cohort event balancing mode: 'target_count' (fixed quota), 'min_sample' (auto-balance to smallest sample), 'stratified', or 'none'.",
    )
    batch_p.add_argument(
        "--no-downsample",
        dest="no_downsample",
        action="store_true",
        help="Disable event downsampling (preserve all clean events).",
    )
    batch_p.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Random seed for reproducible deterministic downsampling (default: 42).",
    )
    batch_p.add_argument(
        "--export-fcs",
        dest="export_fcs",
        action="store_true",
        help="Export individual normalized/downsampled events to standardized .fcs files.",
    )
    batch_p.add_argument(
        "--no-pooled",
        dest="no_pooled",
        action="store_true",
        help="Disable exporting consolidated pooled cohort CSV dataset.",
    )
    batch_p.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="outputs/batch_prep",
        help="Directory to save summary tables, models, diagnostic figures, and prepared FCS files.",
    )
    batch_p.add_argument(
        "--no-plot",
        dest="no_plot",
        action="store_true",
        help="Skip rendering diagnostic plots for fast headless execution.",
    )
    batch_p.add_argument(
        "--markers-file",
        "--markers-metadata",
        dest="markers_file",
        type=str,
        default=None,
        help="Optional path to curated markers.csv to restrict normalization to biological markers.",
    )
    batch_p.add_argument(
        "--check-only",
        dest="check_only",
        action="store_true",
        help="Perform inter-batch necessity and drift assessment only (skip downsampling/export).",
    )
    batch_p.add_argument(
        "--no-assess",
        dest="no_assess",
        action="store_true",
        help="Disable automated batch effect necessity assessment.",
    )
    batch_p.add_argument(
        "--overwrite-autocheck",
        dest="overwrite_autocheck",
        action="store_true",
        help="Force executing batch normalization even when auto-check returns NOT_NEEDED or NOT_RECOMMENDED_CONFOUNDED.",
    )

    # Command 4.6: batch-check
    check_p = subparsers.add_parser(
        "batch-check",
        aliases=["check-batch"],
        help="Diagnose inter-batch drift and assess whether batch normalization is necessary.",
    )
    check_p.add_argument(
        "targets",
        type=str,
        nargs="*",
        default=[],
        help="Path(s) to FCS files or directory containing cohort FCS files.",
    )
    check_p.add_argument(
        "--manifest",
        dest="manifest",
        type=str,
        default=None,
        help="Path to cohort manifest CSV/JSON specifying sample_id, batch_id, is_control, and filepath.",
    )
    check_p.add_argument(
        "--condition-key",
        dest="condition_key",
        type=str,
        default="condition",
        help="Metadata column/key to test for batch-condition confounding (default: 'condition').",
    )
    check_p.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        default=None,
        help="Path to YAML/JSON pipeline config file (reads input metadata_file/data_dir/fcs_file and condition_column).",
    )
    check_p.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save assessment Markdown report and diagnostic figure (optional).",
    )

    # Command 5: pipeline
    pipe_p = subparsers.add_parser("pipeline", help="Run full end-to-end reproducible analysis pipeline.")
    pipe_p.add_argument(
        "fcs_file",
        type=str,
        nargs="?",
        default=None,
        help="Path to input raw .fcs file (optional if specified in --config).",
    )
    pipe_p.add_argument(
        "--files-list",
        "--file-list",
        dest="files_list",
        type=str,
        default=None,
        help="Path to text file (one filepath per line) or CSV listing FCS files to process in batch mode.",
    )
    pipe_p.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to YAML (.yaml, .yml) or JSON (.json) parameter configuration file.",
    )
    pipe_p.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory for plots and results (default: outputs).",
    )
    pipe_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42).",
    )
    pipe_p.add_argument(
        "--gating-mode",
        type=str,
        default=None,
        choices=["template", "som", "phenograph", "compare"],
        help="Methodology: 'som' (default: unsupervised Self-Organizing Map), 'template' (manual gating), 'phenograph', or 'compare' (all three).",
    )
    pipe_p.add_argument(
        "--qc-method",
        type=str,
        default=None,
        choices=["clustering", "template"],
        help="Quality Control (QC) method: 'clustering' (default: automated GMM scatter and singlet deconvolution) or 'template' (polygon templates).",
    )
    pipe_p.add_argument(
        "--cluster-parent",
        type=str,
        default=None,
        help="Parent population to cluster on: 'viable' (default, excludes dead cells & viability dye), 'singlets', 'cells', or 'all'.",
    )
    pipe_p.add_argument(
        "--legend-mode",
        type=str,
        default=None,
        choices=["centroid", "external"],
        help="Legend style: 'centroid' (direct badges at cluster centers) or 'external' (multi-column legend).",
    )
    pipe_p.add_argument("--no-timestamp", action="store_true", help="Disable timestamped run subdirectories.")
    pipe_p.add_argument("--overwrite", action="store_true", help="Allow overwriting target directory without timestamping.")
    pipe_p.add_argument(
        "--no-time-qc",
        dest="no_time_qc",
        action="store_true",
        help="Disable automatic Step 1 Time QC (fluidic stability and signal drift filtering).",
    )
    pipe_p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Optional path to custom workspace.json or workspace.xml defining custom gating hierarchy.",
    )
    pipe_p.add_argument(
        "--adaptive-gating",
        action="store_true",
        help="Use openCyto/flowDensity data-driven local density minimum identification between peaks.",
    )
    pipe_p.add_argument(
        "--marker-mapping",
        type=str,
        default=None,
        help="Custom marker-to-detector assignments as comma-separated pairs (e.g. 'CD3=FL1-A,CD4=FL2-A') or JSON file path.",
    )
    pipe_p.add_argument(
        "--comp-controls-dir",
        "--compensation-controls",
        dest="comp_controls_dir",
        type=str,
        default=None,
        help="Directory containing single-stained compensation control FCS files.",
    )
    pipe_p.add_argument(
        "--comp-manifest",
        "--control-manifest",
        dest="comp_manifest",
        type=str,
        default=None,
        help="Path to CSV or YAML manifest explicitly mapping detectors/channels to control FCS files.",
    )
    pipe_p.add_argument(
        "--unstained-control",
        dest="unstained_control",
        type=str,
        default=None,
        help="Optional path to separate unstained / negative control FCS file.",
    )
    pipe_p.add_argument(
        "--comp-source",
        dest="comp_source",
        type=str,
        default=None,
        choices=["auto", "controls", "embedded", "none"],
        help="Compensation source: 'controls' (calculate from single-stained controls), 'embedded' (use FCS header), 'auto', or 'none'.",
    )
    pipe_p.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging verbosity level (detailed execution trace is always persisted to pipeline.log in output folder).",
    )
    pipe_p.add_argument(
        "--transform",
        "--transform-type",
        dest="transform_type",
        type=str,
        default=None,
        choices=["logicle", "biexponential", "biex", "asinh", "log"],
        help="Scale transformation: 'logicle' (default: Gating-ML 2.0) or 'biexponential' (FlowJo LUT).",
    )
    pipe_p.add_argument(
        "--umap",
        dest="enable_umap",
        action="store_true",
        default=None,
        help="Enable UMAP 2D non-linear manifold projection, cluster maps, and continuous marker feature heatmaps.",
    )
    pipe_p.add_argument(
        "--no-umap",
        dest="enable_umap",
        action="store_false",
        help="Disable UMAP manifold embedding calculation.",
    )

    # Phase 2 Core Clean-up Gating options
    pipe_p.add_argument(
        "--debris-method",
        dest="debris_method",
        type=str,
        default=None,
        choices=["gmm", "adaptive", "manual"],
        help="Clean-up debris exclusion method: 'gmm' (default: clustering), 'adaptive', or 'manual'.",
    )
    pipe_p.add_argument(
        "--singlet-mode",
        dest="singlet_mode",
        type=str,
        default=None,
        choices=["dual", "single", "one_time", "fsc_only", "ssc_only", "none"],
        help="Clean-up singlet discrimination mode ('dual' [default], 'single'/'one_time', 'fsc_only', etc.).",
    )
    pipe_p.add_argument(
        "--debris-sd-scale",
        dest="debris_sd_scale",
        type=float,
        default=None,
        help="Clean-up debris boundary looseness multiplier (default: 1.0; >1.0 loosens, <1.0 tightens).",
    )
    pipe_p.add_argument(
        "--debris-min-probability",
        dest="debris_min_probability",
        type=float,
        default=None,
        help="Optional minimum GMM posterior probability for retaining debris-boundary events.",
    )
    pipe_p.add_argument(
        "--singlet-method",
        dest="singlet_method",
        type=str,
        default=None,
        choices=["gmm_ratio", "envelope"],
        help="Clean-up singlet discrimination algorithm ('gmm_ratio' [default] or 'envelope').",
    )
    pipe_p.add_argument(
        "--singlet-tolerance",
        dest="singlet_tolerance",
        type=float,
        default=None,
        help="Clean-up relative diagonal envelope width for doublet exclusion (default: 0.25).",
    )
    pipe_p.add_argument(
        "--viability-cutoff",
        dest="viability_cutoff",
        type=float,
        default=None,
        help="Clean-up explicit viability upper threshold cutoff in transformed space.",
    )
    pipe_p.add_argument(
        "--viability-looseness",
        dest="viability_looseness",
        type=float,
        default=None,
        help="Clean-up viability automated threshold looseness multiplier (default: 1.0; >1.0 loosens, <1.0 tightens).",
    )
    pipe_p.add_argument(
        "--dump-looseness",
        dest="dump_looseness",
        type=float,
        default=None,
        help="Clean-up dump automated threshold looseness multiplier (default: 1.0; >1.0 loosens, <1.0 tightens).",
    )
    pipe_p.add_argument(
        "--overwrite-autocheck",
        dest="overwrite_autocheck",
        action="store_true",
        default=None,
        help="Force executing batch normalization even when auto-check returns NOT_NEEDED or NOT_RECOMMENDED_CONFOUNDED.",
    )
    pipe_p.add_argument(
        "--no-viability",
        dest="no_viability",
        action="store_true",
        help="Disable clean-up viability dye exclusion gating.",
    )
    pipe_p.add_argument(
        "--export-clean-fcs",
        dest="export_clean_fcs",
        action="store_true",
        help="Export clean, viable, single events to standardized .fcs file(s) in clean_fcs/ subfolder.",
    )

    # Phase 4 & 5 Cohort Analysis options in pipeline
    pipe_p.add_argument(
        "-m",
        "--metadata-file",
        "--sample-metadata",
        dest="metadata_file",
        type=str,
        default=None,
        help="Path to cohort metadata CSV/YAML annotating experimental conditions, batches, or donor pairs.",
    )
    pipe_p.add_argument(
        "--markers-file",
        "--markers-metadata",
        dest="markers_file",
        type=str,
        default=None,
        help="Path to curated panel markers.csv mapping $PnN detectors to markers, roles, and cell types.",
    )
    pipe_p.add_argument(
        "--cell-signatures",
        dest="cell_signatures",
        type=str,
        default=None,
        help="Optional YAML/JSON file specifying custom marker-combination signatures for transparent cluster annotation.",
    )
    pipe_p.add_argument(
        "--cluster-name-map",
        dest="cluster_name_map",
        type=str,
        default=None,
        help="JSON string or path to JSON file mapping cluster IDs to biological names (e.g. '{\"0\": \"CD4+ Naive\"}').",
    )
    pipe_p.add_argument(
        "--cluster-palette",
        dest="cluster_palette",
        type=str,
        default=None,
        help="Colormap/palette name for clusters across UMAP and heatmaps (default: 'deep').",
    )
    pipe_p.add_argument(
        "--clustering-method",
        dest="clustering_method",
        type=str,
        default=None,
        choices=["som", "phenograph", "compare"],
        help="Unsupervised discovery engine: 'som' (FlowSOM [default]), 'phenograph', or 'compare'.",
    )
    pipe_p.add_argument(
        "--condition-column",
        dest="condition_column",
        type=str,
        default=None,
        help="Metadata column holding experimental conditions/groups for cohort DA/DS testing (default: 'condition').",
    )
    pipe_p.add_argument(
        "--batch-column",
        dest="batch_column",
        type=str,
        default=None,
        help="Metadata column holding batch identifier for cohort analysis.",
    )
    pipe_p.add_argument(
        "--umap-covariates",
        dest="umap_covariates",
        type=str,
        default=None,
        help="Comma-separated list of metadata columns to visualize on UMAP.",
    )
    pipe_p.add_argument(
        "--reference-group",
        "--reference",
        dest="reference_group",
        type=str,
        default=None,
        help="Baseline/reference condition for cohort Differential Abundance and Differential State testing.",
    )
    pipe_p.add_argument(
        "--target-group",
        "--target",
        dest="target_group",
        type=str,
        default=None,
        help="Target/perturbed condition for cohort Differential Abundance and Differential State testing.",
    )
    pipe_p.add_argument(
        "--paired-column",
        dest="paired_column",
        type=str,
        default=None,
        help="Optional metadata column for paired cohort observations (e.g. 'donor_id', 'subject').",
    )
    pipe_p.add_argument(
        "-k",
        "--clusters",
        "--n-clusters",
        dest="n_clusters",
        type=int,
        default=None,
        help="Target metaclusters (caps upper k bound when --auto-k is active; use null in YAML to scan up to k_max).",
    )
    pipe_p.add_argument(
        "--auto-k",
        dest="auto_k",
        action="store_true",
        default=None,
        help="Automatically select optimal metacluster count k via --auto-k-method (default: consensus PAC).",
    )
    pipe_p.add_argument(
        "--no-auto-k",
        dest="auto_k",
        action="store_false",
        help="Disable automatic k selection and strictly use fixed --n-clusters count.",
    )
    pipe_p.add_argument(
        "--auto-k-method",
        dest="auto_k_method",
        type=str,
        default=None,
        choices=["consensus", "davies_bouldin", "silhouette", "calinski_harabasz", "elbow"],
        help="Optimal k selection algorithm: 'consensus' (FlowSOM PAC default), 'davies_bouldin', 'silhouette', 'calinski_harabasz', or 'elbow'.",
    )
    pipe_p.add_argument(
        "--k-min",
        dest="k_min",
        type=int,
        default=None,
        help="Minimum metacluster count for auto-k evaluation (default: 4).",
    )
    pipe_p.add_argument(
        "--k-max",
        dest="k_max",
        type=int,
        default=None,
        help="Maximum metacluster count for auto-k evaluation (default: 20).",
    )
    pipe_p.add_argument(
        "--consensus-subsample",
        dest="consensus_subsample",
        type=float,
        default=None,
        help="Subsampling fraction of SOM nodes per consensus resampling iteration (default: 0.8).",
    )
    pipe_p.add_argument(
        "--consensus-iterations",
        dest="consensus_iterations",
        type=int,
        default=None,
        help="Number of consensus resampling iterations for PAC auto-k estimation (default: 50).",
    )
    pipe_p.add_argument(
        "--umap-events",
        "--umap-events-per-sample",
        dest="umap_events_per_sample",
        type=int,
        default=None,
        help="Maximum events per sample to downsample for UMAP embedding (default: 1000).",
    )
    pipe_p.add_argument(
        "--marker-roles",
        dest="marker_roles",
        type=str,
        default=None,
        help="Comma-separated marker roles from markers.csv for clustering when lineage_markers is 'auto' (e.g. 'lineage' or 'lineage,functional').",
    )
    pipe_p.add_argument(
        "--include-markers",
        dest="include_markers",
        type=str,
        default=None,
        help="Comma-separated markers or channels to always include alongside role-selected markers.",
    )
    pipe_p.add_argument(
        "--exclude-markers",
        dest="exclude_markers",
        type=str,
        default=None,
        help="Comma-separated markers or channels to exclude from clustering.",
    )
    pipe_p.add_argument(
        "--stats-granularity",
        dest="stats_granularity",
        type=str,
        default=None,
        choices=["populations", "clusters", "both"],
        help="Statistical DA/DS resolution in Step 06: 'populations' (default), 'clusters', or 'both'.",
    )
    pipe_p.add_argument(
        "--stats-marker-roles",
        dest="stats_marker_roles",
        type=str,
        default=None,
        help="Comma-separated marker roles from markers.csv for Step 06 DS testing (default: 'lineage,functional').",
    )
    pipe_p.add_argument(
        "--export-augmented-fcs",
        dest="export_augmented_fcs",
        action="store_true",
        default=None,
        help="Export analysis-ready FCS files with Cluster_ID, Lineage_ID, UMAP_1, UMAP_2 channels.",
    )
    pipe_p.add_argument(
        "--no-augmented-fcs",
        dest="export_augmented_fcs",
        action="store_false",
        help="Disable export of augmented FCS files.",
    )
    pipe_p.add_argument(
        "--export-prism",
        dest="export_prism",
        action="store_true",
        default=None,
        help="Export GraphPad Prism formatted grouped tables.",
    )
    pipe_p.add_argument(
        "--no-prism",
        dest="export_prism",
        action="store_false",
        help="Disable GraphPad Prism export.",
    )
    pipe_p.add_argument(
        "--run-cohort",
        dest="run_cohort",
        action="store_true",
        help="Force executing Phase 4 & Phase 5 cohort population analysis when processing multiple samples.",
    )
    pipe_p.add_argument(
        "--qc",
        dest="enable_qc",
        action="store_true",
        default=None,
        help="Enable Quality Control phase (Time QC, compensation, transformation, clean-up gating). Default is enabled.",
    )
    pipe_p.add_argument(
        "--no-qc",
        dest="enable_qc",
        action="store_false",
        help="Disable Quality Control phase.",
    )
    pipe_p.add_argument(
        "--discovery",
        dest="enable_discovery",
        action="store_true",
        default=None,
        help="Enable unsupervised Population Discovery phase (FlowSOM/PhenoGraph clustering and UMAP embeddings). Default is disabled.",
    )
    pipe_p.add_argument(
        "--no-discovery",
        dest="enable_discovery",
        action="store_false",
        help="Disable unsupervised Population Discovery phase.",
    )
    pipe_p.add_argument(
        "--numbered-steps",
        dest="numbered_steps",
        action="store_true",
        default=None,
        help="Organize run outputs into numbered phase directories (01_time_qc, 02_compensation, etc.). Default is enabled.",
    )
    pipe_p.add_argument(
        "--no-numbered-steps",
        dest="numbered_steps",
        action="store_false",
        help="Disable organizing run outputs into numbered phase directories.",
    )
    pipe_p.add_argument(
        "--technology",
        dest="technology",
        type=str,
        default=None,
        choices=["auto", "conventional", "spectral"],
        help="Cytometry acquisition technology: 'auto' (default), 'conventional', or 'spectral'.",
    )
    pipe_p.add_argument(
        "--time-qc-bin-width",
        dest="time_qc_bin_width",
        type=float,
        default=None,
        help="Time bin width (seconds) for Step 01 Time QC stability analysis (default: 5.0).",
    )
    pipe_p.add_argument(
        "--no-time-drift",
        dest="no_time_drift",
        action="store_true",
        help="Disable signal drift detection during Step 01 Time QC.",
    )
    pipe_p.add_argument(
        "--no-time-margins",
        dest="no_time_margins",
        action="store_true",
        help="Disable acquisition start/end margin trimming during Step 01 Time QC.",
    )
    pipe_p.add_argument(
        "--comp-estimation-method",
        dest="comp_estimation_method",
        type=str,
        default=None,
        choices=["hybrid", "median_delta_mfi", "autospill_regression"],
        help="Spillover calculation algorithm from single-stain controls (default: hybrid).",
    )
    pipe_p.add_argument(
        "--voltage-mismatch-policy",
        dest="voltage_mismatch_policy",
        type=str,
        default=None,
        choices=["warn", "strict", "rescale", "ignore"],
        help="Policy when PMT/detector voltages differ between controls and sample.",
    )
    pipe_p.add_argument(
        "--voltage-tolerance",
        dest="voltage_tolerance",
        type=float,
        default=None,
        help="Voltage mismatch tolerance in volts (default: 5.0).",
    )
    pipe_p.add_argument(
        "--unmix-method",
        dest="unmix_method",
        type=str,
        default=None,
        choices=["wls", "ols", "nnls"],
        help="Spectral unmixing solver when technology is spectral ('wls', 'ols', or 'nnls').",
    )
    pipe_p.add_argument(
        "--af-mode",
        dest="af_mode",
        type=str,
        default=None,
        choices=["single", "multi", "none"],
        help="Spectral unmixing autofluorescence handling ('single', 'multi', or 'none').",
    )
    pipe_p.add_argument(
        "--no-transform",
        dest="no_transform",
        action="store_true",
        help="Disable Step 04 scale transformation (for already-transformed inputs).",
    )
    pipe_p.add_argument(
        "--transform-cofactor",
        dest="transform_cofactor",
        type=float,
        default=None,
        help="Arcsinh cofactor when transform method is asinh (default: config or 5.0).",
    )
    pipe_p.add_argument(
        "--transform-channels",
        dest="transform_channels",
        type=str,
        default=None,
        help="Comma-separated fluorescence channels to transform, or 'auto'.",
    )
    pipe_p.add_argument(
        "--transform-exclude",
        dest="transform_exclude",
        type=str,
        default=None,
        help="Comma-separated channels to exclude from transformation.",
    )
    pipe_p.add_argument(
        "--no-cleanup",
        dest="no_cleanup",
        action="store_true",
        help="Disable Step 03 Core Clean-up Gating.",
    )
    pipe_p.add_argument(
        "--debris-n-components",
        dest="debris_n_components",
        type=int,
        default=None,
        help="Number of GMM components for debris exclusion (default: 3).",
    )
    pipe_p.add_argument("--fsc-channel", dest="fsc_channel", type=str, default=None, help="Explicit FSC channel name.")
    pipe_p.add_argument("--ssc-channel", dest="ssc_channel", type=str, default=None, help="Explicit SSC channel name.")
    pipe_p.add_argument("--fsc-height", "--fsc-height-channel", dest="fsc_height_channel", type=str, default=None, help="Explicit FSC-H channel name.")
    pipe_p.add_argument("--ssc-height", "--ssc-height-channel", dest="ssc_height_channel", type=str, default=None, help="Explicit SSC-H channel name.")
    pipe_p.add_argument("--viability-channel", dest="viability_channel", type=str, default=None, help="Explicit viability dye channel name or 'auto'.")
    pipe_p.add_argument("--dump-channel", dest="dump_channel", type=str, default=None, help="Explicit dump channel name or 'auto'.")
    pipe_p.add_argument("--dump-cutoff", dest="dump_cutoff", type=float, default=None, help="Explicit dump gate cutoff in transformed space.")
    pipe_p.add_argument(
        "--no-batch-prep",
        dest="no_batch_prep",
        action="store_true",
        help="Disable Step 04 Batch Preparation (normalization & downsampling).",
    )
    pipe_p.add_argument(
        "--normalization-method",
        dest="normalization_method",
        type=str,
        default=None,
        choices=["cytonorm", "quantile", "none"],
        help="Step 04 batch normalization method: 'cytonorm', 'quantile', or 'none'.",
    )
    pipe_p.add_argument(
        "--batch-goal",
        dest="batch_goal",
        type=str,
        default=None,
        choices=["median", "mean", "reference"],
        help="Target distribution goal for quantile normalization ('median', 'mean', 'reference').",
    )
    pipe_p.add_argument(
        "--downsample-mode",
        dest="downsample_mode",
        type=str,
        default=None,
        choices=["target_count", "min_sample", "stratified", "proportional", "none"],
        help="Step 04 cohort event downsampling mode.",
    )
    pipe_p.add_argument(
        "--target-events",
        dest="target_events",
        type=int,
        default=None,
        help="Target events per sample for Step 04 downsampling.",
    )
    pipe_p.add_argument(
        "--no-batch-check",
        dest="no_batch_check",
        action="store_true",
        help="Disable pre-normalization automated batch effect necessity assessment.",
    )
    pipe_p.add_argument(
        "--lineage-markers",
        dest="lineage_markers",
        type=str,
        default=None,
        help="Comma-separated markers for discovery clustering, or 'auto' to resolve via markers.csv.",
    )
    pipe_p.add_argument(
        "--da-method",
        dest="da_test_method",
        type=str,
        default=None,
        choices=["auto", "mann_whitney", "wilcoxon_paired", "t_test", "welch_ttest", "kruskal", "glm_binomial"],
        help="Step 06 Differential Abundance test method (default: 'auto').",
    )
    pipe_p.add_argument(
        "--ds-method",
        dest="ds_test_method",
        type=str,
        default=None,
        choices=["auto", "mann_whitney", "t_test", "wilcoxon_paired"],
        help="Step 06 Differential State test method (default: 'auto').",
    )
    pipe_p.add_argument(
        "--fdr-threshold",
        dest="fdr_threshold",
        type=float,
        default=None,
        help="Benjamini-Hochberg FDR significance threshold in Step 06 (default: 0.05).",
    )
    pipe_p.add_argument(
        "--log2fc-threshold",
        dest="log2fc_threshold",
        type=float,
        default=None,
        help="Minimum absolute log2 fold-change threshold for DA highlights (default: 1.0).",
    )
    pipe_p.add_argument(
        "--export-flowjo-wsp",
        dest="export_flowjo_wsp",
        action="store_true",
        default=None,
        help="Export FlowJo .wsp XML workspace.",
    )
    pipe_p.add_argument(
        "--no-flowjo-wsp",
        dest="export_flowjo_wsp",
        action="store_false",
        help="Disable FlowJo .wsp XML workspace export.",
    )
    pipe_p.add_argument(
        "--export-gating-ml",
        dest="export_gating_ml",
        action="store_true",
        default=None,
        help="Export Gating-ML 2.0 XML file.",
    )
    pipe_p.add_argument(
        "--no-gating-ml",
        dest="export_gating_ml",
        action="store_false",
        help="Disable Gating-ML 2.0 XML export.",
    )
    pipe_p.add_argument(
        "--miflowcyt-report",
        dest="generate_miflowcyt_report",
        action="store_true",
        default=None,
        help="Generate MIFlowCyt compliance Markdown & HTML reports.",
    )
    pipe_p.add_argument(
        "--no-miflowcyt-report",
        dest="generate_miflowcyt_report",
        action="store_false",
        help="Disable MIFlowCyt compliance report generation.",
    )

    # Command: cohort (Phase 4 Population Discovery & Phase 5 Cohort Statistical Analysis)
    cohort_p = subparsers.add_parser(
        "cohort",
        aliases=["population", "cluster-stats"],
        help="Run Phase 4 Population Discovery & Phase 5 Cohort Statistical Analysis (FlowSOM/PhenoGraph, UMAP, DA/DS testing, augmented FCS, Prism).",
    )
    cohort_p.add_argument(
        "targets",
        type=str,
        nargs="*",
        default=[],
        help="Path(s) to input FCS/CSV file(s) or directory containing cohort files.",
    )
    cohort_p.add_argument(
        "--files-list",
        "--file-list",
        dest="files_list",
        type=str,
        default=None,
        help="Path to text or CSV file listing cohort sample files to process.",
    )
    cohort_p.add_argument(
        "-m",
        "--metadata-file",
        "--sample-metadata",
        dest="metadata_file",
        type=str,
        default=None,
        help="Path to sample metadata CSV/YAML annotating experimental conditions, batches, or donor pairs.",
    )
    cohort_p.add_argument(
        "--markers-file",
        "--markers-metadata",
        dest="markers_file",
        type=str,
        default=None,
        help="Path to curated panel markers.csv mapping $PnN detectors to markers, roles, and cell types.",
    )
    cohort_p.add_argument(
        "--cell-signatures",
        dest="cell_signatures",
        type=str,
        default=None,
        help="Optional YAML/JSON file specifying custom marker-combination signatures for transparent cluster annotation.",
    )
    cohort_p.add_argument(
        "--markers",
        type=str,
        default=None,
        help="Comma-separated list of lineage marker channels (e.g. 'CD3,CD4,CD8,CD19'), or 'auto' to resolve via markers.csv.",
    )
    cohort_p.add_argument(
        "--marker-roles",
        dest="marker_roles",
        type=str,
        default=None,
        help="Comma-separated marker roles from markers.csv to select (e.g. 'lineage' or 'lineage,functional').",
    )
    cohort_p.add_argument(
        "--include-markers",
        dest="include_markers",
        type=str,
        default=None,
        help="Comma-separated markers/channels to force-include when using role-based or auto marker selection.",
    )
    cohort_p.add_argument(
        "--exclude-markers",
        dest="exclude_markers",
        type=str,
        default=None,
        help="Comma-separated markers/channels to exclude from clustering.",
    )
    cohort_p.add_argument(
        "--stats-granularity",
        dest="stats_granularity",
        type=str,
        default=None,
        choices=["populations", "clusters", "both"],
        help="Step 06 DA/DS granularity: 'populations' (default aggregated biological labels), 'clusters', or 'both'.",
    )
    cohort_p.add_argument(
        "--stats-marker-roles",
        dest="stats_marker_roles",
        type=str,
        default=None,
        help="Comma-separated marker roles from markers.csv for Step 06 DS testing (default: 'lineage,functional').",
    )
    cohort_p.add_argument(
        "--method",
        "--clustering-method",
        dest="clustering_method",
        type=str,
        default=None,
        choices=["som", "phenograph"],
        help="Unsupervised clustering algorithm: 'som' (FlowSOM: SOM + consensus metaclustering [default]) or 'phenograph' (Louvain graph).",
    )
    cohort_p.add_argument(
        "-k",
        "--clusters",
        "--n-clusters",
        dest="n_clusters",
        type=int,
        default=None,
        help="Target metaclusters for FlowSOM (caps upper k bound when --auto-k is active; default: config or 10).",
    )
    cohort_p.add_argument(
        "--auto-k",
        dest="auto_k",
        action="store_true",
        default=None,
        help="Automatically select optimal metacluster count k via --auto-k-method (default: consensus PAC).",
    )
    cohort_p.add_argument(
        "--no-auto-k",
        dest="auto_k",
        action="store_false",
        help="Disable automatic k selection and strictly use fixed --n-clusters count.",
    )
    cohort_p.add_argument(
        "--auto-k-method",
        dest="auto_k_method",
        type=str,
        default=None,
        choices=["consensus", "davies_bouldin", "silhouette", "calinski_harabasz", "elbow"],
        help="Optimal k selection algorithm: 'consensus' (FlowSOM PAC default), 'davies_bouldin', 'silhouette', 'calinski_harabasz', or 'elbow'.",
    )
    cohort_p.add_argument(
        "--k-min",
        dest="k_min",
        type=int,
        default=None,
        help="Minimum cluster count for auto-k scan (default: 4).",
    )
    cohort_p.add_argument(
        "--k-max",
        dest="k_max",
        type=int,
        default=None,
        help="Maximum cluster count for auto-k scan (default: 20).",
    )
    cohort_p.add_argument(
        "--consensus-subsample",
        dest="consensus_subsample",
        type=float,
        default=None,
        help="Subsampling fraction per consensus resampling iteration (default: 0.8).",
    )
    cohort_p.add_argument(
        "--consensus-iterations",
        dest="consensus_iterations",
        type=int,
        default=None,
        help="Number of consensus resampling iterations for PAC auto-k estimation (default: 50).",
    )
    cohort_p.add_argument(
        "--umap",
        dest="compute_umap",
        action="store_true",
        default=None,
        help="Compute 2D UMAP non-linear manifold projection and continuous marker feature heatmaps (default: enabled).",
    )
    cohort_p.add_argument(
        "--no-umap",
        dest="compute_umap",
        action="store_false",
        help="Disable UMAP manifold calculation.",
    )
    cohort_p.add_argument(
        "--umap-events",
        "--umap-events-per-sample",
        dest="umap_events_per_sample",
        type=int,
        default=None,
        help="Maximum events per sample to downsample for UMAP embedding (default: 1000).",
    )
    cohort_p.add_argument(
        "--condition-column",
        type=str,
        default=None,
        help="Metadata column holding experimental conditions/groups (default: 'condition').",
    )
    cohort_p.add_argument(
        "--batch-column",
        dest="batch_column",
        type=str,
        default=None,
        help="Metadata column holding batch identifier for cohort analysis.",
    )
    cohort_p.add_argument(
        "--umap-covariates",
        dest="umap_covariates",
        type=str,
        default=None,
        help="Comma-separated list of metadata columns to visualize on UMAP.",
    )
    cohort_p.add_argument(
        "--reference-group",
        "--reference",
        dest="reference_group",
        type=str,
        default=None,
        help="Baseline/reference condition for Differential Abundance and Differential State testing (e.g. 'Control', 'Healthy', '0.2kPa').",
    )
    cohort_p.add_argument(
        "--target-group",
        "--target",
        dest="target_group",
        type=str,
        default=None,
        help="Target/perturbed condition for Differential Abundance and Differential State testing (e.g. 'Treated', 'Disease', '40kPa').",
    )
    cohort_p.add_argument(
        "--paired-column",
        type=str,
        default=None,
        help="Optional metadata column for paired cohort observations (e.g. 'donor_id', 'patient').",
    )
    cohort_p.add_argument(
        "--cluster-name-map",
        type=str,
        default=None,
        help="JSON string or path to JSON file mapping cluster IDs to biological lineage names (e.g. '{\"0\": \"CD4+ T\", \"1\": \"Classical Monocytes\"}').",
    )
    cohort_p.add_argument(
        "--cluster-palette",
        dest="cluster_palette",
        type=str,
        default=None,
        help="Colormap/palette name for clusters across UMAP and heatmaps (default: 'deep').",
    )
    cohort_p.add_argument(
        "--export-augmented-fcs",
        dest="export_augmented_fcs",
        action="store_true",
        default=None,
        help="Export analysis-ready FCS files with Cluster_ID, Lineage_ID, UMAP_1, UMAP_2 channels (default: True).",
    )
    cohort_p.add_argument(
        "--no-augmented-fcs",
        dest="export_augmented_fcs",
        action="store_false",
        help="Disable export of augmented FCS files.",
    )
    cohort_p.add_argument(
        "--export-prism",
        action="store_true",
        default=None,
        help="Export GraphPad Prism formatted grouped tables (default: True).",
    )
    cohort_p.add_argument(
        "--no-prism",
        dest="export_prism",
        action="store_false",
        help="Disable GraphPad Prism export.",
    )
    cohort_p.add_argument(
        "--clean-up",
        action="store_true",
        help="Execute Step 01-03 cleanup gating on raw FCS targets before clustering.",
    )
    cohort_p.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to YAML or JSON parameter configuration file.",
    )
    cohort_p.add_argument(
        "-o",
        "--output-dir",
        "--output",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save figures, tables, augmented FCS, and reports (default: outputs/cohort_population).",
    )
    cohort_p.add_argument(
        "--da-method",
        dest="da_test_method",
        type=str,
        default=None,
        choices=["auto", "mann_whitney", "wilcoxon_paired", "t_test", "welch_ttest", "kruskal", "glm_binomial"],
        help="Step 06 Differential Abundance test method (default: 'auto').",
    )
    cohort_p.add_argument(
        "--ds-method",
        dest="ds_test_method",
        type=str,
        default=None,
        choices=["auto", "mann_whitney", "t_test", "wilcoxon_paired"],
        help="Step 06 Differential State test method (default: 'auto').",
    )
    cohort_p.add_argument(
        "--fdr-threshold",
        dest="fdr_threshold",
        type=float,
        default=None,
        help="Benjamini-Hochberg FDR threshold (default: 0.05).",
    )
    cohort_p.add_argument(
        "--log2fc-threshold",
        dest="log2fc_threshold",
        type=float,
        default=None,
        help="Minimum |log2FC| significance cutoff (default: 1.0).",
    )
    cohort_p.add_argument(
        "--export-flowjo-wsp",
        dest="export_flowjo_wsp",
        action="store_true",
        default=None,
        help="Export FlowJo .wsp workspace (default: True).",
    )
    cohort_p.add_argument(
        "--no-flowjo-wsp",
        dest="export_flowjo_wsp",
        action="store_false",
        help="Disable FlowJo .wsp workspace export.",
    )
    cohort_p.add_argument(
        "--export-gating-ml",
        dest="export_gating_ml",
        action="store_true",
        default=None,
        help="Export Gating-ML 2.0 XML (default: True).",
    )
    cohort_p.add_argument(
        "--no-gating-ml",
        dest="export_gating_ml",
        action="store_false",
        help="Disable Gating-ML 2.0 XML export.",
    )
    cohort_p.add_argument(
        "--miflowcyt-report",
        dest="generate_miflowcyt_report",
        action="store_true",
        default=None,
        help="Generate MIFlowCyt compliance report (default: True).",
    )
    cohort_p.add_argument(
        "--no-miflowcyt-report",
        dest="generate_miflowcyt_report",
        action="store_false",
        help="Disable MIFlowCyt compliance report.",
    )
    cohort_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic clustering and UMAP projection (default: 42).",
    )


    # Command 6: replay
    replay_p = subparsers.add_parser("replay", help="Replay a saved workspace session and verify statistics.")
    replay_p.add_argument("workspace_file", type=str, help="Path to workspace.json or workspace.xml.")
    replay_p.add_argument("--fcs-file", type=str, default=None, help="Optional override for FCS file path.")

    # Command 7: config
    cfg_p = subparsers.add_parser("config", help="Generate or validate pipeline configuration parameter files.")
    cfg_p.add_argument(
        "--template",
        type=str,
        default=None,
        help="Generate a starter configuration template and save to specified path (e.g. 'run_config.yaml').",
    )
    cfg_p.add_argument(
        "--format",
        type=str,
        default="yaml",
        choices=["yaml", "json"],
        help="Template format ('yaml' or 'json', default: yaml).",
    )
    cfg_p.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Validate an existing YAML or JSON configuration file.",
    )

    # Command 9: download / dataset
    dl_p = subparsers.add_parser(
        "download",
        aliases=["dataset"],
        help="List, inspect, verify, and download benchmark conventional and spectral datasets.",
    )
    dl_p.add_argument(
        "datasets",
        type=str,
        nargs="*",
        default=[],
        help="Dataset selector(s): ID, alias, numeric index (e.g. 1, 1,3), or 'all'.",
    )
    dl_p.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available benchmark datasets in a formatted table.",
    )
    dl_p.add_argument(
        "-c",
        "--category",
        type=str,
        default="all",
        choices=["all", "conventional", "spectral"],
        help="Filter datasets by category (default: all).",
    )
    dl_p.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="Display detailed scientific metadata and local health for selected dataset(s).",
    )
    dl_p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing local files and integrity without downloading.",
    )
    dl_p.add_argument(
        "--include-supplementary",
        action="store_true",
        default=False,
        help="Also download paper/publication and non-FCS supplementary files (default: disabled).",
    )
    dl_p.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory containing conventional/ and spectral/ (default: data).",
    )
    dl_p.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format.",
    )

    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    """Execute inspect command for list of FCS files, directories, or manifest lists."""
    try:
        t0 = time.perf_counter()

        # 1. Gather all file paths from CLI targets and/or --files-list
        file_paths: List[Path] = []

        # From --files-list
        if getattr(args, "files_list", None):
            fl_path = Path(args.files_list).resolve()
            if not fl_path.is_file():
                print(f"Error: Files list not found: {args.files_list}", file=sys.stderr)
                return 1

            if fl_path.suffix.lower() == ".csv":
                df_list = pd.read_csv(fl_path)
                col = None
                for candidate in ["filepath", "path", "file", "filename"]:
                    if candidate in [c.lower() for c in df_list.columns]:
                        col = [c for c in df_list.columns if c.lower() == candidate][0]
                        break
                if col:
                    for val in df_list[col].dropna():
                        raw_str = str(val).strip()
                        if not raw_str:
                            continue
                        p = Path(raw_str)
                        if not p.is_absolute():
                            if p.is_file():
                                p = p.resolve()
                            elif fl_path.parent and (fl_path.parent / p).is_file():
                                p = (fl_path.parent / p).resolve()
                            else:
                                p = (Path.cwd() / p).resolve()
                        if p.is_file():
                            file_paths.append(p)
                        else:
                            print(f"Warning: File from list not found: {raw_str}", file=sys.stderr)
                else:
                    print(f"Error: CSV files list '{fl_path.name}' has no 'filepath' or 'filename' column.", file=sys.stderr)
                    return 1
            else:
                for line in fl_path.read_text(encoding="utf-8").splitlines():
                    clean_line = line.strip()
                    if clean_line and not clean_line.startswith("#"):
                        p = Path(clean_line)
                        if not p.is_absolute():
                            if p.is_file():
                                p = p.resolve()
                            elif fl_path.parent and (fl_path.parent / p).is_file():
                                p = (fl_path.parent / p).resolve()
                            else:
                                p = (Path.cwd() / p).resolve()
                        if p.is_file():
                            file_paths.append(p)
                        else:
                            print(f"Warning: File from list not found: {clean_line}", file=sys.stderr)

        # From positional targets
        for target_str in getattr(args, "targets", []):
            tp = Path(target_str).resolve()
            if not tp.exists():
                print(f"Warning: Target path not found: {target_str}", file=sys.stderr)
                continue
            if tp.is_file():
                file_paths.append(tp)
            elif tp.is_dir():
                is_recursive = getattr(args, "recursive", True)
                pattern = getattr(args, "pattern", None)
                max_d = getattr(args, "max_depth", None)
                found = find_fcs_files(tp, recursive=is_recursive, pattern=pattern, max_depth=max_d)
                file_paths.extend(found)

        # Deduplicate paths while preserving order
        seen: Set[str] = set()
        unique_paths: List[Path] = []
        for p in file_paths:
            p_str = str(p.resolve())
            if p_str not in seen:
                seen.add(p_str)
                unique_paths.append(p)

        if not unique_paths:
            print("Error: No FCS files found to inspect. Specify file(s), directory, or --files-list.", file=sys.stderr)
            return 1

        # 2. Destination output directory
        raw_out = getattr(args, "output_dir", "outputs/flowcyt_inspection")
        out_dir = Path(raw_out).resolve()
        if out_dir.suffix.lower() == ".csv":
            # If user provided a specific CSV path, place companions in its parent directory
            summary_csv_path = out_dir
            out_dir = out_dir.parent
        else:
            summary_csv_path = out_dir / "summary.csv"

        out_dir.mkdir(parents=True, exist_ok=True)

        # 3. Perform inspection
        results, summary = inspect_fcs_files(
            file_paths=unique_paths,
            population_map=getattr(args, "population_map", None),
            include_event_stats=args.event_stats,
        )

        # 4. Export artifacts
        export_inspection_csv(results, summary_csv_path)

        channel_csv_path = Path(args.channels_csv).resolve() if getattr(args, "channels_csv", None) else (out_dir / "channel_details.csv")
        export_channel_details_csv(results, channel_csv_path)

        report_md_path = None
        if not getattr(args, "no_markdown", False):
            report_md_path = out_dir / "inspection_report.md"
            export_inspection_markdown(
                summary=summary,
                results=results,
                output_md=report_md_path,
                generated_by="FlowX Automated FCS Inspector v1.0",
            )

        # Export partitioned QC status file lists (PASS vs FLAGGED/FAIL) for downstream handoff
        pass_list_path, flagged_list_path = export_qc_file_lists(results, out_dir)

        # 5. Render Rich console outputs
        render_fcs_batch_summary(
            summary,
            csv_path=summary_csv_path,
            channel_csv_path=channel_csv_path,
        )
        render_fcs_inspection_table(results, max_rows=args.max_rows)

        harm_csv_path = None
        harm_md_path = None
        if getattr(args, "harmonize", None):
            try:
                from flowx.panels.harmonization import load_panel_harmonizer
                from flowx.io.parser import parse_fcs
                print(f"\nApplying cohort panel harmonization from '{args.harmonize}'...")
                harmonizer = load_panel_harmonizer(args.harmonize)
                sample_objs = [parse_fcs(p) for p in unique_paths]
                harm_report = harmonizer.harmonize_cohort(sample_objs)
                harm_csv_path, harm_md_path = harm_report.export(out_dir)
            except Exception as h_exc:
                print(f"Warning: Panel harmonization audit encountered an error: {h_exc}", file=sys.stderr)

        elapsed = time.perf_counter() - t0
        print(f"\nInspection complete in {format_duration(elapsed)} ({elapsed:.2f}s).")
        print(f"  • Summary CSV:      {sanitize_path(summary_csv_path)}")
        print(f"  • Channels CSV:     {sanitize_path(channel_csv_path)}")
        if report_md_path:
            print(f"  • Markdown Report:  {sanitize_path(report_md_path)}")
        if harm_csv_path:
            print(f"  • Harmonization Matrix: {sanitize_path(harm_csv_path)}")
            print(f"  • Harmonization Report: {sanitize_path(harm_md_path)}")
        print(f"  • Passed QC List:   {sanitize_path(pass_list_path)}")
        print(f"  • Flagged QC List:  {sanitize_path(flagged_list_path)}")
        return 0

    except Exception as exc:
        print(f"Inspection error: {exc}", file=sys.stderr)
        return 1



def cmd_info(args: argparse.Namespace) -> int:

    """Execute info command."""
    try:
        t0 = time.perf_counter()
        fcs = parse_fcs(args.fcs_file)
        render_fcs_info(fcs, duration=time.perf_counter() - t0)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_plot(args: argparse.Namespace) -> int:
    """Execute plot command."""
    try:
        t0 = time.perf_counter()
        fcs = parse_fcs(args.fcs_file)
        x_det = fcs.resolve_channel_name(args.x)
        y_det = fcs.resolve_channel_name(args.y)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plot_scatter_2d(
            df=fcs.events_df,
            x_channel=x_det,
            y_channel=y_det,
            x_label=fcs.get_channel_label(x_det),
            y_label=fcs.get_channel_label(y_det),
            title=args.title or f"{fcs.get_channel_label(x_det)} vs {fcs.get_channel_label(y_det)}",
            output_path=out_path,
        )
        elapsed = time.perf_counter() - t0
        print(f"Plot saved successfully to '{sanitize_path(out_path)}' in {format_duration(elapsed)} ({elapsed:.2f}s).")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _flag_in_argv(*flags: str) -> bool:
    """Return True when any of the given CLI flags was explicitly passed on sys.argv."""
    raw = sys.argv[1:]
    return any(
        token in flags or any(token.startswith(f"{flag}=") for flag in flags)
        for token in raw
    )


def _load_cli_config(args: argparse.Namespace) -> Optional[Any]:
    """Load PipelineConfig when --config is supplied on a CLI subcommand."""
    cfg_path = getattr(args, "config", None)
    if not cfg_path:
        return None
    from flowx.io.config import PipelineConfig

    return PipelineConfig.load(cfg_path)


def _resolve_cli_marker_metadata(
    args: argparse.Namespace,
    config_obj: Optional[Any],
    dataset_dir: Optional[Path],
    fcs_data: Any,
) -> Any:
    """Resolve MarkerMetadataResolution via --markers-file, config, or dataset discovery."""
    from flowx.io.metadata import resolve_marker_metadata

    markers_path = getattr(args, "markers_file", None) or (
        getattr(config_obj, "markers_file", None) if config_obj is not None else None
    )
    return resolve_marker_metadata(
        markers_file=markers_path,
        dataset_dir=dataset_dir,
        fcs_pnn=getattr(fcs_data, "pnn_labels", None),
        fcs_pns=getattr(fcs_data, "pns_labels", None),
        marker_mapping=(getattr(config_obj, "marker_mapping", None) if config_obj is not None else None),
    )


def cmd_time_qc(args: argparse.Namespace) -> int:
    """Execute Phase 1 Time QC command across a single file, directory, or file list."""
    try:
        from flowx.qc.models import TimeQCConfig
        from flowx.qc.runner import run_cohort_time_qc

        config_obj = _load_cli_config(args)
        qc_cfg = config_obj.to_time_qc_config() if config_obj is not None else TimeQCConfig()

        if getattr(args, "bin_width", None) is not None:
            qc_cfg.bin_width_sec = float(args.bin_width)
        if getattr(args, "no_drift", False):
            qc_cfg.enable_signal_drift = False
        if getattr(args, "no_margins", False):
            qc_cfg.trim_margins = False

        source = (
            getattr(args, "files_list", None)
            or getattr(args, "target", None)
            or (getattr(config_obj, "files_list", None) if config_obj else None)
            or (getattr(config_obj, "fcs_file", None) if config_obj else None)
            or (getattr(config_obj, "data_dir", None) if config_obj else None)
        )
        if not source:
            print("Error: Provide an FCS file/directory, --files-list, or --config with input paths.", file=sys.stderr)
            return 1

        out_dir = getattr(args, "output_dir", None)
        if not out_dir:
            if config_obj is not None and getattr(config_obj, "output_dir", None):
                out_dir = str(Path(config_obj.output_dir) / "01_time_qc")
            else:
                out_dir = "outputs/time_qc"

        summary_df, csv_path, report_path = run_cohort_time_qc(
            file_list_or_paths=source,
            output_dir=out_dir,
            config=qc_cfg,
            generate_plots=not getattr(args, "no_plot", False),
            min_control_events=int(getattr(args, "min_control_events", 500)),
            min_sample_events=int(getattr(args, "min_sample_events", 5000)),
            export_clean_fcs=bool(getattr(args, "export_clean_fcs", False)),
        )
        print(f"Time QC complete across {len(summary_df)} file(s).")
        print(f"  • Summary CSV:    {sanitize_path(csv_path)}")
        print(f"  • Time QC Report: {sanitize_path(report_path)}")
        return 0
    except Exception as exc:
        print(f"Error in time-qc: {exc}", file=sys.stderr)
        return 1


def cmd_compensate(args: argparse.Namespace) -> int:
    """Execute compensation command."""
    try:
        from flowx.math.compensation import (
            CompensationConfig,
            run_compensation_step,
            run_dataset_compensation,
        )

        t0 = time.perf_counter()
        config_obj = _load_cli_config(args)

        target_str = (
            getattr(args, "fcs_file", None)
            or (getattr(config_obj, "fcs_file", None) if config_obj else None)
            or (getattr(config_obj, "data_dir", None) if config_obj else None)
        )
        if not target_str:
            print("Error: Provide an FCS file/directory or --config with input.fcs_file / input.data_dir.", file=sys.stderr)
            return 1
        target_p = Path(target_str)

        parsed_overrides: Dict[str, float] = {}
        if getattr(args, "manual_overrides", None):
            for item in args.manual_overrides:
                if "=" in item and "->" in item:
                    k_part, v_part = item.split("=", 1)
                    parsed_overrides[k_part.strip()] = float(v_part.strip())

        cfg = config_obj.to_compensation_config() if config_obj is not None else CompensationConfig()
        if _flag_in_argv("--method"):
            cfg.estimation_method = getattr(args, "estimation_method", "hybrid")
        if _flag_in_argv("--no-time-qc"):
            cfg.apply_time_qc_to_controls = False
            cfg.apply_time_qc_to_sample = False
        if parsed_overrides:
            cfg.manual_overrides = {**(cfg.manual_overrides or {}), **parsed_overrides}
        if _flag_in_argv("--voltage-policy"):
            cfg.voltage_mismatch_policy = getattr(args, "voltage_policy", "warn")
        if _flag_in_argv("--voltage-tolerance"):
            cfg.voltage_tolerance_v = float(getattr(args, "voltage_tolerance", 5.0))

        controls_dir = getattr(args, "comp_controls_dir", None) or (config_obj.comp_controls_dir if config_obj else None)
        comp_manifest = getattr(args, "comp_manifest", None) or (config_obj.comp_manifest if config_obj else None)
        unstained_ctrl = getattr(args, "unstained_control", None) or (config_obj.unstained_control if config_obj else None)
        out_dir_arg = getattr(args, "output_dir", None)
        if not out_dir_arg and config_obj is not None and getattr(config_obj, "output_dir", None):
            out_dir_arg = str(Path(config_obj.output_dir) / "02_compensation")

        has_explicit_controls = controls_dir is not None or comp_manifest is not None
        if target_p.is_dir() or has_explicit_controls:
            suite_res = run_dataset_compensation(
                dataset_dir_or_sample=target_p,
                output_dir=out_dir_arg,
                controls_dir=controls_dir,
                manifest_path=comp_manifest,
                unstained_control=unstained_ctrl,
                qc_manifest_path=getattr(args, "qc_manifest", None),
                config=cfg,
                generate_plots=not getattr(args, "no_plots", False),
            )
            elapsed = time.perf_counter() - t0
            print(f"Completed dataset Time QC + Compensation Suite in {format_duration(elapsed)} ({elapsed:.2f}s).")
            print(f"  • Output Directory:  {sanitize_path(Path(suite_res['output_dir']))}")
            print(f"  • Compensation Report: {sanitize_path(Path(suite_res['report_md_path']))}")
            return 0

        fcs = parse_fcs(target_p)

        comp: Optional[CompensationMatrix] = None
        ctrl_res = None
        matrix_csv_arg = getattr(args, "matrix_csv", None)
        if matrix_csv_arg:
            mat_df = pd.read_csv(matrix_csv_arg, index_col=0)
            comp = CompensationMatrix(detectors=list(mat_df.columns), matrix=mat_df.to_numpy())
            comp_df = comp.apply_dataframe(fcs.events_df)
        else:
            step_res = run_compensation_step(
                fcs_data=fcs,
                working_df=fcs.events_df,
                enable_compensation=True,
                compensation_source=getattr(config_obj, "compensation_source", "auto") if config_obj else "auto",
                technology=getattr(config_obj, "technology", "auto") if config_obj else "auto",
                comp_controls_dir=controls_dir,
                comp_manifest=comp_manifest,
                unstained_control=unstained_ctrl,
                comp_channel_mapping=getattr(config_obj, "comp_channel_mapping", None) if config_obj else None,
                comp_config=cfg,
                spectral_config=config_obj.to_unmixing_config() if (config_obj and getattr(config_obj, "technology", "auto") == "spectral") else None,
                plots_dir=Path(out_dir_arg) / "plots" if out_dir_arg else None,
                tables_dir=Path(out_dir_arg) / "tables" if out_dir_arg else None,
                sample_stem=target_p.stem,
                random_seed=getattr(config_obj, "seed", 42) if config_obj else 42,
                generate_plots=not getattr(args, "no_plots", False),
                generate_nxn=False,
            )
            comp = step_res.comp_matrix
            ctrl_res = step_res.comp_ctrl_result
            comp_df = step_res.compensated_df

        if comp is None:
            print("Error: No spillover matrix found in FCS file and no --comp-controls-dir, --comp-manifest, or --matrix-csv provided.", file=sys.stderr)
            return 1

        if comp.dataframe.shape[0] >= 2:
            spill_arr = comp.dataframe.values.copy()
            np.fill_diagonal(spill_arr, 0.0)
            if np.max(spill_arr) > 0.0:
                max_idx = np.unravel_index(np.argmax(spill_arr), spill_arr.shape)
                ch1 = comp.detectors[max_idx[0]]
                ch2 = comp.detectors[max_idx[1]]
            else:
                ch1 = comp.detectors[0]
                ch2 = comp.detectors[1]
        else:
            ch1 = comp.detectors[0]
            ch2 = comp.detectors[1]

        out_dir = Path(out_dir_arg) if out_dir_arg else None
        if out_dir is not None:
            (out_dir / "plots").mkdir(parents=True, exist_ok=True)
            (out_dir / "tables").mkdir(parents=True, exist_ok=True)
            (out_dir / "compensated_samples").mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "plots" / f"{target_p.stem}_compensation.png"
        else:
            out_path = Path(args.output_plot)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        plot_compensation_comparison(
            uncomp_df=fcs.events_df,
            comp_df=comp_df,
            x_channel=ch1,
            y_channel=ch2,
            x_label=fcs.get_channel_label(ch1),
            y_label=fcs.get_channel_label(ch2),
            output_path=out_path,
        )

        if out_dir is not None:
            mat_path = out_dir / "tables" / f"{target_p.stem}_spillover_matrix.csv"
            comp.dataframe.to_csv(mat_path)
            comp_csv = out_dir / "compensated_samples" / f"{target_p.stem}_compensated.csv"
            comp_df.to_csv(comp_csv, index=False)

            report_path = out_dir / "compensation_report.md"
            with open(report_path, "w", encoding="utf-8") as rf:
                rf.write(f"# Flow Cytometry Compensation Report — `{target_p.name}`\n\n")
                rf.write(f"> **Mode:** Single-File Embedded Spillover Compensation  \n")
                rf.write(f"> **File:** `{target_p.name}` ({len(fcs.events_df):,} events)  \n")
                rf.write(f"> **Channels Compensated:** {len(comp.detectors)} ({', '.join(comp.detectors)})  \n")
                rf.write(f"> **Condition Number κ(S):** {comp.condition_number:.2f}  \n\n")
                rf.write(f"## Diagnostic Verification\n\n")
                rf.write(f"- Diagnostic Plot: [`plots/{out_path.name}`](plots/{out_path.name})\n")
                rf.write(f"- Compensated Events: [`compensated_samples/{comp_csv.name}`](compensated_samples/{comp_csv.name})\n")
                rf.write(f"- Spillover Matrix: [`tables/{mat_path.name}`](tables/{mat_path.name})\n")

        elapsed = time.perf_counter() - t0
        source_label = "controls" if ctrl_res else "embedded matrix"
        print(f"Compensated {len(comp.detectors)} channels via {source_label} in {format_duration(elapsed)} ({elapsed:.2f}s). Diagnostic plot saved to '{sanitize_path(out_path)}'.")
        if out_dir is not None:
            print(f"  • Output Directory:  {sanitize_path(out_dir)}")
            print(f"  • Compensation Report: {sanitize_path(out_dir / 'compensation_report.md')}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_transform(args: argparse.Namespace) -> int:
    """Execute transform command using unified run_transformation_step."""
    try:
        from flowx.math.transforms import TransformConfig, run_transformation_step

        t0 = time.perf_counter()
        config_obj = _load_cli_config(args)

        fcs_path_str = getattr(args, "fcs_file", None) or (config_obj.fcs_file if config_obj else None)
        if not fcs_path_str and config_obj is not None and getattr(config_obj, "data_dir", None):
            found = sorted(Path(config_obj.data_dir).rglob("*.fcs"))
            if found:
                fcs_path_str = str(found[0])
        if not fcs_path_str:
            print("Error: Provide an FCS file path or --config with input.fcs_file.", file=sys.stderr)
            return 1

        fcs = parse_fcs(fcs_path_str)
        x_cfg = config_obj.to_transform_config() if config_obj is not None else TransformConfig()
        if getattr(args, "transform_type", None) is not None:
            x_cfg.method = args.transform_type
        if getattr(args, "cofactor", None) is not None:
            x_cfg.cofactor = float(args.cofactor)
        if getattr(args, "exclude_channels", None):
            x_cfg.exclude_channels = [c.strip() for c in args.exclude_channels.split(",") if c.strip()]

        target_channels: Optional[List[str]] = None
        if getattr(args, "channel", None):
            target_channels = [fcs.resolve_channel_name(args.channel)]

        marker_meta = None
        if config_obj is not None and getattr(config_obj, "markers_file", None):
            mp = Path(config_obj.markers_file)
            if mp.is_file():
                marker_meta = pd.read_csv(mp)
        marker_meta = _resolve_cli_marker_metadata(
            args=args,
            config_obj=config_obj,
            dataset_dir=Path(fcs_path_str).parent,
            fcs_data=fcs,
        )

        out_dir_str = getattr(args, "output_dir", None)
        if not out_dir_str and config_obj is not None and getattr(config_obj, "output_dir", None):
            out_dir_str = str(Path(config_obj.output_dir) / "03_transformation")

        out_plot_path = (
            Path(out_dir_str) / f"{Path(fcs_path_str).stem}_transform.png"
            if out_dir_str
            else Path(getattr(args, "output_plot", "outputs/transform_diagnostic.png"))
        )

        step_res = run_transformation_step(
            fcs_data=fcs,
            df_events=fcs.events_df,
            config=x_cfg,
            target_channels=target_channels,
            marker_metadata=marker_meta,
            output_plot_path=out_plot_path,
            generate_plot=True,
        )

        if out_dir_str:
            out_dir_p = Path(out_dir_str)
            out_dir_p.mkdir(parents=True, exist_ok=True)
            csv_out = out_dir_p / f"{Path(fcs_path_str).stem}_transformed.csv"
            step_res.transformed_df.to_csv(csv_out, index=False)

        elapsed = time.perf_counter() - t0
        xform_name = (
            step_res.transform.__class__.__name__
            if step_res.transform is not None
            else x_cfg.method
        )
        print(
            f"Transformed {len(step_res.channels_transformed)} channel(s) with {xform_name} "
            f"in {format_duration(elapsed)} ({elapsed:.2f}s). "
            f"Diagnostic plot saved to '{sanitize_path(out_plot_path)}'."
        )
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Execute Phase 2 Core Clean-up Gating command."""
    try:
        from flowx.gating.cleanup import CleanUpGateConfig
        from flowx.gating.runner import run_cleanup_step
        from flowx.math.compensation import run_compensation_step
        from flowx.math.transforms import TransformConfig, run_transformation_step

        config_obj = _load_cli_config(args)

        target = (
            getattr(args, "target", None)
            or (getattr(config_obj, "fcs_file", None) if config_obj else None)
            or (getattr(config_obj, "data_dir", None) if config_obj else None)
        )
        files_list = getattr(args, "files_list", None) or (getattr(config_obj, "files_list", None) if config_obj else None)

        if not target and not files_list:
            print("Error: Please provide a path to an FCS file or directory, --files-list, or --config.", file=sys.stderr)
            return 1

        cfg = config_obj.to_cleanup_config() if config_obj is not None else CleanUpGateConfig()
        if _flag_in_argv("--debris-method"):
            cfg.debris_method = args.debris_method
        if _flag_in_argv("--debris-n-components"):
            cfg.debris_n_components = int(args.debris_n_components)
        if _flag_in_argv("--debris-sd-scale"):
            cfg.debris_sd_scale = float(args.debris_sd_scale)
        if _flag_in_argv("--debris-min-probability"):
            cfg.debris_min_probability = float(args.debris_min_probability) if args.debris_min_probability is not None else None
        if getattr(args, "fsc_channel", None):
            cfg.fsc_channel = args.fsc_channel
        if getattr(args, "ssc_channel", None):
            cfg.ssc_channel = args.ssc_channel
        if _flag_in_argv("--singlet-mode"):
            cfg.singlet_mode = args.singlet_mode
        if _flag_in_argv("--no-fsc-singlets"):
            cfg.singlet_fsc_enabled = False
        if _flag_in_argv("--no-ssc-singlets"):
            cfg.singlet_ssc_enabled = False
        if _flag_in_argv("--singlet-tolerance"):
            cfg.singlet_tolerance = float(args.singlet_tolerance)
        if _flag_in_argv("--singlet-method"):
            cfg.singlet_method = args.singlet_method
        if getattr(args, "fsc_height_channel", None):
            cfg.fsc_height_channel = args.fsc_height_channel
        if getattr(args, "ssc_height_channel", None):
            cfg.ssc_height_channel = args.ssc_height_channel
        if _flag_in_argv("--no-viability"):
            cfg.viability_enabled = False
        if getattr(args, "viability_channel", None):
            cfg.viability_channel = args.viability_channel
        if getattr(args, "viability_cutoff", None) is not None:
            cfg.viability_cutoff = float(args.viability_cutoff)
        if _flag_in_argv("--viability-looseness"):
            cfg.viability_looseness = float(args.viability_looseness)
        if getattr(args, "dump_channel", None):
            cfg.dump_enabled = True
            cfg.dump_channel = args.dump_channel
        if getattr(args, "dump_cutoff", None) is not None:
            cfg.dump_cutoff = float(args.dump_cutoff)
        if _flag_in_argv("--dump-looseness"):
            cfg.dump_looseness = float(args.dump_looseness)

        out_dir_str = getattr(args, "output_dir", "outputs/cleanup")
        if not _flag_in_argv("-o", "--output-dir") and config_obj is not None and getattr(config_obj, "output_dir", None):
            out_dir_str = str(Path(config_obj.output_dir) / "03_cleanup_gating")
        out_dir = Path(out_dir_str)

        target_path = Path(target) if target else None
        is_cohort = files_list is not None or (target_path is not None and target_path.is_dir())
        markers_path_arg = getattr(args, "markers_file", None) or (config_obj.markers_file if config_obj else None)
        xform_cfg = config_obj.to_transform_config() if config_obj is not None else TransformConfig()
        if getattr(args, "transform_type", None) is not None:
            xform_cfg.method = args.transform_type
        if getattr(args, "cofactor", None) is not None:
            xform_cfg.cofactor = float(args.cofactor)

        if is_cohort:
            # Batch cohort clean-up
            source = files_list if files_list else target
            export_fcs = getattr(args, "export_clean_fcs", False) or (config_obj.export_clean_fcs if config_obj else False)
            generate_plots = not getattr(args, "no_plot", False)
            comp_matrix = getattr(args, "compensation_matrix", None)
            apply_comp = (
                False
                if _flag_in_argv("--no-compensation")
                else (config_obj.enable_compensation if config_obj else True)
            )

            summary_df, summary_csv, report_path = run_cohort_cleanup(
                file_list_or_paths=source,
                output_dir=out_dir,
                config=cfg,
                generate_plots=generate_plots,
                export_clean_fcs=export_fcs,
                apply_embedded_compensation=apply_comp,
                compensation_matrix=comp_matrix,
                markers_file=markers_path_arg,
                transform_config=xform_cfg,
            )
            print(f"Cohort Clean-up complete across {len(summary_df)} files.")
            print(f"Summary CSV: {sanitize_path(summary_csv)}")
            print(f"Summary Report: {sanitize_path(report_path)}")
            return 0

        # Single FCS file clean-up delegated through unified step runners
        fcs = parse_fcs(target_path)
        events_df = fcs.events_df.copy()

        marker_meta = None
        if config_obj is not None and getattr(config_obj, "markers_file", None):
            mp = Path(config_obj.markers_file)
            if mp.is_file():
                marker_meta = pd.read_csv(mp)
        marker_meta = _resolve_cli_marker_metadata(
            args=args,
            config_obj=config_obj,
            dataset_dir=target_path.parent if target_path is not None else None,
            fcs_data=fcs,
        )

        comp_matrix_arg = getattr(args, "compensation_matrix", None)
        no_comp = _flag_in_argv("--no-compensation") or (
            config_obj is not None and not getattr(config_obj, "enable_compensation", True)
        )
        if not no_comp:
            if comp_matrix_arg is not None and Path(comp_matrix_arg).is_file():
                csv_df = pd.read_csv(comp_matrix_arg, index_col=0)
                comp_obj = CompensationMatrix(detectors=list(csv_df.columns), matrix=csv_df.to_numpy())
                events_df = comp_obj.apply_dataframe(events_df)
            else:
                comp_res = run_compensation_step(
                    fcs_data=fcs,
                    working_df=events_df,
                    enable_compensation=True,
                    compensation_source=config_obj.compensation_source if config_obj else "auto",
                    technology=config_obj.technology if config_obj else "auto",
                    comp_controls_dir=config_obj.comp_controls_dir if config_obj else None,
                    comp_manifest=config_obj.comp_manifest if config_obj else None,
                    unstained_control=config_obj.unstained_control if config_obj else None,
                    comp_config=config_obj.to_compensation_config() if config_obj else None,
                    generate_plots=False,
                    generate_nxn=False,
                )
                events_df = comp_res.compensated_df

        xform_cfg = config_obj.to_transform_config() if config_obj is not None else TransformConfig()
        xform_res = run_transformation_step(
            fcs_data=fcs,
            df_events=events_df,
            config=xform_cfg,
            marker_metadata=marker_meta,
            generate_plot=False,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = Path(getattr(args, "output_plot", None) or (out_dir / f"{fcs.sample_name}_cleanup.png"))
        clean_fcs_out = out_dir / f"{fcs.sample_name}_clean.fcs"
        do_export = getattr(args, "export_clean_fcs", False) or (config_obj.export_clean_fcs if config_obj else False)

        step_res = run_cleanup_step(
            fcs_data=fcs,
            df_events=events_df,
            df_transformed=xform_res.transformed_df,
            config=cfg,
            marker_metadata=marker_meta,
            output_plot_path=plot_path,
            generate_plots=not getattr(args, "no_plot", False),
            export_clean_fcs=do_export,
            clean_fcs_path=clean_fcs_out,
            is_compensated=not getattr(args, "export_uncompensated", False),
        )

        render_cleanup_summary(step_res.cleanup_result)
        if step_res.plot_path:
            print(f"Diagnostic plot saved to: '{sanitize_path(step_res.plot_path)}'")
        if step_res.clean_fcs_path:
            print(f"Clean FCS exported to: '{sanitize_path(step_res.clean_fcs_path)}'")

        return 0

    except Exception as exc:
        print(f"Error in cleanup: {exc}", file=sys.stderr)
        return 1


def cmd_batch_prepare(args: argparse.Namespace) -> int:
    """Execute Step 3 Batch Preparation: Normalization and Cohort Downsampling."""
    try:
        config_obj = _load_cli_config(args)
        targets = getattr(args, "targets", [])
        manifest = getattr(args, "manifest", None)
        files_list = getattr(args, "files_list", None)
        cfg_path_str = getattr(args, "config", None)
        manifest = getattr(args, "manifest", None) or (
            getattr(config_obj, "metadata_file", None) if config_obj else None
        )
        files_list = getattr(args, "files_list", None) or (
            getattr(config_obj, "files_list", None) if config_obj else None
        )

        cfg_data: Dict[str, Any] = {}
        if cfg_path_str:
            c_path = Path(cfg_path_str)
            if not c_path.is_file():
                print(f"Error: Config file not found: {cfg_path_str}", file=sys.stderr)
                return 1
            if c_path.suffix.lower() in (".yaml", ".yml"):
                import yaml
                cfg_data = yaml.safe_load(c_path.read_text(encoding="utf-8")) or {}
            elif c_path.suffix.lower() == ".json":
                cfg_data = json.loads(c_path.read_text(encoding="utf-8")) or {}

            input_sec = cfg_data.get("input", {}) if isinstance(cfg_data.get("input"), dict) else {}
            if not manifest:
                manifest = cfg_data.get("manifest") or input_sec.get("metadata_file")
            if not files_list and input_sec.get("files_list"):
                files_list = input_sec.get("files_list")
        if config_obj is not None:
            if not targets and not manifest and not files_list:
                cfg_target = getattr(config_obj, "data_dir", None) or getattr(config_obj, "fcs_file", None)
                if cfg_target:
                    targets = [str(cfg_target)]
            if not _flag_in_argv("-o", "--output-dir"):
                if getattr(config_obj, "output_dir", None):
                    args.output_dir = str(Path(config_obj.output_dir) / "04_batch_preparation")
                elif "output_dir" in cfg_data:
                    args.output_dir = cfg_data["output_dir"]
                elif isinstance(cfg_data.get("output"), dict) and cfg_data["output"].get("output_dir"):
                    args.output_dir = str(Path(cfg_data["output"]["output_dir"]) / "04_batch_preparation")

        if not targets and not manifest and not files_list:
            print("Error: Please provide target FCS files/directory, --manifest, --files-list, or --config.", file=sys.stderr)
            return 1

        if config_obj is not None:
            if (
                cfg_data
                and ("batch_normalization" in cfg_data or "downsampling" in cfg_data)
                and "04_batch_preparation" not in cfg_data
            ):
                prep_cfg = BatchPreparationConfig.from_dict(cfg_data)
            else:
                prep_cfg = config_obj.to_batch_preparation_config()
            if _flag_in_argv("--no-normalize"):
                prep_cfg.batch_normalization.enabled = False
            elif _flag_in_argv("--method"):
                prep_cfg.batch_normalization.enabled = args.method != "none"
                prep_cfg.batch_normalization.method = args.method
            if _flag_in_argv("--goal"):
                prep_cfg.batch_normalization.goal = args.goal
            if _flag_in_argv("--reference-batch"):
                prep_cfg.batch_normalization.reference_batch = args.reference_batch
            if _flag_in_argv("--clusters"):
                prep_cfg.batch_normalization.n_clusters = int(args.clusters)
            if _flag_in_argv("--quantiles"):
                prep_cfg.batch_normalization.n_quantiles = int(args.quantiles)
            if _flag_in_argv("--no-downsample"):
                prep_cfg.downsampling.enabled = False
            if _flag_in_argv("--downsample-mode"):
                prep_cfg.downsampling.mode = args.downsample_mode
                if args.downsample_mode == "none":
                    prep_cfg.downsampling.enabled = False
            if _flag_in_argv("--downsample"):
                prep_cfg.downsampling.target_events = int(args.downsample)
            if _flag_in_argv("--seed"):
                prep_cfg.random_seed = int(args.seed)
                prep_cfg.downsampling.random_seed = int(args.seed)
            if _flag_in_argv("--export-fcs"):
                prep_cfg.export_individual_fcs = True
                prep_cfg.downsampling.export_fcs = True
            if _flag_in_argv("--no-pooled"):
                prep_cfg.export_pooled_cohort = False
            if _flag_in_argv("--no-plot"):
                prep_cfg.generate_diagnostics = False
            if _flag_in_argv("-o", "--output-dir"):
                prep_cfg.output_dir = args.output_dir
            elif getattr(args, "output_dir", None) and args.output_dir != "outputs/batch_prep":
                prep_cfg.output_dir = args.output_dir
            if _flag_in_argv("--no-assess"):
                prep_cfg.assess_batch_effect = False
            if _flag_in_argv("--overwrite-autocheck"):
                prep_cfg.overwrite_autocheck = True
        else:
            norm_enabled = not getattr(args, "no_normalize", False)
            norm_method = getattr(args, "method", "cytonorm")
            if getattr(args, "no_normalize", False) or norm_method == "none":
                norm_enabled = False

            norm_cfg = BatchCorrectionConfig(
                enabled=norm_enabled,
                method=norm_method,
                goal=getattr(args, "goal", "median"),
                reference_batch=getattr(args, "reference_batch", None),
                n_clusters=int(getattr(args, "clusters", 10)),
                n_quantiles=int(getattr(args, "quantiles", 101)),
                save_model=True,
            )

            down_enabled = not getattr(args, "no_downsample", False)
            down_mode = getattr(args, "downsample_mode", "target_count")
            if getattr(args, "no_downsample", False) or down_mode == "none":
                down_enabled = False

            down_cfg = DownsampleConfig(
                enabled=down_enabled,
                mode=down_mode,
                target_events=int(getattr(args, "downsample", 50000)),
                random_seed=int(getattr(args, "seed", 42)),
                export_fcs=bool(getattr(args, "export_fcs", False)),
            )

            prep_cfg = BatchPreparationConfig(
                batch_normalization=norm_cfg,
                downsampling=down_cfg,
                output_dir=Path(getattr(args, "output_dir", "outputs/batch_prep")),
                export_pooled_cohort=not getattr(args, "no_pooled", False),
                export_individual_fcs=bool(getattr(args, "export_fcs", False)),
                generate_diagnostics=not getattr(args, "no_plot", False),
                random_seed=int(getattr(args, "seed", 42)),
                assess_batch_effect=not getattr(args, "no_assess", False),
                overwrite_autocheck=bool(getattr(args, "overwrite_autocheck", False)),
            )

        # Auto-resolve markers.csv for biological channel subsetting when not explicitly overridden
        if prep_cfg.batch_normalization.markers is None:
            ds_dir = Path(targets[0]) if (targets and Path(targets[0]).exists()) else (Path(manifest).parent if manifest else None)
            if ds_dir is not None and ds_dir.is_file():
                ds_dir = ds_dir.parent
            markers_path_arg = getattr(args, "markers_file", None) or (
                getattr(config_obj, "markers_file", None) if config_obj else None
            )
            if markers_path_arg or ds_dir is not None:
                try:
                    from flowx.io.metadata import resolve_marker_metadata

                    marker_meta = resolve_marker_metadata(markers_file=markers_path_arg, dataset_dir=ds_dir)
                    if marker_meta.clustering_channels:
                        prep_cfg.batch_normalization.markers = list(marker_meta.clustering_channels)
                except Exception:
                    pass

        # Resolve cohort input targets
        cohort_input: Any
        if files_list:
            fl_path = Path(files_list)
            if not fl_path.is_file():
                print(f"Error: files-list not found: {files_list}", file=sys.stderr)
                return 1
            lines = [line.strip() for line in fl_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
            cohort_input = lines
            if isinstance(files_list, (list, tuple)):
                cohort_input = [str(p) for p in files_list if p]
            else:
                fl_path = Path(files_list)
                if not fl_path.is_file():
                    print(f"Error: files-list not found: {files_list}", file=sys.stderr)
                    return 1
                lines = [line.strip() for line in fl_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
                cohort_input = lines
        elif manifest:
            cohort_input = None
        elif len(targets) == 1:
            cohort_input = targets[0]
        else:
            cohort_input = targets

        if getattr(args, "check_only", False):
            # Run assessment only
            return cmd_batch_check(args)

        res = run_batch_preparation(
            cohort_targets=cohort_input,
            manifest_path=manifest,
            config=prep_cfg,
            output_dir=prep_cfg.output_dir,
        )

        print("\n" + "=" * 60)
        print("FlowX Batch Preparation & Cohort Balancing Complete")
        print("=" * 60)
        print(f"Cohort Samples:     {res.sample_count}")
        print(f"Total Prepared:     {res.total_prepared_events:,} events")
        print(f"Normalization:      {res.normalization_model.__class__.__name__ if res.normalization_model else 'Disabled'}")
        if res.batch_assessment is not None:
            ba = res.batch_assessment
            print(f"Batch Assessment:   {ba.recommendation} (Drift Score: {ba.drift_score:.1f}/100, Conf: {ba.confidence*100:.0f}%)")
        print(f"Output Directory:   {sanitize_path(res.output_dir)}")
        print(f"Summary Table:      {sanitize_path(res.generated_tables.get('cohort_batch_summary', ''))}")
        if res.pooled_df is not None:
            print(f"Pooled Dataset:     {sanitize_path(res.generated_tables.get('pooled_cohort_events', ''))}")
        if res.exported_fcs_paths:
            print(f"Exported FCS:       {len(res.exported_fcs_paths)} files in {sanitize_path(Path(res.output_dir)/'fcs')}")
        if res.generated_plots:
            print(f"Diagnostic Plots:   {len(res.generated_plots)} figures generated in {sanitize_path(Path(res.output_dir)/'plots')}")
        print("=" * 60 + "\n")

        return 0

    except Exception as exc:
        print(f"Error in batch-prepare: {exc}", file=sys.stderr)
        return 1


def cmd_batch_check(args: argparse.Namespace) -> int:
    """Execute standalone batch necessity and technical drift assessment."""
    try:
        config_obj = _load_cli_config(args)
        manifest = getattr(args, "manifest", None) or (config_obj.metadata_file if config_obj else None)
        targets = getattr(args, "targets", [])
        if not targets and not manifest and config_obj is not None:
            cfg_target = getattr(config_obj, "data_dir", None) or getattr(config_obj, "fcs_file", None)
            if cfg_target:
                targets = [str(cfg_target)]
        condition_key = getattr(args, "condition_key", "condition")
        if not _flag_in_argv("--condition-key") and config_obj is not None and getattr(config_obj, "condition_column", None):
            condition_key = config_obj.condition_column
        output_dir = getattr(args, "output_dir", None)
        if not output_dir and config_obj is not None and getattr(config_obj, "output_dir", None):
            output_dir = str(Path(config_obj.output_dir) / "04_batch_check")

        if not manifest and not targets:
            print("Error: Must provide either cohort target paths or a --manifest CSV/JSON.", file=sys.stderr)
            return 1

        resolved_samples: List[CohortSampleInfo] = []
        if manifest:
            resolved_samples = load_cohort_manifest(manifest)
        elif targets:
            cohort_input = targets[0] if len(targets) == 1 else targets
            resolved_samples = discover_cohort_samples(cohort_input)

        if not resolved_samples:
            print("Error: No cohort samples discovered.", file=sys.stderr)
            return 1

        print(f"Loading {len(resolved_samples)} cohort sample files for batch assessment...")
        working_dfs: Dict[str, pd.DataFrame] = {}
        for s in resolved_samples:
            if s.filepath and Path(s.filepath).exists():
                fcs = parse_fcs(s.filepath)
                working_dfs[s.sample_id] = fcs.events_df

        if not working_dfs:
            print("Error: Could not load any valid FCS event data for assessment.", file=sys.stderr)
            return 1

        assessment = assess_batch_effect_need(
            cohort_dfs=working_dfs,
            sample_info=resolved_samples,
            condition_key=condition_key,
        )

        print("\n" + "=" * 65)
        print("FlowX Batch Effect Necessity Assessment")
        print("=" * 65)
        print(f"Recommendation:     {assessment.recommendation}")
        print(f"Drift Score:        {assessment.drift_score:.1f} / 100.0")
        print(f"Confidence:         {assessment.confidence * 100:.1f}%")
        print(f"Batches Evaluated:  {assessment.batch_count} ({', '.join(assessment.unique_batches) if assessment.unique_batches else 'None'})")
        print(f"Control Anchors:    {'Yes (' + str(len(assessment.control_batches)) + ' batches)' if assessment.has_control_anchors else 'No'}")
        print("\nRationale:")
        for r in assessment.rationale:
            print(f"  • {r}")

        df_markers = assessment.to_dataframe()
        if not df_markers.empty:
            print("\nMarker-Level Drift Breakdown:")
            display_cols = ["marker", "mean_pairwise_emd", "max_median_ratio", "drift_severity", "is_drifting"]
            avail_cols = [c for c in display_cols if c in df_markers.columns]
            print(df_markers[avail_cols].to_string(index=False))

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            rep_path = out_p / "batch_necessity_report.md"
            rep_path.write_text(assessment.to_markdown(), encoding="utf-8")
            plot_path = out_p / "batch_necessity_summary.png"
            plot_batch_assessment_summary(assessment, output_path=plot_path)
            print(f"\nSaved report to:    {sanitize_path(rep_path)}")
            print(f"Saved plot to:      {sanitize_path(plot_path)}")

        print("=" * 65 + "\n")
        return 0

    except Exception as exc:
        print(f"Error in batch-check: {exc}", file=sys.stderr)
        return 1


def _parse_cluster_name_map(raw_name_map: Any) -> Optional[Dict[str, str]]:
    """Parse cluster_name_map uniformly from a dict, JSON string, or file path."""
    if not raw_name_map:
        return None
    if isinstance(raw_name_map, dict):
        return {str(k): str(v) for k, v in raw_name_map.items()}
    if isinstance(raw_name_map, str):
        p_map = Path(raw_name_map)
        if p_map.is_file():
            loaded = json.loads(p_map.read_text(encoding="utf-8"))
        else:
            loaded = json.loads(raw_name_map)
        return {str(k): str(v) for k, v in loaded.items()}
    return None


def cmd_cohort(args: argparse.Namespace) -> int:
    """Execute Phase 4 Population Discovery & Phase 5 Quantitative Cohort Analysis."""
    try:
        from flowx.clustering.cohort_runner import run_cohort_population_analysis
        from flowx.io.config import PipelineConfig
        from flowx.io.parser import parse_fcs
        from flowx.gating.cleanup import CleanUpGateConfig, run_cleanup_gating
        from flowx.math.transforms import apply_transform_to_dataframe, get_optimal_transform

        config_obj: Optional[PipelineConfig] = None
        if getattr(args, "config", None):
            config_obj = PipelineConfig.load(args.config)

        # 1. Resolve targets (positional targets, files_list, metadata_file, or config)
        targets = getattr(args, "targets", [])
        files_list = getattr(args, "files_list", None)
        metadata_file = getattr(args, "metadata_file", None) or (config_obj.metadata_file if config_obj else None)

        file_paths: List[Path] = []
        if files_list:
            fl_path = Path(files_list).resolve()
            if not fl_path.is_file():
                print(f"Error: Files list not found: {args.files_list}", file=sys.stderr)
                return 1
            if fl_path.suffix.lower() == ".csv":
                df_fl = pd.read_csv(fl_path)
                col = next((c for c in df_fl.columns if c.lower() in ["filepath", "path", "file", "filename"]), None)
                if col:
                    for val in df_fl[col].dropna():
                        raw_str = str(val).strip()
                        if raw_str:
                            p = Path(raw_str)
                            if not p.is_absolute():
                                p = (fl_path.parent / p).resolve() if (fl_path.parent / p).is_file() else (Path.cwd() / p).resolve()
                            if p.is_file():
                                file_paths.append(p)
            else:
                for line in fl_path.read_text(encoding="utf-8").splitlines():
                    clean = line.strip()
                    if clean and not clean.startswith("#"):
                        p = Path(clean)
                        if not p.is_absolute():
                            p = (fl_path.parent / p).resolve() if (fl_path.parent / p).is_file() else (Path.cwd() / p).resolve()
                        if p.is_file():
                            file_paths.append(p)

        for t in targets:
            tp = Path(t).resolve()
            if tp.is_file():
                file_paths.append(tp)
            elif tp.is_dir():
                found_fcs = sorted(tp.glob("*.fcs")) + sorted(tp.glob("*.FCS"))
                if found_fcs:
                    file_paths.extend(found_fcs)
                else:
                    found_csv = sorted(tp.glob("*.csv"))
                    file_paths.extend(found_csv)

        if not file_paths and metadata_file:
            meta_path = Path(metadata_file).resolve()
            if meta_path.is_file() and meta_path.suffix.lower() == ".csv":
                df_meta = pd.read_csv(meta_path)
                col = next((c for c in df_meta.columns if c.lower() in ["filepath", "path", "file", "filename"]), None)
                if col:
                    for val in df_meta[col].dropna():
                        raw_str = str(val).strip()
                        if raw_str:
                            p = Path(raw_str)
                            if not p.is_absolute():
                                p = (meta_path.parent / p).resolve() if (meta_path.parent / p).is_file() else (Path.cwd() / p).resolve()
                            if p.is_file():
                                file_paths.append(p)

        # De-duplicate preserving order
        unique_paths: List[Path] = []
        seen = set()
        for p in file_paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        file_paths = unique_paths

        if not file_paths:
            print(
                "Error: No cohort files found. Please specify target FCS/CSV files or directory, --files-list, or --metadata-file.",
                file=sys.stderr,
            )
            return 1

        print(f"\n=======================================================")
        print(f" FlowX Cohort Population Discovery & Statistical Analysis")
        print(f" Loading {len(file_paths)} sample file(s)...")
        print(f"=======================================================\n")

        cohort_dfs: Dict[str, pd.DataFrame] = {}
        do_cleanup = getattr(args, "clean_up", False)
        cohort_marker_meta = None

        for fp in file_paths:
            sample_id = fp.stem
            if sample_id.endswith("_clean"):
                sample_id = sample_id[:-6]
            elif sample_id.endswith("_downsampled"):
                sample_id = sample_id[:-12]

            if fp.suffix.lower() == ".csv":
                df = pd.read_csv(fp)
                cohort_dfs[sample_id] = df
            elif fp.suffix.lower() in [".fcs", ".lmd"]:
                from flowx.gating.cleanup import CleanUpGateConfig
                from flowx.gating.runner import run_cleanup_step
                from flowx.math.compensation import run_compensation_step
                from flowx.math.transforms import TransformConfig, run_transformation_step

                fcs = parse_fcs(fp)
                if cohort_marker_meta is None:
                    cohort_marker_meta = _resolve_cli_marker_metadata(
                        args=args,
                        config_obj=config_obj,
                        dataset_dir=fp.parent,
                        fcs_data=fcs,
                    )
                comp_step = run_compensation_step(
                    fcs_data=fcs,
                    working_df=fcs.events_df,
                    enable_compensation=getattr(config_obj, "enable_compensation", True) if config_obj else True,
                    compensation_source=getattr(config_obj, "compensation_source", "auto") if config_obj else "auto",
                    technology=getattr(config_obj, "technology", "auto") if config_obj else "auto",
                    comp_controls_dir=getattr(config_obj, "comp_controls_dir", None) if config_obj else None,
                    comp_manifest=getattr(config_obj, "comp_manifest", None) if config_obj else None,
                    unstained_control=getattr(config_obj, "unstained_control", None) if config_obj else None,
                    comp_config=config_obj.to_compensation_config() if config_obj else None,
                    spectral_config=(
                        config_obj.to_unmixing_config()
                        if config_obj and getattr(config_obj, "technology", "auto") == "spectral"
                        else None
                    ),
                    generate_plots=False,
                    generate_nxn=False,
                )
                xform_cfg = config_obj.to_transform_config() if config_obj else TransformConfig()
                xform_step = run_transformation_step(
                    fcs_data=fcs,
                    df_events=comp_step.compensated_df,
                    config=xform_cfg,
                    marker_metadata=cohort_marker_meta,
                    generate_plot=False,
                )
                if do_cleanup:
                    c_cfg = config_obj.to_cleanup_config() if config_obj else CleanUpGateConfig()
                    c_step = run_cleanup_step(
                        fcs_data=fcs,
                        df_events=comp_step.compensated_df,
                        df_transformed=xform_step.transformed_df,
                        config=c_cfg,
                        marker_metadata=cohort_marker_meta,
                        generate_plots=False,
                    )
                    cohort_dfs[sample_id] = xform_step.transformed_df[c_step.cleanup_result.final_clean_mask].copy()
                else:
                    cohort_dfs[sample_id] = xform_step.transformed_df.copy()

        if not cohort_dfs:
            print("Error: Could not load event data from specified targets.", file=sys.stderr)
            return 1

        # Marker selection
        markers = config_obj.lineage_markers if config_obj else None
        if getattr(args, "markers", None):
            raw_m = args.markers.strip()
            if raw_m.lower() == "auto":
                markers = "auto"
            else:
                markers = [m.strip() for m in raw_m.split(",") if m.strip()]
        marker_roles = (
            [r.strip() for r in args.marker_roles.split(",") if r.strip()]
            if getattr(args, "marker_roles", None)
            else (config_obj.marker_roles if config_obj else None)
        )
        include_markers = (
            [m.strip() for m in args.include_markers.split(",") if m.strip()]
            if getattr(args, "include_markers", None)
            else (config_obj.include_markers if config_obj else None)
        )
        exclude_markers = (
            [m.strip() for m in args.exclude_markers.split(",") if m.strip()]
            if getattr(args, "exclude_markers", None)
            else (config_obj.exclude_markers if config_obj else None)
        )
        stats_granularity = (
            getattr(args, "stats_granularity", None)
            or (getattr(config_obj, "stats_granularity", None) if config_obj else None)
            or "populations"
        )
        stats_marker_roles = (
            [r.strip() for r in args.stats_marker_roles.split(",") if r.strip()]
            if getattr(args, "stats_marker_roles", None)
            else (getattr(config_obj, "stats_marker_roles", None) if config_obj else None)
        )

        # Cluster name map, markers metadata, & custom signatures
        cluster_name_map = _parse_cluster_name_map(
            getattr(args, "cluster_name_map", None) or (config_obj.cluster_name_map if config_obj else None)
        )
        markers_file = getattr(args, "markers_file", None) or (config_obj.markers_file if config_obj else None)
        cell_signatures = getattr(args, "cell_signatures", None) or (config_obj.cell_signatures if config_obj else None)

        clustering_method = getattr(args, "clustering_method", None) or (config_obj.clustering_method if config_obj else "som")
        n_clusters = args.n_clusters if getattr(args, "n_clusters", None) is not None else (config_obj.n_clusters if config_obj else 10)
        auto_k = (
            getattr(args, "auto_k")
            if getattr(args, "auto_k", None) is not None
            else (config_obj.auto_k if config_obj else True)
        )
        auto_k_method = getattr(args, "auto_k_method", None) or (config_obj.auto_k_method if config_obj else "consensus")
        k_min = getattr(args, "k_min", None) if getattr(args, "k_min", None) is not None else (config_obj.k_min if config_obj else 4)
        k_max = getattr(args, "k_max", None) if getattr(args, "k_max", None) is not None else (config_obj.k_max if config_obj else 20)
        consensus_subsample = (
            getattr(args, "consensus_subsample")
            if getattr(args, "consensus_subsample", None) is not None
            else (config_obj.consensus_subsample if config_obj else 0.8)
        )
        consensus_iterations = (
            getattr(args, "consensus_iterations")
            if getattr(args, "consensus_iterations", None) is not None
            else (config_obj.consensus_iterations if config_obj else 50)
        )
        compute_umap = getattr(args, "compute_umap", None) if getattr(args, "compute_umap", None) is not None else (config_obj.compute_umap if config_obj else True)
        umap_events = getattr(args, "umap_events_per_sample", None) if getattr(args, "umap_events_per_sample", None) is not None else (config_obj.umap_events_per_sample if config_obj else 1000)
        cond_col = getattr(args, "condition_column", None) or (config_obj.condition_column if config_obj else None)
        batch_col = getattr(args, "batch_column", None) or (config_obj.batch_column if config_obj else None)
        umap_covs_raw = getattr(args, "umap_covariates", None) or (config_obj.umap_covariates if config_obj else None)
        umap_covs = [c.strip() for c in umap_covs_raw.split(",") if c.strip()] if isinstance(umap_covs_raw, str) else umap_covs_raw
        ref_grp = getattr(args, "reference_group", None) or (config_obj.reference_group if config_obj else None)
        tgt_grp = getattr(args, "target_group", None) or (config_obj.target_group if config_obj else None)
        pair_col = getattr(args, "paired_column", None) or (config_obj.paired_column if config_obj else None)
        cluster_pal = getattr(args, "cluster_palette", None) or (getattr(config_obj, "cluster_palette", None) if config_obj else None) or "tab20"
        da_method_arg = getattr(args, "da_test_method", None) or (getattr(config_obj, "da_test_method", None) if config_obj else None) or "auto"
        ds_method_arg = getattr(args, "ds_test_method", None) or (getattr(config_obj, "ds_test_method", None) if config_obj else None) or "auto"
        fdr_arg = float(args.fdr_threshold) if getattr(args, "fdr_threshold", None) is not None else float(getattr(config_obj, "fdr_threshold", 0.05) if config_obj else 0.05)
        log2fc_arg = float(args.log2fc_threshold) if getattr(args, "log2fc_threshold", None) is not None else float(getattr(config_obj, "log2fc_threshold", 1.0) if config_obj else 1.0)
        export_prism_arg = getattr(args, "export_prism", None) if getattr(args, "export_prism", None) is not None else (getattr(config_obj, "export_prism", True) if config_obj else True)
        export_augmented_fcs = getattr(args, "export_augmented_fcs", None) if getattr(args, "export_augmented_fcs", None) is not None else (config_obj.export_augmented_fcs if config_obj else True)
        export_wsp_arg = getattr(args, "export_flowjo_wsp", None) if getattr(args, "export_flowjo_wsp", None) is not None else (getattr(config_obj, "export_flowjo_wsp", True) if config_obj else True)
        export_gml_arg = getattr(args, "export_gating_ml", None) if getattr(args, "export_gating_ml", None) is not None else (getattr(config_obj, "export_gating_ml", True) if config_obj else True)
        gen_miflow_arg = getattr(args, "generate_miflowcyt_report", None) if getattr(args, "generate_miflowcyt_report", None) is not None else (getattr(config_obj, "generate_miflowcyt_report", True) if config_obj else True)
        output_dir = getattr(args, "output_dir", None) or (config_obj.output_dir if config_obj else "outputs/cohort_population")
        seed = getattr(args, "seed", None) if getattr(args, "seed", None) is not None else (config_obj.seed if config_obj else 42)

        pop_res = run_cohort_population_analysis(
            cohort_dfs=cohort_dfs,
            sample_metadata=metadata_file,
            markers=markers,
            markers_file=markers_file,
            marker_roles=marker_roles,
            include_markers=include_markers,
            exclude_markers=exclude_markers,
            cell_signatures=cell_signatures,
            clustering_method=clustering_method,
            n_clusters=n_clusters,
            auto_k=auto_k,
            auto_k_method=auto_k_method,
            k_min=k_min,
            k_max=k_max,
            consensus_subsample=consensus_subsample,
            consensus_iterations=consensus_iterations,
            compute_umap=compute_umap,
            umap_events_per_sample=umap_events,
            condition_column=cond_col,
            batch_column=batch_col,
            umap_covariates=umap_covs,
            reference_group=ref_grp,
            target_group=tgt_grp,
            paired_column=pair_col,
            stats_granularity=stats_granularity,
            stats_marker_roles=stats_marker_roles,
            da_test_method=da_method_arg,
            ds_test_method=ds_method_arg,
            fdr_threshold=fdr_arg,
            log2fc_threshold=log2fc_arg,
            stratified_density_contours=(getattr(config_obj, "stratified_density_contours", True) if config_obj else True),
            generate_volcano_plot=(getattr(config_obj, "generate_volcano_plot", True) if config_obj else True),
            generate_forest_plot=(getattr(config_obj, "generate_forest_plot", True) if config_obj else True),
            generate_abundance_boxplots=(getattr(config_obj, "generate_abundance_boxplots", True) if config_obj else True),
            generate_ds_heatmap=(getattr(config_obj, "generate_ds_heatmap", True) if config_obj else True),
            export_prism=export_prism_arg,
            cluster_name_map=cluster_name_map,
            cluster_palette=cluster_pal,
            export_augmented_fcs=export_augmented_fcs,
            export_flowjo_wsp=export_wsp_arg,
            export_gating_ml=export_gml_arg,
            generate_miflowcyt_report=gen_miflow_arg,
            output_dir=output_dir,
            random_seed=seed,
            generate_plots=True,
            numbered_steps=True,
        )

        # Print structured console completion summary
        print("\n" + "=" * 65)
        print(" Cohort Population Discovery & Statistical Analysis Complete")
        print("=" * 65)
        print(f" Samples Processed:      {len(pop_res.samples)}")
        print(f" Clustering Method:      {pop_res.clustering_result.method.upper()} (Metaclusters: {pop_res.clustering_result.cluster_count})")
        if auto_k and hasattr(pop_res.clustering_result, "optimal_k"):
            print(f" Auto-k Selected Count:  {pop_res.clustering_result.optimal_k}")
        print(f" Lineage Populations:    {len(pop_res.feature_set.population_names)} ({', '.join(pop_res.feature_set.population_names)})")
        if pop_res.da_result is not None:
            sig_pops = pop_res.da_result.significant_populations
            print(f" DA Comparison:          {pop_res.da_result.target_group} vs {pop_res.da_result.reference_group}")
            print(f" DA Significant Shifts:  {len(sig_pops)} ({', '.join(sig_pops) if sig_pops else 'None at FDR < 0.05'})")
        if pop_res.ds_result is not None:
            print(f" DS Modulated Markers:   {len(pop_res.ds_result.significant_states)} significant population-marker pairs")
        print(f" Output Directory:       {sanitize_path(pop_res.output_dir)}")
        print(f" Generated Deliverables:")
        print(f"   - Figures:            {len(pop_res.generated_plots)} publication plots saved")
        print(f"   - Tables:             {len(pop_res.generated_tables)} audit tables saved")
        if export_augmented_fcs:
            print(f"   - Augmented FCS:      Saved to {sanitize_path(pop_res.output_dir)}/exports/augmented_fcs/")
        print(f"   - FlowJo & Gating-ML: Saved to {sanitize_path(pop_res.output_dir)}/exports/workspaces/")
        print(f"   - MIFlowCyt Report:   Saved to {sanitize_path(pop_res.output_dir)}/exports/reports/")
        print("=" * 65 + "\n")
        return 0

    except Exception as exc:
        print(f"Error in cohort analysis: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Execute full pipeline command."""
    try:
        from flowx.io.config import PipelineConfig

        config_obj: Optional[PipelineConfig] = None
        if getattr(args, "config", None):
            config_obj = PipelineConfig.load(args.config)

        # 1. Resolve target FCS file(s) (CLI fcs_file, --files-list, or config file)
        target_files: List[Path] = []
        fl_source = getattr(args, "files_list", None) or (getattr(config_obj, "files_list", None) if config_obj else None)
        if fl_source and not getattr(args, "fcs_file", None):
            if isinstance(fl_source, (list, tuple)):
                for raw_item in fl_source:
                    raw_str = str(raw_item).strip()
                    if not raw_str:
                        continue
                    p = Path(raw_str)
                    if not p.is_absolute():
                        p = (p if p.is_file() else (Path.cwd() / p)).resolve()
                    if p.is_file():
                        target_files.append(p)
                    else:
                        print(f"Warning: File from list not found: {raw_str}", file=sys.stderr)
            else:
                fl_path = Path(fl_source).resolve()
                if not fl_path.is_file():
                    print(f"Error: Files list not found: {fl_source}", file=sys.stderr)
                    return 1
                if fl_path.suffix.lower() == ".csv":
                    df_list = pd.read_csv(fl_path)
                    col = next((c for c in df_list.columns if c.lower() in ["relative_path", "filepath", "path", "file", "filename"]), None)
                    if col:
                        for val in df_list[col].dropna():
                            raw_str = str(val).strip()
                            if raw_str:
                                p = Path(raw_str)
                                if not p.is_absolute():
                                    if p.is_file():
                                        p = p.resolve()
                                    elif fl_path.parent and (fl_path.parent / p).is_file():
                                        p = (fl_path.parent / p).resolve()
                                    else:
                                        p = (Path.cwd() / p).resolve()
                                if p.is_file():
                                    target_files.append(p)
                                else:
                                    print(f"Warning: File from list not found: {raw_str}", file=sys.stderr)
                else:
                    for line in fl_path.read_text(encoding="utf-8").splitlines():
                        clean_line = line.strip()
                        if clean_line and not clean_line.startswith("#"):
                            p = Path(clean_line)
                            if not p.is_absolute():
                                if p.is_file():
                                    p = p.resolve()
                                elif fl_path.parent and (fl_path.parent / p).is_file():
                                    p = (fl_path.parent / p).resolve()
                                else:
                                    p = (Path.cwd() / p).resolve()
                            if p.is_file():
                                target_files.append(p)
                            else:
                                print(f"Warning: File from list not found: {clean_line}", file=sys.stderr)

        if not target_files:
            single_arg = (
                args.fcs_file
                or (config_obj.fcs_file if config_obj else None)
                or (getattr(config_obj, "data_dir", None) if config_obj else None)
            )
            if single_arg:
                target_files.append(Path(single_arg).resolve())

        if not target_files:
            print(
                "Error: No FCS file specified. Please provide an FCS file path, specify --files-list, or set 'fcs_file' in your config.",
                file=sys.stderr,
            )
            return 1

        # 2. Resolve options (explicit CLI flags take precedence over config file defaults)
        output_dir = args.output_dir if args.output_dir is not None else (config_obj.output_dir if config_obj else "outputs")
        seed = args.seed if args.seed is not None else (config_obj.seed if config_obj else 42)
        gating_mode = args.gating_mode if args.gating_mode is not None else (config_obj.gating_mode if config_obj else "som")
        cluster_parent = args.cluster_parent if getattr(args, "cluster_parent", None) is not None else (config_obj.cluster_parent if config_obj else "viable")
        qc_method = args.qc_method if getattr(args, "qc_method", None) is not None else (config_obj.qc_method if config_obj else "clustering")
        legend_mode = args.legend_mode if args.legend_mode is not None else (config_obj.legend_mode if config_obj else "centroid")

        raw_argv = sys.argv
        if "--adaptive-gating" in raw_argv:
            adaptive_gating = True
        elif config_obj is not None:
            adaptive_gating = config_obj.adaptive_gating
        else:
            adaptive_gating = False

        if getattr(args, "overwrite", False) or "--overwrite" in raw_argv:
            overwrite = True
        elif config_obj is not None:
            overwrite = config_obj.overwrite
        else:
            overwrite = False

        if "--no-timestamp" in raw_argv or overwrite:
            timestamp_dir = False
        elif config_obj is not None:
            timestamp_dir = config_obj.timestamp_dir
        else:
            timestamp_dir = True

        workspace_file = args.workspace if args.workspace is not None else (config_obj.workspace_file if config_obj else None)
        custom_hierarchy = None
        if workspace_file:
            ws_path = Path(workspace_file)
            if ws_path.suffix.lower() == ".xml":
                session = WorkspaceSession.load_xml(ws_path)
            else:
                session = WorkspaceSession.load_json(ws_path)
            custom_hierarchy = session.hierarchy

        # Parse optional marker mapping
        marker_map = None
        if getattr(args, "marker_mapping", None):
            map_str = args.marker_mapping.strip()
            if map_str.startswith("{") or Path(map_str).is_file():
                import json
                if Path(map_str).is_file():
                    marker_map = json.loads(Path(map_str).read_text(encoding="utf-8"))
                else:
                    marker_map = json.loads(map_str)
            else:
                marker_map = {}
                for pair in map_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        marker_map[k.strip()] = v.strip()
        elif config_obj and config_obj.marker_mapping:
            marker_map = config_obj.marker_mapping

        t0 = time.perf_counter()
        pipeline_log_level = getattr(args, "log_level", None) or (config_obj.log_level if config_obj else "INFO")

        # Render Biotech Laboratory Header Banner
        display_sample_name = target_files[0].name if len(target_files) == 1 else f"{len(target_files)} Cohort Samples"
        tech_arg = getattr(args, "technology", None) or (config_obj.technology if config_obj else "auto")
        render_banner(
            sample_name=display_sample_name,
            mode=gating_mode,
            instrument="FCS Header / Auto",
            technology=tech_arg.title() if tech_arg != "auto" else "Auto-detected",
        )

        # Compensation controls and source options
        comp_controls_dir = getattr(args, "comp_controls_dir", None) or (config_obj.comp_controls_dir if config_obj else None)
        comp_manifest = getattr(args, "comp_manifest", None) or (config_obj.comp_manifest if config_obj else None)
        unstained_control = getattr(args, "unstained_control", None) or (config_obj.unstained_control if config_obj else None)
        comp_source = getattr(args, "comp_source", None) or (config_obj.compensation_source if config_obj else "auto")
        enable_comp = (
            False
            if comp_source == "none"
            else (getattr(config_obj, "enable_compensation", True) if config_obj else True)
        )
        transform_type = getattr(args, "transform_type", None) or (config_obj.transform_type if config_obj else "logicle")
        enable_transform = (
            False
            if _flag_in_argv("--no-transform")
            else (getattr(config_obj, "enable_transformation", True) if config_obj else True)
        )
        transform_cofactor = (
            float(args.transform_cofactor)
            if getattr(args, "transform_cofactor", None) is not None
            else float(getattr(config_obj, "transform_cofactor", 5.0) if config_obj else 5.0)
        )
        enable_time_qc = (
            not getattr(args, "no_time_qc", False)
            if "--no-time-qc" in raw_argv
            else (config_obj.enable_time_qc if config_obj else True)
        )
        enable_cleanup = (
            False
            if _flag_in_argv("--no-cleanup")
            else (getattr(config_obj, "enable_cleanup_gating", True) if config_obj else True)
        )

        # Clean-up Phase 2 options
        debris_method = getattr(args, "debris_method", None) or (config_obj.debris_method if config_obj else "gmm")
        debris_n_comp = (
            int(args.debris_n_components)
            if getattr(args, "debris_n_components", None) is not None
            else int(getattr(config_obj, "debris_n_components", 3) if config_obj else 3)
        )
        debris_sd_scale = (
            float(getattr(args, "debris_sd_scale"))
            if getattr(args, "debris_sd_scale", None) is not None
            else (getattr(config_obj, "debris_sd_scale", 1.0) if config_obj else 1.0)
        )
        debris_min_probability = (
            float(getattr(args, "debris_min_probability"))
            if getattr(args, "debris_min_probability", None) is not None
            else (getattr(config_obj, "debris_min_probability", None) if config_obj else None)
        )
        singlet_mode = getattr(args, "singlet_mode", None) or (config_obj.singlet_mode if config_obj else "dual")
        singlet_method = getattr(args, "singlet_method", None) or (config_obj.singlet_method if config_obj else "gmm_ratio")
        singlet_tolerance = (
            float(getattr(args, "singlet_tolerance"))
            if getattr(args, "singlet_tolerance", None) is not None
            else (config_obj.singlet_tolerance if config_obj else 0.25)
        )
        viability_cutoff = (
            float(args.viability_cutoff)
            if getattr(args, "viability_cutoff", None) is not None
            else (config_obj.viability_cutoff if config_obj else None)
        )
        viability_looseness = (
            float(getattr(args, "viability_looseness"))
            if getattr(args, "viability_looseness", None) is not None
            else (getattr(config_obj, "viability_looseness", 1.0) if config_obj else 1.0)
        )
        dump_looseness = (
            float(getattr(args, "dump_looseness"))
            if getattr(args, "dump_looseness", None) is not None
            else (getattr(config_obj, "dump_looseness", 1.0) if config_obj else 1.0)
        )
        fsc_ch_arg = getattr(args, "fsc_channel", None) or (getattr(config_obj, "fsc_channel", None) if config_obj else None)
        ssc_ch_arg = getattr(args, "ssc_channel", None) or (getattr(config_obj, "ssc_channel", None) if config_obj else None)
        fsc_h_arg = getattr(args, "fsc_height_channel", None) or (getattr(config_obj, "fsc_height_channel", None) if config_obj else None)
        ssc_h_arg = getattr(args, "ssc_height_channel", None) or (getattr(config_obj, "ssc_height_channel", None) if config_obj else None)
        via_ch_arg = getattr(args, "viability_channel", None) or (getattr(config_obj, "viability_channel", None) if config_obj else None)
        dump_ch_arg = getattr(args, "dump_channel", None) or (getattr(config_obj, "dump_channel", None) if config_obj else None)
        dump_cutoff_arg = (
            float(args.dump_cutoff)
            if getattr(args, "dump_cutoff", None) is not None
            else (getattr(config_obj, "dump_cutoff", None) if config_obj else None)
        )
        dump_enabled_arg = bool(dump_ch_arg) or (getattr(config_obj, "dump_enabled", False) if config_obj else False)
        overwrite_autocheck = (
            bool(getattr(args, "overwrite_autocheck"))
            if getattr(args, "overwrite_autocheck", None) is not None
            else (getattr(config_obj, "overwrite_autocheck", False) if config_obj else False)
        )
        enable_viability = (
            not getattr(args, "no_viability", False)
            if "--no-viability" in raw_argv
            else (config_obj.enable_viability if config_obj else True)
        )
        export_clean_fcs = (
            getattr(args, "export_clean_fcs", False)
            if "--export-clean-fcs" in raw_argv
            else (config_obj.export_clean_fcs if config_obj else False)
        )
        enable_qc = (
            getattr(args, "enable_qc")
            if getattr(args, "enable_qc", None) is not None
            else (getattr(config_obj, "enable_qc", True) if config_obj else True)
        )
        enable_discovery = (
            getattr(args, "enable_discovery")
            if getattr(args, "enable_discovery", None) is not None
            else (getattr(config_obj, "enable_discovery", False) if config_obj else False)
        )
        numbered_steps = (
            getattr(args, "numbered_steps")
            if getattr(args, "numbered_steps", None) is not None
            else (getattr(config_obj, "numbered_steps", True) if config_obj else True)
        )

        meta_file_arg = getattr(args, "metadata_file", None) or (config_obj.metadata_file if config_obj else None)
        markers_file_arg = getattr(args, "markers_file", None) or (config_obj.markers_file if config_obj else None)
        cell_sigs_arg = getattr(args, "cell_signatures", None) or (config_obj.cell_signatures if config_obj else None)
        ref_group_arg = getattr(args, "reference_group", None) or (config_obj.reference_group if config_obj else None)
        tgt_group_arg = getattr(args, "target_group", None) or (config_obj.target_group if config_obj else None)
        paired_col_arg = getattr(args, "paired_column", None) or (config_obj.paired_column if config_obj else None)
        cl_name_map_arg = _parse_cluster_name_map(
            getattr(args, "cluster_name_map", None) or (getattr(config_obj, "cluster_name_map", None) if config_obj else None)
        )
        auto_k_arg = (
            getattr(args, "auto_k")
            if getattr(args, "auto_k", None) is not None
            else (config_obj.auto_k if config_obj else True)
        )
        auto_k_method_arg = getattr(args, "auto_k_method", None) or (config_obj.auto_k_method if config_obj else "consensus")
        k_min_arg = getattr(args, "k_min", None) if getattr(args, "k_min", None) is not None else (config_obj.k_min if config_obj else 4)
        k_max_arg = getattr(args, "k_max", None) if getattr(args, "k_max", None) is not None else (config_obj.k_max if config_obj else 20)
        cons_sub_arg = (
            getattr(args, "consensus_subsample")
            if getattr(args, "consensus_subsample", None) is not None
            else (config_obj.consensus_subsample if config_obj else 0.8)
        )
        cons_iter_arg = (
            getattr(args, "consensus_iterations")
            if getattr(args, "consensus_iterations", None) is not None
            else (config_obj.consensus_iterations if config_obj else 50)
        )

        # Build step configs merging config defaults and explicit CLI overrides
        from flowx.qc.models import TimeQCConfig
        from flowx.math.compensation import CompensationConfig
        from flowx.math.transforms import TransformConfig
        from flowx.gating.cleanup import CleanUpGateConfig

        tqc_cfg = config_obj.to_time_qc_config() if config_obj else TimeQCConfig.for_sample()
        tqc_bin_width_arg = (
            float(args.time_qc_bin_width)
            if getattr(args, "time_qc_bin_width", None) is not None
            else float(tqc_cfg.bin_width_sec)
        )
        if getattr(args, "time_qc_bin_width", None) is not None:
            tqc_cfg.bin_width_sec = tqc_bin_width_arg
        if getattr(args, "no_time_drift", False):
            tqc_cfg.enable_signal_drift = False
        if getattr(args, "no_time_margins", False):
            tqc_cfg.trim_margins = False

        comp_cfg = config_obj.to_compensation_config() if config_obj else CompensationConfig(source=comp_source, random_seed=seed)
        comp_cfg.source = comp_source
        comp_cfg.apply_time_qc_to_sample = enable_time_qc
        comp_cfg.apply_time_qc_to_controls = enable_time_qc
        if getattr(args, "comp_estimation_method", None):
            comp_cfg.estimation_method = args.comp_estimation_method
        if getattr(args, "voltage_mismatch_policy", None):
            comp_cfg.voltage_mismatch_policy = args.voltage_mismatch_policy
        if getattr(args, "voltage_tolerance", None) is not None:
            comp_cfg.voltage_tolerance_v = float(args.voltage_tolerance)

        use_spectral = tech_arg == "spectral" or comp_source == "spectral" or getattr(args, "unmix_method", None) is not None or getattr(args, "af_mode", None) is not None
        spectral_cfg = None
        if use_spectral or (config_obj is not None and config_obj.technology == "spectral"):
            from flowx.spectral.models import UnmixingConfig

            spectral_cfg = config_obj.to_unmixing_config() if config_obj else UnmixingConfig(random_seed=seed)
            if getattr(args, "unmix_method", None):
                spectral_cfg.method = args.unmix_method
            if getattr(args, "af_mode", None):
                spectral_cfg.autofluorescence_mode = args.af_mode

        xform_cfg = config_obj.to_transform_config() if config_obj else TransformConfig()
        xform_cfg.enabled = enable_transform
        xform_cfg.method = transform_type
        xform_cfg.cofactor = transform_cofactor
        if getattr(args, "transform_channels", None):
            raw_tc = args.transform_channels.strip()
            xform_cfg.channels = "auto" if raw_tc.lower() == "auto" else [c.strip() for c in raw_tc.split(",") if c.strip()]
        if getattr(args, "transform_exclude", None):
            xform_cfg.exclude_channels = [c.strip() for c in args.transform_exclude.split(",") if c.strip()]

        cleanup_cfg = config_obj.to_cleanup_config() if config_obj else CleanUpGateConfig()
        cleanup_cfg.debris_method = debris_method
        cleanup_cfg.debris_n_components = debris_n_comp
        cleanup_cfg.debris_sd_scale = debris_sd_scale
        cleanup_cfg.debris_min_probability = debris_min_probability
        cleanup_cfg.fsc_channel = fsc_ch_arg
        cleanup_cfg.ssc_channel = ssc_ch_arg
        cleanup_cfg.fsc_height_channel = fsc_h_arg
        cleanup_cfg.ssc_height_channel = ssc_h_arg
        cleanup_cfg.singlet_mode = singlet_mode
        cleanup_cfg.singlet_method = singlet_method
        cleanup_cfg.singlet_tolerance = singlet_tolerance
        cleanup_cfg.viability_enabled = enable_viability
        cleanup_cfg.viability_channel = via_ch_arg
        cleanup_cfg.viability_cutoff = viability_cutoff
        cleanup_cfg.viability_looseness = viability_looseness
        cleanup_cfg.dump_enabled = dump_enabled_arg
        cleanup_cfg.dump_channel = dump_ch_arg
        cleanup_cfg.dump_cutoff = dump_cutoff_arg
        cleanup_cfg.dump_looseness = dump_looseness

        enable_batch_prep = (
            False
            if _flag_in_argv("--no-batch-prep")
            else (getattr(config_obj, "enable_batch_preparation", True) if config_obj else True)
        )
        batch_prep_dict = dict(getattr(config_obj, "batch_preparation", {}) or {}) if config_obj else {}
        if getattr(args, "normalization_method", None):
            batch_prep_dict["normalization_method"] = args.normalization_method
        if getattr(args, "batch_goal", None):
            batch_prep_dict["goal"] = args.batch_goal
        if getattr(args, "downsample_mode", None):
            batch_prep_dict["mode"] = args.downsample_mode
        if getattr(args, "target_events", None) is not None:
            batch_prep_dict["target_events"] = int(args.target_events)
        if getattr(args, "no_batch_check", False):
            batch_prep_dict["assess_batch_effect"] = False

        lineage_markers_arg: Any = getattr(config_obj, "lineage_markers", None) if config_obj else None
        if getattr(args, "lineage_markers", None):
            raw_lm = args.lineage_markers.strip()
            lineage_markers_arg = "auto" if raw_lm.lower() == "auto" else [m.strip() for m in raw_lm.split(",") if m.strip()]

        da_method_arg = getattr(args, "da_test_method", None) or (getattr(config_obj, "da_test_method", "auto") if config_obj else "auto")
        ds_method_arg = getattr(args, "ds_test_method", None) or (getattr(config_obj, "ds_test_method", "auto") if config_obj else "auto")
        fdr_arg = float(args.fdr_threshold) if getattr(args, "fdr_threshold", None) is not None else float(getattr(config_obj, "fdr_threshold", 0.05) if config_obj else 0.05)
        log2fc_arg = float(args.log2fc_threshold) if getattr(args, "log2fc_threshold", None) is not None else float(getattr(config_obj, "log2fc_threshold", 1.0) if config_obj else 1.0)
        cluster_pal_arg = getattr(args, "cluster_palette", None) or (getattr(config_obj, "cluster_palette", "tab20") if config_obj else "tab20")
        export_prism_arg = getattr(args, "export_prism", None) if getattr(args, "export_prism", None) is not None else (getattr(config_obj, "export_prism", True) if config_obj else True)
        export_aug_arg = getattr(args, "export_augmented_fcs", None) if getattr(args, "export_augmented_fcs", None) is not None else (getattr(config_obj, "export_augmented_fcs", True) if config_obj else True)
        export_wsp_arg = getattr(args, "export_flowjo_wsp", None) if getattr(args, "export_flowjo_wsp", None) is not None else (getattr(config_obj, "export_flowjo_wsp", True) if config_obj else True)
        export_gml_arg = getattr(args, "export_gating_ml", None) if getattr(args, "export_gating_ml", None) is not None else (getattr(config_obj, "export_gating_ml", True) if config_obj else True)
        gen_miflow_arg = getattr(args, "generate_miflowcyt_report", None) if getattr(args, "generate_miflowcyt_report", None) is not None else (getattr(config_obj, "generate_miflowcyt_report", True) if config_obj else True)

        pipeline = FlowCytometryPipeline(
            output_dir=output_dir,
            random_seed=seed,
            timestamp_dir=timestamp_dir,
            overwrite=overwrite,
            gating_mode=gating_mode,
            cluster_parent=cluster_parent,
            qc_method=qc_method,
            legend_mode=legend_mode,
            adaptive_gating=adaptive_gating,
            custom_hierarchy=custom_hierarchy,
            marker_mapping=marker_map,
            comp_controls_dir=comp_controls_dir,
            comp_manifest=comp_manifest,
            unstained_control=unstained_control,
            compensation_source=comp_source,
            comp_config=comp_cfg,
            enable_time_qc=enable_time_qc,
            time_qc_config=tqc_cfg,
            config_file=getattr(args, "config", None),
            config_source=config_obj,
            log_level=pipeline_log_level,
            transform_type=transform_type,
            enable_transformation=enable_transform,
            transform_cofactor=transform_cofactor,
            transform_config=xform_cfg,
            enable_umap=getattr(args, "enable_umap", None) if getattr(args, "enable_umap", None) is not None else (config_obj.enable_umap if config_obj else True),
            debris_method=debris_method,
            debris_n_components=debris_n_comp,
            debris_sd_scale=debris_sd_scale,
            debris_min_probability=debris_min_probability,
            singlet_mode=singlet_mode,
            singlet_method=singlet_method,
            singlet_tolerance=singlet_tolerance,
            viability_channel=via_ch_arg,
            viability_cutoff=viability_cutoff,
            viability_looseness=viability_looseness,
            dump_channel=dump_ch_arg,
            dump_enabled=dump_enabled_arg,
            dump_looseness=dump_looseness,
            enable_viability=enable_viability,
            export_clean_fcs=export_clean_fcs,
            enable_qc=enable_qc,
            enable_discovery=enable_discovery,
            numbered_steps=numbered_steps,
            technology=tech_arg,
            metadata_file=meta_file_arg,
            markers_file=markers_file_arg,
            cell_signatures=cell_sigs_arg,
            clustering_method=getattr(args, "clustering_method", None) or (config_obj.clustering_method if config_obj else "som"),
            n_clusters=args.n_clusters if getattr(args, "n_clusters", None) is not None else (config_obj.n_clusters if config_obj else None),
            auto_k=auto_k_arg,
            auto_k_method=auto_k_method_arg,
            k_min=k_min_arg,
            k_max=k_max_arg,
            consensus_subsample=cons_sub_arg,
            consensus_iterations=cons_iter_arg,
            umap_events_per_sample=getattr(args, "umap_events_per_sample", None) if getattr(args, "umap_events_per_sample", None) is not None else (config_obj.umap_events_per_sample if config_obj else 1000),
            condition_column=getattr(args, "condition_column", None) or (config_obj.condition_column if config_obj else None),
            batch_column=getattr(args, "batch_column", None) or (config_obj.batch_column if config_obj else None),
            umap_covariates=(
                [c.strip() for c in getattr(args, "umap_covariates", "").split(",") if c.strip()]
                if getattr(args, "umap_covariates", None)
                else (config_obj.umap_covariates if config_obj else None)
            ),
            reference_group=ref_group_arg,
            target_group=tgt_group_arg,
            paired_column=paired_col_arg,
            da_test_method=da_method_arg,
            ds_test_method=ds_method_arg,
            fdr_threshold=fdr_arg,
            log2fc_threshold=log2fc_arg,
            stratified_density_contours=(getattr(config_obj, "stratified_density_contours", True) if config_obj else True),
            generate_forest_plot=(getattr(config_obj, "generate_forest_plot", True) if config_obj else True),
            generate_ds_heatmap=(getattr(config_obj, "generate_ds_heatmap", True) if config_obj else True),
            generate_abundance_boxplots=(getattr(config_obj, "generate_abundance_boxplots", True) if config_obj else True),
            generate_volcano_plot=(getattr(config_obj, "generate_volcano_plot", True) if config_obj else True),
            cluster_palette=cluster_pal_arg,
            export_prism=export_prism_arg,
            export_augmented_fcs=export_aug_arg,
            export_flowjo_wsp=export_wsp_arg,
            export_gating_ml=export_gml_arg,
            generate_miflowcyt_report=gen_miflow_arg,
            batch_preparation=batch_prep_dict,
            enable_batch_preparation=enable_batch_prep,
            overwrite_autocheck=overwrite_autocheck,
            cluster_name_map=cl_name_map_arg,
            enable_compensation=enable_comp,
            enable_cleanup_gating=enable_cleanup,
            cleanup_config=cleanup_cfg,
            lineage_markers=lineage_markers_arg,
            marker_roles=(
                [r.strip() for r in args.marker_roles.split(",") if r.strip()]
                if getattr(args, "marker_roles", None)
                else (getattr(config_obj, "marker_roles", None) if config_obj else None)
            ),
            include_markers=(
                [m.strip() for m in args.include_markers.split(",") if m.strip()]
                if getattr(args, "include_markers", None)
                else (getattr(config_obj, "include_markers", None) if config_obj else None)
            ),
            exclude_markers=(
                [m.strip() for m in args.exclude_markers.split(",") if m.strip()]
                if getattr(args, "exclude_markers", None)
                else (getattr(config_obj, "exclude_markers", None) if config_obj else None)
            ),
            stats_granularity=(
                getattr(args, "stats_granularity", None)
                or (getattr(config_obj, "stats_granularity", None) if config_obj else None)
                or "populations"
            ),
            stats_marker_roles=(
                [r.strip() for r in args.stats_marker_roles.split(",") if r.strip()]
                if getattr(args, "stats_marker_roles", None)
                else (getattr(config_obj, "stats_marker_roles", None) if config_obj else None)
            ),
            spectral_config=spectral_cfg,
        )

        run_cohort_flag = getattr(args, "run_cohort", False) or (meta_file_arg is not None)
        if len(target_files) > 1 and (run_cohort_flag or enable_discovery or enable_batch_prep):
            with maybe_progress(progress_enabled(pipeline_log_level)):
                res = pipeline.run(input_path=[str(tf) for tf in target_files])

            # Render artifacts panel and success footer (population results are kept in report files only)
            render_artifacts_panel(res)
            render_success_footer(
                duration=time.perf_counter() - t0,
                log_path=res.log_file_path or "",
            )
            return 0
        elif len(target_files) > 1:
            success_count = 0
            failed_count = 0
            cohort_clean_dfs: Dict[str, pd.DataFrame] = {}
            use_bar = progress_enabled(pipeline_log_level)
            with maybe_progress(use_bar, n_samples=len(target_files), with_cohort_stage=False) as prog:
                if prog is None:
                    print(f"\n[Batch Processing] Initiating independent sample pipeline for {len(target_files)} samples from files list...")
                for idx, tf in enumerate(target_files):
                    if prog is not None:
                        prog.begin_sample(idx, tf.name)
                    else:
                        print(f"\n=======================================================")
                        print(f" Processing sample [{idx+1}/{len(target_files)}]: {tf.name}")
                        print(f"=======================================================\n")
                    try:
                        s_res = pipeline.run(fcs_filepath=tf)
                        success_count += 1
                        s_name = s_res.sample_name
                        if s_res.cleanup_result is not None and s_res.cleanup_result.final_clean_mask is not None:
                            cohort_clean_dfs[s_name] = s_res.transformed_df[s_res.cleanup_result.final_clean_mask].copy()
                        else:
                            cohort_clean_dfs[s_name] = s_res.transformed_df.copy()
                    except Exception as exc:
                        print(f"Error processing {tf.name}: {exc}", file=sys.stderr)
                        failed_count += 1
                if prog is not None:
                    prog.begin_cohort_stage() if (run_cohort_flag and len(cohort_clean_dfs) >= 2) else None
                elapsed_batch = time.perf_counter() - t0
                if prog is None:
                    print(f"\n[Batch Processing Complete] {success_count} succeeded, {failed_count} failed out of {len(target_files)} samples in {format_duration(elapsed_batch)} ({elapsed_batch:.2f}s).")
                if run_cohort_flag and len(cohort_clean_dfs) >= 2:
                    if prog is None:
                        print(f"\n[Cohort Analysis] Executing Phase 4 Population Discovery & Phase 5 Statistical Analysis across {len(cohort_clean_dfs)} samples...")
                    from flowx.clustering.cohort_runner import run_cohort_population_analysis
                    cohort_dir = Path(output_dir) / "cohort_population"
                    run_cohort_population_analysis(
                        cohort_dfs=cohort_clean_dfs,
                        sample_metadata=meta_file_arg,
                        markers_file=markers_file_arg,
                        cell_signatures=cell_sigs_arg,
                        clustering_method=getattr(args, "clustering_method", None) or (config_obj.clustering_method if config_obj else "som"),
                        n_clusters=args.n_clusters if getattr(args, "n_clusters", None) is not None else (config_obj.n_clusters if config_obj else None),
                        auto_k=auto_k_arg,
                        auto_k_method=auto_k_method_arg,
                        k_min=k_min_arg,
                        k_max=k_max_arg,
                        consensus_subsample=cons_sub_arg,
                        consensus_iterations=cons_iter_arg,
                        compute_umap=getattr(args, "enable_umap", None) if getattr(args, "enable_umap", None) is not None else (config_obj.enable_umap if config_obj else True),
                        condition_column=getattr(args, "condition_column", None) or (config_obj.condition_column if config_obj else None),
                        batch_column=getattr(args, "batch_column", None) or (config_obj.batch_column if config_obj else None),
                        umap_covariates=(
                            [c.strip() for c in getattr(args, "umap_covariates", "").split(",") if c.strip()]
                            if getattr(args, "umap_covariates", None)
                            else (config_obj.umap_covariates if config_obj else None)
                        ),
                        reference_group=ref_group_arg,
                        target_group=tgt_group_arg,
                        paired_column=paired_col_arg,
                        stats_granularity=(
                            getattr(args, "stats_granularity", None)
                            or (getattr(config_obj, "stats_granularity", None) if config_obj else None)
                            or "populations"
                        ),
                        stats_marker_roles=(
                            [r.strip() for r in args.stats_marker_roles.split(",") if r.strip()]
                            if getattr(args, "stats_marker_roles", None)
                            else (getattr(config_obj, "stats_marker_roles", None) if config_obj else None)
                        ),
                        marker_roles=(
                            [r.strip() for r in args.marker_roles.split(",") if r.strip()]
                            if getattr(args, "marker_roles", None)
                            else (getattr(config_obj, "marker_roles", None) if config_obj else None)
                        ),
                        include_markers=(
                            [m.strip() for m in args.include_markers.split(",") if m.strip()]
                            if getattr(args, "include_markers", None)
                            else (getattr(config_obj, "include_markers", None) if config_obj else None)
                        ),
                        exclude_markers=(
                            [m.strip() for m in args.exclude_markers.split(",") if m.strip()]
                            if getattr(args, "exclude_markers", None)
                            else (getattr(config_obj, "exclude_markers", None) if config_obj else None)
                        ),
                        cluster_name_map=cl_name_map_arg,
                        cluster_palette=getattr(args, "cluster_palette", None) or (getattr(config_obj, "cluster_palette", None) if config_obj else None) or "tab20",
                        export_augmented_fcs=getattr(args, "export_augmented_fcs", None) if getattr(args, "export_augmented_fcs", None) is not None else (config_obj.export_augmented_fcs if config_obj else True),
                        output_dir=cohort_dir,
                        random_seed=seed,
                    )
                    if prog is None:
                        print(f"[Cohort Analysis Complete] Deliverables saved to {sanitize_path(cohort_dir)}/")

            elapsed_batch = time.perf_counter() - t0
            if use_bar:
                get_console().print(
                    f"[{COLOR_TINT_1}]✓[/] {success_count} succeeded, {failed_count} failed "
                    f"out of {len(target_files)} samples in {format_duration(time.perf_counter() - t0)}."
                    f"out of {len(target_files)} samples in {format_duration(elapsed_batch)}."
                )
            else:
                print(f"\n[Batch Processing Complete] {success_count} succeeded, {failed_count} failed out of {len(target_files)} samples in {format_duration(elapsed_batch)} ({elapsed_batch:.2f}s).")
            return 0 if failed_count == 0 else 1
        else:
            with maybe_progress(progress_enabled(pipeline_log_level)):
                res = pipeline.run(fcs_filepath=target_files[0])

            # Render artifacts panel and success footer (population results are kept in report files only)
            render_artifacts_panel(res)
            render_success_footer(
                duration=time.perf_counter() - t0,
                log_path=res.log_file_path or "",
            )
            return 0
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_replay(args: argparse.Namespace) -> int:
    """Execute replay command."""
    try:
        t0 = time.perf_counter()
        ws_path = Path(args.workspace_file)
        if ws_path.suffix.lower() == ".xml":
            session = WorkspaceSession.load_xml(ws_path)
        else:
            session = WorkspaceSession.load_json(ws_path)

        fcs_path = args.fcs_file or session.sample_path
        fcs_data = parse_fcs(fcs_path)
        _, stats = session.replay(fcs_data)
        elapsed = time.perf_counter() - t0

        print("=" * 60)
        print(f"REPLAY RESULTS: {session.name} (completed in {format_duration(elapsed)} / {elapsed:.2f}s)")
        print("=" * 60)
        print(stats.to_string(index=False))
        print("=" * 60)
        return 0
    except Exception as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Execute config command."""
    from flowx.io.config import PipelineConfig

    if args.template:
        out_path = Path(args.template)
        content = PipelineConfig.generate_template(format=args.format)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(f"Generated {args.format.upper()} configuration template at: {sanitize_path(out_path)}")
        return 0

    if args.validate:
        try:
            cfg = PipelineConfig.load(args.validate)
            print(f"Configuration file '{sanitize_path(args.validate)}' is valid!")
            print(f"  - FCS File:        {sanitize_path(cfg.fcs_file) if cfg.fcs_file else '(Not specified, supply via CLI)'}")
            print(f"  - Metadata File:   {sanitize_path(cfg.metadata_file) if cfg.metadata_file else '(Auto-discover metadata/samples.csv)'}")
            print(f"  - Markers File:    {sanitize_path(cfg.markers_file) if cfg.markers_file else '(Auto-discover metadata/markers.csv)'}")
            print(f"  - Gating Mode:     {cfg.gating_mode}")
            print(f"  - Clustering:      {cfg.clustering_method} (k={cfg.n_clusters}, auto_k={cfg.auto_k}, method={cfg.auto_k_method})")
            print(f"  - Adaptive Gating: {cfg.adaptive_gating}")
            print(f"  - Output Dir:      {sanitize_path(cfg.output_dir)}")
            return 0
        except Exception as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 1

    print("Please specify --template <filepath> or --validate <filepath>.", file=sys.stderr)
    return 1


def cmd_unmix(args: argparse.Namespace) -> int:
    """Execute spectral unmixing deconvolution command."""
    from flowx.spectral.models import UnmixingConfig
    from flowx.spectral.runner import run_spectral_unmixing_pipeline

    config_obj = _load_cli_config(args)
    cfg = config_obj.to_unmixing_config() if config_obj is not None else UnmixingConfig()
    if _flag_in_argv("--method"):
        cfg.method = args.method
    if _flag_in_argv("--alpha-poisson"):
        cfg.alpha_poisson = float(args.alpha_poisson)
    if _flag_in_argv("--electronic-noise"):
        cfg.electronic_noise = float(args.electronic_noise)
    if _flag_in_argv("--af-mode"):
        cfg.autofluorescence_mode = args.af_mode
    if _flag_in_argv("--multi-af-components"):
        cfg.multi_af_components = int(args.multi_af_components)

    sample_input = (
        getattr(args, "sample", None)
        or (config_obj.fcs_file if config_obj else None)
        or (getattr(config_obj, "data_dir", None) if config_obj else None)
    )
    if not sample_input:
        print("Error: Must specify --sample / -s or provide --config with input.fcs_file.", file=sys.stderr)
        return 1

    controls_dir = getattr(args, "controls_dir", None) or (config_obj.comp_controls_dir if config_obj else None)
    unstained_ctrl = getattr(args, "unstained", None) or (config_obj.unstained_control if config_obj else None)
    out_dir = getattr(args, "output_dir", "outputs/spectral_unmixing")
    if not _flag_in_argv("-o", "--output-dir") and config_obj is not None and getattr(config_obj, "output_dir", None):
        out_dir = str(Path(config_obj.output_dir) / "02_spectral_unmixing")

    try:
        res = run_spectral_unmixing_pipeline(
            sample_input=sample_input,
            controls_dir_or_files=controls_dir,
            unstained_control=unstained_ctrl,
            output_dir=out_dir,
            config=cfg,
        )
        print(f"Spectral unmixing successful for '{res.sample_name}':")
        print(
            f"  - Parameters unmixed: {res.spectral_library.fluorophores_count} "
            f"({len(res.unmixing_result.fluorophore_channels)} markers + {len(res.unmixing_result.af_channels)} AF)"
        )
        print(f"  - Solver:             {res.unmixing_result.method_used.upper()}")
        print(f"  - Complexity Index:   {res.metrics_result.complexity_index:.2f}")
        print(f"  - Matrix Condition:   {res.spectral_library.condition_number:.2f}")
        print(f"  - Median Residual:    {res.unmixing_result.median_residual_norm:.2f}")
        print(f"  - Unmixed FCS:        {res.unmixed_fcs_path}")
        print(f"  - Report:             {res.report_md_path}")
        return 0
    except Exception as exc:
        logger.exception("Spectral unmixing failed")
        print(f"Error during spectral unmixing: {exc}", file=sys.stderr)
        return 1


def cmd_download_dataset(args: argparse.Namespace) -> int:
    """List, inspect, verify, or download benchmark conventional and spectral datasets."""
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from flowx.ui.console import get_console

    console = get_console()
    data_dir = Path(getattr(args, "data_dir", "data")).resolve()

    # 1. Resolve category filter
    category = getattr(args, "category", "all")
    if category in ("*", "all"):
        category = None

    # 2. Check if --list requested or no dataset target specified and no --verify-only
    raw_targets = getattr(args, "datasets", [])
    show_list = getattr(args, "list", False)
    show_info = getattr(args, "info", False)
    verify_only = getattr(args, "verify_only", False)
    include_supplementary = getattr(args, "include_supplementary", False)
    as_json = getattr(args, "json", False)

    # If --list flag or (no targets provided and not verify-only and not info)
    if show_list or (not raw_targets and not verify_only and not show_info):
        datasets = list_datasets(category)
        verifications = {ds.id: verify_dataset(ds, data_dir) for ds in datasets}

        if as_json:
            out_data = [
                {**ds.__dict__, "verification": verifications[ds.id].summary_dict()}
                for ds in datasets
            ]
            print(json.dumps(out_data, indent=2))
            return 0

        render_datasets_table(datasets, verifications, console=console)
        return 0

    # 3. If no targets provided but --verify-only requested, verify all in category
    if not raw_targets and verify_only:
        targets_to_process = list_datasets(category)
    else:
        # Resolve target query
        targets_to_process = resolve_datasets(raw_targets, category=category)

    if not targets_to_process:
        console.print(
            f"[bold red]Error:[/bold red] No matching benchmark datasets found for: [bold white]{', '.join(raw_targets)}[/bold white]"
        )
        console.print("[dim]Use [bold #ff6600]flowx download --list[/bold #ff6600] to see available datasets.[/dim]")
        return 1

    # 4. If --info requested: show detailed specification cards
    if show_info:
        verifications = {ds.id: verify_dataset(ds, data_dir) for ds in targets_to_process}
        if as_json:
            out_data = [
                {**ds.__dict__, "verification": verifications[ds.id].summary_dict()}
                for ds in targets_to_process
            ]
            print(json.dumps(out_data, indent=2))
            return 0

        for ds in targets_to_process:
            render_dataset_card(ds, verifications.get(ds.id), console=console)
        return 0

    # 5. Process download / verification
    mode_label = "Verification" if verify_only else "Download & Verification"
    console.print(
        f"\n[bold #ff6600]◈ FlowX Dataset Manager[/bold #ff6600] │ "
        f"[bold white]{mode_label}[/bold white] for [bold #ff6600]{len(targets_to_process)}[/bold #ff6600] dataset(s)...\n"
    )

    results: Dict[str, bool] = {}
    for ds in targets_to_process:
        console.print(f"[#ff6600]▶ Processing [bold white]{ds.id}[/bold white] ({ds.category})...[/#ff6600]")
        success = download_dataset(
            ds,
            data_dir,
            verify_only=verify_only,
            include_supplementary=include_supplementary,
        )
        results[ds.id] = success

    # Update summary files only during download/write operations (never clobber in --verify-only)
    if not verify_only:
        update_catalog_summaries(data_dir)

    # Post-check verification
    verifications = {ds.id: verify_dataset(ds, data_dir) for ds in targets_to_process}

    if as_json:
        out_data = {
            ds.id: {
                "success": results.get(ds.id, False),
                "verification": verifications[ds.id].summary_dict(),
            }
            for ds in targets_to_process
        }
        print(json.dumps(out_data, indent=2))
        return 0

    render_datasets_table(targets_to_process, verifications, console=console)

    all_succeeded = all(results.values())
    status_title = "[bold #ff9955]✓ Dataset Operation Completed Successfully[/bold #ff9955]" if all_succeeded else "[bold yellow]⚠ Completed With Some Warnings[/bold yellow]"
    border_style = "#ff9955" if all_succeeded else "yellow"

    summary_panel = Panel(
        Text.from_markup(
            f"• [dim white]Datasets Processed:[/dim white] [bold white]{len(targets_to_process)}[/bold white]\n"
            f"• [dim white]Operation Mode:[/dim white]     [#ff6600]{mode_label}[/#ff6600]\n"
            f"• [dim white]Storage Directory:[/dim white]  [bold white]{sanitize_path(data_dir)}[/bold white]\n"
            f"• [dim white]Ready For Pipeline:[/dim white] [bold #ff9955]{sum(1 for v in verifications.values() if v.status == 'Ready')}[/bold #ff9955] / {len(targets_to_process)}"
        ),
        title=status_title,
        border_style="#444444" if all_succeeded else "yellow",
        box=box.ROUNDED,
        width=min(80, max(40, getattr(console.size, "width", 80) or 80)),
    )
    console.print(summary_panel)
    return 0 if all_succeeded else 1


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        render_cli_home()
        return 0

    dispatch = {
        "inspect": cmd_inspect,
        "info": cmd_info,
        "plot": cmd_plot,
        "time-qc": cmd_time_qc,
        "qc": cmd_time_qc,
        "compensate": cmd_compensate,
        "unmix": cmd_unmix,
        "transform": cmd_transform,
        "cleanup": cmd_cleanup,
        "batch-prepare": cmd_batch_prepare,
        "batch": cmd_batch_prepare,
        "normalize": cmd_batch_prepare,
        "batch-check": cmd_batch_check,
        "check-batch": cmd_batch_check,
        "cohort": cmd_cohort,
        "population": cmd_cohort,
        "cluster-stats": cmd_cohort,
        "pipeline": cmd_pipeline,
        "replay": cmd_replay,
        "config": cmd_config,
        "download": cmd_download_dataset,
        "dataset": cmd_download_dataset,
    }

    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
