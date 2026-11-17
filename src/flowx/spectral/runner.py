"""High-level runner and orchestrator for full-spectrum flow cytometry workflows.

Coordinates reference spectral library assembly, single/multi autofluorescence extraction,
spectral deconvolution (WLS/OLS), panel metrics evaluation, unmixed FCS export, and
diagnostic figure generation.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from flowx.io.parser import FCSData, parse_fcs
from flowx.spectral.autofluorescence import (
    attach_autofluorescence_to_library,
    extract_autofluorescence_spectrum,
    extract_multi_autofluorescence_spectra,
)
from flowx.spectral.library import (
    ReferenceSpectralLibraryBuilder,
    filter_optical_channels,
)
from flowx.spectral.metrics import evaluate_spectral_panel
from flowx.spectral.models import (
    ReferenceSpectralLibrary,
    SpectralMetricsResult,
    UnmixingConfig,
    UnmixingResult,
)
from flowx.spectral.unmixing import unmix_dataframe
from flowx.spectral.visualization import (
    plot_deconvolution_goodness_of_fit,
    plot_spectral_ribbons,
    plot_spectral_similarity_heatmap,
    plot_unmixing_residuals_distribution,
)

logger = logging.getLogger(__name__)


@dataclass
class SpectralPipelineResult:
    """Container holding results, paths, and diagnostics from an unmixing pipeline run."""

    sample_name: str
    raw_fcs: FCSData
    spectral_library: ReferenceSpectralLibrary
    metrics_result: SpectralMetricsResult
    unmixing_result: UnmixingResult
    unmixed_fcs_path: str = ""
    report_md_path: str = ""
    spill_matrix_csv_path: Optional[str] = None
    generated_tables: Dict[str, str] = field(default_factory=dict)
    generated_plots: Dict[str, str] = field(default_factory=dict)
    execution_time_seconds: float = 0.0


def run_spectral_unmixing_pipeline(
    sample_input: Union[str, Path, FCSData],
    controls_dir_or_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    unstained_control: Optional[Union[str, Path, FCSData]] = None,
    output_dir: Union[str, Path] = "outputs/spectral_unmixing",
    tables_dir: Optional[Union[str, Path]] = None,
    plots_dir: Optional[Union[str, Path]] = None,
    config: Optional[UnmixingConfig] = None,
    control_mapping: Optional[Dict[str, str]] = None,
) -> SpectralPipelineResult:
    """Execute end-to-end spectral flow cytometry deconvolution workflow.

    Args:
        sample_input: Filepath to raw spectral sample FCS or parsed FCSData.
        controls_dir_or_files: Directory containing single-stain controls or explicit list of files.
        unstained_control: Optional unstained control for autofluorescence extraction.
            If None and sample_input is unstained or has biological AF, extracts from sample.
        output_dir: Output directory for unmixed FCS, diagnostic plots, and reports.
        tables_dir: Optional directory for CSV matrix/SSM exports (defaults to ``output_dir/tables``).
        plots_dir: Optional directory for diagnostic plots (defaults to ``output_dir/plots``).
        config: Optional UnmixingConfig specifying solver, noise model, and AF mode.
        control_mapping: Optional dict mapping fluorophore name to control file.

    Returns:
        SpectralPipelineResult container with unmixed data, metrics, and generated artifacts.
    """
    t_start = time.time()
    cfg = config or UnmixingConfig()
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    t_dir = Path(tables_dir) if tables_dir is not None else (out_dir_path / "tables")
    t_dir.mkdir(parents=True, exist_ok=True)
    p_dir = Path(plots_dir) if plots_dir is not None else (out_dir_path / "plots")
    p_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(sample_input, (str, Path)):
        sample_fcs = parse_fcs(sample_input)
    else:
        sample_fcs = sample_input

    sample_name = sample_fcs.sample_name
    optical_channels = filter_optical_channels(sample_fcs.pnn_labels)
    logger.info(
        "Initiating spectral unmixing for '%s' (%d events, %d optical channels)",
        sample_name,
        sample_fcs.event_count,
        len(optical_channels),
    )

    # 1. Build Reference Spectral Library from single-stained controls
    builder = ReferenceSpectralLibraryBuilder(optical_channels=optical_channels)

    if controls_dir_or_files is not None:
        if isinstance(controls_dir_or_files, (str, Path)) and Path(controls_dir_or_files).is_dir():
            builder.add_controls_directory(controls_dir_or_files, file_mapping=control_mapping)
        elif isinstance(controls_dir_or_files, (list, tuple)):
            for f_p in controls_dir_or_files:
                f_path = Path(f_p)
                if any(w in f_path.name.lower() for w in ["unstained", "blank"]):
                    continue
                fl_name = control_mapping.get(f_path.name) if control_mapping else None
                try:
                    builder.add_control_file(f_path, fluorophore_name=fl_name)
                except Exception as exc:
                    logger.warning("Could not add control %s: %s", f_path.name, exc)

    # 2. Extract Autofluorescence if configured
    af_refs = []
    if cfg.autofluorescence_mode != "none":
        af_source = unstained_control or sample_fcs
        try:
            if cfg.autofluorescence_mode == "multi":
                af_refs = extract_multi_autofluorescence_spectra(
                    af_source,
                    n_components=cfg.multi_af_components,
                    optical_channels=optical_channels,
                    random_seed=cfg.random_seed,
                )
            else:
                af_refs = [
                    extract_autofluorescence_spectrum(
                        af_source,
                        label="Autofluorescence",
                        optical_channels=optical_channels,
                    )
                ]
        except Exception as exc:
            logger.warning("Autofluorescence extraction notice: %s", exc)

    # Build base library
    if not builder.references:
        # Fall back to embedded unmixing / spillover matrix if present in FCS metadata
        from flowx.spectral.library import extract_embedded_unmixing_matrix
        from flowx.spectral.models import SpectralReference
        embedded = extract_embedded_unmixing_matrix(sample_fcs)
        if embedded is not None:
            logger.info("Reconstructing reference spectral library from embedded FCS matrix...")
            det_names = embedded.detectors
            spill_mat = embedded.matrix
            builder.optical_channels = det_names
            for j, ch in enumerate(det_names):
                builder.add_reference(SpectralReference(
                    fluorophore=ch,
                    carrier="beads",
                    channel_names=det_names,
                    normalized_spectrum=spill_mat[:, j] if spill_mat.shape[0] == len(det_names) else spill_mat[j, :],
                    raw_positive_mfi=np.zeros(len(det_names)),
                    raw_negative_mfi=np.zeros(len(det_names)),
                    peak_channel=ch,
                    is_autofluorescence=False,
                ))
        else:
            raise ValueError(
                f"No single-stained reference controls provided or found for '{sample_name}', "
                f"and raw sample contains no embedded unmixing matrix. "
                f"Please provide controls_dir_or_files pointing to single-stained reference controls."
            )

    library = builder.build()
    if af_refs:
        library = attach_autofluorescence_to_library(library, af_refs)

    logger.info(
        "Spectral library compiled: %d optical detectors -> %d parameters (condition κ=%.2f)",
        library.detectors_count,
        library.fluorophores_count,
        library.condition_number,
    )

    # 3. Evaluate Spectral Metrics & Export Unified Tables
    from flowx.math.compensation import export_compensation_tables

    metrics_res = evaluate_spectral_panel(library)
    unmix_matrix_df = library.compute_unmixing_dataframe()
    detector_spill_df = library.compute_detector_spillover_dataframe()
    tables_map = export_compensation_tables(
        t_dir,
        {
            "spectral_signature_matrix": library.dataframe,
            "spectral_unmixing_matrix": unmix_matrix_df,
            "spectral_similarity_matrix": metrics_res.similarity_matrix,
            "spillover_spreading_matrix_ssm": metrics_res.ssm_dataframe,
            "spillover_matrix_calculated": detector_spill_df,
        },
    )
    spill_matrix_csv = (
        tables_map.get("spillover_matrix_calculated")
        or tables_map.get("spectral_unmixing_matrix")
    )
    logger.info("Exported %d spectral compensation tables to %s", len(tables_map), t_dir)

    # 4. Perform Spectral Deconvolution
    unmix_res = unmix_dataframe(sample_fcs.events_df, library, config=cfg)

    # 5. Export Unmixed FCS File
    from flowx.io.parser import write_fcs

    unmixed_fcs_path = out_dir_path / f"{sample_name}_unmixed.fcs"
    unmixed_pnn = list(unmix_res.unmixed_df.columns)
    unmixed_pns = [
        sample_fcs.detector_to_marker.get(ch, ch) for ch in unmixed_pnn
    ]

    write_fcs(
        output_path=unmixed_fcs_path,
        events_df=unmix_res.unmixed_df,
        pnn_labels=unmixed_pnn,
        pns_labels=unmixed_pns,
        metadata=sample_fcs.metadata,
        channel_ranges=sample_fcs.channel_ranges,
        is_compensated=True,
    )
    logger.info("Exported unmixed FCS file: %s", unmixed_fcs_path.name)

    # 6. Generate Publication-Ready Visualizations
    plots_map: Dict[str, str] = {}

    p_ribbons = p_dir / f"{sample_name}_spectral_ribbons.png"
    plot_spectral_ribbons(library, output_path=p_ribbons)
    plots_map["spectral_ribbons"] = str(p_ribbons)

    p_sim = p_dir / f"{sample_name}_spectral_similarity_matrix.png"
    plot_spectral_similarity_heatmap(metrics_res, output_path=p_sim)
    plots_map["similarity_matrix"] = str(p_sim)

    p_gof = p_dir / f"{sample_name}_deconvolution_goodness_of_fit.png"
    plot_deconvolution_goodness_of_fit(unmix_res, output_path=p_gof)
    plots_map["deconvolution_goodness_of_fit"] = str(p_gof)

    try:
        from flowx.visualization.comp_plots import plot_nxn_compensation_matrix
        nxn_dir = out_dir_path / "nxn_matrices"
        nxn_samples_dir = nxn_dir / "samples"
        nxn_ctrls_dir = nxn_dir / "controls"
        nxn_samples_dir.mkdir(parents=True, exist_ok=True)
        nxn_ctrls_dir.mkdir(parents=True, exist_ok=True)

        plots_nxn_samples = p_dir / "nxn_matrices" / "samples"
        plots_nxn_ctrls = p_dir / "nxn_matrices" / "controls"
        plots_nxn_samples.mkdir(parents=True, exist_ok=True)
        plots_nxn_ctrls.mkdir(parents=True, exist_ok=True)

        p_nxn = nxn_samples_dir / f"{sample_name}_nxn_unmixing_matrix.png"
        plot_nxn_compensation_matrix(
            uncomp_df=sample_fcs.events_df,
            comp_df=unmix_res.unmixed_df,
            spectral_library=library,
            sample_name=sample_name,
            output_path=p_nxn,
            lower_triangle_only=True,
            auto_gate_sample=True,
            viable_only=True,
        )
        shutil.copy2(p_nxn, plots_nxn_samples / p_nxn.name)
        shutil.copy2(p_nxn, p_dir / f"{sample_name}_nxn_unmixing_matrix.png")
        plots_map["nxn_unmixing_matrix"] = str(p_nxn)
    except Exception as nxn_exc:
        logger.warning(f"NxN spectral unmixing matrix generation skipped: {nxn_exc}")

    # 7. Generate MIFlowCyt Unmixing Audit Report
    report_path = out_dir_path / f"{sample_name}_unmixing_report.md"
    _generate_unmixing_markdown_report(
        report_path=report_path,
        sample_name=sample_name,
        library=library,
        metrics=metrics_res,
        unmix_res=unmix_res,
        plots=plots_map,
    )

    total_time = time.time() - t_start
    return SpectralPipelineResult(
        sample_name=sample_name,
        raw_fcs=sample_fcs,
        spectral_library=library,
        metrics_result=metrics_res,
        unmixing_result=unmix_res,
        unmixed_fcs_path=str(unmixed_fcs_path),
        report_md_path=str(report_path),
        spill_matrix_csv_path=spill_matrix_csv,
        generated_tables=tables_map,
        generated_plots=plots_map,
        execution_time_seconds=total_time,
    )


def _generate_unmixing_markdown_report(
    report_path: Path,
    sample_name: str,
    library: ReferenceSpectralLibrary,
    metrics: SpectralMetricsResult,
    unmix_res: UnmixingResult,
    plots: Dict[str, str],
) -> None:
    """Write MIFlowCyt-compliant spectral unmixing diagnostic report."""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Spectral Flow Cytometry Deconvolution Report: {sample_name}\n\n")
        f.write("## 1. Executive Summary & Quality Indicators\n\n")
        f.write(f"- **Optical Channels (Detectors):** {library.detectors_count}\n")
        f.write(f"- **Unmixed Parameters:** {library.fluorophores_count} ({len(unmix_res.fluorophore_channels)} markers + {len(unmix_res.af_channels)} AF)\n")
        f.write(f"- **Deconvolution Solver:** {unmix_res.method_used.upper()}\n")
        f.write(f"- **Matrix Condition Number κ(M):** {library.condition_number:.2f}\n")
        f.write(f"- **Panel Complexity Index:** {metrics.complexity_index:.2f}\n")
        f.write(f"- **Median Unmixing Residual ||r||:** {unmix_res.median_residual_norm:.2f}\n")
        f.write(f"- **Residual Outlier Events:** {unmix_res.residual_outliers_count:,} ({unmix_res.residual_outliers_count / max(1, len(unmix_res.residuals_norm)) * 100:.2f}%)\n")
        f.write(f"- **Deconvolution Runtime:** {unmix_res.execution_time_seconds:.2f}s\n")
        if metrics.ssm_dataframe is not None:
            f.write("- **Spillover Spreading Matrix (SSM):** Computed per Nguyen et al. (2013) (`tables/spillover_spreading_matrix_ssm.csv`)\n")
        f.write("- **Exported Spectral Matrices:** `tables/spectral_signature_matrix.csv`, `tables/spectral_unmixing_matrix.csv`, `tables/spectral_similarity_matrix.csv`\n")
        f.write("\n")

        f.write("## 2. High Spectral Collinearity Warnings\n\n")
        if metrics.high_similarity_pairs:
            f.write("| Fluorophore 1 | Fluorophore 2 | Cosine Similarity | Status |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for d1, d2, sim in metrics.high_similarity_pairs:
                status = "CRITICAL (>0.95)" if sim >= 0.95 else "WARNING (>0.85)"
                f.write(f"| `{d1}` | `{d2}` | {sim:.3f} | {status} |\n")
        else:
            f.write("> **Clean Spectral Separation:** All fluorophore pairs exhibit cosine similarities < 0.85.\n")
        f.write("\n")

        f.write("## 3. Diagnostic Figures\n\n")
        for p_name, p_file in plots.items():
            f.write(f"### {p_name.replace('_', ' ').title()}\n")
            f.write(f"![{p_name}]({Path(p_file).name})\n\n")
