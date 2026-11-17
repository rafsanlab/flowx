"""Cohort batch runner for Core Clean-up Gating.

Processes cohorts of FCS files through the sequential clean-up cascade (Debris,
Dual Singlets, Viability, and Dump exclusion), producing cohort summary tables,
diagnostic figure galleries, and standardized clean FCS exports.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import flowx
from flowx.gating.cleanup import (
    CleanUpGateConfig,
    CleanUpGatingResult,
    matches_viability_dye,
    run_cleanup_gating,
)
from flowx.io.parser import FCSData, parse_fcs
from flowx.math.compensation import CompensationMatrix
from flowx.math.transforms import TransformConfig, run_transformation_step
from flowx.visualization.cleanup_plots import plot_cleanup_diagnostics

logger = logging.getLogger(__name__)


@dataclass
class CleanupStepResult:
    """Container holding results from an independent single-sample clean-up gating step."""

    cleanup_result: CleanUpGatingResult
    clean_df: pd.DataFrame
    plot_path: Optional[str] = None
    clean_fcs_path: Optional[str] = None


def _is_auto_val(val: Optional[str]) -> bool:
    """Return True if a channel config field is unset or explicitly set to 'auto'."""
    return val is None or str(val).strip().lower() == "auto"


def resolve_cleanup_channels_from_metadata(
    cfg: CleanUpGateConfig,
    available_columns: Any,
    marker_metadata: Optional[Any] = None,
    spectral_marker_map: Optional[Dict[str, str]] = None,
) -> CleanUpGateConfig:
    """Return a CleanUpGateConfig copy with scatter, viability, and dump channels resolved from metadata if unset or 'auto'."""
    from flowx.io.metadata import resolve_scatter_and_role_channels

    cols_list = list(getattr(available_columns, "pnn_labels", available_columns))
    cols_set = set(cols_list)

    meta_roles = resolve_scatter_and_role_channels(marker_metadata, cols_list)

    resolved_fsc = meta_roles["fsc_channel"] if _is_auto_val(cfg.fsc_channel) else cfg.fsc_channel
    resolved_ssc = meta_roles["ssc_channel"] if _is_auto_val(cfg.ssc_channel) else cfg.ssc_channel
    resolved_fsc_h = meta_roles["fsc_height_channel"] if _is_auto_val(cfg.fsc_height_channel) else cfg.fsc_height_channel
    resolved_ssc_h = meta_roles["ssc_height_channel"] if _is_auto_val(cfg.ssc_height_channel) else cfg.ssc_height_channel
    resolved_viab = meta_roles["viability_channel"] if _is_auto_val(cfg.viability_channel) else cfg.viability_channel
    dump_was_explicit_auto = isinstance(cfg.dump_channel, str) and cfg.dump_channel.strip().lower() == "auto"
    resolved_dump = meta_roles["dump_channel"] if _is_auto_val(cfg.dump_channel) else cfg.dump_channel
    resolved_dump_enabled = cfg.dump_enabled or bool(dump_was_explicit_auto and resolved_dump is not None)

    if resolved_viab is None and spectral_marker_map:
        resolved_viab = next(
            (
                fl_c
                for fl_c, mk_l in spectral_marker_map.items()
                if (matches_viability_dye(fl_c) or matches_viability_dye(mk_l)) and fl_c in cols_set
            ),
            None,
        )

    return replace(
        cfg,
        fsc_channel=resolved_fsc,
        ssc_channel=resolved_ssc,
        fsc_height_channel=resolved_fsc_h,
        ssc_height_channel=resolved_ssc_h,
        viability_channel=resolved_viab,
        dump_channel=resolved_dump,
        dump_enabled=resolved_dump_enabled,
    )


def run_cleanup_step(
    fcs_data: FCSData,
    df_events: pd.DataFrame,
    df_transformed: pd.DataFrame,
    config: Optional[CleanUpGateConfig] = None,
    marker_metadata: Optional[Any] = None,
    spectral_marker_map: Optional[Dict[str, str]] = None,
    output_plot_path: Optional[Union[str, Path]] = None,
    generate_plots: bool = True,
    export_clean_fcs: bool = False,
    clean_fcs_path: Optional[Union[str, Path]] = None,
    is_compensated: bool = True,
) -> CleanupStepResult:
    """Execute independent Core Clean-up Gating step on a single sample.

    Resolves marker roles from metadata if needed, executes `run_cleanup_gating`,
    renders diagnostic plots, and optionally exports clean FCS events.
    """
    base_cfg = config or CleanUpGateConfig()
    eff_cfg = resolve_cleanup_channels_from_metadata(
        base_cfg,
        available_columns=df_transformed.columns,
        marker_metadata=marker_metadata,
        spectral_marker_map=spectral_marker_map,
    )

    res = run_cleanup_gating(
        fcs_data=fcs_data,
        df_events=df_events,
        df_transformed=df_transformed,
        config=eff_cfg,
    )
    if res.final_clean_mask is not None:
        clean_df = df_transformed[res.final_clean_mask].copy()
    else:
        clean_df = df_transformed.copy()

    plot_str: Optional[str] = None
    if generate_plots and output_plot_path is not None:
        p_out = Path(output_plot_path)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        plot_cleanup_diagnostics(
            fcs_data=fcs_data,
            cleanup_result=res,
            df_events=df_events,
            df_transformed=df_transformed,
            output_path=p_out,
            show_cluster_shades=getattr(eff_cfg, "show_cluster_shades", True),
        )
        plot_str = str(p_out)

    fcs_out_str: Optional[str] = None
    if export_clean_fcs and clean_fcs_path is not None and res.clean_event_count > 0:
        f_out = Path(clean_fcs_path)
        f_out.parent.mkdir(parents=True, exist_ok=True)
        res.export_fcs(f_out, fcs_data=fcs_data, events_df=df_events, is_compensated=is_compensated)
        fcs_out_str = str(f_out)

    return CleanupStepResult(
        cleanup_result=res,
        clean_df=clean_df,
        plot_path=plot_str,
        clean_fcs_path=fcs_out_str,
    )


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for NumPy arrays and scalars."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run_cohort_cleanup(
    file_list_or_paths: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path] = "outputs/cleanup",
    config: Optional[CleanUpGateConfig] = None,
    generate_plots: bool = True,
    export_clean_fcs: bool = False,
    clean_fcs_dir: Optional[Union[str, Path]] = None,
    apply_embedded_compensation: bool = True,
    apply_optimal_transform: bool = True,
    compensation_matrix: Optional[Union[pd.DataFrame, np.ndarray, CompensationMatrix, str, Path]] = None,
    markers_file: Optional[Union[str, Path]] = None,
    marker_metadata: Optional[Any] = None,
    transform_config: Optional[TransformConfig] = None,
) -> Tuple[pd.DataFrame, Path, Path]:
    """Execute Core Clean-up Gating across a cohort of FCS files.

    Args:
        file_list_or_paths: Text/CSV file with paths, directory path, or list of FCS paths.
        output_dir: Destination folder for summary tables, reports, and plots.
        config: Optional CleanUpGateConfig configuration override.
        generate_plots: If True, renders multi-panel diagnostic figures.
        export_clean_fcs: If True, exports clean gated events to new .fcs files.
        clean_fcs_dir: Destination directory for clean .fcs files.
        apply_embedded_compensation: If True, applies embedded spillover matrix if present.
        apply_optimal_transform: If True, applies optimal Logicle scaling to fluorescent channels.
        compensation_matrix: Optional calculated spillover matrix from single-stained controls
            (Step 3) or CSV file path overriding embedded matrix.

    Returns:
        Tuple containing:
            - Cohort summary DataFrame.
            - Path to saved summary CSV.
            - Path to saved Markdown report.
    """
    out_dir = Path(output_dir).resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    hierarchies_dir = out_dir / "hierarchies"
    hierarchies_dir.mkdir(parents=True, exist_ok=True)

    resolved_clean_fcs_dir = Path(clean_fcs_dir).resolve() if clean_fcs_dir else out_dir / "processed_fcs"
    if export_clean_fcs:
        resolved_clean_fcs_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or CleanUpGateConfig()

    # Resolve FCS file list
    fcs_files: List[Path] = []
    if isinstance(file_list_or_paths, (str, Path)):
        p = Path(file_list_or_paths)
        if p.is_file() and p.suffix.lower() in [".txt", ".csv"]:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        fcs_files.append(Path(stripped))
        elif p.is_file() and p.suffix.lower() == ".fcs":
            fcs_files.append(p)
        elif p.is_dir():
            fcs_files.extend(sorted(p.glob("*.fcs")))
    elif isinstance(file_list_or_paths, list):
        for item in file_list_or_paths:
            fcs_files.append(Path(item))

    if not fcs_files:
        raise ValueError(f"No FCS files found to process from: {file_list_or_paths}")

    logger.info(f"Starting cohort clean-up gating across {len(fcs_files)} FCS files...")
    t0 = time.perf_counter()

    summary_rows: List[Dict[str, Any]] = []

    for file_path in tqdm(fcs_files, desc="Clean-up Gating", unit="file"):
        try:
            if not file_path.is_file():
                logger.warning(f"File not found: '{file_path}'. Skipping.")
                continue

            fcs_data = parse_fcs(file_path)
            events_df = fcs_data.events_df.copy()

            active_marker_meta = marker_metadata
            if active_marker_meta is None:
                from flowx.io.metadata import resolve_marker_metadata

                active_marker_meta = resolve_marker_metadata(
                    markers_file=markers_file,
                    dataset_dir=file_path.parent,
                    fcs_pnn=fcs_data.pnn_labels,
                    fcs_pns=fcs_data.pns_labels,
                )

            # Apply compensation (prioritizing control-calculated matrix, then embedded FCS matrix)
            comp_obj = None
            comp_source = "none"

            if compensation_matrix is not None:
                if isinstance(compensation_matrix, (str, Path)):
                    mat_path = Path(compensation_matrix)
                    if mat_path.is_file():
                        spill_df = pd.read_csv(mat_path, index_col=0)
                        comp_obj = CompensationMatrix(detectors=list(spill_df.columns), matrix=spill_df.to_numpy())
                        comp_source = f"file:{mat_path.name}"
                elif isinstance(compensation_matrix, CompensationMatrix):
                    comp_obj = compensation_matrix
                    comp_source = "external_matrix"
                elif isinstance(compensation_matrix, pd.DataFrame):
                    comp_obj = CompensationMatrix(detectors=list(compensation_matrix.columns), matrix=compensation_matrix.to_numpy())
                    comp_source = "external_dataframe"
            elif apply_embedded_compensation and fcs_data.spillover_matrix is not None:
                try:
                    comp_obj = CompensationMatrix(
                        detectors=list(fcs_data.spillover_matrix.columns),
                        matrix=fcs_data.spillover_matrix.to_numpy(),
                    )
                    comp_source = "embedded_fcs"
                except Exception as comp_err:
                    logger.warning(f"Failed to parse embedded spillover matrix in {file_path.name}: {comp_err}")

            if comp_obj is not None:
                try:
                    events_df = comp_obj.apply_dataframe(events_df)
                except Exception as comp_err:
                    logger.warning(f"Error applying compensation to {file_path.name}: {comp_err}")

            # Apply transformation to fluorescent channels
            if apply_optimal_transform:
                try:
                    xf_res = run_transformation_step(
                        fcs_data=fcs_data,
                        df_events=events_df,
                        config=transform_config or TransformConfig(enabled=True, method="logicle"),
                        marker_metadata=active_marker_meta,
                        generate_plot=False,
                    )
                    transformed_df = xf_res.transformed_df
                except Exception as xf_err:
                    logger.debug(f"Transformation failed for {file_path.name}: {xf_err}. Using unscaled data.")
                    transformed_df = events_df
            else:
                transformed_df = events_df

            # Run Core Clean-up Gating via shared step function
            stem = file_path.stem
            step_res = run_cleanup_step(
                fcs_data=fcs_data,
                df_events=events_df,
                df_transformed=transformed_df,
                config=cfg,
                marker_metadata=active_marker_meta,
                output_plot_path=(plots_dir / f"{stem}_cleanup.png") if generate_plots else None,
                generate_plots=generate_plots,
                export_clean_fcs=export_clean_fcs,
                clean_fcs_path=(resolved_clean_fcs_dir / file_path.name) if export_clean_fcs else None,
                is_compensated=True,
            )
            res: CleanUpGatingResult = step_res.cleanup_result
            clean_fcs_path = step_res.clean_fcs_path or ""

            # Save MIFlowCyt reproducible hierarchy definition JSON
            h_json = hierarchies_dir / f"{stem}_hierarchy.json"
            with open(h_json, "w", encoding="utf-8") as jf:
                json.dump(res.hierarchy.to_dict(), jf, indent=2, default=_json_default)

            row = res.to_summary_dict()
            row["clean_fcs_path"] = clean_fcs_path
            row["compensation_source"] = comp_source
            row["hierarchy_json_path"] = str(h_json)
            summary_rows.append(row)

        except Exception as exc:
            logger.error(f"Error executing clean-up gating on {file_path.name}: {exc}", exc_info=True)
            summary_rows.append({
                "sample_name": file_path.name,
                "total_events": 0,
                "clean_events": 0,
                "clean_pct_total": 0.0,
                "execution_time_sec": 0.0,
                "status": f"ERROR: {str(exc)}",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = out_dir / "cohort_cleanup_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved cohort clean-up summary CSV to '{summary_csv_path}'")

    # Generate Markdown report
    total_samples = len(summary_rows)
    total_clean = int(summary_df["clean_events"].sum()) if "clean_events" in summary_df.columns else 0
    total_raw = int(summary_df["total_events"].sum()) if "total_events" in summary_df.columns else 0
    cohort_yield_pct = (total_clean / total_raw * 100.0) if total_raw > 0 else 0.0
    mean_sample_pct = float(summary_df["clean_pct_total"].mean()) if "clean_pct_total" in summary_df.columns else 0.0

    # Determine resolved cohort channel statuses
    fsc_status_col = summary_df.get("channel_singlet_fsc_status", pd.Series(dtype=object))
    fsc_statuses = set(fsc_status_col.dropna().unique())
    fsc_summary = f"Active ({', '.join(fsc_statuses)})" if "active" in fsc_statuses else ("Pass-Through (No pulse width/height)" if "pass_through" in fsc_statuses else "Disabled")

    ssc_status_col = summary_df.get("channel_singlet_ssc_status", pd.Series(dtype=object))
    ssc_statuses = set(ssc_status_col.dropna().unique())
    ssc_summary = f"Active ({', '.join(ssc_statuses)})" if "active" in ssc_statuses else ("Pass-Through (No pulse width/height)" if "pass_through" in ssc_statuses else "Disabled")

    viab_ch_col = summary_df.get("channel_viability", pd.Series(dtype=object))
    viab_channels = [str(c) for c in viab_ch_col.dropna().unique() if str(c) not in ("", "none", "None", "nan")]
    viab_summary = f"Active ({', '.join(viab_channels)})" if viab_channels else "None detected / Skipped"

    comp_src_col = summary_df.get("compensation_source", pd.Series(dtype=object))
    comp_sources = set(comp_src_col.dropna().unique())
    comp_summary = ", ".join(comp_sources) if comp_sources else "none"

    report_path = out_dir / "cohort_cleanup_report.md"
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# FlowX Core Clean-up Gating Cohort Report\n\n")
        rf.write(f"- **FlowX Software Version**: `{getattr(flowx, '__version__', '1.0.0')}`\n")
        rf.write(f"- **Execution Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write(f"- **Total Samples Processed**: {total_samples}\n")
        rf.write(f"- **Total Input Events**: {total_raw:,}\n")
        rf.write(f"- **Total Clean Events Retained**: {total_clean:,} ({cohort_yield_pct:.2f}% true cohort yield, {mean_sample_pct:.2f}% mean per-sample yield)\n")
        rf.write(f"- **Debris Strategy**: `{cfg.debris_method}`\n")
        rf.write(f"- **Compensation Source**: `{comp_summary}`\n")
        rf.write(f"- **FSC Singlets**: {fsc_summary}\n")
        rf.write(f"- **SSC Singlets**: {ssc_summary}\n")
        rf.write(f"- **Viability Exclusion**: {viab_summary}\n")
        rf.write(f"- **Lineage Dump Exclusion**: {cfg.dump_enabled} (channel: `{cfg.dump_channel or 'auto'}`)\n\n")
        rf.write("## Cohort Summary Table\n\n")
        rf.write(summary_df.to_markdown(index=False))
        rf.write("\n")

    logger.info(f"Saved cohort clean-up report to '{report_path}'")
    elapsed_total = time.perf_counter() - t0
    logger.info(f"Cohort clean-up completed in {elapsed_total:.2f}s.")

    return summary_df, summary_csv_path, report_path
