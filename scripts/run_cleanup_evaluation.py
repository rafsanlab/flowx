#!/usr/bin/env python3
"""Reusable evaluation and execution script for FlowX Preprocessing & Clean-up Pipeline.

Executes end-to-end preprocessing:
  1. Automated Fluidic Time QC (flowAI/flowCut/PeacoQC style)
  2. Spillover Compensation (Embedded Matrix or Single-Stained Controls)
  3. Scale Transformation (Logicle / Biexponential)
  4. Core Clean-up Gating (Debris exclusion, FSC & SSC dual singlets, Viability, and Dump exclusion)

Supports both the 30-case clinical FlowCyt Class A dataset and the high-parameter
multi-laser FR-FCM-Z2ZH dataset, exporting MIFlowCyt summary statistics, Markdown reports,
and diagnostic figures into outputs/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from flowx.gating.cleanup import CleanUpGateConfig, CleanUpGatingResult, run_cleanup_gating
from flowx.io.parser import FCSData, parse_fcs
from flowx.math.compensation import (
    CompensationConfig,
    CompensationMatrix,
    calculate_spillover_from_controls,
)
from flowx.math.transforms import apply_transform_to_dataframe, get_optimal_transform
from flowx.qc.engine import apply_time_qc
from flowx.qc.models import TimeQCConfig
from flowx.visualization.cleanup_plots import plot_cleanup_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlowX: Preprocessing & Core Clean-up Evaluation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cohort",
        "-c",
        choices=["all", "flowcyt_a", "z2zh"],
        default="all",
        help="Cohort to evaluate: 'flowcyt_a' (30 Class A acute leukemia cases), 'z2zh' (FR-FCM-Z2ZH 28-color sample & controls), or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="outputs/cleanup_evaluation",
        help="Base directory for summary tables, Markdown reports, and diagnostic plots.",
    )
    parser.add_argument(
        "--debris-method",
        choices=["gmm", "adaptive", "manual"],
        default="gmm",
        help="Debris exclusion method: 'gmm' (automated Gaussian Mixture Model clustering, default) or 'adaptive'.",
    )
    parser.add_argument(
        "--singlet-mode",
        choices=["dual", "single", "one_time", "fsc_only", "ssc_only", "none"],
        default="dual",
        help="Singlet gating mode: 'dual' (default: FSC then SSC), 'single'/'one_time' (single active gate), etc.",
    )
    parser.add_argument(
        "--singlet-method",
        choices=["gmm_ratio", "envelope"],
        default="gmm_ratio",
        help="Singlet discrimination algorithm: 'gmm_ratio' (default: dynamic pulse-ratio clustering) or 'envelope'.",
    )
    parser.add_argument(
        "--viability-cutoff",
        type=float,
        default=None,
        help="Explicit upper threshold cutoff for live cells in transformed space.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Generate and save publication diagnostic cascade plots.",
    )
    parser.add_argument(
        "--no-plots",
        dest="save_plots",
        action="store_false",
        help="Disable diagnostic plot generation to maximize throughput.",
    )
    parser.add_argument(
        "--export-clean-fcs",
        action="store_true",
        default=False,
        help="Export clean gated events to new standardized FCS files.",
    )
    return parser.parse_args()


def run_pipeline_on_sample(
    fcs_path: Path,
    config: Optional[CleanUpGateConfig] = None,
    time_qc_enabled: bool = True,
    use_embedded_comp: bool = True,
    use_logicle_xform: bool = True,
    output_plot_path: Optional[Path] = None,
    export_clean_fcs_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute end-to-end pipeline from Time QC to Clean-up Gating on a single FCS file.

    Args:
        fcs_path: Path to raw input FCS file.
        config: CleanUpGateConfig instance with custom parameters.
        time_qc_enabled: Whether to run automated fluidic time quality control.
        use_embedded_comp: Whether to apply embedded spillover matrix if present.
        use_logicle_xform: Whether to apply Logicle transformation to fluorescence channels.
        output_plot_path: Destination path for diagnostic cascade figure.
        export_clean_fcs_path: Destination path for clean gated FCS file.

    Returns:
        Flat dictionary of audit metrics, population yields, and execution timings.
    """
    t_total_start = time.perf_counter()
    cfg = config or CleanUpGateConfig()
    sample_metrics: Dict[str, Any] = {
        "sample": fcs_path.stem,
        "filename": fcs_path.name,
    }

    # Step 0: Parse FCS
    t0 = time.perf_counter()
    fcs_data = parse_fcs(fcs_path)
    sample_metrics["cytometer"] = fcs_data.cytometer or "Unknown"
    sample_metrics["raw_events"] = fcs_data.event_count
    sample_metrics["channel_count"] = fcs_data.channel_count
    sample_metrics["parse_time_sec"] = time.perf_counter() - t0

    # Step 1: Automated Time QC (flowAI/flowCut/PeacoQC style)
    time_ch = fcs_data.get_time_channel()
    events_post_qc = fcs_data.events_df.copy()
    if time_qc_enabled and time_ch and time_ch in fcs_data.events_df.columns and fcs_data.event_count >= 100:
        t0 = time.perf_counter()
        tqc_cfg = TimeQCConfig.for_sample()
        tqc_res = apply_time_qc(fcs_data, config=tqc_cfg)
        decision = tqc_res.evaluate_workflow(fcs_data)
        sample_metrics["time_qc_passed_events"] = tqc_res.passed_events
        sample_metrics["time_qc_passed_pct"] = round(tqc_res.passed_pct, 2)
        sample_metrics["time_qc_action"] = str(decision.action)
        sample_metrics["time_qc_rate_spikes"] = int(np.sum(tqc_res.flow_rate_anomaly_mask))
        sample_metrics["time_qc_drift_channels"] = len(tqc_res.drift_stats) if tqc_res.drift_stats else 0
        sample_metrics["time_qc_margin_saturated"] = int(np.sum(tqc_res.margin_saturation_mask))
        sample_metrics["time_qc_time_sec"] = time.perf_counter() - t0
        if decision.clean_events_df is not None:
            events_post_qc = decision.clean_events_df
    else:
        sample_metrics["time_qc_passed_events"] = fcs_data.event_count
        sample_metrics["time_qc_passed_pct"] = 100.0
        sample_metrics["time_qc_action"] = "SKIPPED"
        sample_metrics["time_qc_time_sec"] = 0.0

    # Step 2: Fluorescence Spillover Compensation
    t0 = time.perf_counter()
    events_post_comp = events_post_qc.copy()
    if use_embedded_comp and fcs_data.spillover_matrix is not None:
        comp_mat = CompensationMatrix(
            detectors=list(fcs_data.spillover_matrix.columns),
            matrix=fcs_data.spillover_matrix.to_numpy(),
        )
        events_post_comp = comp_mat.apply_dataframe(events_post_comp)
        sample_metrics["comp_status"] = "APPLIED_EMBEDDED"
        sample_metrics["comp_detectors"] = len(comp_mat.detectors)
        sample_metrics["comp_cond_num"] = round(float(comp_mat.condition_number), 2)
    else:
        sample_metrics["comp_status"] = "NONE"
    sample_metrics["comp_time_sec"] = time.perf_counter() - t0

    # Step 3: Scale Transformation (Logicle)
    t0 = time.perf_counter()
    fluoro_channels = fcs_data.spillover_channels
    if not fluoro_channels:
        exclude = {
            "fsc-a", "fsc-h", "fsc-w", "ssc-a", "ssc-h", "ssc-w", "time",
            "fs int", "ss int", "ss peak", "ss tof",
        }
        fluoro_channels = [
            ch for ch in fcs_data.pnn_labels
            if ch.lower() not in exclude
            and not any(ch.lower().startswith(p) for p in ["fs ", "fs.", "ss ", "ss.", "time"])
        ]

    events_xformed = events_post_comp.copy()
    if use_logicle_xform and fluoro_channels:
        xform = get_optimal_transform(fcs_data, transform_type="logicle")
        events_xformed = apply_transform_to_dataframe(
            events_post_comp, channels=fluoro_channels, transform=xform
        )
        sample_metrics["xform_status"] = "LOGICLE_APPLIED"
        sample_metrics["xform_channels"] = len(fluoro_channels)
    else:
        sample_metrics["xform_status"] = "SKIPPED"
    sample_metrics["xform_time_sec"] = time.perf_counter() - t0

    # Step 4: Core Clean-up Gating
    t0 = time.perf_counter()
    cleanup_res: CleanUpGatingResult = run_cleanup_gating(
        fcs_data=fcs_data,
        df_events=events_post_comp,
        df_transformed=events_xformed,
        config=cfg,
    )
    sample_metrics["clean_events"] = cleanup_res.clean_event_count
    sample_metrics["clean_pct_post_qc"] = round(cleanup_res.clean_event_pct, 2)
    sample_metrics["clean_pct_raw"] = round(
        float(cleanup_res.clean_event_count / fcs_data.event_count * 100.0), 2
    )

    for p_name in ["Cells", "Singlets_FSC", "Singlets_SSC", "Viable", "Target"]:
        if p_name in cleanup_res.masks:
            cnt = int(np.sum(cleanup_res.masks[p_name]))
            sample_metrics[f"{p_name.lower()}_count"] = cnt
            sample_metrics[f"{p_name.lower()}_pct_qc"] = round(
                cnt / len(events_post_qc) * 100.0, 2
            )

    for k, v in cleanup_res.cutoffs.items():
        sample_metrics[f"cutoff_{k}"] = round(float(v), 4)

    sample_metrics["cleanup_time_sec"] = time.perf_counter() - t0
    sample_metrics["total_pipeline_time_sec"] = time.perf_counter() - t_total_start

    # Step 5: Optional Diagnostic Plots & Clean FCS Export
    if output_plot_path:
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_cleanup_diagnostics(
            fcs_data=fcs_data,
            cleanup_result=cleanup_res,
            df_events=events_post_comp,
            df_transformed=events_xformed,
            output_path=output_plot_path,
        )
        sample_metrics["diagnostic_plot_path"] = str(output_plot_path)

    if export_clean_fcs_path and cleanup_res.clean_event_count > 0:
        export_clean_fcs_path.parent.mkdir(parents=True, exist_ok=True)
        cleanup_res.export_fcs(export_clean_fcs_path, fcs_data, events_df=events_post_comp)
        sample_metrics["clean_fcs_path"] = str(export_clean_fcs_path)

    return sample_metrics


def run_evaluation(
    cohort: str = "all",
    output_dir: str = "outputs/cleanup_evaluation",
    debris_method: str = "gmm",
    singlet_mode: str = "dual",
    singlet_method: str = "gmm_ratio",
    viability_cutoff: Optional[float] = None,
    save_plots: bool = True,
    export_clean_fcs: bool = False,
) -> None:
    """Execute cohort evaluation and save tabular reports."""
    out_path = Path(output_dir).resolve()
    plots_dir = out_path / "plots"
    tables_dir = out_path / "tables"
    clean_fcs_dir = out_path / "clean_fcs"
    out_path.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FLOWX PIPELINE EVALUATION: QUALITY CONTROL -> COMPENSATION -> TRANSFORMATION -> CLEAN-UP")
    print(f"Target Cohort: {cohort.upper()} | Output Directory: {out_path} | Debris Method: {debris_method.upper()}")
    print("=" * 80)

    # -----------------------------------------------------------------
    # COHORT 1: 30 Case*_A Clinical Samples
    # -----------------------------------------------------------------
    if cohort in ("all", "flowcyt_a"):
        raw_dir = Path("data/conventional/bini_2024_flowcyt/raw")
        case_a_files = sorted(raw_dir.glob("Case*_A.fcs"))
        print(f"\n[Cohort 1] Found {len(case_a_files)} Case*_A files in {raw_dir}")

        cfg_cohort1 = CleanUpGateConfig(
            sample_type="general",
            debris_method=debris_method,
            singlet_mode=singlet_mode,
            singlet_method=singlet_method,
            viability_cutoff=viability_cutoff,
            viability_enabled=True,
        )

        cohort1_metrics = []
        for idx, f_path in enumerate(case_a_files, start=1):
            plot_path = plots_dir / f"{f_path.stem}_cleanup_diagnostics.png" if save_plots else None
            fcs_out = clean_fcs_dir / f"{f_path.stem}_clean.fcs" if export_clean_fcs else None
            print(f"  [{idx:02d}/30] Processing {f_path.name}...", end="", flush=True)
            res = run_pipeline_on_sample(
                f_path,
                config=cfg_cohort1,
                output_plot_path=plot_path,
                export_clean_fcs_path=fcs_out,
            )
            cohort1_metrics.append(res)
            print(
                f" Done ({res['raw_events']:,} raw -> {res.get('clean_events', 0):,} clean | "
                f"{res.get('clean_pct_raw', 0.0):.1f}% yield | {res['total_pipeline_time_sec']:.2f}s)"
            )

        df_c1 = pd.DataFrame(cohort1_metrics)
        c1_csv = tables_dir / "cohort1_flowcyt_class_a_summary.csv"
        df_c1.to_csv(c1_csv, index=False)
        print(f"\nSaved Cohort 1 summary table: {c1_csv}")

    # -----------------------------------------------------------------
    # COHORT 2: FR-FCM-Z2ZH 28-Color Dataset
    # -----------------------------------------------------------------
    if cohort in ("all", "z2zh"):
        z2zh_dir = Path("data/conventional/somasundaram_2020_frfcmz2zh")
        sample_z2zh = z2zh_dir / "samples_WBC 9196_002.fcs"
        print("\n" + "=" * 80)
        print(f"[Cohort 2] Evaluating FR-FCM-Z2ZH sample: {sample_z2zh.name}")
        print("=" * 80)

        # Single-stained compensation controls vs embedded matrix
        fcs_z2zh = parse_fcs(sample_z2zh)
        ctrl_res = calculate_spillover_from_controls(
            controls_dir_or_files=z2zh_dir,
            channels=fcs_z2zh.spillover_channels,
            fcs_sample=fcs_z2zh,
            fallback_matrix=fcs_z2zh.spillover_matrix,
        )
        print(f"  Mapped {len(ctrl_res.control_files_mapped)}/28 controls (Condition: {ctrl_res.matrix.condition_number:.2f})")
        print(f"  Agreement with Cytometer Matrix: Pearson r={ctrl_res.embedded_correlation:.6f}, MAE={ctrl_res.embedded_mae:.6f}")

        # Execute Clean-up Gating
        plot_path = plots_dir / "FR_FCM_Z2ZH_samples_WBC_9196_002_cleanup_diagnostics.png" if save_plots else None
        fcs_out = clean_fcs_dir / "FR_FCM_Z2ZH_samples_WBC_9196_002_clean.fcs" if export_clean_fcs else None

        cfg_z2zh = CleanUpGateConfig(
            sample_type="general",
            debris_method=debris_method,
            singlet_mode=singlet_mode,
            singlet_method=singlet_method,
            viability_cutoff=viability_cutoff,
            viability_enabled=True,
        )
        res_z2zh = run_pipeline_on_sample(
            sample_z2zh,
            config=cfg_z2zh,
            output_plot_path=plot_path,
            export_clean_fcs_path=fcs_out,
        )
        print(
            f"  Done ({res_z2zh['raw_events']:,} raw -> {res_z2zh['clean_events']:,} clean | "
            f"{res_z2zh['clean_pct_raw']:.1f}% yield | Cutoff: {res_z2zh.get('cutoff_viability_cutoff', 'N/A')} | "
            f"{res_z2zh['total_pipeline_time_sec']:.2f}s)"
        )

        df_c2 = pd.DataFrame([res_z2zh])
        c2_csv = tables_dir / "cohort2_fr_fcm_z2zh_summary.csv"
        df_c2.to_csv(c2_csv, index=False)
        print(f"Saved Cohort 2 summary table: {c2_csv}")

    # Generate Markdown Summary Report
    report_path = out_path / "evaluation_report.md"
    generate_markdown_report(report_path, cohort1_df=df_c1 if cohort in ("all", "flowcyt_a") else None, cohort2_dict=res_z2zh if cohort in ("all", "z2zh") else None)
    print(f"\nSaved Comprehensive Markdown Report: {report_path}")
    print("=" * 80)


def generate_markdown_report(
    report_path: Path,
    cohort1_df: Optional[pd.DataFrame] = None,
    cohort2_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate MIFlowCyt-standard Markdown summary report of the evaluation."""
    lines = [
        "# FlowX Pipeline Cohort Evaluation Report",
        "",
        "## Executive Summary",
        "This report documents the performance, quality control metrics, compensation accuracy, and population yields of the updated preprocessing pipeline (Automated Time QC → Spillover Compensation → Logicle Transformation → Core Clean-up Gating) across two benchmark datasets.",
        "",
    ]

    if cohort1_df is not None:
        lines.extend([
            "## Cohort 1: Clinical FlowCyt Class A (30 Patients)",
            f"- **Dataset**: 30 Acute Leukemia patient samples (Beckman Coulter Navios / Kaluza)",
            f"- **Total Raw Events Processed**: {cohort1_df['raw_events'].sum():,}",
            f"- **Average Raw Events per Sample**: {cohort1_df['raw_events'].mean():,.0f}",
            f"- **Average Time QC Retention**: {cohort1_df['time_qc_passed_pct'].mean():.2f}%",
            f"- **Average Intact Cells (% of QC)**: {cohort1_df['cells_pct_qc'].mean():.2f}%",
            f"- **Average SSC Singlets (% of QC)**: {cohort1_df['singlets_ssc_pct_qc'].mean():.2f}%",
            f"- **Average Clean Yield (% of Raw)**: {cohort1_df['clean_pct_raw'].mean():.2f}%",
            f"- **Total Runtime**: {cohort1_df['total_pipeline_time_sec'].sum():.2f}s (Average {cohort1_df['total_pipeline_time_sec'].mean():.2f}s/sample)",
            "",
            "### Cohort 1 Patient Summary Table",
            cohort1_df[[
                "sample", "raw_events", "time_qc_passed_pct", "time_qc_action",
                "cells_pct_qc", "singlets_ssc_pct_qc", "clean_pct_raw", "total_pipeline_time_sec"
            ]].to_markdown(index=False),
            "",
        ])

    if cohort2_dict is not None:
        lines.extend([
            "## Cohort 2: FR-FCM-Z2ZH Multi-Laser High-Parameter Dataset",
            "- **Sample**: `samples_WBC 9196_002.fcs` (BD LSRFortessa)",
            f"- **Raw Events**: {cohort2_dict['raw_events']:,} across {cohort2_dict['channel_count']} channels",
            f"- **Time QC Passed**: {cohort2_dict['time_qc_passed_events']:,} ({cohort2_dict['time_qc_passed_pct']}%, Action: {cohort2_dict['time_qc_action']})",
            f"- **Debris Gated Cells**: {cohort2_dict.get('cells_count', 'N/A'):,} ({cohort2_dict.get('cells_pct_qc', 'N/A')}%)",
            f"- **FSC Singlets**: {cohort2_dict.get('singlets_fsc_count', 'N/A'):,} ({cohort2_dict.get('singlets_fsc_pct_qc', 'N/A')}%)",
            f"- **SSC Singlets**: {cohort2_dict.get('singlets_ssc_count', 'N/A'):,} ({cohort2_dict.get('singlets_ssc_pct_qc', 'N/A')}%)",
            f"- **Viable Cells**: {cohort2_dict.get('viable_count', 'N/A'):,} ({cohort2_dict.get('viable_pct_qc', 'N/A')}%, Cutoff: {cohort2_dict.get('cutoff_viability_cutoff', 'N/A')})",
            f"- **Total Clean Events**: {cohort2_dict['clean_events']:,} ({cohort2_dict['clean_pct_raw']}% of raw)",
            f"- **Total Runtime**: {cohort2_dict['total_pipeline_time_sec']:.2f}s",
            "",
        ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    run_evaluation(
        cohort=args.cohort,
        output_dir=args.output_dir,
        debris_method=args.debris_method,
        singlet_mode=args.singlet_mode,
        singlet_method=args.singlet_method,
        viability_cutoff=args.viability_cutoff,
        save_plots=args.save_plots,
        export_clean_fcs=args.export_clean_fcs,
    )


if __name__ == "__main__":
    main()
