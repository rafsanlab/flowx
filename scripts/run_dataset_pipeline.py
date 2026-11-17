#!/usr/bin/env python3
"""Unified End-to-End Dataset Pipeline Runner for FlowX.

Executes the complete preprocessing workflow from raw FCS files:
  1. Inspection: Channel extraction, cytometer detection, summary & audit report.
  2. Time QC: Flow rate anomaly detection, signal drift screening, margin saturation trimming.
  3. Compensation: Matrix calculation from single-stained controls or embedded matrix verification.
  4. Core Clean-up Gating: GMM clustering debris exclusion, dual singlets (FSC/SSC), viability dye exclusion.

Automatically organizes outputs into structured subdirectories per dataset:
  outputs/dataset_<name>/
    ├── inspection/
    ├── time_qc/
    ├── compensation/
    └── cleanup/
"""

from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure src and scripts are in sys.path
_project_root = str(Path(__file__).resolve().parent.parent / "src")
_scripts_dir = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from flowx.batch.models import (
    BatchCorrectionConfig,
    BatchPreparationConfig,
    CohortSampleInfo,
    DownsampleConfig,
)
from flowx.batch.runner import run_batch_preparation
from flowx.cli import cmd_inspect
from flowx.gating.cleanup import CleanUpGateConfig, CleanUpGatingResult, run_cleanup_gating
from flowx.gating.runner import run_cohort_cleanup
from flowx.io.parser import FCSData, parse_fcs
from flowx.math.compensation import (
    CompensationConfig,
    CompensationMatrix,
    calculate_spillover_from_controls,
    run_dataset_compensation,
)
from flowx.math.transforms import apply_transform_to_dataframe, get_optimal_transform
from flowx.qc.engine import apply_time_qc
from flowx.qc.models import TimeQCConfig
from flowx.qc.runner import run_cohort_time_qc
from flowx.visualization.cleanup_plots import plot_cleanup_diagnostics
from flowx.visualization.comp_plots import plot_nxn_compensation_matrix
from flowx.visualization.plotting import plot_compensation_comparison, plot_spillover_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlowX: End-to-End Dataset Pipeline Runner (Inspection -> Time QC -> Compensation -> Clean-up)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["all", "flowcyt", "fr_fcm_z2zh", "tqc_tutorial"],
        default="all",
        help="Dataset to process: 'flowcyt' (30 Case A clinical samples), 'fr_fcm_z2zh' (28 controls + WBC sample), 'tqc_tutorial' (8 BD LSRII samples), or 'all'.",
    )
    parser.add_argument(
        "--base-output-dir",
        "-o",
        default="outputs/internal_validation",
        help="Base outputs directory. Each dataset creates outputs/internal_validation/dataset_<name>/ with step subfolders.",
    )
    parser.add_argument(
        "--debris-method",
        choices=["gmm", "adaptive", "manual"],
        default="gmm",
        help="Debris exclusion method for clean-up gating ('gmm' clustering [default] or 'adaptive').",
    )
    parser.add_argument(
        "--save-plots",
        dest="save_plots",
        action="store_true",
        default=True,
        help="Generate and save diagnostic cascade and verification plots.",
    )
    parser.add_argument(
        "--no-save-plots",
        dest="save_plots",
        action="store_false",
        help="Disable generating and saving diagnostic cascade and verification plots.",
    )
    parser.add_argument(
        "--export-clean-fcs",
        dest="export_clean_fcs",
        action="store_true",
        default=False,
        help="Export clean FCS files at Time QC and Clean-up stages (default: False to conserve disk space).",
    )
    parser.add_argument(
        "--no-export-clean-fcs",
        dest="export_clean_fcs",
        action="store_false",
        help="Disable exporting clean FCS files at Time QC and Clean-up stages.",
    )
    parser.add_argument(
        "--viability-cutoff",
        type=float,
        default=None,
        help="Optional explicit user-defined upper threshold for live cells in transformed space.",
    )
    return parser.parse_args()


# =====================================================================
# PIPELINE STEP IMPLEMENTATIONS
# =====================================================================

def run_step1_inspection(files: List[Path], output_dir: Path) -> Path:
    """Step 1: Inspect FCS files, detect instruments & channels, export summary CSV & report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    from inspect_fcs import build_script_parser
    args = build_script_parser().parse_args([*[str(f) for f in files], "-o", str(output_dir)])
    cmd_inspect(args)
    return output_dir / "summary.csv"


def run_step2_time_qc(
    files: List[Path],
    output_dir: Path,
    export_clean_fcs: bool = True,
    generate_plots: bool = True,
) -> Tuple[pd.DataFrame, Path, Path]:
    """Step 2: Automated fluidic Time QC (flowAI/flowCut/PeacoQC style)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_cohort_time_qc(
        file_list_or_paths=files,
        output_dir=output_dir,
        generate_plots=generate_plots,
        export_clean_fcs=export_clean_fcs,
    )


def run_step3_flowcyt_compensation(
    files: List[Path],
    output_dir: Path,
    report_title: str = "Flow Cytometry Compensation Report",
    generate_plots: bool = True,
) -> Path:
    """Step 3: Apply embedded spillover matrix, export tables & diagnostic plots."""
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    samples_dir = output_dir / "compensated_samples"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        f"# {report_title}",
        "",
        "> **Mode:** Embedded Spillover Matrix Compensation",
        f"> **Total Samples:** {len(files)}",
        "",
        "| Sample | Events | Channels | Matrix Present | Condition Number κ(S) | Primary Spillover Pair |",
        "|:---|:---:|:---:|:---:|:---:|:---|",
    ]

    for idx, f_path in enumerate(tqdm(files, desc="Step 3 Compensation", unit="file")):
        fcs = parse_fcs(f_path)
        if fcs.spillover_matrix is not None:
            comp_mat = CompensationMatrix(
                detectors=list(fcs.spillover_matrix.columns),
                matrix=fcs.spillover_matrix.to_numpy(),
            )
            # Save matrix CSV
            mat_csv = tables_dir / f"{f_path.stem}_spillover_matrix.csv"
            fcs.spillover_matrix.to_csv(mat_csv)

            # Apply compensation
            comp_df = comp_mat.apply_dataframe(fcs.events_df)
            comp_sample_csv = samples_dir / f"{f_path.stem}_compensated.csv"
            comp_df.head(1000).to_csv(comp_sample_csv, index=False)

            # Diagnostic plot for representative or first samples
            if generate_plots:
                spill_plot = plots_dir / f"{f_path.stem}_compensation.png"
                if len(comp_mat.detectors) >= 2:
                    plot_compensation_comparison(
                        uncomp_df=fcs.events_df,
                        comp_df=comp_df,
                        x_channel=comp_mat.detectors[0],
                        y_channel=comp_mat.detectors[1],
                        output_path=spill_plot,
                    )
                    # Generate NxN compensation matrix for representative samples
                    if idx < 3 or "case1_a" in f_path.name.lower() or len(files) <= 5:
                        nxn_dir = output_dir / "nxn_matrices"
                        nxn_samples = nxn_dir / "samples"
                        nxn_controls = nxn_dir / "controls"
                        nxn_samples.mkdir(parents=True, exist_ok=True)
                        nxn_controls.mkdir(parents=True, exist_ok=True)

                        plots_nxn_samples = plots_dir / "nxn_matrices" / "samples"
                        plots_nxn_controls = plots_dir / "nxn_matrices" / "controls"
                        plots_nxn_samples.mkdir(parents=True, exist_ok=True)
                        plots_nxn_controls.mkdir(parents=True, exist_ok=True)

                        nxn_plot = nxn_samples / f"{f_path.stem}_nxn_matrix.png"
                        try:
                            plot_nxn_compensation_matrix(
                                uncomp_df=fcs.events_df,
                                comp_df=comp_df,
                                comp_matrix=comp_mat,
                                channels=comp_mat.detectors,
                                marker_map=fcs.detector_to_marker,
                                sample_name=f_path.stem,
                                output_path=nxn_plot,
                                lower_triangle_only=True,
                                auto_gate_sample=True,
                                viable_only=True,
                            )
                            shutil.copy2(nxn_plot, plots_nxn_samples / nxn_plot.name)
                            shutil.copy2(nxn_plot, plots_dir / f"{f_path.stem}_nxn_compensation_matrix.png")
                        except Exception as exc:
                            logger.debug(f"NxN plot skipped for {f_path.name}: {exc}")

            top_pair = f"{comp_mat.detectors[0]} -> {comp_mat.detectors[1]}"
            report_lines.append(
                f"| `{f_path.name}` | {fcs.event_count:,} | {len(comp_mat.detectors)} | Yes | {comp_mat.condition_number:.2f} | {top_pair} |"
            )
        else:
            report_lines.append(
                f"| `{f_path.name}` | {fcs.event_count:,} | N/A | No | N/A | None |"
            )

    report_path = output_dir / "compensation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def run_step4_cleanup(
    files: List[Path],
    output_dir: Path,
    debris_method: str = "gmm",
    generate_plots: bool = True,
    export_clean_fcs: bool = True,
    viability_cutoff: Optional[float] = None,
    compensation_matrix: Optional[Union[str, Path, pd.DataFrame, CompensationMatrix]] = None,
) -> Tuple[pd.DataFrame, Path, Path]:
    """Step 4: Core Clean-up Gating (Debris clustering, dual singlets, viability, dump)."""
    cfg = CleanUpGateConfig(
        sample_type="general",
        debris_method=debris_method,
        singlet_fsc_enabled=True,
        singlet_ssc_enabled=True,
        viability_enabled=True,
        viability_cutoff=viability_cutoff,
    )
    return run_cohort_cleanup(
        file_list_or_paths=files,
        output_dir=output_dir,
        config=cfg,
        generate_plots=generate_plots,
        export_clean_fcs=export_clean_fcs,
        compensation_matrix=compensation_matrix,
    )


# =====================================================================
# DATASET ORCHESTRATORS
# =====================================================================

def process_dataset_flowcyt(
    base_output_dir: Path,
    debris_method: str = "gmm",
    save_plots: bool = True,
    export_clean_fcs: bool = True,
    viability_cutoff: Optional[float] = None,
) -> None:
    """Run full 4-step pipeline on Clinical FlowCyt Class A dataset."""
    ds_out = base_output_dir / "dataset_flowcyt"
    raw_dir = Path("data/conventional/bini_2024_flowcyt/raw")
    case_a_files = sorted(raw_dir.glob("Case*_A.fcs"))

    print("\n" + "=" * 80)
    print(f"PROCESSING DATASET: FlowCyt Class A (30 Patients)")
    print(f"Destination Root: {ds_out}")
    print("=" * 80)

    # Step 1: Inspection
    print("\n[Step 1/4] Running Dataset Inspection...")
    t0 = time.perf_counter()
    run_step1_inspection(case_a_files, ds_out / "inspection")
    print(f"  Inspection completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'inspection'}")

    # Step 2: Time QC
    print("\n[Step 2/4] Running Automated Fluidic Time QC...")
    t0 = time.perf_counter()
    df_qc, qc_csv, qc_rep = run_step2_time_qc(
        case_a_files,
        ds_out / "time_qc",
        export_clean_fcs=export_clean_fcs,
        generate_plots=save_plots,
    )
    print(f"  Time QC completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'time_qc'}")

    # Step 3: Compensation
    print("\n[Step 3/4] Applying Spillover Compensation & Verification...")
    t0 = time.perf_counter()
    run_step3_flowcyt_compensation(
        case_a_files,
        ds_out / "compensation",
        generate_plots=save_plots,
    )
    print(f"  Compensation completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'compensation'}")

    # Step 4: Core Clean-up Gating
    print(f"\n[Step 4/4] Executing Core Clean-Up Gating (Debris Method: {debris_method.upper()})...")
    t0 = time.perf_counter()
    # If clean FCS files were exported from Time QC, gate those clean files!
    clean_fcs_inputs = sorted((ds_out / "time_qc/clean_fcs").glob("*.fcs"))
    target_inputs = clean_fcs_inputs if clean_fcs_inputs else case_a_files
    df_clean, clean_csv, clean_rep = run_step4_cleanup(
        target_inputs,
        ds_out / "cleanup",
        debris_method=debris_method,
        generate_plots=save_plots,
        export_clean_fcs=export_clean_fcs,
        viability_cutoff=viability_cutoff,
    )
    print(f"  Clean-up Gating completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'cleanup'}")

    # Step 5: Batch Preparation & Downsampling
    print("\n[Step 5/5] Executing Cohort Downsampling & Balancing (Bottleneck Auto-Balancing)...")
    t0 = time.perf_counter()
    clean_fcs_inputs = sorted((ds_out / "cleanup/clean_fcs").glob("*.fcs"))
    if clean_fcs_inputs:
        batch_out = ds_out / "batch"
        # Recover upstream raw and clean counts from cleanup summary if present
        cleanup_summary_file = ds_out / "cleanup/cohort_cleanup_summary.csv"
        sample_info_list = []
        if cleanup_summary_file.exists():
            clean_df = pd.read_csv(cleanup_summary_file)
            name_c = "sample_name" if "sample_name" in clean_df.columns else "sample_id"
            clean_lookup = {str(r[name_c]).strip(): r for _, r in clean_df.iterrows()}
            for f in clean_fcs_inputs:
                s_id = f.stem
                row = clean_lookup.get(s_id)
                raw_c = int(row["total_events"]) if row is not None and "total_events" in row else 0
                cln_c = int(row["clean_events"]) if row is not None and "clean_events" in row else 0
                sample_info_list.append(
                    CohortSampleInfo(
                        sample_id=s_id,
                        batch_id="Batch_1",
                        is_control=False,
                        filepath=str(f.resolve()),
                        raw_event_count=raw_c,
                        clean_event_count=cln_c,
                    )
                )
        batch_cfg = BatchPreparationConfig(
            downsampling=DownsampleConfig(
                enabled=True,
                mode="min_sample",
                target_events=None,
                random_seed=42,
            ),
            batch_normalization=BatchCorrectionConfig(enabled=False),
            export_individual_fcs=export_clean_fcs,
            export_pooled_cohort=True,
        )
        run_batch_preparation(
            sample_info=sample_info_list if sample_info_list else None,
            cohort_targets=clean_fcs_inputs if not sample_info_list else None,
            config=batch_cfg,
            output_dir=batch_out,
        )
        print(f"  Batch Preparation & Downsampling completed in {time.perf_counter()-t0:.2f}s -> {batch_out}")


def process_dataset_fr_fcm_z2zh(
    base_output_dir: Path,
    debris_method: str = "gmm",
    save_plots: bool = True,
    export_clean_fcs: bool = True,
    viability_cutoff: Optional[float] = None,
) -> None:
    """Run full 4-step pipeline on FR-FCM-Z2ZH 28-color dataset."""
    ds_out = base_output_dir / "dataset_fr_fcm_z2zh"
    z2zh_dir = Path("data/conventional/somasundaram_2020_frfcmz2zh")
    raw_dir = z2zh_dir / "raw" if (z2zh_dir / "raw").exists() else z2zh_dir
    sample_wbc = raw_dir / "samples_WBC 9196_002.fcs"
    ctrl_files = sorted(raw_dir.glob("Compensation Controls_Ex*.fcs"))
    all_files = ctrl_files + ([sample_wbc] if sample_wbc.exists() else [])

    print("\n" + "=" * 80)
    print(f"PROCESSING DATASET: FR-FCM-Z2ZH (28-Color High-Parameter WBC + Controls)")
    print(f"Destination Root: {ds_out}")
    print("=" * 80)

    # Step 1: Inspection
    print("\n[Step 1/4] Running Dataset Inspection...")
    t0 = time.perf_counter()
    run_step1_inspection(all_files, ds_out / "inspection")
    print(f"  Inspection completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'inspection'}")

    # Step 2: Time QC
    print("\n[Step 2/4] Running Automated Fluidic Time QC...")
    t0 = time.perf_counter()
    df_qc, qc_csv, qc_rep = run_step2_time_qc(
        all_files,
        ds_out / "time_qc",
        export_clean_fcs=export_clean_fcs,
        generate_plots=save_plots,
    )
    print(f"  Time QC completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'time_qc'}")

    # Step 3: Compensation (Controls calculation + verification suite)
    print("\n[Step 3/4] Calculating Controls Spillover & Biological Verification Suite...")
    t0 = time.perf_counter()
    run_dataset_compensation(
        dataset_dir_or_sample=raw_dir,
        output_dir=ds_out / "compensation",
        qc_manifest_path=qc_csv,
        generate_plots=save_plots,
    )
    print(f"  Compensation completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'compensation'}")

    # Step 4: Core Clean-up Gating
    print(f"\n[Step 4/4] Executing Core Clean-Up Gating on WBC Sample (Debris: {debris_method.upper()})...")
    t0 = time.perf_counter()
    clean_sample_fcs = ds_out / f"time_qc/clean_fcs/{sample_wbc.name}"
    target_sample = clean_sample_fcs if clean_sample_fcs.is_file() else sample_wbc

    calculated_mat = ds_out / "compensation/tables/spillover_matrix_calculated.csv"
    comp_input = calculated_mat if calculated_mat.is_file() else None

    df_clean, clean_csv, clean_rep = run_step4_cleanup(
        [target_sample],
        ds_out / "cleanup",
        debris_method=debris_method,
        generate_plots=save_plots,
        export_clean_fcs=export_clean_fcs,
        viability_cutoff=viability_cutoff,
        compensation_matrix=comp_input,
    )
    print(f"  Clean-up Gating completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'cleanup'}")

    # Step 5: Batch Preparation & Downsampling
    print("\n[Step 5/5] Executing Downsampling on WBC Sample (Target: 50,000 events)...")
    t0 = time.perf_counter()
    clean_wbc_inputs = sorted((ds_out / "cleanup/clean_fcs").glob("*.fcs"))
    if clean_wbc_inputs:
        batch_out = ds_out / "batch"
        cleanup_summary_file = ds_out / "cleanup/cohort_cleanup_summary.csv"
        sample_info_list = []
        if cleanup_summary_file.exists():
            clean_df = pd.read_csv(cleanup_summary_file)
            name_c = "sample_name" if "sample_name" in clean_df.columns else "sample_id"
            clean_lookup = {str(r[name_c]).strip(): r for _, r in clean_df.iterrows()}
            for f in clean_wbc_inputs:
                s_id = f.stem
                row = clean_lookup.get(s_id)
                raw_c = int(row["total_events"]) if row is not None and "total_events" in row else 0
                cln_c = int(row["clean_events"]) if row is not None and "clean_events" in row else 0
                sample_info_list.append(
                    CohortSampleInfo(
                        sample_id=s_id,
                        batch_id="Batch_1",
                        is_control=False,
                        filepath=str(f.resolve()),
                        raw_event_count=raw_c,
                        clean_event_count=cln_c,
                    )
                )
        batch_cfg = BatchPreparationConfig(
            downsampling=DownsampleConfig(
                enabled=True,
                mode="target_count",
                target_events=50000,
                random_seed=42,
            ),
            batch_normalization=BatchCorrectionConfig(enabled=False),
            export_individual_fcs=export_clean_fcs,
            export_pooled_cohort=True,
        )
        run_batch_preparation(
            sample_info=sample_info_list if sample_info_list else None,
            cohort_targets=clean_wbc_inputs if not sample_info_list else None,
            config=batch_cfg,
            output_dir=batch_out,
        )
        print(f"  Batch Preparation & Downsampling completed in {time.perf_counter()-t0:.2f}s -> {batch_out}")


def process_dataset_tqc_tutorial(
    base_output_dir: Path,
    debris_method: str = "gmm",
    save_plots: bool = True,
    export_clean_fcs: bool = True,
    viability_cutoff: Optional[float] = None,
) -> None:
    """Run full 5-step pipeline on TQC Basic Tutorial Dataset (8 BD LSRII samples)."""
    ds_out = base_output_dir / "dataset_tqc_tutorial"
    tqc_dir = Path("data/conventional/tqc_2020_basic_tutorial/raw")
    tqc_files = sorted(tqc_dir.glob("*.fcs"))

    print("\n" + "=" * 80)
    print(f"PROCESSING DATASET: TQC Basic Tutorial Data ({len(tqc_files)} BD LSRII Samples)")
    print(f"Destination Root: {ds_out}")
    print("=" * 80)

    # Step 1: Inspection
    print("\n[Step 1/5] Running Dataset Inspection...")
    t0 = time.perf_counter()
    run_step1_inspection(tqc_files, ds_out / "inspection")
    print(f"  Inspection completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'inspection'}")

    # Step 2: Time QC
    print("\n[Step 2/5] Running Automated Fluidic Time QC...")
    t0 = time.perf_counter()
    df_qc, qc_csv, qc_rep = run_step2_time_qc(
        tqc_files,
        ds_out / "time_qc",
        export_clean_fcs=export_clean_fcs,
        generate_plots=save_plots,
    )
    print(f"  Time QC completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'time_qc'}")

    # Step 3: Compensation
    print("\n[Step 3/5] Applying Spillover Compensation & Verification...")
    t0 = time.perf_counter()
    run_step3_flowcyt_compensation(
        tqc_files,
        ds_out / "compensation",
        report_title="Flow Cytometry Compensation Report — TQC Basic Tutorial Data",
        generate_plots=save_plots,
    )
    print(f"  Compensation completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'compensation'}")

    # Step 4: Core Clean-up Gating
    print(f"\n[Step 4/5] Executing Core Clean-Up Gating (Debris Method: {debris_method.upper()})...")
    t0 = time.perf_counter()
    clean_fcs_inputs = sorted((ds_out / "time_qc/clean_fcs").glob("*.fcs"))
    target_inputs = clean_fcs_inputs if clean_fcs_inputs else tqc_files
    df_clean, clean_csv, clean_rep = run_step4_cleanup(
        target_inputs,
        ds_out / "cleanup",
        debris_method=debris_method,
        generate_plots=save_plots,
        export_clean_fcs=export_clean_fcs,
        viability_cutoff=viability_cutoff,
    )
    print(f"  Clean-up Gating completed in {time.perf_counter()-t0:.2f}s -> {ds_out / 'cleanup'}")

    # Step 5: Batch Preparation & Multi-Day Quantile Normalization
    print("\n[Step 5/5] Executing Multi-Day Quantile Normalization & Downsampling...")
    t0 = time.perf_counter()
    clean_tqc_inputs = sorted((ds_out / "cleanup/clean_fcs").glob("*.fcs"))
    if clean_tqc_inputs:
        batch_out = ds_out / "batch"
        sample_info_list = []
        cleanup_summary_file = ds_out / "cleanup/cohort_cleanup_summary.csv"
        clean_lookup = {}
        if cleanup_summary_file.exists():
            clean_df = pd.read_csv(cleanup_summary_file)
            name_c = "sample_name" if "sample_name" in clean_df.columns else "sample_id"
            clean_lookup = {str(r[name_c]).strip(): r for _, r in clean_df.iterrows()}
        for f in clean_tqc_inputs:
            day = "Day1" if "LD1" in f.name else "Day2"
            s_name = f.stem
            row = clean_lookup.get(s_name)
            raw_c = int(row["total_events"]) if row is not None and "total_events" in row else 0
            cln_c = int(row["clean_events"]) if row is not None and "clean_events" in row else 0
            sample_info_list.append(
                CohortSampleInfo(
                    sample_id=s_name,
                    batch_id=day,
                    is_control=False,
                    filepath=str(f.resolve()),
                    raw_event_count=raw_c,
                    clean_event_count=cln_c,
                )
            )
        batch_cfg = BatchPreparationConfig(
            downsampling=DownsampleConfig(
                enabled=True,
                mode="min_sample",
                target_events=None,
                random_seed=42,
            ),
            batch_normalization=BatchCorrectionConfig(
                enabled=True,
                method="quantile",
                goal="median",
            ),
            export_individual_fcs=export_clean_fcs,
            export_pooled_cohort=True,
        )
        run_batch_preparation(
            sample_info=sample_info_list,
            config=batch_cfg,
            output_dir=batch_out,
        )
        print(f"  Batch Preparation & Normalization completed in {time.perf_counter()-t0:.2f}s -> {batch_out}")


def main():
    args = parse_args()
    base_out = Path(args.base_output_dir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()
    if args.dataset in ("all", "flowcyt"):
        process_dataset_flowcyt(
            base_output_dir=base_out,
            debris_method=args.debris_method,
            save_plots=args.save_plots,
            export_clean_fcs=args.export_clean_fcs,
            viability_cutoff=args.viability_cutoff,
        )

    if args.dataset in ("all", "fr_fcm_z2zh"):
        process_dataset_fr_fcm_z2zh(
            base_output_dir=base_out,
            debris_method=args.debris_method,
            save_plots=args.save_plots,
            export_clean_fcs=args.export_clean_fcs,
            viability_cutoff=args.viability_cutoff,
        )

    if args.dataset in ("all", "tqc_tutorial"):
        process_dataset_tqc_tutorial(
            base_output_dir=base_out,
            debris_method=args.debris_method,
            save_plots=args.save_plots,
            export_clean_fcs=args.export_clean_fcs,
            viability_cutoff=args.viability_cutoff,
        )

    print("\n" + "=" * 80)
    print(f"ALL PIPELINE RUNS COMPLETED in {time.perf_counter()-t_total:.2f}s")
    print(f"Outputs saved in: {base_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
