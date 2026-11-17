#!/usr/bin/env python3
"""Command-line script for executing Flow Cytometry Time Quality Control (QC).

Implements 1.1 Automated QC (fluidic flow rate stability, signal drift, saturation)
and 1.2 Manual QC (Time interval gating) across individual files or cohort manifests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowx.qc.models import TimeQCConfig
from flowx.qc.runner import run_cohort_time_qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlowX QC: Automated Fluidic Quality Control & Manual Time Gating",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to text manifest file (.txt/.csv) containing FCS file paths, directory of FCS files, or single FCS file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="outputs/flowcyt_preprocessing",
        help="Output directory for reports, summaries, and diagnostic plots.",
    )
    parser.add_argument(
        "--sample-type",
        "-t",
        choices=["auto", "sample", "control", "single_stain", "fmo"],
        default="auto",
        help="Type of sample / QC preset: 'auto' (default: automatically detects controls vs samples from filenames), 'sample' (full QC), 'control' / 'single_stain' (saturation + permissive flow rate, signal drift bypassed), or 'fmo'.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=None,
        help="Width of discrete time bins in seconds (default: 0.5s for samples, 0.25s for controls).",
    )
    parser.add_argument(
        "--mad-factor",
        type=float,
        default=None,
        help="Robust MAD multiplier for flow rate anomaly detection (default: 3.5 for samples, 4.5 for controls).",
    )
    parser.add_argument(
        "--min-flow-rate",
        type=float,
        default=None,
        help="Minimum events per second before flagging as a clog (default: 50 eps for samples, 20 eps for controls).",
    )
    parser.add_argument(
        "--drift-mad-factor",
        type=float,
        default=None,
        help="Robust MAD multiplier for channel signal drift detection.",
    )
    parser.add_argument(
        "--manual-time-min",
        type=float,
        default=None,
        help="Optional manual lower bound for time gate in seconds (e.g. 1.0 or 2.0 to discard warm-up).",
    )
    parser.add_argument(
        "--manual-time-max",
        type=float,
        default=None,
        help="Optional manual upper bound for time gate in seconds.",
    )
    parser.add_argument(
        "--min-control-events",
        type=int,
        default=500,
        help="Minimum clean events required for a compensation control to proceed with spillover calculation (default: 500).",
    )
    parser.add_argument(
        "--min-sample-events",
        type=int,
        default=5000,
        help="Minimum clean events required for a biological sample to proceed downstream (default: 5000).",
    )
    parser.add_argument(
        "--pass-threshold-pct",
        type=float,
        default=None,
        help="Custom pass rate percentage threshold for PASS status (default: 90.0).",
    )
    parser.add_argument(
        "--warn-threshold-pct",
        type=float,
        default=None,
        help="Custom pass rate percentage threshold for WARNING status (default: 75.0).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable rendering of publication-quality diagnostic plots for faster execution.",
    )
    parser.add_argument(
        "--export-clean-fcs",
        action="store_true",
        help="Export clean, time-gated .fcs files into a clean_fcs/ subfolder for seamless multi-step pipelines.",
    )
    parser.add_argument(
        "--clean-fcs-dir",
        type=str,
        default=None,
        help="Custom destination directory for exported clean .fcs files (defaults to <output_dir>/clean_fcs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Collect explicitly provided overrides
    overrides = {}
    if args.bin_width is not None:
        overrides["bin_width_sec"] = args.bin_width
    if args.mad_factor is not None:
        overrides["flow_rate_mad_factor"] = args.mad_factor
    if args.min_flow_rate is not None:
        overrides["flow_rate_min_eps"] = args.min_flow_rate
    if args.drift_mad_factor is not None:
        overrides["signal_drift_mad_factor"] = args.drift_mad_factor
    if args.manual_time_min is not None:
        overrides["manual_time_min"] = args.manual_time_min
    if args.manual_time_max is not None:
        overrides["manual_time_max"] = args.manual_time_max
    if args.pass_threshold_pct is not None:
        overrides["pass_threshold_pct"] = args.pass_threshold_pct
    if args.warn_threshold_pct is not None:
        overrides["warn_threshold_pct"] = args.warn_threshold_pct

    if args.sample_type == "auto":
        config = TimeQCConfig(sample_type="auto", **overrides)
    else:
        config = TimeQCConfig.from_sample_type(args.sample_type, **overrides)

    print("=" * 70)
    print("FlowX QC: Automated Fluidic Quality Control & Manual Time Gating")
    print(f"Input: {args.input}")
    print(f"Sample Type Preset: {args.sample_type.upper()}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Time Bin Width: {config.bin_width_sec}s | Flow Rate MAD Multiplier: {config.flow_rate_mad_factor}")
    print(f"Signal Drift Monitored: {config.enable_signal_drift} | Margin Saturation Trimming: {config.trim_margins}")
    print(f"Thresholds: Control >= {args.min_control_events} events | Sample >= {args.min_sample_events} events")
    print("=" * 70)

    try:
        summary_df, csv_path, md_path = run_cohort_time_qc(
            file_list_or_paths=args.input,
            output_dir=args.output_dir,
            config=config,
            generate_plots=not args.no_plots,
            min_control_events=args.min_control_events,
            min_sample_events=args.min_sample_events,
            export_clean_fcs=getattr(args, "export_clean_fcs", False),
            clean_fcs_dir=getattr(args, "clean_fcs_dir", None),
        )

        total_ev = summary_df["total_events"].sum()
        passed_ev = summary_df["passed_events"].sum()
        passed_pct = (passed_ev / total_ev * 100.0) if total_ev > 0 else 0.0

        n_clean = (summary_df["workflow_action"] == "PROCEED_CLEAN").sum() if "workflow_action" in summary_df.columns else 0
        n_caution = (summary_df["workflow_action"] == "PROCEED_WITH_CAUTION").sum() if "workflow_action" in summary_df.columns else 0
        n_salvage = (summary_df["workflow_action"] == "PROCEED_MASKED_SALVAGE").sum() if "workflow_action" in summary_df.columns else 0
        n_reject = (summary_df["workflow_action"].isin(["FALLBACK_INSTRUMENT_MATRIX", "REJECT_SAMPLE"])).sum() if "workflow_action" in summary_df.columns else 0

        print("\n" + "=" * 70)
        print("QC EXECUTION SUMMARY")
        print(f"Total Files Processed: {len(summary_df)}")
        print(f"Total Events: {total_ev:,}")
        print(f"Events Passing Quality Control: {passed_ev:,} ({passed_pct:.2f}%)")
        print(f"Events Rejected: {total_ev - passed_ev:,} ({100.0 - passed_pct:.2f}%)")
        print(f"Workflow Decisions: {n_clean} Proceed Clean │ {n_caution} Caution │ {n_salvage} Masked Salvage │ {n_reject} Fallback/Reject")
        print(f"Summary CSV: {csv_path}")
        print(f"Detailed Report: {md_path}")
        print(f"Diagnostic Plots: {Path(args.output_dir) / 'plots'}")
        if getattr(args, "export_clean_fcs", False):
            clean_dir_display = Path(args.clean_fcs_dir) if args.clean_fcs_dir else (Path(args.output_dir) / "clean_fcs")
            print(f"Exported Clean FCS Files: {clean_dir_display}")
        print("=" * 70)
        return 0
    except Exception as exc:
        print(f"\nExecution error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
