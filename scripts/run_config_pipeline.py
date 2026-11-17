#!/usr/bin/env python3
"""Universal Config-Driven Flow Cytometry Pipeline Runner.

Enables universal execution across any conventional or spectral flow cytometry dataset
where the user only supplies and edits a YAML/JSON configuration file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root and src to sys.path
_project_root = Path(__file__).resolve().parent.parent
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from flowx.pipeline import FlowCytometryPipeline
from flowx.io.paths import sanitize_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FlowX analysis pipeline using a user-specified configuration file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to YAML or JSON configuration file (e.g. configs/config_run/conventional/coburn_2021_thp1_stiffness.yaml).",
    )
    parser.add_argument(
        "--fcs-file",
        type=str,
        default=None,
        help="Optional override for target raw FCS file (defaults to fcs_file in config).",
    )
    parser.add_argument(
        "--discovery",
        dest="discovery",
        action="store_true",
        default=None,
        help="Enable unsupervised Population Discovery phase (FlowSOM/PhenoGraph clustering & UMAP).",
    )
    parser.add_argument(
        "--no-discovery",
        dest="discovery",
        action="store_false",
        help="Disable unsupervised Population Discovery phase (default).",
    )
    parser.add_argument(
        "--qc",
        dest="qc",
        action="store_true",
        default=None,
        help="Enable Quality Control phase (Time QC, clean-up gating). Default is enabled.",
    )
    parser.add_argument(
        "--no-qc",
        dest="qc",
        action="store_false",
        help="Disable Quality Control phase.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"Error: Configuration file not found at '{args.config}'", file=sys.stderr)
        return 1

    print("=" * 70)
    print(" FlowX Constant Config Runner")
    print("=" * 70)
    print(f" Config File: {sanitize_path(config_path)}")

    t0 = time.perf_counter()
    try:
        pipeline, target_fcs = FlowCytometryPipeline.from_config(config_path)
        fcs_to_run = args.fcs_file or target_fcs
        if not fcs_to_run:
            print("Error: No FCS file specified in config (input.fcs_file) or via --fcs-file.", file=sys.stderr)
            return 1

        # Check phase overrides
        enable_qc = args.qc if args.qc is not None else pipeline.enable_qc
        enable_discovery = args.discovery if args.discovery is not None else pipeline.enable_discovery

        print(f" Target FCS:  {sanitize_path(fcs_to_run)}")
        print(f" Output Dir:  {sanitize_path(pipeline.output_dir)}")
        print(f" Gating Mode: {pipeline.gating_mode.upper()} | Transform: {pipeline.transform_type}")
        print(f" Phases:      QC={'ENABLED' if enable_qc else 'DISABLED'} | Discovery={'ENABLED' if enable_discovery else 'DISABLED'}")
        print("-" * 70)

        result = pipeline.run(
            fcs_filepath=fcs_to_run,
            enable_qc=enable_qc,
            enable_discovery=enable_discovery,
        )

        duration = time.perf_counter() - t0
        print("-" * 70)
        print(" Run Completed Successfully!")
        print(f" Total Events:       {result.event_count:,}")
        print(f" Execution Duration: {result.execution_duration_formatted} (elapsed: {duration:.2f}s)")
        print(f" Deliverables in:    {sanitize_path(result.run_dir)}")
        print(f" Log File:           {sanitize_path(result.log_file_path or '')}")
        print("=" * 70)
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Pipeline run failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
