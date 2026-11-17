#!/usr/bin/env python3
"""Standalone CLI script for inspecting flow cytometry (.fcs) files, directories, and cohorts.

Usage:
    # Inspect entire directory into outputs/flowcyt_inspection/ with population config
    python scripts/inspect_fcs.py data/conventional/bini_2024_flowcyt/raw/ -m configs/population_maps/flowcyt_benchmark.json

    # Inspect arbitrary list of files
    python scripts/inspect_fcs.py data/conventional/bini_2024_flowcyt/raw/Case1_A.fcs data/conventional/bini_2024_flowcyt/raw/Case2_A.fcs

    # Inspect from a text file manifest
    python scripts/inspect_fcs.py --files-list my_files.txt -o outputs/custom_inspection/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
_project_root = str(Path(__file__).resolve().parent.parent / "src")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from flowx.cli import cmd_inspect


def build_script_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone inspector script."""
    parser = argparse.ArgumentParser(
        prog="inspect_fcs",
        description="FlowX Screener: Inspect FCS files, detect instruments & channels, and export summary CSV & report.",
    )
    parser.add_argument(
        "targets",
        type=str,
        nargs="*",
        default=[],
        help="Path(s) to raw .fcs file(s) or directory containing .fcs files.",
    )
    parser.add_argument(
        "--files-list",
        "--file-list",
        dest="files_list",
        type=str,
        default=None,
        help="Path to text file (one filepath per line) or CSV file listing FCS files to inspect.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--output",
        dest="output_dir",
        type=str,
        default="outputs/flowcyt_inspection",
        help="Directory to save inspection summary, channel details, and report (default: outputs/flowcyt_inspection).",
    )
    parser.add_argument(
        "-m",
        "--population-map",
        "--pop-map",
        dest="population_map",
        type=str,
        default=None,
        help="Optional path to JSON/YAML population mapping configuration (e.g. configs/population_maps/flowcyt_benchmark.json).",
    )
    parser.add_argument(
        "--channels-csv",
        dest="channels_csv",
        type=str,
        default=None,
        help="Custom filename/path for detailed per-channel configuration CSV (default: channel_details.csv in output dir).",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Disable automatic generation of inspection_report.md Markdown audit.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.fcs",
        help="File pattern for directory search (default: *.fcs).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Recursively scan all nested subdirectories for FCS files (default: True).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive directory scanning; only inspect top-level target directory.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum subdirectory depth for recursive scanning (e.g. --max-depth 2).",
    )
    parser.add_argument(
        "--event-stats",
        action="store_true",
        help="Calculate deep event-level statistics (negative event rate, detector saturation).",
    )
    parser.add_argument(
        "-H",
        "--harmonize",
        type=str,
        default=None,
        help="Optional path to panel harmonization specification YAML/JSON (e.g. configs/panel_harmonization.yaml).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=25,
        help="Maximum sample rows to show in console table (default: 25).",
    )
    return parser


def main() -> int:
    """Entry point for standalone script."""
    parser = build_script_parser()
    args = parser.parse_args()
    return cmd_inspect(args)


if __name__ == "__main__":
    sys.exit(main())
