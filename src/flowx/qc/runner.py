"""Cohort runner and batch processing module for Flow Cytometry Time Quality Control.

Processes cohorts of FCS files through automated and manual time QC, generating
tabular summaries, diagnostic plot galleries, and MIFlowCyt-compliant reports.
Includes intelligent automatic detection and profiling of single-stain compensation
controls, FMOs, and biological samples.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from flowx.io.parser import FCSData, parse_fcs
from flowx.qc.engine import apply_time_qc
from flowx.qc.models import QCWorkflowDecision, TimeQCConfig, TimeQCResult
from flowx.visualization.qc_plots import plot_time_qc_diagnostics

logger = logging.getLogger(__name__)


@dataclass
class TimeQCStepResult:
    """Container holding results from an independent single-sample Time QC step."""

    clean_df: pd.DataFrame
    qc_result: Optional[TimeQCResult] = None
    decision: Optional[QCWorkflowDecision] = None
    plot_path: Optional[str] = None
    summary_csv_path: Optional[str] = None
    skipped: bool = False


def run_time_qc_step(
    fcs_data: FCSData,
    config: Optional[TimeQCConfig] = None,
    min_control_events: int = 500,
    min_sample_events: int = 5000,
    output_plot_path: Optional[Union[str, Path]] = None,
    output_summary_csv: Optional[Union[str, Path]] = None,
    generate_plots: bool = True,
) -> TimeQCStepResult:
    """Execute independent Time QC step on a parsed FCSData sample.

    Evaluates fluidic stability, signal drift, and margin saturation, renders
    diagnostic plots, and optionally writes the single-sample summary CSV.
    """
    time_ch = fcs_data.get_time_channel()
    if not time_ch or time_ch not in fcs_data.events_df.columns or len(fcs_data.events_df) < 100:
        return TimeQCStepResult(
            clean_df=fcs_data.events_df.copy(),
            qc_result=None,
            decision=None,
            skipped=True,
        )

    cfg = config or TimeQCConfig.for_sample()
    tqc_res = apply_time_qc(fcs_data, config=cfg)
    decision = tqc_res.evaluate_workflow(
        fcs_data,
        min_control_events=min_control_events,
        min_sample_events=min_sample_events,
    )
    if decision.can_proceed and decision.clean_events_df is not None:
        clean_df = decision.clean_events_df
    else:
        clean_df = fcs_data.events_df.copy()

    plot_str: Optional[str] = None
    if generate_plots and output_plot_path is not None:
        p_out = Path(output_plot_path)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        plot_time_qc_diagnostics(fcs_data, tqc_res, output_path=p_out)
        plot_str = str(p_out)

    csv_str: Optional[str] = None
    if output_summary_csv is not None:
        c_out = Path(output_summary_csv)
        c_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([tqc_res.to_summary_dict(decision=decision)]).to_csv(c_out, index=False)
        csv_str = str(c_out)

    return TimeQCStepResult(
        clean_df=clean_df,
        qc_result=tqc_res,
        decision=decision,
        plot_path=plot_str,
        summary_csv_path=csv_str,
        skipped=False,
    )


def detect_sample_type_from_path(file_path: Path) -> str:
    """Infer sample type ('control', 'fmo', or 'sample') from file path and naming conventions.

    Args:
        file_path: Path object of FCS file.

    Returns:
        Sample type string ('control', 'fmo', or 'sample').
    """
    name_low = file_path.name.lower()
    parent_low = file_path.parent.name.lower()

    if any(k in name_low for k in ["control", "unstained", "single_stain", "singlestain", "comp_ctrl", "compensation controls"]) or "control" in parent_low:
        if "fmo" in name_low:
            return "fmo"
        return "control"
    elif "fmo" in name_low:
        return "fmo"
    return "sample"


def run_cohort_time_qc(
    file_list_or_paths: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path] = "outputs/flowcyt_preprocessing",
    config: Optional[TimeQCConfig] = None,
    generate_plots: bool = True,
    min_control_events: int = 500,
    min_sample_events: int = 5000,
    export_clean_fcs: bool = False,
    clean_fcs_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Path, Path]:
    """Execute Time Quality Control across a cohort of FCS files.

    Args:
        file_list_or_paths: Filepath to text file with FCS paths, or list of FCS filepaths.
        output_dir: Target directory to save reports and diagnostic plots.
        config: Optional TimeQCConfig configuration override. If None or sample_type='auto',
            automatically applies tailored QC profiles to controls vs samples.
        generate_plots: If True, renders publication-quality diagnostic plots.
        min_control_events: Minimum clean events required for a compensation control to proceed (default: 500).
        min_sample_events: Minimum clean events required for a biological sample to proceed (default: 5000).
        export_clean_fcs: If True, exports clean time-gated events as standardized .fcs files.
        clean_fcs_dir: Destination folder for exported clean .fcs files (defaults to output_dir/clean_fcs).

    Returns:
        Tuple of (summary_dataframe, summary_csv_path, report_md_path).
    """
    out_dir = Path(output_dir).resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    resolved_clean_fcs_dir = Path(clean_fcs_dir).resolve() if clean_fcs_dir else out_dir / "clean_fcs"
    if export_clean_fcs:
        resolved_clean_fcs_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or TimeQCConfig(sample_type="auto")
    auto_detect_type = (cfg.sample_type == "auto")

    # Resolve input file paths
    fcs_files: List[Path] = []
    if isinstance(file_list_or_paths, (str, Path)):
        p = Path(file_list_or_paths)
        if p.is_file() and p.suffix in [".txt", ".csv"]:
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

    logger.info(f"Starting Time QC cohort analysis for {len(fcs_files)} files...")
    t0 = time.time()

    summary_rows: List[Dict[str, Any]] = []
    qc_results: List[TimeQCResult] = []

    for file_path in tqdm(fcs_files, desc="Running Time QC", unit="file"):
        try:
            if not file_path.is_file():
                logger.warning(f"File not found: {file_path}. Skipping.")
                continue

            # Resolve per-file configuration
            if auto_detect_type:
                st = detect_sample_type_from_path(file_path)
                file_cfg = TimeQCConfig.from_sample_type(st)
                # Propagate all user overrides provided in cfg (T8)
                default_cfg = TimeQCConfig()
                for field_name in cfg.__dataclass_fields__:
                    if field_name not in ("sample_type",):
                        user_val = getattr(cfg, field_name)
                        def_val = getattr(default_cfg, field_name)
                        if user_val != def_val:
                            setattr(file_cfg, field_name, user_val)
            else:
                file_cfg = cfg

            fcs_data = parse_fcs(file_path)
            res = apply_time_qc(fcs_data, config=file_cfg)
            decision = res.evaluate_workflow(
                fcs_data, min_control_events=min_control_events, min_sample_events=min_sample_events
            )
            qc_results.append(res)

            clean_fcs_path = ""
            if export_clean_fcs and decision.can_proceed and decision.clean_events_df is not None:
                target_clean_fcs = resolved_clean_fcs_dir / file_path.name
                fcs_data.export_fcs(target_clean_fcs, events_df=decision.clean_events_df)
                clean_fcs_path = str(target_clean_fcs)
                logger.info(f"Exported clean FCS ({len(decision.clean_events_df):,} events) to '{target_clean_fcs}'")

            row_summary = res.to_summary_dict(decision=decision)
            row_summary["clean_fcs_path"] = clean_fcs_path
            summary_rows.append(row_summary)

            if generate_plots:
                stem = file_path.stem
                plot_file = plots_dir / f"{stem}_qc_time.png"
                plot_time_qc_diagnostics(fcs_data, res, output_path=plot_file)

        except Exception as exc:
            logger.error(f"Error processing {file_path.name}: {exc}", exc_info=True)
            summary_rows.append(
                {
                    "sample_name": file_path.name,
                    "sample_type": "unknown",
                    "total_events": 0,
                    "passed_events": 0,
                    "rejected_events": 0,
                    "passed_pct": 0.0,
                    "flow_rate_anomalies": 0,
                    "flow_rate_anomalies_pct": 0.0,
                    "signal_drift_anomalies": 0,
                    "signal_drift_anomalies_pct": 0.0,
                    "margin_saturation_events": 0,
                    "margin_saturation_pct": 0.0,
                    "manual_time_gated_events": 0,
                    "manual_time_gated_pct": 0.0,
                    "duration_sec": 0.0,
                    "time_channel": "N/A",
                    "qc_status": "ERROR",
                    "workflow_action": "REJECT_SAMPLE",
                    "recommended_action": f"Processing error: {exc}",
                    "qc_messages": str(exc),
                    "clean_fcs_path": "",
                }
            )

    elapsed = time.time() - t0
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "qc_time_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    report_md = out_dir / "qc_time_report.md"
    generate_qc_report(summary_df, qc_results, report_md, elapsed_sec=elapsed)

    logger.info(f"Completed Time QC for {len(fcs_files)} files in {elapsed:.1f}s. Report: '{report_md}'")
    return summary_df, summary_csv, report_md


def generate_qc_report(
    summary_df: pd.DataFrame,
    qc_results: List[TimeQCResult],
    output_path: Path,
    elapsed_sec: float,
) -> None:
    """Generate comprehensive MIFlowCyt-compliant Markdown report summarizing cohort Time QC."""
    total_files = len(summary_df)
    total_events = summary_df["total_events"].sum()
    total_passed = summary_df["passed_events"].sum()
    total_rejected = summary_df["rejected_events"].sum()
    overall_pass_pct = (total_passed / total_events * 100.0) if total_events > 0 else 0.0

    pass_count = (summary_df["qc_status"] == "PASS").sum()
    warn_count = (summary_df["qc_status"] == "WARNING").sum()
    fail_count = (summary_df["qc_status"] == "FAIL").sum()
    err_count = (summary_df["qc_status"] == "ERROR").sum()

    n_ctrl = (summary_df["sample_type"] == "control").sum() if "sample_type" in summary_df.columns else 0
    n_sample = (summary_df["sample_type"] == "sample").sum() if "sample_type" in summary_df.columns else 0
    n_fmo = (summary_df["sample_type"] == "fmo").sum() if "sample_type" in summary_df.columns else 0

    tot_flow_anom = summary_df["flow_rate_anomalies"].sum()
    tot_drift_anom = summary_df["signal_drift_anomalies"].sum()
    tot_margin = summary_df["margin_saturation_events"].sum()
    n_proceed_clean = int((summary_df["workflow_action"] == "PROCEED_CLEAN").sum()) if "workflow_action" in summary_df.columns else pass_count
    n_proceed_caution = int((summary_df["workflow_action"] == "PROCEED_WITH_CAUTION").sum()) if "workflow_action" in summary_df.columns else warn_count
    n_proceed_salvage = int((summary_df["workflow_action"] == "PROCEED_MASKED_SALVAGE").sum()) if "workflow_action" in summary_df.columns else 0
    n_fallback = int((summary_df["workflow_action"].isin(["FALLBACK_INSTRUMENT_MATRIX", "REJECT_SAMPLE"])).sum()) if "workflow_action" in summary_df.columns else 0

    lines: List[str] = [
        "# Flow Cytometry Time Quality Control (QC) Cohort Report",
        "",
        "> **Pipeline:** FlowX QC v1.0 (Automated & Manual Time Gating)  ",
        f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Standard Compliance:** MIFlowCyt & ISAC Standards  ",
        f"> **Summary Manifest:** [`qc_time_summary.csv`](qc_time_summary.csv)  ",
        f"> **Diagnostic Gallery:** [`plots/`](plots/)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Value |",
        "|:-------|:------|",
        f"| **Total Evaluated Files** | {total_files} files |",
        f"| **Cohort Composition** | **{n_sample} Biological Samples** │ **{n_ctrl} Compensation Controls** │ **{n_fmo} FMO Controls** |",
        f"| **Total Recorded Events** | {total_events:,} events |",
        f"| **Clean / Passed Events** | **{total_passed:,} events ({overall_pass_pct:.2f}%)** |",
        f"| **Excluded Events (All QC)** | {total_rejected:,} events ({100.0 - overall_pass_pct:.2f}%) |",
        f"| **Flow Rate / Fluidic Anomalies** | {tot_flow_anom:,} events ({(tot_flow_anom / total_events)*100:.2f}%) |",
        f"| **Signal Drift Anomalies** | {tot_drift_anom:,} events ({(tot_drift_anom / total_events)*100:.2f}%) |",
        f"| **Margin / Saturation Artifacts** | {tot_margin:,} events ({(tot_margin / total_events)*100:.2f}%) |",
        f"| **Cohort QC Quality Status** | **{pass_count} PASS** │ **{warn_count} WARNING** │ **{fail_count + err_count} FLAG/FAIL** |",
        f"| **Workflow Decisions** | **{n_proceed_clean} Proceed Clean** │ **{n_proceed_caution} Proceed with Caution** │ **{n_proceed_salvage} Proceed Masked Salvage** │ **{n_fallback} Fallback/Reject** |",
        f"| **Processing Execution Time** | {elapsed_sec:.1f} seconds ({elapsed_sec / max(1, total_files):.2f}s / file) |",
        "",
        "---",
        "",
        "## 2. Methodological Standards (MIFlowCyt Preprocessing Adherence)",
        "",
        "1. **Flow Rate Stability (flowAI / flowCut Methodology)**:  ",
        "   Instantaneous event rates were computed across discrete time intervals (0.5s for samples, 0.25s for controls). Time bins exhibiting event rates outside the robust envelope ($Median \\pm MAD \\times factor$) or experiencing fluidic clogs were automatically excised.",
        "",
        "2. **Signal Drift Monitoring (PeacoQC Methodology)**:  ",
        "   Fluorescence intensities were monitored for baseline shifts across all operational channels on biological samples and FMOs. Single-stain controls bypassed multi-channel drift monitoring to protect single-positive bead populations from false-positive clipping.",
        "",
        "3. **Margin & Dynamic Range Filtering**:  ",
        "   Events saturated on instrument hardware thresholds ($PnR$) or lower sensor floor were purged to prevent downstream biexponential/logicle distortion and compensation slope flattening.",
        "",
        "---",
        "",
        "## 3. Sample-Level Quality Control Audit Roster",
        "",
        "| Sample File | Type | Total Events | Passed Events | Pass Rate | Status | Workflow Action | Operational Guidance | Diagnostic Plot |",
        "|:------------|:----:|:------------:|:-------------:|:---------:|:------:|:---------------:|:---------------------|:---------------:|",
    ]

    for _, row in summary_df.iterrows():
        s_name = row["sample_name"]
        st = f"`{row.get('sample_type', 'sample').upper()}`"
        t_ev = f"{int(row['total_events']):,}"
        p_ev = f"{int(row['passed_events']):,}"
        p_rate = f"{row['passed_pct']:.1f}%"
        stat = f"`{row['qc_status']}`"
        action = f"`{row.get('workflow_action', 'PROCEED_CLEAN')}`"
        rec = str(row.get("recommended_action", "Proceed with clean events."))
        rec_first = rec.split(". ")[0] + ("." if not rec.endswith(".") else "")
        stem = Path(s_name).stem
        plot_link = f"[`{stem}_qc_time.png`](plots/{stem}_qc_time.png)"

        lines.append(
            f"| `{s_name}` | {st} | {t_ev} | {p_ev} | **{p_rate}** | {stat} | {action} | {rec_first} | {plot_link} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Operational Workflow Actions & Downstream Usage",
            "",
            "### How to Handle Passed vs Flagged Files with `passed_mask`:",
            "",
            "1. **`PROCEED_CLEAN` (Pass Rate >= 90%):**  ",
            "   - **Action:** Subset raw events using `events_df[result.passed_mask]`.  ",
            "   - **Usage:** Feed directly into **Step 1.3 (Compensation)** and **Step 1.4 (Transformation)** without manual intervention.",
            "",
            "2. **`PROCEED_WITH_CAUTION` (Pass Rate 75% - 90%):**  ",
            "   - **Action:** Proceed with `events_df[result.passed_mask]`.  ",
            "   - **Usage:** Moderate fluidic turbulence or transient spikes were excised. Review the linked diagnostic plot to confirm remaining event count satisfies downstream population statistics.",
            "",
            "3. **`PROCEED_MASKED_SALVAGE` (Pass Rate < 75% with sufficient clean events):**  ",
            "   - **Controls (>=500 clean events):** Single-stain beads captured during the stable interval are retained. Excising the stalled/turbulent interval **prevents** corrupted MFIs and distorted spillover slopes. Proceed with masked events for spillover matrix calculation.  ",
            "   - **Biological Samples (>=5,000 clean events):** Stable cell events are salvaged. Downstream gating proceeds with masked events if target cell morphology (FSC/SSC) is preserved.",
            "",
            "4. **`FALLBACK_INSTRUMENT_MATRIX` (Controls < 500 clean events):**  ",
            "   - **Action:** Do not calculate spillover from this control. Automatically fall back to the cytometer's embedded `$SPILLOVER` / `$COMP` keyword matrix recorded in the FCS header, or use a matched control from another well.",
            "",
            "5. **`REJECT_SAMPLE` (Samples < 5,000 clean events or severe multi-channel signal drift):**  ",
            "   - **Action:** Exclude sample from downstream analysis to prevent misclassification; document exclusion per MIFlowCyt guidelines or re-acquire sample.",
            "",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
