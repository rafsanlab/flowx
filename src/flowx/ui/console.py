"""Biotech Laboratory terminal UI presentation module for FlowX.

Provides rich styling, molecular icons, ASCII yield bars, and structured tables
matching the 'Biotech Laboratory' (Option 3) visual design language.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from flowx.io.logger import DEFAULT_LOGGER_NAME, format_duration, set_console_level_override
from flowx.io.paths import sanitize_path

# Standard shared console instance
_console: Optional[Console] = None
MAX_UI_WIDTH = 80


def get_console() -> Console:
    """Retrieve or create the shared Rich console instance."""
    global _console
    if _console is None:
        _console = Console()
    return _console


def _ui_width(console: Console) -> int:
    """Return the capped UI width (maximum 80 characters)."""
    w = getattr(console.size, "width", MAX_UI_WIDTH) or MAX_UI_WIDTH
    return min(MAX_UI_WIDTH, max(40, w))



# ---------------------------------------------------------------------------
# FlowX colour scheme (white + orange on a dark charcoal terminal)
# ---------------------------------------------------------------------------
COLOR_ACCENT = "#ff6600"   # primary orange
COLOR_TEXT = "#f9f9f9"     # near-white
COLOR_TINT_1 = "#ff9955"   # light orange
COLOR_TINT_2 = "#ffccaa"   # pale orange
COLOR_TINT_3 = "#ffe6d5"   # blush
COLOR_MUTED = "#777777"    # grey text
COLOR_BORDER = "#444444"   # dark grey box borders

# Pixel-art "FX" logo: W=white, O=orange, 1/2/3 = orange tints (blush→light), .=empty.
# Faithful 7x7 block grid matching the FlowX emblem:
# F top bar is 5 squares wide; stem is 7 squares tall.
# X is a 5x5 diagonal cross whose center aligns with the F's gradient crossbar.
_LOGO_PIXELS: List[str] = [
    "WWWWW..",
    "W.O...O",
    "W..O.O.",
    "W321O..",
    "W..O.O.",
    "W.O...O",
    "W......",
]
_LOGO_COLORS: Dict[str, str] = {
    "W": COLOR_TEXT,
    "O": COLOR_ACCENT,
    "1": COLOR_TINT_1,
    "2": COLOR_TINT_2,
    "3": COLOR_TINT_3,
}


def build_logo(wordmark: str = "FLOWX") -> Text:
    """Build the pixel-art FX text logo with the module name beside it.

    Args:
        wordmark: Module name rendered in capital letters next to the logo.

    Returns:
        Rich ``Text`` containing the multi-line colour logo.
    """
    name_row = 2
    out = Text()
    for r, row in enumerate(_LOGO_PIXELS):
        for ch in row:
            color = _LOGO_COLORS.get(ch)
            out.append("██" if color else "  ", style=color or "")
        if r == name_row:
            out.append("    ")
            out.append(wordmark[:-1], style=f"bold {COLOR_TEXT}")
            out.append(wordmark[-1], style=f"bold {COLOR_ACCENT}")
        elif r == name_row + 1:
            out.append("    Computational Flow Cytometry", style=COLOR_MUTED)
        elif r == name_row + 2:
            out.append("    Developed by Rafsanlab (2026)", style=COLOR_MUTED)
        if r < len(_LOGO_PIXELS) - 1:
            out.append("\n")
    return out


def render_logo(console: Optional[Console] = None, width: Optional[int] = None) -> None:
    """Print the FX pixel logo and FLOWX wordmark enclosed in an orange-bordered panel."""
    c = console or get_console()
    panel_width = min(MAX_UI_WIDTH, width) if width is not None else _ui_width(c)
    panel = Panel(
        Align.center(build_logo()),
        title="[bold #ff6600]◈ FlowX Platform[/bold #ff6600]",
        title_align="center",
        border_style=COLOR_ACCENT,
        box=box.ROUNDED,
        padding=(1, 2),
        width=panel_width,
        expand=True,
    )
    c.print(panel)


def make_yield_bar(pct: float, length: int = 12) -> str:
    """Generate a visual ASCII percentage fill bar (e.g. [█████████░░░])."""
    try:
        val = float(pct)
    except (ValueError, TypeError):
        val = 0.0
    filled = int(round((val / 100.0) * length))
    filled = max(0, min(length, filled))
    return f"[{'█' * filled}{'░' * (length - filled)}]"


def render_banner(
    sample_name: str,
    mode: str,
    instrument: str = "FCS Header / Standard",
    technology: str = "Auto-detected",
    console: Optional[Console] = None,
) -> None:
    """Render the header logo and metadata grid panels aligned and styled."""
    c = console or get_console()
    panel_width = _ui_width(c)

    # 1. Logo Panel (only orange-bordered box)
    render_logo(c, width=panel_width)

    # 2. Run Details Panel (dark grey border, one line per field, no blank gap)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=COLOR_MUTED, justify="left", width=10)
    grid.add_column(style=f"bold {COLOR_ACCENT}", justify="left")

    grid.add_row("Sample:", sample_name)
    grid.add_row("Model:", mode.upper())
    grid.add_row("Tech:", technology)
    grid.add_row("Hardware:", instrument)

    details_panel = Panel(
        grid,
        title="[dim white]Run Details[/dim white]",
        title_align="left",
        border_style=COLOR_BORDER,
        box=box.ROUNDED,
        padding=(0, 2),
        width=panel_width,
        expand=True,
    )
    c.print(details_panel)


def render_cli_home(console: Optional[Console] = None) -> None:
    """Render the unified 80-char FlowX CLI landing screen when invoked without subcommands."""
    c = console or get_console()
    panel_width = _ui_width(c)

    # 1. Logo Panel (orange border)
    render_logo(c, width=panel_width)

    # 2. Commands Panel (dark grey border, stacked seamlessly)
    cmd_grid = Table.grid(padding=(0, 2))
    cmd_grid.add_column(style=f"bold {COLOR_ACCENT}", justify="left", width=16)
    cmd_grid.add_column(style=COLOR_TEXT, justify="left")

    commands = [
        ("pipeline", "Run end-to-end single-sample or cohort workflow"),
        ("cohort", "Cohort clustering (FlowSOM/PhenoGraph), UMAP & stats"),
        ("inspect / info", "Inspect FCS headers, detectors & panel metadata"),
        ("time-qc", "Step 1: Fluidic stability & time-gating QC"),
        ("compensate", "Step 2: Compensation / spectral unmixing"),
        ("transform", "Step 3: Logicle, biexponential, or arcsinh scaling"),
        ("cleanup", "Step 4: Debris, singlet, viability & dump gating"),
        ("batch-prepare", "Step 4b: Batch drift check, CytoNorm & downsampling"),
        ("replay", "Replay saved workspace (.json/.wsp) on new FCS files"),
        ("config", "Generate or validate pipeline YAML configurations"),
        ("download", "Download & verify curated benchmark FCS datasets"),
    ]
    for name, desc in commands:
        cmd_grid.add_row(name, desc)

    cmd_panel = Panel(
        cmd_grid,
        title="[dim white]Commands[/dim white]",
        title_align="left",
        border_style=COLOR_BORDER,
        box=box.ROUNDED,
        padding=(0, 2),
        width=panel_width,
        expand=True,
    )
    c.print(cmd_panel)

    # 3. Quick Start Panel (dark grey border, stacked seamlessly)
    usage = Text()
    usage.append("• Run pipeline:  ", style=COLOR_MUTED)
    usage.append("flowx pipeline --config <config.yaml>\n", style=f"bold {COLOR_TEXT}")
    usage.append("• Command help:  ", style=COLOR_MUTED)
    usage.append("flowx <command> --help", style=f"bold {COLOR_TEXT}")

    usage_panel = Panel(
        usage,
        title="[dim white]Quick Start[/dim white]",
        title_align="left",
        border_style=COLOR_BORDER,
        box=box.ROUNDED,
        padding=(0, 2),
        width=panel_width,
        expand=True,
    )
    c.print(usage_panel)


def render_parser_help(parser: Any, file: Optional[Any] = None) -> None:
    """Render standard argparse --help layout (no boxes) using the FlowX colour palette."""
    import argparse
    import shutil
    import textwrap

    out_stream = file if file is not None else sys.stdout
    width = max(40, shutil.get_terminal_size(fallback=(100, 24)).columns)
    c = Console(file=out_stream, width=width, highlight=False)

    # 1. Usage line (preserve argparse's exact indentation, no Rich double-wrapping)
    usage_str = parser.format_usage().rstrip()
    if usage_str.lower().startswith("usage:"):
        usage_body = usage_str[6:].lstrip()
        u_txt = Text()
        u_txt.append("usage: ", style=f"bold {COLOR_ACCENT}")
        u_txt.append(usage_body, style=COLOR_TEXT)
        c.print(u_txt, soft_wrap=True)
    else:
        c.print(Text(usage_str, style=COLOR_TEXT), soft_wrap=True)

    if getattr(parser, "description", None):
        c.print()
        c.print(Text(parser.description, style=COLOR_MUTED))

    # 2. Classic argparse action groups (no boxes, 2-space indent, 24-char action col)
    action_col_w = 24
    for group in getattr(parser, "_action_groups", []):
        actions = [a for a in group._group_actions if a.help != argparse.SUPPRESS]
        if not actions:
            continue

        c.print()
        c.print(Text(f"{group.title}:", style=f"bold {COLOR_ACCENT}"))

        for action in actions:
            left = Text("  ")
            if action.option_strings:
                left.append(", ".join(action.option_strings), style=f"bold {COLOR_ACCENT}")
                if action.nargs != 0:
                    metavar = action.metavar or (action.dest.upper() if action.dest else "VAL")
                    if isinstance(metavar, tuple):
                        metavar = " ".join(metavar)
                    left.append(f" {metavar}", style=COLOR_TINT_1)
            else:
                metavar = action.metavar or action.dest
                left.append(str(metavar), style=f"bold {COLOR_ACCENT}")

            help_msg = (action.help or "").replace("%%", "%")
            if "%(default)" in help_msg:
                try:
                    help_msg = help_msg % {"default": action.default}
                except Exception:
                    pass
            if action.choices:
                choices_str = ", ".join(str(ch) for ch in action.choices)
                help_msg = f"{help_msg} (choices: {choices_str})".strip()

            wrap_w = max(20, width - action_col_w)
            wrapped = textwrap.wrap(help_msg, width=wrap_w) if help_msg else [""]

            if len(left.plain) < action_col_w:
                pad = " " * (action_col_w - len(left.plain))
                line = Text()
                line.append_text(left)
                line.append(pad)
                line.append(wrapped[0], style=COLOR_TEXT)
                c.print(line)
                for cont in wrapped[1:]:
                    c.print(Text(" " * action_col_w + cont, style=COLOR_TEXT))
            else:
                c.print(left)
                for cont in wrapped:
                    c.print(Text(" " * action_col_w + cont, style=COLOR_TEXT))


def print_step(
    step_idx: int,
    total_steps: int,
    title: str,
    duration: Optional[float] = None,
    sub_items: Optional[List[Tuple[str, str]]] = None,
    console: Optional[Console] = None,
) -> None:
    """Print an execution step with molecular glyphs, dot leaders, and timing."""
    c = console or get_console()

    dot_count = max(1, 54 - len(title))
    dots = "." * dot_count

    line = Text()
    line.append(f"  [{step_idx}/{total_steps}] ", style="dim #ff6600")
    line.append("◈ ", style="bold #ff6600")
    line.append(f"{title} ", style="white")
    line.append(dots + " ", style="dim gray")

    if duration is not None:
        line.append(f"[DONE {duration:.2f}s]", style="bold #ff9955")
    else:
        line.append("[RUNNING]", style="bold yellow")
    c.print(line)

    if sub_items:
        for i, (label, val) in enumerate(sub_items):
            branch = "└─ " if i == len(sub_items) - 1 else "├─ "
            sub_line = Text()
            sub_line.append("        " + branch, style="dim #ff6600")
            sub_line.append(f"{label}: ", style="dim white")
            sub_line.append(str(val), style="bold #ff6600")
            c.print(sub_line)


def render_subpopulation_matrix(
    res: Any,
    console: Optional[Console] = None,
) -> None:
    """Render the Discovered Subpopulation Matrix table with visual yield bars."""
    c = console or get_console()

    # Optional: Adaptive Cutoffs Table
    if getattr(res, "adaptive_cutoffs", None):
        ac_table = Table(
            title="◈ Adaptive Density Valley-Seeking Cutoffs",
            title_style="dim white",
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            header_style="bold white",
            width=_ui_width(c),
        )
        ac_table.add_column("Gate Name", style="bold white")
        ac_table.add_column("Channel", style="dim white")
        ac_table.add_column("Cutoff", style="bold #ff9955", justify="right")
        ac_table.add_column("Peaks", style="#ff6600")
        ac_table.add_column("Valley", style="#ff6600")
        ac_table.add_column("Confidence", style="bold #ff9955", justify="right")
        for gate_name, cut_res in res.adaptive_cutoffs.items():
            peaks_str = ", ".join(f"{p:.3f}" for p in cut_res.peaks) if cut_res.peaks else "None"
            valley_str = f"{cut_res.valley:.3f}" if cut_res.valley is not None else "N/A"
            ac_table.add_row(
                gate_name,
                cut_res.channel,
                f"{cut_res.cutoff:.3f}",
                f"[{peaks_str}]",
                valley_str,
                f"{cut_res.confidence:.2f}",
            )
        c.print(ac_table)

    table = Table(
        title="Discovered Subpopulation Matrix",
        title_style="dim white",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold white",
        show_lines=False,
        width=_ui_width(c),
    )

    # 1. SOM Metaclusters & Biological Macro-Groups
    if res.som_result and res.som_result.macro_groups_df is not None and not res.som_result.macro_groups_df.empty:
        table.add_column("Cluster", style="bold white", justify="center")
        table.add_column("Lineage Designation", style="bold #ff9955")
        table.add_column("Cells", style="white", justify="right")
        table.add_column("% Total", style="#ff6600", justify="right")
        table.add_column("Visual Yield", style="bold #ff9955", justify="left")

        df = res.som_result.macro_groups_df
        for _, row in df.iterrows():
            pct = float(row.get("% Total Events", 0.0))
            cells = f"{int(row.get('Total Cells', 0)):,}"
            clusters = str(row.get("Clusters", row.get("Macro Group ID", "")))
            lineage = str(row.get("Lineage Name", "Unassigned"))
            table.add_row(clusters, lineage, cells, f"{pct:.1f}%", make_yield_bar(pct))

    elif res.som_result and not res.som_result.cluster_statistics.empty:
        table.add_column("Cluster", style="bold white", justify="center")
        table.add_column("Designation", style="bold #ff9955")
        table.add_column("Cells", style="white", justify="right")
        table.add_column("% Total", style="#ff6600", justify="right")
        table.add_column("Visual Yield", style="bold #ff9955", justify="left")

        df = res.som_result.cluster_statistics
        for _, row in df.iterrows():
            cid = f"C{int(row.get('Cluster ID', 0))}"
            pct = float(row.get("% Total", 0.0))
            cells = f"{int(row.get('Count', 0)):,}"
            lineage = str(row.get("Phenotype", f"Metacluster {cid}"))
            table.add_row(cid, lineage, cells, f"{pct:.1f}%", make_yield_bar(pct))

    # 2. Template Gating Hierarchy
    elif res.statistics_df is not None and not res.statistics_df.empty:
        table.add_column("Population", style="bold white", justify="left")
        table.add_column("Parent", style="dim #ff6600", justify="left")
        table.add_column("Count", style="white", justify="right")
        table.add_column("% Parent", style="#ff6600", justify="right")
        table.add_column("% Total", style="#ff6600", justify="right")
        table.add_column("Visual Yield", style="bold #ff9955", justify="left")

        df = res.statistics_df[res.statistics_df["Gate"].str.lower() != "root"]
        for _, row in df.iterrows():
            gate = str(row["Gate"])
            parent = str(row.get("Parent", "-"))
            cnt = f"{int(row['Count']):,}"
            pct_p = f"{float(row.get('% Parent', 0.0)):.1f}%"
            pct_t = float(row.get("% Total", 0.0))
            table.add_row(gate, parent, cnt, pct_p, f"{pct_t:.1f}%", make_yield_bar(pct_t))

    c.print(table)

    # 3. Optional Concordance Metrics (Compare Mode)
    if getattr(res, "three_way_metrics_df", None) is not None and not res.three_way_metrics_df.empty:
        metrics_table = Table(
            title="◈ Cross-Method Concordance Metrics (ARI & NMI)",
            title_style="dim white",
            box=box.ROUNDED,
            border_style=COLOR_BORDER,
            header_style="bold white",
            width=_ui_width(c),
        )
        for col in res.three_way_metrics_df.columns:
            metrics_table.add_column(col, style="bold white" if col == "Comparison Pair" else "#ff6600")
        for _, row in res.three_way_metrics_df.iterrows():
            metrics_table.add_row(*[str(val) for val in row])
        c.print(metrics_table)


def render_artifacts_panel(
    res: Any,
    console: Optional[Console] = None,
) -> None:
    """Render the generated artifacts panel."""
    c = console or get_console()

    art_text = Text()

    # Plots
    plot_count = len(res.generated_plots)
    plots_dir = sanitize_path(Path(res.run_dir) / "plots")
    art_text.append("• Plots:      ", style="dim #ff6600")
    art_text.append(f"{plots_dir} ({plot_count} publication figures)\n", style="white")

    # Heatmap
    if "step5_som_marker_heatmap" in res.generated_plots:
        art_text.append("• Heatmap:    ", style="dim #ff6600")
        art_text.append(f"{sanitize_path(res.generated_plots['step5_som_marker_heatmap'])}\n", style="white")

    # Workspaces
    art_text.append("• Workspaces: ", style="dim #ff6600")
    art_text.append(
        f"{sanitize_path(res.workspace_json_path)} & {sanitize_path(res.workspace_xml_path)} "
        f"({'100% Replay Verified' if res.reproducibility_verified else 'Exported'})\n",
        style="white",
    )

    # Reports & Stats
    art_text.append("• Reports:    ", style="dim #ff6600")
    art_text.append(f"{sanitize_path(res.report_md_path)} & {sanitize_path(res.statistics_csv_path)}\n", style="white")

    # Audit Log
    art_text.append("• Audit Log:  ", style="dim #ff6600")
    art_text.append(f"{sanitize_path(res.log_file_path)}\n", style="white")

    # Run Duration
    dur_sec = getattr(res, "execution_time_seconds", 0.0)
    dur_fmt = getattr(res, "execution_duration_formatted", "") or format_duration(dur_sec)
    art_text.append("• Run Time:   ", style="dim #ff6600")
    art_text.append(f"{dur_fmt} ({dur_sec:.2f}s)", style="bold #ff9955")

    panel = Panel(
        art_text,
        title="[dim white]◈ Generated Pipeline Artifacts[/dim white]",
        title_align="left",
        border_style=COLOR_BORDER,
        box=box.ROUNDED,
        width=_ui_width(c),
        expand=True,
    )
    c.print(panel)


def render_success_footer(
    duration: float,
    log_path: str,
    console: Optional[Console] = None,
) -> None:
    """Print the final success banner with elapsed time and audit log reference."""
    c = console or get_console()
    dur_fmt = format_duration(duration)
    time_display = f"{dur_fmt} ({duration:.2f}s)" if duration >= 60 else f"{duration:.2f}s"

    footer = Text()
    footer.append("✓ Success: ", style="bold #ff9955")
    footer.append("Analysis complete in ", style="white")
    footer.append(time_display, style="bold #ff9955")
    footer.append(f". Full audit log persisted in '{sanitize_path(log_path)}'.", style="dim white")
    c.print(footer)


def render_fcs_info(
    fcs: Any,
    duration: Optional[float] = None,
    console: Optional[Console] = None,
) -> None:
    """Render structured FCS file inspection information."""
    c = console or get_console()

    header = Text()
    header.append("⬢ flowx info │ ", style="bold #ff6600")
    header.append(f"FCS File Inspection: {fcs.filename}", style="bold white")

    info_grid = Table.grid(padding=(0, 2))
    info_grid.add_column(style="dim white")
    info_grid.add_column(style="bold #ff6600")
    info_grid.add_column(style="dim white")
    info_grid.add_column(style="bold #ff6600")

    info_grid.add_row("Filepath:", sanitize_path(fcs.filepath), "Total Events:", f"{fcs.event_count:,}")
    info_grid.add_row("Cytometer:", fcs.cytometer or "Standard / Detected", "Channel Count:", str(fcs.channel_count))

    c.print(Panel(info_grid, title=header, title_align="left", border_style=COLOR_MUTED, box=box.ROUNDED, width=_ui_width(c)))

    # Channel detectors table
    table = Table(
        title="◈ Channel Detector & Marker Configuration",
        title_style="dim white",
        box=box.ROUNDED,
        border_style=COLOR_BORDER,
        header_style="bold white",
        width=_ui_width(c),
    )
    table.add_column("Idx", style="dim white", justify="center")
    table.add_column("Detector (PnN)", style="bold white")
    table.add_column("Stain / Marker (PnS)", style="bold #ff9955")
    table.add_column("Resolved Label", style="#ff6600")

    for i, (pnn, pns) in enumerate(zip(fcs.pnn_labels, fcs.pns_labels)):
        table.add_row(str(i), pnn, pns or "-", fcs.get_channel_label(pnn))

    c.print(table)

    has_spill = fcs.spillover_matrix is not None
    spill_status = Text()
    spill_status.append("◈ Spillover Matrix: ", style="bold #ff6600")
    spill_status.append("Present " if has_spill else "None detected", style="bold #ff9955" if has_spill else "dim yellow")
    if has_spill and fcs.spillover_channels:
        spill_status.append(f"({len(fcs.spillover_channels)} compensated channels: {', '.join(fcs.spillover_channels)})", style="dim white")
    c.print(spill_status)

    if duration is not None:
        timing_status = Text()
        timing_status.append("◈ Elapsed Inspection Time: ", style="bold #ff6600")
        timing_status.append(f"{format_duration(duration)} ({duration:.2f}s)", style="bold #ff9955")
        c.print(timing_status)


def render_fcs_batch_summary(
    summary: Any,
    csv_path: Optional[Union[str, Path]] = None,
    channel_csv_path: Optional[Union[str, Path]] = None,
    console: Optional[Console] = None,
) -> None:
    """Render comprehensive cohort-level FCS batch inspection and quality control report."""
    c = console or get_console()

    # 1. Overview Banner
    header = Text()
    header.append("⬢ flowx inspect │ ", style="bold #ff6600")
    header.append("FCS Cohort Metadata & Quality Control Audit", style="bold white")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim white", justify="left")
    grid.add_column(style="bold #ff6600", justify="left")
    grid.add_column(style="dim white", justify="left")
    grid.add_column(style="bold #ff6600", justify="left")

    grid.add_row("Total FCS Files:", f"{summary.total_files} files", "Total Cell Events:", f"{summary.total_events:,}")
    grid.add_row("Dataset Storage:", f"{summary.total_size_mb:.1f} MB ({summary.total_size_mb/1024:.2f} GB)", "Event Range:", f"{summary.min_events:,} – {summary.max_events:,} (median: {summary.median_events:,.0f})")
    grid.add_row("Cytometers:", ", ".join(summary.unique_cytometers) or "Standard", "Manufacturers:", ", ".join(summary.unique_vendors) or "Standard")
    
    pass_pct = (summary.qc_pass_count / summary.total_files * 100.0) if summary.total_files > 0 else 0.0
    warn_pct = (summary.qc_warning_count / summary.total_files * 100.0) if summary.total_files > 0 else 0.0
    qc_summary_str = f"{summary.qc_pass_count} Passed ({pass_pct:.1f}%) │ {summary.qc_warning_count} Flagged ({warn_pct:.1f}%)"
    grid.add_row("Quality Status:", qc_summary_str, "Consensus Panel:", summary.consensus_panel)

    c.print(Panel(grid, title=header, title_align="left", border_style="#ff6600", box=box.ROUNDED))

    # 2. Population Breakdown Table
    if summary.population_counts:
        pop_table = Table(
            title="◈ Cell Population Distribution Across Cohort",
            title_style="bold #ff6600",
            box=box.ROUNDED,
            border_style="#ff6600",
            header_style="bold #ff6600",
        )
        pop_table.add_column("Class", style="bold #ff6600", justify="center")
        pop_table.add_column("Population Identity", style="bold white")
        pop_table.add_column("Files", style="#ff6600", justify="right")
        pop_table.add_column("Total Events", style="white", justify="right")
        pop_table.add_column("% Cohort", style="#ff6600", justify="right")
        pop_table.add_column("Event Proportion", style="bold #ff9955", justify="left")

        # Sort populations (A first, then others by event count descending)
        sorted_pops = sorted(
            summary.population_counts.items(),
            key=lambda kv: (-1 if kv[0] == "A" else 0, -kv[1]["event_count"]),
        )
        for pop_code, pdata in sorted_pops:
            ev = pdata["event_count"]
            pct = (ev / summary.total_events * 100.0) if summary.total_events > 0 else 0.0
            pop_table.add_row(
                pop_code,
                pdata["name"],
                str(pdata["file_count"]),
                f"{ev:,}",
                f"{pct:.2f}%",
                make_yield_bar(pct, length=14),
            )

        c.print(pop_table)

    # 3. Optical Panel Variants & Concordance Table
    if summary.panel_variants:
        panel_table = Table(
            title="◈ Optical Panel Configurations & Cross-File Consistency",
            title_style="bold #ff6600",
            box=box.ROUNDED,
            border_style="#ff6600",
            header_style="bold #ff6600",
        )
        panel_table.add_column("Panel Variant Signature", style="bold white")
        panel_table.add_column("Matching Files", style="#ff6600", justify="right")
        panel_table.add_column("% Files", style="#ff6600", justify="right")
        panel_table.add_column("Status / Observation", style="bold #ff9955")

        for sig, count in summary.panel_variants.items():
            pct = (count / summary.total_files * 100.0) if summary.total_files > 0 else 0.0
            if "Variant-1" in sig:
                status_desc = "[bold #ff9955]Consensus Standard[/bold #ff9955] (Navios raw with embedded $SPILLOVER)"
            elif "Variant-2" in sig:
                status_desc = "[bold yellow]Modified Naming[/bold yellow] (flowCore dot notation, no $SPILLOVER)"
            elif "Variant-3" in sig:
                status_desc = "[bold yellow]Clean Stain Naming[/bold yellow] (Direct marker detector names, no $SPILLOVER)"
            else:
                status_desc = "[dim white]Custom / Variant Layout[/dim white]"

            panel_table.add_row(sig, str(count), f"{pct:.1f}%", status_desc)

        c.print(panel_table)

    # 4. Quality Control & Audit Findings
    qc_text = Text()
    qc_text.append(f"• Spillover Matrices:  ", style="dim #ff6600")
    qc_text.append(f"{summary.spillover_present_count} files contain embedded $SPILLOVER, {summary.spillover_missing_count} lack embedded spillover\n", style="white")

    if summary.common_warnings:
        qc_text.append(f"• Detected Anomalies:  ", style="dim #ff6600")
        warn_bullets = [f"{cat} ({cnt}x)" for cat, cnt in summary.common_warnings.items()]
        qc_text.append(f"{'; '.join(warn_bullets)}\n", style="yellow")

    if csv_path:
        qc_text.append(f"• Exported Summary:    ", style="dim #ff6600")
        qc_text.append(f"{sanitize_path(csv_path)} (Full standardized MIFlowCyt report)\n", style="bold #ff9955")

    if channel_csv_path:
        qc_text.append(f"• Channel Details CSV: ", style="dim #ff6600")
        qc_text.append(f"{sanitize_path(channel_csv_path)} (Per-channel detector, voltage, range audit)\n", style="bold #ff9955")

    c.print(Panel(qc_text, title="◈ Quality Control & Audit Persisted Artifacts", title_align="left", border_style=COLOR_MUTED, box=box.ROUNDED, width=_ui_width(c)))


def render_fcs_inspection_table(
    results: List[Any],
    max_rows: int = 25,
    console: Optional[Console] = None,
) -> None:
    """Render a compact tabular listing of individual inspected FCS files."""
    c = console or get_console()

    table = Table(
        title=f"◈ Sample-Level FCS Inspection Roster ({min(len(results), max_rows)} of {len(results)} files)",
        title_style="bold #ff6600",
        box=box.ROUNDED,
        border_style="#ff6600",
        header_style="bold #ff6600",
    )
    table.add_column("Filename", style="bold white")
    table.add_column("Case", style="#ff6600", justify="center")
    table.add_column("Pop", style="bold #ff9955", justify="center")
    table.add_column("Events", style="white", justify="right")
    table.add_column("Duration", style="dim white", justify="right")
    table.add_column("Flow Rate", style="dim #ff6600", justify="right")
    table.add_column("Spillover", style="#ff6600", justify="center")
    table.add_column("QC", style="bold", justify="center")

    for r in results[:max_rows]:
        spill_badge = "[bold #ff9955]YES[/bold #ff9955]" if r.has_spillover else "[dim yellow]NO[/dim yellow]"
        qc_badge = "[bold #ff9955]PASS[/bold #ff9955]" if r.qc_status == "PASS" else "[bold yellow]WARN[/bold yellow]"
        dur_str = f"{r.acquisition_duration_sec:.1f}s" if r.acquisition_duration_sec > 0 else "-"
        rate_str = f"{r.event_rate_eps:.0f} eps" if r.event_rate_eps > 0 else "-"

        table.add_row(
            r.filename,
            r.patient_case_id,
            r.population_class,
            f"{r.event_count:,}",
            dur_str,
            rate_str,
            spill_badge,
            qc_badge,
        )

    if len(results) > max_rows:
        table.caption = f"Showing first {max_rows} of {len(results)} files. See full CSV report for complete cohort."
        table.caption_style = "dim #ff6600"

    c.print(table)


def render_cleanup_summary(
    result: Any,
    console: Optional[Console] = None,
) -> None:
    """Render a minimal completion notice for Core Clean-up Gating.

    Population counts and percentages are intentionally not echoed to the terminal;
    they are persisted in the run's statistics tables and report.

    Args:
        result: CleanUpGatingResult instance.
        console: Optional Rich console instance.
    """
    c = console or get_console()
    line = Text()
    line.append("✓ ", style=f"bold {COLOR_TINT_1}")
    line.append("Clean-up gating complete", style=COLOR_TEXT)
    line.append(f" in {result.execution_time_seconds:.2f}s", style=f"bold {COLOR_TINT_1}")
    line.append(f" — {result.sample_name}", style=COLOR_MUTED)
    c.print(line)


def render_datasets_table(
    datasets: List[Any],
    verifications: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
) -> None:
    """Render a formatted table of benchmark datasets with local verification status.

    Parameters
    ----------
    datasets : List[DatasetInfo]
        List of dataset metadata objects.
    verifications : Optional[Dict[str, DatasetVerificationResult]]
        Mapping of dataset ID to local verification status.
    console : Optional[Console]
        Optional Rich console instance.
    """
    c = console or get_console()

    table = Table(
        title="◈ FlowX Benchmark Datasets Catalog (MIFlowCyt Compliant)",
        title_style="bold #ff6600",
        box=box.ROUNDED,
        border_style="#ff6600",
        header_style="bold #ff6600",
    )
    table.add_column("#", style="dim #ff6600", justify="right", width=3)
    table.add_column("Dataset ID", style="bold white")
    table.add_column("Type", justify="center")
    table.add_column("Cytometer & Panel", style="dim white")
    table.add_column("Status", justify="center")
    table.add_column("Files (Ctrl/Samp)", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Recommended Use Case", style="#ff6600")

    for i, ds in enumerate(datasets, start=1):
        type_str = f"[#ff6600]Conv[/#ff6600]" if ds.category == "conventional" else f"[#ff6600]Spectral[/#ff6600]"
        cyto_panel = f"{ds.cytometer.split('(')[0].strip()[:18]} ({ds.colors})"

        status_str = "[dim red]Missing[/dim red]"
        files_str = f"- / {ds.files_expected}"
        size_str = f"{ds.size_mb_approx:.1f} MB"

        if verifications and ds.id in verifications:
            v = verifications[ds.id]
            if v.status == "Ready":
                status_str = "[bold #ff9955]✓ Ready[/bold #ff9955]"
            elif v.status == "Partial":
                status_str = "[bold yellow]⚠ Partial[/bold yellow]"
            else:
                status_str = "[dim red]✗ Missing[/dim red]"

            if v.exists and v.total_fcs > 0:
                files_str = f"{v.controls_count}c / {v.samples_count}s"
                size_str = f"{v.total_size_mb:.1f} MB"

        table.add_row(
            str(i),
            ds.id,
            type_str,
            cyto_panel,
            status_str,
            files_str,
            size_str,
            ds.recommended_for,
        )

    c.print(table)
    c.print(
        "[dim white]Commands:[/dim white] "
        "[bold #ff6600]flowx download <id>[/bold #ff6600] [dim](download dataset)[/dim] │ "
        "[bold #ff6600]flowx download <id> --info[/bold #ff6600] [dim](scientific details)[/dim] │ "
        "[bold #ff6600]flowx download --verify-only[/bold #ff6600] [dim](verify local files)[/dim]\n"
    )


def render_dataset_card(
    dataset: Any,
    verification: Optional[Any] = None,
    console: Optional[Console] = None,
) -> None:
    """Render a detailed scientific and technical card for a single dataset.

    Parameters
    ----------
    dataset : DatasetInfo
        Dataset specification.
    verification : Optional[DatasetVerificationResult]
        Verification result if verified locally.
    console : Optional[Console]
        Optional Rich console instance.
    """
    c = console or get_console()

    category_badge = "[bold #ff6600]Conventional Flow[/bold #ff6600]" if dataset.category == "conventional" else "[bold #ff6600]Spectral Flow[/bold #ff6600]"

    content = Text()
    content.append(f"• Dataset ID:    ", style="dim white")
    content.append(f"{dataset.id}\n", style="bold white")
    content.append(f"• Type:          ", style="dim white")
    content.append(f"{category_badge}\n")
    content.append(f"• Description:   ", style="dim white")
    content.append(f"{dataset.description}\n", style="white")
    content.append(f"• Organism:      ", style="dim white")
    content.append(f"{dataset.organism} ({dataset.specimen})\n", style="white")
    content.append(f"• Cytometer:     ", style="dim white")
    content.append(f"{dataset.cytometer}\n", style="white")
    content.append(f"• Colors/Panel:  ", style="dim white")
    content.append(f"{dataset.colors}\n", style="bold #ff6600")
    content.append(f"• Repository:    ", style="dim white")
    content.append(f"{dataset.repository} (Accession/DOI: {dataset.accession})\n", style="white")
    content.append(f"• Publication:   ", style="dim white")
    content.append(f"{dataset.publication}\n", style="white")
    content.append(f"• Article DOI:   ", style="dim white")
    content.append(f"{dataset.doi}\n", style="#ff6600")
    content.append(f"• Recommended:   ", style="dim white")
    content.append(f"{dataset.recommended_for}\n", style="bold #ff9955")

    if verification:
        status_text = {
            "Ready": "[bold #ff9955]✓ Ready (Fully Downloaded & Verified)[/bold #ff9955]",
            "Partial": "[bold yellow]⚠ Partial (Some files missing or mismatch)[/bold yellow]",
            "Missing": "[dim red]✗ Not Downloaded Yet[/dim red]",
        }.get(verification.status, verification.status)

        content.append("\n[bold #ff6600]Local Storage Status:[/bold #ff6600]\n")
        content.append(f"  - Health:        {status_text}\n")
        content.append(f"  - Local Path:    {sanitize_path(verification.dataset_dir)}\n")
        content.append(f"  - FCS Files:     {verification.total_fcs} total ({verification.controls_count} controls, {verification.samples_count} samples)\n")
        content.append(f"  - Storage Size:  {verification.total_size_mb:.2f} MB\n")
        content.append(f"  - Manifest:      {'Valid (' + str(verification.manifest_entries) + ' entries)' if verification.manifest_valid else 'Missing/Invalid'}\n")
        content.append(f"  - Papers / Supp: {verification.publications_count} publication files, {verification.supplementary_count} supplementary files\n")
        if verification.issues:
            content.append(f"  - Issues:        {', '.join(verification.issues)}\n", style="yellow")

    panel = Panel(
        content,
        title=f"◈ [bold white]{dataset.name}[/bold white]",
        border_style=COLOR_BORDER,
        box=box.ROUNDED,
        width=_ui_width(c),
    )
    c.print(panel)


# ---------------------------------------------------------------------------
# Live progress bar with status (replaces line-by-line log streaming)
# ---------------------------------------------------------------------------
SAMPLE_STEP_COUNT = 8  # main pipeline steps per sample (Step 1 ... Step 8)
COHORT_STAGE_UNITS = 2  # cohort-level discovery + statistics stages
_MAIN_STEP_DONE = re.compile(r"^<<< Completed Step (\d+):")
_COHORT_PREPROC = re.compile(r"^\[(\d+)/(\d+)\] Preprocessing sample")
_STATUS_PREFIX = re.compile(r"^(>>> Starting |<<< Completed |\[\d+:\d+:\d+\]\s*)")


class PipelineProgress:
    """Rich progress bar whose status text follows the pipeline logger.

    While active, console log lines are suppressed (the full trace still goes to
    ``pipeline.log``); the latest INFO message is shown as a one-line status and
    WARNING/ERROR messages are printed above the bar.
    """

    def __init__(self, n_samples: int = 1, with_cohort_stage: bool = False, console: Optional[Console] = None) -> None:
        """Initialise progress state.

        Args:
            n_samples: Number of samples that will be processed sequentially.
            with_cohort_stage: Whether cohort-level stages follow the per-sample runs.
            console: Optional Rich console.
        """
        self.console = console or get_console()
        self._width = _ui_width(self.console)
        self.n_samples = max(1, n_samples)
        self.units_per_sample = SAMPLE_STEP_COUNT
        self.total = self.n_samples * self.units_per_sample + (COHORT_STAGE_UNITS if with_cohort_stage else 0)
        self.sample_idx = 0
        self.phase = "sample"
        self.mode = "single"
        self.cohort_samples = 0
        self.progress = Progress(
            SpinnerColumn(style=COLOR_ACCENT),
            TextColumn("[bold " + COLOR_ACCENT + "]Running:[/]"),
            BarColumn(bar_width=44, complete_style=COLOR_ACCENT, finished_style=COLOR_TINT_1, pulse_style=COLOR_TINT_2),
            TaskProgressColumn(style=COLOR_TEXT),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        self._status_text = Text("  Initialising…", style=COLOR_MUTED)
        self.panel = Panel(
            Group(self.progress, self._status_text),
            title="[dim white]Progress[/dim white]",
            title_align="left",
            border_style=COLOR_BORDER,
            box=box.ROUNDED,
            padding=(0, 1),
            width=self._width,
        )
        self.live = Live(
            self.panel,
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self.task: Optional[TaskID] = None
        self._handler: Optional[logging.Handler] = None
        self._prev_override: Optional[int] = None
        self._sample_label = ""

    # -- status helpers ---------------------------------------------------
    def _render_status_line(self, plain_msg: str, style: str = COLOR_TEXT) -> None:
        max_len = max(20, self._width - 6)
        prefix_plain = f"{self._sample_label} " if self._sample_label else ""
        avail = max(10, max_len - len(prefix_plain))
        clean = plain_msg if len(plain_msg) <= avail else (plain_msg[: avail - 1] + "…")
        self._status_text.plain = ""
        self._status_text.spans.clear()
        self._status_text.append("  ")
        if self._sample_label:
            self._status_text.append(f"{self._sample_label} ", style=COLOR_MUTED)
        self._status_text.append(clean, style=style)

    def set_status(self, text: str) -> None:
        """Update the one-line status shown below the progress bar."""
        if self.task is None:
            return
        clean = _STATUS_PREFIX.sub("", text).strip()
        self._render_status_line(clean, style=COLOR_TEXT)

    def begin_sample(self, idx: int, name: str) -> None:
        """Mark the start of the idx-th (0-based) sample."""
        self.phase = "sample"
        self.sample_idx = idx
        self._sample_label = f"[{idx + 1}/{self.n_samples}] {name}" if self.n_samples > 1 else ""
        if self.task is not None:
            self.progress.update(self.task, completed=idx * self.units_per_sample)

    def begin_cohort_stage(self) -> None:
        """Switch to the cohort-level discovery/statistics phase."""
        self.phase = "cohort"
        self._sample_label = "cohort"
        if self.task is not None:
            self.progress.update(self.task, completed=self.n_samples * self.units_per_sample)

    def advance_cohort(self, units: int = 1) -> None:
        """Advance the bar during the cohort phase."""
        if self.task is not None:
            self.progress.advance(self.task, units)

    def _set_total(self, total: int) -> None:
        """Change the bar's total number of work units."""
        self.total = total
        if self.task is not None:
            self.progress.update(self.task, total=total)

    def _complete_to(self, units: int) -> None:
        """Move the bar forward to ``units`` (never backwards)."""
        if self.task is None:
            return
        current = self.progress.tasks[0].completed
        if units > current:
            self.progress.update(self.task, completed=min(units, self.total))

    def _on_record(self, record: logging.LogRecord) -> None:
        """Translate a log record into a status update / bar advance."""
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:
            self.live.console.print(f"[yellow]![/yellow] {msg.replace('[', chr(92) + '[')}", highlight=False)
            return

        pre = _COHORT_PREPROC.match(msg)
        if pre:
            idx, n = int(pre.group(1)), int(pre.group(2))
            if self.mode != "cohort":
                self.mode = "cohort"
                self.cohort_samples = n
                self._set_total(n + 4)
            self._sample_label = f"[{idx}/{n}]"
            self._complete_to(idx - 1)
        elif self.mode == "cohort":
            if msg.startswith("<<< Completed Step 4:"):
                self._complete_to(self.cohort_samples + 1)
                self._sample_label = "cohort"
            elif "Clustering complete" in msg:
                self._complete_to(self.cohort_samples + 2)
            elif msg.startswith("<<< Completed Step 5 & 6"):
                self._complete_to(self.cohort_samples + 4)
        elif self.phase == "sample":
            m = _MAIN_STEP_DONE.match(msg)
            if m:
                done = self.sample_idx * self.units_per_sample + min(int(m.group(1)), self.units_per_sample)
                self._complete_to(done)
        elif "Clustering complete" in msg:
            self.advance_cohort(1)
        self.set_status(msg)

    # -- context management ----------------------------------------------
    def __enter__(self) -> "PipelineProgress":
        """Start the live display and hook the pipeline logger."""
        self._prev_override = set_console_level_override(logging.CRITICAL + 10)
        outer = self

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    outer._on_record(record)
                except Exception:
                    pass

        self._handler = _Handler(level=logging.INFO)
        logging.getLogger(DEFAULT_LOGGER_NAME).addHandler(self._handler)
        self.task = self.progress.add_task("flowx", total=self.total)
        self.live.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Complete (or abort) the bar and restore normal logging."""
        if self.task is not None and exc_type is None:
            self.progress.update(self.task, completed=self.total)
            self._render_status_line("Done", style=f"bold {COLOR_TINT_1}")
        elif self.task is not None:
            self._render_status_line("Failed", style="bold red")
        self.live.refresh()
        self.live.stop()
        if self._handler is not None:
            logging.getLogger(DEFAULT_LOGGER_NAME).removeHandler(self._handler)
        set_console_level_override(self._prev_override)


def progress_enabled(log_level: Optional[str] = None, console: Optional[Console] = None) -> bool:
    """Return True when the live progress bar should replace streamed log lines.

    The bar is used on interactive terminals at the default INFO level; DEBUG
    verbosity or non-interactive output (pipes, CI) keeps plain line logging.
    """
    c = console or get_console()
    level = (log_level or "INFO").upper()
    return bool(c.is_terminal) and level in ("INFO", "WARNING", "ERROR")


@contextmanager
def maybe_progress(
    enabled: bool,
    n_samples: int = 1,
    with_cohort_stage: bool = False,
) -> Generator[Optional[PipelineProgress], None, None]:
    """Yield a ``PipelineProgress`` when enabled, else ``None``."""
    if not enabled:
        yield None
        return
    with PipelineProgress(n_samples=n_samples, with_cohort_stage=with_cohort_stage) as prog:
        yield prog
