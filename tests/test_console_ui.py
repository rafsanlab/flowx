"""Unit tests for FlowX Biotech Laboratory console UI presentation module."""

import io
from pathlib import Path
from types import SimpleNamespace
from rich.console import Console

from flowx.ui.console import (
    make_yield_bar,
    render_banner,
    print_step,
    render_subpopulation_matrix,
    render_artifacts_panel,
    render_fcs_info,
    render_success_footer,
)
from flowx.io.parser import parse_fcs


def test_make_yield_bar():
    """Verify ASCII yield bar generates expected visual fills."""
    bar_0 = make_yield_bar(0.0, length=10)
    assert bar_0 == "[░░░░░░░░░░]"

    bar_50 = make_yield_bar(50.0, length=10)
    assert bar_50 == "[█████░░░░░]"

    bar_100 = make_yield_bar(100.0, length=10)
    assert bar_100 == "[██████████]"


def test_render_banner_output():
    """Verify render_banner produces formatted Biotech Laboratory panel."""
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=True, width=100)

    render_banner(
        sample_name="sample.fcs",
        mode="SOM",
        instrument="FACSDiva",
        technology="Conventional",
        console=test_console,
    )

    rendered = output.getvalue()
    assert "flowx" in rendered.lower()
    assert "FlowX Platform" in rendered
    assert "sample.fcs" in rendered
    assert "Conventional" in rendered
    assert "SOM" in rendered


def test_print_step_output():
    """Verify print_step renders molecular glyphs and timing markers."""
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=True, width=100)

    print_step(
        step_idx=1,
        total_steps=6,
        title="Parse FCS Stream",
        duration=0.45,
        sub_items=[("Yield", "96.2%")],
        console=test_console,
    )

    rendered = output.getvalue()
    assert "[1/6]" in rendered
    assert "◈" in rendered
    assert "Parse FCS Stream" in rendered
    assert "[DONE 0.45s]" in rendered
    assert "Yield" in rendered
    assert "96.2%" in rendered


def test_render_fcs_info(tqc_sample_fcs_path: Path):
    """Verify render_fcs_info displays structured table for parsed FCS data."""
    fcs = parse_fcs(tqc_sample_fcs_path)
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=True, width=120)

    render_fcs_info(fcs, console=test_console)

    rendered = output.getvalue()
    assert "flowx info" in rendered
    assert fcs.filename in rendered
    assert "FSC-A" in rendered
    assert "PE-TxRed-A" in rendered
    assert "CD4" in rendered


def test_render_success_footer():
    """Verify render_success_footer formats elapsed time and log file path."""
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=True, width=100)

    # Sub-minute duration
    render_success_footer(duration=3.14, log_path="outputs/run.log", console=test_console)
    rendered = output.getvalue()
    assert "Success" in rendered
    assert "3.14s" in rendered
    assert "outputs/run.log" in rendered

    # Multi-minute duration
    output2 = io.StringIO()
    test_console2 = Console(file=output2, force_terminal=True, width=100)
    render_success_footer(duration=75.4, log_path="outputs/run.log", console=test_console2)
    rendered2 = output2.getvalue()
    assert "1m 15.40s" in rendered2
    assert "75.40s" in rendered2


def test_render_artifacts_panel_timing():
    """Verify render_artifacts_panel displays elapsed execution time."""
    mock_res = SimpleNamespace(
        generated_plots={"step1": "plot1.png"},
        run_dir="outputs/sample_run",
        workspace_json_path="outputs/sample_run/workspace.json",
        workspace_xml_path="outputs/sample_run/workspace.xml",
        reproducibility_verified=True,
        report_md_path="outputs/sample_run/report.md",
        statistics_csv_path="outputs/sample_run/population_statistics.csv",
        log_file_path="outputs/sample_run/pipeline.log",
        execution_time_seconds=14.85,
        execution_duration_formatted="14.85s",
    )
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=True, width=100)

    render_artifacts_panel(mock_res, console=test_console)
    rendered = output.getvalue()
    assert "Run Time:" in rendered
    assert "14.85s" in rendered


def test_render_fcs_info_timing(tqc_sample_fcs_path: Path):
    """Verify render_fcs_info displays optional inspection duration timer."""
    fcs = parse_fcs(tqc_sample_fcs_path)
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=True, width=120)

    render_fcs_info(fcs, duration=0.15, console=test_console)
    rendered = output.getvalue()
    assert "Elapsed Inspection Time:" in rendered
    assert "0.15s" in rendered
