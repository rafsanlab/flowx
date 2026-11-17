"""Unit and integration tests for Core Clean-up Gating (Phase 2)."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from flowx.cli import main
from flowx.gating.cleanup import (
    CleanUpGateConfig,
    CleanUpGatingResult,
    build_debris_gate,
    build_dump_gate,
    build_singlet_gate,
    build_viability_gate,
    detect_dump_channel,
    detect_viability_channel,
    resolve_scatter_channels,
    run_cleanup_gating,
)
from flowx.gating.gate import GatingHierarchy, GMMClusterGate, PolygonGate, RangeGate
from flowx.gating.runner import run_cohort_cleanup
from flowx.io.parser import parse_fcs
from flowx.math.compensation import CompensationMatrix
from flowx.math.transforms import apply_transform_to_dataframe, get_optimal_transform
from flowx.visualization.cleanup_plots import plot_cleanup_diagnostics

@pytest.fixture(scope="module")
def prepared_sample(tqc_sample_fcs_path: Path):
    """Load, compensate, and transform sample file once for clean-up gating tests."""
    fcs = parse_fcs(tqc_sample_fcs_path)
    events_df = fcs.events_df.copy()

    # Compensate with embedded spillover matrix
    if fcs.spillover_matrix is not None:
        comp_mat = CompensationMatrix(
            detectors=list(fcs.spillover_matrix.columns),
            matrix=fcs.spillover_matrix.to_numpy(),
        )
        events_df = comp_mat.apply_dataframe(events_df)

    # Transform fluorescence channels with Logicle
    xform = get_optimal_transform(fcs, transform_type="logicle")
    fluoro_channels = fcs.spillover_channels
    xformed_df = apply_transform_to_dataframe(events_df, channels=fluoro_channels, transform=xform)

    return fcs, events_df, xformed_df


def test_resolve_scatter_channels(prepared_sample):
    """Verify scatter channel resolution accurately identifies Area and Height channels."""
    fcs, events_df, _ = prepared_sample
    channels = resolve_scatter_channels(fcs, events_df)

    assert channels["fsc_area"] == "FSC-A"
    assert channels["ssc_area"] == "SSC-A"
    assert channels["fsc_height"] == "FSC-H"


def test_debris_gate_adaptive_envelope(prepared_sample):
    """Verify adaptive debris removal creates a valid data-driven polygon gate."""
    fcs, events_df, _ = prepared_sample

    cfg = CleanUpGateConfig(debris_method="adaptive")
    gate = build_debris_gate(fcs, events_df, config=cfg)

    assert isinstance(gate, PolygonGate)
    assert gate.name == "Cells"
    assert len(gate.channels) == 2
    assert len(gate.vertices) >= 4

    mask = gate.apply(events_df)
    intact_count = int(np.sum(mask))
    assert intact_count > 0.50 * fcs.event_count


def test_debris_gate_gmm_method(prepared_sample):
    """Verify GMM-based debris exclusion builds convex hull polygon gate and cluster info metadata."""
    fcs, events_df, _ = prepared_sample
    cfg = CleanUpGateConfig(debris_method="gmm")
    gate = build_debris_gate(fcs, events_df, config=cfg)

    assert isinstance(gate, PolygonGate)
    assert gate.name == "Cells"
    assert len(gate.vertices) >= 3
    assert "cluster_info" in gate.metadata
    cinfo = gate.metadata["cluster_info"]
    assert cinfo["method"] == "gmm"
    assert "Intact Cells" in cinfo["cluster_names"].values()
    assert len(cinfo["means"]) == 3
    assert len(cinfo["covariances"]) == 3

    mask = gate.apply(events_df)
    assert np.sum(mask) > 0.40 * fcs.event_count


def test_singlet_gate_fsc_and_ssc(prepared_sample):
    """Verify forward scatter singlet gating evaluates slope and encloses singlets."""
    fcs, events_df, _ = prepared_sample
    gate_cells = build_debris_gate(fcs, events_df)
    cells_mask = gate_cells.apply(events_df)

    # FSC Singlets
    gate_fsc, slope_fsc = build_singlet_gate(
        fcs,
        events_df,
        scatter_axis="fsc",
        parent_mask=cells_mask,
        config=CleanUpGateConfig(singlet_tolerance=0.25),
    )
    assert isinstance(gate_fsc, PolygonGate)
    assert gate_fsc.name == "Singlets_FSC"
    assert 0.3 <= slope_fsc <= 2.0

    mask_fsc = gate_fsc.apply(events_df)
    assert np.sum(mask_fsc & cells_mask) > 0.60 * np.sum(cells_mask)


def test_singlet_gate_fallback_when_height_missing(prepared_sample):
    """Verify singlet gating provides graceful pass-through when height channel is absent."""
    fcs, events_df, _ = prepared_sample

    # Drop height channel to simulate instrument without pulse height
    df_no_h = events_df.drop(columns=["FSC-H"])
    gate_fallback, slope = build_singlet_gate(fcs, df_no_h, scatter_axis="fsc")

    assert isinstance(gate_fallback, PolygonGate)
    mask = gate_fallback.apply(df_no_h)
    # Pass-through gate should retain events
    assert np.sum(mask) == len(df_no_h)


def test_detect_viability_channel(prepared_sample):
    """Verify viability channel auto-detection identifies AARD-A / Dead in test sample."""
    fcs, _, _ = prepared_sample
    ch = detect_viability_channel(fcs)
    assert ch == "AARD-A"


def test_viability_gate(prepared_sample):
    """Verify viability gating creates a valid RangeGate with bimodal density valley cutoff."""
    fcs, events_df, xformed_df = prepared_sample
    cells_gate = build_debris_gate(fcs, events_df)
    cells_mask = cells_gate.apply(events_df)

    fsc_gate, _ = build_singlet_gate(fcs, events_df, scatter_axis="fsc", parent_mask=cells_mask)
    singlets_mask = cells_mask & fsc_gate.apply(events_df)

    viab_gate, cutoff = build_viability_gate(fcs, xformed_df, parent_mask=singlets_mask)
    assert viab_gate is not None
    assert isinstance(viab_gate, RangeGate)
    assert viab_gate.name == "Viable"
    assert viab_gate.channel == "AARD-A"
    assert 0.20 <= cutoff <= 0.85

    live_mask = singlets_mask & viab_gate.apply(xformed_df)
    assert np.sum(live_mask) > 0.70 * np.sum(singlets_mask)


def test_viability_explicit_user_cutoff_override(prepared_sample):
    """Verify explicit viability_cutoff configured by user takes precedence."""
    fcs, _, xformed_df = prepared_sample
    cfg = CleanUpGateConfig(viability_cutoff=0.523)
    viab_gate, cutoff = build_viability_gate(fcs, xformed_df, config=cfg)
    assert viab_gate is not None
    assert viab_gate.max_val == 0.523
    assert cutoff == 0.523


def test_viability_unimodal_robust_tail_option_c(prepared_sample):
    """Verify Option C dynamically estimates cutoff via robust mode/MAD statistics when unimodal."""
    fcs, events_df, xformed_df = prepared_sample
    cells_gate = build_debris_gate(fcs, events_df)
    cells_mask = cells_gate.apply(events_df)
    fsc_gate, _ = build_singlet_gate(fcs, events_df, scatter_axis="fsc", parent_mask=cells_mask)
    singlets_mask = cells_mask & fsc_gate.apply(events_df)

    viab_gate, cutoff = build_viability_gate(fcs, xformed_df, parent_mask=singlets_mask)
    assert viab_gate is not None
    # For LD1_NS+NS_A01, Option C places cutoff at ~0.40 - 0.45 (live ~95%)
    assert 0.35 <= cutoff <= 0.60
    live_mask = singlets_mask & viab_gate.apply(xformed_df)
    assert np.mean(live_mask[singlets_mask]) >= 0.90


def test_dump_channel_gating(prepared_sample):
    """Verify dump channel gating creates RangeGate and filters out positive lineage events."""
    fcs, _, xformed_df = prepared_sample

    # Simulate dump channel on Ax647-A (Blank in this panel)
    cfg = CleanUpGateConfig(dump_enabled=True, dump_channel="Ax647-A", dump_cutoff=0.30)
    dump_gate, cutoff = build_dump_gate(fcs, xformed_df, config=cfg)

    assert dump_gate is not None
    assert isinstance(dump_gate, RangeGate)
    assert dump_gate.name == "Target"
    assert dump_gate.channel == "Ax647-A"
    assert cutoff == 0.30


def test_run_cleanup_gating_end_to_end(prepared_sample):
    """Verify full Core Clean-up Gating cascade executes sequentially and builds statistics."""
    fcs, events_df, xformed_df = prepared_sample

    cfg = CleanUpGateConfig(
        singlet_tolerance=0.25,
        viability_enabled=True,
    )

    result = run_cleanup_gating(
        fcs_data=fcs,
        df_events=events_df,
        df_transformed=xformed_df,
        config=cfg,
    )

    assert isinstance(result, CleanUpGatingResult)
    assert result.total_events == fcs.event_count
    assert result.clean_event_count > 0.65 * fcs.event_count
    assert 65.0 <= result.clean_event_pct <= 95.0

    # Verify gates in hierarchy
    assert "Cells" in result.hierarchy.gates
    assert "Singlets_FSC" in result.hierarchy.gates
    assert "Viable" in result.hierarchy.gates

    # Verify statistics table
    stats = result.statistics_df
    assert "Gate" in stats.columns
    assert "Count" in stats.columns
    assert "% Parent" in stats.columns
    assert "% Total" in stats.columns
    assert len(stats) >= 4

    # Verify clean events extractor
    clean_df = result.get_clean_events(events_df)
    assert len(clean_df) == result.clean_event_count
    assert len(clean_df.columns) == len(events_df.columns)


def test_one_time_singlet_mode(prepared_sample):
    """Verify singlet_mode='single' or 'one_time' executes exactly one singlet discrimination gate."""
    fcs, events_df, xformed_df = prepared_sample

    cfg_single = CleanUpGateConfig(singlet_mode="single")
    res_single = run_cleanup_gating(fcs, events_df, xformed_df, config=cfg_single)

    assert "Cells" in res_single.hierarchy.gates
    assert "Singlets_FSC" in res_single.hierarchy.gates
    assert "Singlets_SSC" not in res_single.hierarchy.gates

    # Test explicit SSC only
    cfg_ssc = CleanUpGateConfig(singlet_mode="ssc_only")
    res_ssc = run_cleanup_gating(fcs, events_df, xformed_df, config=cfg_ssc)

    assert "Cells" in res_ssc.hierarchy.gates
    assert "Singlets_FSC" not in res_ssc.hierarchy.gates
    assert "Singlets_SSC" in res_ssc.hierarchy.gates


def test_export_clean_fcs(prepared_sample, tmp_path: Path):
    """Verify export_fcs writes a valid FCS file with exact clean event count and preserved metadata."""
    fcs, events_df, xformed_df = prepared_sample
    result = run_cleanup_gating(fcs, events_df, xformed_df)

    out_clean_fcs = tmp_path / "test_sample_clean.fcs"
    exported_path = result.export_fcs(out_clean_fcs, fcs_data=fcs, events_df=events_df)

    assert Path(exported_path).is_file()
    assert Path(exported_path).stat().st_size > 0

    # Re-parse exported FCS and verify event count matches
    reloaded_fcs = parse_fcs(exported_path)
    assert reloaded_fcs.event_count == result.clean_event_count
    assert reloaded_fcs.channel_count == fcs.channel_count
    assert reloaded_fcs.spillover_matrix is not None

    # M1 check: verify identity spillover matrix and compensated metadata
    spill = reloaded_fcs.spillover_matrix.to_numpy()
    assert np.allclose(spill, np.eye(len(spill))), "Exported compensated FCS must embed identity spillover to prevent double compensation"
    comp_tag = reloaded_fcs.metadata.get("comp") or reloaded_fcs.metadata.get("$comp") or reloaded_fcs.metadata.get("$COMP")
    assert comp_tag == "COMPENSATED"
    spill_applied = reloaded_fcs.metadata.get("spillover_applied") or reloaded_fcs.metadata.get("SPILLOVER_APPLIED")
    assert spill_applied == "TRUE"


def test_plot_cleanup_diagnostics(prepared_sample, tmp_path: Path):
    """Verify publication multi-panel clean-up diagnostic figure is generated and non-empty."""
    fcs, events_df, xformed_df = prepared_sample
    result = run_cleanup_gating(fcs, events_df, xformed_df)

    out_plot = tmp_path / "cleanup_diagnostics.png"
    saved_plot = plot_cleanup_diagnostics(
        fcs_data=fcs,
        cleanup_result=result,
        df_events=events_df,
        df_transformed=xformed_df,
        output_path=out_plot,
        dpi=150,
    )

    assert Path(saved_plot).is_file()
    assert Path(saved_plot).stat().st_size > 10000


def test_cli_cleanup_execution(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify CLI 'flowx cleanup' runs smoothly from command line."""
    out_dir = tmp_path / "cli_cleanup_out"
    ret = main([
        "cleanup",
        str(tqc_sample_fcs_path),
        "--output-dir",
        str(out_dir),
        "--export-clean-fcs",
    ])

    assert ret == 0
    assert (out_dir / "LD1_NS+NS_A01_exp_clean.fcs").is_file()
    assert (out_dir / "LD1_NS+NS_A01_exp_cleanup.png").is_file()


def test_run_cohort_cleanup(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify cohort runner processes multiple files and generates summary CSV & Markdown report."""
    out_dir = tmp_path / "cohort_out"
    summary_df, csv_p, report_p = run_cohort_cleanup(
        file_list_or_paths=[tqc_sample_fcs_path],
        output_dir=out_dir,
        generate_plots=True,
        export_clean_fcs=True,
    )

    assert len(summary_df) == 1
    assert Path(csv_p).is_file()
    assert Path(report_p).is_file()
    assert (out_dir / "processed_fcs" / "LD1_NS+NS_A01_exp.fcs").is_file()
    assert (out_dir / "plots" / "LD1_NS+NS_A01_exp_cleanup.png").is_file()


def test_debris_gate_manual_vertices(prepared_sample):
    """Verify manual polygon vertices configuration for debris gate."""
    fcs, events_df, _ = prepared_sample
    custom_verts = [[10000.0, 5000.0], [200000.0, 5000.0], [200000.0, 200000.0], [10000.0, 200000.0]]
    cfg = CleanUpGateConfig(debris_method="manual", debris_vertices=custom_verts)
    gate = build_debris_gate(fcs, events_df, config=cfg)

    assert isinstance(gate, PolygonGate)
    assert gate.vertices.tolist() == custom_verts


def test_cleanup_when_no_viability(prepared_sample):
    """Verify clean-up gating operates correctly when viability gating is disabled."""
    fcs, events_df, xformed_df = prepared_sample
    cfg = CleanUpGateConfig(viability_enabled=False)
    result = run_cleanup_gating(fcs, events_df, xformed_df, config=cfg)

    assert "Viable" not in result.hierarchy.gates
    assert "Singlets_FSC" in result.hierarchy.gates
    assert result.clean_event_count > 0


def test_cleanup_result_summary_dict(prepared_sample):
    """Verify to_summary_dict serializes clean-up metrics correctly."""
    fcs, events_df, xformed_df = prepared_sample
    result = run_cleanup_gating(fcs, events_df, xformed_df)
    summary = result.to_summary_dict()

    assert summary["sample_name"] == "LD1_NS+NS_A01_exp"
    assert summary["total_events"] == fcs.event_count
    assert "clean_events" in summary
    assert "clean_pct_total" in summary
    assert "execution_time_sec" in summary


def test_cleanup_kaluza_dataset(navios_suite_dir: Path):
    """Verify clean-up gating on clinical Kaluza sample with FS INT, SS INT, SS PEAK."""
    kaluza_path = navios_suite_dir / "Case10_A.fcs"
    fcs = parse_fcs(kaluza_path)
    cfg = CleanUpGateConfig(singlet_tolerance=0.25)
    result = run_cleanup_gating(fcs, config=cfg)

    assert result.total_events == fcs.event_count
    assert "Cells" in result.hierarchy.gates
    assert "Singlets_SSC" in result.hierarchy.gates
    assert result.clean_event_count > 0.50 * fcs.event_count


def test_singlet_gate_small_events_no_unbound_error(prepared_sample):
    """Verify build_singlet_gate does not raise UnboundLocalError on tiny subsets."""
    from flowx.gating.cleanup import build_singlet_gate
    fcs, events_df, _ = prepared_sample
    tiny_df = events_df.head(20).copy()
    gate, slope = build_singlet_gate(fcs, tiny_df, scatter_axis="fsc", config=CleanUpGateConfig())
    assert gate is not None
    assert isinstance(slope, float)


def test_one_time_singlet_mode_prefers_active_channel(navios_suite_dir: Path):
    """Verify singlet_mode='single' activates SSC when FSC height is missing."""
    kaluza_path = navios_suite_dir / "Case10_A.fcs"
    fcs = parse_fcs(kaluza_path)
    cfg = CleanUpGateConfig(singlet_mode="single")
    result = run_cleanup_gating(fcs, config=cfg)
    assert "Singlets_SSC" in result.hierarchy.gates
    assert "Singlets_FSC" not in result.hierarchy.gates
    assert result.channels_used["singlet_ssc_status"] == "active"


def test_singlet_tolerance_scaling(prepared_sample):
    """Verify singlet_tolerance modifies gate width in both GMM and envelope modes."""
    from flowx.gating.cleanup import build_singlet_gate
    fcs, events_df, _ = prepared_sample
    gate_narrow, _ = build_singlet_gate(fcs, events_df, scatter_axis="fsc", config=CleanUpGateConfig(singlet_tolerance=0.10))
    gate_wide, _ = build_singlet_gate(fcs, events_df, scatter_axis="fsc", config=CleanUpGateConfig(singlet_tolerance=0.50))
    y_span_narrow = gate_narrow.vertices[2][1] - gate_narrow.vertices[1][1]
    y_span_wide = gate_wide.vertices[2][1] - gate_wide.vertices[1][1]
    assert y_span_wide > y_span_narrow * 1.5


def test_cleanup_hierarchy_serialization_reproducibility(prepared_sample):
    """Verify hierarchy serialization and deserialization produces identical event counts."""
    fcs, events_df, xformed_df = prepared_sample
    result = run_cleanup_gating(fcs, events_df, xformed_df)

    serialized = result.hierarchy.to_dict()
    reloaded_hierarchy = GatingHierarchy.from_dict(serialized)

    re_masks = reloaded_hierarchy.evaluate(events_df, transformed_df=xformed_df)
    last_gate = reloaded_hierarchy.order[-1]
    re_clean_count = int(np.sum(re_masks[last_gate]))

    assert re_clean_count == result.clean_event_count
    assert int(np.sum(re_masks["Cells"])) == int(np.sum(result.masks["Cells"]))


def test_viability_and_dump_keyword_matching_precision():
    """Verify robust viability and dump keyword matching prevents substring false positives."""
    from flowx.gating.cleanup import matches_dump_channel, matches_viability_dye

    # Negative controls that previously matched substring 'pi', 'efluor', 'dead'
    assert not matches_viability_dye("CD3 eFluor450")
    assert not matches_viability_dye("Ki-67 eFluor 660")
    assert not matches_viability_dye("Spike-RBD")
    assert not matches_viability_dye("Lipid")
    assert not matches_viability_dye("PI3K")
    assert not matches_viability_dye("Capillary")
    assert not matches_viability_dye("Deadbolt")

    # True positive viability dyes
    assert matches_viability_dye("AARD-A")
    assert matches_viability_dye("Dead")
    assert matches_viability_dye("Zombie Aqua")
    assert matches_viability_dye("7-AAD")
    assert matches_viability_dye("DAPI")
    assert matches_viability_dye("FL3-A (PI)")
    assert matches_viability_dye("Propidium Iodide")
    assert matches_viability_dye("Fixable Viability Dye eFluor 780")

    # Dump keywords
    assert not matches_dump_channel("Linker")
    assert matches_dump_channel("CD45RA lin-")
    assert matches_dump_channel("Dump Channel")
    assert matches_dump_channel("CD20 Exclusion?")


def test_cohort_cleanup_with_custom_compensation_matrix(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify run_cohort_cleanup correctly prioritizes and applies custom compensation matrix (M6)."""
    fcs = parse_fcs(tqc_sample_fcs_path)
    detectors = fcs.spillover_channels
    n = len(detectors)
    custom_mat = pd.DataFrame(np.eye(n), index=detectors, columns=detectors)
    if n > 1:
        custom_mat.iloc[0, 1] = 0.05

    out_dir = tmp_path / "custom_comp_cleanup"
    df_sum, _, _ = run_cohort_cleanup(
        [tqc_sample_fcs_path],
        output_dir=out_dir,
        compensation_matrix=custom_mat,
        generate_plots=False,
        export_clean_fcs=True,
    )
    assert not df_sum.empty
    assert df_sum["compensation_source"].iloc[0] == "external_dataframe"
    clean_fcs_file = out_dir / "processed_fcs" / tqc_sample_fcs_path.name
    assert clean_fcs_file.is_file()
    reloaded = parse_fcs(clean_fcs_file)
    comp_tag = reloaded.metadata.get("comp") or reloaded.metadata.get("$comp") or reloaded.metadata.get("$COMP")
    assert comp_tag == "COMPENSATED"


def test_cleanup_low_retention_warning(caplog):
    """Verify run_cleanup_gating logs an actionable warning when retention falls below 25%."""
    import logging
    from types import SimpleNamespace

    rng = np.random.default_rng(42)
    n = 1000
    events_df = pd.DataFrame({
        "FSC-A": rng.uniform(50000.0, 200000.0, n),
        "FSC-H": rng.uniform(50000.0, 200000.0, n),
        "SSC-A": rng.uniform(50000.0, 200000.0, n),
        "SSC-H": rng.uniform(50000.0, 200000.0, n),
    })
    fcs_mock = SimpleNamespace(
        sample_name="Fragile_Tissue_Sample",
        file_path="Fragile_Tissue_Sample.fcs",
        events_df=events_df,
        event_count=n,
        pnn_labels=list(events_df.columns),
        pns_labels=[""] * 4,
        channel_ranges={c: 262144.0 for c in events_df.columns},
    )
    tiny_verts = [[10000.0, 5000.0], [15000.0, 5000.0], [15000.0, 15000.0], [10000.0, 15000.0]]
    cfg = CleanUpGateConfig(debris_method="manual", debris_vertices=tiny_verts, viability_enabled=False)
    with caplog.at_level(logging.WARNING):
        res = run_cleanup_gating(fcs_mock, events_df, events_df, config=cfg)
    assert res.clean_event_pct < 25.0
    assert any("Low clean-up retention yield" in rec.message for rec in caplog.records)


