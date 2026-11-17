"""Unit tests for automated and manual Time Quality Control (QC)."""

import numpy as np
import pandas as pd
import pytest

from flowx.qc.automated import (
    compute_time_bins,
    filter_flow_rate_anomalies,
    filter_margin_events,
    filter_signal_drift,
)
from flowx.qc.engine import apply_time_qc
from flowx.qc.manual import apply_manual_time_gate
from flowx.qc.models import TimeQCConfig, TimeQCResult


def test_compute_time_bins():
    """Verify time binning partitions time array accurately."""
    time_sec = np.linspace(0.0, 10.0, 1000)
    cfg = TimeQCConfig(bin_width_sec=1.0)
    bin_idx, starts, ends, counts = compute_time_bins(time_sec, 10.0, cfg)

    assert len(starts) >= 10
    assert len(ends) == len(starts)
    assert np.sum(counts) == 1000
    assert bin_idx.min() == 0
    assert bin_idx.max() == len(starts) - 1


def test_filter_flow_rate_anomalies():
    """Verify flow rate outlier detection flags spikes and drops."""
    cfg = TimeQCConfig(flow_rate_mad_factor=3.0, flow_rate_min_eps=50.0)

    # 20 bins with 500 eps, except bin 5 is a surge (2500 eps) and bin 12 is a clog (10 eps)
    bin_counts = np.array([500] * 20)
    bin_counts[5] = 2500  # Spike
    bin_counts[12] = 10   # Clog

    bin_starts = np.arange(20, dtype=float)
    bin_ends = bin_starts + 1.0
    bin_indices = np.repeat(np.arange(20), bin_counts)

    anom_events, anom_bins, stats = filter_flow_rate_anomalies(
        bin_counts=bin_counts,
        bin_starts=bin_starts,
        bin_ends=bin_ends,
        bin_indices=bin_indices,
        config=cfg,
    )

    assert anom_bins[5] == True
    assert anom_bins[12] == True
    assert anom_bins[0] == False
    assert anom_bins[1] == False
    assert np.sum(anom_bins) >= 2
    assert stats["median_eps"] == pytest.approx(500.0, rel=0.1)


def test_filter_signal_drift():
    """Verify channel drift detection identifies shifting populations."""
    cfg = TimeQCConfig(signal_drift_mad_factor=3.0)
    n_bins = 20
    events_per_bin = 100
    total_events = n_bins * events_per_bin
    bin_indices = np.repeat(np.arange(n_bins), events_per_bin)

    # Channel 1: steady around 1000
    ch1 = np.random.normal(1000, 50, total_events)
    # Channel 2: steady around 500, but shifts to 2000 in bins 15-17
    ch2 = np.random.normal(500, 30, total_events)
    ch2[bin_indices >= 15] += 1500

    df = pd.DataFrame({"FL1": ch1, "FL2": ch2})
    anom_events, anom_bins, drift_stats = filter_signal_drift(
        events_df=df,
        bin_indices=bin_indices,
        n_bins=n_bins,
        channels=["FL1", "FL2"],
        config=cfg,
    )

    assert "FL1" in drift_stats
    assert "FL2" in drift_stats
    assert drift_stats["FL2"].is_flagged == True
    assert np.any(anom_bins[15:]) == True


def test_filter_margin_events():
    """Verify hardware margin and saturation clipping."""
    cfg = TimeQCConfig(trim_margins=True, margin_tolerance=0.001)
    # Range is 1,048,576
    vals = np.array([100.0, 500.0, 1048576.0, 0.0, 50000.0])
    df = pd.DataFrame({"FL1": vals})
    ranges = {"FL1": 1048576.0}

    mask = filter_margin_events(df, ranges, ["FL1"], cfg)
    assert mask[0] == False
    assert mask[1] == False
    assert mask[2] == True  # Saturation
    assert mask[3] == True  # Floor
    assert mask[4] == False


def test_apply_manual_time_gate():
    """Verify manual time interval exclusion."""
    time_sec = np.array([1.0, 2.5, 5.0, 10.0, 15.0, 25.0])
    cfg = TimeQCConfig(
        manual_time_min=2.0,
        manual_time_max=20.0,
        manual_exclude_intervals=[(8.0, 12.0)],
    )

    excluded, stats = apply_manual_time_gate(time_sec, cfg)

    # 1.0 < 2.0 -> excluded
    assert excluded[0] == True
    # 2.5 in [2.0, 20.0] -> passed
    assert excluded[1] == False
    # 5.0 in [2.0, 20.0] -> passed
    assert excluded[2] == False
    # 10.0 in exclude interval [8.0, 12.0] -> excluded
    assert excluded[3] == True
    # 15.0 -> passed
    assert excluded[4] == False
    # 25.0 > 20.0 -> excluded
    assert excluded[5] == True


def test_sample_type_presets():
    """Verify factory presets configure appropriate parameters for controls vs samples."""
    cfg_sample = TimeQCConfig.for_sample()
    assert cfg_sample.sample_type == "sample"
    assert cfg_sample.enable_signal_drift == True
    assert cfg_sample.flow_rate_mad_factor == 3.5

    cfg_ctrl = TimeQCConfig.for_controls()
    assert cfg_ctrl.sample_type == "control"
    assert cfg_ctrl.enable_signal_drift == False
    assert cfg_ctrl.flow_rate_mad_factor == 4.5
    assert cfg_ctrl.trim_margins == True
    assert cfg_ctrl.manual_time_min is None

    cfg_fmo = TimeQCConfig.for_fmo()
    assert cfg_fmo.sample_type == "fmo"
    assert cfg_fmo.enable_signal_drift == True

    # Test from_sample_type string resolver
    resolved_ctrl = TimeQCConfig.from_sample_type("single_stain", manual_time_min=0.5)
    assert resolved_ctrl.sample_type == "control"
    assert resolved_ctrl.enable_signal_drift == False
    assert resolved_ctrl.manual_time_min == 0.5


def test_apply_time_qc_control_profile():
    """Verify that applying control QC profile bypasses signal drift but enforces margin trimming."""
    from flowx.io.parser import FCSData

    # Create synthetic mock FCSData
    n_events = 2000
    t = np.linspace(0.0, 20.0, n_events)
    # CD4-PE positive bead population in FL2
    fl1 = np.random.normal(50, 10, n_events)
    fl2 = np.random.normal(50000, 2000, n_events)
    # Add a couple saturated events
    fl2[100] = 1048576.0
    fl2[101] = 1048576.0

    df = pd.DataFrame({"TIME": t, "FS INT": np.random.normal(100000, 5000, n_events), "FL1 INT": fl1, "FL2 INT": fl2})

    mock_fcs = FCSData(
        filepath="dummy_ctrl.fcs",
        filename="dummy_ctrl.fcs",
        event_count=n_events,
        channel_count=4,
        pnn_labels=["TIME", "FS INT", "FL1 INT", "FL2 INT"],
        pns_labels=["Time", "FSC", "CD3", "CD4"],
        events_df=df,
        channel_ranges={"TIME": 1000000.0, "FS INT": 1048576.0, "FL1 INT": 1048576.0, "FL2 INT": 1048576.0},
        metadata={"@acqtime": "20.0"},
    )

    ctrl_cfg = TimeQCConfig.for_controls(manual_time_min=0.0)
    res = apply_time_qc(mock_fcs, config=ctrl_cfg)

    assert res.sample_type == "control"
    assert res.signal_drift_mask.sum() == 0  # Signal drift bypassed for controls
    assert res.margin_saturation_mask.sum() >= 2  # Saturation caught
    assert res.margin_saturation_mask[100] == True
    assert res.margin_saturation_mask[101] == True
    assert res.passed_events > 0


def test_qc_workflow_decision_matrix():
    """Verify decision logic for PROCEED_CLEAN, PROCEED_WITH_CAUTION, PROCEED_MASKED_SALVAGE, FALLBACK, and REJECT."""
    from flowx.io.parser import FCSData
    from flowx.qc.models import QCWorkflowDecision, evaluate_qc_workflow_decision

    n_events = 1000
    df = pd.DataFrame({"TIME": np.linspace(0, 10, n_events), "FL1": np.ones(n_events)})
    mock_fcs = FCSData(
        filepath="mock.fcs",
        filename="mock.fcs",
        event_count=n_events,
        channel_count=2,
        pnn_labels=["TIME", "FL1"],
        pns_labels=["Time", "CD3"],
        events_df=df,
        channel_ranges={"TIME": 1000.0, "FL1": 10000.0},
        metadata={},
    )

    # 1. PASS status -> PROCEED_CLEAN
    pass_mask = np.ones(n_events, dtype=bool)
    qc_pass = TimeQCResult(
        sample_name="sample.fcs",
        sample_type="sample",
        total_events=n_events,
        passed_events=n_events,
        rejected_events=0,
        passed_pct=100.0,
        passed_mask=pass_mask,
        flow_rate_anomaly_mask=~pass_mask,
        signal_drift_mask=~pass_mask,
        margin_saturation_mask=~pass_mask,
        manual_time_mask=~pass_mask,
        duration_sec=10.0,
        time_channel="TIME",
        qc_status="PASS",
    )
    dec_pass = evaluate_qc_workflow_decision(mock_fcs, qc_pass)
    assert dec_pass.can_proceed == True
    assert dec_pass.action == "PROCEED_CLEAN"
    clean_df = qc_pass.get_clean_dataframe(mock_fcs)
    assert len(clean_df) == n_events

    # 2. WARNING status -> PROCEED_WITH_CAUTION
    warn_mask = np.ones(n_events, dtype=bool)
    warn_mask[:200] = False
    qc_warn = TimeQCResult(
        sample_name="sample_warn.fcs",
        sample_type="sample",
        total_events=n_events,
        passed_events=800,
        rejected_events=200,
        passed_pct=80.0,
        passed_mask=warn_mask,
        flow_rate_anomaly_mask=~warn_mask,
        signal_drift_mask=np.zeros(n_events, dtype=bool),
        margin_saturation_mask=np.zeros(n_events, dtype=bool),
        manual_time_mask=np.zeros(n_events, dtype=bool),
        duration_sec=10.0,
        time_channel="TIME",
        qc_status="WARNING",
    )
    dec_warn = evaluate_qc_workflow_decision(mock_fcs, qc_warn)
    assert dec_warn.can_proceed == True
    assert dec_warn.action == "PROCEED_WITH_CAUTION"

    # 3. Control with FAIL status but >500 passed events -> PROCEED_MASKED_SALVAGE
    ctrl_mask = np.zeros(n_events, dtype=bool)
    ctrl_mask[:600] = True  # 600 events passed
    qc_ctrl_salvage = TimeQCResult(
        sample_name="ctrl_fail.fcs",
        sample_type="control",
        total_events=n_events,
        passed_events=600,
        rejected_events=400,
        passed_pct=60.0,
        passed_mask=ctrl_mask,
        flow_rate_anomaly_mask=~ctrl_mask,
        signal_drift_mask=np.zeros(n_events, dtype=bool),
        margin_saturation_mask=np.zeros(n_events, dtype=bool),
        manual_time_mask=np.zeros(n_events, dtype=bool),
        duration_sec=10.0,
        time_channel="TIME",
        qc_status="FAIL",
    )
    dec_ctrl_salvage = qc_ctrl_salvage.evaluate_workflow(mock_fcs, min_control_events=500)
    assert dec_ctrl_salvage.can_proceed == True
    assert dec_ctrl_salvage.action == "PROCEED_MASKED_SALVAGE"
    assert len(dec_ctrl_salvage.clean_events_df) == 600

    # 4. Control with FAIL status and <500 passed events -> FALLBACK_INSTRUMENT_MATRIX
    ctrl_sparse_mask = np.zeros(n_events, dtype=bool)
    ctrl_sparse_mask[:150] = True  # only 150 events passed
    qc_ctrl_sparse = TimeQCResult(
        sample_name="ctrl_sparse.fcs",
        sample_type="control",
        total_events=n_events,
        passed_events=150,
        rejected_events=850,
        passed_pct=15.0,
        passed_mask=ctrl_sparse_mask,
        flow_rate_anomaly_mask=~ctrl_sparse_mask,
        signal_drift_mask=np.zeros(n_events, dtype=bool),
        margin_saturation_mask=np.zeros(n_events, dtype=bool),
        manual_time_mask=np.zeros(n_events, dtype=bool),
        duration_sec=10.0,
        time_channel="TIME",
        qc_status="FAIL",
    )
    dec_ctrl_sparse = qc_ctrl_sparse.evaluate_workflow(mock_fcs, min_control_events=500)
    assert dec_ctrl_sparse.can_proceed == False
    assert dec_ctrl_sparse.action == "FALLBACK_INSTRUMENT_MATRIX"
    assert "below minimum threshold" in dec_ctrl_sparse.rejection_reason

    # 5. Biological Sample with FAIL status and <5000 events -> REJECT_SAMPLE
    sample_fail_sparse = TimeQCResult(
        sample_name="sample_fail.fcs",
        sample_type="sample",
        total_events=n_events,
        passed_events=400,
        rejected_events=600,
        passed_pct=40.0,
        passed_mask=ctrl_mask,
        flow_rate_anomaly_mask=~ctrl_mask,
        signal_drift_mask=np.zeros(n_events, dtype=bool),
        margin_saturation_mask=np.zeros(n_events, dtype=bool),
        manual_time_mask=np.zeros(n_events, dtype=bool),
        duration_sec=10.0,
        time_channel="TIME",
        qc_status="FAIL",
    )
    dec_sample_reject = sample_fail_sparse.evaluate_workflow(mock_fcs, min_sample_events=5000)
    assert dec_sample_reject.can_proceed == False
    assert dec_sample_reject.action == "REJECT_SAMPLE"


def test_export_clean_fcs_and_summary_generation(tmp_path: Path, tqc_sample_fcs_path: Path):
    """Verify run_cohort_time_qc exports clean .fcs files and records paths in summary CSV."""
    from pathlib import Path
    from flowx.qc.runner import run_cohort_time_qc
    from flowx.io.parser import parse_fcs

    out_dir = tmp_path / "qc_output"

    summary_df, csv_path, md_path = run_cohort_time_qc(
        file_list_or_paths=[tqc_sample_fcs_path],
        output_dir=out_dir,
        generate_plots=False,
        export_clean_fcs=True,
    )

    assert csv_path.is_file()
    assert md_path.is_file()
    assert len(summary_df) == 1
    assert "clean_fcs_path" in summary_df.columns

    clean_fcs_str = summary_df.iloc[0]["clean_fcs_path"]
    assert clean_fcs_str != ""
    clean_fcs_file = Path(clean_fcs_str)
    assert clean_fcs_file.is_file()

    # Re-read exported clean FCS file and verify event count matches passed_events
    reloaded = parse_fcs(clean_fcs_file)
    assert reloaded.event_count == summary_df.iloc[0]["passed_events"]
    assert reloaded.channel_count == 14


def test_extract_time_in_seconds_flowkit_scaling():
    """Verify extract_time_in_seconds does not double-scale preprocessed FlowKit time arrays (T1)."""
    from flowx.io.parser import FCSData
    from flowx.qc.automated import extract_time_in_seconds

    n_events = 1000
    # True acquisition: 120 seconds
    t_seconds = np.linspace(0.0, 120.0, n_events)
    df = pd.DataFrame({"Time": t_seconds, "FL1": np.ones(n_events)})
    # Metadata includes $TIMESTEP = 0.01 (100 Hz counter tick)
    fcs_mock = FCSData(
        filepath="mock.fcs",
        filename="mock.fcs",
        event_count=n_events,
        channel_count=2,
        pnn_labels=["Time", "FL1"],
        pns_labels=["Time", "CD3"],
        events_df=df,
        channel_ranges={"Time": 1000.0, "FL1": 1000.0},
        metadata={"$timestep": "0.01", "$pnstep": "0.01", "is_preprocessed": "True"},
    )

    time_sec, duration, time_ch = extract_time_in_seconds(fcs_mock)
    assert time_ch == "Time"
    # Duration must be 120 seconds, NOT compressed by 100x to 1.2 seconds!
    assert duration == pytest.approx(120.0, rel=1e-2)
    assert time_sec[-1] == pytest.approx(120.0, rel=1e-2)


def test_compute_time_bins_equal_events():
    """Verify compute_time_bins supports equal-event chunking via events_per_bin (T7)."""
    t = np.linspace(0.0, 50.0, 1000)
    cfg = TimeQCConfig(events_per_bin=200)

    bin_idx, starts, ends, counts = compute_time_bins(t, 50.0, cfg)
    # 1000 events / 200 per bin = exactly 5 bins
    assert len(starts) == 5
    assert len(ends) == 5
    assert list(counts) == [200, 200, 200, 200, 200]
    assert np.all(bin_idx >= 0) and np.all(bin_idx <= 4)


def test_extract_time_duration_clock_fallback():
    """Verify extract_time_in_seconds falls back to $BTIM / $ETIM clock timestamps (T10)."""
    from flowx.io.parser import FCSData
    from flowx.qc.automated import extract_time_in_seconds

    n_events = 500
    t = np.linspace(0.0, 180.0, n_events)
    df = pd.DataFrame({"TIME": t, "FL1": np.ones(n_events)})
    # No @acqtime, but valid $BTIM and $ETIM (duration = 3 minutes 30 seconds = 210 seconds)
    fcs_mock = FCSData(
        filepath="mock.fcs",
        filename="mock.fcs",
        event_count=n_events,
        channel_count=2,
        pnn_labels=["TIME", "FL1"],
        pns_labels=["Time", "CD3"],
        events_df=df,
        channel_ranges={"TIME": 1000.0, "FL1": 1000.0},
        metadata={"$btim": "10:00:00", "$etim": "10:03:30"},
    )

    time_sec, duration, time_ch = extract_time_in_seconds(fcs_mock)
    assert duration == pytest.approx(210.0, rel=1e-2)


def test_extract_time_non_monotonic_unwrapping():
    """Verify extract_time_in_seconds unwraps non-monotonic timestamps from file concatenation (T11)."""
    from flowx.io.parser import FCSData
    from flowx.qc.automated import extract_time_in_seconds

    # Concatenated run: runs from 0..10, then resets to 0..10
    t1 = np.linspace(0.0, 10.0, 50)
    t2 = np.linspace(0.0, 10.0, 50)
    raw_time = np.concatenate([t1, t2])
    n_events = len(raw_time)

    df = pd.DataFrame({"Time": raw_time, "FL1": np.ones(n_events)})
    fcs_mock = FCSData(
        filepath="mock.fcs",
        filename="mock.fcs",
        event_count=n_events,
        channel_count=2,
        pnn_labels=["Time", "FL1"],
        pns_labels=["Time", "CD3"],
        events_df=df,
        channel_ranges={"Time": 1000.0, "FL1": 1000.0},
        metadata={},
    )

    time_sec, duration, time_ch = extract_time_in_seconds(fcs_mock)
    # Output must be non-decreasing
    assert np.all(np.diff(time_sec) >= -1e-5)
    assert duration >= 19.0


def test_filter_flow_rate_low_rate_envelope():
    """Verify filter_flow_rate_anomalies does not flag steady low-flow-rate acquisitions (T4)."""
    cfg = TimeQCConfig(flow_rate_mad_factor=3.0, flow_rate_min_eps=50.0)

    # Low flow rate acquisition: steady around 20 eps across 20 bins, with one true clog at bin 15 (0 eps)
    bin_counts = np.array([20] * 20)
    bin_counts[15] = 0  # True clog dropout

    bin_starts = np.arange(20, dtype=float)
    bin_ends = bin_starts + 1.0
    bin_indices = np.repeat(np.arange(20), bin_counts)

    anom_events, anom_bins, stats = filter_flow_rate_anomalies(
        bin_counts=bin_counts,
        bin_starts=bin_starts,
        bin_ends=bin_ends,
        bin_indices=bin_indices,
        config=cfg,
    )

    # Bin 15 should be flagged as clog
    assert anom_bins[15] == True
    # Normal bins (e.g. 0-14) at 20 eps should NOT be flagged despite being < 50 eps
    assert anom_bins[0] == False
    assert anom_bins[1] == False
    assert anom_bins[5] == False
    assert stats["median_eps"] == pytest.approx(20.0, rel=0.1)


def test_filter_signal_drift_robust_stability():
    """Verify filter_signal_drift does not produce false positives on steady wide-variance populations (T6)."""
    np.random.seed(42)
    cfg = TimeQCConfig(signal_drift_mad_factor=3.5)
    n_bins = 25
    events_per_bin = 200
    total_events = n_bins * events_per_bin
    bin_indices = np.repeat(np.arange(n_bins), events_per_bin)

    # Wide variance normal distribution (mean=5000, std=1500) that is completely stationary across all bins
    ch_steady = np.random.normal(5000, 1500, total_events)
    df = pd.DataFrame({"FL1": ch_steady})

    anom_events, anom_bins, drift_stats = filter_signal_drift(
        events_df=df,
        bin_indices=bin_indices,
        n_bins=n_bins,
        channels=["FL1"],
        config=cfg,
    )

    assert "FL1" in drift_stats
    # Stationary distribution should NOT be flagged as drifting
    assert drift_stats["FL1"].is_flagged == False
    assert np.sum(anom_bins) == 0


def test_configurable_qc_pass_warn_thresholds():
    """Verify configurable pass_threshold_pct and warn_threshold_pct in TimeQCConfig (T12)."""
    from flowx.io.parser import FCSData
    from flowx.qc.engine import apply_time_qc

    # Create mock data with exactly 88% passing events (120 margin events)
    n_events = 1000
    t = np.linspace(0.0, 20.0, n_events)
    fl1 = np.ones(n_events) * 500.0
    fl1[:120] = 1048576.0

    df = pd.DataFrame({"TIME": t, "FL1": fl1})
    fcs_mock = FCSData(
        filepath="mock.fcs",
        filename="mock.fcs",
        event_count=n_events,
        channel_count=2,
        pnn_labels=["TIME", "FL1"],
        pns_labels=["Time", "CD3"],
        events_df=df,
        channel_ranges={"TIME": 1000.0, "FL1": 1048576.0},
        metadata={},
    )

    # Default thresholds: pass >= 90%, warn >= 75% -> 88% is WARNING
    cfg_default = TimeQCConfig(sample_type="sample", enable_signal_drift=False)
    res_def = apply_time_qc(fcs_mock, config=cfg_default)
    assert res_def.passed_pct == 88.0
    assert res_def.qc_status == "WARNING"

    # Custom relaxed thresholds: pass >= 85%, warn >= 70% -> 88% becomes PASS
    cfg_relaxed = TimeQCConfig(
        sample_type="sample", enable_signal_drift=False, pass_threshold_pct=85.0, warn_threshold_pct=70.0
    )
    res_rel = apply_time_qc(fcs_mock, config=cfg_relaxed)
    assert res_rel.qc_status == "PASS"

    # Custom strict thresholds: pass >= 95%, warn >= 90% -> 88% becomes FAIL
    cfg_strict = TimeQCConfig(
        sample_type="sample", enable_signal_drift=False, pass_threshold_pct=95.0, warn_threshold_pct=90.0
    )
    res_str = apply_time_qc(fcs_mock, config=cfg_strict)
    assert res_str.qc_status == "FAIL"


def test_time_qc_on_case1_variant1(navios_suite_dir):
    """Verify Time QC runs on Variant 1 Navios file with 'TIME' channel and 20-bit range (T2, T3)."""
    from flowx.io.parser import parse_fcs
    from flowx.qc.engine import apply_time_qc

    case1_path = navios_suite_dir / "Case1_A.fcs"
    fcs_data = parse_fcs(case1_path)
    res = apply_time_qc(fcs_data, config=TimeQCConfig.for_sample())

    assert res.time_channel == "TIME"
    assert res.passed_pct > 30.0
    assert res.passed_events > 500
    assert res.duration_sec > 300.0
    assert res.total_events == fcs_data.event_count



