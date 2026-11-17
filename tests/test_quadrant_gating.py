"""Unit tests for 2D quadrant gating and bivariate density cutoff detection."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from flowx.gating.gate import QuadrantGate, create_gate_from_dict
from flowx.gating.density_gating import find_quadrant_cutoffs
from flowx.visualization.plotting import plot_quadrant_gate


def test_quadrant_gate_masks_and_frequencies():
    """Verify QuadrantGate produces accurate Boolean masks and percentage frequencies."""
    # Create 4 distinct clusters in quadrants
    rng = np.random.RandomState(42)
    q1_pts = np.column_stack([rng.normal(0.2, 0.05, 50), rng.normal(0.8, 0.05, 50)])
    q2_pts = np.column_stack([rng.normal(0.8, 0.05, 100), rng.normal(0.8, 0.05, 100)])
    q3_pts = np.column_stack([rng.normal(0.2, 0.05, 150), rng.normal(0.2, 0.05, 150)])
    q4_pts = np.column_stack([rng.normal(0.8, 0.05, 200), rng.normal(0.2, 0.05, 200)])

    all_pts = np.vstack([q1_pts, q2_pts, q3_pts, q4_pts])
    df = pd.DataFrame(all_pts, columns=["CD80", "CD206"])

    gate = QuadrantGate(
        name="Macrophage_Polarization",
        channels=["CD80", "CD206"],
        x_cutoff=0.50,
        y_cutoff=0.50,
        quadrant="all",
    )

    counts = gate.get_quadrant_counts(df)
    freqs = gate.get_quadrant_frequencies(df)

    assert counts["Q1"] == 50   # UL: CD80- CD206+
    assert counts["Q2"] == 100  # UR: CD80+ CD206+
    assert counts["Q3"] == 150  # LL: CD80- CD206-
    assert counts["Q4"] == 200  # LR: CD80+ CD206-

    assert pytest.approx(freqs["Q1"], abs=0.1) == 10.0
    assert pytest.approx(freqs["Q2"], abs=0.1) == 20.0
    assert pytest.approx(freqs["Q3"], abs=0.1) == 30.0
    assert pytest.approx(freqs["Q4"], abs=0.1) == 40.0

    # Specific quadrant application
    gate_q2 = QuadrantGate(name="M1_M2_Dual", channels=["CD80", "CD206"], x_cutoff=0.50, y_cutoff=0.50, quadrant="Q2")
    m2 = gate_q2.apply(df)
    assert np.sum(m2) == 100


def test_quadrant_gate_serialization_roundtrip():
    """Verify QuadrantGate can be serialized to and deserialized from dictionary."""
    gate = QuadrantGate(
        name="TestQuad",
        channels=["FITC", "PE"],
        x_cutoff=0.35,
        y_cutoff=0.45,
        quadrant="Q2",
        parent_gate="Live",
        metadata={"author": "FlowX"},
    )
    d = gate.to_dict()
    assert d["type"] == "quadrant"
    assert d["x_cutoff"] == 0.35
    assert d["y_cutoff"] == 0.45

    reconstructed = create_gate_from_dict(d)
    assert isinstance(reconstructed, QuadrantGate)
    assert reconstructed.name == "TestQuad"
    assert reconstructed.x_cutoff == 0.35
    assert reconstructed.y_cutoff == 0.45
    assert reconstructed.quadrant == "Q2"


def test_find_quadrant_cutoffs_density():
    """Verify automated 2D quadrant cutoff detection finds natural saddle points."""
    rng = np.random.RandomState(42)
    # Bimodal distributions along both X and Y
    x = np.concatenate([rng.normal(0.25, 0.05, 500), rng.normal(0.75, 0.05, 500)])
    y = np.concatenate([rng.normal(0.20, 0.05, 500), rng.normal(0.80, 0.05, 500)])
    df = pd.DataFrame({"MarkerX": x, "MarkerY": y})

    res = find_quadrant_cutoffs(
        data=df,
        x_channel="MarkerX",
        y_channel="MarkerY",
        x_range=(0.30, 0.70),
        y_range=(0.30, 0.70),
        refine_2d=True,
    )

    assert 0.40 <= res.x_cutoff <= 0.60
    assert 0.40 <= res.y_cutoff <= 0.60
    assert res.gate is not None
    assert sum(res.quadrant_counts.values()) == 1000


def test_plot_quadrant_gate(tmp_path: Path):
    """Verify plot_quadrant_gate exports cleanly."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "CD80": rng.uniform(0, 1, 200),
        "CD206": rng.uniform(0, 1, 200),
    })
    out_img = tmp_path / "test_quad_gate.png"
    fig = plot_quadrant_gate(df, "CD80", "CD206", x_cutoff=0.5, y_cutoff=0.5, output_path=out_img)
    assert out_img.exists()
    assert out_img.stat().st_size > 1000

