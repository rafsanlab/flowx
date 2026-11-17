"""Unit tests for gating operations and hierarchy."""

import numpy as np
import pandas as pd
import pytest

from flowx.gating.density_gating import (
    DensityCutoffResult,
    build_adaptive_density_hierarchy,
    find_density_minimum,
)
from flowx.gating.gate import (
    GatingHierarchy,
    PolygonGate,
    RangeGate,
    RectangleGate,
)


def test_rectangle_gate():
    """Verify RectangleGate filters points correctly."""
    df = pd.DataFrame({
        "X": [10, 20, 30, 40],
        "Y": [100, 200, 300, 400],
    })
    gate = RectangleGate(name="Box", channels=["X", "Y"], x_min=15, x_max=35, y_min=150, y_max=350)
    mask = gate.apply(df)
    # Events at index 1 (20, 200) and 2 (30, 300) should be inside
    assert np.array_equal(mask, [False, True, True, False])


def test_polygon_gate():
    """Verify PolygonGate point-in-polygon containment."""
    df = pd.DataFrame({
        "X": [0.0, 5.0, 10.0, 5.0],
        "Y": [0.0, 5.0, 0.0, 20.0],
    })
    # Triangle with vertices (0,0), (10,0), (5,10)
    gate = PolygonGate(
        name="Triangle",
        channels=["X", "Y"],
        vertices=[[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]],
    )
    mask = gate.apply(df)
    # (5, 5) is strictly inside; (5, 20) is outside
    assert mask[1] is np.True_ or mask[1] == True
    assert mask[3] is np.False_ or mask[3] == False


def test_range_gate():
    """Verify RangeGate 1D thresholds."""
    df = pd.DataFrame({"Val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    gate = RangeGate(name="Threshold", channel="Val", min_val=2.5, max_val=4.5)
    mask = gate.apply(df)
    assert np.array_equal(mask, [False, False, True, True, False])


def test_gating_hierarchy():
    """Verify parent-child cascading membership and population statistics."""
    df = pd.DataFrame({
        "FSC": [100, 200, 300, 400, 500],
        "SSC": [10, 20, 30, 40, 50],
        "CD3": [1, 2, 8, 9, 10],
    })
    hierarchy = GatingHierarchy()

    # Gate 1: FSC >= 250 (indices 2, 3, 4)
    g1 = RectangleGate(name="FSC_Positive", channels=["FSC"], x_min=250)
    hierarchy.add_gate(g1)

    # Gate 2: Child of Gate 1, CD3 >= 8.5 (indices 3, 4)
    g2 = RangeGate(name="CD3_High", channel="CD3", min_val=8.5, parent_gate="FSC_Positive")
    hierarchy.add_gate(g2)

    masks = hierarchy.evaluate(df)
    assert np.sum(masks["root"]) == 5
    assert np.sum(masks["FSC_Positive"]) == 3
    assert np.sum(masks["CD3_High"]) == 2

    stats = hierarchy.get_population_statistics(df)
    assert len(stats) == 3
    assert stats.loc[stats["Gate"] == "CD3_High", "Count"].iloc[0] == 2
    assert stats.loc[stats["Gate"] == "CD3_High", "% Total"].iloc[0] == 40.0


def test_find_density_minimum_bimodal():
    """Verify density valley identification on bimodal distribution."""
    np.random.seed(42)
    # Mixture of two Gaussians: mode 1 at 0.15, mode 2 at 0.50
    vals1 = np.random.normal(loc=0.15, scale=0.03, size=2000)
    vals2 = np.random.normal(loc=0.50, scale=0.03, size=2000)
    vals = np.concatenate([vals1, vals2])

    res = find_density_minimum(vals, channel_name="TestBimodal", gate_range=(0.10, 0.60))

    assert res.is_bimodal is True
    assert len(res.peaks) >= 2
    assert res.valley is not None
    # Valley should be located between the two modes (~0.25 - 0.40)
    assert 0.25 <= res.cutoff <= 0.40
    assert res.confidence > 0.5


def test_find_density_minimum_unimodal():
    """Verify fallback behavior on unimodal distribution."""
    np.random.seed(42)
    vals = np.random.normal(loc=0.25, scale=0.03, size=2000)

    res = find_density_minimum(vals, channel_name="TestUnimodal", default_cutoff=0.35)

    assert res.is_bimodal is False
    assert res.cutoff == 0.35
    assert res.valley is None


def test_find_density_minimum_multimodal_subpeaks():
    """Verify candidate pair search finds inter-cluster valley when main cluster has sub-modes."""
    np.random.seed(42)
    # Simulate live cluster with two overlapping sub-peaks (e.g. lymphocytes 0.38, monocytes 0.46)
    # and a distinct secondary dead cluster (0.85)
    vals = np.concatenate([
        np.random.normal(0.38, 0.03, 1500),
        np.random.normal(0.46, 0.03, 2000),
        np.random.normal(0.85, 0.03, 300),
    ])

    res = find_density_minimum(vals, channel_name="TestMultimodal", gate_range=(0.25, 0.85))

    assert res.is_bimodal is True
    # The valley separating the main cluster (0.38-0.46) from the dead cluster (0.85)
    assert 0.60 <= res.cutoff <= 0.80
    assert res.confidence >= 0.15


def test_build_adaptive_density_hierarchy(tqc_sample_fcs_path):
    """Verify end-to-end adaptive hierarchy generation on FCS data."""
    from flowx.io.parser import parse_fcs
    from flowx.math.transforms import LogicleTransform, apply_transform_to_dataframe

    fcs = parse_fcs(tqc_sample_fcs_path)
    transform = LogicleTransform()
    fluoro_ch = [ch for ch in fcs.pnn_labels if "fsc" not in ch.lower() and "ssc" not in ch.lower() and "time" not in ch.lower()]
    df_xform = apply_transform_to_dataframe(fcs.events_df, channels=fluoro_ch, transform=transform)

    hierarchy, cutoffs = build_adaptive_density_hierarchy(fcs, df_xform)

    # Verify standard gates are present
    assert "Cells" in hierarchy.order
    assert "Singlets" in hierarchy.order
    assert "Viable" in hierarchy.order
    assert "CD3+" in hierarchy.order
    assert "CD4+ T Cells" in hierarchy.order
    assert "CD8+ T Cells" in hierarchy.order

    # Verify cutoffs were estimated
    assert "CD3+" in cutoffs
    cd3_cutoff = cutoffs["CD3+"].cutoff
    assert 0.20 <= cd3_cutoff <= 0.65

    masks = hierarchy.evaluate(df_xform)
    assert np.sum(masks["Cells"]) > 0
    assert np.sum(masks["Singlets"]) > 0
    assert np.sum(masks["Viable"]) > 0
    assert np.sum(masks["CD3+"]) > 0
    assert np.sum(masks["CD4+ T Cells"]) > 0
    assert np.sum(masks["CD8+ T Cells"]) > 0

