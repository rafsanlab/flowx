"""Unit tests for transformations module."""

import numpy as np
import pandas as pd
import pytest

from flowx.math.transforms import (
    AsinhTransform,
    BaseTransform,
    BiexponentialTransform,
    LogicleTransform,
    LogTransform,
    apply_transform_to_dataframe,
    create_transform,
    get_optimal_transform,
)


def test_logicle_invertibility():
    """Verify Logicle forward and inverse transformation round-trip."""
    xform = LogicleTransform(param_t=262144.0, param_w=0.5, param_m=4.5, param_a=0.0)
    test_vals = np.array([0.0, 10.0, 500.0, 10000.0, 200000.0])

    transformed = xform.apply(test_vals)
    # Check bounded within [0, 1] approximately for positive range
    assert np.all(transformed >= 0.0)
    assert np.all(transformed <= 1.05)

    inverted = xform.inverse(transformed)
    assert np.allclose(test_vals, inverted, atol=1e-5)


def test_biexponential_invertibility():
    """Verify FlowJo Biexponential forward and inverse transformation round-trip."""
    xform = BiexponentialTransform(
        max_value=262144.0,
        positive=4.418540,
        width_basis=-10.0,
        negative=0.0,
        channel_range=4096,
        normalized=True,
    )
    test_vals = np.array([-100.0, -10.0, 0.0, 10.0, 100.0, 1000.0, 10000.0, 262144.0])

    transformed = xform.apply(test_vals)
    # Check normalized bounded within [0, 1]
    assert np.all(transformed >= 0.0)
    assert np.all(transformed <= 1.0)
    # Monotonicity check
    assert np.all(np.diff(transformed) >= 0)

    inverted = xform.inverse(transformed)
    # Reconstructed values should be close for non-extreme ranges
    assert np.allclose(test_vals[1:-1], inverted[1:-1], rtol=1e-2, atol=5.0)


def test_biexponential_unnormalized():
    """Verify unnormalized FlowJo Biexponential outputs [0, 4096] range."""
    xform = BiexponentialTransform(
        max_value=262144.0,
        positive=4.418540,
        width_basis=-10.0,
        negative=0.0,
        channel_range=4096,
        normalized=False,
    )
    test_vals = np.array([0.0, 100.0, 10000.0, 262144.0])
    transformed = xform.apply(test_vals)

    assert np.all(transformed >= 0.0)
    assert np.all(transformed <= 4096.0)
    assert np.isclose(transformed[-1], 4096.0)

    inverted = xform.inverse(transformed)
    assert np.allclose(test_vals, inverted, rtol=1e-2, atol=5.0)


def test_biexponential_properties_and_dict():
    """Verify property aliases and serialization for BiexponentialTransform."""
    xform = BiexponentialTransform(
        max_value=262144.0,
        positive=4.5,
        width_basis=-12.0,
        negative=0.5,
        channel_range=4096,
        normalized=True,
    )
    assert xform.param_t == 262144.0
    assert xform.param_m == 4.5
    assert xform.param_w == -12.0
    assert xform.param_a == 0.5

    d = xform.to_dict()
    assert d["type"] == "biexponential"
    assert d["max_value"] == 262144.0
    assert d["positive"] == 4.5
    assert d["width_basis"] == -12.0
    assert d["normalized"] is True


def test_asinh_invertibility():
    """Verify Asinh transform forward and inverse."""
    xform = AsinhTransform(cofactor=150.0)
    test_vals = np.array([-50.0, 0.0, 150.0, 3000.0])

    transformed = xform.apply(test_vals)
    inverted = xform.inverse(transformed)
    assert np.allclose(test_vals, inverted, atol=1e-5)


def test_log_transform():
    """Verify LogTransform works with floor."""
    xform = LogTransform(floor=1.0, base=10.0)
    test_vals = np.array([1.0, 10.0, 100.0, 1000.0])
    transformed = xform.apply(test_vals)
    assert np.allclose(transformed, [0.0, 1.0, 2.0, 3.0])

    inverted = xform.inverse(transformed)
    assert np.allclose(test_vals, inverted)


def test_apply_transform_to_dataframe():
    """Verify transform applied only to designated columns."""
    df = pd.DataFrame({
        "FSC-A": [10000.0, 20000.0],
        "CD3": [150.0, 3000.0],
    })
    xform = AsinhTransform(cofactor=150.0)
    res_df = apply_transform_to_dataframe(df, ["CD3"], xform)

    # FSC-A unchanged
    assert np.array_equal(res_df["FSC-A"].values, df["FSC-A"].values)
    # CD3 transformed
    assert not np.array_equal(res_df["CD3"].values, df["CD3"].values)
    assert np.isclose(res_df["CD3"].iloc[0], np.arcsinh(1.0))


def test_create_transform_factory():
    """Verify transform creation from dictionary config."""
    cfg_logicle = {"type": "logicle", "param_t": 10000.0, "param_w": 0.5, "param_m": 4.5, "param_a": 0.0}
    xform_l = create_transform(cfg_logicle)
    assert isinstance(xform_l, LogicleTransform)
    assert xform_l.param_t == 10000.0

    cfg_biex = {"type": "biexponential", "max_value": 262144.0, "positive": 4.4, "width_basis": -10.0}
    xform_b = create_transform(cfg_biex)
    assert isinstance(xform_b, BiexponentialTransform)
    assert xform_b.max_value == 262144.0


def test_get_optimal_transform():
    """Verify get_optimal_transform returns correct transform types."""
    class DummyFCS:
        def detect_max_range(self):
            return 262144.0

    fcs = DummyFCS()
    l_xform = get_optimal_transform(fcs, transform_type="logicle")
    assert isinstance(l_xform, LogicleTransform)

    b_xform = get_optimal_transform(fcs, transform_type="biexponential")
    assert isinstance(b_xform, BiexponentialTransform)
    assert b_xform.max_value == 262144.0

