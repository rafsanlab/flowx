"""Unit tests for FlowX spectral flow cytometry unmixing module.

Tests reference library construction, Cosine Spectral Similarity assessment,
Panel Complexity Index computation, OLS and WLS deconvolution solvers, and
autofluorescence profile extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowx.spectral.autofluorescence import (
    attach_autofluorescence_to_library,
    extract_autofluorescence_spectrum,
)
from flowx.spectral.library import (
    ReferenceSpectralLibraryBuilder,
    filter_optical_channels,
    is_optical_fluorescence_channel,
)
from flowx.spectral.metrics import (
    compute_panel_complexity_index,
    compute_spectral_similarity_matrix,
    evaluate_spectral_panel,
)
from flowx.spectral.models import (
    ReferenceSpectralLibrary,
    SpectralReference,
    UnmixingConfig,
)
from flowx.spectral.unmixing import (
    unmix_dataframe,
    unmix_events_ols,
    unmix_events_wls,
)


def test_optical_channel_filtering() -> None:
    """Test filtering of non-fluorescent scatter and time parameters."""
    channels = [
        "FSC-A",
        "FSC-H",
        "SSC-A",
        "SSC-B-A",
        "V-SSC-A",
        "VSSC-A",
        "SSC-V-A",
        "Time",
        "UV1-A",
        "B525-A",
        "R780-A",
    ]
    optical = filter_optical_channels(channels)
    assert "FSC-A" not in optical
    assert "SSC-A" not in optical
    assert "SSC-B-A" not in optical
    assert "V-SSC-A" not in optical
    assert "VSSC-A" not in optical
    assert "SSC-V-A" not in optical
    assert "Time" not in optical
    assert "UV1-A" in optical
    assert "B525-A" in optical
    assert "R780-A" in optical


def test_spectral_similarity_matrix() -> None:
    """Test cosine similarity matrix calculation."""
    # 4 detectors, 2 distinct dyes + 1 duplicate dye
    spec_mat = np.array([
        [1.0, 0.0, 1.0],
        [0.5, 0.2, 0.5],
        [0.0, 1.0, 0.0],
        [0.0, 0.5, 0.0],
    ])
    dyes = ["DyeA", "DyeB", "DyeA_copy"]
    sim_df = compute_spectral_similarity_matrix(spec_mat, fluorophore_names=dyes)

    assert sim_df.shape == (3, 3)
    assert np.isclose(sim_df.loc["DyeA", "DyeA"], 1.0)
    assert np.isclose(sim_df.loc["DyeA", "DyeA_copy"], 1.0)
    assert sim_df.loc["DyeA", "DyeB"] < 0.5


def test_panel_complexity_index() -> None:
    """Test panel complexity index computation."""
    # Orthogonal panel should have low complexity
    ortho_mat = np.eye(4)
    comp_ortho = compute_panel_complexity_index(ortho_mat)
    assert np.isclose(comp_ortho, 4.0)

    # Collinear panel should have high complexity
    collinear_mat = np.array([
        [1.0, 0.99],
        [0.1, 0.11],
        [0.0, 0.01],
    ])
    comp_collinear = compute_panel_complexity_index(collinear_mat)
    assert comp_collinear > comp_ortho


def test_spectral_spillover_spreading_matrix() -> None:
    """Test analytical post-unmixing Spillover Spreading Matrix (SSM) computation."""
    from flowx.spectral.metrics import compute_spectral_spillover_spreading_matrix

    M = np.array([
        [1.0, 0.1, 0.0],
        [0.8, 0.5, 0.1],
        [0.1, 1.0, 0.4],
        [0.0, 0.2, 1.0],
    ])
    dyes = ["DyeA", "DyeB", "DyeC"]
    ssm_df = compute_spectral_spillover_spreading_matrix(M, alpha_poisson=1.0)
    ssm_df.index = dyes
    ssm_df.columns = dyes

    assert ssm_df.shape == (3, 3)
    # Diagonal self-spreading must be 0.0
    for d in dyes:
        assert np.isclose(ssm_df.loc[d, d], 0.0)
    # Overlapping dyes have positive spreading error
    assert ssm_df.loc["DyeC", "DyeB"] > 0.1


def test_ols_unmixing_synthetic() -> None:
    """Test Ordinary Least Squares recovery of ground-truth abundances."""
    np.random.seed(42)
    # 5 detectors x 3 fluorophores
    M = np.array([
        [1.0, 0.1, 0.0],
        [0.8, 0.3, 0.0],
        [0.1, 1.0, 0.2],
        [0.0, 0.5, 0.9],
        [0.0, 0.0, 1.0],
    ])
    # Ground truth abundances for 100 events
    X_true = np.random.uniform(100.0, 5000.0, size=(100, 3))
    # Simulated detector signals
    Y = X_true @ M.T

    X_pred, res = unmix_events_ols(Y, M)

    assert np.allclose(X_pred, X_true, atol=1e-5)
    assert np.all(res < 1e-4)


def test_wls_unmixing_synthetic() -> None:
    """Test Weighted Least Squares deconvolution."""
    np.random.seed(42)
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.8, 0.0],
        [0.0, 1.0, 0.3],
        [0.0, 0.2, 1.0],
        [0.0, 0.0, 0.8],
    ])
    X_true = np.random.uniform(50.0, 2000.0, size=(50, 3))
    Y = X_true @ M.T

    X_pred, res = unmix_events_wls(Y, M, alpha_poisson=1.0, electronic_noise=10.0)
    assert np.allclose(X_pred, X_true, atol=1e-3)


def test_unmix_dataframe_workflow() -> None:
    """Test end-to-end unmix_dataframe with preserved non-optical columns."""
    detectors = ["D1", "D2", "D3", "D4"]
    fluorophores = ["FITC", "PE"]

    M = np.array([
        [1.0, 0.1],
        [0.8, 0.4],
        [0.2, 1.0],
        [0.0, 0.7],
    ])
    spec_lib = ReferenceSpectralLibrary(
        detectors=detectors,
        fluorophores=fluorophores,
        matrix=M,
        dataframe=pd.DataFrame(M, index=detectors, columns=fluorophores),
    )

    # Simulated DataFrame with FSC, SSC, optical detectors, and Time
    df_raw = pd.DataFrame({
        "FSC-A": [100000, 120000],
        "SSC-A": [40000, 45000],
        "Time": [1.0, 2.0],
        "D1": [1000.0, 200.0],
        "D2": [900.0, 500.0],
        "D3": [300.0, 1100.0],
        "D4": [50.0, 750.0],
    })

    cfg = UnmixingConfig(method="ols", compute_residuals=True)
    res = unmix_dataframe(df_raw, spec_lib, config=cfg)

    assert "FSC-A" in res.unmixed_df.columns
    assert "SSC-A" in res.unmixed_df.columns
    assert "Time" in res.unmixed_df.columns
    assert "FITC" in res.unmixed_df.columns
    assert "PE" in res.unmixed_df.columns
    assert "Unmixing_Residual" in res.unmixed_df.columns
    assert len(res.residuals_norm) == 2
    assert res.mean_residual_norm >= 0.0

