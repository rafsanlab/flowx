"""Spectral deconvolution (unmixing) engine for full-spectrum flow cytometry.

Solves overdetermined linear systems (D detectors >= P fluorophores + autofluorescence)
via Weighted Least Squares (WLS) with Poisson photon noise weighting, Ordinary Least
Squares (OLS), and regularized bounded solvers. Computes per-event unmixing residuals
for quality auditing in compliance with MIFlowCyt guidelines.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from flowx.spectral.models import (
    ReferenceSpectralLibrary,
    UnmixingConfig,
    UnmixingResult,
)

logger = logging.getLogger(__name__)


def unmix_events_ols(
    events_matrix: np.ndarray,
    spectral_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Ordinary Least Squares (OLS) spectral deconvolution.

    Formula: X = Y * M * (M^T * M)^(-1)

    Args:
        events_matrix: 2D NumPy array of shape (N_events, D_detectors).
        spectral_matrix: 2D NumPy array of shape (D_detectors, P_fluorophores).

    Returns:
        Tuple of (unmixed_abundances (N, P), residuals_norm (N,)).
    """
    # M has shape (D, P)
    # events_matrix has shape (N, D)
    # Using np.linalg.lstsq(M, Y.T) -> shape (P, N), then transpose to (N, P)
    unmixed_t, residuals, rank, s = np.linalg.lstsq(
        spectral_matrix, events_matrix.T, rcond=None
    )
    unmixed = unmixed_t.T  # Shape: (N, P)

    # Compute reconstructed detector intensities and residual norms
    reconstructed = unmixed @ spectral_matrix.T  # Shape: (N, D)
    diff = events_matrix - reconstructed
    residuals_norm = np.linalg.norm(diff, axis=1)  # Shape: (N,)

    return unmixed, residuals_norm


def unmix_events_wls(
    events_matrix: np.ndarray,
    spectral_matrix: np.ndarray,
    alpha_poisson: float = 1.0,
    electronic_noise: float = 50.0,
    batch_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Weighted Least Squares (WLS) spectral deconvolution with Poisson photon noise weighting.

    In spectral cytometry, detector noise variance scales with signal intensity:
      sigma_{i,d}^2 = alpha * max(y_{i,d}, 0) + electronic_noise^2
      W_i = diag(1 / sigma_{i,d}^2)
      x_i = (M^T W_i M)^(-1) M^T W_i y_i

    To prevent noise amplification from negative fluctuations, expected intensities are derived
    from an initial stable OLS projection.

    Args:
        events_matrix: 2D NumPy array of shape (N_events, D_detectors).
        spectral_matrix: 2D NumPy array of shape (D_detectors, P_fluorophores).
        alpha_poisson: Poisson variance scaling coefficient.
        electronic_noise: Base electronic read noise variance floor.
        batch_size: Number of events to process per vectorized block.

    Returns:
        Tuple of (unmixed_abundances (N, P), residuals_norm (N,)).
    """
    n_events, n_detectors = events_matrix.shape
    n_fluorophores = spectral_matrix.shape[1]

    # Initial OLS pass to get smooth non-negative expected detector intensities
    ols_unmixed, _ = unmix_events_ols(events_matrix, spectral_matrix)
    expected_detectors = np.maximum(ols_unmixed @ spectral_matrix.T, 0.0)  # Shape: (N, D)

    # Compute detector variance: Var(y) = alpha * y + noise^2
    var_matrix = alpha_poisson * expected_detectors + (electronic_noise ** 2)
    weights_matrix = 1.0 / np.maximum(var_matrix, 1.0)  # Shape: (N, D)

    unmixed = np.zeros((n_events, n_fluorophores), dtype=np.float64)
    residuals_norm = np.zeros(n_events, dtype=np.float64)

    # Process in vectorized batches
    for start_idx in range(0, n_events, batch_size):
        end_idx = min(start_idx + batch_size, n_events)
        b_events = events_matrix[start_idx:end_idx]  # Shape: (B, D)
        b_weights = weights_matrix[start_idx:end_idx]  # Shape: (B, D)
        B = end_idx - start_idx

        # Vectorized batch normal equations: (M^T W_i M) x_i = M^T W_i y_i
        # A has shape (B, P, P); rhs has shape (B, P)
        A = np.einsum("dp,dq,bd->bpq", spectral_matrix, spectral_matrix, b_weights)
        rhs = np.einsum("dp,bd,bd->bp", spectral_matrix, b_weights, b_events)

        try:
            x_b = np.linalg.solve(A, rhs[:, :, np.newaxis]).squeeze(-1)
        except np.linalg.LinAlgError:
            # Add small Tikhonov ridge regularization if A is ill-conditioned
            reg = 1e-7 * np.eye(n_fluorophores, dtype=np.float64)
            A_reg = A + reg[np.newaxis, :, :]
            try:
                x_b = np.linalg.solve(A_reg, rhs[:, :, np.newaxis]).squeeze(-1)
            except np.linalg.LinAlgError:
                # Robust fallback to lstsq if batch solve fails
                sqrt_w = np.sqrt(b_weights)
                x_b = np.zeros((B, n_fluorophores), dtype=np.float64)
                for i in range(B):
                    w_diag = sqrt_w[i]
                    M_w = spectral_matrix * w_diag[:, np.newaxis]
                    y_w = b_events[i] * w_diag
                    x_i, _, _, _ = np.linalg.lstsq(M_w, y_w, rcond=None)
                    x_b[i] = x_i

        unmixed[start_idx:end_idx] = x_b

        # Residual norm in unweighted physical detector space (fully vectorized)
        reconstructed = x_b @ spectral_matrix.T  # Shape: (B, D)
        residuals_norm[start_idx:end_idx] = np.linalg.norm(b_events - reconstructed, axis=1)

    return unmixed, residuals_norm


def unmix_dataframe(
    events_df: pd.DataFrame,
    spectral_library: ReferenceSpectralLibrary,
    config: Optional[UnmixingConfig] = None,
) -> UnmixingResult:
    """Deconvolve multi-parameter events DataFrame using a ReferenceSpectralLibrary.

    Preserves non-optical channels (FSC, SSC, Time) and appends newly unmixed fluorophore
    parameters, extracted autofluorescence channels, and the per-event unmixing residual.

    Args:
        events_df: DataFrame containing measured flow cytometry events.
        spectral_library: Configured ReferenceSpectralLibrary containing optical detectors.
        config: Optional UnmixingConfig specifying solver, weights, and slack.

    Returns:
        UnmixingResult container with unmixed DataFrame and quality diagnostics.

    Raises:
        KeyError: If any optical detector channel in the library is missing from events_df.
    """
    t0 = time.time()
    cfg = config or UnmixingConfig()
    detectors = spectral_library.detectors
    fluorophores = spectral_library.fluorophores

    missing_detectors = [d for d in detectors if d not in events_df.columns]
    if missing_detectors:
        raise KeyError(
            f"Events DataFrame missing {len(missing_detectors)} optical detectors required "
            f"by the spectral library: {missing_detectors[:5]}"
        )

    raw_optical = events_df[detectors].to_numpy(dtype=np.float64)
    spec_mat = spectral_library.matrix

    # Select solver
    method_name = cfg.method.lower().strip()
    if method_name == "wls":
        unmixed_mat, res_norm = unmix_events_wls(
            raw_optical,
            spec_mat,
            alpha_poisson=cfg.alpha_poisson,
            electronic_noise=cfg.electronic_noise,
        )
    elif method_name in ("ols", "linear"):
        unmixed_mat, res_norm = unmix_events_ols(raw_optical, spec_mat)
    elif method_name in ("nnls", "projected_ols"):
        # Projected non-negative with slack
        unmixed_mat, _ = unmix_events_ols(raw_optical, spec_mat)
        slack_floor = cfg.negative_slack_floor if cfg.allow_negative_slack else 0.0
        unmixed_mat = np.maximum(unmixed_mat, slack_floor)
        # Recompute physical residuals with the clamped abundances
        reconstructed = unmixed_mat @ spec_mat.T
        res_norm = np.linalg.norm(raw_optical - reconstructed, axis=1)
    else:
        logger.warning("Unknown unmixing method '%s'; defaulting to WLS", cfg.method)
        unmixed_mat, res_norm = unmix_events_wls(raw_optical, spec_mat)

    # Separate biological fluorophores from autofluorescence
    af_pattern = re.compile(r"^(af|autofluorescence)([_\-\s].*)?$", re.IGNORECASE)
    af_channels: List[str] = []
    marker_channels: List[str] = []
    for fluoro in fluorophores:
        ref = spectral_library.references.get(fluoro)
        if ref and ref.is_autofluorescence:
            af_channels.append(fluoro)
        elif af_pattern.match(fluoro.strip()):
            af_channels.append(fluoro)
        else:
            marker_channels.append(fluoro)

    # Build resulting DataFrame: non-optical channels + unmixed channels + residuals
    non_optical_cols = [c for c in events_df.columns if c not in detectors]
    res_df = pd.DataFrame(index=events_df.index)

    # Keep scatter and time
    for col in non_optical_cols:
        res_df[col] = events_df[col]

    # Add unmixed fluorophores and AF channels
    for j, fluoro in enumerate(fluorophores):
        res_df[fluoro] = unmixed_mat[:, j]

    # Add residual channel if configured
    residual_col = "Unmixing_Residual"
    if cfg.compute_residuals:
        res_df[residual_col] = res_norm

    # Calculate outlier statistics
    med_res = float(np.median(res_norm))
    mad_res = float(np.median(np.abs(res_norm - med_res)))
    rsd_res = max(1.4826 * mad_res, 1.0)
    outlier_threshold = med_res + (cfg.residual_outlier_sigma * rsd_res)
    outliers_cnt = int(np.sum(res_norm > outlier_threshold))

    elapsed = time.time() - t0

    return UnmixingResult(
        unmixed_df=res_df,
        residuals_norm=res_norm,
        spectral_library=spectral_library,
        method_used=method_name,
        residual_channel_name=residual_col,
        fluorophore_channels=marker_channels,
        af_channels=af_channels,
        scatter_and_time_channels=non_optical_cols,
        mean_residual_norm=float(np.mean(res_norm)),
        median_residual_norm=med_res,
        residual_outliers_count=outliers_cnt,
        execution_time_seconds=elapsed,
    )

