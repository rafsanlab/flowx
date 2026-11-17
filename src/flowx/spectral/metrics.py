"""Spectral flow cytometry metrics and quality assessment.

Calculates Cosine Spectral Similarity Matrices, Panel Complexity Index, condition numbers,
and post-unmixing Spillover Spreading Matrices (SSM) to assess panel health and deconvolution
stability.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from flowx.spectral.models import ReferenceSpectralLibrary, SpectralMetricsResult

logger = logging.getLogger(__name__)


def compute_spectral_similarity_matrix(
    spectral_library: Union[ReferenceSpectralLibrary, pd.DataFrame, np.ndarray],
    fluorophore_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute pairwise Cosine Spectral Similarity Matrix between reference fluorophore spectra.

    Formula:
      Similarity(v_i, v_j) = (v_i . v_j) / (||v_i||_2 * ||v_j||_2)

    Values range from 0.0 (completely distinct / orthogonal emission) to 1.0 (identical emission).

    Args:
        spectral_library: ReferenceSpectralLibrary instance, DataFrame, or 2D NumPy array.
        fluorophore_names: Optional list of fluorophore names if 2D NumPy array provided.

    Returns:
        Pandas DataFrame of shape (P, P) with pairwise cosine similarities.
    """
    if isinstance(spectral_library, ReferenceSpectralLibrary):
        mat = spectral_library.matrix
        names = spectral_library.fluorophores
    elif isinstance(spectral_library, pd.DataFrame):
        mat = spectral_library.to_numpy(dtype=np.float64)
        names = list(spectral_library.columns)
    else:
        mat = np.asarray(spectral_library, dtype=np.float64)
        names = fluorophore_names or [f"param_{i}" for i in range(mat.shape[1])]

    n_params = mat.shape[1]
    # Compute L2 norms per column (fluorophore vector)
    norms = np.linalg.norm(mat, axis=0)
    norms = np.maximum(norms, 1e-12)

    # Normalized unit vectors
    unit_mat = mat / norms[np.newaxis, :]  # Shape: (D, P)

    # Cosine similarity is the dot product of unit vectors
    sim_mat = unit_mat.T @ unit_mat  # Shape: (P, P)
    # Numerical clip to [0.0, 1.0]
    sim_mat = np.clip(sim_mat, 0.0, 1.0)
    np.fill_diagonal(sim_mat, 1.0)

    return pd.DataFrame(sim_mat, index=names, columns=names)


def compute_panel_complexity_index(
    spectral_library: Union[ReferenceSpectralLibrary, pd.DataFrame, np.ndarray],
) -> float:
    """Calculate the overall Panel Complexity Index for a spectral panel.

    Measures the condition of the overdetermined spectral reference matrix M:
      Complexity Index = Tr((M^T M)^(-1))

    A lower complexity index reflects a panel well-distributed across laser lines with minimal
    collinearity, whereas a high complexity index flags severe variance inflation and spreading.

    Args:
        spectral_library: ReferenceSpectralLibrary, DataFrame, or 2D array of shape (D, P).

    Returns:
        Floating point Panel Complexity Index.
    """
    if isinstance(spectral_library, ReferenceSpectralLibrary):
        mat = spectral_library.matrix
    elif isinstance(spectral_library, pd.DataFrame):
        mat = spectral_library.to_numpy(dtype=np.float64)
    else:
        mat = np.asarray(spectral_library, dtype=np.float64)

    gram_matrix = mat.T @ mat  # Shape: (P, P)

    try:
        inv_gram = np.linalg.inv(gram_matrix)
        complexity = float(np.trace(inv_gram))
    except np.linalg.LinAlgError:
        # Singular or near-singular: use Moore-Penrose pseudo-inverse
        pinv_gram = np.linalg.pinv(gram_matrix)
        complexity = float(np.trace(pinv_gram))

    return max(complexity, 1.0)


def compute_spectral_spillover_spreading_matrix(
    spectral_library: Union[ReferenceSpectralLibrary, pd.DataFrame, np.ndarray],
    alpha_poisson: float = 1.0,
) -> pd.DataFrame:
    """Calculate analytical post-unmixing Spillover Spreading Matrix (SSM) per Nguyen et al. (2013).

    In spectral flow cytometry deconvolution, the unmixing pseudoinverse is U = (M^T M)^(-1) M^T.
    Under Poisson photon statistics, the intrinsic spreading error introduced by fluorophore i
    into unmixed channel j per square root of intensity is:
      SSM[i, j] = sqrt(alpha * sum_{d=1}^D (U_{j, d}^2 * M_{d, i}))

    Diagonal elements SSM[i, i] are set to 0.0 by convention.

    Args:
        spectral_library: ReferenceSpectralLibrary, DataFrame, or 2D array of shape (D, P).
        alpha_poisson: Poisson variance scaling factor (default: 1.0).

    Returns:
        Pandas DataFrame of shape (P, P) containing pairwise spillover spreading coefficients.
    """
    if isinstance(spectral_library, ReferenceSpectralLibrary):
        mat = spectral_library.matrix
        names = spectral_library.fluorophores
    elif isinstance(spectral_library, pd.DataFrame):
        mat = spectral_library.to_numpy(dtype=np.float64)
        names = list(spectral_library.columns)
    else:
        mat = np.asarray(spectral_library, dtype=np.float64)
        names = [f"param_{i}" for i in range(mat.shape[1])]

    # Compute unmixing pseudoinverse U = (M^T M)^(-1) M^T of shape (P, D)
    try:
        gram = mat.T @ mat
        inv_gram = np.linalg.inv(gram)
        U = inv_gram @ mat.T
    except np.linalg.LinAlgError:
        U = np.linalg.pinv(mat)

    # SSM^2[j, i] = sum_d U[j, d]^2 * M[d, i] = (U^2) @ M
    ssm_sq = (U ** 2) @ mat  # Shape: (P, P)
    ssm_mat = np.sqrt(np.maximum(alpha_poisson * ssm_sq.T, 0.0))  # Shape: (P, P) where [i, j] is dye i spreading into j
    np.fill_diagonal(ssm_mat, 0.0)

    return pd.DataFrame(ssm_mat, index=names, columns=names)


def evaluate_spectral_panel(
    spectral_library: ReferenceSpectralLibrary,
    warning_similarity_threshold: float = 0.85,
    danger_similarity_threshold: float = 0.95,
) -> SpectralMetricsResult:
    """Evaluate comprehensive health metrics for a spectral flow cytometry panel.

    Args:
        spectral_library: Configured ReferenceSpectralLibrary.
        warning_similarity_threshold: Similarity cutoff to flag potential unmixing overlap (default: 0.85).
        danger_similarity_threshold: Similarity cutoff to flag severe collinearity (default: 0.95).

    Returns:
        SpectralMetricsResult container with similarity matrix, complexity index, condition number,
        list of overlapping fluorophore pairs, and Spillover Spreading Matrix (SSM).
    """
    sim_df = compute_spectral_similarity_matrix(spectral_library)
    complexity = compute_panel_complexity_index(spectral_library)
    cond_num = spectral_library.condition_number
    ssm_df = compute_spectral_spillover_spreading_matrix(spectral_library)

    high_sim_pairs: List[Tuple[str, str, float]] = []
    names = list(sim_df.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            val = float(sim_df.iloc[i, j])
            if val >= warning_similarity_threshold:
                high_sim_pairs.append((names[i], names[j], val))
                if val >= danger_similarity_threshold:
                    logger.warning(
                        "High spectral collinearity between '%s' and '%s' (similarity = %.3f >= %.2f)",
                        names[i],
                        names[j],
                        val,
                        danger_similarity_threshold,
                    )

    # Sort pairs by descending similarity
    high_sim_pairs.sort(key=lambda x: x[2], reverse=True)

    return SpectralMetricsResult(
        similarity_matrix=sim_df,
        complexity_index=complexity,
        condition_number=cond_num,
        high_similarity_pairs=high_sim_pairs,
        ssm_dataframe=ssm_df,
    )

