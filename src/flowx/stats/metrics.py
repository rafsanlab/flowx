"""Statistical metrics and mathematical primitives for cytometry cohort analysis.

Provides robust location estimators (MFI, GeoMFI), scale estimators (rSD, MAD, IQR),
effect size metrics (Log2 Fold Change, Cohen's d), and multiple testing corrections (Benjamini-Hochberg FDR)
conforming to MIFlowCyt standards and biostatistical best practices.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np


def median_fluorescence_intensity(values: Union[np.ndarray, list]) -> float:
    """Calculate the Median Fluorescence Intensity (MFI).

    Args:
        values: 1D array of single-cell marker intensity values.

    Returns:
        Median intensity, or 0.0 if array is empty.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    return float(np.median(arr))


def geometric_mean(
    values: Union[np.ndarray, list],
    pseudocount: float = 1.0,
) -> float:
    """Calculate Geometric Mean Fluorescence Intensity (GeoMFI).

    Essential for log-normally distributed fluorescence intensity channels.
    Values <= 0 are bounded by pseudocount to ensure strictly positive log evaluation.

    Args:
        values: 1D array of marker intensity values.
        pseudocount: Lower boundary floor for non-positive events (default: 1.0).

    Returns:
        Geometric mean intensity, or 0.0 if array is empty.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0

    clipped = np.maximum(arr, pseudocount)
    log_vals = np.log(clipped)
    return float(np.exp(np.mean(log_vals)))


def robust_sd(
    values: Union[np.ndarray, list],
    method: Literal["mad", "iqr"] = "mad",
) -> float:
    """Calculate the Robust Standard Deviation (rSD) of a marker distribution.

    Adheres to the MIFlowCyt framework to quantify population variance without
    distortion from bimodal staining tails, non-specific background, or outliers.

    Methods:
        - 'mad': Median Absolute Deviation scaled for normal consistency:
                 rSD = 1.4826 * median(|x - median(x)|)
        - 'iqr': Normalized Interquartile Range:
                 rSD = (Q75 - Q25) / 1.349

    Args:
        values: 1D array of single-cell marker values.
        method: Estimation method ('mad' or 'iqr'). Default 'mad'.

    Returns:
        Robust standard deviation estimate.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0

    if method == "mad":
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        return float(1.4826 * mad)
    elif method == "iqr":
        q75, q25 = np.percentile(arr, [75, 25])
        return float((q75 - q25) / 1.349)
    else:
        raise ValueError(f"Unknown robust_sd method: '{method}'. Expected 'mad' or 'iqr'.")


def log2_fold_change(
    ref_value: float,
    target_value: float,
    pseudocount: float = 0.01,
) -> float:
    """Calculate log2 fold change between target and reference cohort values.

    Formula:
        log2_FC = log2((target + pseudocount) / (ref + pseudocount))

    Args:
        ref_value: Baseline/reference group metric (e.g. control mean frequency).
        target_value: Experimental/target group metric (e.g. disease mean frequency).
        pseudocount: Small positive constant preventing division by zero (default: 0.01).

    Returns:
        log2 fold change.
    """
    ref_adj = max(float(ref_value), 0.0) + pseudocount
    target_adj = max(float(target_value), 0.0) + pseudocount
    return float(np.log2(target_adj / ref_adj))


def cohens_d(
    group1: Union[np.ndarray, list],
    group2: Union[np.ndarray, list],
) -> float:
    """Calculate Cohen's d standardized effect size between two groups.

    Formula:
        d = (mean(group2) - mean(group1)) / s_pooled
        s_pooled = sqrt(((n1 - 1)*s1^2 + (n2 - 1)*s2^2) / (n1 + n2 - 2))

    Args:
        group1: Values for baseline/reference group.
        group2: Values for experimental/target group.

    Returns:
        Cohen's d effect size. Positive indicates higher values in group2.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1, m2 = np.mean(g1), np.mean(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

    pooled_variance = ((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1)
    if pooled_variance <= 1e-12:
        return 0.0

    s_pooled = np.sqrt(pooled_variance)
    return float((m2 - m1) / s_pooled)


def fdr_correction(
    p_values: Union[np.ndarray, list],
    method: Literal["bh", "bonferroni"] = "bh",
) -> np.ndarray:
    """Perform multiple hypothesis testing correction.

    Methods:
        - 'bh': Benjamini-Hochberg False Discovery Rate (FDR / q-value).
                Guarantees monotonicity: q_(i) = min_{j >= i} (K * p_(j) / j).
        - 'bonferroni': Family-Wise Error Rate (FWER) control:
                q = min(1.0, p * K).

    Args:
        p_values: 1D array of raw p-values from statistical tests.
        method: Correction algorithm ('bh' or 'bonferroni').

    Returns:
        1D array of adjusted q-values aligned to original input indices.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.clip(p, 0.0, 1.0)

    # Handle NaNs: preserve NaN locations, adjust valid p-values
    valid_mask = ~np.isnan(p)
    if not np.any(valid_mask):
        return p.copy()

    valid_p = p[valid_mask]
    m = len(valid_p)

    if method == "bonferroni":
        adj_valid = np.clip(valid_p * m, 0.0, 1.0)
    elif method == "bh":
        order = np.argsort(valid_p)
        sorted_p = valid_p[order]

        # Calculate BH adjusted values: p * (m / rank)
        ranks = np.arange(1, m + 1)
        bh_vals = sorted_p * (m / ranks)

        # Monotonize from right to left (step-up procedure)
        monotonized = np.minimum.accumulate(bh_vals[::-1])[::-1]
        monotonized = np.clip(monotonized, 0.0, 1.0)

        # Invert order back to original valid_p indexing
        adj_valid = np.empty_like(monotonized)
        adj_valid[order] = monotonized
    else:
        raise ValueError(f"Unknown FDR correction method: '{method}'. Expected 'bh' or 'bonferroni'.")

    adjusted = np.full_like(p, fill_value=np.nan)
    adjusted[valid_mask] = adj_valid
    return adjusted
