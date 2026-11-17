"""Statistical and structural concordance metrics for external benchmark validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class SpilloverConcordanceMetrics:
    """Summary metrics comparing an estimated spillover matrix to a ground truth matrix.

    Attributes:
        pearson_r: Pearson correlation coefficient across all matrix entries.
        r_squared: Coefficient of determination (``pearson_r ** 2``).
        off_diag_mae: Mean absolute error (fraction) across off-diagonal entries.
        off_diag_max_abs_error: Maximum absolute error (fraction) across off-diagonal entries.
        condition_number: Spectral condition number ``kappa(S)`` of the estimated matrix.
        diff_pct_dataframe: Percentage-point difference matrix ``(calculated - reference) * 100.0``.
    """

    pearson_r: float
    r_squared: float
    off_diag_mae: float
    off_diag_max_abs_error: float
    condition_number: float
    diff_pct_dataframe: pd.DataFrame


def compute_spillover_concordance(
    calculated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> SpilloverConcordanceMetrics:
    """Quantify agreement between a calculated spillover matrix and a reference matrix.

    Args:
        calculated_df: Square DataFrame estimated by FlowX (rows=donor, cols=detector).
        reference_df: Square DataFrame extracted from an external workspace or instrument.

    Returns:
        :class:`SpilloverConcordanceMetrics` populated with correlation, errors, and
        percentage-point differences.

    Raises:
        ValueError: If fewer than 2 shared channels exist between matrices.
    """
    common = [ch for ch in calculated_df.columns if ch in reference_df.columns and ch in calculated_df.index and ch in reference_df.index]
    if len(common) < 2:
        raise ValueError(f"Need >= 2 overlapping detector channels to compute concordance; found {len(common)}.")

    calc_aligned = calculated_df.loc[common, common].astype(np.float64)
    ref_aligned = reference_df.loc[common, common].astype(np.float64)

    calc_arr = calc_aligned.to_numpy()
    ref_arr = ref_aligned.to_numpy()

    corr_val, _ = pearsonr(calc_arr.flatten(), ref_arr.flatten())
    corr_f = float(corr_val) if np.isfinite(corr_val) else 0.0

    off_diag = ~np.eye(len(common), dtype=bool)
    off_diffs = np.abs((calc_arr - ref_arr)[off_diag])
    off_mae = float(np.mean(off_diffs)) if off_diffs.size > 0 else 0.0
    off_max = float(np.max(off_diffs)) if off_diffs.size > 0 else 0.0
    cond_num = float(np.linalg.cond(calc_arr))

    return SpilloverConcordanceMetrics(
        pearson_r=corr_f,
        r_squared=corr_f ** 2,
        off_diag_mae=off_mae,
        off_diag_max_abs_error=off_max,
        condition_number=cond_num,
        diff_pct_dataframe=(calc_aligned - ref_aligned) * 100.0,
    )


def compute_replicate_concordance(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> Dict[str, float]:
    """Compute correlation and error agreement between two paired frequency/metric vectors.

    Args:
        values_a: Values from replicate or method A.
        values_b: Paired values from replicate or method B.

    Returns:
        Dictionary with keys ``pearson_r``, ``r_squared``, ``mae``, and ``max_abs_delta``.

    Raises:
        ValueError: If lengths mismatch or fewer than 2 values are provided.
    """
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch in replicate comparison: {arr_a.shape} vs {arr_b.shape}")
    if arr_a.size < 2:
        raise ValueError("Need at least 2 paired entries to compute replicate concordance.")

    if np.std(arr_a) < 1e-12 or np.std(arr_b) < 1e-12:
        corr_f = 1.0 if np.allclose(arr_a, arr_b) else 0.0
    else:
        corr_val, _ = pearsonr(arr_a, arr_b)
        corr_f = float(corr_val) if np.isfinite(corr_val) else 0.0

    abs_delta = np.abs(arr_a - arr_b)
    return {
        "pearson_r": corr_f,
        "r_squared": corr_f ** 2,
        "mae": float(np.mean(abs_delta)),
        "max_abs_delta": float(np.max(abs_delta)),
    }


def evaluate_cluster_manual_alignment(
    manual_labels: Sequence[Any],
    cluster_labels: Sequence[Any],
) -> Dict[str, Any]:
    """Quantify concordance between manual ground-truth gating labels and unsupervised clusters.

    Args:
        manual_labels: Event-level labels derived from manual workspace gating.
        cluster_labels: Corresponding event-level cluster assignments from FlowX.

    Returns:
        Dictionary containing ``ari`` (Adjusted Rand Index), ``nmi`` (Normalized
        Mutual Information), ``purity`` (weighted dominant-label purity), and
        ``contingency_df`` (row-normalized cross-tabulation DataFrame).

    Raises:
        ValueError: If arrays are empty or differ in length.
    """
    man_arr = np.asarray(manual_labels)
    cl_arr = np.asarray(cluster_labels)
    if man_arr.shape != cl_arr.shape or man_arr.size == 0:
        raise ValueError(f"Invalid label shapes for alignment: {man_arr.shape} vs {cl_arr.shape}")

    ari = float(adjusted_rand_score(man_arr, cl_arr))
    nmi = float(normalized_mutual_info_score(man_arr, cl_arr))

    raw_ct = pd.crosstab(pd.Series(cl_arr, name="Cluster"), pd.Series(man_arr, name="Manual"))
    purity = float(raw_ct.max(axis=1).sum() / len(cl_arr))
    norm_ct = raw_ct.div(raw_ct.sum(axis=1), axis=0) * 100.0

    return {
        "ari": ari,
        "nmi": nmi,
        "purity": purity,
        "contingency_df": norm_ct,
    }


def compute_cohort_cleanup_concordance(
    records_df: pd.DataFrame,
    flowx_col: str = "flowx_final_pct",
    ref_col: str = "ref_final_pct",
) -> Dict[str, Any]:
    """Summarize clean-up gating yield agreement across 1 or more samples in a cohort.

    Args:
        records_df: Per-sample summary DataFrame containing FlowX and reference retention percentages.
        flowx_col: Column holding FlowX retention percentages.
        ref_col: Column holding reference workspace retention percentages.

    Returns:
        Dictionary with ``n_samples``, ``pearson_r`` (None if ``n_samples < 2``),
        ``r_squared`` (None if ``n_samples < 2``), ``mae``, ``max_abs_delta``, and ``mean_bias_pct``.

    Raises:
        ValueError: If ``records_df`` is empty or required columns are absent.
    """
    if records_df.empty:
        raise ValueError("Cannot compute cohort clean-up concordance on an empty DataFrame.")
    if flowx_col not in records_df.columns or ref_col not in records_df.columns:
        raise KeyError(f"Expected columns '{flowx_col}' and '{ref_col}' in records_df.")

    valid = records_df[[flowx_col, ref_col]].dropna()
    n_samples = len(valid)
    if n_samples == 0:
        raise ValueError("No non-NaN paired sample retention entries found.")

    fx = valid[flowx_col].to_numpy(dtype=np.float64)
    rf = valid[ref_col].to_numpy(dtype=np.float64)
    delta = fx - rf
    abs_delta = np.abs(delta)

    pearson_val: float | None = None
    r2_val: float | None = None
    if n_samples >= 2:
        if np.std(fx) < 1e-12 or np.std(rf) < 1e-12:
            pearson_val = 1.0 if np.allclose(fx, rf) else 0.0
        else:
            r_raw, _ = pearsonr(fx, rf)
            pearson_val = float(r_raw) if np.isfinite(r_raw) else 0.0
        r2_val = pearson_val ** 2

    return {
        "n_samples": int(n_samples),
        "pearson_r": pearson_val,
        "r_squared": r2_val,
        "mae": float(np.mean(abs_delta)),
        "max_abs_delta": float(np.max(abs_delta)),
        "mean_bias_pct": float(np.mean(delta)),
    }


def compute_cohort_frequency_concordance(
    flowx_freq_df: pd.DataFrame,
    ref_freq_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare per-sample lineage/population frequencies across matching samples and columns.

    Args:
        flowx_freq_df: Sample-by-population percentage table from FlowX.
        ref_freq_df: Corresponding sample-by-population percentage table from Ground Truth.

    Returns:
        Dictionary with ``n_samples``, ``n_populations``, ``pearson_r``, ``mae``,
        ``max_abs_delta``, and ``paired_df`` (long-format per-sample concordance table).

    Raises:
        ValueError: If fewer than 1 sample and 1 population overlap.
    """
    shared_samples = [s for s in flowx_freq_df.index if s in ref_freq_df.index]
    shared_pops = [c for c in flowx_freq_df.columns if c in ref_freq_df.columns]
    if not shared_samples or not shared_pops:
        raise ValueError(
            f"Need >= 1 shared sample and population column (samples={len(shared_samples)}, pops={len(shared_pops)})."
        )

    rows = []
    for sample_id in shared_samples:
        for pop in shared_pops:
            fx_val = float(flowx_freq_df.loc[sample_id, pop])
            rf_val = float(ref_freq_df.loc[sample_id, pop])
            rows.append({
                "sample_id": sample_id,
                "population": pop,
                "flowx_pct": fx_val,
                "ref_pct": rf_val,
                "delta_pct": fx_val - rf_val,
            })

    paired_df = pd.DataFrame(rows)
    fx_arr = paired_df["flowx_pct"].to_numpy(dtype=np.float64)
    rf_arr = paired_df["ref_pct"].to_numpy(dtype=np.float64)

    corr_f: float | None = None
    if len(paired_df) >= 2:
        if np.std(fx_arr) < 1e-12 or np.std(rf_arr) < 1e-12:
            corr_f = 1.0 if np.allclose(fx_arr, rf_arr) else 0.0
        else:
            r_raw, _ = pearsonr(fx_arr, rf_arr)
            corr_f = float(r_raw) if np.isfinite(r_raw) else 0.0

    abs_delta = np.abs(fx_arr - rf_arr)
    return {
        "n_samples": len(shared_samples),
        "n_populations": len(shared_pops),
        "pearson_r": corr_f,
        "mae": float(np.mean(abs_delta)),
        "max_abs_delta": float(np.max(abs_delta)),
        "paired_df": paired_df,
    }

