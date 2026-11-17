"""Differential Abundance (DA) Hypothesis Testing Engine for Flow Cytometry.

Identifies statistically significant immunological population expansions and contractions
between experimental conditions (e.g. Healthy vs Disease, Treated vs Untreated)
using non-parametric rank-sum tests and false discovery rate (FDR) adjustments.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, ttest_ind, wilcoxon

from flowx.io.logger import get_logger
from flowx.stats.metrics import fdr_correction, log2_fold_change
from flowx.stats.models import CohortFeatureSet, DAConfig, DAResult


def run_differential_abundance(
    features: CohortFeatureSet,
    condition_column: str = "condition",
    reference_group: Optional[str] = None,
    target_group: Optional[str] = None,
    config: Optional[DAConfig] = None,
) -> DAResult:
    """Execute Differential Abundance hypothesis testing on population frequencies across cohorts.

    Args:
        features: CohortFeatureSet containing frequencies and sample metadata.
        condition_column: Column name in sample_metadata defining experimental groups.
        reference_group: Identifier for the baseline/control condition.
        target_group: Identifier for the experimental/treated condition.
        config: DAConfig options (method, thresholds, pseudocount).

    Returns:
        DAResult containing per-population p-values, q-values, log2FC, and significance tags.
    """
    logger = get_logger()
    cfg = config or DAConfig()

    if features.sample_metadata is None or condition_column not in features.sample_metadata.columns:
        raise ValueError(
            f"Sample metadata must contain condition column '{condition_column}' for DA testing."
        )

    meta = features.sample_metadata
    freq_matrix = (
        features.parent_freq_matrix
        if cfg.frequency_type == "parent"
        else features.viable_freq_matrix
    )
    count_matrix = features.count_matrix

    # Resolve reference and target groups
    unique_conditions = sorted(meta[condition_column].dropna().unique())
    if len(unique_conditions) < 2:
        raise ValueError(
            f"Differential abundance testing requires >= 2 condition groups, found: {unique_conditions}"
        )

    if reference_group is None or target_group is None:
        if len(unique_conditions) == 2 and reference_group is None and target_group is None:
            get_logger().warning(
                "Neither 'reference_group' nor 'target_group' was explicitly declared. "
                "Defaulting contrast to '%s' (reference) vs '%s' (target) based on alphabetical sorting. "
                "Specify 'reference_group' and 'target_group' in your configuration to avoid unintended directionality.",
                unique_conditions[0],
                unique_conditions[1],
            )
            ref = str(unique_conditions[0])
            target = str(unique_conditions[1])
        else:
            raise ValueError(
                f"Explicit 'reference_group' and 'target_group' must be declared for differential abundance testing "
                f"across conditions: {unique_conditions}. Please declare 'reference_group' (e.g. baseline/control) "
                f"and 'target_group' (e.g. treated) in your configuration."
            )
    else:
        ref = reference_group
        target = target_group

    if ref not in unique_conditions:
        raise ValueError(f"Reference group '{ref}' not found in conditions: {unique_conditions}")
    if target not in unique_conditions:
        raise ValueError(f"Target group '{target}' not found in conditions: {unique_conditions}")

    samples_ref = meta[meta[condition_column] == ref].index.intersection(freq_matrix.index).tolist()
    samples_target = meta[meta[condition_column] == target].index.intersection(freq_matrix.index).tolist()

    if not samples_ref:
        raise ValueError(f"No samples matched reference group '{ref}'.")
    if not samples_target:
        raise ValueError(f"No samples matched target group '{target}'.")

    n_ref = len(samples_ref)
    n_target = len(samples_target)

    logger.info(
        f"Differential Abundance: comparing '{target}' (n={n_target}) vs "
        f"'{ref}' (n={n_ref}) using {cfg.test_method.upper()} on {cfg.frequency_type}."
    )

    min_achievable_p: Optional[float] = None
    underpowered_warning: Optional[str] = None
    if cfg.test_method != "glm_binomial" and (n_ref < 2 or n_target < 2):
        underpowered_warning = (
            f"Underpowered cohort contrast ('{target}' n={n_target} vs '{ref}' n={n_ref}): "
            f"sample-level {cfg.test_method} testing requires n >= 2 biological replicates per group. "
            "Inferential p-values and q-values are set to NaN; effect sizes (delta_pct, log2fc) are descriptive only."
        )
        logger.warning(underpowered_warning)
    elif cfg.test_method == "mann_whitney" and n_ref >= 2 and n_target >= 2:
        min_achievable_p = float(2.0 / math.comb(n_ref + n_target, n_ref))
        if min_achievable_p > cfg.fdr_threshold:
            underpowered_warning = (
                f"Underpowered Mann-Whitney U contrast ('{target}' n={n_target} vs '{ref}' n={n_ref}): "
                f"minimum mathematically achievable two-sided p-value is p_min = {min_achievable_p:.4f} "
                f"(> fdr_threshold = {cfg.fdr_threshold}). No population can reach FDR significance "
                "without larger cohort sizes or parametric/count-based testing (e.g., t_test or glm_binomial)."
            )
            logger.warning(underpowered_warning)

    rows: List[dict] = []
    raw_p_values: List[float] = []

    for pop in features.population_names:
        # Check total count across evaluated samples
        total_events = int(count_matrix.loc[samples_ref + samples_target, pop].sum())
        if total_events < cfg.min_events:
            logger.debug(f"Population '{pop}' skipped: total events ({total_events}) < min_events ({cfg.min_events}).")
            raw_p_values.append(np.nan)
            rows.append({
                "population": pop,
                "mean_ref_pct": np.nan,
                "median_ref_pct": np.nan,
                "mean_target_pct": np.nan,
                "median_target_pct": np.nan,
                "delta_pct": np.nan,
                "log2fc": np.nan,
                "p_value": np.nan,
                "q_value": np.nan,
                "is_significant": False,
                "direction": "unchanged",
                "total_events": total_events,
            })
            continue

        v_ref = freq_matrix.loc[samples_ref, pop].dropna().to_numpy()
        v_target = freq_matrix.loc[samples_target, pop].dropna().to_numpy()

        m_ref = float(np.mean(v_ref)) if len(v_ref) > 0 else 0.0
        med_ref = float(np.median(v_ref)) if len(v_ref) > 0 else 0.0
        m_target = float(np.mean(v_target)) if len(v_target) > 0 else 0.0
        med_target = float(np.median(v_target)) if len(v_target) > 0 else 0.0

        l2fc = log2_fold_change(m_ref, m_target, pseudocount=cfg.pseudocount)
        delta_pct = med_target - med_ref

        # Compute p-value
        p_val: float = 1.0
        if cfg.test_method != "glm_binomial" and (len(v_ref) < 2 or len(v_target) < 2):
            p_val = np.nan
        else:
            try:
                if cfg.test_method == "mann_whitney":
                    # Check for zero variance
                    if np.array_equal(v_ref, v_target) or (np.var(v_ref) == 0 and np.var(v_target) == 0 and v_ref[0] == v_target[0]):
                        p_val = 1.0
                    else:
                        res = mannwhitneyu(v_target, v_ref, alternative="two-sided")
                        p_val = float(res.pvalue)
                elif cfg.test_method in ("t_test", "welch_ttest"):
                    res = ttest_ind(v_target, v_ref, equal_var=False)
                    p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
                elif cfg.test_method == "glm_binomial":
                    # Two-proportion z-test / Binomial GLM using pooled single-cell events
                    k_target = float(count_matrix.loc[samples_target, pop].sum())
                    n_target_ev = float(count_matrix.loc[samples_target].sum().sum())
                    k_ref = float(count_matrix.loc[samples_ref, pop].sum())
                    n_ref_ev = float(count_matrix.loc[samples_ref].sum().sum())

                    if n_target_ev > 0 and n_ref_ev > 0:
                        p1 = k_target / n_target_ev
                        p2 = k_ref / n_ref_ev
                        p_pool = (k_target + k_ref) / (n_target_ev + n_ref_ev)
                        se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_target_ev + 1.0 / n_ref_ev))
                        if se > 0:
                            from scipy.stats import norm
                            z = (p1 - p2) / se
                            p_val = float(2.0 * (1.0 - norm.cdf(abs(z))))
                        else:
                            p_val = 1.0
                    else:
                        p_val = 1.0
                elif cfg.test_method == "wilcoxon_paired":
                    min_len = min(len(v_ref), len(v_target))
                    diff = v_target[:min_len] - v_ref[:min_len]
                    if np.all(diff == 0):
                        p_val = 1.0
                    else:
                        res = wilcoxon(diff, alternative="two-sided")
                        p_val = float(res.pvalue)
                elif cfg.test_method == "kruskal":
                    res = kruskal(v_target, v_ref)
                    p_val = float(res.pvalue)
                else:
                    raise ValueError(f"Unknown DA test_method: '{cfg.test_method}'")
            except Exception as exc:
                logger.debug(f"Test calculation exception for population '{pop}': {exc}")
                p_val = 1.0

        raw_p_values.append(p_val)
        rows.append({
            "population": pop,
            "mean_ref_pct": round(m_ref, 4),
            "median_ref_pct": round(med_ref, 4),
            "mean_target_pct": round(m_target, 4),
            "median_target_pct": round(med_target, 4),
            "delta_pct": round(delta_pct, 4),
            "log2fc": round(l2fc, 4),
            "p_value": p_val,
            "q_value": np.nan,  # updated after FDR correction
            "is_significant": False,
            "direction": "unchanged",
            "total_events": total_events,
        })

    # Apply FDR correction
    q_values = fdr_correction(np.array(raw_p_values), method="bh")
    sig_pops: List[str] = []

    for idx, row in enumerate(rows):
        q = q_values[idx]
        row["q_value"] = round(float(q), 6) if not np.isnan(q) else np.nan
        l2fc = row["log2fc"]

        if not np.isnan(q) and not np.isnan(l2fc):
            is_sig = (q <= cfg.fdr_threshold) and (abs(l2fc) >= cfg.log2fc_threshold)
            row["is_significant"] = is_sig
            if is_sig:
                direction = "expanded" if l2fc > 0 else "contracted"
                row["direction"] = direction
                sig_pops.append(row["population"])

    summary_df = pd.DataFrame(rows)
    # Sort by q_value ascending, then abs(log2fc) descending
    summary_df = summary_df.sort_values(
        by=["q_value", "log2fc"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    comparison_desc = f"{target} vs {ref}"
    logger.info(
        f"DA Complete for '{comparison_desc}': {len(sig_pops)} of {len(features.population_names)} "
        f"populations significant (FDR < {cfg.fdr_threshold}, |log2FC| >= {cfg.log2fc_threshold})."
    )

    return DAResult(
        comparison_name=comparison_desc,
        reference_group=ref,
        target_group=target,
        summary_df=summary_df,
        significant_populations=sig_pops,
        config=cfg,
        n_ref=n_ref,
        n_target=n_target,
        min_achievable_p=min_achievable_p,
        underpowered_warning=underpowered_warning,
    )
