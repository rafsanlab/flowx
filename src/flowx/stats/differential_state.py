"""Differential State (DS) Functional Marker Modulation Engine for Flow Cytometry.

Evaluates activation, signaling, checkpoint, or phenotypic marker expression shifts
within defined cell lineages across experimental conditions using count-weighted non-parametric
and parametric statistical testing with false discovery rate (FDR) control.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon

from flowx.io.logger import get_logger
from flowx.stats.metrics import cohens_d, fdr_correction
from flowx.stats.models import CohortFeatureSet, DSConfig, DSResult


def run_differential_state(
    features: CohortFeatureSet,
    condition_column: str = "condition",
    reference_group: Optional[str] = None,
    target_group: Optional[str] = None,
    config: Optional[DSConfig] = None,
) -> DSResult:
    """Execute Differential State (DS) marker modulation analysis within populations across cohorts.

    Args:
        features: CohortFeatureSet containing expression tensors and sample metadata.
        condition_column: Column name in sample_metadata defining experimental groups.
        reference_group: Identifier for the baseline/control condition.
        target_group: Identifier for the experimental/treated condition.
        config: DSConfig options (method, statistic, thresholds, markers).

    Returns:
        DSResult containing per-(population, marker) p-values, q-values, delta MFI, and effect sizes.
    """
    logger = get_logger()
    cfg = config or DSConfig()

    if features.sample_metadata is None or condition_column not in features.sample_metadata.columns:
        raise ValueError(
            f"Sample metadata must contain condition column '{condition_column}' for DS testing."
        )

    meta = features.sample_metadata
    unique_conditions = sorted(meta[condition_column].dropna().unique())
    if len(unique_conditions) < 2:
        raise ValueError(
            f"Differential state testing requires >= 2 condition groups, found: {unique_conditions}"
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
                f"Explicit 'reference_group' and 'target_group' must be declared for differential state testing "
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

    samples_ref = meta[meta[condition_column] == ref].index.intersection(features.sample_ids).tolist()
    samples_target = meta[meta[condition_column] == target].index.intersection(features.sample_ids).tolist()

    if not samples_ref or not samples_target:
        raise ValueError("Insufficient sample representation for DS reference and target cohorts.")

    # Select target statistic tensor
    tensor = features.mfi_tensor if cfg.statistic == "mfi" else features.geomfi_tensor
    eval_markers = cfg.functional_markers or features.marker_names

    n_ref = len(samples_ref)
    n_target = len(samples_target)

    logger.info(
        f"Differential State: comparing '{target}' (n={n_target}) vs '{ref}' (n={n_ref}) across "
        f"{len(features.population_names)} populations and {len(eval_markers)} functional markers "
        f"using {cfg.test_method.upper()} on {cfg.statistic.upper()}."
    )

    min_achievable_p: Optional[float] = None
    underpowered_warning: Optional[str] = None
    if n_ref < 2 or n_target < 2:
        underpowered_warning = (
            f"Underpowered cohort contrast ('{target}' n={n_target} vs '{ref}' n={n_ref}): "
            f"sample-level {cfg.test_method} testing requires n >= 2 biological replicates per group. "
            "Inferential p-values, q-values, and Cohen's d are set to NaN; delta MFI values are descriptive only."
        )
        logger.warning(underpowered_warning)
    elif cfg.test_method == "mann_whitney" and n_ref >= 2 and n_target >= 2:
        min_achievable_p = float(2.0 / math.comb(n_ref + n_target, n_ref))
        if min_achievable_p > cfg.fdr_threshold:
            underpowered_warning = (
                f"Underpowered Mann-Whitney U contrast ('{target}' n={n_target} vs '{ref}' n={n_ref}): "
                f"minimum mathematically achievable two-sided p-value is p_min = {min_achievable_p:.4f} "
                f"(> fdr_threshold = {cfg.fdr_threshold}). No marker state can reach FDR significance "
                "without larger cohort sizes or parametric testing (e.g., t_test)."
            )
            logger.warning(underpowered_warning)

    rows: List[dict] = []
    raw_p_values: List[float] = []
    tested_pairs: List[Tuple[str, str]] = []

    for pop in features.population_names:
        for m in eval_markers:
            if m not in tensor:
                continue

            stat_df = tensor[m]
            if pop not in stat_df.columns:
                continue

            v_ref = stat_df.loc[samples_ref, pop].dropna().to_numpy()
            v_target = stat_df.loc[samples_target, pop].dropna().to_numpy()

            tested_pairs.append((pop, m))

            if len(v_ref) == 0 or len(v_target) == 0:
                raw_p_values.append(np.nan)
                rows.append({
                    "population": pop,
                    "marker": m,
                    "mean_ref": np.nan,
                    "median_ref": np.nan,
                    "mean_target": np.nan,
                    "median_target": np.nan,
                    "delta_stat": np.nan,
                    "cohens_d": np.nan,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "is_significant": False,
                    "direction": "unchanged",
                })
                continue

            m_ref = float(np.mean(v_ref))
            med_ref = float(np.median(v_ref))
            m_target = float(np.mean(v_target))
            med_target = float(np.median(v_target))
            delta = m_target - m_ref

            if len(v_ref) < 2 or len(v_target) < 2:
                raw_p_values.append(np.nan)
                rows.append({
                    "population": pop,
                    "marker": m,
                    "mean_ref": round(m_ref, 4),
                    "median_ref": round(med_ref, 4),
                    "mean_target": round(m_target, 4),
                    "median_target": round(med_target, 4),
                    "delta_stat": round(delta, 4),
                    "cohens_d": np.nan,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "is_significant": False,
                    "direction": "unchanged",
                })
                continue

            d_val = cohens_d(v_ref, v_target)

            p_val = 1.0
            try:
                if cfg.test_method == "mann_whitney":
                    if np.array_equal(v_ref, v_target) or (np.var(v_ref) == 0 and np.var(v_target) == 0 and v_ref[0] == v_target[0]):
                        p_val = 1.0
                    else:
                        res = mannwhitneyu(v_target, v_ref, alternative="two-sided")
                        p_val = float(res.pvalue)
                elif cfg.test_method == "t_test":
                    res = ttest_ind(v_target, v_ref, equal_var=False)
                    p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
                elif cfg.test_method == "wilcoxon_paired":
                    min_len = min(len(v_ref), len(v_target))
                    diff = v_target[:min_len] - v_ref[:min_len]
                    if np.all(diff == 0):
                        p_val = 1.0
                    else:
                        res = wilcoxon(diff, alternative="two-sided")
                        p_val = float(res.pvalue)
                else:
                    raise ValueError(f"Unknown DS test_method: '{cfg.test_method}'")
            except Exception as exc:
                logger.debug(f"DS test calculation exception for ({pop}, {m}): {exc}")
                p_val = 1.0

            raw_p_values.append(p_val)
            rows.append({
                "population": pop,
                "marker": m,
                "mean_ref": round(m_ref, 4),
                "median_ref": round(med_ref, 4),
                "mean_target": round(m_target, 4),
                "median_target": round(med_target, 4),
                "delta_stat": round(delta, 4),
                "cohens_d": round(d_val, 4),
                "p_value": p_val,
                "q_value": np.nan,
                "is_significant": False,
                "direction": "unchanged",
            })

    # Apply FDR correction across all (population, marker) hypotheses
    q_values = fdr_correction(np.array(raw_p_values), method="bh")
    sig_states: List[Tuple[str, str]] = []

    for idx, row in enumerate(rows):
        q = q_values[idx]
        row["q_value"] = round(float(q), 6) if not np.isnan(q) else np.nan
        d_val = row["cohens_d"]

        if not np.isnan(q):
            is_sig = (q <= cfg.fdr_threshold) and (abs(d_val) >= 0.5 or abs(row["delta_stat"]) > 0.5)
            row["is_significant"] = is_sig
            if is_sig:
                direction = "upregulated" if row["delta_stat"] > 0 else "downregulated"
                row["direction"] = direction
                sig_states.append((row["population"], row["marker"]))

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        by=["q_value", "cohens_d"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    comparison_desc = f"{target} vs {ref}"
    logger.info(
        f"DS Complete for '{comparison_desc}': {len(sig_states)} of {len(tested_pairs)} "
        f"marker states significant (FDR < {cfg.fdr_threshold})."
    )

    return DSResult(
        comparison_name=comparison_desc,
        reference_group=ref,
        target_group=target,
        summary_df=summary_df,
        significant_states=sig_states,
        config=cfg,
        n_ref=n_ref,
        n_target=n_target,
        min_achievable_p=min_achievable_p,
        underpowered_warning=underpowered_warning,
    )
