"""Unit tests for Phase 5 Quantitative Cohort & Statistical Analysis."""

import numpy as np
import pandas as pd
import pytest

from flowx.stats.differential_abundance import run_differential_abundance
from flowx.stats.differential_state import run_differential_state
from flowx.stats.extraction import CohortFeatureExtractor
from flowx.stats.metrics import (
    cohens_d,
    fdr_correction,
    geometric_mean,
    log2_fold_change,
    median_fluorescence_intensity,
    robust_sd,
)
from flowx.stats.models import (
    CohortFeatureSet,
    DAConfig,
    DAResult,
    DSConfig,
    DSResult,
)


def test_metrics_primitives():
    """Verify statistical metrics (MFI, GeoMFI, rSD, Log2FC, Cohen's d, FDR)."""
    vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert median_fluorescence_intensity(vals) == 30.0
    assert 20.0 < geometric_mean(vals) < 30.0
    assert robust_sd(vals, method="mad") > 0.0
    assert robust_sd(vals, method="iqr") > 0.0

    # Log2FC
    l2fc = log2_fold_change(10.0, 40.0, pseudocount=0.0)
    assert np.isclose(l2fc, 2.0)

    # Cohen's d
    g1 = np.array([10.0, 11.0, 12.0, 10.5])
    g2 = np.array([20.0, 21.0, 22.0, 20.5])
    d = cohens_d(g1, g2)
    assert d > 5.0  # Large positive effect

    # FDR correction
    raw_p = np.array([0.001, 0.01, 0.04, 0.05, 0.50])
    q = fdr_correction(raw_p, method="bh")
    assert len(q) == 5
    assert np.all(q <= 1.0)
    assert np.all(np.diff(q) >= 0)  # Monotonicity check


def test_cohort_feature_extractor():
    """Verify feature extractor builds counts, percentages, and MFI tensors correctly."""
    rng = np.random.default_rng(42)

    # Sample 1: 100 cells, 60 in pop 0, 40 in pop 1
    df1 = pd.DataFrame({
        "cluster_id": [0] * 60 + [1] * 40,
        "CD3": rng.normal(10, 1, 100),
        "CD4": rng.normal(5, 1, 100),
    })

    # Sample 2: 100 cells, 20 in pop 0, 80 in pop 1
    df2 = pd.DataFrame({
        "cluster_id": [0] * 20 + [1] * 80,
        "CD3": rng.normal(12, 1, 100),
        "CD4": rng.normal(8, 1, 100),
    })

    cohort_dfs = {"S1": df1, "S2": df2}
    meta = pd.DataFrame({"condition": ["Control", "Treated"]}, index=["S1", "S2"])

    extractor = CohortFeatureExtractor(population_column="cluster_id")
    features = extractor.extract(cohort_dfs, sample_metadata=meta)

    assert isinstance(features, CohortFeatureSet)
    assert features.sample_ids == ["S1", "S2"]
    assert len(features.population_names) == 2
    assert features.count_matrix.loc["S1", "Cluster_0"] == 60
    assert features.count_matrix.loc["S2", "Cluster_0"] == 20
    assert features.viable_freq_matrix.loc["S1", "Cluster_0"] == 60.0
    assert "CD3" in features.mfi_tensor
    assert "CD4" in features.mfi_tensor


def test_differential_abundance():
    """Verify DA testing identifies expanding and contracting populations."""
    # Synthetic cohort: 3 controls (high in Cluster_0, low in Cluster_1)
    # and 3 treated (low in Cluster_0, high in Cluster_1)
    rng = np.random.default_rng(42)

    count_data = {
        "Cluster_0": [80, 85, 90, 10, 15, 12],
        "Cluster_1": [20, 15, 10, 90, 85, 88],
    }
    sample_ids = ["C1", "C2", "C3", "T1", "T2", "T3"]
    counts = pd.DataFrame(count_data, index=sample_ids)
    viable_freqs = counts.copy()  # 100 events total each

    meta = pd.DataFrame({
        "condition": ["Control", "Control", "Control", "Treated", "Treated", "Treated"]
    }, index=sample_ids)

    features = CohortFeatureSet(
        sample_ids=sample_ids,
        population_names=["Cluster_0", "Cluster_1"],
        marker_names=["CD3"],
        count_matrix=counts,
        parent_freq_matrix=viable_freqs,
        viable_freq_matrix=viable_freqs,
        sample_metadata=meta,
    )

    cfg = DAConfig(fdr_threshold=0.10, log2fc_threshold=1.0)
    res = run_differential_abundance(
        features=features,
        condition_column="condition",
        reference_group="Control",
        target_group="Treated",
        config=cfg,
    )

    assert isinstance(res, DAResult)
    assert len(res.summary_df) == 2
    assert "Cluster_1" in res.significant_populations
    assert "Cluster_0" in res.significant_populations

    c1_row = res.summary_df[res.summary_df["population"] == "Cluster_1"].iloc[0]
    assert c1_row["log2fc"] > 1.5
    assert c1_row["direction"] == "expanded"

    c0_row = res.summary_df[res.summary_df["population"] == "Cluster_0"].iloc[0]
    assert c0_row["log2fc"] < -1.5
    assert c0_row["direction"] == "contracted"


def test_differential_state():
    """Verify DS testing identifies marker upregulation within a population."""
    sample_ids = ["C1", "C2", "C3", "T1", "T2", "T3"]
    meta = pd.DataFrame({
        "condition": ["Control", "Control", "Control", "Treated", "Treated", "Treated"]
    }, index=sample_ids)

    # CD80 upregulated in Cluster_0 in treated group
    mfi_cd80 = pd.DataFrame({
        "Cluster_0": [2.0, 2.2, 2.1, 8.5, 8.9, 9.1],
    }, index=sample_ids)

    counts = pd.DataFrame({"Cluster_0": [100] * 6}, index=sample_ids)
    freqs = pd.DataFrame({"Cluster_0": [100.0] * 6}, index=sample_ids)

    features = CohortFeatureSet(
        sample_ids=sample_ids,
        population_names=["Cluster_0"],
        marker_names=["CD80"],
        count_matrix=counts,
        parent_freq_matrix=freqs,
        viable_freq_matrix=freqs,
        mfi_tensor={"CD80": mfi_cd80},
        sample_metadata=meta,
    )

    cfg = DSConfig(fdr_threshold=0.10)
    res = run_differential_state(
        features=features,
        condition_column="condition",
        reference_group="Control",
        target_group="Treated",
        config=cfg,
    )

    assert isinstance(res, DSResult)
    assert len(res.summary_df) == 1
    assert ("Cluster_0", "CD80") in res.significant_states
    row = res.summary_df.iloc[0]
    assert row["direction"] == "upregulated"
    assert row["delta_stat"] > 5.0
    assert row["cohens_d"] > 3.0


def test_differential_abundance_ttest_and_min_events():
    """Verify t_test method and min_events filtering in DA."""
    sample_ids = ["C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"]
    meta = pd.DataFrame({
        "condition": ["Control"] * 4 + ["Treated"] * 4
    }, index=sample_ids)

    counts = pd.DataFrame({
        "Abundant": [100, 105, 95, 100, 20, 25, 22, 18],
        "Rare": [1, 0, 2, 0, 1, 0, 1, 0],  # Total = 5 < 10 min_events
    }, index=sample_ids)

    freqs = counts / 100.0 * 100.0

    features = CohortFeatureSet(
        sample_ids=sample_ids,
        population_names=["Abundant", "Rare"],
        marker_names=["CD4"],
        count_matrix=counts,
        parent_freq_matrix=freqs,
        viable_freq_matrix=freqs,
        sample_metadata=meta,
    )

    cfg = DAConfig(test_method="t_test", fdr_threshold=0.05, min_events=10)
    res = run_differential_abundance(features, config=cfg)

    assert "Abundant" in res.significant_populations
    assert "Rare" not in res.significant_populations
    rare_row = res.summary_df[res.summary_df["population"] == "Rare"].iloc[0]
    assert np.isnan(rare_row["p_value"])


def test_differential_state_geomfi():
    """Verify differential state analysis with geometric mean MFI."""
    sample_ids = ["C1", "C2", "C3", "T1", "T2", "T3"]
    meta = pd.DataFrame({
        "condition": ["Baseline", "Baseline", "Baseline", "Activated", "Activated", "Activated"]
    }, index=sample_ids)

    geomfi_cd25 = pd.DataFrame({
        "T_cells": [10.0, 12.0, 11.0, 150.0, 160.0, 145.0],
    }, index=sample_ids)

    counts = pd.DataFrame({"T_cells": [200] * 6}, index=sample_ids)
    freqs = pd.DataFrame({"T_cells": [100.0] * 6}, index=sample_ids)

    features = CohortFeatureSet(
        sample_ids=sample_ids,
        population_names=["T_cells"],
        marker_names=["CD25"],
        count_matrix=counts,
        parent_freq_matrix=freqs,
        viable_freq_matrix=freqs,
        geomfi_tensor={"CD25": geomfi_cd25},
        sample_metadata=meta,
    )

    cfg = DSConfig(statistic="geomfi", test_method="t_test", fdr_threshold=0.05)
    res = run_differential_state(
        features,
        reference_group="Baseline",
        target_group="Activated",
        config=cfg,
    )

    assert ("T_cells", "CD25") in res.significant_states
    assert res.summary_df.iloc[0]["direction"] == "upregulated"


def test_differential_abundance_glm_binomial():
    """Verify glm_binomial detects significance even with N=2 small animal cohorts."""
    sample_ids = ["Ctrl_1", "Ctrl_2", "Treat_1", "Treat_2"]
    meta = pd.DataFrame({"condition": ["Ctrl", "Ctrl", "Treat", "Treat"]}, index=sample_ids)

    # 10,000 cells per sample: Ctrl has 1% in PopA (100 cells), Treat has 25% in PopA (2500 cells)
    counts = pd.DataFrame({
        "PopA": [100, 110, 2500, 2480],
        "PopB": [9900, 9890, 7500, 7520],
    }, index=sample_ids)

    freqs = counts.div(counts.sum(axis=1), axis=0) * 100.0

    features = CohortFeatureSet(
        sample_ids=sample_ids,
        population_names=["PopA", "PopB"],
        marker_names=[],
        count_matrix=counts,
        parent_freq_matrix=freqs,
        viable_freq_matrix=freqs,
        sample_metadata=meta,
    )

    cfg = DAConfig(test_method="glm_binomial", fdr_threshold=0.01, log2fc_threshold=0.30)
    res = run_differential_abundance(features, condition_column="condition", reference_group="Ctrl", target_group="Treat", config=cfg)

    assert "PopA" in res.significant_populations
    assert "PopB" in res.significant_populations
    popa_row = res.summary_df[res.summary_df["population"] == "PopA"].iloc[0]
    assert popa_row["p_value"] < 1e-10
    assert popa_row["q_value"] < 1e-10
    assert popa_row["direction"] == "expanded"
    popb_row = res.summary_df[res.summary_df["population"] == "PopB"].iloc[0]
    assert popb_row["direction"] == "contracted"


def test_small_n_guardrails_and_mann_whitney_power_floor():
    """Verify n=1 descriptive NaN p-values and n=3 Mann-Whitney p_min > FDR warning."""
    # Case 1: n=1 per group -> p_value and q_value must be NaN, descriptive log2FC and delta_stat preserved
    sample_ids_n1 = ["Ref_1", "Tgt_1"]
    meta_n1 = pd.DataFrame({"condition": ["Ref", "Tgt"]}, index=sample_ids_n1)
    counts_n1 = pd.DataFrame({"PopA": [200, 800], "PopB": [800, 200]}, index=sample_ids_n1)
    freqs_n1 = counts_n1 / 1000.0 * 100.0
    mfi_n1 = {"CD69": pd.DataFrame({"PopA": [10.0, 45.0], "PopB": [5.0, 6.0]}, index=sample_ids_n1)}

    features_n1 = CohortFeatureSet(
        sample_ids=sample_ids_n1,
        population_names=["PopA", "PopB"],
        marker_names=["CD69"],
        count_matrix=counts_n1,
        parent_freq_matrix=freqs_n1,
        viable_freq_matrix=freqs_n1,
        mfi_tensor=mfi_n1,
        sample_metadata=meta_n1,
    )

    da_n1 = run_differential_abundance(
        features_n1,
        condition_column="condition",
        reference_group="Ref",
        target_group="Tgt",
        config=DAConfig(test_method="mann_whitney", fdr_threshold=0.05),
    )
    assert da_n1.n_ref == 1 and da_n1.n_target == 1
    assert da_n1.underpowered_warning is not None
    assert "requires n >= 2" in da_n1.underpowered_warning
    assert da_n1.summary_df["p_value"].isna().all()
    assert da_n1.summary_df["q_value"].isna().all()
    assert da_n1.summary_df["log2fc"].notna().all()

    ds_n1 = run_differential_state(
        features_n1,
        condition_column="condition",
        reference_group="Ref",
        target_group="Tgt",
        config=DSConfig(test_method="mann_whitney", fdr_threshold=0.05),
    )
    assert ds_n1.n_ref == 1 and ds_n1.n_target == 1
    assert ds_n1.underpowered_warning is not None
    assert ds_n1.summary_df["p_value"].isna().all()
    assert ds_n1.summary_df["q_value"].isna().all()
    assert ds_n1.summary_df["delta_stat"].notna().all()
    popa_ds = ds_n1.summary_df[ds_n1.summary_df["population"] == "PopA"].iloc[0]
    assert np.isclose(popa_ds["delta_stat"], 35.0)

    # Case 2: n=3 vs n=3 Mann-Whitney with fdr_threshold=0.05 -> min_achievable_p = 2 / comb(6, 3) = 0.10 > 0.05
    sample_ids_n3 = ["C1", "C2", "C3", "T1", "T2", "T3"]
    meta_n3 = pd.DataFrame({"condition": ["Control"] * 3 + ["Treated"] * 3}, index=sample_ids_n3)
    counts_n3 = pd.DataFrame({"PopA": [100, 110, 120, 500, 520, 540]}, index=sample_ids_n3)
    freqs_n3 = pd.DataFrame({"PopA": [10.0, 11.0, 12.0, 50.0, 52.0, 54.0]}, index=sample_ids_n3)
    features_n3 = CohortFeatureSet(
        sample_ids=sample_ids_n3,
        population_names=["PopA"],
        marker_names=[],
        count_matrix=counts_n3,
        parent_freq_matrix=freqs_n3,
        viable_freq_matrix=freqs_n3,
        sample_metadata=meta_n3,
    )
    da_n3 = run_differential_abundance(
        features_n3,
        condition_column="condition",
        reference_group="Control",
        target_group="Treated",
        config=DAConfig(test_method="mann_whitney", fdr_threshold=0.05),
    )
    assert da_n3.n_ref == 3 and da_n3.n_target == 3
    assert np.isclose(da_n3.min_achievable_p or 0.0, 0.10)
    assert da_n3.underpowered_warning is not None
    assert "p_min = 0.1000" in da_n3.underpowered_warning



