"""Unit and integration tests for Cohort Population Discovery & Statistical Analysis Orchestrator."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from flowx.clustering.cohort_runner import (
    CohortPopulationResult,
    run_cohort_population_analysis,
)


def test_run_cohort_population_analysis(tmp_path: Path):
    """Verify cohort clustering, UMAP embedding, feature extraction, and DA testing."""
    rng = np.random.default_rng(42)

    # 4 samples (2 Control, 2 Treated)
    # Control has higher Pop A; Treated has higher Pop B
    def make_sample(shift_b: float):
        # 150 cells in cluster 1 (CD3 high, CD4 high)
        c1 = np.column_stack([rng.normal(12, 1, 150), rng.normal(8, 1, 150)])
        # 150 cells in cluster 2 (CD3 low, CD4 high + shift)
        c2 = np.column_stack([rng.normal(2, 0.8, 150), rng.normal(8 + shift_b, 1, 150)])
        data = np.vstack([c1, c2])
        return pd.DataFrame(data, columns=["CD3", "CD4"])

    cohort_dfs = {
        "C1": make_sample(shift_b=0.0),
        "C2": make_sample(shift_b=0.0),
        "T1": make_sample(shift_b=5.0),
        "T2": make_sample(shift_b=5.0),
    }

    meta = pd.DataFrame({
        "condition": ["Control", "Control", "Treated", "Treated"]
    }, index=["C1", "C2", "T1", "T2"])

    out_dir = tmp_path / "cohort_pop_out"
    res = run_cohort_population_analysis(
        cohort_dfs=cohort_dfs,
        sample_metadata=meta,
        markers=["CD3", "CD4"],
        clustering_method="som",
        n_clusters=4,
        compute_umap=True,
        umap_events_per_sample=100,
        condition_column="condition",
        reference_group="Control",
        target_group="Treated",
        output_dir=out_dir,
        generate_plots=True,
    )

    assert isinstance(res, CohortPopulationResult)
    assert len(res.samples) == 4
    assert len(res.labeled_dfs) == 4

    for s_id, df in res.labeled_dfs.items():
        assert "cluster_id" in df.columns
        assert "lineage_id" in df.columns
        assert len(df) == 300

    assert res.clustering_result.cluster_count >= 2
    n_pops = len(res.feature_set.population_names)
    assert res.feature_set.count_matrix.shape == (4, n_pops)
    assert res.feature_set.viable_freq_matrix.shape == (4, n_pops)

    # UMAP checks
    assert res.umap_embedding is not None
    assert len(res.umap_embedding) <= 400

    # Verify tables and structured Step 5 cluster/population plot directories
    assert (out_dir / "05_population_discovery/tables/population_counts.csv").is_file()
    assert (out_dir / "05_population_discovery/tables/population_frequencies_viable.csv").is_file()
    assert (out_dir / "05_population_discovery/tables/cluster_phenotypes_summary.csv").is_file()
    assert (out_dir / "05_population_discovery/tables/cluster_to_population_map.csv").is_file()
    assert (out_dir / "05_population_discovery/plots/clusters/cluster_expression_heatmap.png").is_file()
    assert (out_dir / "05_population_discovery/plots/clusters/cluster_frequency_clustermap.png").is_file()
    assert (out_dir / "05_population_discovery/plots/clusters/cluster_qc_metrics.png").is_file()
    assert (out_dir / "05_population_discovery/plots/clusters/umap_clusters.png").is_file()
    assert (out_dir / "05_population_discovery/plots/populations/population_expression_heatmap.png").is_file()
    assert (out_dir / "05_population_discovery/plots/populations/population_frequency_clustermap.png").is_file()
    assert (out_dir / "05_population_discovery/plots/populations/umap_populations.png").is_file()
    assert (out_dir / "05_population_discovery/plots/cluster_to_population_bridge.png").is_file()

    # Verify GraphPad Prism exports
    prism_freq = out_dir / "05_population_discovery/tables/prism/prism_population_frequencies_viable.csv"
    prism_counts = out_dir / "05_population_discovery/tables/prism/prism_population_counts.csv"
    assert prism_freq.is_file()
    assert prism_counts.is_file()
    prism_text = prism_freq.read_text()
    assert "Control" in prism_text
    assert "Treated" in prism_text

    # Verify mandatory deliverable exports
    assert (out_dir / "exports/workspaces/flowjo_workspace.wsp").is_file()
    assert (out_dir / "exports/workspaces/gating_ml_2.0.xml").is_file()
    assert (out_dir / "exports/reports/miflowcyt_analysis_report.md").is_file()
    assert (out_dir / "exports/reports/miflowcyt_analysis_report.html").is_file()
    assert (out_dir / "exports/augmented_fcs/augmented_C1.fcs").is_file()

    # Verify new Phase 5 & 6 diagnostic plots
    assert (out_dir / "05_population_discovery/plots/populations/umap_stratified_contours.png").is_file()
    assert (out_dir / "06_statistical_analysis/plots/differential_abundance_forest.png").is_file()
    assert (out_dir / "06_statistical_analysis/plots/differential_state_heatmap.png").is_file()

    # Test cluster renaming post-discovery
    name_map = {0: "CD4+ T Cells", 1: "CD8+ T Cells"}
    res.rename_clusters(name_map, update_plots=True)
    assert "CD4+ T Cells" in res.feature_set.population_names
    assert "CD4+ T Cells" in res.feature_set.viable_freq_matrix.columns
    for df in res.labeled_dfs.values():
        assert "cluster_name" in df.columns
    # Check updated Prism export contains new name
    updated_prism = prism_freq.read_text()
    assert "CD4+ T Cells" in updated_prism


def test_auto_k_and_optimization_metrics(tmp_path: Path):
    """Verify auto_k optimal metacluster evaluation via Davies-Bouldin index."""
    rng = np.random.default_rng(999)
    c1 = np.column_stack([rng.normal(15, 0.5, 80), rng.normal(15, 0.5, 80)])
    c2 = np.column_stack([rng.normal(5, 0.5, 80), rng.normal(5, 0.5, 80)])
    df = pd.DataFrame(np.vstack([c1, c2]), columns=["M1", "M2"])
    cohort_dfs = {"S1": df, "S2": df.copy()}

    out_dir = tmp_path / "auto_k_test"
    res = run_cohort_population_analysis(
        cohort_dfs=cohort_dfs,
        markers=["M1", "M2"],
        auto_k=True,
        k_min=3,
        k_max=6,
        output_dir=out_dir,
        generate_plots=False,
    )
    assert res.clustering_result.cluster_count >= 3
    assert hasattr(res.clustering_result, "k_optimization_scores")
    k_metrics_csv = out_dir / "05_population_discovery/tables/metacluster_k_metrics.csv"
    assert k_metrics_csv.is_file()
    metrics_df = pd.read_csv(k_metrics_csv)
    assert any(col in metrics_df.columns for col in ["consensus_pac", "davies_bouldin_index"])
    assert len(metrics_df) >= 2



def test_cluster_name_map_upfront(tmp_path: Path):
    """Verify passing cluster_name_map directly into run_cohort_population_analysis."""
    rng = np.random.default_rng(123)
    c1 = np.column_stack([rng.normal(10, 1, 100), rng.normal(5, 1, 100)])
    c2 = np.column_stack([rng.normal(2, 1, 100), rng.normal(8, 1, 100)])
    df = pd.DataFrame(np.vstack([c1, c2]), columns=["CD3", "CD4"])
    cohort_dfs = {"S1": df, "S2": df.copy()}
    meta = pd.DataFrame({"condition": ["A", "B"]}, index=["S1", "S2"])

    out_dir = tmp_path / "upfront_rename"
    name_map = {0: "Lineage_Alpha", 1: "Lineage_Beta"}
    res = run_cohort_population_analysis(
        cohort_dfs=cohort_dfs,
        sample_metadata=meta,
        markers=["CD3", "CD4"],
        n_clusters=2,
        cluster_name_map=name_map,
        output_dir=out_dir,
        generate_plots=False,
    )
    assert "Lineage_Alpha" in res.feature_set.population_names
    assert "Lineage_Beta" in res.feature_set.population_names
    prism_file = out_dir / "05_population_discovery/tables/prism/prism_population_frequencies_viable.csv"
    assert prism_file.is_file()
    assert "Lineage_Alpha" in prism_file.read_text()


def test_many_to_one_cluster_aggregation(tmp_path: Path):
    """Verify many-to-one cluster mapping aggregates event counts and frequencies."""
    rng = np.random.default_rng(42)
    c1 = np.column_stack([rng.normal(10, 1, 100), rng.normal(5, 1, 100)])
    c2 = np.column_stack([rng.normal(8, 1, 100), rng.normal(5, 1, 100)])
    c3 = np.column_stack([rng.normal(2, 1, 100), rng.normal(12, 1, 100)])
    df = pd.DataFrame(np.vstack([c1, c2, c3]), columns=["CD3", "CD4"])
    cohort_dfs = {"S1": df.copy(), "S2": df.copy()}
    meta = pd.DataFrame({"condition": ["Ctrl", "Treat"]}, index=["S1", "S2"])

    out_dir = tmp_path / "aggregation_test"
    res = run_cohort_population_analysis(
        cohort_dfs=cohort_dfs,
        sample_metadata=meta,
        markers=["CD3", "CD4"],
        clustering_method="som",
        n_clusters=3,
        compute_umap=False,
        output_dir=out_dir,
        generate_plots=False,
    )

    # Record initial counts
    orig_counts = res.feature_set.count_matrix.copy()
    orig_freqs = res.feature_set.viable_freq_matrix.copy()
    col0 = orig_counts.columns[0]
    col1 = orig_counts.columns[1]
    expected_sum_count = orig_counts[col0] + orig_counts[col1]
    expected_sum_freq = orig_freqs[col0] + orig_freqs[col1]

    # Map both col0 and col1 to 'CD4+ T Cells'
    many_to_one_map = {col0: "CD4+ T Cells", col1: "CD4+ T Cells"}
    res.rename_clusters(many_to_one_map, update_plots=False)

    # Verify column deduplication
    assert "CD4+ T Cells" in res.feature_set.population_names
    assert len(res.feature_set.population_names) == len(set(res.feature_set.population_names))
    assert res.feature_set.count_matrix.columns.is_unique
    assert res.feature_set.viable_freq_matrix.columns.is_unique

    # Verify mathematical aggregation
    np.testing.assert_allclose(res.feature_set.count_matrix["CD4+ T Cells"], expected_sum_count)
    np.testing.assert_allclose(res.feature_set.viable_freq_matrix["CD4+ T Cells"], expected_sum_freq)

    # Verify cell-level labeling
    df_s1 = res.labeled_dfs["S1"]
    mask_c01 = df_s1["cluster_id"].isin([int(str(col0).replace("Cluster_", "")), int(str(col1).replace("Cluster_", ""))])
    assert (df_s1.loc[mask_c01, "lineage_id"] == "CD4+ T Cells").all()


def test_load_sample_metadata_and_cohort_integration(tmp_path: Path):
    """Verify load_sample_metadata cleans .fcs extensions and integrates into cohort analysis."""
    from flowx.io.metadata import load_sample_metadata

    meta_csv = tmp_path / "cohort_metadata.csv"
    meta_csv.write_text(
        "sample_id,condition,batch\n"
        "Sample_A.fcs,Healthy,Batch1\n"
        "Sample_B.fcs,Disease,Batch1\n"
    )

    meta_df = load_sample_metadata(meta_csv)
    assert list(meta_df.index) == ["Sample_A", "Sample_B"]
    assert meta_df.index.name == "sample_id"
    assert "condition" in meta_df.columns

    # Test integration into run_cohort_population_analysis via metadata_file argument
    rng = np.random.default_rng(77)
    df_a = pd.DataFrame(rng.normal(10, 1, (80, 2)), columns=["CD3", "CD4"])
    df_b = pd.DataFrame(rng.normal(5, 1, (80, 2)), columns=["CD3", "CD4"])
    cohort_dfs = {"Sample_A": df_a, "Sample_B": df_b}

    out_dir = tmp_path / "metadata_file_test"
    res = run_cohort_population_analysis(
        cohort_dfs=cohort_dfs,
        metadata_file=meta_csv,
        markers=["CD3", "CD4"],
        n_clusters=2,
        compute_umap=False,
        output_dir=out_dir,
        generate_plots=False,
    )

    assert res.feature_set.sample_metadata is not None
    assert "condition" in res.feature_set.sample_metadata.columns
    assert res.feature_set.sample_metadata.loc["Sample_A", "condition"] == "Healthy"


def test_population_granularity_aggregates_default_signatures_and_records_provenance(tmp_path: Path):
    """Verify stats_granularity='populations' aggregates default signature labels and records markers_csv provenance."""
    import yaml
    from flowx.io.metadata import MarkerMetadataResolution
    from flowx.pipeline import FlowCytometryPipeline

    rng = np.random.default_rng(11)
    # 3 clusters: two CD3+ CD4+ CD8- (should merge into CD4+ T-Cell Lineage) and one CD3- CD19+ (B-Cell Lineage)
    c1 = np.column_stack([rng.normal(10, 0.5, 60), rng.normal(10, 0.5, 60), rng.normal(1, 0.3, 60), rng.normal(1, 0.3, 60)])
    c2 = np.column_stack([rng.normal(8, 0.5, 60), rng.normal(9, 0.5, 60), rng.normal(1, 0.3, 60), rng.normal(1, 0.3, 60)])
    c3 = np.column_stack([rng.normal(1, 0.3, 60), rng.normal(1, 0.3, 60), rng.normal(1, 0.3, 60), rng.normal(10, 0.5, 60)])
    df = pd.DataFrame(np.vstack([c1, c2, c3]), columns=["V450-A", "B710-A", "R780-A", "APC-A"])
    cohort_dfs = {"S1": df.copy(), "S2": df.copy()}
    meta = pd.DataFrame({"condition": ["Ctrl", "Treat"]}, index=["S1", "S2"])

    out_dir = tmp_path / "pop_agg_sig"
    res = run_cohort_population_analysis(
        cohort_dfs=cohort_dfs,
        sample_metadata=meta,
        markers=["V450-A", "B710-A", "R780-A", "APC-A"],
        marker_name_map={"V450-A": "CD3", "B710-A": "CD4", "R780-A": "CD8", "APC-A": "CD19"},
        n_clusters=3,
        auto_k=False,
        compute_umap=False,
        stats_granularity="populations",
        output_dir=out_dir,
        generate_plots=False,
    )
    assert not any(p.startswith("Cluster_") for p in res.feature_set.population_names)
    assert len(res.feature_set.population_names) == 2

    # Verify _write_resolved_config records markers_csv source and path
    mm_res = MarkerMetadataResolution(provenance_source="markers_csv:data/spectral/test/metadata/markers.csv")
    cfg_path = FlowCytometryPipeline._write_resolved_config(run_dir=out_dir, marker_meta_res=mm_res)
    parsed = yaml.safe_load(cfg_path.read_text())
    assert parsed["provenance"]["markers_metadata_source"] == "markers_csv"
    assert parsed["provenance"]["markers_metadata_path"] == "data/spectral/test/metadata/markers.csv"

    # Verify MIFlowCyt report resolves marker names, auto-k method, phenotype counts, and n<2 power warning
    miflow_md = Path(res.generated_tables["miflowcyt_report_md"]).read_text(encoding="utf-8")
    assert "**CD3** (Phenotypic Surface/Intracellular Marker)" in miflow_md
    assert "Fixed Metacluster Count (k=3)" in miflow_md
    assert "### Discovered Population Phenotypes & Abundances" in miflow_md
    assert "120" in miflow_md
    assert "Statistical Power Notice:" in miflow_md
    assert "N/A (n<2)" in miflow_md


def test_step06_ds_auto_includes_functional_markers_and_override(tmp_path: Path):
    """Step 06 DS evaluates lineage+functional by default while keeping clustering on lineage markers."""
    rng = np.random.default_rng(19)
    markers_csv = tmp_path / "markers.csv"
    pd.DataFrame(
        [
            {"channel_name": "V450-A", "marker_name": "CD3", "marker_class": "lineage"},
            {"channel_name": "B710-A", "marker_name": "CD4", "marker_class": "lineage"},
            {"channel_name": "PE-A", "marker_name": "IFNg", "marker_class": "functional"},
        ]
    ).to_csv(markers_csv, index=False)

    def make_sample(ifng_shift: float) -> pd.DataFrame:
        c1 = np.column_stack([
            rng.normal(10, 0.4, 80),
            rng.normal(10, 0.4, 80),
            rng.normal(2 + ifng_shift, 0.4, 80),
        ])
        c2 = np.column_stack([
            rng.normal(1, 0.3, 80),
            rng.normal(1, 0.3, 80),
            rng.normal(1, 0.3, 80),
        ])
        return pd.DataFrame(np.vstack([c1, c2]), columns=["V450-A", "B710-A", "PE-A"])

    cohort_dfs = {
        "C1": make_sample(0.0),
        "C2": make_sample(0.05),
        "C3": make_sample(-0.05),
        "C4": make_sample(0.1),
        "T1": make_sample(6.0),
        "T2": make_sample(6.1),
        "T3": make_sample(6.2),
        "T4": make_sample(6.3),
    }
    meta = pd.DataFrame(
        {"condition": ["Control"] * 4 + ["Treated"] * 4},
        index=["C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"],
    )

    # Default: clustering on lineage only, DS evaluates lineage + functional
    res_default = run_cohort_population_analysis(
        cohort_dfs=cohort_dfs,
        sample_metadata=meta,
        markers="auto",
        markers_file=markers_csv,
        marker_roles=["lineage"],
        n_clusters=2,
        auto_k=False,
        compute_umap=False,
        condition_column="condition",
        reference_group="Control",
        target_group="Treated",
        ds_test_method="t_test",
        output_dir=tmp_path / "ds_default",
        generate_plots=False,
    )
    assert list(res_default.clustering_result.markers_used) == ["V450-A", "B710-A"]
    assert set(res_default.ds_markers or []) == {"V450-A", "B710-A", "PE-A"}
    ds_markers_default = set(res_default.ds_result.summary_df["marker"].unique())
    assert ds_markers_default == {"V450-A", "B710-A", "PE-A"}
    assert "PE-A" in set(m for _, m in res_default.ds_result.significant_states)

    # Override: stats_marker_roles=["functional"] limits DS to PE-A only
    res_func_only = run_cohort_population_analysis(
        cohort_dfs=cohort_dfs,
        sample_metadata=meta,
        markers=["V450-A", "B710-A"],
        markers_file=markers_csv,
        marker_roles=["lineage"],
        stats_marker_roles=["functional"],
        n_clusters=2,
        auto_k=False,
        compute_umap=False,
        condition_column="condition",
        reference_group="Control",
        target_group="Treated",
        output_dir=tmp_path / "ds_func_only",
        generate_plots=False,
    )
    assert list(res_func_only.clustering_result.markers_used) == ["V450-A", "B710-A"]
    assert res_func_only.ds_markers == ["PE-A"]
    assert set(res_func_only.ds_result.summary_df["marker"].unique()) == {"PE-A"}




