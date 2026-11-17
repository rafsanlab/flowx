"""Unit tests for unsupervised clustering algorithms (SOM & PhenoGraph)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from flowx.clustering.cluster import (
    ClusteringResult,
    find_optimal_metaclusters,
    run_phenograph,
    run_som,
)
from flowx.pipeline import FlowCytometryPipeline


@pytest.fixture
def synthetic_flow_data() -> pd.DataFrame:
    """Generate synthetic flow data with two distinct clusters."""
    rng = np.random.RandomState(42)
    # Cluster 1: CD3+ CD4+
    c1 = rng.normal(loc=[0.8, 0.7, 0.1, 0.1], scale=0.05, size=(500, 4))
    # Cluster 2: CD3+ CD8+
    c2 = rng.normal(loc=[0.8, 0.1, 0.7, 0.1], scale=0.05, size=(500, 4))
    data = np.vstack([c1, c2])
    cols = ["Ax700-A", "PE-TxRed-A", "PacBlue-A", "AARD-A"]
    return pd.DataFrame(data, columns=cols)


def test_som_execution(synthetic_flow_data: pd.DataFrame) -> None:
    """Test that SOM clusters data into expected metaclusters with parent population stats."""
    markers = ["Ax700-A", "PE-TxRed-A", "PacBlue-A"]
    result = run_som(
        df=synthetic_flow_data,
        markers=markers,
        x_dim=5,
        y_dim=5,
        n_metaclusters=4,
        random_state=42,
        parent_population="viable",
        total_sample_events=2000,
    )

    assert isinstance(result, ClusteringResult)
    assert result.method == "som"
    assert len(result.cluster_labels) == len(synthetic_flow_data)
    assert result.cluster_count == 4
    assert result.parent_population == "viable"
    assert result.parent_count == 1000
    assert result.total_sample_count == 2000
    assert not result.cluster_statistics.empty
    assert "Cluster ID" in result.cluster_statistics.columns
    assert "Count" in result.cluster_statistics.columns
    assert "% Parent" in result.cluster_statistics.columns
    assert "% Total" in result.cluster_statistics.columns
    assert result.cluster_statistics["Count"].sum() == len(synthetic_flow_data)
    assert np.isclose(result.cluster_statistics["% Parent"].sum(), 100.0, atol=0.1)
    assert np.isclose(result.cluster_statistics["% Total"].sum(), 50.0, atol=0.1)


def test_phenograph_execution(synthetic_flow_data: pd.DataFrame) -> None:
    """Test that PhenoGraph builds k-NN graph and partitions communities."""
    markers = ["Ax700-A", "PE-TxRed-A", "PacBlue-A"]
    result = run_phenograph(
        df=synthetic_flow_data,
        markers=markers,
        k=15,
        random_state=42,
    )

    assert isinstance(result, ClusteringResult)
    assert result.method == "phenograph"
    assert len(result.cluster_labels) == len(synthetic_flow_data)
    assert result.cluster_count >= 2
    assert not result.cluster_statistics.empty
    assert result.cluster_statistics["Count"].sum() == len(synthetic_flow_data)


def test_phenograph_missing_import_error(synthetic_flow_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that an informative ImportError is raised when phenograph is not installed."""
    import builtins
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "phenograph":
            raise ImportError("No module named 'phenograph'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    markers = ["Ax700-A", "PE-TxRed-A", "PacBlue-A"]
    with pytest.raises(ImportError, match="PhenoGraph is an optional unsupervised clustering engine"):
        run_phenograph(synthetic_flow_data, markers=markers)


def test_pipeline_clustering_compare_mode(tmp_path: Path, tqc_sample_fcs_path: Path) -> None:
    """Test end-to-end pipeline in compare mode on a tutorial sample."""
    out_run_dir = tmp_path / "clustering_test_run"
    pipeline = FlowCytometryPipeline(
        output_dir=out_run_dir,
        random_seed=42,
        timestamp_dir=False,
        overwrite=True,
        enable_time_qc=False,
        enable_discovery=True,
        enable_umap=False,
        numbered_steps=False,
        gating_mode="compare",
        cluster_parent="viable",
    )
    result = pipeline.run(fcs_filepath=tqc_sample_fcs_path)

    assert result.gating_mode == "compare"
    assert result.cluster_parent == "viable"
    assert result.som_result is not None
    assert result.som_result.parent_population == "Viable"
    assert result.som_result.parent_count > 3000
    assert result.som_result.total_sample_count == 6000
    assert result.som_result.full_cluster_labels is not None
    assert len(result.som_result.full_cluster_labels) == 6000
    assert "AARD-A" not in result.som_result.markers_used
    assert result.phenograph_result is not None
    assert result.phenograph_result.parent_population == "Viable"
    assert "AARD-A" not in result.phenograph_result.markers_used
    assert result.three_way_metrics_df is not None
    assert result.three_way_lineages_df is not None
    assert result.reproducibility_verified is True
    assert (out_run_dir / "tables" / "som_statistics.csv").exists()
    assert (out_run_dir / "tables" / "phenograph_statistics.csv").exists()
    assert (out_run_dir / "tables" / "som_macro_groups.csv").exists()
    assert (out_run_dir / "tables" / "phenograph_macro_groups.csv").exists()
    assert (out_run_dir / "tables" / "comparison_crosstab_som.csv").exists()
    assert (out_run_dir / "tables" / "comparison_three_methods_metrics.csv").exists()
    assert (out_run_dir / "tables" / "comparison_three_methods_lineages.csv").exists()
    assert Path(result.generated_plots["step5_som_clusters_scatter"]).is_file()
    assert Path(result.generated_plots["step5_clustering_comparison_centroids"]).is_file()
    assert Path(result.generated_plots["step5_clustering_comparison_external_legend"]).is_file()


def test_pipeline_clustering_all_events_mode(tmp_path: Path, tqc_sample_fcs_path: Path) -> None:
    """Test clustering when cluster_parent='all' (clusters all un-subsetted events)."""
    pipeline = FlowCytometryPipeline(
        output_dir=tmp_path / "clustering_all_run",
        random_seed=42,
        timestamp_dir=False,
        overwrite=True,
        enable_time_qc=False,
        enable_discovery=True,
        enable_umap=False,
        gating_mode="som",
        cluster_parent="all",
    )
    result = pipeline.run(fcs_filepath=tqc_sample_fcs_path)

    assert result.gating_mode == "som"
    assert result.cluster_parent == "all"
    assert result.som_result is not None
    assert result.som_result.parent_population == "All Events"
    assert result.som_result.parent_count == 6000
    assert len(result.som_result.cluster_labels) == 6000
    assert "AARD-A" in result.som_result.markers_used


def test_find_optimal_metaclusters_all_algorithms() -> None:
    """Verify that all 5 auto_k evaluation algorithms execute and return valid optimal k."""
    rng = np.random.default_rng(42)
    # 4 distinct synthetic clusters in 6D
    centers = [
        [10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 10.0, 10.0],
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    ]
    data = []
    for c in centers:
        data.append(rng.normal(loc=c, scale=0.5, size=(25, 6)))
    weights = np.vstack(data)

    methods = ["consensus", "davies_bouldin", "silhouette", "calinski_harabasz", "elbow"]
    for m in methods:
        best_k, scores, metric_name = find_optimal_metaclusters(
            weights=weights,
            k_min=3,
            k_max=8,
            method=m,
            consensus_subsample=0.8,
            consensus_iterations=20,
            random_state=42,
        )
        assert 3 <= best_k <= 8
        assert len(scores) >= 2
        assert metric_name != ""
        assert all(isinstance(v, float) for v in scores.values())


def test_transparent_cluster_annotation_signature_and_no_hallucination(tmp_path: Path) -> None:
    """Verify transparent marker-driven annotation and zero biological hallucination when unmapped."""
    import pandas as pd
    from flowx.clustering.cluster import annotate_clusters_transparently
    from flowx.io.metadata import resolve_marker_metadata

    stats_df = pd.DataFrame([
        {"Cluster ID": 0, "Count": 500, "% Parent": 50.0, "% Total": 50.0, "V450-A Median": 0.72, "B710-A Median": 0.78, "R780-A Median": 0.05},
        {"Cluster ID": 1, "Count": 300, "% Parent": 30.0, "% Total": 30.0, "V450-A Median": 0.70, "B710-A Median": 0.08, "R780-A Median": 0.85},
        {"Cluster ID": 2, "Count": 200, "% Parent": 20.0, "% Total": 20.0, "V450-A Median": 0.08, "B710-A Median": 0.10, "R780-A Median": 0.05},
    ])
    markers = ["V450-A", "B710-A", "R780-A"]

    # Case 1: Neither markers.csv nor $PnS biological names supplied -> MUST NOT guess T-cell names
    res_unmapped = resolve_marker_metadata(
        dataset_dir=tmp_path,
        fcs_pnn=markers,
        fcs_pns=["", "", ""],
    )
    assert res_unmapped.has_verified_biological_markers is False
    assert len(res_unmapped.warnings) >= 1

    pheno_unmapped, _, audit_unmapped, _ = annotate_clusters_transparently(
        cluster_stats_df=stats_df,
        markers=markers,
        marker_name_map=None,
    )
    assert all("empirical_detectors_unmapped" == src for src in audit_unmapped["Annotation Source"])
    assert "CD4" not in pheno_unmapped[0]
    assert "Lymphocyte" not in pheno_unmapped[2]

    # Case 2: Curated markers.csv provided -> resolves canonical biological rules cleanly
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    markers_csv = meta_dir / "markers.csv"
    markers_csv.write_text(
        "channel_name,marker_name,fluorophore,clone_or_catalog,marker_class,cell_type_association,provenance_source\n"
        "V450-A,CD3,V450,NA,lineage,Pan-T,test\n"
        "B710-A,CD4,PerCP-Cy5.5,NA,lineage;functional,T-Helper,test\n"
        "R780-A,CD8,APC-H7,NA,lineage,Cytotoxic T,test\n",
        encoding="utf-8",
    )
    res_mapped = resolve_marker_metadata(dataset_dir=tmp_path, fcs_pnn=markers)
    assert res_mapped.has_verified_biological_markers is True
    assert res_mapped.marker_classes["B710-A"] == ["lineage", "functional"]

    pheno_mapped, macro_mapped, audit_mapped, _ = annotate_clusters_transparently(
        cluster_stats_df=stats_df,
        markers=markers,
        marker_name_map=res_mapped.detector_to_marker,
    )
    assert pheno_mapped[0] == "CD4+ T-Cell Lineage"
    assert pheno_mapped[1] == "CD8+ T-Cell Lineage"
    assert pheno_mapped[2] == "CD3- Non-T Lymphocytes"
    assert "signature_rule:" in audit_mapped.loc[0, "Annotation Source"]

    # Case 3: No markers.csv, but verified FCS $PnS populated -> resolves from $PnS cleanly with warning
    res_pns = resolve_marker_metadata(
        dataset_dir=tmp_path / "empty_dir",
        fcs_pnn=markers,
        fcs_pns=["CD3 V450", "CD4 PcpCy55", "CD8 APCH7"],
    )
    assert res_pns.has_verified_biological_markers is True
    assert res_pns.detector_to_marker["B710-A"] == "CD4"
    assert len(res_pns.warnings) == 1


def test_cd20_confirmatory_cd19_and_cd34_ckit_signature_rules() -> None:
    """Verify CD20+ guards against CD19- when measured, and CD34+ c-Kit+ outranks generic CD34+."""
    from flowx.clustering.cluster import annotate_clusters_transparently

    # 1. Both CD19 and CD20 measured: CD20+ CD19- CD3- must NOT be called B-Cell Lineage (CD20+)
    stats_b = pd.DataFrame([
        {"Cluster ID": 0, "Count": 200, "% Parent": 40.0, "% Total": 40.0, "M1": 0.05, "M2": 0.85, "M3": 0.80},  # CD19+ CD20+ CD3-
        {"Cluster ID": 1, "Count": 200, "% Parent": 40.0, "% Total": 40.0, "M1": 0.05, "M2": 0.05, "M3": 0.80},  # CD19- CD20+ CD3-
        {"Cluster ID": 2, "Count": 100, "% Parent": 20.0, "% Total": 20.0, "M1": 0.85, "M2": 0.05, "M3": 0.05},  # CD3+
    ])
    pheno_b, _, _, _ = annotate_clusters_transparently(
        cluster_stats_df=stats_b,
        markers=["M1", "M2", "M3"],
        marker_name_map={"M1": "CD3", "M2": "CD19", "M3": "CD20"},
    )
    assert "B-Cell Lineage" in pheno_b[0]
    assert pheno_b[1] != "B-Cell Lineage (CD20+)"

    # 2. Panel without CD19 still annotates CD20+ CD3- as B-Cell Lineage (CD20+)
    stats_no_cd19 = pd.DataFrame([
        {"Cluster ID": 0, "Count": 200, "% Parent": 50.0, "% Total": 50.0, "M1": 0.05, "M3": 0.85},
        {"Cluster ID": 1, "Count": 200, "% Parent": 50.0, "% Total": 50.0, "M1": 0.85, "M3": 0.05},
    ])
    pheno_no_cd19, _, _, _ = annotate_clusters_transparently(
        cluster_stats_df=stats_no_cd19,
        markers=["M1", "M3"],
        marker_name_map={"M1": "CD3", "M3": "CD20"},
    )
    assert pheno_no_cd19[0] == "B-Cell Lineage (CD20+)"

    # 3. CD34 + c-Kit panel distinguishes HSPC (CD34+ c-Kit+) vs Endothelial/Stromal (CD34+ c-Kit-)
    stats_hspc = pd.DataFrame([
        {"Cluster ID": 0, "Count": 150, "% Parent": 50.0, "% Total": 50.0, "M1": 0.85, "M2": 0.82},
        {"Cluster ID": 1, "Count": 150, "% Parent": 50.0, "% Total": 50.0, "M1": 0.85, "M2": 0.04},
    ])
    pheno_hspc, _, _, _ = annotate_clusters_transparently(
        cluster_stats_df=stats_hspc,
        markers=["M1", "M2"],
        marker_name_map={"M1": "CD34", "M2": "c-Kit"},
    )
    assert pheno_hspc[0] == "Hematopoietic Stem & Progenitor Cells (CD34+ c-Kit+)"
    assert pheno_hspc[1] == "CD34+ c-Kit- Endothelial / Stromal Lineage"

    # Also verify CD117 alias normalizes identically to c-Kit
    pheno_cd117, _, _, _ = annotate_clusters_transparently(
        cluster_stats_df=stats_hspc,
        markers=["M1", "M2"],
        marker_name_map={"M1": "CD34", "M2": "CD117"},
    )
    assert pheno_cd117[0] == "Hematopoietic Stem & Progenitor Cells (CD34+ c-Kit+)"
    assert pheno_cd117[1] == "CD34+ c-Kit- Endothelial / Stromal Lineage"


def test_cd34_lineage_exclusions_and_immature_thymocytes() -> None:
    """Verify CD34+ HSPC rule does not hijack T/NK/immature thymocyte clusters, and CD1a+ DP thymocytes are resolved."""
    from flowx.clustering.cluster import annotate_clusters_transparently

    stats_df = pd.DataFrame([
        # C0: Mature CD3+ CD4+ CD8+ DP T cell with dim CD34 spread -> must NOT be HSPC
        {"Cluster ID": 0, "Count": 100, "% Parent": 20.0, "% Total": 20.0, "M1": 0.70, "M2": 0.75, "M3": 0.72, "M4": 0.05, "M5": 0.05, "M6": 0.55},
        # C1: Immature cortical DP thymocyte (CD3- CD1a+ CD4+ CD8+) -> Immature Thymocytes
        {"Cluster ID": 1, "Count": 100, "% Parent": 20.0, "% Total": 20.0, "M1": 0.05, "M2": 0.78, "M3": 0.76, "M4": 0.82, "M5": 0.05, "M6": 0.05},
        # C2: NK cell (CD3- CD56+) with dim CD34 spread -> must be NK Cells, NOT HSPC
        {"Cluster ID": 2, "Count": 100, "% Parent": 20.0, "% Total": 20.0, "M1": 0.05, "M2": 0.05, "M3": 0.05, "M4": 0.05, "M5": 0.80, "M6": 0.52},
        # C3: True Lin- CD34+ HSPC -> Hematopoietic Stem & Progenitor Cells (CD34+)
        {"Cluster ID": 3, "Count": 100, "% Parent": 20.0, "% Total": 20.0, "M1": 0.05, "M2": 0.05, "M3": 0.05, "M4": 0.05, "M5": 0.05, "M6": 0.88},
        # C4: True CD3- Non-T Lymphocyte (Lin- CD1a- CD56- CD34-) -> CD3- Non-T Lymphocytes
        {"Cluster ID": 4, "Count": 100, "% Parent": 20.0, "% Total": 20.0, "M1": 0.05, "M2": 0.05, "M3": 0.05, "M4": 0.05, "M5": 0.05, "M6": 0.05},
    ])
    pheno, macro, _, _ = annotate_clusters_transparently(
        cluster_stats_df=stats_df,
        markers=["M1", "M2", "M3", "M4", "M5", "M6"],
        marker_name_map={
            "M1": "CD3",
            "M2": "CD4",
            "M3": "CD8",
            "M4": "CD1a",
            "M5": "CD56",
            "M6": "CD34",
        },
    )
    assert pheno[0] == "CD4+ CD8+ DP T Cells"
    assert pheno[1] == "Immature Thymocytes (CD1a+ CD4+ CD8+)"
    assert macro[1] == "CD4+ CD8+ DP T Cells"
    assert pheno[2] == "NK Cells (CD56+)"
    assert pheno[3] == "Hematopoietic Stem & Progenitor Cells (CD34+)"
    assert pheno[4] == "CD3- Non-T Lymphocytes"



