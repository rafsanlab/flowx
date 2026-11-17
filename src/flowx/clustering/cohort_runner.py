"""Cohort Population Discovery & Statistical Analysis Orchestrator.

Coordinates multi-sample unsupervised clustering (FlowSOM as primary discovery engine,
PhenoGraph), reproducible 2D UMAP manifold embedding, automated lineage annotation,
quantitative cohort feature extraction, and differential abundance & state testing.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from flowx.clustering.cluster import (
    ClusteringResult,
    run_phenograph,
    run_som,
)
from flowx.clustering.qc_clustering import run_clustering_qc
from flowx.clustering.reduction import compute_umap_embedding
from flowx.io.logger import format_duration, get_logger
from flowx.io.parser import write_fcs
from flowx.stats import (
    CohortFeatureExtractor,
    CohortFeatureSet,
    DAConfig,
    DAResult,
    DSConfig,
    DSResult,
    export_prism_tables,
    run_differential_abundance,
    run_differential_state,
)
from flowx.visualization.plotting import (
    plot_cluster_heatmap,
    plot_cluster_population_bridge,
    plot_clusters_scatter_2d,
    plot_cohort_umap_covariates,
    plot_population_expression_heatmap,
    plot_stratified_umap_density_contours,
    plot_umap_marker_features,
    plot_umap_projection,
)
from flowx.visualization.stats_plots import (
    plot_cohort_heatmap,
    plot_differential_state_heatmap,
    plot_forest_da,
    plot_population_abundance_boxplots,
    plot_volcano_da,
)


def _strip_cluster_prefix(lbl: Any) -> str:
    """Strip any leading 'C0: ' or 'C0 - ' prefix to return the pure biological population name."""
    raw = str(lbl).strip()
    has_prefix = bool(re.match(r"^C\d+\s*[:\-]\s*", raw))
    cleaned = re.sub(r"^C\d+\s*[:\-]\s*", "", raw).strip()
    if has_prefix and cleaned:
        if cleaned == "Low/Neg Across Panel" or re.fullmatch(
            r"[A-Za-z0-9_/\-\(\)]+(?:\+{1,2}|\-)(\s+[A-Za-z0-9_/\-\(\)]+(?:\+{1,2}|\-))*",
            cleaned,
        ):
            return f"Unassigned ({cleaned})"
    return cleaned or raw


def _apply_cluster_rename_and_aggregate(
    feature_set: CohortFeatureSet,
    labeled_dfs: Dict[str, pd.DataFrame],
    umap_embedding: Optional[pd.DataFrame],
    cluster_statistics: Optional[pd.DataFrame],
    name_map: Dict[Union[int, str], str],
    logger: Any = None,
) -> Tuple[CohortFeatureSet, Dict[str, str]]:
    """Rename cluster IDs/populations and aggregate counts, frequencies, and MFIs if many-to-one.

    Handles biological macro-grouping where multiple computational clusters are assigned the
    same lineage label (e.g. Cluster 0 and Cluster 2 both mapped to 'CD4+ T Cells').
    """
    clean_map: Dict[str, str] = {}
    for k, v in name_map.items():
        str_k = str(k)
        clean_map[str_k] = str(v)
        if str_k.isdigit():
            clean_map[f"Cluster_{str_k}"] = str(v)
            clean_map[f"Cluster {str_k}"] = str(v)

    col_rename_dict: Dict[str, str] = {}
    for p in feature_set.population_names:
        str_p = str(p)
        col_rename_dict[p] = clean_map.get(str_p, clean_map.get(f"Cluster_{str_p}", str_p))

    is_many_to_one = len(set(col_rename_dict.values())) < len(col_rename_dict)
    if is_many_to_one:
        pre_counts = feature_set.count_matrix.copy()
        feature_set.count_matrix = (
            feature_set.count_matrix.rename(columns=col_rename_dict).T.groupby(level=0, sort=False).sum().T
        )
        feature_set.viable_freq_matrix = (
            feature_set.viable_freq_matrix.rename(columns=col_rename_dict).T.groupby(level=0, sort=False).sum().T
        )
        feature_set.parent_freq_matrix = (
            feature_set.parent_freq_matrix.rename(columns=col_rename_dict).T.groupby(level=0, sort=False).sum().T
        )
        total_counts = feature_set.count_matrix.replace(0, np.nan)
        for m in feature_set.mfi_tensor:
            mfi_df = feature_set.mfi_tensor[m]
            w_sum = (mfi_df * pre_counts).rename(columns=col_rename_dict).T.groupby(level=0, sort=False).sum().T
            feature_set.mfi_tensor[m] = w_sum.div(total_counts).fillna(0.0)

        for m in feature_set.geomfi_tensor:
            geomfi_df = feature_set.geomfi_tensor[m]
            w_sum = (geomfi_df * pre_counts).rename(columns=col_rename_dict).T.groupby(level=0, sort=False).sum().T
            feature_set.geomfi_tensor[m] = w_sum.div(total_counts).fillna(0.0)

        for m in feature_set.rsd_tensor:
            feature_set.rsd_tensor[m] = (
                feature_set.rsd_tensor[m].rename(columns=col_rename_dict).T.groupby(level=0, sort=False).mean().T
            )

        seen = set()
        new_pop_names = []
        for p in feature_set.population_names:
            mapped = col_rename_dict.get(p, p)
            if mapped not in seen:
                seen.add(mapped)
                new_pop_names.append(mapped)
        feature_set.population_names = new_pop_names
    else:
        feature_set.population_names = [col_rename_dict[p] for p in feature_set.population_names]
        feature_set.count_matrix.rename(columns=col_rename_dict, inplace=True)
        feature_set.viable_freq_matrix.rename(columns=col_rename_dict, inplace=True)
        feature_set.parent_freq_matrix.rename(columns=col_rename_dict, inplace=True)
        for m in feature_set.mfi_tensor:
            feature_set.mfi_tensor[m].rename(columns=col_rename_dict, inplace=True)
        for m in feature_set.geomfi_tensor:
            feature_set.geomfi_tensor[m].rename(columns=col_rename_dict, inplace=True)
        for m in feature_set.rsd_tensor:
            feature_set.rsd_tensor[m].rename(columns=col_rename_dict, inplace=True)

    # Update event-level DataFrames
    for s_id, df in labeled_dfs.items():
        if "cluster_id" in df.columns:
            df["lineage_id"] = df["cluster_id"].map(
                lambda cid: clean_map.get(str(cid), clean_map.get(f"Cluster_{cid}", str(cid)))
            )
            df["cluster_name"] = df["lineage_id"]

    # Update UMAP DataFrame
    if umap_embedding is not None and "cluster_id" in umap_embedding.columns:
        umap_embedding["lineage_id"] = umap_embedding["cluster_id"].map(
            lambda cid: clean_map.get(str(cid), clean_map.get(f"Cluster_{cid}", str(cid)))
        )
        umap_embedding["cluster_name"] = umap_embedding["lineage_id"]

    # Update cluster statistics summary
    if cluster_statistics is not None and "Cluster ID" in cluster_statistics.columns:
        cluster_statistics["Assigned Lineage"] = cluster_statistics["Cluster ID"].map(
            lambda cid: clean_map.get(str(cid), clean_map.get(f"Cluster_{cid}", str(cid)))
        )

    return feature_set, col_rename_dict


def _build_cluster_population_crosswalk(
    clustering_result: ClusteringResult,
    labeled_dfs: Dict[str, pd.DataFrame],
    name_map: Optional[Dict[Union[int, str], str]] = None,
) -> pd.DataFrame:
    """Build a canonical crosswalk table mapping technical cluster IDs (C0..Ck) to biological populations."""
    all_frames = [df for df in labeled_dfs.values() if "cluster_id" in df.columns]
    if all_frames:
        merged = pd.concat(
            [df[[c for c in ("cluster_id", "lineage_id") if c in df.columns]] for df in all_frames],
            ignore_index=True,
        )
    else:
        merged = pd.DataFrame({"cluster_id": clustering_result.cluster_labels})

    total_events = max(1, len(merged))
    counts_by_cid = merged["cluster_id"].value_counts().to_dict()
    macro_map = getattr(clustering_result, "cluster_to_macro_group", None) or {}
    pheno_map = getattr(clustering_result, "phenotype_labels", None) or {}
    str_override = {str(k): str(v) for k, v in (name_map or {}).items()}

    rows: List[Dict[str, Any]] = []
    for cid in sorted(counts_by_cid.keys(), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
        cnt = int(counts_by_cid[cid])
        pop_label = str_override.get(str(cid)) or pheno_map.get(cid)
        if not pop_label and "lineage_id" in merged.columns:
            mode_vals = merged.loc[merged["cluster_id"] == cid, "lineage_id"].mode()
            if not mode_vals.empty:
                pop_label = str(mode_vals.iloc[0])
        if not pop_label:
            pop_label = f"Cluster_{cid}"
        clean_pop = _strip_cluster_prefix(pop_label)
        macro = macro_map.get(cid, macro_map.get(str(cid), clean_pop))
        rows.append(
            {
                "Cluster_ID": cid,
                "Cluster_Label": f"C{cid}",
                "Annotated_Population": clean_pop,
                "Macro_Group": str(macro),
                "Event_Count": cnt,
                "Pct_Total_Events": round(100.0 * cnt / total_events, 3),
            }
        )
    return pd.DataFrame(rows)


def _render_cluster_qc_metrics_plot(
    clustering_result: ClusteringResult,
    crosswalk_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render technical cluster QC summary (cluster sizes and optional auto-k validation curve)."""
    import matplotlib.pyplot as plt
    from flowx.visualization.style import apply_nature_axis_style, set_nature_style

    set_nature_style()
    scores = getattr(clustering_result, "k_optimization_scores", None) or {}
    metric_name = getattr(clustering_result, "k_optimization_metric", None) or "Validation Score"

    has_k_curve = isinstance(scores, dict) and len(scores) >= 2
    fig, axes = plt.subplots(1, 2 if has_k_curve else 1, figsize=(8.6 if has_k_curve else 4.8, 3.4))
    if not has_k_curve:
        axes = [axes]

    ax0 = axes[0]
    if not crosswalk_df.empty:
        x_lbls = crosswalk_df["Cluster_Label"].astype(str).tolist()
        x_pos = np.arange(len(x_lbls))
        ax0.bar(x_pos, crosswalk_df["Pct_Total_Events"].astype(float), color="#4C72B0", alpha=0.88)
        ax0.set_xticks(x_pos)
        ax0.set_xticklabels(x_lbls, rotation=35, ha="right", fontsize=9.5, fontweight="bold")
    ax0.tick_params(axis="y", labelsize=9.5)
    ax0.set_title("Unsupervised Cluster Size Distribution", fontsize=11.5, fontweight="bold", pad=8)
    ax0.set_xlabel("Cluster ID", fontsize=10.5, fontweight="bold")
    ax0.set_ylabel("% of Clean Events", fontsize=10.5, fontweight="bold")
    apply_nature_axis_style(ax0)

    if has_k_curve:
        ax1 = axes[1]
        ks = sorted(scores.keys(), key=lambda v: int(v))
        vals = [float(scores[k]) for k in ks]
        ax1.plot([int(k) for k in ks], vals, marker="o", color="#DD8452", linewidth=2.0, markersize=5.5)
        sel_k = getattr(clustering_result, "cluster_count", None)
        if sel_k is not None and sel_k in scores:
            ax1.axvline(int(sel_k), color="#55A868", linestyle="--", linewidth=1.6, label=f"Selected k={sel_k}")
            ax1.legend(frameon=False, fontsize=9.5)
        ax1.tick_params(axis="both", labelsize=9.5)
        ax1.set_title(f"Cluster Optimization ({metric_name})", fontsize=11.5, fontweight="bold", pad=8)
        ax1.set_xlabel("Candidate Cluster Count (k)", fontsize=10.5, fontweight="bold")
        ax1.set_ylabel(str(metric_name), fontsize=10.5, fontweight="bold")
        apply_nature_axis_style(ax1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@dataclass
class CohortPopulationResult:
    """Container holding cohort population discovery, clustering, and statistical analysis outputs.

    Attributes:
        samples: List of sample identifiers processed.
        labeled_dfs: Per-sample event DataFrames augmented with 'cluster_id' and 'lineage_id'.
        clustering_result: Unsupervised clustering model output (FlowSOM or PhenoGraph).
        feature_set: CohortFeatureSet containing count, frequency, and expression tensors.
        da_result: Differential Abundance result if condition metadata was evaluated.
        ds_result: Differential State result if condition metadata was evaluated.
        umap_embedding: Optional DataFrame of 2D UMAP coordinates across cohort events.
        condition_column: Experimental condition column in metadata if evaluated.
        reference_group: Reference cohort condition string.
        target_group: Target cohort condition string.
        paired_column: Optional paired observation column.
        generated_plots: Mapping of plot description to saved image filepath.
        generated_tables: Mapping of table description to saved CSV/JSON filepath.
        output_dir: Root output directory for this cohort analysis run.
        execution_time_seconds: Total wall-clock processing time.
    """

    samples: List[str]
    labeled_dfs: Dict[str, pd.DataFrame]
    clustering_result: ClusteringResult
    feature_set: CohortFeatureSet
    da_result: Optional[DAResult] = None
    ds_result: Optional[DSResult] = None
    umap_embedding: Optional[pd.DataFrame] = None
    condition_column: Optional[str] = None
    batch_column: Optional[str] = None
    umap_covariates: Optional[List[str]] = None
    cluster_palette: str = "tab20"
    reference_group: Optional[str] = None
    target_group: Optional[str] = None
    paired_column: Optional[str] = None
    generated_plots: Dict[str, str] = field(default_factory=dict)
    generated_tables: Dict[str, str] = field(default_factory=dict)
    output_dir: str = ""
    execution_time_seconds: float = 0.0
    cluster_feature_set: Optional[CohortFeatureSet] = None
    stats_granularity: str = "populations"
    ds_markers: Optional[List[str]] = None

    def rename_clusters(
        self,
        name_map: Dict[Union[int, str], str],
        update_plots: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "CohortPopulationResult":
        """Rename discovered cluster IDs to biological lineage names and update all tables & figures.

        Allows users to inspect initial unsupervised clustering phenotypes, assign biological names
        (e.g., {0: "CD4+ Naive T", 1: "Classical Monocytes", 2: "CD8+ Effector"}), and propagate
        them across all feature tensors, sample DataFrames, Prism exports, and publication plots.
        If multiple clusters are assigned the same biological name, event counts and frequencies
        are automatically aggregated into a single unified population.

        Args:
            name_map: Dictionary mapping numeric cluster ID (int or str) or current population name
                      to the new biological name string.
            update_plots: If True, re-renders and overwrites publication figures with new labels.
            output_dir: Directory where updated tables and plots should be saved (defaults to self.output_dir).

        Returns:
            Self with updated metadata, feature tensors, and regenerated tables/plots.
        """
        logger = get_logger()
        out_base = Path(output_dir or self.output_dir or "outputs/cohort_population")

        cl_cols = (
            set(self.cluster_feature_set.count_matrix.columns)
            if (self.cluster_feature_set is not None and hasattr(self.cluster_feature_set, "count_matrix"))
            else set()
        )
        is_numeric_or_cluster_key = all(
            isinstance(k, int) or str(k).strip().isdigit() or str(k).strip() in cl_cols
            for k in name_map.keys()
        )
        base_fs = (
            copy.deepcopy(self.cluster_feature_set)
            if (is_numeric_or_cluster_key and self.cluster_feature_set is not None)
            else self.feature_set
        )
        self.feature_set, col_rename_dict = _apply_cluster_rename_and_aggregate(
            feature_set=base_fs,
            labeled_dfs=self.labeled_dfs,
            umap_embedding=self.umap_embedding,
            cluster_statistics=self.clustering_result.cluster_statistics if hasattr(self.clustering_result, "cluster_statistics") else None,
            name_map=name_map,
            logger=logger,
        )

        # Re-save updated tables to disk
        step_clustering_tables = out_base / "05_population_discovery" / "tables"
        if not step_clustering_tables.exists():
            step_clustering_tables = out_base / "population_discovery" / "tables"
        if not step_clustering_tables.exists():
            step_clustering_tables = out_base / "tables"
        step_clustering_tables.mkdir(parents=True, exist_ok=True)

        counts_csv = step_clustering_tables / "population_counts.csv"
        self.feature_set.count_matrix.to_csv(counts_csv)
        self.generated_tables["population_counts"] = str(counts_csv)

        freqs_csv = step_clustering_tables / "population_frequencies_viable.csv"
        self.feature_set.viable_freq_matrix.to_csv(freqs_csv)
        self.generated_tables["population_frequencies"] = str(freqs_csv)

        parent_freqs_csv = step_clustering_tables / "population_frequencies_parent.csv"
        self.feature_set.parent_freq_matrix.to_csv(parent_freqs_csv)
        self.generated_tables["population_frequencies_parent"] = str(parent_freqs_csv)

        if hasattr(self.clustering_result, "cluster_statistics") and self.clustering_result.cluster_statistics is not None:
            cl_stats_csv = step_clustering_tables / "cluster_phenotypes_summary.csv"
            self.clustering_result.cluster_statistics.to_csv(cl_stats_csv, index=False)
            self.generated_tables["cluster_phenotypes"] = str(cl_stats_csv)

        crosswalk_df = _build_cluster_population_crosswalk(
            self.clustering_result, self.labeled_dfs, name_map=name_map
        )
        crosswalk_csv = step_clustering_tables / "cluster_to_population_map.csv"
        crosswalk_df.to_csv(crosswalk_csv, index=False)
        self.generated_tables["cluster_to_population_map"] = str(crosswalk_csv)

        # Resolve condition column for DA / DS and Prism
        cond_col = self.condition_column
        if not cond_col and self.feature_set.sample_metadata is not None:
            for c in ["condition", "Condition", "group", "Group", "batch_id"]:
                if c in self.feature_set.sample_metadata.columns:
                    cond_col = c
                    break

        # Re-run DA & DS if conditions are present
        if cond_col and self.feature_set.sample_metadata is not None and cond_col in self.feature_set.sample_metadata.columns:
            from flowx.stats.differential_abundance import DAConfig, run_differential_abundance
            from flowx.stats.differential_state import DSConfig, run_differential_state

            step_stats_tables = out_base / "06_statistical_analysis" / "tables"
            if not step_stats_tables.exists():
                step_stats_tables = out_base / "statistical_analysis" / "tables"
            if not step_stats_tables.exists():
                step_stats_tables = out_base / "tables"
            step_stats_tables.mkdir(parents=True, exist_ok=True)

            try:
                self.da_result = run_differential_abundance(
                    features=self.feature_set,
                    condition_column=cond_col,
                    reference_group=self.reference_group,
                    target_group=self.target_group,
                    config=DAConfig(paired_column=self.paired_column),
                )
                da_csv = step_stats_tables / "differential_abundance_summary.csv"
                self.da_result.summary_df.to_csv(da_csv, index=False)
                self.generated_tables["differential_abundance"] = str(da_csv)
            except Exception as exc:
                logger.debug(f"DA recalculation on renamed populations failed: {exc}")

            try:
                self.ds_result = run_differential_state(
                    features=self.feature_set,
                    condition_column=cond_col,
                    reference_group=self.reference_group,
                    target_group=self.target_group,
                    config=DSConfig(
                        paired_column=self.paired_column,
                        functional_markers=self.ds_markers,
                    ),
                )
                ds_csv = step_stats_tables / "differential_state_summary.csv"
                self.ds_result.summary_df.to_csv(ds_csv, index=False)
                self.generated_tables["differential_state"] = str(ds_csv)
            except Exception as exc:
                logger.debug(f"DS recalculation on renamed populations failed: {exc}")

        # Update GraphPad Prism tables with new names
        prism_files = export_prism_tables(
            feature_set=self.feature_set,
            output_dir=step_clustering_tables,
            condition_column=cond_col,
        )
        for k, p in prism_files.items():
            self.generated_tables[k] = str(p)

        # Re-render publication plots with updated lineage names
        if update_plots:
            step_clustering_plots = out_base / "05_population_discovery" / "plots"
            if not step_clustering_plots.exists():
                step_clustering_plots = out_base / "population_discovery" / "plots"
            if not step_clustering_plots.exists():
                step_clustering_plots = out_base / "plots"
            step_clusters_plots = step_clustering_plots / "clusters"
            step_populations_plots = step_clustering_plots / "populations"
            step_clusters_plots.mkdir(parents=True, exist_ok=True)
            step_populations_plots.mkdir(parents=True, exist_ok=True)

            step_stats_plots = out_base / "06_statistical_analysis" / "plots"
            if not step_stats_plots.exists():
                step_stats_plots = out_base / "statistical_analysis" / "plots"
            if not step_stats_plots.exists():
                step_stats_plots = out_base / "plots"
            step_stats_plots.mkdir(parents=True, exist_ok=True)

            try:
                p_cohort_hm = step_populations_plots / "population_frequency_clustermap.png"
                plot_cohort_heatmap(
                    self.feature_set,
                    matrix_type="frequency",
                    condition_column=cond_col,
                    group_by_population=True,
                    output_path=p_cohort_hm,
                )
                self.generated_plots["cohort_frequency_clustermap"] = str(p_cohort_hm)
            except Exception as exc:
                logger.debug(f"Cohort clustermap re-plotting failed: {exc}")

            if self.umap_embedding is not None and "lineage_id" in self.umap_embedding.columns:
                try:
                    cluster_only_labels = (
                        [f"C{c}" for c in self.umap_embedding["cluster_id"]]
                        if "cluster_id" in self.umap_embedding.columns
                        else []
                    )
                    if cluster_only_labels:
                        p_umap_cl = step_clusters_plots / "umap_clusters.png"
                        plot_umap_projection(
                            embedding=self.umap_embedding[["UMAP_1", "UMAP_2"]].to_numpy(),
                            labels=cluster_only_labels,
                            output_path=str(p_umap_cl),
                            title="Cohort 2D UMAP (Clusters)",
                            legend_mode="external",
                            palette=getattr(self, "cluster_palette", "tab20"),
                        )
                        self.generated_plots["cohort_umap_clusters"] = str(p_umap_cl)

                    pop_only_labels = np.asarray(
                        [_strip_cluster_prefix(v) for v in self.umap_embedding["lineage_id"]]
                    )
                    p_umap_pop = step_populations_plots / "umap_populations.png"
                    plot_umap_projection(
                        embedding=self.umap_embedding[["UMAP_1", "UMAP_2"]].to_numpy(),
                        labels=pop_only_labels,
                        output_path=str(p_umap_pop),
                        title="Cohort 2D UMAP (Annotated Lineages)",
                        legend_mode="external",
                        palette=getattr(self, "cluster_palette", "tab20"),
                    )
                    self.generated_plots["cohort_umap_populations"] = str(p_umap_pop)

                    p_bridge = step_clustering_plots / "cluster_to_population_bridge.png"
                    plot_cluster_population_bridge(
                        embedding=self.umap_embedding[["UMAP_1", "UMAP_2"]].to_numpy(),
                        cluster_labels=cluster_only_labels or pop_only_labels,
                        population_labels=pop_only_labels,
                        mapping_df=crosswalk_df,
                        palette=getattr(self, "cluster_palette", "tab20"),
                        output_path=p_bridge,
                    )
                    self.generated_plots["cluster_to_population_bridge"] = str(p_bridge)
                except Exception as exc:
                    logger.debug(f"UMAP cluster/population re-plotting failed: {exc}")

                if "sample_id" in self.umap_embedding.columns:
                    try:
                        p_umap_cov = step_populations_plots / "umap_covariates.png"
                        plot_cohort_umap_covariates(
                            embedding=self.umap_embedding[["UMAP_1", "UMAP_2"]].to_numpy(),
                            sample_ids=self.umap_embedding["sample_id"].tolist(),
                            lineage_or_cluster_labels=np.asarray(
                                [_strip_cluster_prefix(v) for v in self.umap_embedding["lineage_id"]]
                            ),
                            sample_metadata=self.feature_set.sample_metadata,
                            condition_column=cond_col,
                            batch_column=self.batch_column,
                            covariate_columns=self.umap_covariates,
                            cluster_palette=getattr(self, "cluster_palette", "tab20"),
                            output_path=str(p_umap_cov),
                        )
                        self.generated_plots["cohort_umap_covariates"] = str(p_umap_cov)
                    except Exception as exc:
                        logger.debug(f"Cohort UMAP covariates re-plotting failed: {exc}")

            if self.da_result is not None:
                try:
                    p_box = step_stats_plots / "population_abundance_boxplots.png"
                    plot_population_abundance_boxplots(
                        features=self.feature_set,
                        condition_column=cond_col or "condition",
                        output_path=p_box,
                    )
                    self.generated_plots["abundance_boxplots"] = str(p_box)
                except Exception as exc:
                    logger.debug(f"Boxplot re-plotting failed: {exc}")

                try:
                    p_volcano = step_stats_plots / "differential_abundance_volcano.png"
                    plot_volcano_da(
                        da_result=self.da_result,
                        output_path=p_volcano,
                    )
                    self.generated_plots["da_volcano"] = str(p_volcano)
                except Exception as exc:
                    logger.debug(f"Volcano re-plotting failed: {exc}")

                try:
                    p_forest = step_stats_plots / "differential_abundance_forest.png"
                    plot_forest_da(
                        da_result=self.da_result,
                        output_path=p_forest,
                    )
                    self.generated_plots["da_forest"] = str(p_forest)
                except Exception as exc:
                    logger.debug(f"Forest plot re-plotting failed: {exc}")

            if self.ds_result is not None:
                try:
                    p_ds_hm = step_stats_plots / "differential_state_heatmap.png"
                    plot_differential_state_heatmap(
                        ds_result=self.ds_result,
                        output_path=p_ds_hm,
                    )
                    self.generated_plots["ds_heatmap"] = str(p_ds_hm)
                except Exception as exc:
                    logger.debug(f"DS heatmap re-plotting failed: {exc}")

        logger.info(f"Successfully renamed and aggregated clusters to biological lineage names.")
        return self


def run_cohort_population_analysis(
    cohort_dfs: Dict[str, pd.DataFrame],
    sample_metadata: Optional[Union[pd.DataFrame, Dict[str, Dict[str, Any]], str, Path]] = None,
    metadata_file: Optional[Union[str, Path]] = None,
    markers: Optional[Union[str, List[str]]] = None,
    markers_file: Optional[Union[str, Path]] = None,
    marker_roles: Optional[List[str]] = None,
    stats_marker_roles: Optional[List[str]] = None,
    ds_markers: Optional[Union[str, List[str]]] = None,
    include_markers: Optional[List[str]] = None,
    exclude_markers: Optional[List[str]] = None,
    marker_name_map: Optional[Dict[str, str]] = None,
    cell_signatures: Optional[Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]] = None,
    clustering_method: Literal["som", "phenograph"] = "som",
    n_clusters: Optional[int] = 10,
    auto_k: bool = True,
    auto_k_method: str = "consensus",
    k_min: int = 4,
    k_max: int = 20,
    consensus_subsample: float = 0.8,
    consensus_iterations: int = 50,
    compute_umap: bool = True,
    umap_events_per_sample: int = 1000,
    condition_column: Optional[str] = None,
    batch_column: Optional[str] = None,
    umap_covariates: Optional[List[str]] = None,
    reference_group: Optional[str] = None,
    target_group: Optional[str] = None,
    paired_column: Optional[str] = None,
    cluster_name_map: Optional[Dict[Union[int, str], str]] = None,
    cluster_palette: str = "tab20",
    stats_granularity: Literal["populations", "clusters", "both"] = "populations",
    da_test_method: str = "auto",
    ds_test_method: str = "auto",
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    stratified_density_contours: bool = True,
    generate_volcano_plot: bool = True,
    generate_forest_plot: bool = True,
    generate_abundance_boxplots: bool = True,
    generate_ds_heatmap: bool = True,
    export_prism: bool = True,
    export_augmented_fcs: bool = True,
    export_flowjo_wsp: bool = True,
    export_gating_ml: bool = True,
    preprocessing_wsp_path: Optional[Union[str, Path]] = None,
    preprocessing_gating_ml_path: Optional[Union[str, Path]] = None,
    generate_miflowcyt_report: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    random_seed: int = 42,
    generate_plots: bool = True,
    numbered_steps: bool = True,
) -> CohortPopulationResult:
    """Execute end-to-end Population Identification & Quantitative Cohort Statistical Analysis.

    Pipeline sequence:
    1. Pool balanced/clean events from cohort samples.
    2. Fit FlowSOM (primary discovery engine) or PhenoGraph across lineage markers.
    3. Generate automated lineage annotations and macro-groupings.
    4. Compute reproducible 2D UMAP manifold projections on clean event matrices.
    5. Assign 'cluster_id', 'lineage_id', and 'umap_1', 'umap_2' to each sample DataFrame.
    6. Extract quantitative cohort feature set (counts, frequencies, MFI, GeoMFI, rSD).
    7. Execute Differential Abundance (DA) and Differential State (DS) if conditions are present.
    8. Generate publication-ready Nature-style figures and structured audit tables.

    Args:
        cohort_dfs: Dictionary mapping sample_id to clean event DataFrame.
        sample_metadata: Metadata DataFrame or dict with condition/batch annotations.
        markers: Lineage markers for clustering and UMAP (auto-detected if None).
        clustering_method: 'som' (FlowSOM, recommended) or 'phenograph'.
        n_clusters: Target number of metaclusters for FlowSOM (default: 10).
        compute_umap: Whether to compute 2D UMAP projection (default: True).
        umap_events_per_sample: Maximum events per sample to include in UMAP embedding.
        condition_column: Metadata column holding experimental conditions (e.g. 'condition').
        reference_group: Baseline condition for DA/DS comparisons.
        target_group: Target condition for DA/DS comparisons.
        output_dir: Destination path for figures, tables, and reports.
        random_seed: Deterministic random seed for reproducibility.
        generate_plots: Whether to render and save publication figures.

    Returns:
        CohortPopulationResult holding models, labeled DataFrames, feature sets, and statistical tests.
    """
    t_start = time.perf_counter()
    out_base = Path(output_dir or "outputs/cohort_population")
    out_base.mkdir(parents=True, exist_ok=True)

    # Initialize structured run logging by default (both console stream and outputs/.../pipeline.log)
    # Use overwrite=False so Step 01-04 logs are preserved when called from FlowCytometryPipeline._run_cohort
    from flowx.io.logger import setup_run_logger
    logger, _ = setup_run_logger(run_dir=out_base, log_filename="pipeline.log", overwrite=False)

    sample_ids = sorted(cohort_dfs.keys())
    if not sample_ids:
        raise ValueError("Must provide at least one sample DataFrame for cohort population analysis.")

    # Load metadata from file path if supplied
    if sample_metadata is None and metadata_file is not None:
        from flowx.io.metadata import load_sample_metadata
        sample_metadata = load_sample_metadata(metadata_file)
    elif isinstance(sample_metadata, (str, Path)):
        from flowx.io.metadata import load_sample_metadata
        sample_metadata = load_sample_metadata(sample_metadata)

    # Standardized step-named output subdirectories for cohesive pipeline structure
    if numbered_steps:
        step_clustering_plots = out_base / "05_population_discovery" / "plots"
        step_clustering_tables = out_base / "05_population_discovery" / "tables"
        step_stats_plots = out_base / "06_statistical_analysis" / "plots"
        step_stats_tables = out_base / "06_statistical_analysis" / "tables"
        step_workspaces = out_base / "exports" / "workspaces"
        step_fcs = out_base / "exports" / "augmented_fcs"
        step_reports = out_base / "exports" / "reports"
    else:
        step_clustering_plots = out_base / "population_discovery" / "plots"
        step_clustering_tables = out_base / "population_discovery" / "tables"
        step_stats_plots = out_base / "statistical_analysis" / "plots"
        step_stats_tables = out_base / "statistical_analysis" / "tables"
        step_workspaces = out_base / "exports" / "workspaces"
        step_fcs = out_base / "exports" / "augmented_fcs"
        step_reports = out_base / "exports" / "reports"

    step_clusters_plots = step_clustering_plots / "clusters"
    step_populations_plots = step_clustering_plots / "populations"

    for d in [
        step_clustering_plots,
        step_clusters_plots,
        step_populations_plots,
        step_clustering_tables,
        step_stats_plots,
        step_stats_tables,
        step_workspaces,
        step_fcs,
        step_reports,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    generated_plots: Dict[str, str] = {}
    generated_tables: Dict[str, str] = {}

    # 1. Resolve Markers & Biological Marker Name Mapping (markers.csv / explicit / fallback)
    first_df = cohort_dfs[sample_ids[0]]
    resolved_marker_map: Dict[str, str] = dict(marker_name_map or {})
    m_res = None
    if markers_file is not None:
        from flowx.io.metadata import resolve_marker_metadata
        m_res = resolve_marker_metadata(
            markers_file=markers_file,
            fcs_pnn=list(first_df.columns),
            marker_mapping=resolved_marker_map or None,
            logger=logger,
        )
        for ch, sym in m_res.detector_to_marker.items():
            if ch not in resolved_marker_map or not resolved_marker_map[ch]:
                resolved_marker_map[ch] = sym

    if isinstance(markers, str) or marker_roles is not None or include_markers or exclude_markers or (not markers and m_res is not None):
        from flowx.io.metadata import select_markers_from_metadata
        sel_chs = select_markers_from_metadata(
            marker_metadata=m_res,
            available_columns=list(first_df.columns),
            explicit_markers=markers,
            marker_roles=marker_roles,
            include_markers=include_markers,
            exclude_markers=exclude_markers,
        )
        if sel_chs:
            markers = [c for c in sel_chs if c in first_df.columns]
        elif isinstance(markers, str) and markers.strip().lower() == "auto":
            markers = None

    if not markers:
        exclude_kw = ["time", "event", "orig", "sample", "batch", "fsc", "ssc", "umap", "cluster", "lineage", "autofluorescence", "af "]
        markers = [
            c for c in first_df.columns
            if not any(c.lower().startswith(ex) for ex in exclude_kw)
        ]

    logger.info(
        f"Initiating Cohort Population Discovery across {len(sample_ids)} samples and "
        f"{len(markers)} markers ({', '.join(markers)})."
    )

    # 2. Pool events with sample tracking indices
    pooled_rows = []
    sample_index_map: List[Tuple[str, int]] = []  # (sample_id, local_index)

    for s_id in sample_ids:
        df = cohort_dfs[s_id]
        n_ev = len(df)
        for i in range(n_ev):
            sample_index_map.append((s_id, i))
        df_annot = df[markers].copy()
        df_annot["__sample_id"] = s_id
        pooled_rows.append(df_annot)

    pooled_df = pd.concat(pooled_rows, ignore_index=True)
    total_pooled_events = len(pooled_df)
    logger.info(f"Pooled {total_pooled_events:,} events across {len(sample_ids)} samples for cohort clustering.")

    # 3. Unsupervised Clustering (FlowSOM or PhenoGraph)
    logger.info(f"Executing unsupervised clustering using {clustering_method.upper()}...")
    cluster_matrix = pooled_df[markers].to_numpy()

    if clustering_method == "phenograph":
        # PhenoGraph on large datasets: subsample if total events > 100,000 for tractability
        if total_pooled_events > 80000:
            logger.info(f"Subsampling to 80,000 events for PhenoGraph graph construction...")
            rng = np.random.default_rng(random_seed)
            sub_idx = rng.choice(total_pooled_events, size=80000, replace=False)
            sub_df = pooled_df.iloc[sub_idx][markers]
            cl_res = run_phenograph(
                sub_df,
                markers=markers,
                cluster_name_map=cluster_name_map,
                marker_name_map=resolved_marker_map or None,
                cell_signatures=cell_signatures,
            )
            # Assign remaining cells via 1-NN
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(sub_df.to_numpy())
            _, full_indices = nbrs.kneighbors(cluster_matrix)
            full_labels = cl_res.cluster_labels[full_indices.ravel()]
            cl_res.cluster_labels = full_labels
        else:
            cl_res = run_phenograph(
                pooled_df[markers],
                markers=markers,
                cluster_name_map=cluster_name_map,
                marker_name_map=resolved_marker_map or None,
                cell_signatures=cell_signatures,
            )
    else:
        # Native Permissive FlowSOM (MiniSom + Agglomerative Metaclustering)
        if auto_k:
            effective_k_max = max(k_max, n_clusters) if n_clusters is not None else k_max
            effective_k_min = min(k_min, max(2, effective_k_max - 1))
        else:
            effective_k_max = k_max
            effective_k_min = k_min

        # Determine SOM grid dimensions (e.g. 10x10 = 100 nodes for high-resolution topology)
        grid_target = n_clusters or effective_k_max or 10
        grid_dim = max(6, int(np.sqrt(grid_target * 6)))
        grid_dim = min(grid_dim, 12)
        cl_res = run_som(
            df=pooled_df[markers],
            markers=markers,
            x_dim=grid_dim,
            y_dim=grid_dim,
            n_metaclusters=n_clusters or 10,
            auto_k=auto_k,
            auto_k_method=auto_k_method,
            k_min=effective_k_min,
            k_max=effective_k_max,
            consensus_subsample=consensus_subsample,
            consensus_iterations=consensus_iterations,
            random_state=random_seed,
            cluster_name_map=cluster_name_map,
            marker_name_map=resolved_marker_map or None,
            cell_signatures=cell_signatures,
        )

    logger.info(
        f"Clustering complete: {cl_res.cluster_count} metaclusters identified across {total_pooled_events:,} events."
    )

    # Save transparent cluster annotation audit & empirical positivity thresholds
    if getattr(cl_res, "annotation_audit_df", None) is not None and not cl_res.annotation_audit_df.empty:
        audit_path = step_clustering_tables / "cluster_annotation_audit.csv"
        cl_res.annotation_audit_df.to_csv(audit_path, index=False)
        generated_tables["cluster_annotation_audit"] = str(audit_path)

    if getattr(cl_res, "marker_thresholds", None):
        th_df = pd.DataFrame(list(cl_res.marker_thresholds.values()))
        if not th_df.empty:
            if resolved_marker_map:
                th_df.insert(1, "marker_name", th_df["channel"].map(lambda c: resolved_marker_map.get(c, c)))
            th_path = step_clustering_tables / "cluster_marker_thresholds.csv"
            th_df.to_csv(th_path, index=False)
            generated_tables["cluster_marker_thresholds"] = str(th_path)

    # Save metacluster k optimization metric scores if auto_k was run
    if hasattr(cl_res, "k_optimization_scores") and cl_res.k_optimization_scores:
        metric_col = getattr(cl_res, "k_optimization_metric", "") or "score"
        k_df = pd.DataFrame([
            {"k": k, metric_col: score, "is_optimal": (k == cl_res.cluster_count)}
            for k, score in sorted(cl_res.k_optimization_scores.items())
        ])
        k_path = step_clustering_tables / "metacluster_k_metrics.csv"
        k_df.to_csv(k_path, index=False)
        generated_tables["k_optimization_metrics"] = str(k_path)

    # 4. Optional 2D UMAP Manifold Projection
    umap_df: Optional[pd.DataFrame] = None
    all_umap_coords: Optional[np.ndarray] = None
    if compute_umap:
        total_umap_quota = min(total_pooled_events, umap_events_per_sample * len(sample_ids))
        logger.info(f"Computing 2D UMAP manifold embedding (target: {total_umap_quota:,} events across cohort)...")
        try:
            umap_coords, sampled_indices = compute_umap_embedding(
                df=pooled_df[markers],
                markers=markers,
                subsample=total_umap_quota,
                random_state=random_seed,
            )
            if len(sampled_indices) < total_pooled_events:
                from sklearn.neighbors import NearestNeighbors

                nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
                    pooled_df.iloc[sampled_indices][markers].to_numpy()
                )
                full_nn_indices = nn.kneighbors(pooled_df[markers].to_numpy(), return_distance=False).ravel()
                all_umap_coords = umap_coords[full_nn_indices]
            else:
                all_umap_coords = umap_coords

            umap_df = pd.DataFrame(
                umap_coords,
                columns=["UMAP_1", "UMAP_2"],
                index=sampled_indices,
            )
            umap_df["cluster_id"] = cl_res.cluster_labels[sampled_indices]
            umap_df["sample_id"] = [sample_index_map[i][0] for i in sampled_indices]
            if cl_res.phenotype_labels:
                umap_df["lineage_id"] = [
                    _strip_cluster_prefix(cl_res.phenotype_labels.get(c, f"Cluster_{c}"))
                    for c in umap_df["cluster_id"]
                ]
            else:
                umap_df["lineage_id"] = [f"Cluster_{c}" for c in umap_df["cluster_id"]]
            logger.info(f"UMAP projection successfully computed for {len(sampled_indices):,} representative cells.")
        except Exception as exc:
            logger.warning(f"UMAP manifold projection encountered an error: {exc}. Continuing without UMAP.")
            umap_df = None
            all_umap_coords = None

    # 5. Partition Cluster Labels back into Individual Sample DataFrames
    labeled_dfs: Dict[str, pd.DataFrame] = {}
    curr_offset = 0

    for s_id in sample_ids:
        orig_df = cohort_dfs[s_id].copy()
        n_ev = len(orig_df)
        sample_clusters = cl_res.cluster_labels[curr_offset : curr_offset + n_ev]

        orig_df["cluster_id"] = sample_clusters
        if all_umap_coords is not None:
            sample_umap = all_umap_coords[curr_offset : curr_offset + n_ev]
            orig_df["UMAP_1"] = sample_umap[:, 0]
            orig_df["UMAP_2"] = sample_umap[:, 1]

        # Assign phenotype/lineage descriptions
        if cl_res.phenotype_labels:
            orig_df["lineage_id"] = [
                _strip_cluster_prefix(cl_res.phenotype_labels.get(c, f"Cluster_{c}"))
                for c in sample_clusters
            ]
        else:
            orig_df["lineage_id"] = [f"Cluster_{c}" for c in sample_clusters]

        labeled_dfs[s_id] = orig_df
        curr_offset += n_ev

    # 6. Extract Quantitative Cohort Feature Set (Phase 5.1)
    logger.info("Extracting quantitative cohort feature tensors (counts, frequencies, MFI, GeoMFI, rSD)...")
    active_granularity = str(stats_granularity or "populations").strip().lower()
    if active_granularity not in {"populations", "clusters", "both"}:
        active_granularity = "populations"

    first_sample_cols = list(cohort_dfs[sample_ids[0]].columns) if sample_ids else []
    first_sample_col_set = set(first_sample_cols)
    resolved_ds_markers: List[str] = list(markers)
    if ds_markers is not None and not isinstance(ds_markers, str):
        explicit_ds = [m for m in ds_markers if m in first_sample_col_set]
        if m_res is not None and not explicit_ds:
            ds_sel = select_markers_from_metadata(
                marker_metadata=m_res,
                available_columns=first_sample_cols,
                explicit_markers=list(ds_markers),
                marker_roles=stats_marker_roles,
                include_markers=None,
                exclude_markers=exclude_markers,
            )
            explicit_ds = [c for c in (ds_sel or []) if c in first_sample_col_set]
        if explicit_ds:
            resolved_ds_markers = list(dict.fromkeys(explicit_ds))
    elif m_res is not None and sample_ids:
        target_ds_roles = list(stats_marker_roles) if stats_marker_roles else ["lineage", "functional"]
        ds_sel = select_markers_from_metadata(
            marker_metadata=m_res,
            available_columns=first_sample_cols,
            explicit_markers="auto",
            marker_roles=target_ds_roles,
            include_markers=include_markers,
            exclude_markers=exclude_markers,
        )
        ds_role_channels = [c for c in (ds_sel or []) if c in first_sample_col_set]
        if stats_marker_roles:
            if ds_role_channels:
                resolved_ds_markers = list(dict.fromkeys(ds_role_channels))
        else:
            resolved_ds_markers = list(dict.fromkeys([*markers, *ds_role_channels]))
        if resolved_ds_markers != list(markers):
            logger.info(
                f"Step 06 DS marker resolution (roles={target_ds_roles}): expanded {len(markers)} clustering "
                f"markers -> {len(resolved_ds_markers)} DS markers."
            )

    extraction_markers = list(dict.fromkeys([*markers, *resolved_ds_markers]))
    extractor = CohortFeatureExtractor(
        population_column="cluster_id",
        markers=extraction_markers,
    )
    cluster_feature_set = extractor.extract(
        cohort_dfs=labeled_dfs,
        sample_metadata=sample_metadata,
    )

    # Build biological population feature set (explicit cluster_name_map or verified biological annotations > Cluster_i)
    population_feature_set = copy.deepcopy(cluster_feature_set)
    effective_name_map: Optional[Dict[Union[int, str], str]] = None
    if cluster_name_map:
        effective_name_map = cluster_name_map
    elif (cell_signatures is not None or markers_file is not None or bool(marker_name_map)) and getattr(cl_res, "phenotype_labels", None):
        audit_df = getattr(cl_res, "annotation_audit_df", None)
        has_sig_match = (
            audit_df is not None
            and not audit_df.empty
            and "Annotation Source" in audit_df.columns
            and audit_df["Annotation Source"].astype(str).str.contains("signature", case=False).any()
        )
        stripped_labels = {k: _strip_cluster_prefix(v) for k, v in cl_res.phenotype_labels.items()}
        has_merged_labels = len(set(stripped_labels.values())) < len(stripped_labels)
        if has_sig_match or has_merged_labels:
            effective_name_map = stripped_labels

    if effective_name_map:
        population_feature_set, _ = _apply_cluster_rename_and_aggregate(
            feature_set=population_feature_set,
            labeled_dfs=labeled_dfs,
            umap_embedding=umap_df,
            cluster_statistics=cl_res.cluster_statistics if hasattr(cl_res, "cluster_statistics") else None,
            name_map=effective_name_map,
            logger=logger,
        )

    feature_set = cluster_feature_set if active_granularity == "clusters" else population_feature_set

    # Save feature matrices & cluster-to-population crosswalk CSV
    counts_csv = step_clustering_tables / "population_counts.csv"
    feature_set.count_matrix.to_csv(counts_csv)
    generated_tables["population_counts"] = str(counts_csv)

    freqs_csv = step_clustering_tables / "population_frequencies_viable.csv"
    feature_set.viable_freq_matrix.to_csv(freqs_csv)
    generated_tables["population_frequencies"] = str(freqs_csv)

    parent_freqs_csv = step_clustering_tables / "population_frequencies_parent.csv"
    feature_set.parent_freq_matrix.to_csv(parent_freqs_csv)
    generated_tables["population_frequencies_parent"] = str(parent_freqs_csv)

    # Save metacluster statistics & cluster-to-population crosswalk
    cl_stats_csv = step_clustering_tables / "cluster_phenotypes_summary.csv"
    cl_res.cluster_statistics.to_csv(cl_stats_csv, index=False)
    generated_tables["cluster_phenotypes"] = str(cl_stats_csv)

    crosswalk_df = _build_cluster_population_crosswalk(
        cl_res, labeled_dfs, name_map=effective_name_map
    )
    crosswalk_csv = step_clustering_tables / "cluster_to_population_map.csv"
    crosswalk_df.to_csv(crosswalk_csv, index=False)
    generated_tables["cluster_to_population_map"] = str(crosswalk_csv)

    # GraphPad Prism compatible Grouped Table Exports
    if export_prism:
        try:
            prism_files = export_prism_tables(
                feature_set=feature_set,
                output_dir=step_clustering_tables,
                condition_column=condition_column,
            )
            for k, p in prism_files.items():
                generated_tables[k] = str(p)
        except Exception as exc:
            logger.warning(f"GraphPad Prism export failed: {exc}")

    # 7. Differential Abundance (DA) and Differential State (DS) Analysis
    da_result: Optional[DAResult] = None
    ds_result: Optional[DSResult] = None
    cl_da_result: Optional[DAResult] = None
    cl_ds_result: Optional[DSResult] = None
    ref_grp: Optional[str] = None
    tgt_grp: Optional[str] = None

    if condition_column is None and feature_set.sample_metadata is not None:
        for cand_col in ("condition", "Condition", "group", "Group", "treatment", "Treatment"):
            if cand_col in feature_set.sample_metadata.columns:
                condition_column = cand_col
                logger.info(f"Step 06: Auto-resolved condition_column='{condition_column}' from sample metadata.")
                break

    if condition_column is None:
        logger.info(
            "Step 06: 'condition_column' was not specified. Skipping differential testing (DA/DS). "
            "To execute cohort statistical comparisons, declare 'condition_column' under '06_statistical_analysis' in your configuration or CLI."
        )
    elif feature_set.sample_metadata is None or condition_column not in feature_set.sample_metadata.columns:
        avail_cols = list(feature_set.sample_metadata.columns) if feature_set.sample_metadata is not None else []
        logger.warning(
            f"Step 06: Specified condition column '{condition_column}' was not found in sample metadata columns ({avail_cols}). "
            "Skipping differential testing."
        )
    else:
        cond_series = feature_set.sample_metadata[condition_column].dropna()
        unique_conds = sorted(cond_series.unique())

        if len(unique_conds) < 2:
            logger.info(
                f"Step 06: Condition column '{condition_column}' has only 1 unique group ('{unique_conds[0]}'). "
                "Differential Abundance and State testing require at least 2 distinct groups. Skipping."
            )
        else:
            ref_grp = reference_group if (reference_group and reference_group in unique_conds) else None
            tgt_grp = target_group if (target_group and target_group in unique_conds) else None
            if ref_grp is None:
                ctrl_keywords = (
                    "control", "healthy", "untreated", "vehicle", "baseline",
                    "wt", "wildtype", "pbs", "mock", "naive", "unstim", "normal", "donor",
                )
                for c in unique_conds:
                    if any(kw in str(c).lower() for kw in ctrl_keywords):
                        ref_grp = str(c)
                        break
                if ref_grp is None:
                    ref_grp = str(unique_conds[0])
            if tgt_grp is None or tgt_grp == ref_grp:
                non_ref = [str(c) for c in unique_conds if str(c) != ref_grp]
                if len(non_ref) == 1:
                    tgt_grp = non_ref[0]
                else:
                    # Select the non-reference group with the highest sample count (or last sorted group)
                    counts_by_cond = cond_series.value_counts()
                    tgt_grp = max(non_ref, key=lambda g: (int(counts_by_cond.get(g, 0)), g))

            if reference_group is None or target_group is None:
                logger.info(
                    f"Step 06: Evaluating cohort contrast across {len(unique_conds)} condition(s) "
                    f"({unique_conds}) at granularity='{active_granularity}': target '{tgt_grp}' vs reference '{ref_grp}'."
                )
            else:
                logger.info(
                    f"Step 06: Executing statistical testing at granularity='{active_granularity}' "
                    f"for contrast '{tgt_grp}' vs '{ref_grp}'..."
                )

            def _resolve_da_method(raw_method: Optional[str]) -> str:
                m = str(raw_method or "auto").strip().lower()
                if m == "auto":
                    return "wilcoxon_paired" if paired_column else "mann_whitney"
                if m == "wilcoxon":
                    return "wilcoxon_paired" if paired_column else "mann_whitney"
                aliases = {"mann_whitney_u": "mann_whitney", "ttest": "t_test"}
                m = aliases.get(m, m)
                valid_da = {"mann_whitney", "wilcoxon_paired", "t_test", "welch_ttest", "kruskal", "glm_binomial"}
                return m if m in valid_da else "mann_whitney"

            def _resolve_ds_method(raw_method: Optional[str]) -> str:
                m = str(raw_method or "auto").strip().lower()
                if m == "auto":
                    return "wilcoxon_paired" if paired_column else "mann_whitney"
                if m == "wilcoxon":
                    return "wilcoxon_paired" if paired_column else "mann_whitney"
                aliases = {"mann_whitney_u": "mann_whitney", "ttest": "t_test", "welch_ttest": "t_test"}
                m = aliases.get(m, m)
                valid_ds = {"mann_whitney", "t_test", "wilcoxon_paired"}
                return m if m in valid_ds else "mann_whitney"

            resolved_da_method = _resolve_da_method(da_test_method)
            resolved_ds_method = _resolve_ds_method(ds_test_method)

            try:
                da_cfg = DAConfig(
                    test_method=resolved_da_method,  # type: ignore[arg-type]
                    fdr_threshold=float(fdr_threshold),
                    log2fc_threshold=float(log2fc_threshold),
                )
                da_result = run_differential_abundance(
                    features=feature_set,
                    condition_column=condition_column,
                    reference_group=ref_grp,
                    target_group=tgt_grp,
                    config=da_cfg,
                )
                da_csv = step_stats_tables / "differential_abundance_summary.csv"
                da_result.summary_df.to_csv(da_csv, index=False)
                generated_tables["differential_abundance"] = str(da_csv)
            except Exception as exc:
                logger.warning(f"Differential Abundance testing encountered an error: {exc}")

            try:
                ds_cfg = DSConfig(
                    test_method=resolved_ds_method,  # type: ignore[arg-type]
                    statistic="mfi",
                    fdr_threshold=float(fdr_threshold),
                    functional_markers=resolved_ds_markers or None,
                )
                ds_result = run_differential_state(
                    features=feature_set,
                    condition_column=condition_column,
                    reference_group=ref_grp,
                    target_group=tgt_grp,
                    config=ds_cfg,
                )
                ds_csv = step_stats_tables / "differential_state_summary.csv"
                ds_result.summary_df.to_csv(ds_csv, index=False)
                generated_tables["differential_state"] = str(ds_csv)
            except Exception as exc:
                logger.warning(f"Differential State testing encountered an error: {exc}")

            if active_granularity == "both":
                cl_stats_tables_dir = step_stats_tables / "clusters"
                cl_stats_tables_dir.mkdir(parents=True, exist_ok=True)
                try:
                    cl_da_result = run_differential_abundance(
                        features=cluster_feature_set,
                        condition_column=condition_column,
                        reference_group=ref_grp,
                        target_group=tgt_grp,
                        config=DAConfig(
                            test_method=resolved_da_method,  # type: ignore[arg-type]
                            fdr_threshold=float(fdr_threshold),
                            log2fc_threshold=float(log2fc_threshold),
                        ),
                    )
                    cl_da_csv = cl_stats_tables_dir / "differential_abundance_summary.csv"
                    cl_da_result.summary_df.to_csv(cl_da_csv, index=False)
                    generated_tables["cluster_differential_abundance"] = str(cl_da_csv)
                except Exception as exc:
                    logger.debug(f"Cluster-level DA testing failed: {exc}")
                try:
                    cl_ds_result = run_differential_state(
                        features=cluster_feature_set,
                        condition_column=condition_column,
                        reference_group=ref_grp,
                        target_group=tgt_grp,
                        config=DSConfig(
                            test_method=resolved_ds_method,  # type: ignore[arg-type]
                            statistic="mfi",
                            fdr_threshold=float(fdr_threshold),
                            functional_markers=resolved_ds_markers or None,
                        ),
                    )
                    cl_ds_csv = cl_stats_tables_dir / "differential_state_summary.csv"
                    cl_ds_result.summary_df.to_csv(cl_ds_csv, index=False)
                    generated_tables["cluster_differential_state"] = str(cl_ds_csv)
                except Exception as exc:
                    logger.debug(f"Cluster-level DS testing failed: {exc}")

    # 8. Publication Diagnostic Figures (Phase 5 & 6 — Two-Tier Organized Folders)
    if generate_plots:
        logger.info("Generating organized cluster (clusters/*) and population (populations/*) diagnostic figures...")
        macro_map = None
        if hasattr(cl_res, "cluster_annotation_audit") and cl_res.cluster_annotation_audit is not None:
            audit_df = cl_res.cluster_annotation_audit
            if "Cluster ID" in audit_df.columns and "Macro Group" in audit_df.columns:
                macro_map = dict(zip(audit_df["Cluster ID"].astype(int), audit_df["Macro Group"].astype(str)))
        elif cl_res.phenotype_labels:
            macro_map = {int(k): _strip_cluster_prefix(v) for k, v in cl_res.phenotype_labels.items()}

        clean_pheno_map = {int(k): _strip_cluster_prefix(v) for k, v in (cl_res.phenotype_labels or {}).items()}
        raw_pheno_map = {int(k): str(v) for k, v in (cl_res.phenotype_labels or {}).items()}

        try:
            # clusters: Individual Cluster MFI Heatmap (C0, C1, ...)
            p_heatmap = step_clusters_plots / "cluster_expression_heatmap.png"
            plot_cluster_heatmap(
                cluster_stats_df=cl_res.cluster_statistics,
                marker_cols=cl_res.markers_used,
                ordered_indices=cl_res.ordered_indices,
                cluster_to_macro_group=macro_map,
                cluster_to_label=raw_pheno_map,
                marker_name_map=resolved_marker_map,
                cluster_palette=cluster_palette,
                output_path=str(p_heatmap),
            )
            generated_plots["cluster_expression_heatmap"] = str(p_heatmap)
        except Exception as exc:
            logger.debug(f"Cluster heatmap plotting failed: {exc}")

        try:
            # populations: Overall Identified Population MFI Heatmap (X, Y, ... merged across clusters)
            p_pop_heatmap = step_populations_plots / "population_expression_heatmap.png"
            plot_population_expression_heatmap(
                cluster_stats_df=cl_res.cluster_statistics,
                marker_cols=cl_res.markers_used,
                cluster_to_macro_group=macro_map,
                cluster_to_label=clean_pheno_map,
                marker_name_map=resolved_marker_map,
                cluster_palette=cluster_palette,
                output_path=str(p_pop_heatmap),
            )
            generated_plots["population_expression_heatmap"] = str(p_pop_heatmap)
        except Exception as exc:
            logger.debug(f"Population heatmap plotting failed: {exc}")

        try:
            # clusters: Technical Cluster QC Metrics
            p_cl_qc = step_clusters_plots / "cluster_qc_metrics.png"
            _render_cluster_qc_metrics_plot(cl_res, crosswalk_df, p_cl_qc)
            generated_plots["cluster_qc_metrics"] = str(p_cl_qc)
        except Exception as exc:
            logger.debug(f"Cluster QC metrics plotting failed: {exc}")

        try:
            # clusters: Multi-sample per-cluster frequency clustermap (C0: X, C1: X, ...)
            p_cl_freq_hm = step_clusters_plots / "cluster_frequency_clustermap.png"
            plot_cohort_heatmap(
                cluster_feature_set,
                matrix_type="frequency",
                condition_column=condition_column,
                population_name_map=raw_pheno_map,
                group_by_population=False,
                output_path=p_cl_freq_hm,
            )
            generated_plots["cluster_frequency_clustermap"] = str(p_cl_freq_hm)
        except Exception as exc:
            logger.debug(f"Cluster frequency clustermap plotting failed: {exc}")

        try:
            # populations: Multi-sample overall identified population frequency clustermap (X, Y, ...)
            p_cohort_hm = step_populations_plots / "population_frequency_clustermap.png"
            plot_cohort_heatmap(
                population_feature_set,
                matrix_type="frequency",
                condition_column=condition_column,
                population_name_map=clean_pheno_map,
                group_by_population=True,
                output_path=p_cohort_hm,
            )
            generated_plots["cohort_frequency_clustermap"] = str(p_cohort_hm)
        except Exception as exc:
            logger.debug(f"Cohort clustermap plotting failed: {exc}")

        # UMAP Embeddings (clusters vs populations + paired bridge)
        if umap_df is not None:
            umap_labels = (
                np.asarray([_strip_cluster_prefix(v) for v in umap_df["lineage_id"]])
                if "lineage_id" in umap_df.columns
                else umap_df["cluster_id"].to_numpy()
            )
            cluster_only_labels = np.asarray([f"C{c}" for c in umap_df["cluster_id"]])

            try:
                p_umap_cl = step_clusters_plots / "umap_clusters.png"
                plot_umap_projection(
                    embedding=umap_df[["UMAP_1", "UMAP_2"]].to_numpy(),
                    labels=cluster_only_labels,
                    output_path=str(p_umap_cl),
                    title=f"Cohort 2D UMAP ({clustering_method.upper()} Clusters)",
                    legend_mode="external",
                    palette=cluster_palette,
                )
                generated_plots["cohort_umap_clusters"] = str(p_umap_cl)
            except Exception as exc:
                logger.debug(f"UMAP cluster plot failed: {exc}")

            try:
                p_umap_pop = step_populations_plots / "umap_populations.png"
                plot_umap_projection(
                    embedding=umap_df[["UMAP_1", "UMAP_2"]].to_numpy(),
                    labels=umap_labels,
                    output_path=str(p_umap_pop),
                    title=f"Cohort 2D UMAP ({clustering_method.upper()} Populations)",
                    legend_mode="external" if cl_res.phenotype_labels else "centroid",
                    palette=cluster_palette,
                )
                generated_plots["cohort_umap_populations"] = str(p_umap_pop)
            except Exception as exc:
                logger.debug(f"UMAP populations plot failed: {exc}")

            try:
                p_bridge = step_clustering_plots / "cluster_to_population_bridge.png"
                plot_cluster_population_bridge(
                    embedding=umap_df[["UMAP_1", "UMAP_2"]].to_numpy(),
                    cluster_labels=cluster_only_labels,
                    population_labels=umap_labels,
                    mapping_df=crosswalk_df,
                    palette=cluster_palette,
                    output_path=p_bridge,
                )
                generated_plots["cluster_to_population_bridge"] = str(p_bridge)
            except Exception as exc:
                logger.debug(f"Cluster-to-population bridge plot failed: {exc}")

            try:
                # populations: Marker feature intensities overlaid on UMAP
                sub_marker_df = pooled_df.iloc[umap_df.index][markers].copy()
                p_umap_feats = step_populations_plots / "umap_marker_features.png"
                plot_umap_marker_features(
                    embedding=umap_df[["UMAP_1", "UMAP_2"]].to_numpy(),
                    df_features=sub_marker_df,
                    markers=markers,
                    marker_name_map=resolved_marker_map,
                    output_path=str(p_umap_feats),
                )
                generated_plots["cohort_umap_marker_features"] = str(p_umap_feats)
            except Exception as exc:
                logger.debug(f"UMAP marker feature plot failed: {exc}")

            try:
                # populations: Multi-panel UMAP Covariates
                p_umap_cov = step_populations_plots / "umap_covariates.png"
                plot_cohort_umap_covariates(
                    embedding=umap_df[["UMAP_1", "UMAP_2"]].to_numpy(),
                    sample_ids=umap_df["sample_id"].tolist(),
                    lineage_or_cluster_labels=umap_labels,
                    sample_metadata=feature_set.sample_metadata,
                    condition_column=condition_column,
                    batch_column=batch_column,
                    covariate_columns=umap_covariates,
                    cluster_palette=cluster_palette,
                    output_path=p_umap_cov,
                )
                generated_plots["cohort_umap_covariates"] = str(p_umap_cov)
            except Exception as exc:
                logger.debug(f"Cohort UMAP covariates plot failed: {exc}")

            try:
                # populations: Stratified UMAP Density Contours
                if (
                    stratified_density_contours
                    and condition_column
                    and feature_set.sample_metadata is not None
                    and condition_column in feature_set.sample_metadata.columns
                ):
                    p_strat = step_populations_plots / "umap_stratified_contours.png"
                    plot_stratified_umap_density_contours(
                        embedding=umap_df[["UMAP_1", "UMAP_2"]].to_numpy(),
                        sample_ids=umap_df["sample_id"].tolist(),
                        sample_metadata=feature_set.sample_metadata,
                        condition_column=condition_column,
                        output_path=p_strat,
                    )
                    generated_plots["cohort_umap_stratified_contours"] = str(p_strat)
            except Exception as exc:
                logger.debug(f"Stratified UMAP density contours plot failed: {exc}")
        else:
            # Non-UMAP bridge plot fallback (bar crosswalk + mapping table)
            try:
                p_bridge = step_clustering_plots / "cluster_to_population_bridge.png"
                all_cids = [f"C{c}" for c in cl_res.cluster_labels]
                all_pops = [
                    _strip_cluster_prefix(cl_res.phenotype_labels.get(c, f"Cluster_{c}"))
                    if cl_res.phenotype_labels
                    else f"Cluster_{c}"
                    for c in cl_res.cluster_labels
                ]
                plot_cluster_population_bridge(
                    embedding=None,
                    cluster_labels=all_cids,
                    population_labels=all_pops,
                    mapping_df=crosswalk_df,
                    palette=cluster_palette,
                    output_path=p_bridge,
                )
                generated_plots["cluster_to_population_bridge"] = str(p_bridge)
            except Exception as exc:
                logger.debug(f"Fallback bridge plot failed: {exc}")

        # Step 6: Differential Abundance Plots
        if da_result is not None:
            if generate_volcano_plot:
                try:
                    p_volcano = step_stats_plots / "differential_abundance_volcano.png"
                    plot_volcano_da(da_result, output_path=p_volcano)
                    generated_plots["differential_abundance_volcano"] = str(p_volcano)
                except Exception as exc:
                    logger.debug(f"DA volcano plot failed: {exc}")

            if generate_forest_plot:
                try:
                    p_forest = step_stats_plots / "differential_abundance_forest.png"
                    plot_forest_da(da_result, output_path=p_forest)
                    generated_plots["differential_abundance_forest"] = str(p_forest)
                except Exception as exc:
                    logger.debug(f"DA forest plot failed: {exc}")

            if generate_abundance_boxplots:
                try:
                    p_boxplots = step_stats_plots / "population_abundance_boxplots.png"
                    plot_population_abundance_boxplots(
                        features=feature_set,
                        condition_column=condition_column or "condition",
                        output_path=p_boxplots,
                    )
                    generated_plots["population_abundance_boxplots"] = str(p_boxplots)
                except Exception as exc:
                    logger.debug(f"Abundance boxplot failed: {exc}")

        # Differential State (DS) Modulation Heatmap
        if ds_result is not None and generate_ds_heatmap:
            try:
                p_ds_hm = step_stats_plots / "differential_state_heatmap.png"
                plot_differential_state_heatmap(ds_result, output_path=p_ds_hm)
                generated_plots["differential_state_heatmap"] = str(p_ds_hm)
            except Exception as exc:
                logger.debug(f"DS heatmap plot failed: {exc}")

        # Optional cluster-resolution Step 6 plots when stats_granularity == 'both'
        if active_granularity == "both" and cl_da_result is not None:
            cl_stats_plots_dir = step_stats_plots / "clusters"
            cl_stats_plots_dir.mkdir(parents=True, exist_ok=True)
            if generate_volcano_plot:
                try:
                    p_cl_vol = cl_stats_plots_dir / "differential_abundance_volcano.png"
                    plot_volcano_da(cl_da_result, output_path=p_cl_vol)
                    generated_plots["cluster_differential_abundance_volcano"] = str(p_cl_vol)
                except Exception as exc:
                    logger.debug(f"Cluster DA volcano plot failed: {exc}")
            if generate_forest_plot:
                try:
                    p_cl_for = cl_stats_plots_dir / "differential_abundance_forest.png"
                    plot_forest_da(cl_da_result, output_path=p_cl_for)
                    generated_plots["cluster_differential_abundance_forest"] = str(p_cl_for)
                except Exception as exc:
                    logger.debug(f"Cluster DA forest plot failed: {exc}")
            if cl_ds_result is not None and generate_ds_heatmap:
                try:
                    p_cl_ds = cl_stats_plots_dir / "differential_state_heatmap.png"
                    plot_differential_state_heatmap(cl_ds_result, output_path=p_cl_ds)
                    generated_plots["cluster_differential_state_heatmap"] = str(p_cl_ds)
                except Exception as exc:
                    logger.debug(f"Cluster DS heatmap plot failed: {exc}")

    # 9. Step 07: Workspaces & Reporting (Mandatory Step)
    augmented_fcs_paths: List[Path] = []
    aug_fcs_info: Dict[str, Tuple[Path, List[str], float]] = {}
    if export_augmented_fcs:
        logger.info(f"Exporting augmented FCS files to '{step_fcs.name}/'...")
        for s_id, df in labeled_dfs.items():
            try:
                out_fcs_path = step_fcs / f"augmented_{s_id}.fcs"
                # Export numeric channels into FCS
                numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
                write_fcs(
                    output_path=out_fcs_path,
                    events_df=df[numeric_cols],
                    pnn_labels=numeric_cols,
                )
                augmented_fcs_paths.append(out_fcs_path)
                vals = df[numeric_cols].to_numpy(dtype=np.float64, copy=False)
                finite_vals = vals[np.isfinite(vals)]
                max_rng = float(np.max(finite_vals)) if finite_vals.size > 0 else 262144.0
                aug_fcs_info[s_id] = (out_fcs_path, numeric_cols, max(max_rng, 25.0))
            except Exception as exc:
                logger.debug(f"Failed exporting augmented FCS for {s_id}: {exc}")

    wsp_file = step_workspaces / "flowjo_workspace.wsp"
    gml_file = step_workspaces / "gating_ml_2.0.xml"
    if export_flowjo_wsp or export_gating_ml:
        try:
            from flowx.gating.gate import GatingHierarchy, RangeGate
            from flowx.workspace.gating_ml_export import (
                SampleWorkspaceSpec,
                export_gating_ml as _export_gml,
                export_preprocessing_flowjo_wsp as _export_wsp_v10,
            )

            # Build per-cluster RangeGates on numeric `cluster_id` channel ([cid - 0.5, cid + 0.5])
            cid_to_label: Dict[int, str] = {}
            for df in labeled_dfs.values():
                if "cluster_id" not in df.columns:
                    continue
                has_lin = "lineage_id" in df.columns
                for raw_cid in df["cluster_id"].dropna().unique():
                    cid = int(raw_cid)
                    if cid in cid_to_label:
                        continue
                    if has_lin:
                        modes = df.loc[df["cluster_id"] == raw_cid, "lineage_id"].dropna().mode()
                        label = str(modes.iloc[0]) if not modes.empty else f"Cluster_{cid}"
                    else:
                        label = f"Cluster_{cid}"
                    cid_to_label[cid] = label

            if not cid_to_label:
                for idx_p, pop_name in enumerate(feature_set.population_names):
                    cid_to_label[idx_p] = str(pop_name)

            label_counts: Dict[str, int] = {}
            for lbl in cid_to_label.values():
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            discovery_hierarchy = GatingHierarchy()
            cid_to_gate_name: Dict[int, str] = {}
            for cid in sorted(cid_to_label):
                lbl = cid_to_label[cid]
                gate_name = f"{lbl}_C{cid}" if label_counts[lbl] > 1 else lbl
                cid_to_gate_name[cid] = gate_name
                discovery_hierarchy.add_gate(
                    RangeGate(
                        name=gate_name,
                        channel="cluster_id",
                        min_val=float(cid) - 0.5,
                        max_val=float(cid) + 0.5,
                    )
                )

            exp_name = f"Cohort_{clustering_method.upper()}_Workspace"
            if export_flowjo_wsp:
                aug_specs: List[SampleWorkspaceSpec] = []
                for s_id, df in labeled_dfs.items():
                    fcs_path_obj, num_cols, max_rng = aug_fcs_info.get(
                        s_id,
                        (
                            step_fcs / f"augmented_{s_id}.fcs",
                            [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)],
                            262144.0,
                        ),
                    )
                    s_counts: Dict[str, int] = {}
                    if "cluster_id" in df.columns:
                        vc = df["cluster_id"].value_counts().to_dict()
                        for cid, gname in cid_to_gate_name.items():
                            s_counts[gname] = int(vc.get(cid, vc.get(float(cid), 0)))
                    aug_specs.append(
                        SampleWorkspaceSpec(
                            sample_filename=fcs_path_obj.name,
                            sample_path=str(fcs_path_obj) if fcs_path_obj.exists() else "",
                            total_events=len(df),
                            channels=num_cols,
                            max_range=max_rng,
                            hierarchy=discovery_hierarchy,
                            gate_counts=s_counts,
                        )
                    )
                _export_wsp_v10(aug_specs, wsp_file, experiment_name=exp_name)
                generated_tables["flowjo_workspace"] = str(wsp_file)
            if export_gating_ml:
                _export_gml(discovery_hierarchy, gml_file, experiment_name=exp_name)
                generated_tables["gating_ml"] = str(gml_file)
        except Exception as exc:
            logger.debug(f"Workspace export failed: {exc}")

    elapsed = time.perf_counter() - t_start
    dur_str = format_duration(elapsed)

    cohort_result_obj = CohortPopulationResult(
        samples=sample_ids,
        labeled_dfs=labeled_dfs,
        clustering_result=cl_res,
        feature_set=feature_set,
        da_result=da_result,
        ds_result=ds_result,
        umap_embedding=umap_df,
        condition_column=condition_column,
        batch_column=batch_column,
        umap_covariates=umap_covariates,
        cluster_palette=cluster_palette,
        reference_group=reference_group,
        target_group=target_group,
        paired_column=paired_column,
        generated_plots=generated_plots,
        generated_tables=generated_tables,
        output_dir=str(out_base),
        execution_time_seconds=round(elapsed, 2),
        cluster_feature_set=cluster_feature_set,
        stats_granularity=active_granularity,
        ds_markers=resolved_ds_markers,
    )

    # Compile MIFlowCyt-Compliant Markdown & HTML Reports
    if generate_miflowcyt_report:
        try:
            from flowx.workspace.miflowcyt_report import generate_miflowcyt_report as _gen_miflow

            rep_dict = _gen_miflow(
                output_dir=step_reports,
                experiment_name=f"FlowX {clustering_method.upper()} Cohort Discovery",
                cohort_result=cohort_result_obj,
                samples=sample_ids,
                total_events=int(feature_set.count_matrix.values.sum()),
                markers=markers,
                marker_name_map=resolved_marker_map,
                clustering_method=clustering_method,
                cluster_count=cl_res.cluster_count,
                auto_k_method=(auto_k_method if auto_k else "fixed"),
                da_result=da_result,
                ds_result=ds_result,
                flowjo_wsp_path=wsp_file if export_flowjo_wsp else None,
                gating_ml_path=gml_file if export_gating_ml else None,
                preprocessing_wsp_path=preprocessing_wsp_path,
                preprocessing_gating_ml_path=preprocessing_gating_ml_path,
                augmented_fcs_paths=augmented_fcs_paths,
                pipeline_step="exports",
                numbered_steps=numbered_steps,
            )
            generated_tables["miflowcyt_report_md"] = str(rep_dict["markdown"])
            generated_tables["miflowcyt_report_html"] = str(rep_dict["html"])
        except Exception as exc:
            logger.debug(f"MIFlowCyt report generation failed: {exc}")

    logger.info(
        f"Cohort Population Analysis completed in {dur_str} ({elapsed:.2f}s). "
        f"Results saved to '{out_base}'."
    )

    return cohort_result_obj
