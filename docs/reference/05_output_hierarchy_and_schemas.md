# Reference: Output Hierarchy & Table Schemas

FlowX organizes every pipeline and step-runner execution into a deterministic, numbered directory tree under `outputs/config_run/<platform>/<dataset_name>/` (or `outputs/<platform>/<dataset_name>/` for ad-hoc CLI runs). Every run preserves both the user's input configuration (`config.yaml`) and the effective resolved parameters (`resolved_config.yaml`).

---

## Standard Output Directory Tree

```text
outputs/config_run/<platform>/<dataset_name>/
├── config.yaml                          # Input YAML snapshot
├── resolved_config.yaml                 # Effective parameters after metadata & auto-resolution
├── pipeline.log                         # Timestamped execution log
│
├── 01_time_qc/
│   ├── plots/<sample>_time_qc_diagnostics.png
│   └── tables/
│
├── 02_compensation/
│   ├── plots/                           # Spillover / unmixing & AF diagnostics (when generated)
│   └── tables/                          # Estimated or extracted spillover matrices (when generated)
│
├── 03_cleanup_gating/
│   ├── plots/<sample>_cleanup_diagnostics.png
│   └── tables/
│
├── 04_batch_preparation/
│   ├── batch_necessity_report.md        # Markdown summary of batch verdict & CV/MMD/EMD checks
│   ├── plots/
│   │   ├── batch_necessity_summary.png
│   │   ├── step3_batch_density_comparison.png
│   │   ├── step3_batch_metrics_reduction.png
│   │   └── step3_cohort_downsampling_balance.png
│   └── tables/
│       ├── batch_invariance_metrics.csv
│       ├── batch_necessity_assessment.json
│       ├── batch_necessity_metrics.csv
│       ├── cohort_batch_summary.csv
│       └── downsampling_summary.csv
│
├── 05_population_discovery/
│   ├── plots/
│   │   ├── cluster_to_population_bridge.png
│   │   ├── clusters/
│   │   │   ├── cluster_expression_heatmap.png
│   │   │   ├── cluster_frequency_clustermap.png
│   │   │   ├── cluster_qc_metrics.png
│   │   │   └── umap_clusters.png
│   │   └── populations/
│   │       ├── population_expression_heatmap.png
│   │       ├── population_frequency_clustermap.png
│   │       ├── umap_covariates.png
│   │       ├── umap_marker_features.png
│   │       ├── umap_populations.png
│   │       └── umap_stratified_contours.png
│   └── tables/
│       ├── cluster_annotation_audit.csv
│       ├── cluster_marker_thresholds.csv
│       ├── cluster_phenotypes_summary.csv
│       ├── cluster_to_population_map.csv
│       ├── metacluster_k_metrics.csv
│       ├── population_counts.csv
│       ├── population_frequencies_parent.csv
│       ├── population_frequencies_viable.csv
│       └── prism/                               # (untested — reserved for future release)
│           └── prism_marker_mfi_<channel>.csv   # (untested — reserved for future release)
│
├── 06_statistical_analysis/
│   ├── plots/
│   │   ├── differential_abundance_forest.png
│   │   ├── differential_abundance_volcano.png
│   │   ├── differential_state_heatmap.png
│   │   └── population_abundance_boxplots.png
│   └── tables/
│       ├── differential_abundance_summary.csv
│       └── differential_state_summary.csv
│
└── exports/
    ├── augmented_fcs/
    │   └── augmented_<sample>.fcs       # Clean events + numeric cluster_id, UMAP_1, UMAP_2
    ├── reports/
    │   ├── miflowcyt_analysis_report.html
    │   ├── miflowcyt_analysis_report.md
    │   └── report.md
    └── workspaces/
        ├── flowjo_workspace.wsp                 # (untested — reserved for future release)
        ├── gating_ml_2.0.xml
        ├── preprocessing_gating.wsp             # (untested — reserved for future release)
        ├── preprocessing_gating_ml_2.0.xml
        ├── workspace.json
        └── workspace.xml
```

> [!NOTE]
> Inside `05_population_discovery/plots/`, plots are strictly separated into `plots/clusters/` (unsupervised SOM/PhenoGraph clusters) and `plots/populations/` (annotated biological lineages) without `01_`/`02_` or `cl_`/`pop_` filename prefixes.

---

## Core Output Table Schemas

### `04_batch_preparation/tables/batch_necessity_assessment.json`
Machine-readable batch verdict and rationale serialized from `BatchNecessityAssessment` (`assess_batch_effect_need()`):

| Key | Type | Description |
| :--- | :--- | :--- |
| `recommendation` | `str` | One of `NOT_NEEDED`, `NOT_RECOMMENDED_CONFOUNDED`, `CAUTION_ANCHOR_FREE`, `RECOMMENDED_CYTONORM`. |
| `confidence` | `float` | Assessment confidence in `[0.0, 1.0]`. |
| `drift_score` | `float` | Composite `0–100` inter-batch drift score. |
| `batch_count` | `int` | Distinct `batch_id` count across cohort samples. |
| `unique_batches` | `list[str]` | Sorted list of unique batch identifiers. |
| `has_control_anchors` | `bool` | Whether `is_control: true` bridge anchors span $\ge 2$ batches. |
| `control_batches` | `list[str]` | Batches containing bridge anchor controls. |
| `sample_count` | `int` | Total cohort samples evaluated. |
| `confounded_with_condition` | `bool` | True when batch and condition assignment are completely collinear/disjoint. |
| `rationale` | `list[str]` | Ordered narrative explanation items rendered in `batch_necessity_report.md`. |
| `marker_metrics` | `list[dict]` | Per-marker EMD, KS, median ratio, and severity records. |

### `05_population_discovery/tables/metacluster_k_metrics.csv`
Cluster validation sweep evaluated across candidate $k$ values:

| Column | Type | Description |
| :--- | :--- | :--- |
| `k` | `int` | Candidate metacluster count. |
| `<metric_name>` | `float` | Optimization metric column (`consensus_pac`, `silhouette_score`, `davies_bouldin_index`, `calinski_harabasz_score`, or `wcss_inertia`). |
| `is_optimal` | `bool` | Whether this $k$ was selected as optimal for downstream annotation. |

### `05_population_discovery/tables/cluster_annotation_audit.csv`
Transparent audit trail mapping each raw cluster ID to its biological lineage label:

| Column | Type | Description |
| :--- | :--- | :--- |
| `Cluster ID` | `int` | Unsupervised FlowSOM metacluster or PhenoGraph community ID. |
| `Assigned Label` | `str` | Assigned canonical lineage label (e.g., `CD4+ Naive T Cells`) or empirical profile (`C0: ...`). |
| `Macro Group` | `str` | Top-level biological lineage compartment. |
| `Annotation Source` | `str` | Rule source (`signature_rule:...`, `user_cluster_name_map`, `empirical_markers`, etc.). |
| `Confidence` | `str` | Constraint satisfaction string (e.g., `1.00 (4/4 constraints)`). |
| `Positive Markers` | `str` | Semicolon-delimited markers exceeding Otsu/positivity cutoffs. |
| `Negative Markers` | `str` | Semicolon-delimited markers below positivity cutoffs. |

### `05_population_discovery/tables/population_frequencies_viable.csv` & `population_counts.csv`
Wide-format sample $\times$ population matrices ready for statistical testing and external plotting:
- Rows: `sample_id` (plus merged `condition` and `batch_id` covariates).
- Columns: one column per annotated biological population (percentage of total viable singlets in `population_frequencies_viable.csv`, percentage of parent lineage in `population_frequencies_parent.csv`, or integer cell counts in `population_counts.csv`).

### `06_statistical_analysis/tables/differential_abundance_summary.csv`
Population-level DA hypothesis test results:

| Column | Type | Description |
| :--- | :--- | :--- |
| `population` | `str` | Biological population name (or cluster label). |
| `mean_ref_pct` / `median_ref_pct` | `float` | Reference group mean and median population frequency (`%`). |
| `mean_target_pct` / `median_target_pct` | `float` | Target group mean and median population frequency (`%`). |
| `delta_pct` | `float` | Frequency shift (`target - reference`, in percentage points). |
| `log2fc` | `float` | $\log_2$ ratio of group means with `0.01` pseudocount. |
| `p_value` | `float` | Raw two-sided hypothesis test $p$-value. |
| `q_value` | `float` | Benjamini-Hochberg FDR adjusted $q$-value. |
| `is_significant` | `bool` | Passes `q_value < fdr_threshold` and `|log2fc| >= log2fc_threshold`. |
| `direction` / `total_events` | `str` / `int` | Effect direction (`Enriched` / `Depleted` / `Unchanged`) and pooled population cell count. |

### `06_statistical_analysis/tables/differential_state_summary.csv`
Within-population functional marker DS results:

| Column | Type | Description |
| :--- | :--- | :--- |
| `population` | `str` | Biological population name (or cluster label). |
| `marker` | `str` | Marker tested (`marker_roles` from `06_statistical_analysis`). |
| `mean_ref` / `median_ref` | `float` | Reference group mean and median per-sample MFI/GeoMFI. |
| `mean_target` / `median_target` | `float` | Target group mean and median per-sample MFI/GeoMFI. |
| `delta_stat` | `float` | Difference in group means (`target - reference`). |
| `cohens_d` | `float` | Standardized effect size between conditions. |
| `p_value` | `float` | Raw two-sided $p$-value. |
| `q_value` | `float` | Benjamini-Hochberg adjusted $q$-value. |
| `is_significant` / `direction` | `bool` / `str` | FDR significance flag and direction (`Up` / `Down` / `Unchanged`). |

---

## References & Scientific Standards

* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
* **Gating-ML 2.0 Standard:** [](doi:10.1002/cyto.a.22690)
* **Benjamini-Hochberg FDR Control:** [](doi:10.1111/j.2517-6161.1995.tb02031.x)
