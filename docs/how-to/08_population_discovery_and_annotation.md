# 08. Population Discovery, Auto-$k$ & Phenotypic Annotation (`flowx cohort`)

Configure FlowSOM (MiniSom + Consensus PAC) or PhenoGraph clustering, customize biological signature rules, separate technical cluster plots from biological population plots, and rename clusters cleanly.

---

## 1. Two-Tier Plot Organization (`clusters/` vs `populations/`)

Inside `05_population_discovery/`, FlowX separates **unsupervised technical artifacts** from **annotated biological lineages**:
- `plots/clusters/`: `cluster_qc_metrics.png`, `cluster_expression_heatmap.png`, `cluster_frequency_clustermap.png`, `umap_clusters.png` (using `cluster_palette: tab20` by default).
- `plots/populations/`: `population_expression_heatmap.png`, `population_frequency_clustermap.png`, `umap_populations.png`, `umap_marker_features.png`, `umap_covariates.png`, `umap_stratified_contours.png`.
- `plots/cluster_to_population_bridge.png` & `tables/cluster_to_population_map.csv`: Reconciles raw `Cluster_ID`s to merged `Annotated_Population`s.

---

## 2. Configure Clustering & Auto-$k$ on PBMC Cohorts

**Examples — `spectral/kreutmair_2026_icu_pbmc` & `spectral/michaelis_2025_c9orf72_als_pbmc`:**
Both PBMC cohorts map 100% of clusters (`22` and `20` respectively) into **10 canonical PBMC subsets** with **0 unassigned clusters**.

**YAML (`05_population_discovery`):**
```yaml
05_population_discovery:
  enabled: true
  lineage_markers: auto
  marker_roles: ["lineage"]      # Or ["lineage", "functional"] (e.g. crespo_2026_crispr_cd8_tcells)
  clustering_method: som         # som | phenograph
  auto_k: true
  auto_k_method: consensus       # consensus (PAC) | davies_bouldin | silhouette | calinski_harabasz | elbow
  k_min: 6
  k_max: 22
  consensus_subsample: 0.8
  consensus_iterations: 50
  compute_umap: true
  umap_events_per_sample: 1000
  cluster_palette: tab20
```

**Standalone CLI (`flowx cohort`):**
```bash
flowx cohort -c configs/config_run/spectral/kreutmair_2026_icu_pbmc.yaml \
  --auto-k-method consensus --k-min 6 --k-max 22
```

---

## 3. Custom Cell Signatures & Post-Hoc Cluster Renaming

1. **Declarative Signature Rules (`--cell-signatures` / `cell_signatures`):**
   Supply a JSON or YAML list of `{name, macro_group, required_positive, required_negative, exclude_if_positive, priority}` rules. Rules only fire when **all** required markers are measured in `markers.csv`.
2. **Explicit Overrides (`cluster_name_map` / `--cluster-name-map`):**
   ```yaml
   05_population_discovery:
     cluster_name_map:
       "0": "CD4+ Naive T Cells"
       "1": "Classical Monocytes (CD14+ CD16-)"
   ```
3. **Python Post-Hoc Renaming:**
   Call `res.rename_clusters({0: "CD4+ T Cells", 1: "CD4+ T Cells"})` on `CohortPopulationResult` to automatically regenerate `population_counts.csv`, `cluster_to_population_map.csv`, `cluster_to_population_bridge.png`, `plots/populations/`, and Step 06 DA/DS tables.

---

## References & Scientific Standards

* **FlowSOM Clustering:** [](doi:10.1002/cyto.a.22625)
* **PhenoGraph Algorithm:** [](doi:10.1016/j.cell.2015.05.047)
* **UMAP Dimensionality Reduction:** [](doi:10.48550/arXiv.1802.03426)
