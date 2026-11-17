# 04. Python API & Unified Step Runners Reference

Authoritative reference covering **both** the high-level pipeline/step-runner APIs and core algorithmic classes in `src/flowx/`.

---

## 1. High-Level Pipeline & Config API

### `PipelineConfig` & `FlowCytometryPipeline` (`flowx.io.config`, `flowx.pipeline`)

```python
from flowx.io.config import PipelineConfig

cfg = PipelineConfig.load("configs/config_run/conventional/finak_2014_cytotrol_pbmc.yaml")
cfg.validate()
pipeline, input_path = cfg.to_pipeline()
result = pipeline.run(input_path)
```

**Canonical Step Config Builders on `PipelineConfig`:**
- `cfg.to_time_qc_config() -> TimeQCConfig` (uses `sample_type` from `01_time_qc`, or call `TimeQCConfig.for_controls()` / `TimeQCConfig.for_fmo()`)
- `cfg.to_compensation_config() -> CompensationConfig`
- `cfg.to_unmixing_config() -> UnmixingConfig`
- `cfg.to_transform_config() -> TransformConfig`
- `cfg.to_cleanup_config() -> CleanUpGateConfig`
- `cfg.to_batch_preparation_config(output_dir=None) -> BatchPreparationConfig`

---

## 2. Unified Independent Step Runners (`*Config` $\to$ `run_<step>_step` $\to$ `*Result`)

| Step | Module | Runner Function(s) | Config & Result Dataclasses |
| :--- | :--- | :--- | :--- |
| **01. Time QC** | `flowx.qc.runner` | `run_time_qc_step()`, `run_cohort_time_qc()` | `TimeQCConfig`, `TimeQCResult`, `TimeQCStepResult` |
| **02A. Compensation** | `flowx.math.compensation` | `run_compensation_step()`, `run_dataset_compensation()` | `CompensationConfig`, `CompensationMatrix`, `CompensationStepResult` |
| **02B. Spectral Unmixing** | `flowx.spectral.runner` | `run_spectral_unmixing_pipeline()`, `unmix_dataframe()` | `UnmixingConfig`, `ReferenceSpectralLibrary`, `UnmixingResult` |
| **03A. Transformation** | `flowx.math.transforms` | `run_transformation_step()`, `resolve_fluorescence_channels()` | `TransformConfig`, `TransformStepResult` |
| **03B. Cleanup Gating** | `flowx.gating.runner` | `run_cleanup_step()`, `run_cohort_cleanup()` | `CleanUpGateConfig`, `CleanUpGatingResult`, `CleanupStepResult` |
| **04B. Batch Prep** | `flowx.batch.runner` | `run_batch_preparation()`, `assess_batch_effect_need()` | `BatchPreparationConfig`, `BatchCorrectionConfig`, `DownsampleConfig`, `CohortBatchResult`, `BatchNecessityAssessment` |
| **05–06. Discovery & Stats** | `flowx.clustering.cohort_runner` | `run_cohort_population_analysis()` | `CohortPopulationResult` (supports `.rename_clusters({...})`) |

---

## 3. Core Algorithmic Classes

### 3.1 Scale Transformations (`flowx.math.transforms`)
All transforms inherit from `BaseTransform` (`apply(values)`, `inverse(values)`, `to_dict()`):
- `LogicleTransform(param_t=262144.0, param_w=0.5, param_m=4.5, param_a=0.0)`
- `BiexponentialTransform(max_value=262144.0, positive=4.418540, width_basis=-10.0, negative=0.0, channel_range=4096, normalized=True)`
- `AsinhTransform(cofactor=150.0, max_value=262144.0, normalized=False)`
- `LogTransform(floor=1.0, base=10.0)`
- `PreTransformedScaleTransform(low_val=-1.0, high_val=4.5)`

### 3.2 Hierarchical Gates, FMO & Workspaces (`flowx.gating.gate`, `flowx.gating.fmo`, `flowx.workspace`)
All gates inherit from `Gate` (`apply(data) -> np.ndarray`, `to_dict()`, `from_dict()`):
- `PolygonGate(name, channels, vertices, parent_gate=None)`
- `GMMClusterGate(name, channels, means, covariances, weights, cell_clusters, scale_diag, debris_sd_scale=1.0, debris_min_probability=None, ...)`
- `RangeGate(name, channel, min_val=None, max_val=None, parent_gate=None)`
- `RectangleGate(name, channels, x_min, x_max, y_min, y_max, parent_gate=None)`
- `QuadrantGate(name, channels, x_cutoff, y_cutoff, quadrant="all", parent_gate=None)`
- `GatingHierarchy` — directed parent-child tree supporting `.add_gate()`, `.evaluate(df, transformed_df)`, and `.get_population_statistics()`.
- Data-assisted & FMO helpers: `find_density_minimum()`, `find_quadrant_cutoffs()` (`flowx.gating.density_gating`), and `calibrate_fmo_thresholds()` (`flowx.gating.fmo`).
- Workspace I/O & replay: `WorkspaceSession` (`flowx.workspace.session`) and `export_gating_ml()` (`flowx.workspace.gating_ml_export`).

### 3.3 Batch Normalization & Clustering Models (`flowx.batch.normalization`, `flowx.clustering.cluster`)
- `QuantileSplineMapping(cluster_id, marker, batch_id, input_quantiles, goal_quantiles, method="pchip")`
- `CytoNormModel(markers, batches, n_clusters, centroids, splines, goal="median")` — `.transform(df, batch_id)`, `.save_json(path)`, `.load_json(path)`
- `QuantileNormalizer(markers, batches, mappings, goal="median")` — `.transform(df, batch_id)`
- `ClusteringResult` — holds `cluster_labels`, `cluster_count`, `cluster_statistics`, `phenotype_labels`, `macro_groups_df`, `k_optimization_scores`, `annotation_audit_df`, `marker_thresholds`.

### 3.4 Cohort Statistics Models (`flowx.stats.models`, `flowx.stats.metrics`)
- `CohortFeatureSet` — holds `count_matrix`, `viable_freq_matrix`, `parent_freq_matrix`, `mfi_tensor`, `geomfi_tensor`, `rsd_tensor`.
- `DAConfig`, `DAResult`, `DSConfig`, `DSResult` — produced by `run_differential_abundance()` and `run_differential_state()`.
- Metric primitives: `median_fluorescence_intensity()`, `geometric_mean(values, pseudocount=1.0)`, `robust_sd(values, method="mad"|"iqr")`, `log2_fold_change()`, `cohens_d()`, `fdr_correction(p_values, method="bh"|"bonferroni")`.

---

## References & Scientific Standards

* **FlowKit Computational Toolkit:** [](doi:10.3389/fimmu.2021.768541)
* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
