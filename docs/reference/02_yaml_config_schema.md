# 02. Unified YAML Configuration Schema (`PipelineConfig`)

`PipelineConfig` (`src/flowx/io/config.py`, **114 dataclass fields**) parses hierarchical numbered YAML/JSON configurations, forwards parameters to each `to_<step>_config()` builder, and writes `<output_dir>/resolved_config.yaml`.

---

## Complete Numbered YAML Schema

```yaml
input:
  fcs_file: null                           # File, directory, or samples.csv (or CLI target)
  data_dir: null                           # Alternative cohort directory path
  files_list: null                         # Optional explicit list or .txt/.csv path
  technology: auto                         # auto | conventional | spectral
  metadata_file: null                      # Auto-discovers <dataset>/metadata/samples.csv
  markers_file: null                       # Auto-discovers <dataset>/metadata/markers.csv
  harmonization_file: null                 # Optional panel harmonization YAML
  enable_harmonization: false

analysis:
  gating_mode: som                         # som | template | phenograph | compare
  cluster_parent: viable                   # viable | singlet | cells | all
  qc_method: clustering                    # clustering | template
  adaptive_gating: false                   # True recommended in cohort presets
  enable_qc: true
  enable_discovery: false                  # True recommended for end-to-end discovery
  numbered_steps: true
  seed: 42
  log_level: INFO
  umap_subsample: 20000

visualization:
  legend_mode: centroid                    # centroid | external

01_time_qc:
  enabled: true
  bin_width: null                          # Float seconds (step default: 0.5 sample, 0.25 control)
  enable_signal_drift: true
  trim_margins: true
  manual_exclude_intervals: null           # e.g. [[12.0, 18.5]]

02_compensation:
  enabled: true
  source: auto                             # auto | controls | embedded | csv | none
  method: hybrid                           # hybrid | median_delta_mfi | autospill_regression
  controls_dir: null
  unstained_control: null
  voltage_mismatch_policy: warn            # warn | fallback_embedded | scale_power_law | strict_error
  voltage_tolerance_v: 5.0
  # Spectral-only keys (when technology: spectral):
  unmixing_method: wls                     # wls | ols | nnls
  autofluorescence_mode: single            # single | multi | none
  multi_af_components: 2
  alpha_poisson: 1.0
  electronic_noise: 50.0

03_cleanup_gating:
  enabled: true
  debris_method: gmm                       # gmm | adaptive | manual
  debris_n_components: 2                   # PipelineConfig default: 2 (CleanUpGateConfig/PBMC preset: 3)
  debris_sd_scale: 1.0                     # >1.0 loosens, <1.0 tightens
  debris_min_probability: null             # Optional posterior floor [0.05..0.90]
  singlet_mode: dual                       # dual | single | fsc_only | ssc_only | none
  singlet_method: gmm_ratio                # gmm_ratio | envelope
  singlet_tolerance: 0.25                  # [0.05..0.85]
  fsc_channel: null                        # null or "auto"
  ssc_channel: null
  fsc_height_channel: null
  ssc_height_channel: null
  enable_viability: true
  viability_channel: null                  # null or "auto"
  viability_cutoff: null
  viability_looseness: 1.0
  dump_enabled: false
  dump_channel: null
  dump_cutoff: null
  dump_looseness: 1.0
  export_clean_fcs: false

04_transformation:
  enabled: true
  method: logicle                          # logicle | biexponential | arcsinh | log
  cofactor: 5.0                            # Cohort preset: arcsinh with 150-500 (conv) | 2000 (spectral)
  channels: null                           # null or "auto" (excludes NON_FLUORESCENCE_MARKER_CLASSES)
  exclude_channels: null

04_batch_preparation:
  enabled: true
  assess_batch_effect: true                # Auto-check enabled by default
  overwrite_autocheck: false               # Safety gate against confounded/unnecessary norm
  normalization_method: none               # none | cytonorm | quantile | combat | auto
  batch_column: batch_id
  goal: median                             # median | mean | reference
  n_clusters: 10
  n_quantiles: 101
  mode: min_sample                         # min_sample | target_count | stratified | proportional | none
  target_events: 50000
  min_events_threshold: 2500

05_population_discovery:
  enabled: false                           # True when analysis.enable_discovery is true
  lineage_markers: null                    # "auto" recommended when markers.csv is present
  marker_roles: null                       # Defaults to ["lineage"] when resolving markers.csv
  include_markers: null
  exclude_markers: null
  cell_signatures: null                    # Optional custom JSON/YAML signature rules
  clustering_method: som                   # som | phenograph
  n_clusters: 10
  auto_k: true
  auto_k_method: consensus                 # consensus | davies_bouldin | silhouette | calinski_harabasz | elbow
  k_min: 4
  k_max: 20
  consensus_subsample: 0.8
  consensus_iterations: 50
  compute_umap: true
  umap_events_per_sample: 1000
  umap_covariates: null
  stratified_density_contours: true
  cluster_palette: tab20
  cluster_name_map: null

06_statistical_analysis:
  enabled: true
  condition_column: null                   # Auto-resolves "condition" when samples.csv is present
  reference_group: null
  target_group: null
  paired_column: null
  da_test_method: auto                     # auto | mann_whitney | t_test | welch_ttest | wilcoxon_paired | kruskal | glm_binomial
  ds_test_method: auto                     # auto | mann_whitney | t_test | wilcoxon_paired
  fdr_threshold: 0.05
  log2fc_threshold: 1.0
  stats_granularity: populations           # populations | clusters | both
  marker_roles: null                       # Maps to PipelineConfig.stats_marker_roles
  generate_abundance_boxplots: true
  generate_volcano_plot: true
  generate_forest_plot: true
  generate_ds_heatmap: true

exports:
  enabled: true
  export_augmented_fcs: true
  export_gating_ml: true
  generate_miflowcyt_report: true

output:
  output_dir: outputs                      # Config runs set outputs/config_run/<platform>/<dataset>
  timestamp_dir: true                      # False when overwrite: true or --no-timestamp
  overwrite: false
```

---

## `resolved_config.yaml` Audit Snapshot

Every `FlowCytometryPipeline.run()` execution writes `<output_dir>/resolved_config.yaml`, capturing:
- Exact resolved `fsc_channel`, `ssc_channel`, `viability_channel`, `dump_channel`,
- Exact list of transformed channels and clustering `lineage_markers`,
- Effective looseness parameters, batch assessment verdict, and metadata file paths.

---

## References & Scientific Standards

* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
* **Gating-ML 2.0 Standard:** [](doi:10.1002/cyto.a.22690)
