# 01. CLI Reference (`flowx`)

Authoritative lookup for all 15 `flowx` CLI subcommands, aliases, key flags, and underlying step runners (`src/flowx/cli.py`). Every step subcommand accepts `-c / --config` to resolve settings from a unified YAML/JSON file, with explicit CLI flags overriding config values.

---

## Subcommands & Aliases

| Subcommand | Aliases | Purpose | Underlying Runner / Target |
| :--- | :--- | :--- | :--- |
| `flowx inspect` | — | Pre-flight FCS audit, QC lists & panel harmonization | `inspect_fcs_files`, `PanelHarmonizer.harmonize_cohort` |
| `flowx info` | — | Single-file FCS header & channel summary | `parse_fcs`, `flowx.ui.console.render_fcs_info` |
| `flowx plot` | — | Quick 2D scatter/density plot (`--x`, `--y`) | `flowx.visualization.plotting.plot_scatter_2d` |
| `flowx time-qc` | `qc` | Step 01 Time QC (flow rate, drift, margins) | `run_cohort_time_qc` / `run_time_qc_step` |
| `flowx compensate` | — | Step 02A Conventional spillover & voltage audit | `run_dataset_compensation` / `run_compensation_step` |
| `flowx unmix` | — | Step 02B Full-spectrum deconvolution (WLS/OLS/NNLS) | `run_spectral_unmixing_pipeline` |
| `flowx transform` | — | Step 03A/04A Scale transform (Logicle/Biex/Arcsinh) | `run_transformation_step` |
| `flowx cleanup` | — | Step 03B 4-stage cleanup cascade & looseness tuning | `run_cohort_cleanup` / `run_cleanup_step` |
| `flowx batch-prepare` | `batch`, `normalize` | Step 04B Batch normalization & downsampling | `run_batch_preparation` |
| `flowx batch-check` | `check-batch` | Diagnose inter-batch drift & condition confounding | `assess_batch_effect_need` |
| `flowx cohort` | `population`, `cluster-stats` | Step 05 & 06 Discovery, annotation, DA/DS & exports | `run_cohort_population_analysis` |
| `flowx pipeline` | — | Full end-to-end single-file or cohort pipeline | `FlowCytometryPipeline.run` |
| `flowx replay` | — | Replay saved `workspace.json` / `workspace.xml` | `WorkspaceSession.replay` |
| `flowx config` | — | Generate (`--template`) or validate (`--validate`) YAML | `PipelineConfig.generate_template` / `.load` |
| `flowx download` | `dataset` | List, verify & download the 20 benchmark datasets | `list_datasets` (`registry`), `download_dataset`, `verify_dataset` (`downloader`) |

---

## Key Flags by Subcommand

### `flowx inspect`
`targets` (files/dirs/globs), `--files-list`, `-o/--output-dir` (`outputs/flowcyt_inspection`), `-m/--population-map`, `--channels-csv`, `--no-markdown`, `--pattern` (`*.fcs`), `-r/--recursive` / `--no-recursive`, `--max-depth`, `--event-stats`, `-H/--harmonize` (`DEFAULT` or path), `--max-rows` (`25`).

### `flowx time-qc` (alias `qc`)
`target`, `--files-list`, `-c/--config`, `--bin-width`, `--no-drift`, `--no-margins`, `--min-sample-events` (`5000`), `--min-control-events` (`500`), `--export-clean-fcs`, `--no-plot`, `-o/--output-dir`.

### `flowx compensate`
`fcs_file`, `-c/--config`, `-o/--output-dir`, `--method` (`hybrid`, `median_delta_mfi`, `autospill_regression`), `--no-time-qc`, `--override` (`Primary->Secondary=Val`), `--matrix-csv`, `--comp-controls-dir`, `--unstained-control`, `--comp-manifest`, `--qc-manifest`, `--output-matrix`, `--voltage-policy` (`warn`, `fallback_embedded`, `scale_power_law`, `strict_error`), `--voltage-tolerance` (`5.0`), `--no-plots`, `--output-plot`.

### `flowx unmix`
`-s/--sample`, `--config`, `-c/--controls-dir`, `-u/--unstained`, `--method` (`wls`, `ols`, `nnls`), `--af-mode` (`single`, `multi`, `none`), `--multi-af-components` (`2`), `--alpha-poisson` (`1.0`), `--electronic-noise` (`50.0`), `-o/--output-dir`.

### `flowx transform`
`fcs_file`, `-c/--config`, `--channel`, `--transform/--type` (`logicle`, `biexponential`, `biex`, `asinh`, `arcsinh`, `log`), `--cofactor`, `--exclude-channels`, `--markers-file`, `-o/--output-dir`, `--output-plot`.

### `flowx cleanup`
`target`, `--files-list`, `-c/--config`, `--markers-file`, `--debris-method` (`gmm`, `adaptive`, `manual`), `--debris-n-components` (`3`), `--debris-sd-scale` (`1.0`), `--debris-min-probability`, `--singlet-mode` (`dual`, `single`, `one_time`, `fsc_only`, `ssc_only`, `none`), `--singlet-method` (`gmm_ratio`, `envelope`), `--singlet-tolerance` (`0.25`), `--no-fsc-singlets`, `--no-ssc-singlets`, `--fsc-channel`, `--ssc-channel`, `--fsc-height`, `--ssc-height`, `--viability-channel`, `--viability-cutoff`, `--viability-looseness` (`1.0`), `--no-viability`, `--dump-channel`, `--dump-cutoff`, `--dump-looseness` (`1.0`), `--compensation-matrix`, `--no-compensation`, `--export-clean-fcs`, `--export-uncompensated`, `--transform`, `--cofactor`, `--no-plot`, `-o/--output-dir`.

### `flowx batch-prepare` (aliases `batch`, `normalize`) & `flowx batch-check`
* **`batch-prepare`:** `-c/--config`, `targets`, `--manifest`, `--files-list`, `--method` (`cytonorm`, `quantile`, `none`), `--no-normalize`, `--goal` (`median`, `mean`, `reference`), `--reference-batch`, `--clusters` (`10`), `--quantiles` (`101`), `--downsample` (`50000`), `--downsample-mode` (`target_count`, `min_sample`, `stratified`, `proportional`, `none`), `--no-downsample`, `--seed` (`42`), `--export-fcs`, `--no-pooled`, `--markers-file`, `--check-only`, `--no-assess`, `--overwrite-autocheck`, `--no-plot`, `-o/--output-dir`.
* **`batch-check`:** `targets`, `--manifest`, `--condition-key` (`condition`), `-c/--config`, `-o/--output-dir`.

### `flowx cohort` (aliases `population`, `cluster-stats`)
`targets`, `--files-list`, `-c/--config`, `-m/--metadata-file`, `--markers-file`, `--cell-signatures`, `--markers`, `--marker-roles`, `--include-markers`, `--exclude-markers`, `--stats-granularity` (`populations`, `clusters`, `both`), `--stats-marker-roles`, `--method/--clustering-method` (`som`, `phenograph`), `-k/--n-clusters`, `--auto-k/--no-auto-k`, `--auto-k-method` (`consensus`, `davies_bouldin`, `silhouette`, `calinski_harabasz`, `elbow`), `--k-min` (`4`), `--k-max` (`20`), `--consensus-subsample` (`0.8`), `--consensus-iterations` (`50`), `--umap/--no-umap`, `--umap-events` (`1000`), `--condition-column`, `--batch-column`, `--umap-covariates`, `--reference-group`, `--target-group`, `--paired-column`, `--cluster-name-map`, `--cluster-palette` (`tab20`), `--da-method`, `--ds-method`, `--fdr-threshold` (`0.05`), `--log2fc-threshold` (`1.0`), `--export-augmented-fcs/--no-augmented-fcs`, `--export-gating-ml/--no-gating-ml`, `--miflowcyt-report/--no-miflowcyt-report`, `--clean-up`, `--seed`, `-o/--output-dir`.

### `flowx pipeline`
`fcs_file`, `-c/--config`, `-o/--output-dir/--output`, `--overwrite` (implies `--no-timestamp` and cleans stale files), `--no-timestamp`, `--qc/--no-qc`, `--discovery/--no-discovery`, `--numbered-steps/--no-numbered-steps`, `--technology` (`auto`, `conventional`, `spectral`), `--gating-mode` (`template`, `som`, `phenograph`, `compare`), `--comp-source`, `--comp-estimation-method`, `--unmix-method`, `--voltage-mismatch-policy`, `--voltage-tolerance`, plus all CLI overrides for Steps 01–06 and exports.

### `flowx replay`, `flowx config`, & `flowx download` (`dataset`)
* **`replay`:** `workspace_file` (`.json` or `.xml`), `--fcs-file`.
* **`config`:** `--template <path>`, `--format` (`yaml`, `json`), `--validate <path>`.
* **`download` (`dataset`):** `datasets` (names or IDs), `-l/--list`, `-c/--category` (`all`, `conventional`, `spectral`), `-i/--info`, `--verify-only`, `-d/--data-dir` (`data`), `--json`.

---

## References & Scientific Standards

* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
* **FCS 3.1 Data File Standard:** [](doi:10.1002/cyto.a.20825)
