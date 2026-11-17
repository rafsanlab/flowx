```{image} images/flowx-logo.svg
:alt: FlowX Logo
:width: 120px
:align: center
:class: dark:hidden
```

```{image} images/flowx-logo-dark.svg
:alt: FlowX Logo
:width: 120px
:align: center
:class: hidden dark:block
```

# FlowX Documentation

Welcome to **FlowX** — a source-available, modular, Python-native bioinformatics framework for conventional and full-spectrum flow cytometry preprocessing, quality control, compensation/unmixing, batch correction, unsupervised population discovery, cohort statistics, and MIFlowCyt-compliant reporting — free for academic and non-commercial research, with enterprise BYOC licenses available for industry.

FlowX documentation is structured around the **[Diátaxis framework](https://diataxis.fr/)**, separating hands-on learning from task recipes, technical lookup, and mathematical theory.

---

## Choose Your Path

| Quadrant | Purpose | Start Here |
| :--- | :--- | :--- |
| **1. Tutorials** *(Learning-Oriented)* | Guided, executable **Jupyter notebooks (`.ipynb`)** walking through real human PBMC cohorts (plus a 64-channel raw spectral unmixing complement) from `.fcs` to biological discovery. Fetch raw `.fcs` data with `flowx download <dataset_id>`. | • [Quickstart: PBMC CLI & Config](tutorials/01_quickstart_cli_and_config.ipynb)<br>• [Conventional PBMC Cohort Walkthrough](tutorials/02_conventional_pbmc_cohort_walkthrough.ipynb)<br>• [Spectral 32-Color PBMC Cohort](tutorials/03_spectral_pbmc_cohort_walkthrough.ipynb)<br>• [Raw 64-Channel Spectral Unmixing Complement](tutorials/04_raw_spectral_unmixing_complement.ipynb) |
| **2. How-To Guides** *(Goal-Oriented)* | Focused operational recipes for metadata curation, Time QC, spillover/voltage mismatches, cleanup looseness tuning, batch safety guards, and Gating-ML/MIFlowCyt exports. | • [Curate Metadata & Auto-Resolve Roles](how-to/01_curate_metadata_and_auto_roles.md)<br>• [Preflight Inspection & Harmonization](how-to/02_preflight_inspection_and_harmonization.md)<br>• [Time QC & Control Salvage](how-to/03_time_qc_and_control_salvage.md)<br>• [Conventional Compensation & Voltages](how-to/04_conventional_compensation_and_voltages.md)<br>• [Spectral Unmixing & Autofluorescence](how-to/05_spectral_unmixing_and_autofluorescence.md)<br>• [Tune Cleanup Gate Looseness](how-to/06_tune_cleanup_gate_looseness.md)<br>• [Batch Assessment & Normalization](how-to/07_batch_assessment_and_normalization.md)<br>• [Population Discovery & Annotation](how-to/08_population_discovery_and_annotation.md)<br>• [Cohort Statistical Analysis](how-to/09_cohort_statistical_analysis.md)<br>• [Export Gating-ML & MIFlowCyt](how-to/10_export_gating_ml_and_miflowcyt.md)<br>• [Manual & Hierarchical Gating](how-to/11_manual_hierarchical_gating.md) |
| **3. Reference** *(Information-Oriented)* | Authoritative lookup for the 15 `flowx` CLI subcommands, 114-field `PipelineConfig` YAML schema, `samples.csv`/`markers.csv` specs, Python API, and 20-dataset / 21-config catalog. | • [CLI Command Reference](reference/01_cli_reference.md)<br>• [YAML Configuration Schema](reference/02_yaml_config_schema.md)<br>• [Metadata CSV Specifications](reference/03_metadata_csv_specifications.md)<br>• [Python API & Step Runners](reference/04_python_api_and_step_runners.md)<br>• [Output Hierarchy & Table Schemas](reference/05_output_hierarchy_and_schemas.md)<br>• [Benchmark Datasets & Config Catalog](reference/06_datasets_catalog.md)<br>• [Benchmarks & Validation Matrix](reference/07_benchmarks_and_validation.md) |
| **4. Explanation** *(Understanding-Oriented)* | Unified step-runner architecture, behavioral flowcharts, exact implemented mathematical equations, and clean-room licensing governance. | • [Unified Step Architecture & Orchestration](explanation/01_unified_step_architecture.md)<br>• [Time QC Mathematics](explanation/02_time_qc_mathematics.md)<br>• [Compensation & Spectral Deconvolution](explanation/03_compensation_and_spectral_deconvolution.md)<br>• [Scale Transformation & Cleanup Gating Math](explanation/04_scaling_and_cleanup_gating_math.md)<br>• [Batch Effect & Normalization Math](explanation/05_batch_effect_and_normalization_math.md)<br>• [Clustering, Manifolds & Annotation Math](explanation/06_clustering_manifolds_and_annotation_math.md)<br>• [Statistical Testing & Power Theory](explanation/07_statistical_testing_and_power_theory.md)<br>• [Licensing & Clean-Room Governance](explanation/08_licensing_and_cleanroom_governance.md) |

---

## Core Capabilities at a Glance

1. **Pre-Flight Cohort Inspection (`flowx inspect`) & Metadata Auto-Resolution (`markers.csv` / `samples.csv`):** Zero-guesswork channel role mapping (`lineage`, `functional`, `viability`, `dump`, `autofluorescence`), panel harmonization, and `resolved_config.yaml` snapshotting.
2. **Step 01 — Time Quality Control (`flowx time-qc`):** Flow-rate MAD stability, bin-median & OLS slope signal drift, hardware margin saturation trimming, and manual time-window gating.
3. **Step 02 — Compensation & Full-Spectrum Unmixing (`flowx compensate`, `flowx unmix`):** Single-stain bead/cell carrier pre-gating, Median $\Delta$MFI, AutoSpill 1-step Huber IRLS, PMT voltage concordance (`warn`, `fallback_embedded`, `scale_power_law`, `strict_error`), Spillover Spreading Matrix (SSM), plus Poisson-weighted WLS/OLS/Projected-OLS spectral deconvolution with single/multi-component autofluorescence extraction.
4. **Step 03 & 04A — Scale Transformation & Core Clean-up Gating (`flowx transform`, `flowx cleanup`):** Gating-ML 2.0 Logicle, FlowJo Biexponential LUT, Arcsinh, pre-transformed scale guards, and 4-stage cleanup (`Debris` $\to$ `Dual Singlets` $\to$ `Viability` $\to$ `Dump`) with continuous boundary looseness controls (`debris_sd_scale`, `debris_min_probability`, `singlet_tolerance`, `viability_looseness`, `dump_looseness`).
5. **Step 04B — Batch Necessity Assessment & Normalization (`flowx batch-check`, `flowx batch-prepare`):** Automated confounding guard (`NOT_RECOMMENDED_CONFOUNDED`), 0–100 Wasserstein EMD/KS Drift Score, CytoNorm quantile splines, anchor-free quantile alignment, and deterministic cohort downsampling.
6. **Step 05 & 06 — Population Discovery, Cohort Statistics & Interoperability (`flowx cohort`, `flowx pipeline`):** MiniSom + Consensus PAC auto-$k$ (plus PhenoGraph), regime-aware Otsu phenotypic annotation, two-tier `clusters/` vs `populations/` UMAP & heatmap galleries, Differential Abundance (DA) & Differential State (DS) with Benjamini-Hochberg FDR, and exports to Gating-ML 2.0, augmented `.fcs`, and MIFlowCyt HTML/Markdown reports.

---

(references)=
## References & Scientific Standards

* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
* **FCS 3.1 Data File Standard:** [](doi:10.1002/cyto.a.20825)
* **Gating-ML 2.0 Standard:** [](doi:10.1002/cyto.a.22690)
* **flowAI Algorithm:** [](doi:10.1093/bioinformatics/btw191)
* **flowCut Algorithm:** [](doi:10.1002/cyto.a.24670)
* **PeacoQC Algorithm:** [](doi:10.1002/cyto.a.24501)
* **AutoSpill Compensation Framework:** [](doi:10.1038/s41467-021-23126-8)
* **Spillover Spreading Matrix (SSM):** [](doi:10.1002/cyto.a.22251)
* **Logicle Data Scaling:** [](doi:10.1002/cyto.a.20258)
* **openCyto Automated Gating:** [](doi:10.1371/journal.pcbi.1003806)
* **CytoNorm Batch Normalization:** [](doi:10.1002/cyto.a.23904)
* **FlowSOM Clustering:** [](doi:10.1002/cyto.a.22625)
* **PhenoGraph Algorithm:** [](doi:10.1016/j.cell.2015.05.047)
* **UMAP Dimensionality Reduction:** [](doi:10.48550/arXiv.1802.03426)
* **FlowKit Computational Toolkit:** [](doi:10.3389/fimmu.2021.768541)
* **Spectral Flow Cytometry Overview:** [](doi:10.1002/cyto.a.23779)
* **OMIP-098 39-Color Full Spectrum Panel:** [](doi:10.1002/cyto.a.24789)
