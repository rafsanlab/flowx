# Explanation: Unified Step Architecture & Orchestration

FlowX is architected around a **Unified Step Runner** design: every analytical stage—from Time QC and spectral unmixing through cohort statistical testing—is encapsulated in a standalone, deterministic step runner paired with typed Config and Result dataclasses. Both the end-to-end `FlowCytometryPipeline` orchestrator and the 15 standalone `flowx` CLI subcommands invoke the exact same step runners.

---

## End-to-End Pipeline Orchestration Flowchart

```mermaid
flowchart TD
    Start(["<code>flowx pipeline -c config.yaml</code><br/>or Standalone CLI Subcommand"]) --> Meta["<b>Metadata &amp; Role Resolution</b><br/><code>load_sample_metadata()</code> + <code>resolve_marker_metadata()</code><br/>(<code>lineage_markers: auto</code>)"]
    Meta --> Step1["<b>Step 1: Time QC</b><br/><code>run_time_qc_step()</code><br/>Flow rate MAD, bin median drift, OLS slope, margin saturation"]
    Step1 --> Step2{"Platform Mode?"}
    Step2 -->|"Conventional"| Comp["<b>Step 2a: Compensation</b><br/><code>run_compensation_step()</code><br/>Controls priority vs embedded SPILL,<br/>PMT voltage power-law check, AutoSpill/Median"]
    Step2 -->|"Spectral"| Unmix["<b>Step 2b: Spectral Unmixing</b><br/><code>run_spectral_unmixing_pipeline()</code><br/>OLS / Two-Pass Poisson WLS / Projected OLS,<br/>Single or Multi-AF subtraction"]
    Comp --> Transform["<b>Scale Transformation</b><br/><code>run_transformation_step()</code><br/>Logicle / Biexponential LUT / Arcsinh<br/>(+ <code>is_empirically_pre_transformed</code> guard)"]
    Unmix --> Transform
    Transform --> Step3["<b>Step 3: Core Cleanup Gating</b><br/><code>run_cleanup_step()</code><br/>2D GMM Debris &rarr; Pulse-Ratio Singlet &rarr;<br/>1D KDE Viability &rarr; Optional Dump"]
    Step3 --> Step4["<b>Step 4: Batch Assessment &amp; Prep</b><br/><code>run_batch_preparation()</code><br/>Confounding Guard &rarr; 0&ndash;100 Drift Score &rarr;<br/>CytoNorm / Quantile &rarr; Stratified Downsampling"]
    Step4 --> Step5["<b>Step 5: Population Discovery</b><br/><code>run_cohort_population_analysis()</code><br/>MiniSom + Ward / PhenoGraph &rarr; Auto-k &rarr;<br/>Otsu Signature Annotation &rarr; UMAP"]
    Step5 --> Step6["<b>Step 6: Cohort Statistics &amp; Exports</b><br/>DA &amp; DS Hypothesis Tests (BH-FDR) &rarr;<br/>Augmented FCS, Gating-ML 2.0<br/>&amp; MIFlowCyt Report"]
```

---

## Why Unified Step Runners?

1. **Zero CLI-vs-Pipeline Drift:** Previously, standalone subcommands could diverge from the full pipeline in default parameters, plot styling, or table schemas. By routing `flowx time-qc`, `flowx compensate`, `flowx unmix`, `flowx transform`, `flowx cleanup`, `flowx batch-prepare`, and `flowx cohort` directly into the shared step runners (`run_time_qc_step`, `run_compensation_step`, `run_spectral_unmixing_pipeline`, `run_transformation_step`, `run_cleanup_step`, `run_batch_preparation`, and `run_cohort_population_analysis`), standalone runs and `flowx pipeline` produce identical outputs and directory layouts (`01_time_qc/` through `06_statistical_analysis/` and `exports/`).
2. **Typed Contracts (`Config` $\to$ `Result`):** Each runner accepts a structured configuration dataclass (`TimeQCConfig`, `CompensationConfig`, `UnmixingConfig`, `TransformConfig`, `CleanUpGateConfig`, `BatchPreparationConfig`) built by `PipelineConfig.to_<step>_config()` and returns a structured result dataclass (`TimeQCStepResult`, `CompensationStepResult`, `UnmixingResult`, `TransformStepResult`, `CleanupStepResult`, `CohortBatchResult`, `CohortPopulationResult`).
3. **Config Auditability (`resolved_config.yaml`):** Whenever `lineage_markers: auto` or metadata auto-discovery runs, FlowX serializes the exact resolved channel lists, gating modes, and looseness values into `resolved_config.yaml` alongside the original `config.yaml`.

---

## Granularity Routing in Cohort Discovery & Statistics

During Step 5 and Step 6, `stats_granularity` controls whether hypothesis tests and exports operate on:
- **`populations` (Default):** Merged biological lineages (e.g., `CD4+ Naive T Cells`, `Classical Monocytes (CD14+ CD16-)`) written to `06_statistical_analysis/{tables,plots}/`.
- **`clusters`:** Raw unsupervised SOM metaclusters or PhenoGraph communities ($C_0 \dots C_{k-1}$) written to `06_statistical_analysis/{tables,plots}/`.
- **`both`:** Biological populations in `06_statistical_analysis/{tables,plots}/` **and** raw clusters isolated in `06_statistical_analysis/{tables,plots}/clusters/`.

---

## References & Scientific Standards

* **FlowKit Framework:** [](doi:10.3389/fimmu.2021.768541)
* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
* **Gating-ML 2.0 Standard:** [](doi:10.1002/cyto.a.22690)
