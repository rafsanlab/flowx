# Reference: Benchmarks & Validation Matrix

All **21 config-driven pipeline runs** across the 20 registered datasets have been executed end-to-end with `flowx pipeline -c <config> --overwrite`, generating full step outputs under `outputs/config_run/<platform>/<dataset_name>/` and per-dataset benchmark comparison reports under `outputs/reports/<platform>/<dataset_name>/`.

---

## Conventional Benchmark Validation Summary (12 Config Runs)

| Dataset / Config | Samples | Batch Verdict | Discovery ($k \to$ Populations) | Benchmark Report |
| :--- | :---: | :--- | :---: | :--- |
| `arguello_2020_scenith_metabolism` | 2 | `NOT_RECOMMENDED_CONFOUNDED` | $k = 20 \to 12$ pops | `outputs/reports/conventional/arguello_2020_scenith_metabolism/arguello_2020_scenith_metabolism_benchmark_comparison_report.md` |
| `brouwers_2026_whole_blood_omip` | 1 | N/A (single sample) | Single-file SOM ($k = 10$ clusters) | `outputs/reports/conventional/brouwers_2026_whole_blood_omip/brouwers_2026_whole_blood_omip_benchmark_comparison_report.md` |
| `chen_2025_placenta_hspc` | 7 (+2 ctrl in `samples.csv`) | `NOT_NEEDED` | $k = 14 \to 5$ pops | `outputs/reports/conventional/chen_2025_placenta_hspc/chen_2025_placenta_hspc_benchmark_comparison_report.md` |
| `coburn_2021_thp1_stiffness` | 4 | `NOT_NEEDED` | $k = 12 \to 2$ pops | `outputs/reports/conventional/coburn_2021_thp1_stiffness/coburn_2021_thp1_stiffness_benchmark_comparison_report.md` |
| `crespo_2026_crispr_cd8_tcells` | 8 | `NOT_NEEDED` | $k = 14 \to 9$ pops | `outputs/reports/conventional/crespo_2026_crispr_cd8_tcells/crespo_2026_crispr_cd8_tcells_benchmark_comparison_report.md` |
| `finak_2014_cytotrol_pbmc` | 2 | `NOT_NEEDED` | $k = 13 \to 6$ pops | `outputs/reports/conventional/finak_2014_cytotrol_pbmc/finak_2014_cytotrol_pbmc_benchmark_comparison_report.md` |
| `li_2025_caerin_macrophage_cd24` | 12 | `NOT_NEEDED` | $k = 14 \to 4$ pops | `outputs/reports/conventional/li_2025_caerin_macrophage_cd24/li_2025_caerin_macrophage_cd24_benchmark_comparison_report.md` |
| `li_2025_caerin_macrophage_ly6c` | 12 | `NOT_NEEDED` | $k = 14 \to 7$ pops | `outputs/reports/conventional/li_2025_caerin_macrophage_ly6c/li_2025_caerin_macrophage_ly6c_benchmark_comparison_report.md` |
| `quintana_2026_skull_marrow_chimera` | 4 | `NOT_NEEDED` | $k = 16 \to 7$ pops | `outputs/reports/conventional/quintana_2026_skull_marrow_chimera/quintana_2026_skull_marrow_chimera_benchmark_comparison_report.md` |
| `quintelier_2023_cytonorm_meso` | 12 | `CAUTION_ANCHOR_FREE` | $k = 16 \to 15$ pops | `outputs/reports/conventional/quintelier_2023_cytonorm_meso/quintelier_2023_cytonorm_meso_benchmark_comparison_report.md` |
| `weber_2025_bk_polyomavirus` | 15 | `NOT_NEEDED` | $k = 10 \to 3$ pops | `outputs/reports/conventional/weber_2025_bk_polyomavirus/weber_2025_bk_polyomavirus_benchmark_comparison_report.md` |
| `white_2021_flowkit_ics` | 3 | `NOT_NEEDED` | $k = 14 \to 5$ pops | `outputs/reports/conventional/white_2021_flowkit_ics/white_2021_flowkit_ics_benchmark_comparison_report.md` |

---

## Spectral Benchmark Validation Summary (9 Config Runs)

| Dataset / Config | Samples | Batch Verdict | Discovery ($k \to$ Populations) | Benchmark Report |
| :--- | :---: | :--- | :---: | :--- |
| `baglio_2024_osteosarcoma_spectral` | 2 | `NOT_NEEDED` | $k = 20 \to 11$ pops | `outputs/reports/spectral/baglio_2024_osteosarcoma_spectral/baglio_2024_osteosarcoma_spectral_benchmark_comparison_report.md` |
| `balin_2025_thymopoiesis_omip116` | 2 | `NOT_RECOMMENDED_CONFOUNDED` | $k = 22 \to 10$ pops | `outputs/reports/spectral/balin_2025_thymopoiesis_omip116/balin_2025_thymopoiesis_omip116_benchmark_comparison_report.md` |
| `barrachina_2026_ipsc_organoid_ccl5` | 6 | `NOT_NEEDED` | $k = 18 \to 5$ pops | `outputs/reports/spectral/barrachina_2026_ipsc_organoid_ccl5/barrachina_2026_ipsc_organoid_ccl5_benchmark_comparison_report.md` |
| `de_vries_2023_crc_til_mucosal` | 2 | `NOT_NEEDED` | $k = 22 \to 8$ pops | `outputs/reports/spectral/de_vries_2023_crc_til_mucosal/de_vries_2023_crc_til_mucosal_benchmark_comparison_report.md` |
| `flagr_2024_cytobatch_pbmc` | 4 | `CAUTION_ANCHOR_FREE` | $k = 20 \to 8$ pops | `outputs/reports/spectral/flagr_2024_cytobatch_pbmc/flagr_2024_cytobatch_pbmc_benchmark_comparison_report.md` |
| `korotkaja_2024_breast_tumor_tme` | 6 | `NOT_NEEDED` | $k = 16 \to 3$ pops (64 raw APD $\to$ 8 fl) | `outputs/reports/spectral/korotkaja_2024_breast_tumor_tme/korotkaja_2024_breast_tumor_tme_benchmark_comparison_report.md` |
| `kreutmair_2026_icu_pbmc` | 8 | `CAUTION_ANCHOR_FREE` | $k = 22 \to 10$ pops (0 unassigned) | `outputs/reports/spectral/kreutmair_2026_icu_pbmc/kreutmair_2026_icu_pbmc_benchmark_comparison_report.md` |
| `michaelis_2025_c9orf72_als_pbmc` | 8 | `NOT_RECOMMENDED_CONFOUNDED` | $k = 20 \to 10$ pops (0 unassigned) | `outputs/reports/spectral/michaelis_2025_c9orf72_als_pbmc/michaelis_2025_c9orf72_als_pbmc_benchmark_comparison_report.md` |
| `rockstein_2026_omip_whole_blood` | 6 | `NOT_RECOMMENDED_CONFOUNDED` | $k = 24 \to 16$ pops | `outputs/reports/spectral/rockstein_2026_omip_whole_blood/rockstein_2026_omip_whole_blood_benchmark_comparison_report.md` |

---

## Key Cross-Cohort Validation Takeaways

1. **Zero Unassigned PBMC Clusters on High-Plex Spectral Panels:** Both `kreutmair_2026_icu_pbmc` ($k=22 \to 10$ pops) and `michaelis_2025_c9orf72_als_pbmc` ($k=20 \to 10$ pops) achieved zero unassigned clusters using `lineage_markers: auto` driven by `metadata/markers.csv`.
2. **Confounding Safety Guard Verification:** Datasets where `batch_id` is collinear with `condition` (`arguello_2020_scenith_metabolism`, `balin_2025_thymopoiesis_omip116`, `michaelis_2025_c9orf72_als_pbmc`, `rockstein_2026_omip_whole_blood`) were automatically flagged as `NOT_RECOMMENDED_CONFOUNDED`, preventing biological signal erasure.
3. **Raw Spectral Unmixing:** `korotkaja_2024_breast_tumor_tme` successfully unmixed 64 raw Cytek Aurora APD detectors into 8 clean fluorophore parameters using single-stained controls prior to cleanup gating and clustering.

---

## References & Scientific Standards

* **FlowSOM Clustering:** [](doi:10.1002/cyto.a.22625)
* **CytoNorm Batch Normalization:** [](doi:10.1002/cyto.a.23904)
* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
