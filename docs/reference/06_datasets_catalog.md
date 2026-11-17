# Reference: Benchmark Datasets & Config Catalog

FlowX ships with a curated registry of **20 benchmark datasets** (**11 conventional** and **9 spectral**) spanning human PBMCs, whole blood, bone marrow, tumor microenvironment, organoids, and macrophage studies, paired with **21 ready-to-run YAML configurations** under `configs/config_run/`.

> [!TIP]
> Download any registered dataset into `data/<platform>/<dataset_name>/` via:
> ```bash
> flowx download <dataset_name>
> # or inspect available datasets:
> flowx download --list
> ```

---

## Conventional Cytometry Datasets (11 Datasets, 12 Configs)

| Dataset ID | Tissue / Focus | Accession / Repo | Publication DOI | Config Path |
| :--- | :--- | :--- | :--- | :--- |
| `arguello_2020_scenith_metabolism` | Human PBMC SCENITH immunometabolism (NovoCyte Penteon, 24-ch) | Zenodo `10.5281/zenodo.15350040` | [10.5281/zenodo.15350040](https://doi.org/10.5281/zenodo.15350040) | `configs/config_run/conventional/arguello_2020_scenith_metabolism.yaml` |
| `brouwers_2026_whole_blood_omip` | Human whole blood OMIP panel (single-file SOM) | Zenodo `10.5281/zenodo.21869988` | [10.5281/zenodo.21869988](https://doi.org/10.5281/zenodo.21869988) | `configs/config_run/conventional/brouwers_2026_whole_blood_omip.yaml` |
| `chen_2025_placenta_hspc` | Placenta hematopoietic stem/progenitor cells (with 2 bridge controls) | Figshare `10.6084/m9.figshare.27271968` | [10.1371/journal.pbio.3003003](https://doi.org/10.1371/journal.pbio.3003003) | `configs/config_run/conventional/chen_2025_placenta_hspc.yaml` |
| `coburn_2021_thp1_stiffness` | THP-1 macrophage hydrogel stiffness polarization (CD33/CD11b/CD80/CD206) | Figshare `10.6084/m9.figshare.14450370` | [10.1002/jbm.a.37125](https://doi.org/10.1002/jbm.a.37125) | `configs/config_run/conventional/coburn_2021_thp1_stiffness.yaml` |
| `crespo_2026_crispr_cd8_tcells` | CRISPR-perturbed CD8+ T cells (`lineage;functional` dual roles) | Zenodo `10.5281/zenodo.20526379` | [10.5281/zenodo.20526379](https://doi.org/10.5281/zenodo.20526379) | `configs/config_run/conventional/crespo_2026_crispr_cd8_tcells.yaml` |
| `finak_2014_cytotrol_pbmc` | Human CytoTrol PBMC reference (FlowWorkspaceData) | GitHub / RGLab `flowWorkspaceData` | [10.1371/journal.pcbi.1003806](https://doi.org/10.1371/journal.pcbi.1003806) | `configs/config_run/conventional/finak_2014_cytotrol_pbmc.yaml` |
| `li_2025_caerin_macrophage` *(CD24 panel)* | Tumor-associated macrophage Ly6C/CD24 cohort | Zenodo `10.5281/zenodo.13976956` | [10.1038/s41598-025-87687-0](https://doi.org/10.1038/s41598-025-87687-0) | `configs/config_run/conventional/li_2025_caerin_macrophage_cd24.yaml` |
| `li_2025_caerin_macrophage` *(Ly6C panel)* | Tumor-associated macrophage Ly6C cohort (simultaneous viability+dump+DA/DS) | Zenodo `10.5281/zenodo.13976956` | [10.1038/s41598-025-87687-0](https://doi.org/10.1038/s41598-025-87687-0) | `configs/config_run/conventional/li_2025_caerin_macrophage_ly6c.yaml` |
| `quintana_2026_skull_marrow_chimera` | Skull bone marrow chimera (elevated Time QC debris/clots) | Zenodo `10.5281/zenodo.22808501` | [10.1371/journal.pbio.3002389](https://doi.org/10.1371/journal.pbio.3002389) | `configs/config_run/conventional/quintana_2026_skull_marrow_chimera.yaml` |
| `quintelier_2023_cytonorm_meso` | Mesothelioma multi-batch PBMC cohort (CytoNorm benchmark) | Zenodo `10.5281/zenodo.13928969` | [10.1136/jitc-2023-007070](https://doi.org/10.1136/jitc-2023-007070) | `configs/config_run/conventional/quintelier_2023_cytonorm_meso.yaml` |
| `weber_2025_bk_polyomavirus` | BK polyomavirus T-cell cohort (dual-detector deduplication + FDR-sig DS) | Figshare `10.6084/m9.figshare.29601881` | [10.1371/journal.ppat.1013122](https://doi.org/10.1371/journal.ppat.1013122) | `configs/config_run/conventional/weber_2025_bk_polyomavirus.yaml` |
| `white_2021_flowkit_ics` | FlowKit 8-color human PBMC intracellular cytokine staining | GitHub / FlowKit `8_color_data_set` | [10.3389/fimmu.2021.768541](https://doi.org/10.3389/fimmu.2021.768541) | `configs/config_run/conventional/white_2021_flowkit_ics.yaml` |

---

## Spectral Cytometry Datasets (9 Datasets, 9 Configs)

| Dataset ID | Tissue / Focus | Accession / Repo | Publication DOI | Config Path |
| :--- | :--- | :--- | :--- | :--- |
| `baglio_2024_osteosarcoma_spectral` | Osteosarcoma spectral panel (3 autofluorescence channels) | Zenodo `10.5281/zenodo.14513596` | [10.1002/cyto.a.70034](https://doi.org/10.1002/cyto.a.70034) | `configs/config_run/spectral/baglio_2024_osteosarcoma_spectral.yaml` |
| `balin_2025_thymopoiesis_omip116` | Thymopoiesis OMIP-116 high-dimensional spectral panel | Zenodo `10.5281/zenodo.20133311` | [10.1002/cyto.a.24951](https://doi.org/10.1002/cyto.a.24951) | `configs/config_run/spectral/balin_2025_thymopoiesis_omip116.yaml` |
| `barrachina_2026_ipsc_organoid_ccl5` | iPSC organoid CCL5 spectral cohort | Zenodo `10.5281/zenodo.22783306` | [10.1016/j.stemcr.2026.103014](https://doi.org/10.1016/j.stemcr.2026.103014) | `configs/config_run/spectral/barrachina_2026_ipsc_organoid_ccl5.yaml` |
| `de_vries_2023_crc_til_mucosal` | Colorectal cancer TIL mucosal spectral panel | Zenodo `10.5281/zenodo.21643327` | [10.5281/zenodo.21643327](https://doi.org/10.5281/zenodo.21643327) | `configs/config_run/spectral/de_vries_2023_crc_til_mucosal.yaml` |
| `flagr_2024_cytobatch_pbmc` | Multi-batch spectral PBMC cohort (CytoBatch benchmark) | Zenodo `10.5281/zenodo.15388817` | [10.5281/zenodo.15388817](https://doi.org/10.5281/zenodo.15388817) | `configs/config_run/spectral/flagr_2024_cytobatch_pbmc.yaml` |
| `korotkaja_2024_breast_tumor_tme` | Breast tumor TME (64 raw APD detectors $\to$ 8 unmixed fluorophores) | Zenodo `10.5281/zenodo.17457137` | [10.1016/j.omton.2025.201124](https://doi.org/10.1016/j.omton.2025.201124) | `configs/config_run/spectral/korotkaja_2024_breast_tumor_tme.yaml` |
| `kreutmair_2026_icu_pbmc` | Critical-care ICU PBMC 32-color spectral cohort | Zenodo `10.5281/zenodo.22079206` | [10.1016/j.xcrm.2026.102919](https://doi.org/10.1016/j.xcrm.2026.102919) | `configs/config_run/spectral/kreutmair_2026_icu_pbmc.yaml` |
| `michaelis_2025_c9orf72_als_pbmc` | C9orf72 ALS PBMC spectral cohort (confounded batch guard) | Zenodo `10.5281/zenodo.16749283` | [10.1038/s41586-025-09588-6](https://doi.org/10.1038/s41586-025-09588-6) | `configs/config_run/spectral/michaelis_2025_c9orf72_als_pbmc.yaml` |
| `rockstein_2026_omip_whole_blood` | Human whole blood OMIP high-plex spectral cohort | Zenodo `10.5281/zenodo.23950826` | [10.5281/zenodo.23950826](https://doi.org/10.5281/zenodo.23950826) | `configs/config_run/spectral/rockstein_2026_omip_whole_blood.yaml` |

---

## References & Scientific Standards

* **FlowKit Framework:** [](doi:10.3389/fimmu.2021.768541)
* **CytoNorm Batch Normalization:** [](doi:10.1002/cyto.a.23904)
* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
