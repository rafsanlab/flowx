# 05. Full-Spectrum Unmixing & Autofluorescence Handling (`flowx unmix`)

Use `flowx unmix` (or `02_compensation` with `technology: spectral` in YAML) to deconvolve overdetermined ($D > P$) spectral detector arrays and prevent cellular autofluorescence (AF) from distorting downstream discovery.

---

## 1. Exclude Pre-Unmixed Autofluorescence Channels (`rockstein_2026_omip_whole_blood`)

For cohorts already unmixed by SpectroFlo (such as `rockstein_2026_omip_whole_blood` or `kreutmair_2026_icu_pbmc`), keep `02_compensation.source: embedded` (or `none`) and tag `AF-A` as `marker_class: autofluorescence` in `metadata/markers.csv`:
- `AF-A` is retained in the event table for inspection,
- Automatically excluded from `04_transformation` (`channels: auto`) and `05_population_discovery` (`lineage_markers: auto`).

---

## 2. Run Live Spectral Deconvolution (`korotkaja_2024_breast_tumor_tme` Complement)

When raw $D$-detector spectra and single-stain `raw/controls/` are available (e.g., 64 Cytek Aurora APD channels $\to$ 8 fluorophores in `korotkaja_2024_breast_tumor_tme`):

**YAML (`02_compensation`):**
```yaml
input:
  technology: spectral
02_compensation:
  enabled: true
  source: controls
  controls_dir: data/spectral/korotkaja_2024_breast_tumor_tme/raw/controls
  unmixing_method: wls        # wls (default) | ols | nnls (projected_ols)
  autofluorescence_mode: single # single | multi | none
  alpha_poisson: 1.0
  electronic_noise: 50.0
```

**Standalone CLI (`flowx unmix`):**
```bash
# Config-driven unmixing
flowx unmix --config configs/config_run/spectral/korotkaja_2024_breast_tumor_tme.yaml

# Or explicit CLI arguments (pass -u when a separate unstained FCS tube is present)
flowx unmix -s data/spectral/korotkaja_2024_breast_tumor_tme/raw/samples/PBS_001.fcs \
  -c data/spectral/korotkaja_2024_breast_tumor_tme/raw/controls \
  --method wls --af-mode single \
  -o outputs/unmix/korotkaja
```

---

## 3. Multi-Component Autofluorescence (`baglio_2024_osteosarcoma_spectral` Complement)

For heterogeneous solid tumors or tissues containing multiple distinct autofluorescent cell populations (e.g., macrophages vs. stromal cells), set `--af-mode multi --multi-af-components 2` (or `3`). FlowX fits a GMM on the top-5 highest-variance optical channels of the unstained control and appends one normalized AF endmember per component to the reference matrix $M$.

---

## References & Scientific Standards

* **Spectral Flow Cytometry Overview:** [](doi:10.1002/cyto.a.23779)
* **OMIP-098 39-Color Full Spectrum Panel:** [](doi:10.1002/cyto.a.24789)
* **Spillover Spreading Matrix (SSM):** [](doi:10.1002/cyto.a.22251)
