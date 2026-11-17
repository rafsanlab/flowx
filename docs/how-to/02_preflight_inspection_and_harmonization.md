# 02. Pre-Flight Inspection & Panel Harmonization

Run `flowx inspect` before preprocessing a cohort to audit FCS 3.1 headers, catch missing spillover matrices or low-event tubes, verify cytometer modality, and optionally harmonize channel aliases across multi-center studies.

---

## 1. Inspect a Cohort Before Running the Pipeline

**Example — 24-Color Human PBMC Cohort (`conventional/arguello_2020_scenith_metabolism`):**
Inspecting `$CYT` (`NovoCyte Penteon`) and `$SPILLOVER` confirms instrument modality and control availability before execution:

```bash
flowx inspect data/conventional/arguello_2020_scenith_metabolism \
  --event-stats \
  -o outputs/inspect/arguello_2020
```

**Single-File Header Quick Check (`flowx info`):**
```bash
flowx info data/conventional/finak_2014_cytotrol_pbmc/raw/samples/CytoTrol_CytoTrol_1.fcs
```

**Generated Audit Artifacts (`outputs/inspect/arguello_2020/`):**
- `summary.csv`: Per-file `$TOT`, `$PAR`, `$CYT`, `$DATE`, `$BTIM`, `$SPILLOVER` presence, classified role, and QC pass/flag status.
- `channel_details.csv`: Per-channel `$PnN`, `$PnS`, `$PnV`, `$PnR`, `$PnG`.
- `inspection_report.md`: Human-readable cohort audit.
- `files_pass.txt` / `files_flagged.txt`: Partitioned file lists ready for `--files-list`.

---

## 2. Deduplicate Dual Pulse Detectors (`weber_2025_bk_polyomavirus` Complement)

When an instrument records both area and height for the same antigen (e.g., `FL5-A` and `FL5-H` both measuring `BKPyV_VP1`), `markers.csv` paired with `select_markers_from_metadata()` deduplicates both detectors to a single biological target during clustering, while `flowx inspect` flags detector-level mismatches upfront.

---

## 3. Harmonize Multi-Center Panel Aliases (`--harmonize`)

When combining FCS files across cytometers with inconsistent `$PnN` / `$PnS` naming (e.g., `CD3-FITC` vs `FITC-A (CD3)`), pass `-H` / `--harmonize`:

```bash
# Use built-in harmonization rules (configs/panel_harmonization.yaml)
flowx inspect data/my_cohort --harmonize DEFAULT -o outputs/inspect/harmonized

# Or supply a custom harmonization YAML
flowx inspect data/my_cohort -H configs/panel_harmonization.yaml -o outputs/inspect/harmonized
```

This emits `panel_harmonization_matrix.csv` and `panel_harmonization_report.md` in the inspection directory. In end-to-end YAML configs, enable harmonization via `input.harmonization_file` (or `input.harmonization: true`) and audit `01_time_qc/tables/panel_harmonization_mapping.csv`.

---

## References & Scientific Standards

* **FCS 3.1 Data File Standard:** [](doi:10.1002/cyto.a.20825)
* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
