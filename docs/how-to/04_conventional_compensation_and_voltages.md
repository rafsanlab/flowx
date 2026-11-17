# 04. Conventional Spillover Compensation & PMT Voltage Mismatches

Use `flowx compensate` (or `02_compensation` in YAML) to calculate spillover matrices from single-stain controls, handle bead vs. cell carriers, audit the Spillover Spreading Matrix (SSM), and resolve PMT voltage discordance.

---

## 1. Spillover Source Priority

FlowX resolves spillover matrices in strict priority order:
1. **External CSV (`--matrix-csv` on `flowx compensate`)**
2. **Single-stain control directory (`raw/controls/` / `--comp-controls-dir`)** — prioritized over embedded headers whenever present
3. **Embedded FCS `$SPILLOVER` / `$SPILL` keyword (`source: embedded`)**
4. **None (`source: none`)**

---

## 2. Calculate Spillover from Single-Stain Controls

**Example — 24-Color Human PBMC (`conventional/arguello_2020_scenith_metabolism`) & 22-Color Whole Blood (`conventional/brouwers_2026_whole_blood_omip`):**

```bash
# Config-driven compensation
flowx compensate -c configs/config_run/conventional/arguello_2020_scenith_metabolism.yaml

# Standalone CLI using hybrid estimator (AutoSpill IRLS + Median ΔMFI fallback)
flowx compensate data/conventional/arguello_2020_scenith_metabolism/raw/samples \
  --comp-controls-dir data/conventional/arguello_2020_scenith_metabolism/raw/controls \
  --method hybrid \
  -o outputs/comp/arguello
```

FlowX automatically distinguishes `[BEADS]` vs `[CELLS]` carrier scatter profiles, applies control Time QC + singlet gating, selects internal negative vs. external unstained baselines to minimize post-compensation residual $|Z_{\text{res}}| < 0.5$, and writes `spillover_spreading_matrix_ssm.csv` and `post_comp_residual_z_scores.csv`.

---

## 3. Handle PMT Voltage Mismatches (`--voltage-policy`)

When single-stain compensation controls and biological samples were acquired at different detector `$PnV` voltages, FlowX's `audit_pmt_voltage_concordance()` compares every fluorescence detector's `$PnV` against the sample and applies the chosen policy (`--voltage-policy` on `flowx compensate`, `--voltage-mismatch-policy` on `flowx pipeline`, or `02_compensation.voltage_mismatch_policy` in YAML):

| Policy | Behavior when $|\Delta V| > \text{tolerance\_v}$ (default `5.0` V) |
| :--- | :--- |
| `warn` *(Default)* | Logs mismatch in `pmt_voltage_concordance_audit.csv` and proceeds with control matrix. |
| `fallback_embedded` | Replaces discordant channel rows with the sample's embedded `$SPILLOVER` values. |
| `scale_power_law` | Analytically scales spillover using $S^{\text{scaled}}_{i,j} = S_{i,j} (V_{\text{samp},j}/V_{\text{ctrl},j})^{7.2} (V_{\text{ctrl},i}/V_{\text{samp},i})^{7.2}$. |
| `strict_error` | Raises `ValueError` to halt unsafe runs. |

```bash
flowx compensate data/conventional/arguello_2020_scenith_metabolism/raw/samples \
  --comp-controls-dir data/conventional/arguello_2020_scenith_metabolism/raw/controls \
  --voltage-policy scale_power_law \
  --voltage-tolerance 5.0 \
  --override "B615-A->B525-A=0.145" \
  -o outputs/comp/arguello_scaled
```

---

## References & Scientific Standards

* **AutoSpill Compensation Framework:** [](doi:10.1038/s41467-021-23126-8)
* **Spillover Spreading Matrix (SSM):** [](doi:10.1002/cyto.a.22251)
* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
