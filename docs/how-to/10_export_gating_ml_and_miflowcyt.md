# 10. Export Gating-ML 2.0, Augmented FCS & MIFlowCyt Reports

Export every FlowX analysis for external verification in Gating-ML 2.0-compatible tools or regulatory review, and replay sessions deterministically.

---

## 1. Enable Interoperable Exports in YAML or CLI

All PBMC benchmark configs (e.g., `finak_2014_cytotrol_pbmc`, `white_2021_flowkit_ics`, `michaelis_2025_c9orf72_als_pbmc`) enable interoperable exports under `exports:`:

```yaml
exports:
  enabled: true
  export_augmented_fcs: true
  export_gating_ml: true
  generate_miflowcyt_report: true
```

**CLI Flags (`flowx pipeline` / `flowx cohort`):**
`--export-augmented-fcs`, `--export-gating-ml`, `--miflowcyt-report` (or `--no-*` counterparts to disable).

---

## 2. Generated Deliverables (`outputs/config_run/.../`)

| Deliverable | Output Path | Purpose |
| :--- | :--- | :--- |
| **Raw-FCS Preprocessing Gating-ML** | `exports/workspaces/preprocessing_gating_ml_2.0.xml` | Embeds spillover matrix, channel transforms, and inverse-transformed cleanup gates targeting raw `.fcs` files. |
| **Discovery Gating-ML** | `exports/workspaces/gating_ml_2.0.xml` | Downstream discovery & population hierarchy in ISAC Gating-ML 2.0 XML format. |
| **Augmented FCS 3.1 Files** | `exports/augmented_fcs/augmented_<sample_id>.fcs` | Clean events appended with numeric channels `cluster_id`, `UMAP_1`, `UMAP_2` (with biological `lineage_id` labels linked via `RangeGate`s in Gating-ML). |
| **MIFlowCyt Reports** | `exports/reports/miflowcyt_analysis_report.{md,html}` | Standalone styled HTML & Markdown reports covering MIFlowCyt §§1–5. |
| **Deterministic Replay** | `exports/workspaces/workspace.{json,xml}` | Replay saved sessions via `flowx replay <workspace_file>`. |

---

## 3. Replay a Saved Session

```bash
flowx replay outputs/config_run/conventional/finak_2014_cytotrol_pbmc/exports/workspaces/workspace.json
```

---

## References & Scientific Standards

* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
* **Gating-ML 2.0 Standard:** [](doi:10.1002/cyto.a.22690)
* **FCS 3.1 Data File Standard:** [](doi:10.1002/cyto.a.20825)
* **FlowKit Computational Toolkit:** [](doi:10.3389/fimmu.2021.768541)
