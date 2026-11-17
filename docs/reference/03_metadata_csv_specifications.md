# 03. Metadata CSV & Cell Signature Specifications

Exact schemas for `metadata/samples.csv`, `metadata/markers.csv`, and `cell_signatures` parsed by `src/flowx/io/metadata.py` and `src/flowx/clustering/cluster.py`.

---

## 1. `metadata/samples.csv` Specification

Discovered automatically at `data/<platform>/<dataset_name>/metadata/samples.csv`.

| Column | Required | Format / Constraints |
| :--- | :-: | :--- |
| `sample_id` | Yes | Exact `.fcs` filename stem (1:1 with `raw/samples/*.fcs`). |
| `filename` | Yes | `<sample_id>.fcs`. |
| `relative_path` | Yes | Relative path from dataset root (e.g., `raw/samples/<file>.fcs`). |
| `condition` | Yes | Biological group / treatment label (no spaces recommended). |
| `batch_id` | Yes | Strict FCS-derived `YYYY-MM-DD_<Cytometer>` (never guessed from filenames). |
| `is_control` | Yes | Lowercase `true` or `false` (CytoNorm bridge anchor; **never** single-stain bead controls). |
| `provenance_source` | Yes | Citation (`FCS $TEXT`, `DOI:...`). |
| *Optional Covariates* | No | `replicate`, `specimen`, `donor_or_cell_line`, `acquisition_date`, `acquisition_time`, `cytometer`, `total_events`, `paired_id`. |

---

## 2. `metadata/markers.csv` Specification

Keyed 1 row per physical `$PnN` channel in the `.fcs` file.

| Column | Required | Format / Constraints |
| :--- | :-: | :--- |
| `channel_name` | Yes | Exact FCS `$PnN` string verbatim (e.g., `BV510-A`, `AF-A`). |
| `marker_name` | Yes | Standardized antigen symbol (`CD3`, `HLA-DR`), or `FSC-A`/`Time`. |
| `fluorophore` | Yes | Conjugate name (`PE-Cy7`, `BB515`) or `Scatter`/`Time`. |
| `clone_or_catalog` | Yes | Verified clone/catalog or `"Not specified in FCS or publication"`. |
| `marker_class` | Yes | Semicolon-delimited (`;`) controlled vocabulary from `VALID_MARKER_CLASSES`. |
| `cell_type_association` | No | Semicolon-delimited (`;`) expected cell lineages. |
| `provenance_source` | Yes | `FCS $PnS`, publication Methods, or verified single-stain peak. |

### Controlled Vocabularies (`src/flowx/io/metadata.py`)

* **`VALID_MARKER_CLASSES`:** `lineage`, `functional`, `viability`, `dump`, `scatter`, `time`, `autofluorescence`, `unused`, `index`, `qc`.
* **`NON_FLUORESCENCE_MARKER_CLASSES`:** `autofluorescence`, `index`, `qc`, `scatter`, `time`, `unused` (automatically excluded by `resolve_fluorescence_channels()`).

---

## 3. Declarative `cell_signatures` Schema (`src/flowx/clustering/cluster.py`)

Custom biological annotation rules can be passed as a JSON/YAML list of dictionaries. A rule only fires when **all** `required_positive` and `required_negative` markers are present in the panel:

```json
[
  {
    "name": "CD4+ Naive T Cells",
    "macro_group": "CD4+ T-Cell Lineage",
    "required_positive": ["CD3", "CD4", "CCR7", "CD45RA"],
    "required_negative": ["CD8"],
    "exclude_if_positive": ["CD14", "CD19"],
    "priority": 90
  }
]
```

---

## References & Scientific Standards

* **MIFlowCyt Standard:** [](doi:10.1002/cyto.a.20623)
* **FCS 3.1 Data File Standard:** [](doi:10.1002/cyto.a.20825)
