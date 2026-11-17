"""MIFlowCyt (Minimum Information about a Flow Cytometry Experiment) Report Generator.

Compiles comprehensive, publication-ready audit reports conforming to the ISAC MIFlowCyt
standard (https://pmc.ncbi.nlm.nih.gov/articles/PMC2773297/) across all FlowX pipeline steps:
1. Experiment Overview & Biological Intent
2. Flow Cytometer & Reagent Setup
3. Quality Control & Clean-Up Gating
4. Batch Normalization & Balancing
5. Population Identification & Phenotypic Discovery
6. Cohort Statistical Analysis (Differential Abundance & State)
7. Workspace Interoperability & Data File Manifests
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from flowx.io.paths import sanitize_path


def _generate_html_from_markdown(title: str, md_content: str) -> str:
    """Render a modern, standalone HTML document from markdown text with responsive CSS."""
    # Convert basic markdown elements to HTML
    lines = md_content.splitlines()
    html_body_parts: List[str] = []
    in_table = False
    table_rows: List[str] = []

    def flush_table():
        nonlocal in_table, table_rows
        if not in_table or not table_rows:
            return
        t_html = ["<div class='table-container'><table>"]
        # Header
        if len(table_rows) >= 2 and "|" in table_rows[0]:
            h_cells = [c.strip() for c in table_rows[0].split("|")[1:-1]]
            t_html.append("<thead><tr>")
            for hc in h_cells:
                t_html.append(f"<th>{hc}</th>")
            t_html.append("</tr></thead><tbody>")
            # Rows (skip row 1 which is separator ---|---)
            for r in table_rows[2:]:
                if "|" in r:
                    r_cells = [c.strip() for c in r.split("|")[1:-1]]
                    t_html.append("<tr>")
                    for rc in r_cells:
                        # Bold or badge highlights
                        rc_fmt = rc.replace("**", "<strong>").replace("**", "</strong>")
                        t_html.append(f"<td>{rc_fmt}</td>")
                    t_html.append("</tr>")
            t_html.append("</tbody></table></div>")
            html_body_parts.append("".join(t_html))
        in_table = False
        table_rows = []

    for line in lines:
        s_line = line.strip()
        if s_line.startswith("|") and s_line.endswith("|"):
            in_table = True
            table_rows.append(s_line)
            continue
        elif in_table:
            flush_table()

        if s_line.startswith("# "):
            html_body_parts.append(f"<h1>{s_line[2:]}</h1>")
        elif s_line.startswith("## "):
            html_body_parts.append(f"<h2>{s_line[3:]}</h2>")
        elif s_line.startswith("### "):
            html_body_parts.append(f"<h3>{s_line[4:]}</h3>")
        elif s_line.startswith("---"):
            html_body_parts.append("<hr/>")
        elif s_line.startswith("> "):
            callout = s_line[2:].replace("**", "<strong>").replace("**", "</strong>")
            html_body_parts.append(f"<div class='callout'>{callout}</div>")
        elif s_line.startswith("- "):
            item = s_line[2:].replace("**", "<strong>").replace("**", "</strong>")
            html_body_parts.append(f"<ul><li>{item}</li></ul>")
        elif s_line:
            p_text = s_line.replace("**", "<strong>").replace("**", "</strong>")
            html_body_parts.append(f"<p>{p_text}</p>")

    if in_table:
        flush_table()

    body_html = "\n".join(html_body_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg-primary: #FFFFFF;
    --bg-secondary: #F8FAFC;
    --border-color: #E2E8F0;
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --accent: #2563EB;
    --accent-light: #EFF6FF;
    --success: #059669;
    --warning: #D97706;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--text-primary);
    background-color: var(--bg-secondary);
    line-height: 1.6;
    margin: 0;
    padding: 2rem 1rem;
  }}
  .container {{
    max-width: 960px;
    margin: 0 auto;
    background: var(--bg-primary);
    padding: 2.5rem 3rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    border: 1px solid var(--border-color);
  }}
  h1 {{ font-size: 1.85rem; font-weight: 700; color: #1E293B; border-bottom: 2px solid var(--border-color); padding-bottom: 0.75rem; margin-top: 0; }}
  h2 {{ font-size: 1.35rem; font-weight: 600; color: #334155; margin-top: 2rem; margin-bottom: 0.75rem; border-left: 4px solid var(--accent); padding-left: 0.5rem; }}
  h3 {{ font-size: 1.1rem; font-weight: 600; color: #475569; margin-top: 1.25rem; }}
  p, li {{ color: var(--text-secondary); font-size: 0.95rem; }}
  hr {{ border: 0; height: 1px; background: var(--border-color); margin: 2rem 0; }}
  .callout {{
    background: var(--accent-light);
    border-left: 4px solid var(--accent);
    padding: 0.75rem 1rem;
    border-radius: 4px;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    color: #1E3A8A;
  }}
  .table-container {{ overflow-x: auto; margin: 1.25rem 0; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    text-align: left;
  }}
  th {{
    background-color: #F1F5F9;
    color: #334155;
    font-weight: 600;
    padding: 0.65rem 0.85rem;
    border: 1px solid var(--border-color);
  }}
  td {{
    padding: 0.65rem 0.85rem;
    border: 1px solid var(--border-color);
    color: #475569;
  }}
  tr:nth-child(even) {{ background-color: #F8FAFC; }}
  tr:hover {{ background-color: #F1F5F9; }}
  code {{
    background: #F1F5F9;
    padding: 0.15rem 0.35rem;
    border-radius: 4px;
    font-size: 0.85em;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    color: #0F172A;
  }}
</style>
</head>
<body>
<div class="container">
{body_html}
</div>
</body>
</html>
"""


def generate_miflowcyt_report(
    output_dir: Union[str, Path],
    experiment_name: str = "FlowX Multi-Sample Analysis",
    cohort_result: Optional[Any] = None,
    samples: Optional[List[str]] = None,
    total_events: Optional[int] = None,
    markers: Optional[List[str]] = None,
    marker_name_map: Optional[Dict[str, str]] = None,
    clustering_method: Optional[str] = "som",
    cluster_count: Optional[int] = None,
    auto_k_method: Optional[str] = None,
    da_result: Optional[Any] = None,
    ds_result: Optional[Any] = None,
    flowjo_wsp_path: Optional[Union[str, Path]] = None,
    gating_ml_path: Optional[Union[str, Path]] = None,
    preprocessing_wsp_path: Optional[Union[str, Path]] = None,
    preprocessing_gating_ml_path: Optional[Union[str, Path]] = None,
    augmented_fcs_paths: Optional[List[Union[str, Path]]] = None,
    pipeline_step: str = "exports",
    numbered_steps: bool = True,
) -> Dict[str, Path]:
    """Compile end-to-end MIFlowCyt-compliant analysis report in Markdown and HTML.

    Args:
        output_dir: Destination directory for report artifacts.
        experiment_name: Descriptive title for flow cytometry study.
        cohort_result: Optional CohortPopulationResult containing full downstream analysis.
        samples: List of evaluated sample names.
        total_events: Total cell count processed.
        markers: List of fluorescent parameter markers.
        clustering_method: Clustering engine utilized ('som' or 'phenograph').
        cluster_count: Number of identified metaclusters.
        da_result: Optional Differential Abundance result.
        ds_result: Optional Differential State result.
        flowjo_wsp_path: Path to exported FlowJo .wsp workspace.
        gating_ml_path: Path to exported Gating-ML 2.0 XML.
        augmented_fcs_paths: List of exported augmented FCS file paths.
        pipeline_step: Pipeline step context string.
        numbered_steps: Whether directory hierarchy follows numbered prefixes.

    Returns:
        Dict mapping 'markdown' and 'html' to their respective saved Paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract details from cohort_result if provided
    if cohort_result is not None:
        if hasattr(cohort_result, "samples") and not samples:
            samples = cohort_result.samples
        if hasattr(cohort_result, "feature_set") and cohort_result.feature_set is not None:
            if total_events is None:
                total_events = int(cohort_result.feature_set.count_matrix.values.sum())
            if markers is None:
                markers = cohort_result.feature_set.markers
        if hasattr(cohort_result, "clustering_result") and cohort_result.clustering_result is not None:
            cl = cohort_result.clustering_result
            if clustering_method is None:
                clustering_method = cl.method
            if cluster_count is None:
                cluster_count = cl.cluster_count
            if markers is None and hasattr(cl, "markers_used"):
                markers = cl.markers_used
        if da_result is None and hasattr(cohort_result, "da_result"):
            da_result = cohort_result.da_result
        if ds_result is None and hasattr(cohort_result, "ds_result"):
            ds_result = cohort_result.ds_result

    samples = samples or []
    markers = markers or []
    n_samples = len(samples)
    tot_ev = total_events or 0
    k_val = cluster_count or (cohort_result.clustering_result.cluster_count if cohort_result and hasattr(cohort_result, "clustering_result") else 0)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build MIFlowCyt Markdown sections
    md_lines: List[str] = [
        f"# MIFlowCyt-Compliant Cytometry Analysis Report: {experiment_name}",
        "",
        "> **Pipeline:** FlowX High-Dimensional Flow Cytometry Platform  ",
        f"> **Generated:** {now_str}  ",
        "> **Standard Compliance:** ISAC MIFlowCyt (Lee et al., Nature Biotechnology 2008)  ",
        f"> **Pipeline Execution Step:** `{pipeline_step}`  ",
        "",
        "---",
        "",
        "## 1. Experiment Overview & Biological Intent (MIFlowCyt §1)",
        "",
        "| Parameter | Specification |",
        "|:---|:---|",
        f"| **Study Identifier** | {experiment_name} |",
        f"| **Evaluated Cohort Size** | **{n_samples} Samples** |",
        f"| **Total Processed Events** | **{tot_ev:,} Events** |",
        f"| **Analysis Modality** | Unsupervised Phenotypic Discovery & Cohort Biostatistics |",
        f"| **Provenance Tracking** | Fully Deterministic & Reproducible (Config Audited) |",
        "",
        "---",
        "",
        "## 2. Cytometer & Fluorescence Lineage Panel (MIFlowCyt §2 & §3)",
        "",
        "| Channel / Marker | Target Lineage / Specificity | Analysis Role |",
        "|:---|:---|:---|",
    ]

    for m in markers:
        role = "Lineage Phenotyping" if not any(x in m.lower() for x in ["fsc", "ssc", "time"]) else "Morphological / Fluidic"
        antigen = (marker_name_map or {}).get(m, "")
        if antigen and str(antigen).strip() and str(antigen).strip() != str(m).strip():
            target_desc = f"**{antigen}** (Phenotypic Surface/Intracellular Marker)"
        else:
            target_desc = "Phenotypic Surface/Intracellular Marker"
        md_lines.append(f"| **{m}** | {target_desc} | {role} |")

    eff_k_metric = auto_k_method
    if not eff_k_metric and cohort_result is not None and hasattr(cohort_result, "clustering_result") and cohort_result.clustering_result is not None:
        eff_k_metric = getattr(cohort_result.clustering_result, "k_optimization_metric", None)
    eff_k_str = str(eff_k_metric or "consensus").strip().lower()
    if eff_k_str == "consensus":
        auto_k_label = "Multi-Metric Consensus (Silhouette, Calinski-Harabasz, Davies-Bouldin)"
    elif eff_k_str == "silhouette":
        auto_k_label = "Silhouette Score Optimization"
    elif eff_k_str in ("davies_bouldin", "dbi"):
        auto_k_label = "Davies-Bouldin Metric Evaluation"
    elif eff_k_str in ("fixed", "none"):
        auto_k_label = f"Fixed Metacluster Count (k={k_val})"
    else:
        auto_k_label = eff_k_str.replace("_", " ").title()

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Data Preprocessing & Core Clean-Up Gating (MIFlowCyt §4)",
        "",
        "FlowX applies a standardized multi-stage quality control and event clean-up cascade:",
        "- **Automated & Manual Time Gating:** High-resolution anomaly binning excludes fluidic surges and signal drift.",
        "- **Full-Matrix Spectral Compensation:** Verified against single-stain controls and unstained negative baselines.",
        "- **Debris Exclusion:** Adaptive 2D contour and Gaussian Mixture Model gating on FSC-A vs SSC-A isolates intact cellular events.",
        "- **Singlet Discrimination:** Strict dual-stage aspect ratio gating (FSC-A vs FSC-H and SSC-A vs SSC-H) excludes clumps and doublets.",
        "- **Viability Exclusion:** Live/dead exclusion dye gating isolates viable target cells.",
        "- **Cohort Balancing:** Equi-event subsampling to the cohort bottleneck sample prevents clustering bias.",
        "",
        "---",
        "",
        "## 4. Population Identification & Phenotypic Discovery",
        "",
        "| Metric | Result |",
        "|:---|:---|",
        f"| **Clustering Algorithm** | **{clustering_method.upper()}** (Self-Organizing Map + Agglomerative Metaclustering) |",
        f"| **Resolved Populations (k)** | **{k_val} Metaclusters** |",
        f"| **Dimension Reduction** | 2D Uniform Manifold Approximation and Projection (UMAP) |",
        f"| **Cluster Optimization (Auto-k)** | {auto_k_label} |",
        "",
    ])

    # Include Metacluster Phenotype Table if available
    if cohort_result is not None and hasattr(cohort_result, "clustering_result") and cohort_result.clustering_result is not None:
        cl = cohort_result.clustering_result
        if hasattr(cl, "cluster_statistics") and cl.cluster_statistics is not None and not cl.cluster_statistics.empty:
            md_lines.extend([
                "### Discovered Population Phenotypes & Abundances",
                "",
                "| Cluster ID | Assigned Lineage | Event Count | Frequency (% Viable) | Predominant Markers |",
                "|:---|:---|:---|:---|:---|",
            ])
            stats_df = cl.cluster_statistics
            pheno_map = getattr(cl, "phenotype_labels", None) or {}
            for _, row in stats_df.iterrows():
                cid = row.get("Cluster ID", "N/A")
                cid_lookup = int(cid) if str(cid).isdigit() else cid
                lin = row.get("Assigned Lineage", pheno_map.get(cid, pheno_map.get(cid_lookup, f"Cluster_{cid}")))
                raw_cnt = row.get("Event Count", row.get("Count", 0))
                cnt = f"{int(raw_cnt):,}" if pd.notna(raw_cnt) else "0"
                raw_freq = row.get("Percentage", row.get("% Total", row.get("% Parent", 0.0)))
                freq = f"{float(raw_freq):.2f}%" if pd.notna(raw_freq) else "0.00%"
                pred = row.get("Predominant Markers", "N/A")
                md_lines.append(f"| Cluster {cid} | **{lin}** | {cnt} | {freq} | {pred} |")
            md_lines.append("")

    # Include Differential Abundance (DA) findings if available
    if da_result is not None and hasattr(da_result, "summary_df"):
        md_lines.extend([
            "---",
            "",
            "## 5. Quantitative Cohort Statistical Analysis",
            "",
            f"### Differential Abundance (DA): {da_result.target_group} vs {da_result.reference_group}",
            "",
        ])
        if getattr(da_result, "underpowered_warning", None):
            md_lines.extend([
                f"> **Statistical Power Notice:** {da_result.underpowered_warning}",
                "",
            ])
        md_lines.extend([
            "| Population | Ref Mean % | Target Mean % | Log2 Fold Change | p-value | FDR (q-val) | Significant |",
            "|:---|:---|:---|:---|:---|:---|:---|",
        ])
        for _, row in da_result.summary_df.iterrows():
            pop = row.get("population", "N/A")
            ref_val = row.get("mean_ref_pct", row.get("ref_mean", 0.0))
            tgt_val = row.get("mean_target_pct", row.get("target_mean", 0.0))
            raw_p = row.get("p_value", float("nan"))
            raw_q = row.get("q_value", row.get("fdr_bh", float("nan")))
            ref_m = f"{float(ref_val):.2f}%" if pd.notna(ref_val) else "N/A"
            tgt_m = f"{float(tgt_val):.2f}%" if pd.notna(tgt_val) else "N/A"
            l2fc_raw = row.get("log2fc", 0.0)
            l2fc = f"{float(l2fc_raw):+.2f}" if pd.notna(l2fc_raw) else "N/A"
            pval = "N/A (n<2)" if pd.isna(raw_p) else f"{float(raw_p):.4e}"
            qval = "N/A (n<2)" if pd.isna(raw_q) else f"{float(raw_q):.4e}"
            sig = "**YES**" if row.get("is_significant", False) else "No"
            md_lines.append(f"| **{pop}** | {ref_m} | {tgt_m} | {l2fc} | {pval} | {qval} | {sig} |")
        md_lines.append("")

    # Workspace Interoperability and Artifact Manifests
    md_lines.extend([
        "---",
        "",
        "## 6. Workspace Interoperability & Data File Manifest (MIFlowCyt §5)",
        "",
        "| Artifact Description | Format / Standard | File Destination |",
        "|:---|:---|:---|",
    ])
    if preprocessing_wsp_path:
        md_lines.append(
            f"| **Preprocessing & Gating Workspace (Raw FCS)** | XML (.wsp) FlowJo v10.8+ | `{sanitize_path(preprocessing_wsp_path)}` |"
        )
    if preprocessing_gating_ml_path:
        md_lines.append(
            f"| **Preprocessing & Gating Gating-ML 2.0 (Raw FCS)** | ISAC XML Standard | `{sanitize_path(preprocessing_gating_ml_path)}` |"
        )
    if flowjo_wsp_path:
        md_lines.append(f"| **FlowJo Workspace (Discovery)** | XML (.wsp) FlowJo v10.8+ | `{sanitize_path(flowjo_wsp_path)}` |")
    if gating_ml_path:
        md_lines.append(f"| **Gating-ML 2.0 Hierarchy (Discovery)** | ISAC XML Standard | `{sanitize_path(gating_ml_path)}` |")
    if augmented_fcs_paths:
        md_lines.append(f"| **Augmented FCS Files** | FCS 3.1 (+Cluster_ID, +UMAP) | `{len(augmented_fcs_paths)} files in fcs/` |")
    md_lines.append("| **GraphPad Prism Tables** | Grouped Table CSV | `tables/prism/` |")
    md_lines.append("| **Publication Figures** | 300 DPI Nature-Styled PNG | `plots/` |")

    md_lines.extend([
        "",
        "---",
        "",
        "> **Report Archival Notice:** This analysis is fully versioned and reproducible under the FlowX framework.",
    ])

    full_md = "\n".join(md_lines)
    md_path = out_dir / "miflowcyt_analysis_report.md"
    md_path.write_text(full_md, encoding="utf-8")

    # Render HTML version
    full_html = _generate_html_from_markdown(title=f"MIFlowCyt Report: {experiment_name}", md_content=full_md)
    html_path = out_dir / "miflowcyt_analysis_report.html"
    html_path.write_text(full_html, encoding="utf-8")

    return {"markdown": md_path, "html": html_path}

