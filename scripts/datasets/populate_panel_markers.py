#!/usr/bin/env python3
"""Populate panel_markers.csv and update panel.csv with publication-derived biological marker names."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

base_data = Path("data")

# 1. spectral_scenith_metabolism
scenith_markers = [
    ("B525-A", "Puromycin", "AF488"),
    ("B615-A", "CD20", "NFB 610_70S"),
    ("B695-A", "FoxP3", "BB700"),
    ("B725-A", "CD19", "PerCP_eFluor 710"),
    ("R667-A", "CD127", "JF 646"),
    ("R725-A", "CCR6", "AF700"),
    ("R780-A", "CD45RA", "APC_Cy7"),
    ("UV445-A", "Viability", "BUV395"),
    ("UV525-A", "CD27", "BUV496"),
    ("UV586-A", "CXCR5", "BUV563"),
    ("UV615-A", "CD56", "BUV615"),
    ("UV667-A", "CRTh2", "BUV661"),
    ("UV725-A", "IgD", "BUV737"),
    ("V445-A", "CD25", "BV421"),
    ("V586-A", "CD3", "BV570"),
    ("V615-A", "CD16", "BV605"),
    ("V667-A", "CD28", "BV650"),
    ("V725-A", "CD14", "BV711"),
    ("V780-A", "CCR7", "BV786"),
    ("Y586-A", "CD24", "PE"),
    ("Y615-A", "CXCR3", "PE_Dazzle 594"),
    ("Y667-A", "CD38", "PE_Cy5"),
    ("Y725-A", "CD4", "PE_AF700"),
    ("Y780-A", "TCRgd", "PE_Cy7"),
]

# 2. spectral_crc_til_mucosal
crc_markers = [
    ("BUV395-A", "CD39", "BUV395"),
    ("BUV496-A", "CD45", "BUV496"),
    ("BUV563-A", "CD69", "BUV563"),
    ("BUV615-A", "TCRgd", "BUV615"),
    ("BUV661-A", "CD21", "BUV661"),
    ("BUV737-A", "CD127", "BUV737"),
    ("BUV805-A", "CD16", "BUV805"),
    ("BV421-A", "TIGIT", "BV421"),
    ("V500-A", "CD3", "V500"),
    ("BV570-A", "CD19", "BV570"),
    ("BV605-A", "CD103", "BV605"),
    ("BV650-A", "CD25", "BV650"),
    ("BV711-A", "CD56", "BV711"),
    ("BV750-A", "PD-L1", "BV750"),
    ("BV786-A", "CD38", "BV786"),
    ("FITC-A", "CD4", "FITC"),
    ("rb545-A", "CD45RA", "RB545"),
    ("RB613-A", "CTLA-4", "RB613"),
    ("RB670-A", "CD161", "RB670"),
    ("RealBlue 705-A", "TIM-3", "RealBlue 705"),
    ("RealBlue_705", "TIM-3", "RealBlue 705"),
    ("RB744-A", "IgD", "RB744"),
    ("RB780-A", "NKG2A", "RB780"),
    ("PE-A", "CD73", "PE"),
    ("RY586-A", "KLRG1", "RY586"),
    ("RY610-A", "NKG2D", "RY610"),
    ("PE-Cy5-A", "CD45RO", "PE-Cy5"),
    ("RY775-A", "CD27", "RY775"),
    ("APC-A", "CD137", "APC"),
    ("r718-A", "PD-1", "R718"),
    ("Zombie NIR-A", "Viability", "Zombie NIR"),
    ("APC-Cy7-A", "CD8", "APC-Cy7"),
    ("ry 703-A", "LAG-3", "RY703"),
    ("LAG-3 RY 703 (Beads)", "LAG-3", "RY703"),
]

# 3. spectral_mouse_tumor_aurora
mouse_tumor_markers = [
    ("Spark UV 387-A", "CD90.2", "Spark UV 387"),
    ("BUV395-A", "CD45.2", "BUV395"),
    ("LIVE DEAD Blue-A", "Viability", "LIVE DEAD Blue"),
    ("BUV496-A", "MHC-II", "BUV496"),
    ("BUV563-A", "CD86", "BUV563"),
    ("BUV615-A", "CD8a", "BUV615"),
    ("BUV661-A", "CD69", "BUV661"),
    ("BUV737-A", "PD-1", "BUV737"),
    ("BUV805-A", "CD49a", "BUV805"),
    ("BV421-A", "GzmB", "BV421"),
    ("Super Bright 436-A", "SIRPa", "Super Bright 436"),
    ("Pacific Blue-A", "TCF1", "Pacific Blue"),
    ("BV480-A", "PD-L1", "BV480"),
    ("violet fluor 500-A", "CD4", "violet fluor 500"),
    ("BV570-A", "CD11b", "BV570"),
    ("BV605-A", "Tim-3", "BV605"),
    ("BV650-A", "F4/80", "BV650"),
    ("BV711-A", "CD62L", "BV711"),
    ("BV750-A", "Siglec-F", "BV750"),
    ("BV785-A", "Xcr1", "BV785"),
    ("Alexa Fluor 488-A", "Ki-67", "Alexa Fluor 488"),
    ("RB 545-A", "CD14", "RB 545"),
    ("Spark Blue 574-A", "TCRgd", "Spark Blue 574"),
    ("RealBlue 613-A", "CXCR5", "RealBlue 613"),
    ("PerCP-A", "B220", "PerCP"),
    ("BB700-A", "CCR7", "BB700"),
    ("PerCP-eFluor 710-A", "FoxP3", "PerCP-eFluor 710"),
    ("RB744-A", "NK1.1", "RB744"),
    ("NovaFluor Blue 800-A", "CD44", "NovaFluor Blue 800"),
    ("PE-A", "CX3CR1", "PE"),
    ("PE-eFluor 610-A", "Eomes", "PE-eFluor 610"),
    ("PE-Fire 640-A", "ICOS", "PE-Fire 640"),
    ("PE-Cy5-A", "CD11c", "PE-Cy5"),
    ("PE-Fire 700-A", "CD206", "PE-Fire 700"),
    ("PE Fire810-A", "T-bet", "PE Fire810"),
    ("APC-A", "CXCR3", "APC"),
    ("Alexa Fluor 647-A", "Ly6C", "Alexa Fluor 647"),
    ("Spark NIR 685-A", "CCR8", "Spark NIR 685"),
    ("Alexa Fluor 700-A", "CD103", "Alexa Fluor 700"),
    ("APC-Fire 750-A", "CXCR6", "APC-Fire 750"),
    ("APC-Fire 810-A", "Ly6G", "APC-Fire 810"),
]

# 4. spectral_ipsc_organoid_ccl5
ipsc_organoid_markers = [
    ("APC-A", "CD45", "APC"),
    ("FITC-A", "CD34", "FITC"),
    ("PerCP-A", "CD235a", "PerCP"),
    ("PE-A", "CD14", "PE"),
    ("PE-Dazzle 594-A", "CD90", "PE-Dazzle 594"),
    ("PE-Cy7-A", "Lineage", "PE-Cy7"),
    ("PE-Fire 700-A", "CD41", "PE-Fire 700"),
    ("Alexa Fluor 700-A", "CD42b", "Alexa Fluor 700"),
    ("BV650-A", "CD16", "BV650"),
    ("BV711-A", "CD31", "BV711"),
    ("BV785-A", "CD11b", "BV785"),
    ("Pacific Blue-A", "CD271", "Pacific Blue"),
    ("Zombie Aqua-A", "Viability", "Zombie Aqua"),
]

# 5. spectral_healthy_pbmc_cytopheno
healthy_pbmc_markers = [
    ("BUV395-A", "CD4", "BUV395"),
    ("BUV496-A", "CD56", "BUV496"),
    ("BUV563-A", "CD3", "BUV563"),
    ("BUV661-A", "CD19", "BUV661"),
    ("BUV805-A", "CD8", "BUV805"),
    ("BV421-A", "CD103", "BV421"),
    ("Pacific Blue-A", "FOXP3", "Pacific Blue"),
    ("BV510-A", "CD14", "BV510"),
    ("BV605-A", "CD69", "BV605"),
    ("BV650-A", "CCR6", "BV650"),
    ("BV711-A", "Perforin", "BV711"),
    ("Alexa Fluor 488-A", "HELIOS", "Alexa Fluor 488"),
    ("Alexa Fluor 700-A", "CCR7", "Alexa Fluor 700"),
    ("Spark Blue 550-A", "CD45", "Spark Blue 550"),
    ("PerCP-eFluor 710-A", "TIGIT", "PerCP-eFluor 710"),
    ("PE-A", "CD25", "PE"),
    ("PE-Dazzle594-A", "CD39", "PE-Dazzle 594"),
    ("PE-Cy5-A", "CD45RA", "PE-Cy5"),
    ("PE-Cy7-A", "PD1", "PE-Cy7"),
    ("APC-A", "CCR4", "APC"),
    ("APC-Fire 750-A", "GZMB", "APC-Fire 750"),
]

datasets = [
    ("conventional/arguello_2020_scenith_metabolism", scenith_markers),
    ("spectral/de_vries_2023_crc_til_mucosal", crc_markers),
    ("spectral/ghilas_2023_mouse_tumor_aurora", mouse_tumor_markers),
    ("spectral/matsumoto_2026_ipsc_organoid_ccl5", ipsc_organoid_markers),
    ("spectral/albers_2025_healthy_pbmc_cytopheno", healthy_pbmc_markers),
]

for dname, markers in datasets:
    meta_p = base_data / dname / "metadata"
    meta_p.mkdir(parents=True, exist_ok=True)
    csv_p = meta_p / "panel_markers.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel", "marker", "fluorophore"])
        for ch, m, fl in markers:
            w.writerow([ch, m, fl])
    print(f"Wrote {csv_p} ({len(markers)} markers)")

    # Also update panel.csv with biological marker
    panel_p = meta_p / "panel.csv"
    lookup = {ch: (m, fl) for ch, m, fl in markers}
    rows = [["channel", "marker", "fluorophore"]]
    for ch, m, fl in markers:
        rows.append([ch, m, fl])
    with open(panel_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"Updated {panel_p}")

print("All panel markers successfully populated!")

