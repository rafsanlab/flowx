"""Central Registry and Metadata Catalog for FlowX Benchmark Datasets.

Defines the complete catalogue of 20 benchmark flow cytometry datasets (10 conventional,
10 full spectrum), including source repositories, biology, cytometers, markers,
and recommended use cases (tutorials, workspace testing, batch normalization, clustering, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetInfo:
    """Metadata specification for a flow cytometry benchmark dataset."""

    id: str
    name: str
    category: str  # "conventional" or "spectral"
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    organism: str = ""
    specimen: str = ""
    cytometer: str = ""
    colors: str = ""
    repository: str = ""
    accession: str = ""
    publication: str = ""
    doi: str = ""
    recommended_for: str = ""
    files_expected: int = 0
    size_mb_approx: float = 0.0

    def matches_query(self, query: str) -> bool:
        """Check if a query string matches ID, name, or any alias."""
        q = query.strip().lower()
        if not q:
            return False
        if q in (self.id.lower(), self.name.lower()):
            return True
        return any(q == alias.lower() for alias in self.aliases)


BENCHMARK_DATASETS: Dict[str, DatasetInfo] = {
    # =========================================================================
    # Conventional Flow Cytometry Benchmark Datasets (10 Datasets)
    # =========================================================================
    "brouwers_2026_whole_blood_omip": DatasetInfo(
        id="brouwers_2026_whole_blood_omip",
        name="Brouwers Whole Blood OMIP (2026)",
        category="conventional",
        aliases=["brouwers", "omip_blood", "whole_blood", "spaceman", "1_conv"],
        description="Fresh human whole blood 25-color OMIP benchmark with 23 single stains and FlowJo workspace",
        organism="Human",
        specimen="Fresh human whole blood",
        cytometer="BD FACSymphony (5 lasers: UV-V-B-YG-R)",
        colors="25 colors",
        repository="Zenodo",
        accession="10.5281/zenodo.21869988",
        publication="Cytometry Part A (OMIP submission 2026)",
        doi="https://doi.org/10.5281/zenodo.21869988",
        recommended_for="Multi-Laser Spillover & SSM Calibration",
        files_expected=24,
        size_mb_approx=654.1,
    ),
    "chen_2025_placenta_hspc": DatasetInfo(
        id="chen_2025_placenta_hspc",
        name="Chen Embryonic Placenta & Liver HSPC (2025)",
        category="conventional",
        aliases=["placenta", "chen", "singh", "hspc", "2_conv"],
        description="Murine embryonic placenta & fetal liver HSPC lineage (6-color + wsp)",
        organism="Mouse",
        specimen="E12.5 embryonic placenta & fetal liver",
        cytometer="BD LSR II",
        colors="6 colors + 7-AAD",
        repository="Figshare / Zenodo",
        accession="10.6084/m9.figshare.27271968",
        publication="PLOS Biology (2025) 23(1):e3003003",
        doi="https://doi.org/10.1371/journal.pbio.3003003",
        recommended_for="Documentation & Clean Scatter Plot Figures",
        files_expected=16,
        size_mb_approx=182.8,
    ),
    "coburn_2021_thp1_stiffness": DatasetInfo(
        id="coburn_2021_thp1_stiffness",
        name="Coburn THP-1 Macrophage Hydrogel Stiffness (2021)",
        category="conventional",
        aliases=["thp1", "coburn", "wang", "stiffness", "3_conv"],
        description="Human THP-1 macrophages on tunable stiffness hydrogels (4-color, Figshare)",
        organism="Human",
        specimen="THP-1 macrophages on glycol-chitosan hydrogels",
        cytometer="BD LSRFortessa / FACSCanto II",
        colors="4 colors (CD11b, CD33, CD80, CD206)",
        repository="Figshare",
        accession="10.6084/m9.figshare.14450370",
        publication="J Biomed Mater Res A (2021) 109(8):1337-1352",
        doi="https://doi.org/10.1002/jbm.a.37125",
        recommended_for="Interactive Tutorial & Quickstart (< 15s execution)",
        files_expected=9,
        size_mb_approx=6.9,
    ),
    "crespo_2026_crispr_cd8_tcells": DatasetInfo(
        id="crespo_2026_crispr_cd8_tcells",
        name="Crespo In Vivo CRISPR CD8+ T Cells (2026)",
        category="conventional",
        aliases=["crespo", "crispr", "cd8", "4_conv"],
        description="Murine intratumoral CD8+ T cells under in vivo CRISPR knockouts",
        organism="Mouse",
        specimen="Intratumoral CD8+ OT-I T cells",
        cytometer="BD LSRFortessa (FACSDiva v8.0.2)",
        colors="T-cell activation/exhaustion panel",
        repository="Zenodo",
        accession="10.5281/zenodo.20526379",
        publication="Nature Immunology / Ludwig Lausanne (2026)",
        doi="https://doi.org/10.5281/zenodo.20526379",
        recommended_for="In Vivo Perturbation & Differential State (DS)",
        files_expected=8,
        size_mb_approx=684.3,
    ),
    "finak_2014_cytotrol_pbmc": DatasetInfo(
        id="finak_2014_cytotrol_pbmc",
        name="Finak Cyto-Trol PBMC Reference Standard (2014)",
        category="conventional",
        aliases=["cytotrol", "finak", "opencyto", "5_conv"],
        description="Human PBMC Cyto-Trol reference standard with FlowJo manual.xml",
        organism="Human",
        specimen="Lyophilized PBMC reference standard",
        cytometer="BD LSRFortessa / FACSCanto II",
        colors="7-color canonical T-cell panel",
        repository="GitHub / RGLab",
        accession="flowWorkspaceData",
        publication="PLOS Computational Biology (2014) 10(8):e1003806",
        doi="https://doi.org/10.1371/journal.pcbi.1003806",
        recommended_for="Gold-Standard Gating Validation against manual.xml",
        files_expected=2,
        size_mb_approx=10.8,
    ),
    "li_2025_caerin_macrophage": DatasetInfo(
        id="li_2025_caerin_macrophage",
        name="Li Caerin Peptide Macrophage Immunotherapy (2025)",
        category="conventional",
        aliases=["caerin", "li", "huang", "6_conv"],
        description="Murine tumor-infiltrating macrophages under caerin peptide (8-color)",
        organism="Mouse",
        specimen="B16 melanoma tumor-infiltrating leukocytes",
        cytometer="BD FACSAria II",
        colors="8 colors (CD45.2, CD11b, Ly6C, F4/80, CD24, IFNAR1, CD3e, B220)",
        repository="Zenodo",
        accession="10.5281/zenodo.13976956",
        publication="Scientific Reports (2025) 15:3789",
        doi="https://doi.org/10.1038/s41598-025-87687-0",
        recommended_for="High-Powered Differential Abundance (DA) (n=29)",
        files_expected=38,
        size_mb_approx=220.6,
    ),
    "quintana_2026_skull_marrow_chimera": DatasetInfo(
        id="quintana_2026_skull_marrow_chimera",
        name="Quintana Skull Marrow Chimera (2026)",
        category="conventional",
        aliases=["quintana", "skull", "chimera", "7_conv"],
        description="Murine skull marrow chimera with full single stains, unstained baseline & Live/Dead FMO",
        organism="Mouse",
        specimen="Skull bone marrow & meningeal B cells",
        cytometer="BD LSRFortessa (Multi-Laser)",
        colors="11-marker broad panel",
        repository="Zenodo",
        accession="10.5281/zenodo.22808501",
        publication="PLOS Biology (2023) 21(11):e3002389",
        doi="https://doi.org/10.1371/journal.pbio.3002389",
        recommended_for="Empirical FMO Calibration & Boundary Setting",
        files_expected=12,
        size_mb_approx=27.2,
    ),
    "quintelier_2023_cytonorm_meso": DatasetInfo(
        id="quintelier_2023_cytonorm_meso",
        name="Quintelier CytoNorm 2.0 Multi-Batch Benchmark (2023)",
        category="conventional",
        aliases=["cytonorm", "quintelier", "saeys", "8_conv"],
        description="CytoNorm 2.0 multi-batch human PBMC longitudinal benchmark across 3 batches",
        organism="Human",
        specimen="Cryopreserved mesothelioma PBMCs",
        cytometer="BD LSR II / FACSCanto II",
        colors="Longitudinal immune monitoring panel",
        repository="Zenodo",
        accession="10.5281/zenodo.13928969",
        publication="J ImmunoTherapy Cancer (2023) 11:e007070",
        doi="https://doi.org/10.1136/jitc-2023-007070",
        recommended_for="Batch Normalization (CytoNorm 2.0 Benchmark)",
        files_expected=12,
        size_mb_approx=8.4,
    ),
    "weber_2025_bk_polyomavirus": DatasetInfo(
        id="weber_2025_bk_polyomavirus",
        name="Weber BK Polyomavirus Infection Kinetics (2025)",
        category="conventional",
        aliases=["weber", "polyomavirus", "bk_virus", "bkv", "9_conv"],
        description="Primary human renal tubular cells & BKV with FlowJo Fig3.wsp and plate layout xlsx",
        organism="Human",
        specimen="Primary renal tubular cells (RPTECs)",
        cytometer="BD FACSCanto II",
        colors="Multichannel viral antigen & host cell viability",
        repository="Figshare",
        accession="10.6084/m9.figshare.29601881",
        publication="PLOS Pathogens (2025) 21(3):e1013122",
        doi="https://doi.org/10.1371/journal.ppat.1013122",
        recommended_for="FlowJo Workspace Interoperability (Fig3.wsp)",
        files_expected=18,
        size_mb_approx=8.4,
    ),
    "white_2021_flowkit_ics": DatasetInfo(
        id="white_2021_flowkit_ics",
        name="White FlowKit 8-Color ICS PBMC Benchmark (2021)",
        category="conventional",
        aliases=["flowkit", "white", "ics", "10_conv"],
        description="Human PBMC 8-color ICS benchmark with FlowJo 8_color_ICS.wsp and den_comp.csv",
        organism="Human",
        specimen="Human PBMCs",
        cytometer="BD LSR II",
        colors="8-color ICS panel (CD3, CD4, CD8, cytokines)",
        repository="GitHub / FlowKit",
        accession="8_color_data_set",
        publication="Frontiers in Immunology (2021) 12:768541",
        doi="https://doi.org/10.3389/fimmu.2021.768541",
        recommended_for="Boolean Cytokine Gating & Compensation Validation",
        files_expected=3,
        size_mb_approx=49.2,
    ),

    # =========================================================================
    # Spectral Flow Cytometry Benchmark Datasets (10 Datasets)
    # =========================================================================
    "arguello_2020_scenith_metabolism": DatasetInfo(
        id="arguello_2020_scenith_metabolism",
        name="Argüello SCENITH PBMC Metabolism (2020)",
        category="conventional",
        aliases=["scenith", "arguello", "metabolism", "1_conv"],
        description="Agilent NovoCyte Penteon 24-color human PBMC immunometabolism (SCENITH) panel with 24 single-stain compensation controls",
        organism="Human",
        specimen="Human PBMCs",
        cytometer="Agilent NovoCyte Penteon",
        colors="24 colors across 5 lasers",
        repository="Zenodo",
        accession="10.5281/zenodo.15350040",
        publication="Cell Metabolism (2020) 32:1063-1075 (Zenodo dataset Miranda Sole et al. 2025)",
        doi="https://doi.org/10.5281/zenodo.15350040",
        recommended_for="High-dimensional Conventional Compensation & Immunometabolism Discovery",
        files_expected=26,
        size_mb_approx=82.6,
    ),
    "baglio_2024_osteosarcoma_spectral": DatasetInfo(
        id="baglio_2024_osteosarcoma_spectral",
        name="Baglio Multi-Tissue Osteosarcoma OMIP-120 (2024)",
        category="spectral",
        aliases=["baglio", "osteosarcoma", "bone_marrow", "2_spec"],
        description="Cytek Aurora 5-laser 22-marker panel with paired raw and SpectroFlo unmixed FCS files",
        organism="Mouse",
        specimen="Osteosarcoma tumor, bone marrow, and spleen",
        cytometer="Cytek Aurora 5-laser (64 APD channels)",
        colors="22 markers",
        repository="Zenodo",
        accession="10.5281/zenodo.14513596",
        publication="Cytometry Part A (2026) 109:317-349 (OMIP-120)",
        doi="https://doi.org/10.1002/cyto.a.70034",
        recommended_for="Multi-Tissue AF & SpectroFlo Ground Truth Comparison",
        files_expected=3,
        size_mb_approx=2408.5,
    ),
    "balin_2025_thymopoiesis_omip116": DatasetInfo(
        id="balin_2025_thymopoiesis_omip116",
        name="Balin Human Thymopoiesis OMIP-116 (2025)",
        category="spectral",
        aliases=["balin", "thymopoiesis", "omip", "omip116", "omip-116", "park", "3_spec"],
        description="Cytek Aurora 5-laser 39-color human thymopoiesis panel with PeacoQC and FlowJo workspaces",
        organism="Human",
        specimen="Human thymocytes & PBMCs",
        cytometer="Cytek Aurora 5-laser (64 APD channels)",
        colors="39 colors",
        repository="Zenodo",
        accession="10.5281/zenodo.20133311",
        publication="Cytometry Part A (2025) 107:501-512",
        doi="https://doi.org/10.1002/cyto.a.24951",
        recommended_for="Automated Fluidics QC & Maturation Trajectory Gating",
        files_expected=6,
        size_mb_approx=601.0,
    ),
    "barrachina_2026_ipsc_organoid_ccl5": DatasetInfo(
        id="barrachina_2026_ipsc_organoid_ccl5",
        name="Barrachina iPSC Organoid Chemokine Perturbation (2026)",
        category="spectral",
        aliases=["organoid", "ccl5", "barrachina", "matsumoto", "ipsc", "4_spec"],
        description="Cytek Aurora 5-laser human iPSC organoids with raw/unmixed beads, unstained cells & CCL5",
        organism="Human",
        specimen="Human iPSC bone marrow organoids",
        cytometer="Cytek Aurora 5-laser (64 APD channels)",
        colors="13 colors",
        repository="Zenodo",
        accession="10.5281/zenodo.22783306",
        publication="Stem Cell Reports (2026) 21:103014",
        doi="https://doi.org/10.1016/j.stemcr.2026.103014",
        recommended_for="Spectral Quickstart Tutorial & Chemokine Perturbation",
        files_expected=14,
        size_mb_approx=181.5,
    ),
    "de_vries_2023_crc_til_mucosal": DatasetInfo(
        id="de_vries_2023_crc_til_mucosal",
        name="de Vries CRC Mucosal vs TIL Dual-AF (2023)",
        category="spectral",
        aliases=["crc", "de_vries", "til", "mucosal", "5_spec"],
        description="Cytek Aurora 5-laser 32-color human CRC mucosal vs TIL with dual-autofluorescence extraction",
        organism="Human",
        specimen="Primary colorectal cancer TILs vs mucosa",
        cytometer="Cytek Aurora 5-laser (64 APD channels)",
        colors="32 colors",
        repository="Zenodo",
        accession="10.5281/zenodo.21643327",
        publication="LUMC CRC Benchmark (2026)",
        doi="https://doi.org/10.5281/zenodo.21643327",
        recommended_for="Dual-Tissue Autofluorescence (AF) Extraction",
        files_expected=17,
        size_mb_approx=935.9,
    ),
    "flagr_2024_cytobatch_pbmc": DatasetInfo(
        id="flagr_2024_cytobatch_pbmc",
        name="Flagr CytoBatch Multi-Batch PBMC Benchmark (2024)",
        category="spectral",
        aliases=["cytobatch", "flagr", "6_spec"],
        description="Cytek Aurora 5-laser >40-marker longitudinal multi-batch human PBMC benchmark",
        organism="Human",
        specimen="Longitudinal human PBMCs across processing days",
        cytometer="Cytek Aurora 5-laser (64 APD channels)",
        colors=">40 markers",
        repository="Zenodo",
        accession="10.5281/zenodo.15388817",
        publication="CytoBatchFlagR Multi-Batch Study",
        doi="https://doi.org/10.5281/zenodo.15388817",
        recommended_for="Longitudinal Multi-Batch Drift Correction",
        files_expected=4,
        size_mb_approx=382.1,
    ),
    "korotkaja_2024_breast_tumor_tme": DatasetInfo(
        id="korotkaja_2024_breast_tumor_tme",
        name="Korotkaja 4T1 Breast Tumor Immunotherapy (2024)",
        category="spectral",
        aliases=["breast_tumor", "korotkaja", "liu", "tme", "breast", "7_spec"],
        description="Cytek Aurora 5-laser 18-color murine 4T1 breast tumor microenvironment cohort",
        organism="Mouse",
        specimen="4T1 breast cancer tumor homogenate",
        cytometer="Cytek Aurora 5-laser (64 APD channels)",
        colors="18 colors",
        repository="Zenodo",
        accession="10.5281/zenodo.17457137",
        publication="Molecular Therapy Oncology (2026) 34(1):201124",
        doi="https://doi.org/10.1016/j.omton.2025.201124",
        recommended_for="TME Leukocyte Gating & Immunotherapy Response",
        files_expected=14,
        size_mb_approx=612.3,
    ),
    "kreutmair_2026_icu_pbmc": DatasetInfo(
        id="kreutmair_2026_icu_pbmc",
        name="Kreutmair ICU Patient PBMC Clinical Cohort (2026)",
        category="spectral",
        aliases=["kreutmair", "icu", "icu_pbmc", "8_spec"],
        description="Cytek Aurora 5-laser human ICU patient PBMC immunophenotyping cohort with clinical metadata",
        organism="Human",
        specimen="Ventilated ICU patient PBMCs",
        cytometer="Cytek Aurora 5-laser (64 APD channels)",
        colors="39 colors across 5 lasers",
        repository="Zenodo",
        accession="10.5281/zenodo.22079206",
        publication="Cell Reports Medicine / Critical Care (2026)",
        doi="https://doi.org/10.1016/j.xcrm.2026.102919",
        recommended_for="Clinical Cohort Phenotyping & Survival Analysis",
        files_expected=8,
        size_mb_approx=333.3,
    ),
    "michaelis_2025_c9orf72_als_pbmc": DatasetInfo(
        id="michaelis_2025_c9orf72_als_pbmc",
        name="Michaelis ALS PBMC Cohort with 5 Workspaces (2025)",
        category="spectral",
        aliases=["michaelis", "als", "c9orf72", "als_pbmc", "9_spec"],
        description="Cytek Aurora human PBMC immunophenotyping with 5 FlowJo .wsp workspaces",
        organism="Human",
        specimen="Healthy vs ALS human PBMCs",
        cytometer="Cytek Aurora 5-laser (64 APD channels)",
        colors="High-parameter immune panel",
        repository="Zenodo",
        accession="10.5281/zenodo.16749283",
        publication="Nature (2025) 639:456-465",
        doi="https://doi.org/10.1038/s41586-025-09588-6",
        recommended_for="Multi-Workspace FlowJo Interoperability (5 .wsp files)",
        files_expected=8,
        size_mb_approx=397.5,
    ),
    "rockstein_2026_omip_whole_blood": DatasetInfo(
        id="rockstein_2026_omip_whole_blood",
        name="Rockstein 41-Color OMIP Whole Blood (2026)",
        category="spectral",
        aliases=["rockstein", "omip_blood", "whole_blood", "10_spec"],
        description="Cytek Aurora 5-laser 41-color OMIP whole blood panel with donor unstained baselines",
        organism="Human",
        specimen="Human whole blood",
        cytometer="Cytek Aurora 5-laser (64 APD channels)",
        colors="41 colors across 5 lasers",
        repository="Zenodo",
        accession="10.5281/zenodo.23950826",
        publication="Cytometry Part A (OMIP submission 2026)",
        doi="https://doi.org/10.5281/zenodo.23950826",
        recommended_for="Ultra-High Parameter Discovery (FlowSOM / UMAP)",
        files_expected=12,
        size_mb_approx=1341.0,
    ),
}


def list_datasets(category: Optional[str] = None) -> List[DatasetInfo]:
    """Return all datasets, optionally filtered by category ('conventional' or 'spectral')."""
    if not category or category.lower() in ("all", "*"):
        return list(BENCHMARK_DATASETS.values())
    cat = category.strip().lower()
    return [d for d in BENCHMARK_DATASETS.values() if d.category.lower() == cat]


def get_dataset(query: str) -> Optional[DatasetInfo]:
    """Lookup a dataset by exact ID, name, numeric index, or alias."""
    q = query.strip().lower()
    if not q:
        return None

    # Direct ID match
    if q in BENCHMARK_DATASETS:
        return BENCHMARK_DATASETS[q]

    # Check numeric prefix or alias
    for d in BENCHMARK_DATASETS.values():
        if d.matches_query(q):
            return d

    # Numeric selector: 1..10 in conventional or spectral
    if q.isdigit():
        num = int(q)
        conv = [d for d in BENCHMARK_DATASETS.values() if d.category == "conventional"]
        if 1 <= num <= len(conv):
            return conv[num - 1]

    return None


def resolve_datasets(queries: List[str], category: Optional[str] = None) -> List[DatasetInfo]:
    """Resolve a list of queries (e.g. ['all'], ['coburn', 'weber'], ['1,3']) into DatasetInfo instances."""
    selected_map: Dict[str, DatasetInfo] = {}

    for q in queries:
        parts = [p.strip() for p in q.split(",") if p.strip()]
        for part in parts:
            if part.lower() == "all":
                for d in list_datasets(category):
                    selected_map[d.id] = d
                continue

            matched = get_dataset(part)
            if matched:
                if category and category.lower() not in ("all", "*") and matched.category.lower() != category.lower():
                    continue
                selected_map[matched.id] = matched

    return list(selected_map.values())

