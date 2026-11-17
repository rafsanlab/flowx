#!/usr/bin/env python3
"""
Spectral Flow Cytometry Benchmark Datasets Downloader.

Downloads, organizes, and verifies 10 benchmark full-spectrum flow cytometry datasets (.fcs)
from open-access scientific repositories (Zenodo) in compliance with the MIFlowCyt standard.

Datasets Supported:
1. arguello_2020_scenith_metabolism:
   - Source: Zenodo (DOI: 10.5281/zenodo.15350040)
   - Publication: Nature Protocols (2020)
   - Biology: Cytek Aurora 5-laser 24-color human PBMC immunometabolism panel (SCENITH)
2. de_vries_2023_crc_til_mucosal:
   - Source: Zenodo (DOI: 10.5281/zenodo.21643327)
   - Publication: Primary Colorectal Cancer Benchmark
   - Biology: Cytek Aurora 5-laser 32-color human CRC mucosal vs TIL with multi-AF
3. rockstein_2026_omip_whole_blood:
   - Source: Zenodo (DOI: 10.5281/zenodo.23950826)
   - Publication: Cytometry Part A (OMIP Benchmark)
   - Biology: Cytek Aurora 5-laser 41-color human immune whole blood panel with donor unstained baselines
4. barrachina_2026_ipsc_organoid_ccl5:
   - Source: Zenodo (DOI: 10.5281/zenodo.22783306)
   - Publication: Stem Cell Reports 2026;21(8):103014
   - Biology: Cytek Aurora 5-laser human iPSC-derived bone marrow organoids with raw/unmixed beads & CCL5
5. michaelis_2025_c9orf72_als_pbmc:
   - Source: Zenodo (DOI: 10.5281/zenodo.16749283)
   - Publication: Science Translational Medicine / Zenodo
   - Biology: Cytek Aurora human PBMCs (Healthy controls vs ALS cohort) with 5 FlowJo .wsp workspaces
6. balin_2025_thymopoiesis_omip116:
   - Source: Zenodo (DOI: 10.5281/zenodo.20133311)
   - Publication: Cytometry Part A (OMIP-116, DOI: 10.1002/cyto.a.24951)
   - Biology: Cytek Aurora 5-laser 39-color human thymopoiesis panel with PeacoQC events & FlowJo .wsp
7. korotkaja_2024_breast_tumor_tme:
   - Source: Zenodo (DOI: 10.5281/zenodo.17457137)
   - Publication: Molecular Therapy Oncology (2026, DOI: 10.1016/j.omton.2025.201124)
   - Biology: Cytek Aurora 5-laser 18-color murine breast tumor microenvironment with reference controls
8. flagr_2024_cytobatch_pbmc:
   - Source: Zenodo (DOI: 10.5281/zenodo.15388817)
   - Publication: CytoBatchFlagR Multi-Batch Benchmark
   - Biology: Cytek Aurora 5-laser >40-marker longitudinal multi-batch human PBMC benchmark
9. baglio_2024_osteosarcoma_spectral:
   - Source: Zenodo (DOI: 10.5281/zenodo.14513596)
   - Publication: Frontiers in Immunology
   - Biology: Cytek Aurora 5-laser 22-marker panel (bone marrow, spleen, osteosarcoma) with tissue-matched unstained
10. kreutmair_2026_icu_pbmc:
   - Source: Zenodo (DOI: 10.5281/zenodo.22079206 / 10965313)
   - Publication: Cell Reports Medicine / Critical Care 2026
   - Biology: Cytek Aurora 5-laser 39-color human ICU patient PBMC immunophenotyping cohort
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_spectral")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FlowX/1.0"

SPECTRAL_DATASETS: Dict[str, Dict[str, Any]] = {
    "1": {
        "id": "arguello_2020_scenith_metabolism",
        "name": "arguello_2020_scenith_metabolism",
        "aliases": ["scenith", "arguello", "metabolism"],
        "description": "Cytek Aurora 5-laser 24-color human PBMC immunometabolism panel (Zenodo 15350040)",
    },
    "2": {
        "id": "de_vries_2023_crc_til_mucosal",
        "name": "de_vries_2023_crc_til_mucosal",
        "aliases": ["crc", "de_vries", "til", "mucosal"],
        "description": "Cytek Aurora 5-laser 32-color human CRC mucosal vs TIL with multi-AF (Zenodo 21643327)",
    },
    "3": {
        "id": "rockstein_2026_omip_whole_blood",
        "name": "rockstein_2026_omip_whole_blood",
        "aliases": ["rockstein", "omip_blood", "whole_blood"],
        "description": "Cytek Aurora 5-laser 41-color OMIP whole blood panel with donor unstained baselines (Zenodo 23950826)",
    },
    "4": {
        "id": "barrachina_2026_ipsc_organoid_ccl5",
        "name": "barrachina_2026_ipsc_organoid_ccl5",
        "aliases": ["organoid", "ccl5", "barrachina", "matsumoto", "ipsc"],
        "description": "Cytek Aurora 5-laser human iPSC organoids with raw/unmixed beads & CCL5 (Zenodo 22783306)",
    },
    "5": {
        "id": "michaelis_2025_c9orf72_als_pbmc",
        "name": "michaelis_2025_c9orf72_als_pbmc",
        "aliases": ["michaelis", "als", "c9orf72", "als_pbmc"],
        "description": "Cytek Aurora human PBMC immunophenotyping with 5 FlowJo .wsp workspaces (Zenodo 16749283)",
    },
    "6": {
        "id": "balin_2025_thymopoiesis_omip116",
        "name": "balin_2025_thymopoiesis_omip116",
        "aliases": ["balin", "thymopoiesis", "omip", "omip116", "omip-116", "omip098", "park"],
        "description": "Cytek Aurora 5-laser 39-color OMIP-116 human thymopoiesis panel with PeacoQC & .wsp (Zenodo 20133311)",
    },
    "7": {
        "id": "korotkaja_2024_breast_tumor_tme",
        "name": "korotkaja_2024_breast_tumor_tme",
        "aliases": ["breast_tumor", "korotkaja", "liu", "tme", "breast"],
        "description": "Cytek Aurora 5-laser 18-color murine 4T1 breast tumor microenvironment cohort (Zenodo 17457137)",
    },
    "8": {
        "id": "flagr_2024_cytobatch_pbmc",
        "name": "flagr_2024_cytobatch_pbmc",
        "aliases": ["cytobatch", "flagr"],
        "description": "Cytek Aurora 5-laser >40-marker longitudinal multi-batch human PBMC benchmark (Zenodo 15388817)",
    },
    "9": {
        "id": "baglio_2024_osteosarcoma_spectral",
        "name": "baglio_2024_osteosarcoma_spectral",
        "aliases": ["baglio", "osteosarcoma", "bone_marrow"],
        "description": "Cytek Aurora 5-laser 22-marker panel (bone marrow, spleen, osteosarcoma) with unstained (Zenodo 14513596)",
    },
    "10": {
        "id": "kreutmair_2026_icu_pbmc",
        "name": "kreutmair_2026_icu_pbmc",
        "aliases": ["kreutmair", "icu", "icu_pbmc"],
        "description": "Cytek Aurora 5-laser human ICU patient PBMC immunophenotyping cohort (Zenodo 22079206 / 10965313)",
    },
}


def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a local file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_fcs_header(filepath: Path) -> Dict[str, Any]:
    """Inspect FCS header parameters without external dependencies."""
    summary: Dict[str, Any] = {
        "filename": filepath.name,
        "size_bytes": filepath.stat().st_size,
        "md5": compute_file_hash(filepath),
        "fcs_version": "Unknown",
        "cytometer": "Unknown",
        "creator": "Unknown",
        "total_events": 0,
        "channel_count": 0,
        "channels": [],
    }

    try:
        with open(filepath, "rb") as f:
            header = f.read(58)
            fcs_ver = header[:6].decode("ascii", errors="ignore").strip()
            text_start = int(header[10:18].strip())
            text_end = int(header[18:26].strip())

            f.seek(text_start)
            text_bytes = f.read(text_end - text_start + 1)
            delim = chr(text_bytes[0])
            tokens = text_bytes[1:].decode("latin-1", errors="ignore").split(delim)

            kv: Dict[str, str] = {}
            for i in range(0, len(tokens) - 1, 2):
                kv[tokens[i].upper()] = tokens[i + 1]

            summary["fcs_version"] = fcs_ver
            summary["cytometer"] = kv.get("$CYT", kv.get("CYT", "Cytek Aurora"))
            summary["creator"] = kv.get("CREATOR", kv.get("$CREATOR", "SpectroFlo"))
            summary["total_events"] = int(kv.get("$TOT", "0"))
            par = int(kv.get("$PAR", "0"))
            summary["channel_count"] = par

            channels = []
            for p in range(1, par + 1):
                name = kv.get(f"$P{p}N", f"P{p}")
                desc = kv.get(f"$P{p}S", "")
                label = f"{name} ({desc})" if desc else name
                channels.append(label)
            summary["channels"] = channels
    except Exception as e:
        logger.warning("Failed parsing header for %s: %s", filepath.name, e)

    return summary


def download_file(
    url: str,
    destination: Path,
    expected_size: Optional[int] = None,
    max_retries: int = 4,
    verify_only: bool = False,
) -> bool:
    """Download an FCS file with retry logic and caching."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        local_size = destination.stat().st_size
        if expected_size is None or local_size == expected_size:
            logger.info("File already exists and verified: %s (%d bytes)", destination.name, local_size)
            return True

    if verify_only:
        logger.warning("File missing in verify-only mode: %s", destination.name)
        return False

    session = requests.Session()
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)...", destination.name, attempt, max_retries)
            resp = session.get(url, headers=headers, stream=True, timeout=60, verify=False)
            resp.raise_for_status()

            temp_dest = destination.with_suffix(".tmp")
            with open(temp_dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1048576):
                    if chunk:
                        f.write(chunk)

            temp_dest.replace(destination)
            logger.info("Successfully downloaded: %s (%d bytes)", destination.name, destination.stat().st_size)
            return True
        except Exception as e:
            logger.warning("Attempt %d failed for %s: %s", attempt, destination.name, e)
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                logger.error("Failed to download %s after %d attempts", destination.name, max_retries)
                return False

    return False


def download_supplementary_file(
    url: str,
    destination: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> bool:
    """Download non-FCS supplementary files (e.g. .wsp, PDF, guides) resiliently."""
    if not include_supplementary:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return True
    if verify_only:
        return False

    session = requests.Session()
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = session.get(url, headers=headers, stream=True, timeout=30, verify=False)
        resp.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1048576):
                if chunk:
                    f.write(chunk)
        logger.info("Downloaded supplementary: %s (%d bytes)", destination.name, destination.stat().st_size)
        return True
    except Exception as exc:
        logger.warning("Non-FCS supplementary download skipped for %s (%s)", destination.name, exc)
        return False


def build_dataset_metadata(
    dataset_dir: Path,
    file_roles: Dict[str, str],
    file_categories: Dict[str, str],
    file_urls: Dict[str, str],
    study_metadata: Dict[str, Any],
    verify_only: bool = False,
) -> None:
    """Compile dataset_manifest.json, panel.csv, and study_metadata.json."""
    meta_dir = dataset_dir / "metadata"
    if (
        verify_only
        and (meta_dir / "dataset_manifest.json").exists()
        and (meta_dir / "panel.csv").exists()
        and (meta_dir / "study_metadata.json").exists()
    ):
        logger.debug("Verify-only mode: preserving existing metadata in %s", meta_dir)
        return

    raw_dir = dataset_dir / "raw"
    sample_fcs = sorted((raw_dir / "samples").glob("*.fcs")) if (raw_dir / "samples").is_dir() else []
    control_fcs = sorted((raw_dir / "controls").glob("*.fcs")) if (raw_dir / "controls").is_dir() else []
    fcs_files = sample_fcs + control_fcs

    manifest: List[Dict[str, Any]] = []
    panel_rows: List[List[str]] = [["channel", "marker", "fluorophore"]]

    for fcs_path in fcs_files:
        meta = inspect_fcs_header(fcs_path)
        rel_path = str(fcs_path.relative_to(dataset_dir))
        fname = fcs_path.name

        meta["relative_path"] = rel_path
        meta["role"] = file_roles.get(fname, "sample" if "samples" in rel_path else "control")
        meta["category"] = file_categories.get(fname, "spectral_fcs")
        meta["source_url"] = file_urls.get(fname, "")
        manifest.append(meta)

        if not panel_rows[1:] and meta.get("channels"):
            for ch in meta["channels"]:
                parts = ch.split(" (", 1)
                name = parts[0]
                desc = parts[1].rstrip(")") if len(parts) > 1 else ""
                panel_rows.append([name, desc, desc])

    meta_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "publications").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "raw" / "controls").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "raw" / "samples").mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(meta_dir / "panel.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(panel_rows)

    with open(meta_dir / "study_metadata.json", "w", encoding="utf-8") as f:
        json.dump(study_metadata, f, indent=2)

    logger.info("Compiled metadata for %s: %d FCS files", dataset_dir.name, len(manifest))


# =========================================================================
# 1. arguello_2020_scenith_metabolism (Zenodo 15350040)
# =========================================================================
def download_scenith_dataset(
    base_data: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> None:
    """1. arguello_2020_scenith_metabolism (Zenodo 15350040)."""
    logger.info("=== Checking / Downloading: arguello_2020_scenith_metabolism ===")
    d1_dir = base_data / "conventional" / "arguello_2020_scenith_metabolism"
    d1_ctrl_dir = d1_dir / "raw" / "controls"
    d1_samp_dir = d1_dir / "raw" / "samples"
    d1_urls: Dict[str, str] = {}
    d1_roles: Dict[str, str] = {}
    d1_cats: Dict[str, str] = {}

    d1_ctrl_dir.mkdir(parents=True, exist_ok=True)
    d1_samp_dir.mkdir(parents=True, exist_ok=True)

    d1_controls = [
        "Scenith_05062024_Compensation Specimen_A1_B525 AF488 - Puromycin_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_A2_B615 NFB 610_70S - CD20_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_A3_B695 BB700 - FoxP3_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_A4_B725 PerCP_eFluor 710 - CD19_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_A5_R667 JF 646 - CD127_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_A6_R725 AF700 - CCR6_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_A7_R780 APC_Cy7 - CD45RA_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_A8_UV445 BUV395 - Viability_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_B1_UV525 BUV496 - CD27_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_B2_UV586 BUV563 - CXCR5_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_B3_UV615 BUV615 - CD56_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_B4_UV667 BUV661 - CRTh2_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_B5_UV725 BUV737 - IgD_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_B6_V445 BV421 - CD25_20240524.fcs",
        "Scenith_05062024_Compensation Specimen_B7_V586 BV570 - CD3_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_B8_V615 BV605 - CD16_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_C1_V667 BV650 - CD28_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_C2_V725 BV711 - CD14_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_C3_V780 BV786 - CCR7_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_C4_Y586 PE - CD24_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_C5_Y615 PE_Dazzle 594 - CXCR3_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_C6_Y667 PE_Cy5 - CD38_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_C7_Y725 PE_AF700 - CD4_20240521.fcs",
        "Scenith_05062024_Compensation Specimen_C8_Y780 PE_Cy7 - TCRgd_20240521.fcs",
    ]

    for fn in d1_controls:
        u = f"https://zenodo.org/api/records/15350040/files/{urllib.parse.quote(fn)}/content"
        dest = d1_ctrl_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d1_urls[fn] = u
            d1_roles[fn] = "control"
            d1_cats[fn] = "single_stain"

    d1_samples = [
        "Scenith_21062024_Samples_D5_All_20240621.fcs",
        "Scenith_05062024_PMA_Ionomycin_E1_Control_20240605.fcs",
    ]
    for fn in d1_samples:
        u_samp = f"https://zenodo.org/api/records/15350040/files/{urllib.parse.quote(fn)}/content"
        dest_samp = d1_samp_dir / fn
        if download_file(u_samp, dest_samp, verify_only=verify_only):
            d1_urls[fn] = u_samp
            d1_roles[fn] = "sample"
            d1_cats[fn] = "pbmc_sample"

    build_dataset_metadata(
        d1_dir, d1_roles, d1_cats, d1_urls,
        {
            "title": "A 24 colour-panel to functionally characterise energy metabolism from peripheral blood mononuclear cells by advanced flow cytometry",
            "doi": "10.5281/zenodo.15350040",
            "citation": "Miranda Sole, C., Dominguez-Villar, M., & Rowley, J. (2025). Zenodo (DOI: 10.5281/zenodo.15350040); SCENITH assay protocol per Arguello et al. (2020, Cell Metabolism)",
            "publication_date": "2025-05-06",
            "cytometer": "Agilent NovoCyte Penteon",
            "biology": "Human PBMCs immunometabolism profiling (SCENITH Puromycin incorporation)",
            "parameters": "30 total FCS parameters (4 scatter, 24 fluorescence markers/viability, Time/width channels)",
            "af_strategy": "Not applicable (conventional 24-color spillover compensation on NovoCyte Penteon)",
        },
        verify_only=verify_only,
    )


# =========================================================================
# 2. de_vries_2023_crc_til_mucosal (Zenodo 21643327)
# =========================================================================
def download_crc_til_dataset(
    base_data: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> None:
    """2. de_vries_2023_crc_til_mucosal (Zenodo 21643327)."""
    logger.info("=== Checking / Downloading: de_vries_2023_crc_til_mucosal ===")
    d2_dir = base_data / "spectral" / "de_vries_2023_crc_til_mucosal"
    d2_ctrl_dir = d2_dir / "raw" / "controls"
    d2_samp_dir = d2_dir / "raw" / "samples"
    d2_urls: Dict[str, str] = {}
    d2_roles: Dict[str, str] = {}
    d2_cats: Dict[str, str] = {}

    d2_ctrl_dir.mkdir(parents=True, exist_ok=True)
    d2_samp_dir.mkdir(parents=True, exist_ok=True)

    d2_controls = [
        "MB066_TIL_Unstained.fcs", "MB066_ML_Unstained.fcs", "Unstained (Cells).fcs", "Negative_1 (Beads).fcs",
        "CD3 V500 (Cells).fcs", "CD4 FITC (Cells).fcs", "CD8 APC-Cy7 (Cells).fcs", "CD19 BV570 (Cells).fcs",
        "CD21 BUV661 (Cells).fcs", "CD25 BV650 (Cells).fcs", "CD27 RY775 (Cells).fcs", "CD38 BV786 (Cells).fcs",
        "CD45 BUV496 (Cells).fcs", "CD56 BV711 (Cells).fcs", "Zombie NIR Zombie NIR (Cells).fcs",
    ]

    for fn in d2_controls:
        u = f"https://zenodo.org/api/records/21643327/files/{urllib.parse.quote(fn)}/content"
        dest = d2_ctrl_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d2_urls[fn] = u
            d2_roles[fn] = "control"
            d2_cats[fn] = "autofluorescence_control" if "Unstained" in fn else "single_stain"

    d2_samples = ["MB066_TIL_Mix.fcs", "MB066_ML_Mix.fcs"]
    for fn in d2_samples:
        u = f"https://zenodo.org/api/records/21643327/files/{urllib.parse.quote(fn)}/content"
        dest = d2_samp_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d2_urls[fn] = u
            d2_roles[fn] = "sample"
            d2_cats[fn] = "tumor_sample" if "TIL" in fn else "mucosal_sample"

    build_dataset_metadata(
        d2_dir, d2_roles, d2_cats, d2_urls,
        {
            "title": "I-COMET panel v5 (32-color Cytek Aurora-Evo spectral immunophenotyping cohort; benchmark directory retains de_vries_2023_crc_til_mucosal identifier)",
            "doi": "10.5281/zenodo.21643327",
            "citation": "Le Bourhis, L. (2026). I-COMET panel v5 spectral flow cytometry FCS cohort (Zenodo DOI: 10.5281/zenodo.21643327; acquired 2026-04-13 on Cytek Aurora-Evo with SpectroFlo 4.0.2)",
            "publication_date": "2026-04-13",
            "cytometer": "Cytek Aurora-Evo 5-laser (UV-V-B-YG-R, SpectroFlo 4.0.2)",
            "biology": "Human immune cell phenotyping (32-color I-COMET v5 unmixed spectral panel)",
            "parameters": "40 total FCS parameters (32 fluorescence markers/viability + 6 scatter + Time + autofluorescence)",
            "af_strategy": "Pre-unmixed with Cytek_AFExtract and embedded 33x33 spillover matrix",
        },
        verify_only=verify_only,
    )


# =========================================================================
# 3. rockstein_2026_omip_whole_blood (Zenodo 23950826)
# =========================================================================
def download_rockstein_dataset(
    base_data: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> None:
    """3. rockstein_2026_omip_whole_blood (Zenodo 23950826)."""
    logger.info("=== Checking / Downloading: rockstein_2026_omip_whole_blood ===")
    d3_dir = base_data / "spectral" / "rockstein_2026_omip_whole_blood"
    d3_ctrl_dir = d3_dir / "raw" / "controls"
    d3_samp_dir = d3_dir / "raw" / "samples"
    d3_urls: Dict[str, str] = {}
    d3_roles: Dict[str, str] = {}
    d3_cats: Dict[str, str] = {}

    d3_ctrl_dir.mkdir(parents=True, exist_ok=True)
    d3_samp_dir.mkdir(parents=True, exist_ok=True)

    unstained_controls = [
        "ISP V10-Samples-HD1 unstained_Unmixed.fcs",
        "ISP V10-Samples-HD2 unstained V10_Unmixed.fcs",
        "ISP V10-Samples-HD6 unstained V10_Unmixed.fcs",
        "ISP V10-Samples-HD7 unstained V10_Unmixed.fcs",
        "ISP V10-Samples-HD8 unstained V10_Unmixed.fcs",
        "ISP V10-Samples-HD9 unstained V10_Unmixed.fcs",
    ]

    for fn in unstained_controls:
        u = f"https://zenodo.org/api/records/23950826/files/{urllib.parse.quote(fn)}/content"
        dest = d3_ctrl_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d3_urls[fn] = u
            d3_roles[fn] = "control"
            d3_cats[fn] = "donor_unstained_baseline"

    samples = [
        "ISP V10-Samples-HD1 V10_Unmixed - kopie.fcs",
        "ISP V10-Samples-HD6 V10_Unmixed.fcs",
        "ISP V10-Samples-HD7 V10_Unmixed.fcs",
        "ISP V10-Samples-HD8 V10_Unmixed.fcs",
        "ISP V10-PS04 V10_Unmixed.fcs",
        "ISP V10-PS07 V10_Unmixed.fcs",
    ]

    for fn in samples:
        u = f"https://zenodo.org/api/records/23950826/files/{urllib.parse.quote(fn)}/content"
        dest = d3_samp_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d3_urls[fn] = u
            d3_roles[fn] = "sample"
            d3_cats[fn] = "whole_blood_sample"

    build_dataset_metadata(
        d3_dir, d3_roles, d3_cats, d3_urls,
        {
            "title": "OMIP: A 41-colour spectral flow cytometry panel for human immune cell composition in whole blood",
            "doi": "10.5281/zenodo.23950826",
            "citation": "Rockstein et al., Cytometry Part A (OMIP Collection)",
            "cytometer": "Cytek Aurora 5-laser (UV-V-B-YG-R, 64 APD channels)",
            "biology": "Human whole blood immunophenotyping across healthy donors and patient samples",
            "parameters": "41 colors spanning 5 lasers + donor-matched cellular autofluorescence baselines",
            "af_strategy": "Donor-specific baseline autofluorescence subtraction",
        },
        verify_only=verify_only,
    )


# =========================================================================
# 4. barrachina_2026_ipsc_organoid_ccl5 (Zenodo 22783306)
# =========================================================================
def download_ipsc_organoid_dataset(
    base_data: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> None:
    """4. barrachina_2026_ipsc_organoid_ccl5 (Zenodo 22783306)."""
    logger.info("=== Checking / Downloading: barrachina_2026_ipsc_organoid_ccl5 ===")
    d4_dir = base_data / "spectral" / "barrachina_2026_ipsc_organoid_ccl5"
    if not d4_dir.exists() and (base_data / "spectral" / "matsumoto_2026_ipsc_organoid_ccl5").exists():
        d4_dir = base_data / "spectral" / "matsumoto_2026_ipsc_organoid_ccl5"

    d4_ctrl_dir = d4_dir / "raw" / "controls"
    d4_samp_dir = d4_dir / "raw" / "samples"
    d4_supp_dir = d4_dir / "supplementary"
    d4_urls: Dict[str, str] = {}
    d4_roles: Dict[str, str] = {}
    d4_cats: Dict[str, str] = {}

    d4_ctrl_dir.mkdir(parents=True, exist_ok=True)
    d4_samp_dir.mkdir(parents=True, exist_ok=True)
    if include_supplementary:
        d4_supp_dir.mkdir(parents=True, exist_ok=True)
        download_supplementary_file("https://zenodo.org/api/records/22783306/files/manifest.csv/content", d4_supp_dir / "manifest.csv", verify_only=verify_only, include_supplementary=include_supplementary)
        download_supplementary_file("https://zenodo.org/api/records/22783306/files/240228_organoids_staining.xlsx/content", d4_supp_dir / "240228_organoids_staining.xlsx", verify_only=verify_only, include_supplementary=include_supplementary)
        download_supplementary_file("https://zenodo.org/api/records/22783306/files/240228_manuscrip-2.prism/content", d4_supp_dir / "240228_manuscrip-2.prism", verify_only=verify_only, include_supplementary=include_supplementary)

    controls = [
        "raw_reference_control_APC-Beads-2.fcs", "raw_reference_control_Pacific-Blue-Beads-8.fcs",
        "raw_reference_control_FITC-Beads-7.fcs", "raw_reference_control_PE-Beads-9.fcs",
        "raw_reference_control_Unstained-Cells-15.fcs", "unmixed_reference_control_Unstained-Cells-14.fcs",
        "unmixed_fmo_CD34-FMO-3.fcs", "unmixed_fmo_Zombie-FMO-4.fcs",
    ]

    for fn in controls:
        u = f"https://zenodo.org/api/records/22783306/files/{urllib.parse.quote(fn)}/content"
        dest = d4_ctrl_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d4_urls[fn] = u
            d4_roles[fn] = "control"
            d4_cats[fn] = "unstained_cells" if "Unstained" in fn else ("fmo_control" if "fmo" in fn else "reference_bead")

    samples = [
        "unmixed_sample_V1-24h.fcs", "unmixed_sample_V2-24h.fcs", "unmixed_sample_V3-24h.fcs",
        "unmixed_sample_CCL5-1-24h.fcs", "unmixed_sample_CCL5-2-24h.fcs", "unmixed_sample_CCL5-3-24h.fcs",
    ]

    for fn in samples:
        u = f"https://zenodo.org/api/records/22783306/files/{urllib.parse.quote(fn)}/content"
        dest = d4_samp_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d4_urls[fn] = u
            d4_roles[fn] = "sample"
            d4_cats[fn] = "vehicle_control" if "V" in fn else "ccl5_treated"

    build_dataset_metadata(
        d4_dir, d4_roles, d4_cats, d4_urls,
        {
            "title": "Human iPSC-derived bone marrow organoids under CCL5 exposure",
            "doi": "10.5281/zenodo.22783306",
            "citation": "Barrachina et al., Stem Cell Reports 2026;21(8):103014",
            "cytometer": "Cytek Aurora 5-laser (UV-V-B-YG-R, 64 APD channels)",
            "biology": "Human iPSC bone marrow organoids (Vehicle vs CCL5 chemokine treated)",
            "parameters": "13-color hematopoietic & stromal panel with reference beads, unstained cells & FMOs",
            "af_strategy": "Organoid baseline cellular autofluorescence extraction",
        },
        verify_only=verify_only,
    )


# =========================================================================
# 5. michaelis_2025_c9orf72_als_pbmc (Zenodo 16749283)
# =========================================================================
def download_michaelis_dataset(
    base_data: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> None:
    """5. michaelis_2025_c9orf72_als_pbmc (Zenodo 16749283)."""
    logger.info("=== Checking / Downloading: michaelis_2025_c9orf72_als_pbmc ===")
    d5_dir = base_data / "spectral" / "michaelis_2025_c9orf72_als_pbmc"
    d5_samp_dir = d5_dir / "raw" / "samples"
    d5_meta_dir = d5_dir / "metadata"
    d5_urls: Dict[str, str] = {}
    d5_roles: Dict[str, str] = {}
    d5_cats: Dict[str, str] = {}

    d5_samp_dir.mkdir(parents=True, exist_ok=True)
    d5_meta_dir.mkdir(parents=True, exist_ok=True)

    if include_supplementary:
        wsps = ["#1 Spectral.wsp", "#2 Spectral.wsp", "#3 Spectral.wsp", "#4 Spectral.wsp", "#5 Spectral.wsp"]
        for wsp_fn in wsps:
            u = f"https://zenodo.org/api/records/16749283/files/{urllib.parse.quote(wsp_fn)}/content"
            download_supplementary_file(
                u,
                d5_meta_dir / wsp_fn,
                verify_only=verify_only,
                include_supplementary=include_supplementary,
            )

    samples = [
        "A1 6936 ALS.fcs", "A3 8830 ALS.fcs", "A4 8710 ALS.fcs",
        "A5 6807 HC.fcs", "A6 4191 HC.fcs", "A8 3592 HC.fcs", "A8 4660 HC.fcs", "A10 7101 HC.fcs",
    ]

    for fn in samples:
        u = f"https://zenodo.org/api/records/16749283/files/{urllib.parse.quote(fn)}/content"
        dest = d5_samp_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d5_urls[fn] = u
            d5_roles[fn] = "sample"
            d5_cats[fn] = "als_pbmc" if "ALS" in fn else "healthy_control_pbmc"

    build_dataset_metadata(
        d5_dir, d5_roles, d5_cats, d5_urls,
        {
            "title": "Autoimmune response to C9orf72 protein in amyotrophic lateral sclerosis - Spectral data",
            "doi": "10.5281/zenodo.16749283",
            "citation": "Michaelis et al., Nature 2025 (DOI: 10.1038/s41586-025-09588-6)",
            "cytometer": "Cytek Aurora Full Spectrum Flow Cytometer",
            "biology": "Human PBMCs from ALS patients vs Healthy Controls",
            "parameters": "High-parameter spectral immunophenotyping with 5 FlowJo .wsp workspace files",
            "af_strategy": "SpectroFlo unmixing with manual FlowJo gating validation",
        },
        verify_only=verify_only,
    )


# =========================================================================
# 6. balin_2025_thymopoiesis_omip116 (Zenodo 20133311)
# =========================================================================
def download_thymopoiesis_dataset(
    base_data: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> None:
    """6. balin_2025_thymopoiesis_omip116 (Zenodo 20133311)."""
    logger.info("=== Checking / Downloading: balin_2025_thymopoiesis_omip116 ===")
    d6_dir = base_data / "spectral" / "balin_2025_thymopoiesis_omip116"
    d6_ctrl_dir = d6_dir / "raw" / "controls"
    d6_samp_dir = d6_dir / "raw" / "samples"
    d6_meta_dir = d6_dir / "metadata"
    d6_urls: Dict[str, str] = {}
    d6_roles: Dict[str, str] = {}
    d6_cats: Dict[str, str] = {}

    d6_ctrl_dir.mkdir(parents=True, exist_ok=True)
    d6_samp_dir.mkdir(parents=True, exist_ok=True)
    d6_meta_dir.mkdir(parents=True, exist_ok=True)

    if include_supplementary:
        for wsp_fn in ["Healthy Thymus.wsp", "Panel T_39 colori.wsp"]:
            download_supplementary_file(
                f"https://zenodo.org/api/records/20133311/files/{urllib.parse.quote(wsp_fn)}/content",
                d6_meta_dir / wsp_fn,
                verify_only=verify_only,
                include_supplementary=include_supplementary,
            )

        download_supplementary_file(
            "https://zenodo.org/api/records/20133311/files/Metadata.xlsx/content",
            d6_meta_dir / "Metadata.xlsx",
            verify_only=verify_only,
            include_supplementary=include_supplementary,
        )

    controls = [
        "Sample_AF647_unmixing39.fcs", "Sample_BV421_unmixing39.fcs", "Sample_BUV395_unmixing39.fcs",
        "Sample_PE_PBMC_unmixing39.fcs", "PBMCs_PBMC448_FMO CD38_PeacoQCGoodEvents.fcs",
    ]
    for fn in controls:
        u = f"https://zenodo.org/api/records/20133311/files/{urllib.parse.quote(fn)}/content"
        dest = d6_ctrl_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d6_urls[fn] = u
            d6_roles[fn] = "control"
            d6_cats[fn] = "fmo_control" if "FMO" in fn else "single_stain"

    samples = [
        "Thymus_HC22_PeacoQCGoodEvents.fcs",
        "PBMC_448_PeacoQCGoodEvents.fcs",
        "UnmixingErr_Adj_Thymus_HC22_PeacoQCGoodEvents.fcs",
        "UnmixingErr_Adj_PBMC_448_PeacoQCGoodEvents.fcs",
    ]

    for fn in samples:
        u = f"https://zenodo.org/api/records/20133311/files/{urllib.parse.quote(fn)}/content"
        dest = d6_samp_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d6_urls[fn] = u
            d6_roles[fn] = "sample"
            d6_cats[fn] = "thymocyte_sample" if "Thymus" in fn else "pbmc_sample"

    build_dataset_metadata(
        d6_dir, d6_roles, d6_cats, d6_urls,
        {
            "title": "OMIP-116: A 39-Color Full Spectrum Flow Cytometric Panel to Deeply Characterize Human Thymopoiesis",
            "doi": "10.5281/zenodo.20133311",
            "citation": "Balin et al., Cytometry Part A 2025;107:501-512 (OMIP-116; DOI: 10.1002/cyto.a.24951)",
            "cytometer": "BD FACSymphony A5 SE (FACSymphony Analyzer)",
            "biology": "Human thymocytes and peripheral blood mononuclear cells",
            "parameters": "39 colors spanning 5 lasers + PeacoQC automated cleaning + FlowJo workspace",
            "af_strategy": "SpectroFlo unmixing with PeacoQC fluidics anomaly removal",
        },
        verify_only=verify_only,
    )


# =========================================================================
# 7. korotkaja_2024_breast_tumor_tme (Zenodo 17457137)
# =========================================================================
def download_breast_tumor_dataset(
    base_data: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> None:
    """7. korotkaja_2024_breast_tumor_tme (Zenodo 17457137)."""
    logger.info("=== Checking / Downloading: korotkaja_2024_breast_tumor_tme ===")
    d7_dir = base_data / "spectral" / "korotkaja_2024_breast_tumor_tme"
    if not d7_dir.exists() and (base_data / "spectral" / "liu_2024_breast_tumor_tme").exists():
        d7_dir = base_data / "spectral" / "liu_2024_breast_tumor_tme"

    d7_ctrl_dir = d7_dir / "raw" / "controls"
    d7_samp_dir = d7_dir / "raw" / "samples"
    d7_urls: Dict[str, str] = {}
    d7_roles: Dict[str, str] = {}
    d7_cats: Dict[str, str] = {}

    d7_ctrl_dir.mkdir(parents=True, exist_ok=True)
    d7_samp_dir.mkdir(parents=True, exist_ok=True)

    controls = [
        "Reference Group-A2 CD11b BV510 (Beads).fcs", "Reference Group-A3 MHC II APC (Beads).fcs",
        "Reference Group-A4 F4-80 PE-eFluor 610 (Beads).fcs", "Reference Group-A5 CD206 PE (Beads).fcs",
        "Reference Group-A8 CD4 Real Blue RB 545 (Beads).fcs", "Reference Group-A9 CD3 BV605 (Beads).fcs",
        "Reference Group-B1 Viability Live-or-dye_TM (Cells).fcs", "Reference Group-B4 CD8 FITC (Beads).fcs",
    ]

    for fn in controls:
        u = f"https://zenodo.org/api/records/17457137/files/{urllib.parse.quote(fn)}/content"
        dest = d7_ctrl_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d7_urls[fn] = u
            d7_roles[fn] = "control"
            d7_cats[fn] = "viability_control" if "Viability" in fn else "reference_bead"

    samples = [
        "PBS_001.fcs", "PBS_002.fcs", "PBS_003.fcs",
        "SFV-IFNg_020.fcs", "PDT_033.fcs", "SFV-Luc_029.fcs",
    ]

    for fn in samples:
        u = f"https://zenodo.org/api/records/17457137/files/{urllib.parse.quote(fn)}/content"
        dest = d7_samp_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d7_urls[fn] = u
            d7_roles[fn] = "sample"
            d7_cats[fn] = "vehicle_control" if "PBS" in fn else "immunotherapy_treated"

    build_dataset_metadata(
        d7_dir, d7_roles, d7_cats, d7_urls,
        {
            "title": "Virus-based IFNγ gene delivery and photodynamic therapy cooperate to remodel the tumor microenvironment and suppress breast cancer",
            "doi": "10.1016/j.omton.2025.201124",
            "zenodo_doi": "10.5281/zenodo.17457137",
            "citation": "Korotkaja, K. et al., Molecular Therapy Oncology (2026) 34(1):201124 (Zenodo DOI: 10.5281/zenodo.17457137)",
            "cytometer": "Cytek Aurora 5-laser (UV-V-B-YG-R, 64 APD channels)",
            "biology": "Murine 4T1 breast cancer tumor-infiltrating immune cells",
            "parameters": "18 colors across 5 lasers with reference beads & viability controls",
            "af_strategy": "SpectroFlo unmixing with tissue autofluorescence extraction",
        },
        verify_only=verify_only,
    )


# =========================================================================
# 8. flagr_2024_cytobatch_pbmc (Zenodo 15388817)
# =========================================================================
def download_cytobatch_dataset(
    base_data: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> None:
    """8. flagr_2024_cytobatch_pbmc (Zenodo 15388817)."""
    logger.info("=== Checking / Downloading: flagr_2024_cytobatch_pbmc ===")
    d8_dir = base_data / "spectral" / "flagr_2024_cytobatch_pbmc"
    d8_samp_dir = d8_dir / "raw" / "samples"
    d8_urls: Dict[str, str] = {}
    d8_roles: Dict[str, str] = {}
    d8_cats: Dict[str, str] = {}

    d8_samp_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        "B02_tcu2_CD45+CD14-.fcs", "B03_tcu1_CD45+CD14-.fcs", "B03_tcs3_CD45+CD14-.fcs",
        "B04_tcs3_CD45+CD14-.fcs", "B11_tcu1_CD45+CD14-.fcs", "B13_tcs4_CD45+CD14-.fcs",
    ]

    for fn in samples:
        u = f"https://zenodo.org/api/records/15388817/files/{urllib.parse.quote(fn)}/content"
        dest = d8_samp_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d8_urls[fn] = u
            d8_roles[fn] = "sample"
            d8_cats[fn] = "unstimulated_batch" if "_tcu" in fn else "stimulated_batch"

    build_dataset_metadata(
        d8_dir, d8_roles, d8_cats, d8_urls,
        {
            "title": "CytoBatchFlagR: Spectral flow cytometry multi-batch benchmark",
            "doi": "10.5281/zenodo.15388817",
            "citation": "FlagR Multi-Batch Longitudinal Benchmark",
            "cytometer": "Cytek Aurora 5-laser (UV-V-B-YG-R, 64 APD channels)",
            "biology": "Human PBMCs measured across independent longitudinal acquisition batches",
            "parameters": ">40 markers across 5 lasers with batch stimulus variations",
            "af_strategy": "SpectroFlo unmixing with batch effect monitoring",
        },
        verify_only=verify_only,
    )


# =========================================================================
# 9. baglio_2024_osteosarcoma_spectral (Zenodo 14513596)
# =========================================================================
def download_baglio_dataset(
    base_data: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> None:
    """9. baglio_2024_osteosarcoma_spectral (Zenodo 14513596)."""
    logger.info("=== Checking / Downloading: baglio_2024_osteosarcoma_spectral ===")
    d9_dir = base_data / "spectral" / "baglio_2024_osteosarcoma_spectral"
    d9_ctrl_dir = d9_dir / "raw" / "controls"
    d9_samp_dir = d9_dir / "raw" / "samples"
    d9_urls: Dict[str, str] = {}
    d9_roles: Dict[str, str] = {}
    d9_cats: Dict[str, str] = {}

    d9_ctrl_dir.mkdir(parents=True, exist_ok=True)
    d9_samp_dir.mkdir(parents=True, exist_ok=True)

    controls = [
        "Osteosarcoma unstained.fcs", "Bone marrow unstained.fcs",
        "Spleen protocol A unstained_Unmixed.fcs", "Spleen protocol B unstained.fcs",
        "CD69 PE-Cy5 (Cells).fcs",
    ]

    for fn in controls:
        u = f"https://zenodo.org/api/records/14513596/files/{urllib.parse.quote(fn)}/content"
        dest = d9_ctrl_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d9_urls[fn] = u
            d9_roles[fn] = "control"
            d9_cats[fn] = "tissue_unstained_baseline" if "unstained" in fn.lower() else "single_stain"

    samples = [
        "Osteosarcoma.fcs", "Osteosarcoma_Unmixed.fcs",
        "Bone marrow.fcs", "Bone marrow_Unmixed.fcs",
        "Spleen protocol A.fcs", "Spleen protocol B.fcs",
    ]

    for fn in samples:
        u = f"https://zenodo.org/api/records/14513596/files/{urllib.parse.quote(fn)}/content"
        dest = d9_samp_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d9_urls[fn] = u
            d9_roles[fn] = "sample"
            d9_cats[fn] = "osteosarcoma_tumor" if "Osteosarcoma" in fn else ("bone_marrow" if "marrow" in fn else "spleen_tissue")

    build_dataset_metadata(
        d9_dir, d9_roles, d9_cats, d9_urls,
        {
            "title": "OMIP-120: A 22-Marker Spectral Flow Cytometry Panel for the Characterization of Major Immune Populations in Murine Bone Marrow and Osteosarcoma Tissue",
            "doi": "10.5281/zenodo.14513596",
            "citation": "Foderaro et al., Cytometry Part A 2026;109:317-349 (OMIP-120; DOI: 10.1002/cyto.a.70034)",
            "cytometer": "Cytek Aurora 5-laser (UV-V-B-YG-R, 64 APD channels)",
            "biology": "Murine primary osteosarcoma tumor, bone marrow, and spleen leukocytes",
            "parameters": "22 markers across 5 lasers with tissue-specific unstained baselines and raw/unmixed pairs",
            "af_strategy": "Multi-tissue autofluorescence extraction across tumor, spleen, and marrow",
        },
        verify_only=verify_only,
    )


# =========================================================================
# 10. kreutmair_2026_icu_pbmc (Zenodo 22079206 / 10965313)
# =========================================================================
def download_icu_pbmc_dataset(
    base_data: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> None:
    """10. kreutmair_2026_icu_pbmc (Zenodo 22079206 / 10965313)."""
    logger.info("=== Checking / Downloading: kreutmair_2026_icu_pbmc ===")
    d10_dir = base_data / "spectral" / "kreutmair_2026_icu_pbmc"
    d10_samp_dir = d10_dir / "raw" / "samples"
    d10_urls: Dict[str, str] = {}
    d10_roles: Dict[str, str] = {}
    d10_cats: Dict[str, str] = {}

    d10_samp_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        "186.fcs", "202.fcs", "235.fcs", "250.fcs",
        "253.fcs", "278.fcs", "311.fcs", "315.fcs",
    ]

    for fn in samples:
        u = f"https://zenodo.org/api/records/22079206/files/{urllib.parse.quote(fn)}/content"
        dest = d10_samp_dir / fn
        if download_file(u, dest, verify_only=verify_only):
            d10_urls[fn] = u
            d10_roles[fn] = "sample"
            d10_cats[fn] = "icu_patient_pbmc"

    build_dataset_metadata(
        d10_dir, d10_roles, d10_cats, d10_urls,
        {
            "title": "Clinical human ICU patient PBMC immunophenotyping cohort",
            "doi": "10.5281/zenodo.22079206",
            "citation": "Kreutmair et al., Cell Reports Medicine / Critical Care 2026",
            "cytometer": "Cytek Aurora 5-laser (UV-V-B-YG-R, 64 APD channels)",
            "biology": "Human PBMCs from critically ill, mechanically ventilated ICU patients",
            "parameters": "39-color deep immunophenotyping panel across 5 lasers",
            "af_strategy": "SpectroFlo spectral unmixing with donor PBMC autofluorescence extraction",
        },
        verify_only=verify_only,
    )


# ==============================================================================
# PANEL MARKERS POPULATION
# ==============================================================================


def populate_all_spectral_panel_markers(base_data: Path) -> None:
    """Populate panel.csv with biological marker annotations."""
    logger.info("Populating publication-derived panel markers...")
    spectral_dir = base_data / "spectral"

    scenith_markers = [
        ("B525-A", "Puromycin", "AF488"), ("B615-A", "CD20", "NFB 610_70S"),
        ("B695-A", "FoxP3", "BB700"), ("B725-A", "CD19", "PerCP_eFluor 710"),
        ("R667-A", "CD127", "JF 646"), ("R725-A", "CCR6", "AF700"),
        ("R780-A", "CD45RA", "APC_Cy7"), ("UV445-A", "Viability", "BUV395"),
        ("UV525-A", "CD27", "BUV496"), ("UV586-A", "CXCR5", "BUV563"),
        ("UV615-A", "CD56", "BUV615"), ("UV667-A", "CRTh2", "BUV661"),
        ("UV725-A", "IgD", "BUV737"), ("V445-A", "CD25", "BV421"),
        ("V586-A", "CD3", "BV570"), ("V615-A", "CD16", "BV605"),
        ("V667-A", "CD28", "BV650"), ("V725-A", "CD14", "BV711"),
        ("V780-A", "CCR7", "BV786"), ("Y586-A", "CD24", "PE"),
        ("Y615-A", "CXCR3", "PE_Dazzle 594"), ("Y667-A", "CD38", "PE_Cy5"),
        ("Y725-A", "CD4", "PE_AF700"), ("Y780-A", "TCRgd", "PE_Cy7"),
    ]

    crc_markers = [
        ("BUV395-A", "CD39", "BUV395"), ("BUV496-A", "CD45", "BUV496"),
        ("BUV563-A", "CD69", "BUV563"), ("BUV615-A", "TCRgd", "BUV615"),
        ("BUV661-A", "CD21", "BUV661"), ("BUV737-A", "CD127", "BUV737"),
        ("BUV805-A", "CD16", "BUV805"), ("BV421-A", "TIGIT", "BV421"),
        ("V500-A", "CD3", "V500"), ("BV570-A", "CD19", "BV570"),
        ("BV605-A", "CD103", "BV605"), ("BV650-A", "CD8", "BV650"),
        ("BV711-A", "CD56", "BV711"), ("BV750-A", "CD27", "BV750"),
        ("BV785-A", "HLA-DR", "BV785"), ("FITC-A", "CD4", "FITC"),
        ("RB545-A", "CD45RA", "RB545"), ("RB613-A", "CTLA-4", "RB613"),
        ("RB670-A", "CD161", "RB670"), ("RealBlue 705-A", "TIM-3", "RealBlue 705"),
        ("PerCP-A", "CD11c", "PerCP"), ("PE-A", "CD38", "PE"),
        ("PE-Dazzle 594-A", "CD25", "PE-Dazzle 594"), ("RY703-A", "LAG-3", "RY703"),
        ("PE-Cy7-A", "PD-1", "PE-Cy7"), ("APC-A", "FoxP3", "APC"),
        ("APC-R700-A", "CD28", "APC-R700"), ("APC-Cy7-A", "CD8a", "APC-Cy7"),
    ]

    datasets = [
        ("arguello_2020_scenith_metabolism", scenith_markers),
        ("de_vries_2023_crc_til_mucosal", crc_markers),
    ]

    for dname, markers in datasets:
        meta_p = spectral_dir / dname / "metadata"
        if not meta_p.exists():
            continue
        panel_p = meta_p / "panel.csv"
        rows = [["channel", "marker", "fluorophore"]]
        for ch, m, fl in markers:
            rows.append([ch, m, fl])
        with open(panel_p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(rows)


# ==============================================================================
# SUMMARY GENERATOR
# ==============================================================================


def generate_spectral_download_summary(base_data: Path) -> Dict[str, Any]:
    """Generate comprehensive summary JSON covering all 10 spectral datasets."""
    spectral_dir = base_data / "spectral"
    summary: Dict[str, Any] = {}

    dataset_names = [
        "arguello_2020_scenith_metabolism",
        "baglio_2024_osteosarcoma_spectral",
        "balin_2025_thymopoiesis_omip116",
        "barrachina_2026_ipsc_organoid_ccl5",
        "de_vries_2023_crc_til_mucosal",
        "flagr_2024_cytobatch_pbmc",
        "korotkaja_2024_breast_tumor_tme",
        "kreutmair_2026_icu_pbmc",
        "michaelis_2025_c9orf72_als_pbmc",
        "rockstein_2026_omip_whole_blood",
    ]

    for name in dataset_names:
        ds_dir = spectral_dir / name
        if not ds_dir.exists():
            continue

        ctrl_files = list((ds_dir / "raw" / "controls").glob("*.fcs")) if (ds_dir / "raw" / "controls").exists() else []
        samp_files = list((ds_dir / "raw" / "samples").glob("*.fcs")) if (ds_dir / "raw" / "samples").exists() else []
        all_fcs = list((ds_dir / "raw").rglob("*.fcs")) if (ds_dir / "raw").exists() else []

        total_bytes = sum(f.stat().st_size for f in all_fcs)
        manifest_p = ds_dir / "metadata" / "dataset_manifest.json"
        manifest_count = 0
        if manifest_p.exists():
            try:
                with open(manifest_p, "r", encoding="utf-8") as fp:
                    m_data = json.load(fp)
                    manifest_count = len(m_data) if isinstance(m_data, list) else 0
            except Exception:
                pass

        summary[name] = {
            "dataset_name": name,
            "controls_fcs": len(ctrl_files),
            "samples_fcs": len(samp_files),
            "total_fcs": len(all_fcs),
            "manifest_entries": manifest_count or len(all_fcs),
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        }

    summary_file = base_data / "metadata" / "spectral_download_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Spectral download summary written to metadata/%s (%d datasets)", summary_file.name, len(summary))
    return summary


def parse_selected_datasets(selection: str) -> List[str]:
    """Resolve user selection string (e.g. 'all', '1,4', 'scenith,rockstein') to a list of dataset IDs."""
    if not selection or selection.lower() == "all":
        return [item["id"] for item in SPECTRAL_DATASETS.values()]

    selected: List[str] = []
    parts = [p.strip().lower() for p in selection.split(",") if p.strip()]

    for part in parts:
        found = False
        # Check by numeric key
        if part in SPECTRAL_DATASETS:
            ds_id = SPECTRAL_DATASETS[part]["id"]
            if ds_id not in selected:
                selected.append(ds_id)
            found = True
            continue

        # Check by name or alias
        for item in SPECTRAL_DATASETS.values():
            if part == item["id"].lower() or part == item["name"].lower() or part in item["aliases"]:
                if item["id"] not in selected:
                    selected.append(item["id"])
                found = True
                break

        if not found:
            logger.warning("Unrecognized dataset selector '%s'. Use --list to view valid datasets.", part)

    return selected


def print_dataset_list() -> None:
    """Print clean list of available spectral datasets."""
    print("\nAvailable Spectral Flow Cytometry Datasets (10 Total):")
    print("-" * 85)
    for num, item in SPECTRAL_DATASETS.items():
        aliases_str = ", ".join(item["aliases"])
        print(f"[{num:>2}] {item['id']}")
        print(f"     Description : {item['description']}")
        print(f"     Aliases     : {aliases_str}")
    print("-" * 85)
    print("Example usage:")
    print("  python3 scripts/datasets/download_spectral_datasets.py --dataset 1")
    print("  python3 scripts/datasets/download_spectral_datasets.py --dataset 1,4,7")
    print("  python3 scripts/datasets/download_spectral_datasets.py --dataset scenith,rockstein")
    print("  python3 scripts/datasets/download_spectral_datasets.py --dataset all\n")


def main() -> None:
    """CLI entrypoint for spectral datasets download and verification."""
    parser = argparse.ArgumentParser(
        description="Download, organize, and verify 10 benchmark spectral flow cytometry datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base directory containing conventional/ and spectral/ (default: data/)",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Specify which dataset(s) to process: 'all', dataset number (e.g. 1), name, alias, or comma-separated list (e.g. 1,4 or scenith,organoid).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available spectral datasets and exit.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing local files and build manifests without remote downloads",
    )
    parser.add_argument(
        "--include-supplementary",
        action="store_true",
        default=False,
        help="Also download paper/publication and non-FCS supplementary files (default: disabled).",
    )
    args = parser.parse_args()

    if args.list:
        print_dataset_list()
        return

    base_dir = args.data_dir.resolve()
    logger.info("Target data directory: %s (mode: %s)", base_dir, "verify-only" if args.verify_only else "download/verify")

    selected_ids = parse_selected_datasets(args.dataset)
    if not selected_ids:
        logger.error("No valid datasets selected. Run with --list to view options.")
        sys.exit(1)

    logger.info("Selected datasets (%d): %s", len(selected_ids), ", ".join(selected_ids))

    dispatch_map = {
        "arguello_2020_scenith_metabolism": download_scenith_dataset,
        "de_vries_2023_crc_til_mucosal": download_crc_til_dataset,
        "rockstein_2026_omip_whole_blood": download_rockstein_dataset,
        "barrachina_2026_ipsc_organoid_ccl5": download_ipsc_organoid_dataset,
        "matsumoto_2026_ipsc_organoid_ccl5": download_ipsc_organoid_dataset,
        "michaelis_2025_c9orf72_als_pbmc": download_michaelis_dataset,
        "balin_2025_thymopoiesis_omip116": download_thymopoiesis_dataset,
        "korotkaja_2024_breast_tumor_tme": download_breast_tumor_dataset,
        "liu_2024_breast_tumor_tme": download_breast_tumor_dataset,
        "flagr_2024_cytobatch_pbmc": download_cytobatch_dataset,
        "baglio_2024_osteosarcoma_spectral": download_baglio_dataset,
        "kreutmair_2026_icu_pbmc": download_icu_pbmc_dataset,
    }

    for ds_id in selected_ids:
        if ds_id in dispatch_map:
            dispatch_map[ds_id](
                base_dir,
                verify_only=args.verify_only,
                include_supplementary=args.include_supplementary,
            )

    populate_all_spectral_panel_markers(base_dir)
    generate_spectral_download_summary(base_dir)
    logger.info("All selected spectral datasets processed successfully.")


if __name__ == "__main__":
    main()
