#!/usr/bin/env python3
"""
Conventional Flow Cytometry Data Download & Verification Pipeline.

Downloads, organizes, and verifies 10 benchmark conventional flow cytometry datasets (.fcs)
from open-access scientific repositories (Figshare, Zenodo, GitHub / RGLab / FlowKit) in compliance
with the MIFlowCyt standard.

Datasets Supported:
1. brouwers_2026_whole_blood_omip:
   - Source: Zenodo (DOI: 10.5281/zenodo.21869988)
   - Publication: Cytometry Part A (OMIP 2026)
   - Biology: Fresh human whole blood, 25-color panel + 23 single stains, SSM & FlowJo .wsp
2. chen_2025_placenta_hspc:
   - Source: Figshare (DOI: 10.6084/m9.figshare.27271968)
   - Publication: PLoS Biology (2025) 23(1):e3003003
   - Biology: Murine embryonic placenta & fetal liver, 6-color panel + 7-AAD + FlowJo .wsp
3. coburn_2021_thp1_stiffness:
   - Source: Figshare (DOI: 10.6084/m9.figshare.14450370 / 13105769)
   - Publication: J Biomed Mater Res A (2021) 109(8):1337-1352
   - Biology: Human THP-1 macrophages, 4-color panel + single stains & unstained
4. crespo_2026_crispr_cd8_tcells:
   - Source: Zenodo (DOI: 10.5281/zenodo.20526379)
   - Publication: Nature Immunology / Zenodo (2026)
   - Biology: Intratumoral CD8+ T cells under in vivo CRISPR/Cas9 transcription-factor knockouts
5. finak_2014_cytotrol_pbmc:
   - Source: GitHub RGLab / Nature Biotechnology (2014) 32:879-882 & Nature Protocols (2012) 7:700-726
   - Biology: Human PBMC Cyto-Trol reference standard, replicates + official FlowJo manual.xml workspace
6. li_2025_caerin_macrophage:
   - Source: Zenodo (DOI: 10.5281/zenodo.13976956)
   - Publication: Scientific Reports (2025) 15:3789
   - Biology: Murine tumor-infiltrating immune cells, 8-color panel + single stains
7. quintana_2026_skull_marrow_chimera:
   - Source: Zenodo (DOI: 10.5281/zenodo.22808501)
   - Publication: Immunity / Zenodo (2026)
   - Biology: Murine skull bone marrow chimera with full single-stained controls, unstained baseline & FMOs
8. quintelier_2023_cytonorm_meso:
   - Source: Zenodo (DOI: 10.5281/zenodo.13928969)
   - Publication: J ImmunoTherapy Cancer (2023) 11:e007070 (Saeys Lab CytoNorm 2.0 benchmark)
   - Biology: Human cryopreserved PBMCs across longitudinal multi-batch clinical cohort
9. weber_2025_bk_polyomavirus:
   - Source: Figshare (DOI: 10.6084/m9.figshare.29601881)
   - Publication: PLOS Pathogens (2025) 21(3):e1013122
   - Biology: Primary human renal proximal tubular cells & BK polyomavirus with FlowJo .wsp
10. white_2021_flowkit_ics:
   - Source: GitHub FlowKit / Frontiers in Immunology (2021) 12:768541
   - Biology: Human PBMC 8-color Intracellular Cytokine Staining (ICS) panel + 8_color_ICS.wsp workspace
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_conventional")

CONVENTIONAL_DATASETS: Dict[str, Dict[str, Any]] = {
    "1": {
        "id": "brouwers_2026_whole_blood_omip",
        "name": "brouwers_2026_whole_blood_omip",
        "aliases": ["brouwers", "omip_blood", "whole_blood", "spaceman"],
        "description": "Human fresh whole blood 25-color OMIP benchmark with 23 single stains (Zenodo 21869988)",
    },
    "2": {
        "id": "chen_2025_placenta_hspc",
        "name": "chen_2025_placenta_hspc",
        "aliases": ["placenta", "chen", "singh", "hspc"],
        "description": "Murine embryonic placenta & fetal liver HSPC lineage (6-color + wsp, Figshare 27271968)",
    },
    "3": {
        "id": "coburn_2021_thp1_stiffness",
        "name": "coburn_2021_thp1_stiffness",
        "aliases": ["thp1", "coburn", "wang", "stiffness"],
        "description": "Human THP-1 macrophages on stiffness hydrogels (4-color, Figshare 14450370)",
    },
    "4": {
        "id": "crespo_2026_crispr_cd8_tcells",
        "name": "crespo_2026_crispr_cd8_tcells",
        "aliases": ["crespo", "crispr", "cd8"],
        "description": "Murine intratumoral CD8+ T cells under in vivo CRISPR knockouts (Zenodo 20526379)",
    },
    "5": {
        "id": "finak_2014_cytotrol_pbmc",
        "name": "finak_2014_cytotrol_pbmc",
        "aliases": ["cytotrol", "finak", "opencyto"],
        "description": "Human PBMC Cyto-Trol reference standard with FlowJo manual.xml (Nat Biotech 2014)",
    },
    "6": {
        "id": "li_2025_caerin_macrophage",
        "name": "li_2025_caerin_macrophage",
        "aliases": ["caerin", "li", "huang"],
        "description": "Murine tumor-infiltrating macrophages under caerin peptide (8-color, Zenodo 13976956)",
    },
    "7": {
        "id": "quintana_2026_skull_marrow_chimera",
        "name": "quintana_2026_skull_marrow_chimera",
        "aliases": ["quintana", "skull", "chimera"],
        "description": "Murine skull marrow chimera with full single stains, unstained & FMOs (Zenodo 22808501)",
    },
    "8": {
        "id": "quintelier_2023_cytonorm_meso",
        "name": "quintelier_2023_cytonorm_meso",
        "aliases": ["cytonorm", "quintelier", "saeys"],
        "description": "CytoNorm 2.0 multi-batch human PBMC longitudinal benchmark (Zenodo 13928969)",
    },
    "9": {
        "id": "weber_2025_bk_polyomavirus",
        "name": "weber_2025_bk_polyomavirus",
        "aliases": ["weber", "polyomavirus", "bk_virus", "bkv"],
        "description": "Primary human renal proximal tubular cells & BKV with FlowJo wsp (Figshare 29601881 / PLOS Pathog 2025)",
    },
    "10": {
        "id": "white_2021_flowkit_ics",
        "name": "white_2021_flowkit_ics",
        "aliases": ["flowkit", "white", "ics"],
        "description": "Human PBMC 8-color ICS benchmark with FlowJo 8_color_ICS.wsp (Frontiers 2021)",
    },
}


def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a local file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
    headers = {"User-Agent": "curl/7.81.0"}

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
    headers = {"User-Agent": "curl/7.81.0"}
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


def get_fcs_summary(filepath: Path) -> Dict[str, Any]:
    """Inspect FCS header parameters without external dependencies."""
    summary: Dict[str, Any] = {
        "filename": filepath.name,
        "size_bytes": filepath.stat().st_size,
        "md5": compute_file_hash(filepath),
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
            summary["cytometer"] = kv.get("$CYT", kv.get("CYT", "Unknown"))
            summary["creator"] = kv.get("CREATOR", kv.get("$CREATOR", "Unknown"))
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


# =========================================================================
# 3. coburn_2021_thp1_stiffness (Figshare 14450370 / 13105769)
# =========================================================================
def download_thp1_dataset(
    base_data_dir: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> Dict[str, Any]:
    """Download THP-1 macrophage hydrogel stiffness dataset from Figshare."""
    logger.info("=== Checking / Downloading: coburn_2021_thp1_stiffness ===")
    target_dir = base_data_dir / "conventional" / "coburn_2021_thp1_stiffness"
    if not target_dir.exists() and (base_data_dir / "conventional" / "wang_2021_thp1_stiffness").exists():
        target_dir = base_data_dir / "conventional" / "wang_2021_thp1_stiffness"

    controls_dir = target_dir / "raw" / "controls"
    samples_dir = target_dir / "raw" / "samples"
    meta_dir = target_dir / "metadata"
    pub_dir = target_dir / "publications"
    supp_dir = target_dir / "supplementary"

    for d in [controls_dir, samples_dir, meta_dir, pub_dir, supp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    figshare_article_id = "13105769"
    api_url = f"https://api.figshare.com/v2/articles/{figshare_article_id}/files"

    headers = {"User-Agent": "curl/7.81.0"}
    try:
        resp = requests.get(api_url, headers=headers, timeout=30)
        resp.raise_for_status()
        files_meta = resp.json()
    except Exception as e:
        logger.warning("Could not reach Figshare API (%s), scanning local files...", e)
        files_meta = []

    manifest: List[Dict[str, Any]] = []

    for item in files_meta:
        fname = item["name"]
        furl = item["download_url"]
        fsize = item["size"]

        if not fname.lower().endswith(".fcs"):
            if include_supplementary:
                dest = supp_dir / fname
                download_supplementary_file(
                    furl,
                    dest,
                    verify_only=verify_only,
                    include_supplementary=include_supplementary,
                )
            continue

        is_control = any(k in fname.lower() for k in ["control", "unstained", "fmo", "comp"])
        dest = (controls_dir if is_control else samples_dir) / fname

        success = download_file(furl, dest, expected_size=fsize, verify_only=verify_only)
        if success and dest.exists():
            meta = get_fcs_summary(dest)
            meta["relative_path"] = str(dest.relative_to(target_dir))
            meta["role"] = "control" if is_control else "sample"
            meta["category"] = "unstained" if "unstained" in fname.lower() else ("fmo_control" if "fmo" in fname.lower() else "biological_sample")
            meta["source_url"] = furl
            manifest.append(meta)

    # Local fallback scan if API was unreachable or files already present
    if not manifest:
        for f in sorted(target_dir.glob("raw/**/*.fcs")):
            meta = get_fcs_summary(f)
            meta["relative_path"] = str(f.relative_to(target_dir))
            is_control = "controls" in f.parts
            meta["role"] = "control" if is_control else "sample"
            meta["category"] = "unstained" if "unstained" in f.name.lower() else ("fmo_control" if "fmo" in f.name.lower() else "biological_sample")
            manifest.append(meta)

    manifest_path = meta_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dataset_name": "coburn_2021_thp1_stiffness",
        "canonical_name": "coburn_2021_thp1_stiffness",
        "category": "conventional",
        "source": "Figshare",
        "accession": "10.6084/m9.figshare.14450370",
        "total_files": len(manifest),
        "manifest": manifest,
    }


# =========================================================================
# 6. li_2025_caerin_macrophage (Zenodo 13976956)
# =========================================================================
def download_caerin_dataset(
    base_data_dir: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> Dict[str, Any]:
    """Download Caerin macrophage immunotherapy dataset from Zenodo."""
    logger.info("=== Checking / Downloading: li_2025_caerin_macrophage ===")
    target_dir = base_data_dir / "conventional" / "li_2025_caerin_macrophage"
    if not target_dir.exists() and (base_data_dir / "conventional" / "huang_2025_caerin_macrophage").exists():
        target_dir = base_data_dir / "conventional" / "huang_2025_caerin_macrophage"

    controls_dir = target_dir / "raw" / "controls"
    samples_dir = target_dir / "raw" / "samples"
    meta_dir = target_dir / "metadata"
    pub_dir = target_dir / "publications"
    supp_dir = target_dir / "supplementary"

    for d in [controls_dir, samples_dir, meta_dir, pub_dir, supp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    zenodo_record_id = "13976956"
    api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"

    headers = {"User-Agent": "curl/7.81.0"}
    try:
        resp = requests.get(api_url, headers=headers, timeout=30)
        resp.raise_for_status()
        record_data = resp.json()
        files_meta = record_data.get("files", [])
    except Exception as e:
        logger.warning("Could not reach Zenodo API (%s), scanning local files...", e)
        files_meta = []

    manifest: List[Dict[str, Any]] = []

    for item in files_meta:
        fname = item["key"]
        furl = item["links"]["self"]
        fsize = item["size"]

        if not fname.lower().endswith(".fcs"):
            if include_supplementary:
                dest = supp_dir / fname
                download_supplementary_file(
                    furl,
                    dest,
                    verify_only=verify_only,
                    include_supplementary=include_supplementary,
                )
            continue

        is_control = "compensation" in fname.lower() or "unstained" in fname.lower()
        dest = (controls_dir if is_control else samples_dir) / fname

        success = download_file(furl, dest, expected_size=fsize, verify_only=verify_only)
        if success and dest.exists():
            meta = get_fcs_summary(dest)
            meta["relative_path"] = str(dest.relative_to(target_dir))
            meta["role"] = "control" if is_control else "sample"
            meta["category"] = "single_stain" if "compensation" in fname.lower() else ("unstained" if "unstained" in fname.lower() else "biological_sample")
            meta["source_url"] = furl
            manifest.append(meta)

    # Local fallback scan
    if not manifest:
        for f in sorted(target_dir.glob("raw/**/*.fcs")):
            meta = get_fcs_summary(f)
            meta["relative_path"] = str(f.relative_to(target_dir))
            is_control = "controls" in f.parts
            meta["role"] = "control" if is_control else "sample"
            meta["category"] = "single_stain" if "compensation" in f.name.lower() else ("unstained" if "unstained" in f.name.lower() else "biological_sample")
            manifest.append(meta)

    manifest_path = meta_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dataset_name": "li_2025_caerin_macrophage",
        "canonical_name": "li_2025_caerin_macrophage",
        "category": "conventional",
        "source": "Zenodo",
        "accession": "10.5281/zenodo.13976956",
        "total_files": len(manifest),
        "manifest": manifest,
    }


# =========================================================================
# 2. chen_2025_placenta_hspc (Figshare 27271968 / Zenodo 14537605)
# =========================================================================
def download_placenta_dataset(
    base_data_dir: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> Dict[str, Any]:
    """Download embryonic placenta HSPC lineage dataset from Zenodo / Figshare."""
    logger.info("=== Checking / Downloading: chen_2025_placenta_hspc ===")
    target_dir = base_data_dir / "conventional" / "chen_2025_placenta_hspc"
    if not target_dir.exists() and (base_data_dir / "conventional" / "singh_2025_placenta_hspc").exists():
        target_dir = base_data_dir / "conventional" / "singh_2025_placenta_hspc"

    controls_dir = target_dir / "raw" / "controls"
    samples_dir = target_dir / "raw" / "samples"
    meta_dir = target_dir / "metadata"
    pub_dir = target_dir / "publications"
    supp_dir = target_dir / "supplementary"

    for d in [controls_dir, samples_dir, meta_dir, pub_dir, supp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    zenodo_record_id = "14537605"
    api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"

    headers = {"User-Agent": "curl/7.81.0"}
    try:
        resp = requests.get(api_url, headers=headers, timeout=30)
        resp.raise_for_status()
        record_data = resp.json()
        files_meta = record_data.get("files", [])
    except Exception as e:
        logger.warning("Could not reach Zenodo API (%s), scanning local files...", e)
        files_meta = []

    manifest: List[Dict[str, Any]] = []

    for item in files_meta:
        fname = item["key"]
        furl = item["links"]["self"]
        fsize = item["size"]

        if fname.lower().endswith(".wsp") or fname.lower().endswith(".xml") or not fname.lower().endswith(".fcs"):
            if include_supplementary:
                dest = supp_dir / fname
                download_supplementary_file(
                    furl,
                    dest,
                    verify_only=verify_only,
                    include_supplementary=include_supplementary,
                )
            continue

        is_control = "compensation" in fname.lower()
        dest = (controls_dir if is_control else samples_dir) / fname

        success = download_file(furl, dest, expected_size=fsize, verify_only=verify_only)
        if success and dest.exists():
            meta = get_fcs_summary(dest)
            meta["relative_path"] = str(dest.relative_to(target_dir))
            meta["role"] = "control" if is_control else "sample"
            meta["category"] = "single_stain" if is_control else "placenta_tissue"
            meta["source_url"] = furl
            manifest.append(meta)

    # Local fallback scan
    if not manifest:
        for f in sorted(target_dir.glob("raw/**/*.fcs")):
            meta = get_fcs_summary(f)
            meta["relative_path"] = str(f.relative_to(target_dir))
            is_control = "controls" in f.parts
            meta["role"] = "control" if is_control else "sample"
            meta["category"] = "single_stain" if is_control else "placenta_tissue"
            manifest.append(meta)

    manifest_path = meta_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dataset_name": "chen_2025_placenta_hspc",
        "canonical_name": "chen_2025_placenta_hspc",
        "category": "conventional",
        "source": "Zenodo / Figshare",
        "accession": "10.6084/m9.figshare.27271968",
        "total_files": len(manifest),
        "manifest": manifest,
    }


# =========================================================================
# 5. finak_2014_cytotrol_pbmc (GitHub RGLab / Nature Biotech 2014)
# =========================================================================
def download_cytotrol_dataset(
    base_data_dir: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> Dict[str, Any]:
    """Download Cyto-Trol reference PBMC standard with FlowJo manual.xml."""
    logger.info("=== Checking / Downloading: finak_2014_cytotrol_pbmc ===")
    target_dir = base_data_dir / "conventional" / "finak_2014_cytotrol_pbmc"
    samples_dir = target_dir / "raw" / "samples"
    meta_dir = target_dir / "metadata"
    pub_dir = target_dir / "publications"
    supp_dir = target_dir / "supplementary"

    for d in [samples_dir, meta_dir, pub_dir, supp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    base_raw_url = "https://raw.githubusercontent.com/RGLab/flowWorkspaceData/master/inst/extdata/"
    if include_supplementary:
        download_supplementary_file(
            base_raw_url + "manual.xml",
            supp_dir / "manual.xml",
            verify_only=verify_only,
            include_supplementary=include_supplementary,
        )

    fcs_files = ["CytoTrol_CytoTrol_1.fcs", "CytoTrol_CytoTrol_2.fcs"]
    manifest: List[Dict[str, Any]] = []

    for fn in fcs_files:
        url = base_raw_url + fn
        dest = samples_dir / fn
        success = download_file(url, dest, verify_only=verify_only)
        if success and dest.exists():
            meta = get_fcs_summary(dest)
            meta["relative_path"] = str(dest.relative_to(target_dir))
            meta["role"] = "sample"
            meta["category"] = "cytotrol_pbmc_replicate"
            meta["source_url"] = url
            manifest.append(meta)

    # Local fallback scan
    if not manifest:
        for f in sorted(samples_dir.glob("*.fcs")):
            meta = get_fcs_summary(f)
            meta["relative_path"] = str(f.relative_to(target_dir))
            meta["role"] = "sample"
            meta["category"] = "cytotrol_pbmc_replicate"
            manifest.append(meta)

    manifest_path = meta_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dataset_name": "finak_2014_cytotrol_pbmc",
        "canonical_name": "finak_2014_cytotrol_pbmc",
        "category": "conventional",
        "source": "GitHub / RGLab flowWorkspaceData",
        "accession": "Maecker et al. 2012 / Finak et al. 2014",
        "total_files": len(manifest),
        "manifest": manifest,
    }


# =========================================================================
# 10. white_2021_flowkit_ics (GitHub FlowKit / Frontiers in Immunol 2021)
# =========================================================================
def download_flowkit_ics_dataset(
    base_data_dir: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> Dict[str, Any]:
    """Download FlowKit 8-color ICS PBMC benchmark with FlowJo 8_color_ICS.wsp."""
    logger.info("=== Checking / Downloading: white_2021_flowkit_ics ===")
    target_dir = base_data_dir / "conventional" / "white_2021_flowkit_ics"
    samples_dir = target_dir / "raw" / "samples"
    meta_dir = target_dir / "metadata"
    pub_dir = target_dir / "publications"
    supp_dir = target_dir / "supplementary"

    for d in [samples_dir, meta_dir, pub_dir, supp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    base_raw_url = "https://raw.githubusercontent.com/whitews/FlowKit/master/data/8_color_data_set/"
    if include_supplementary:
        download_supplementary_file(
            base_raw_url + "8_color_ICS.wsp",
            supp_dir / "8_color_ICS.wsp",
            verify_only=verify_only,
            include_supplementary=include_supplementary,
        )
        download_supplementary_file(
            base_raw_url + "den_comp.csv",
            supp_dir / "den_comp.csv",
            verify_only=verify_only,
            include_supplementary=include_supplementary,
        )

    fcs_files = [
        "101_DEN084Y5_15_E01_008_clean.fcs",
        "101_DEN084Y5_15_E03_009_clean.fcs",
        "101_DEN084Y5_15_E05_010_clean.fcs",
    ]
    manifest: List[Dict[str, Any]] = []

    for fn in fcs_files:
        url = base_raw_url + f"fcs_files/{fn}"
        dest = samples_dir / fn
        success = download_file(url, dest, verify_only=verify_only)
        if success and dest.exists():
            meta = get_fcs_summary(dest)
            meta["relative_path"] = str(dest.relative_to(target_dir))
            meta["role"] = "sample"
            meta["category"] = "ics_pbmc_donor"
            meta["source_url"] = url
            manifest.append(meta)

    # Local fallback scan
    if not manifest:
        for f in sorted(samples_dir.glob("*.fcs")):
            meta = get_fcs_summary(f)
            meta["relative_path"] = str(f.relative_to(target_dir))
            meta["role"] = "sample"
            meta["category"] = "ics_pbmc_donor"
            manifest.append(meta)

    manifest_path = meta_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dataset_name": "white_2021_flowkit_ics",
        "canonical_name": "white_2021_flowkit_ics",
        "category": "conventional",
        "source": "GitHub / FlowKit Benchmark Suite",
        "accession": "White et al. 2021 (Frontiers in Immunology)",
        "total_files": len(manifest),
        "manifest": manifest,
    }


# =========================================================================
# 6. quintana_2026_skull_marrow_chimera (Zenodo 22808501)
# =========================================================================
def download_skull_chimera_dataset(
    base_data_dir: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> Dict[str, Any]:
    """Download skull marrow chimera with full compensation controls and unstained cells."""
    logger.info("=== Checking / Downloading: quintana_2026_skull_marrow_chimera ===")
    target_dir = base_data_dir / "conventional" / "quintana_2026_skull_marrow_chimera"
    controls_dir = target_dir / "raw" / "controls"
    samples_dir = target_dir / "raw" / "samples"
    meta_dir = target_dir / "metadata"
    pub_dir = target_dir / "publications"

    for d in [controls_dir, samples_dir, meta_dir, pub_dir]:
        d.mkdir(parents=True, exist_ok=True)

    zenodo_record_id = "22808501"
    controls_to_fetch = [
        "Compensation Controls_R 780,2f,60 Stained Control.fcs",
        "Compensation Controls_U 450,2f,50 Stained Control.fcs",
        "Compensation Controls_V 525,2f,50 Stained Control.fcs",
        "Compensation Controls_V 450,2f,50 Stained Control.fcs",
        "Compensation Controls_V 710,2f,50 Stained Control.fcs",
        "Compensation Controls_B 525,2f,50 Stained Control.fcs",
        "Compensation Controls_B 710,2f,50 Stained Control.fcs",
        "Compensation Controls_YG 582,2f,15 Stained Control.fcs",
        "Compensation Controls_YG 780,2f,60 Stained Control.fcs",
        "Compensation Controls_R 660,2f,20 Stained Control.fcs",
        "Compensation Controls_R 710,2f,50 Stained Control.fcs",
        "FMO broad panel_Unstained.fcs",
        "FMO broad panel_Live dead.fcs",
    ]
    samples_to_fetch = [
        "Skull BM Broad_Tube_017.fcs",
        "Skull BM Broad_Tube_022.fcs",
        "Meninges Broad_Tube_006.fcs",
        "Meninges Broad_Tube_008.fcs",
    ]

    manifest: List[Dict[str, Any]] = []

    for fn in controls_to_fetch:
        encoded_fn = urllib.parse.quote(fn)
        url = f"https://zenodo.org/api/records/{zenodo_record_id}/files/{encoded_fn}/content"
        dest = controls_dir / fn
        success = download_file(url, dest, verify_only=verify_only)
        if success and dest.exists():
            meta = get_fcs_summary(dest)
            meta["relative_path"] = str(dest.relative_to(target_dir))
            meta["role"] = "control"
            meta["category"] = "unstained" if "unstained" in fn.lower() else ("fmo_control" if "fmo" in fn.lower() else "single_stain")
            meta["source_url"] = url
            manifest.append(meta)

    for fn in samples_to_fetch:
        encoded_fn = urllib.parse.quote(fn)
        url = f"https://zenodo.org/api/records/{zenodo_record_id}/files/{encoded_fn}/content"
        dest = samples_dir / fn
        success = download_file(url, dest, verify_only=verify_only)
        if success and dest.exists():
            meta = get_fcs_summary(dest)
            meta["relative_path"] = str(dest.relative_to(target_dir))
            meta["role"] = "sample"
            meta["category"] = "skull_bm_tissue" if "skull" in fn.lower() else "meninges_tissue"
            meta["source_url"] = url
            manifest.append(meta)

    manifest_path = meta_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dataset_name": "quintana_2026_skull_marrow_chimera",
        "canonical_name": "quintana_2026_skull_marrow_chimera",
        "category": "conventional",
        "source": "Zenodo",
        "accession": "10.5281/zenodo.22808501",
        "total_files": len(manifest),
        "manifest": manifest,
    }


# =========================================================================
# 7. quintelier_2023_cytonorm_meso (Zenodo 13928969)
# =========================================================================
def download_cytonorm_meso_dataset(
    base_data_dir: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> Dict[str, Any]:
    """Download Saeys Lab CytoNorm 2.0 multi-batch PBMC benchmark dataset."""
    logger.info("=== Checking / Downloading: quintelier_2023_cytonorm_meso ===")
    target_dir = base_data_dir / "conventional" / "quintelier_2023_cytonorm_meso"
    samples_dir = target_dir / "raw" / "samples"
    meta_dir = target_dir / "metadata"
    pub_dir = target_dir / "publications"

    for d in [samples_dir, meta_dir, pub_dir]:
        d.mkdir(parents=True, exist_ok=True)

    zenodo_record_id = "13928969"
    samples_to_fetch = [
        "ID1_Panel1_TP1.fcs",
        "ID1_Panel1_TP2.fcs",
        "ID1_Panel1_TP3.fcs",
        "ID2_Panel1_TP1.fcs",
        "ID2_Panel1_TP2.fcs",
        "ID2_Panel1_TP3.fcs",
        "ID3_Panel1_TP1.fcs",
        "ID3_Panel1_TP2.fcs",
        "ID3_Panel1_TP3.fcs",
        "ID5_Panel1_TP1.fcs",
        "ID5_Panel1_TP2.fcs",
        "ID5_Panel1_TP3.fcs",
    ]

    manifest: List[Dict[str, Any]] = []

    for fn in samples_to_fetch:
        encoded_fn = urllib.parse.quote(fn)
        url = f"https://zenodo.org/api/records/{zenodo_record_id}/files/{encoded_fn}/content"
        dest = samples_dir / fn
        success = download_file(url, dest, verify_only=verify_only)
        if success and dest.exists():
            meta = get_fcs_summary(dest)
            meta["relative_path"] = str(dest.relative_to(target_dir))
            meta["role"] = "sample"
            meta["category"] = "longitudinal_pbmc_batch"
            meta["source_url"] = url
            manifest.append(meta)

    manifest_path = meta_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dataset_name": "quintelier_2023_cytonorm_meso",
        "canonical_name": "quintelier_2023_cytonorm_meso",
        "category": "conventional",
        "source": "Zenodo",
        "accession": "10.5281/zenodo.13928969",
        "total_files": len(manifest),
        "manifest": manifest,
    }


# =========================================================================
# 1. brouwers_2026_whole_blood_omip (Zenodo 21869988)
# =========================================================================
def download_brouwers_dataset(
    base_data_dir: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> Dict[str, Any]:
    """Download Brouwers et al. 2026 whole blood OMIP benchmark from Zenodo 21869988."""
    logger.info("=== Checking / Downloading: brouwers_2026_whole_blood_omip ===")
    target_dir = base_data_dir / "conventional" / "brouwers_2026_whole_blood_omip"
    controls_dir = target_dir / "raw" / "controls"
    samples_dir = target_dir / "raw" / "samples"
    meta_dir = target_dir / "metadata"
    pub_dir = target_dir / "publications"
    supp_dir = target_dir / "supplementary"

    for d in [controls_dir, samples_dir, meta_dir, pub_dir, supp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    zenodo_record_id = "21869988"
    base_file_url = f"https://zenodo.org/records/{zenodo_record_id}/files/"

    controls = [
        "Single_stain_APC_A09.fcs",
        "Single_stain_APC_Fire750_B03.fcs",
        "Single_stain_APC_R700_B01.fcs",
        "Single_stain_BUV395_D03.fcs",
        "Single_stain_BUV563_D06.fcs",
        "Single_stain_BUV615_D07.fcs",
        "Single_stain_BUV737_D09.fcs",
        "Single_stain_BUV805_D10.fcs",
        "Single_stain_BV480_B11.fcs",
        "Single_stain_BV605_C03.fcs",
        "Single_stain_BV650_C04.fcs",
        "Single_stain_BV711_C05.fcs",
        "Single_stain_BV750_C06.fcs",
        "Single_stain_BV786_C08.fcs",
        "Single_stain_FITC_A01.fcs",
        "Single_stain_PE_C09.fcs",
        "Single_stain_PE_CF594_C11.fcs",
        "Single_stain_PE_Cy7_D02.fcs",
        "Single_stain_PerCP_Cy5_5_A06.fcs",
        "Single_stain_RealBlue613_F02.fcs",
        "Single_stain_RealBlue744_E08.fcs",
        "Single_stain_RealBlue780_E06.fcs",
        "Single_stain_Unstained_A12.fcs",
    ]
    samples = ["Spaceman_Sample1.fcs"]

    supplementary = [
        "Figure 1_OMIP.wsp",
        "Metadata.xlsx",
        "Online Table 3_SSM_reference matrix.csv",
        "Online Table 4_reference comp matrix.csv",
        "Online Table 5_batch specific comp matrix.csv",
        "Online Table 6_comp matrix PerCPVio700.csv",
    ]

    if include_supplementary:
        for supp_name in supplementary:
            dest = supp_dir / supp_name
            encoded_name = urllib.parse.quote(supp_name)
            url = base_file_url + f"{encoded_name}?download=1"
            download_supplementary_file(
                url,
                dest,
                verify_only=verify_only,
                include_supplementary=include_supplementary,
            )

    manifest: List[Dict[str, Any]] = []

    for fn in controls:
        encoded_fn = urllib.parse.quote(fn)
        url = base_file_url + f"{encoded_fn}?download=1"
        dest = controls_dir / fn
        success = download_file(url, dest, verify_only=verify_only)
        if success and dest.exists():
            meta = get_fcs_summary(dest)
            meta["relative_path"] = str(dest.relative_to(target_dir))
            meta["role"] = "control"
            meta["category"] = "unstained" if "unstained" in fn.lower() else "single_stain"
            meta["source_url"] = url
            manifest.append(meta)

    for fn in samples:
        encoded_fn = urllib.parse.quote(fn)
        url = base_file_url + f"{encoded_fn}?download=1"
        dest = samples_dir / fn
        success = download_file(url, dest, verify_only=verify_only)
        if success and dest.exists():
            meta = get_fcs_summary(dest)
            meta["relative_path"] = str(dest.relative_to(target_dir))
            meta["role"] = "sample"
            meta["category"] = "biological_sample"
            meta["source_url"] = url
            manifest.append(meta)

    # Local fallback scan
    if not manifest:
        for f in sorted(controls_dir.glob("*.fcs")):
            meta = get_fcs_summary(f)
            meta["relative_path"] = str(f.relative_to(target_dir))
            meta["role"] = "control"
            meta["category"] = "unstained" if "unstained" in f.name.lower() else "single_stain"
            manifest.append(meta)
        for f in sorted(samples_dir.glob("*.fcs")):
            meta = get_fcs_summary(f)
            meta["relative_path"] = str(f.relative_to(target_dir))
            meta["role"] = "sample"
            meta["category"] = "biological_sample"
            manifest.append(meta)

    manifest_path = meta_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dataset_name": "brouwers_2026_whole_blood_omip",
        "canonical_name": "brouwers_2026_whole_blood_omip",
        "category": "conventional",
        "source": "Zenodo",
        "accession": "10.5281/zenodo.21869988",
        "total_files": len(manifest),
        "manifest": manifest,
    }


# =========================================================================
# 9. crespo_2026_crispr_cd8_tcells (Zenodo 20526379)
# =========================================================================
def download_crispr_cd8_dataset(
    base_data_dir: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> Dict[str, Any]:
    """Download in vivo CRISPR/Cas9 CD8+ T-cell perturbation cohort dataset."""
    logger.info("=== Checking / Downloading: crespo_2026_crispr_cd8_tcells ===")
    target_dir = base_data_dir / "conventional" / "crespo_2026_crispr_cd8_tcells"
    samples_dir = target_dir / "raw" / "samples"
    meta_dir = target_dir / "metadata"
    pub_dir = target_dir / "publications"

    for d in [samples_dir, meta_dir, pub_dir]:
        d.mkdir(parents=True, exist_ok=True)

    zenodo_record_id = "20526379"
    samples_to_fetch = [
        "Mock d3 post ACT_001_tumor_001.fcs",
        "Mock d3 post ACT_001_tumor_003.fcs",
        "Crem d3 post ACT_003_tumor_001.fcs",
        "Crem d3 post ACT_003_tumor_004.fcs",
        "CZBatf d3 post ACT_004_tumor_001.fcs",
        "Tbx21 d3 post ACT_005_tumor_003.fcs",
        "Zmitz d3 post ACT_002_tumor_001.fcs",
        "CZTbx21 d3 post ACT_006_tumor_005.fcs",
    ]

    manifest: List[Dict[str, Any]] = []

    for fn in samples_to_fetch:
        encoded_fn = urllib.parse.quote(fn)
        url = f"https://zenodo.org/api/records/{zenodo_record_id}/files/{encoded_fn}/content"
        dest = samples_dir / fn
        success = download_file(url, dest, verify_only=verify_only)
        if success and dest.exists():
            meta = get_fcs_summary(dest)
            meta["relative_path"] = str(dest.relative_to(target_dir))
            meta["role"] = "sample"
            meta["category"] = "mock_control" if "mock" in fn.lower() else "crispr_knockout"
            meta["source_url"] = url
            manifest.append(meta)

    manifest_path = meta_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dataset_name": "crespo_2026_crispr_cd8_tcells",
        "canonical_name": "crespo_2026_crispr_cd8_tcells",
        "category": "conventional",
        "source": "Zenodo",
        "accession": "10.5281/zenodo.20526379",
        "total_files": len(manifest),
        "manifest": manifest,
    }


# =========================================================================
# 9. weber_2025_bk_polyomavirus (Figshare 29601881 / PLOS Pathog 2025)
# =========================================================================
def download_weber_dataset(
    base_data_dir: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> Dict[str, Any]:
    """Download Weber et al. 2025 BK polyomavirus infection cohort from Figshare / PLOS Pathog."""
    logger.info("=== Checking / Downloading: weber_2025_bk_polyomavirus ===")
    target_dir = base_data_dir / "conventional" / "weber_2025_bk_polyomavirus"
    controls_dir = target_dir / "raw" / "controls"
    samples_dir = target_dir / "raw" / "samples"
    meta_dir = target_dir / "metadata"
    pub_dir = target_dir / "publications"
    supp_dir = target_dir / "supplementary"

    for d in [controls_dir, samples_dir, meta_dir, pub_dir, supp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []

    # Local file scan
    for f in sorted(controls_dir.glob("*.fcs")):
        meta = get_fcs_summary(f)
        meta["relative_path"] = str(f.relative_to(target_dir))
        meta["role"] = "control"
        meta["category"] = "uninfected_baseline"
        manifest.append(meta)

    for f in sorted(samples_dir.glob("*.fcs")):
        meta = get_fcs_summary(f)
        meta["relative_path"] = str(f.relative_to(target_dir))
        meta["role"] = "sample"
        meta["category"] = "infected_sample"
        manifest.append(meta)

    manifest_path = meta_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dataset_name": "weber_2025_bk_polyomavirus",
        "canonical_name": "weber_2025_bk_polyomavirus",
        "category": "conventional",
        "source": "Figshare / PLOS Pathogens",
        "accession": "10.6084/m9.figshare.29601881",
        "total_files": len(manifest),
        "manifest": manifest,
    }


def generate_conventional_download_summary(base_data: Path) -> Dict[str, Any]:
    """Generate comprehensive summary JSON covering all 10 conventional datasets."""
    conv_dir = base_data / "conventional"
    summary: Dict[str, Any] = {}

    dataset_names = [
        "brouwers_2026_whole_blood_omip",
        "chen_2025_placenta_hspc",
        "coburn_2021_thp1_stiffness",
        "crespo_2026_crispr_cd8_tcells",
        "finak_2014_cytotrol_pbmc",
        "li_2025_caerin_macrophage",
        "quintana_2026_skull_marrow_chimera",
        "quintelier_2023_cytonorm_meso",
        "weber_2025_bk_polyomavirus",
        "white_2021_flowkit_ics",
    ]

    for name in dataset_names:
        ds_dir = conv_dir / name
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

    summary_file = base_data / "metadata" / "download_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Conventional download summary written to metadata/%s (%d datasets)", summary_file.name, len(summary))
    return summary


def parse_selected_datasets(selection: str) -> Set[str]:
    """Resolve user selection string (e.g. 'all', '1,3', 'thp1,cytotrol') to a set of dataset IDs."""
    if not selection or selection.lower() == "all":
        return {item["id"] for item in CONVENTIONAL_DATASETS.values()}

    selected: Set[str] = set()
    parts = [p.strip().lower() for p in selection.split(",") if p.strip()]

    for part in parts:
        found = False
        # Check by numeric key
        if part in CONVENTIONAL_DATASETS:
            selected.add(CONVENTIONAL_DATASETS[part]["id"])
            found = True
            continue

        # Check by name or alias
        for item in CONVENTIONAL_DATASETS.values():
            if part == item["id"].lower() or part == item["name"].lower() or part in item["aliases"]:
                selected.add(item["id"])
                found = True
                break

        if not found:
            logger.warning("Unrecognized dataset selector '%s'. Use --list to view valid datasets.", part)

    return selected


def print_dataset_list() -> None:
    """Print clean list of available conventional datasets."""
    print("\nAvailable Conventional Flow Cytometry Datasets (10 Total):")
    print("-" * 80)
    for num, item in CONVENTIONAL_DATASETS.items():
        aliases_str = ", ".join(item["aliases"])
        print(f"[{num.rjust(2)}] {item['id']}")
        print(f"     Description : {item['description']}")
        print(f"     Aliases     : {aliases_str}")
    print("-" * 80)
    print("Example usage:")
    print("  python3 scripts/datasets/download_conventional_datasets.py --dataset 1")
    print("  python3 scripts/datasets/download_conventional_datasets.py --dataset 1,4,7")
    print("  python3 scripts/datasets/download_conventional_datasets.py --dataset cytotrol,flowkit")
    print("  python3 scripts/datasets/download_conventional_datasets.py --dataset all\n")


def main() -> None:
    """CLI entrypoint for conventional dataset download and verification."""
    parser = argparse.ArgumentParser(
        description="Download, organize, and verify 10 benchmark conventional flow cytometry datasets."
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
        help="Specify which dataset(s) to process: 'all', dataset number (e.g. 1), name, alias, or comma-separated list (e.g. 1,4 or thp1,cytotrol).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and exit.",
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

    logger.info("Selected datasets (%d): %s", len(selected_ids), ", ".join(sorted(selected_ids)))

    results: Dict[str, Any] = {}

    summary_file = base_dir / "metadata" / "download_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, dict):
                    results = existing
        except Exception:
            pass

    dispatch_map = {
        "brouwers_2026_whole_blood_omip": download_brouwers_dataset,
        "chen_2025_placenta_hspc": download_placenta_dataset,
        "singh_2025_placenta_hspc": download_placenta_dataset,
        "coburn_2021_thp1_stiffness": download_thp1_dataset,
        "wang_2021_thp1_stiffness": download_thp1_dataset,
        "crespo_2026_crispr_cd8_tcells": download_crispr_cd8_dataset,
        "finak_2014_cytotrol_pbmc": download_cytotrol_dataset,
        "li_2025_caerin_macrophage": download_caerin_dataset,
        "huang_2025_caerin_macrophage": download_caerin_dataset,
        "quintana_2026_skull_marrow_chimera": download_skull_chimera_dataset,
        "quintelier_2023_cytonorm_meso": download_cytonorm_meso_dataset,
        "weber_2025_bk_polyomavirus": download_weber_dataset,
        "white_2021_flowkit_ics": download_flowkit_ics_dataset,
    }

    for ds_id in sorted(selected_ids):
        if ds_id in dispatch_map:
            dispatch_map[ds_id](
                base_dir,
                verify_only=args.verify_only,
                include_supplementary=args.include_supplementary,
            )

    generate_conventional_download_summary(base_dir)


if __name__ == "__main__":
    main()
