"""Download and verification engine for FlowX benchmark datasets.

Integrates with conventional and spectral dataset repositories to download,
organize, and verify benchmark datasets according to the MIFlowCyt standard.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from flowx.datasets.registry import BENCHMARK_DATASETS, DatasetInfo

logger = logging.getLogger("flowx.datasets")

# Ensure repository root is on sys.path for scripts.datasets import resolution
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path and (_REPO_ROOT / "scripts" / "datasets").exists():
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class DatasetVerificationResult:
    """Result of local verification for a benchmark dataset."""

    dataset_id: str
    category: str
    dataset_dir: Path
    exists: bool = False
    controls_count: int = 0
    samples_count: int = 0
    total_fcs: int = 0
    total_size_mb: float = 0.0
    manifest_exists: bool = False
    manifest_valid: bool = False
    manifest_entries: int = 0
    publications_count: int = 0
    supplementary_count: int = 0
    status: str = "Missing"  # "Ready", "Partial", "Missing"
    issues: List[str] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        """Convert verification result to dictionary representation."""
        return {
            "dataset_id": self.dataset_id,
            "category": self.category,
            "exists": self.exists,
            "controls": self.controls_count,
            "samples": self.samples_count,
            "total_fcs": self.total_fcs,
            "total_size_mb": round(self.total_size_mb, 2),
            "manifest_entries": self.manifest_entries,
            "publications": self.publications_count,
            "supplementary": self.supplementary_count,
            "status": self.status,
            "issues": self.issues,
        }


def _get_download_dispatcher(dataset_id: str, category: str) -> Optional[Callable[..., Any]]:
    """Dynamically resolve download function from scripts/datasets."""
    try:
        if category == "conventional":
            from scripts.datasets import download_conventional_datasets as dconv
            from scripts.datasets import download_spectral_datasets as dspec

            dispatch_map = {
                "arguello_2020_scenith_metabolism": dspec.download_scenith_dataset,
                "brouwers_2026_whole_blood_omip": dconv.download_brouwers_dataset,
                "chen_2025_placenta_hspc": dconv.download_placenta_dataset,
                "coburn_2021_thp1_stiffness": dconv.download_thp1_dataset,
                "crespo_2026_crispr_cd8_tcells": dconv.download_crispr_cd8_dataset,
                "finak_2014_cytotrol_pbmc": dconv.download_cytotrol_dataset,
                "li_2025_caerin_macrophage": dconv.download_caerin_dataset,
                "quintana_2026_skull_marrow_chimera": dconv.download_skull_chimera_dataset,
                "quintelier_2023_cytonorm_meso": dconv.download_cytonorm_meso_dataset,
                "weber_2025_bk_polyomavirus": dconv.download_weber_dataset,
                "white_2021_flowkit_ics": dconv.download_flowkit_ics_dataset,
            }
            return dispatch_map.get(dataset_id)

        elif category == "spectral":
            from scripts.datasets import download_spectral_datasets as dspec

            dispatch_map = {
                "de_vries_2023_crc_til_mucosal": dspec.download_crc_til_dataset,
                "rockstein_2026_omip_whole_blood": dspec.download_rockstein_dataset,
                "barrachina_2026_ipsc_organoid_ccl5": dspec.download_ipsc_organoid_dataset,
                "michaelis_2025_c9orf72_als_pbmc": dspec.download_michaelis_dataset,
                "balin_2025_thymopoiesis_omip116": dspec.download_thymopoiesis_dataset,
                "korotkaja_2024_breast_tumor_tme": dspec.download_breast_tumor_dataset,
                "flagr_2024_cytobatch_pbmc": dspec.download_cytobatch_dataset,
                "baglio_2024_osteosarcoma_spectral": dspec.download_baglio_dataset,
                "kreutmair_2026_icu_pbmc": dspec.download_icu_pbmc_dataset,
            }
            return dispatch_map.get(dataset_id)
    except ImportError as err:
        logger.warning("Could not import dataset download modules: %s", err)
        return None

    return None


def verify_dataset(info: DatasetInfo, data_dir: Path) -> DatasetVerificationResult:
    """Verify local presence and integrity of a single benchmark dataset.

    Parameters
    ----------
    info : DatasetInfo
        Metadata descriptor for the dataset.
    data_dir : Path
        Base data directory containing conventional/ and spectral/.

    Returns
    -------
    DatasetVerificationResult
        Verification details including file counts, size, and manifest health.
    """
    ds_dir = data_dir / info.category / info.id
    res = DatasetVerificationResult(
        dataset_id=info.id,
        category=info.category,
        dataset_dir=ds_dir,
    )

    if not ds_dir.exists() or not ds_dir.is_dir():
        res.issues.append("Dataset directory does not exist")
        res.status = "Missing"
        return res

    res.exists = True

    # Count FCS files
    ctrl_dir = ds_dir / "raw" / "controls"
    samp_dir = ds_dir / "raw" / "samples"

    ctrl_files = list(ctrl_dir.glob("*.fcs")) if ctrl_dir.exists() else []
    samp_files = list(samp_dir.glob("*.fcs")) if samp_dir.exists() else []
    all_fcs = ctrl_files + samp_files

    res.controls_count = len(ctrl_files)
    res.samples_count = len(samp_files)
    res.total_fcs = len(all_fcs)

    total_bytes = sum(f.stat().st_size for f in all_fcs if f.is_file())
    res.total_size_mb = total_bytes / (1024 * 1024)

    # Check publication and supplementary files
    pub_dir = ds_dir / "publications"
    supp_dir = ds_dir / "supplementary"
    res.publications_count = len(list(pub_dir.glob("*.*"))) if pub_dir.exists() else 0
    res.supplementary_count = len(list(supp_dir.glob("*.*"))) if supp_dir.exists() else 0

    # Check manifest
    manifest_file = ds_dir / "metadata" / "dataset_manifest.json"
    if manifest_file.exists():
        res.manifest_exists = True
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                if isinstance(manifest_data, list):
                    res.manifest_entries = len(manifest_data)
                    res.manifest_valid = res.manifest_entries == res.total_fcs
                    if not res.manifest_valid:
                        res.issues.append(
                            f"Manifest entry count ({res.manifest_entries}) differs from FCS file count ({res.total_fcs})"
                        )
                else:
                    res.issues.append("Manifest is not a valid JSON list")
        except Exception as e:
            res.issues.append(f"Failed to parse manifest: {e}")
    else:
        res.issues.append("dataset_manifest.json missing in metadata/")

    # Determine status
    if res.total_fcs == 0:
        res.status = "Missing"
        res.issues.append("No .fcs files found in raw/controls or raw/samples")
    elif info.files_expected > 0 and res.total_fcs < info.files_expected:
        res.status = "Partial"
        res.issues.append(f"FCS count ({res.total_fcs}) is less than expected ({info.files_expected})")
    elif res.manifest_valid:
        res.status = "Ready"
    else:
        res.status = "Partial"

    return res


def download_dataset(
    info: DatasetInfo,
    data_dir: Path,
    verify_only: bool = False,
    include_supplementary: bool = False,
) -> bool:
    """Download and/or verify a single benchmark dataset.

    Parameters
    ----------
    info : DatasetInfo
        Dataset metadata object.
    data_dir : Path
        Base data directory.
    verify_only : bool
        If True, skip downloading and only verify existing files.
    include_supplementary : bool
        If True, also download paper/publication and non-FCS supplementary
        files. Defaults to False (disabled).

    Returns
    -------
    bool
        True if verification or download completed successfully.
    """
    fn = _get_download_dispatcher(info.id, info.category)
    if fn is None:
        logger.error("No download handler registered for dataset '%s'", info.id)
        return False

    try:
        fn(
            data_dir.resolve(),
            verify_only=verify_only,
            include_supplementary=include_supplementary,
        )
        return True
    except Exception as exc:
        logger.error("Failed to process dataset '%s': %s", info.id, exc)
        return False


def update_catalog_summaries(data_dir: Path) -> None:
    """Regenerate metadata/download_summary.json and spectral_download_summary.json."""
    try:
        from scripts.datasets.download_conventional_datasets import (
            generate_conventional_download_summary,
        )

        generate_conventional_download_summary(data_dir.resolve())
    except Exception as err:
        logger.debug("Conventional summary generation skipped: %s", err)

    try:
        from scripts.datasets.download_spectral_datasets import (
            generate_spectral_download_summary,
        )

        generate_spectral_download_summary(data_dir.resolve())
    except Exception as err:
        logger.debug("Spectral summary generation skipped: %s", err)

