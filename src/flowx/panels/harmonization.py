"""Flow Cytometry Panel Harmonization Module.

Provides automated and user-configurable optical detector-to-canonical marker
harmonization across multi-patient cohorts, diverse cytometers (Beckman Coulter,
BD, Cytek, Sony), and software exports (flowCore, Cytobank, FlowJo).

Ensures all samples in an analysis cohort share identical column schemas and
standardized biological marker labels before downstream batch correction
(CytoNorm / iMUBAC), gating, or high-dimensional clustering (FlowSOM / PhenoGraph).
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from flowx.io.paths import sanitize_path

logger = logging.getLogger("flow_cytometry.harmonization")


@dataclass
class CanonicalChannelSpec:
    """Specification for a single standardized canonical parameter or marker.

    Attributes:
        canonical_name: Standardized target column name (e.g. 'CD14', 'FSC-A', 'HLA-DR').
        aliases: List of detector names, stain labels, or regular expressions to match.
        fluorophore: Standard fluorophore / dye name (e.g. 'FITC', 'PE-Cy7', 'Pacific Blue').
        channel_type: Parameter category ('scatter', 'fluorescent', 'time', 'viability', or 'other').
        required: If True, missing channel triggers error or high-priority warning based on policy.
        description: Biological or optical explanation of this parameter.
    """

    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    fluorophore: Optional[str] = None
    channel_type: str = "fluorescent"
    required: bool = False
    description: Optional[str] = None


@dataclass
class HarmonizationResult:
    """Audit and result container for a single harmonized FCS sample.

    Attributes:
        sample_name: Base filename or identifier of the FCS sample.
        harmonized_df: Pandas DataFrame with column headers harmonized to canonical names.
        channel_mapping: Mapping from canonical target name to original source detector.
        reverse_mapping: Mapping from original source detector to canonical target name.
        missing_channels: List of canonical channels that could not be matched in this sample.
        unmapped_channels: List of original sample detectors not assigned to any canonical target.
        warnings: List of warning strings generated during harmonization.
        success: True if all required canonical channels were successfully mapped.
    """

    sample_name: str
    harmonized_df: pd.DataFrame
    channel_mapping: Dict[str, str] = field(default_factory=dict)
    reverse_mapping: Dict[str, str] = field(default_factory=dict)
    missing_channels: List[str] = field(default_factory=list)
    unmapped_channels: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    success: bool = True

    def summary_dict(self) -> Dict[str, Any]:
        """Convert harmonization summary metrics into a dictionary for auditing."""
        return {
            "sample": self.sample_name,
            "mapped_channels": len(self.channel_mapping),
            "missing_channels": len(self.missing_channels),
            "unmapped_channels": len(self.unmapped_channels),
            "success": self.success,
            "warnings_count": len(self.warnings),
        }


@dataclass
class CohortHarmonizationReport:
    """Comprehensive cohort-level concordance summary across multiple samples.

    Attributes:
        total_samples: Number of samples inspected.
        canonical_channels: List of target canonical channel names.
        sample_results: Dictionary mapping sample names to individual HarmonizationResults.
        concordance_matrix: Matrix DataFrame (samples as rows, canonical channels as columns).
        missing_summary: Dictionary mapping canonical channel names to samples where it is missing.
        is_fully_concordant: True if all samples successfully mapped all required channels.
    """

    total_samples: int
    canonical_channels: List[str]
    sample_results: Dict[str, HarmonizationResult]
    concordance_matrix: pd.DataFrame
    missing_summary: Dict[str, List[str]]
    is_fully_concordant: bool

    def to_markdown(self) -> str:
        """Render a publication-ready Markdown concordance table."""
        lines: List[str] = [
            "# Cohort Panel Harmonization Concordance Report",
            "",
            f"- **Total Samples:** {self.total_samples}",
            f"- **Target Canonical Channels ({len(self.canonical_channels)}):** {', '.join(self.canonical_channels)}",
            f"- **Fully Concordant:** {'✅ YES' if self.is_fully_concordant else '⚠️ NO (Discrepancies Detected)'}",
            "",
            "## 1. Cross-Sample Channel Concordance Matrix",
            "",
        ]

        # Build markdown table header
        cols = ["Sample"] + self.canonical_channels
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join([":---"] + [":---:"] * len(self.canonical_channels)) + " |")

        for sample_name, res in self.sample_results.items():
            row = [f"`{sample_name}`"]
            for ch in self.canonical_channels:
                val = res.channel_mapping.get(ch, "—")
                row.append(f"`{val}`" if val != "—" else "**MISSING**")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        if self.missing_summary:
            lines.append("## 2. Missing Channel Summary")
            for ch, smps in self.missing_summary.items():
                lines.append(f"- **{ch}:** Missing across {len(smps)} sample(s): {', '.join(smps[:5])}")
            lines.append("")

        return "\n".join(lines)

    def export(self, out_dir: Union[str, Path]) -> Tuple[Path, Path]:
        """Export concordance CSV matrix and Markdown report to output directory.

        Args:
            out_dir: Directory where harmonization artifacts will be saved.

        Returns:
            Tuple of (csv_path, markdown_path).
        """
        out_p = Path(out_dir).resolve()
        out_p.mkdir(parents=True, exist_ok=True)

        csv_path = out_p / "panel_harmonization_matrix.csv"
        md_path = out_p / "panel_harmonization_report.md"

        self.concordance_matrix.to_csv(csv_path)
        md_path.write_text(self.to_markdown(), encoding="utf-8")

        return csv_path, md_path


class PanelHarmonizer:
    """Engine for aligning, verifying, and harmonizing flow cytometry parameter channels.

    Supports configurable alias mappings defined in YAML/JSON, automatic marker
    decomposition, synonym resolution, and collision prevention.
    """

    def __init__(
        self,
        channels: Dict[str, CanonicalChannelSpec],
        name: str = "Custom Panel Harmonizer",
        description: str = "",
        on_missing: str = "warn",
        drop_unmapped: bool = False,
        preserve_raw_names: bool = True,
    ) -> None:
        """Initialize PanelHarmonizer.

        Args:
            channels: Dictionary mapping canonical channel name to CanonicalChannelSpec.
            name: Human-readable name of this panel harmonization profile.
            description: Description of the target panel or assay.
            on_missing: Action when a required channel is missing ('warn', 'error', 'ignore').
            drop_unmapped: If True, columns not present in canonical channels are dropped.
            preserve_raw_names: If True, preserves original channel mapping in sample metadata.
        """
        self.channels = channels
        self.name = name
        self.description = description
        self.on_missing = on_missing.lower().strip()
        self.drop_unmapped = drop_unmapped
        self.preserve_raw_names = preserve_raw_names

    @classmethod
    def load(cls, source: Union[str, Path, Dict[str, Any]]) -> PanelHarmonizer:
        """Load PanelHarmonizer from a YAML file, JSON file, or dictionary.

        Supports both comprehensive structured schemas and lightweight flat alias lists.

        Args:
            source: Filepath or dictionary containing the harmonization configuration.

        Returns:
            Configured PanelHarmonizer instance.

        Raises:
            FileNotFoundError: If configuration file does not exist.
            ValueError: If configuration format is invalid.
        """
        data: Dict[str, Any] = {}

        if isinstance(source, dict):
            data = source
        else:
            path_obj = Path(source).resolve()
            if not path_obj.is_file():
                raise FileNotFoundError(f"Panel harmonization config not found: {source}")

            content = path_obj.read_text(encoding="utf-8")
            ext = path_obj.suffix.lower()

            if ext in {".yaml", ".yml"}:
                try:
                    import yaml

                    data = yaml.safe_load(content)
                except Exception as exc:
                    raise ValueError(f"Error parsing YAML file '{path_obj.name}': {exc}") from exc
            else:
                try:
                    data = json.loads(content)
                except Exception as exc:
                    raise ValueError(f"Error parsing JSON file '{path_obj.name}': {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Invalid harmonization configuration structure: expected dict, got {type(data)}")

        # Extract top-level settings and metadata
        name = str(data.get("name", "Custom Panel Harmonizer"))
        description = str(data.get("description", ""))

        settings = data.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}

        on_missing = str(settings.get("on_missing", data.get("on_missing", "warn")))
        drop_unmapped = bool(settings.get("drop_unmapped", data.get("drop_unmapped", False)))
        preserve_raw_names = bool(settings.get("preserve_raw_names", data.get("preserve_raw_names", True)))

        raw_channels = data.get("channels", data.get("canonical_channels", data.get("mappings", {})))
        if not raw_channels and not any(k in data for k in ("name", "settings", "channels")):
            # Fallback for flat dictionary: { "CD14": ["FL1 INT", ...], ... }
            raw_channels = {k: v for k, v in data.items() if not str(k).startswith("_")}

        channels: Dict[str, CanonicalChannelSpec] = {}

        for canonical_name, spec_data in raw_channels.items():
            c_name = str(canonical_name).strip()

            if isinstance(spec_data, list):
                # Flat list of aliases: CD14: ["FL1 INT", "CD14.FITC"]
                aliases = [str(a).strip() for a in spec_data if str(a).strip()]
                spec = CanonicalChannelSpec(
                    canonical_name=c_name,
                    aliases=aliases,
                    required=True,
                )
            elif isinstance(spec_data, dict):
                # Full structured spec
                aliases = [str(a).strip() for a in spec_data.get("aliases", []) if str(a).strip()]
                fluoro = spec_data.get("fluorophore")
                ctype = str(spec_data.get("channel_type", spec_data.get("type", "fluorescent")))
                req = bool(spec_data.get("required", False))
                desc = spec_data.get("description")
                spec = CanonicalChannelSpec(
                    canonical_name=c_name,
                    aliases=aliases,
                    fluorophore=fluoro,
                    channel_type=ctype,
                    required=req,
                    description=desc,
                )
            else:
                # Direct string alias
                spec = CanonicalChannelSpec(
                    canonical_name=c_name,
                    aliases=[str(spec_data).strip()],
                    required=True,
                )

            # Ensure canonical name itself is always included in search candidates
            if c_name not in spec.aliases:
                spec.aliases.append(c_name)

            channels[c_name] = spec

        return cls(
            channels=channels,
            name=name,
            description=description,
            on_missing=on_missing,
            drop_unmapped=drop_unmapped,
            preserve_raw_names=preserve_raw_names,
        )

    def harmonize_sample(
        self,
        fcs_data: Any,
        inplace: bool = False,
    ) -> HarmonizationResult:
        """Harmonize parameter channel headers for an individual FCSData sample.

        Args:
            fcs_data: Parsed FCSData object.
            inplace: If True, updates fcs_data.events_df in-place.

        Returns:
            HarmonizationResult object with mappings, DataFrame, and audit trail.

        Raises:
            ValueError: If on_missing='error' and a required canonical channel is not found.
        """
        sample_name = getattr(fcs_data, "filename", "sample.fcs")
        logger.debug(f"Harmonizing sample '{sample_name}' against panel '{self.name}'...")

        channel_mapping: Dict[str, str] = {}  # {canonical_name: original_detector}
        reverse_mapping: Dict[str, str] = {}  # {original_detector: canonical_name}
        assigned_detectors: Set[str] = set()
        missing_channels: List[str] = []
        warnings: List[str] = []

        available_detectors = list(getattr(fcs_data, "pnn_labels", []))
        if not available_detectors and hasattr(fcs_data, "events_df"):
            available_detectors = list(fcs_data.events_df.columns)

        # Iterate over canonical channel specifications
        for canonical_name, spec in self.channels.items():
            resolved_detector: Optional[str] = None

            # 1. Try fcs_data.resolve_channel_name() directly on canonical_name
            try:
                cand = fcs_data.resolve_channel_name(canonical_name)
                if cand in available_detectors and cand not in assigned_detectors:
                    resolved_detector = cand
            except Exception:
                pass

            # 2. Try each configured alias in order
            if resolved_detector is None:
                for alias in spec.aliases:
                    if alias in available_detectors and alias not in assigned_detectors:
                        resolved_detector = alias
                        break
                    try:
                        cand = fcs_data.resolve_channel_name(alias)
                        if cand in available_detectors and cand not in assigned_detectors:
                            resolved_detector = cand
                            break
                    except Exception:
                        continue

            # 3. Fuzzy punctuation and case-insensitive fallback on aliases
            if resolved_detector is None:
                for alias in spec.aliases:
                    alias_norm = re.sub(r"[^a-zA-Z0-9]", "", alias.lower())
                    for det in available_detectors:
                        if det in assigned_detectors:
                            continue
                        det_norm = re.sub(r"[^a-zA-Z0-9]", "", det.lower())
                        if det_norm == alias_norm:
                            resolved_detector = det
                            break
                    if resolved_detector is not None:
                        break

            # 4. Handle resolution outcome
            if resolved_detector is not None:
                channel_mapping[canonical_name] = resolved_detector
                reverse_mapping[resolved_detector] = canonical_name
                assigned_detectors.add(resolved_detector)
                # Keep record in sample channel resolutions
                if hasattr(fcs_data, "channel_resolutions"):
                    fcs_data.channel_resolutions[canonical_name] = resolved_detector
            else:
                missing_channels.append(canonical_name)
                msg = f"Canonical channel '{canonical_name}' could not be resolved in '{sample_name}'."
                if spec.required:
                    warnings.append(f"REQUIRED MISSING: {msg}")
                    logger.warning(f"[{sample_name}] {msg}")
                else:
                    logger.debug(f"[{sample_name}] Optional channel '{canonical_name}' not detected.")

        # Check required missing channels under policy
        success = True
        required_missing = [ch for ch in missing_channels if self.channels[ch].required]
        if required_missing:
            success = False
            err_msg = (
                f"Harmonization error for '{sample_name}': {len(required_missing)} required canonical "
                f"channels missing: {required_missing}"
            )
            if self.on_missing == "error":
                logger.error(err_msg)
                raise ValueError(err_msg)

        # Detect unmapped original channels
        unmapped_channels = [d for d in available_detectors if d not in assigned_detectors]

        # Construct harmonized DataFrame
        source_df = fcs_data.events_df if hasattr(fcs_data, "events_df") else pd.DataFrame()
        harmonized_df = source_df.copy()

        # Rename mapped columns
        harmonized_df.rename(columns=reverse_mapping, inplace=True)

        if self.drop_unmapped:
            cols_to_keep = [c for c in self.channels.keys() if c in harmonized_df.columns]
            harmonized_df = harmonized_df[cols_to_keep]

        # Save metadata record on fcs_data
        if self.preserve_raw_names and hasattr(fcs_data, "metadata"):
            fcs_data.metadata["harmonized_channels"] = channel_mapping
            fcs_data.metadata["harmonization_profile"] = self.name
            fcs_data.metadata["unmapped_channels"] = unmapped_channels

        if inplace and hasattr(fcs_data, "events_df"):
            fcs_data.events_df = harmonized_df

        return HarmonizationResult(
            sample_name=sample_name,
            harmonized_df=harmonized_df,
            channel_mapping=channel_mapping,
            reverse_mapping=reverse_mapping,
            missing_channels=missing_channels,
            unmapped_channels=unmapped_channels,
            warnings=warnings,
            success=success,
        )

    def harmonize_cohort(
        self,
        samples: List[Any],
        inplace: bool = False,
    ) -> CohortHarmonizationReport:
        """Harmonize a multi-sample cohort and generate a concordance matrix.

        Args:
            samples: List of parsed FCSData sample containers.
            inplace: If True, applies harmonization to sample events_df in place.

        Returns:
            CohortHarmonizationReport containing matrix, results, and concordance check.
        """
        canonical_list = list(self.channels.keys())
        sample_results: Dict[str, HarmonizationResult] = {}
        missing_summary: Dict[str, List[str]] = {ch: [] for ch in canonical_list}
        matrix_rows: List[Dict[str, str]] = []

        all_success = True

        for s in samples:
            res = self.harmonize_sample(s, inplace=inplace)
            sample_results[res.sample_name] = res
            if not res.success:
                all_success = False

            row: Dict[str, str] = {"Sample": res.sample_name}
            for ch in canonical_list:
                det = res.channel_mapping.get(ch)
                if det:
                    row[ch] = det
                else:
                    row[ch] = "MISSING"
                    missing_summary[ch].append(res.sample_name)
            matrix_rows.append(row)

        # Filter empty missing lists
        clean_missing = {k: v for k, v in missing_summary.items() if v}

        concordance_df = pd.DataFrame(matrix_rows).set_index("Sample")

        return CohortHarmonizationReport(
            total_samples=len(samples),
            canonical_channels=canonical_list,
            sample_results=sample_results,
            concordance_matrix=concordance_df,
            missing_summary=clean_missing,
            is_fully_concordant=all_success,
        )

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Serialize this harmonizer configuration to a YAML file."""
        out_p = Path(path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        try:
            import yaml
        except ImportError:
            raise RuntimeError("PyYAML is required to serialize panel harmonizer to YAML.")

        channels_dict = {}
        for c_name, spec in self.channels.items():
            entry: Dict[str, Any] = {
                "aliases": spec.aliases,
                "type": spec.channel_type,
                "required": spec.required,
            }
            if spec.fluorophore:
                entry["fluorophore"] = spec.fluorophore
            if spec.description:
                entry["description"] = spec.description
            channels_dict[c_name] = entry

        doc = {
            "name": self.name,
            "description": self.description,
            "settings": {
                "on_missing": self.on_missing,
                "drop_unmapped": self.drop_unmapped,
                "preserve_raw_names": self.preserve_raw_names,
            },
            "channels": channels_dict,
        }

        with open(out_p, "w", encoding="utf-8") as f:
            yaml.dump(doc, f, sort_keys=False)

        return out_p


def load_panel_harmonizer(
    source: Optional[Union[str, Path, Dict[str, Any]]] = None,
) -> PanelHarmonizer:
    """Convenience helper to load a PanelHarmonizer.

    If source is None, attempts to find 'configs/panel_harmonization.yaml' in the
    workspace; if not found, returns a standard multi-color leukemia/PBMC harmonizer.

    Args:
        source: Optional path or dict.

    Returns:
        Loaded PanelHarmonizer instance.
    """
    if source is not None:
        return PanelHarmonizer.load(source)

    default_path = Path("configs/panel_harmonization.yaml")
    if default_path.is_file():
        return PanelHarmonizer.load(default_path)

    # Return default built-in consensus harmonizer
    builtin_specs = {
        "FSC-A": CanonicalChannelSpec("FSC-A", ["FS INT", "FS.INT", "FSC-A", "FSC_A"], channel_type="scatter", required=True),
        "SSC-A": CanonicalChannelSpec("SSC-A", ["SS INT", "SS.INT", "SSC-A", "SSC_A"], channel_type="scatter", required=True),
        "CD14": CanonicalChannelSpec("CD14", ["FL1 INT", "CD14.FITC", "CD14-FITC", "FITC-A"], fluorophore="FITC", required=True),
        "CD19": CanonicalChannelSpec("CD19", ["FL2 INT", "CD19.PE", "CD19-PE", "PE-A"], fluorophore="PE", required=True),
        "CD13": CanonicalChannelSpec("CD13", ["FL3 INT", "CD13.ECD", "CD13-ECD", "ECD-A"], fluorophore="ECD / PE-TxRed", required=True),
        "CD33": CanonicalChannelSpec("CD33", ["FL4 INT", "CD33.PC5.5", "CD33-PC5.5", "PC5.5-A"], fluorophore="PerCP-Cy5.5 / PC5.5", required=True),
        "CD34": CanonicalChannelSpec("CD34", ["FL5 INT", "CD34.PC7", "CD34-PC7", "PC7-A"], fluorophore="PE-Cy7 / PC7", required=True),
        "CD117": CanonicalChannelSpec("CD117", ["FL6 INT", "CD117.APC", "CD117-APC", "APC-A"], fluorophore="APC", required=True),
        "CD7": CanonicalChannelSpec("CD7", ["FL7 INT", "CD7.APC700", "CD7-APC700", "APC-A700-A"], fluorophore="APC-Alexa700", required=True),
        "CD16": CanonicalChannelSpec("CD16", ["FL8 INT", "CD16.APC750", "CD16-APC750", "APC-A750-A"], fluorophore="APC-Alexa750", required=True),
        "HLA-DR": CanonicalChannelSpec("HLA-DR", ["FL9 INT", "HLA.PB", "HLA-PB", "PacBlue-A"], fluorophore="Pacific Blue", required=True),
        "CD45": CanonicalChannelSpec("CD45", ["FL10 INT", "CD45.KO", "CD45-KO", "KO-A"], fluorophore="Krome Orange", required=True),
    }
    return PanelHarmonizer(
        channels=builtin_specs,
        name="Built-in 10-Color Consensus Harmonizer",
        description="Default consensus harmonizer for 10-color bone marrow/leukemia assays",
    )
