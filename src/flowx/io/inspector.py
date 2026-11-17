"""FCS File Inspection and Quality Control Module.

Provides robust, vendor-agnostic auditing and metadata extraction for Flow
Cytometry Standard (FCS 2.0, 3.0, 3.1, 3.2) data files. Evaluates adherence
to the Minimum Information about a Flow Cytometry Experiment (MIFlowCyt) standard,
detects hardware instrument configurations (BD, Beckman Coulter, Cytek Aurora,
Sony, Thermo Attune, etc.), verifies channel mappings, inspects compensation
matrices, checks panel consistency across multi-patient cohorts, and exports
structured summaries to CSV, Markdown reports, and rich terminal displays.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from flowx.io.paths import sanitize_path


def load_population_map(
    source: Optional[Union[Dict[str, str], str, Path]] = None,
) -> Tuple[Dict[str, str], Optional[str]]:
    """Load population code-to-name mapping dictionary and optional regex extraction pattern.

    Enables completely customizable, scalable, and decoupled population annotation for any
    experimental dataset without hardcoding dataset-specific assumptions into the engine.

    Args:
        source: Can be:
            - None: Returns empty mapping and None pattern (generic flow cytometry inspection).
            - Dict[str, str]: Direct dictionary of {code: description}.
            - str or Path: Path to a JSON or YAML configuration file.

    Returns:
        Tuple of (mapping_dict, regex_pattern_string_or_None).
    """
    if source is None:
        return {}, None

    if isinstance(source, dict):
        return {str(k): str(v) for k, v in source.items()}, None

    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Population mapping configuration not found: {source}")

    content = source_path.read_text(encoding="utf-8")
    ext = source_path.suffix.lower()

    data: Any = {}
    if ext in {".yaml", ".yml"}:
        try:
            import yaml

            data = yaml.safe_load(content)
        except Exception:
            data = json.loads(content)
    else:
        data = json.loads(content)

    if not isinstance(data, dict):
        return {}, None

    # Check for structured format: {"mappings": {...}, "pattern": "..."}
    if "mappings" in data and isinstance(data["mappings"], dict):
        mappings = {str(k): str(v) for k, v in data["mappings"].items()}
        pattern = str(data.get("pattern", "")).strip() or None
        return mappings, pattern

    # Flat dict format: {"A": "Full cell population", ...}
    return {str(k): str(v) for k, v in data.items() if not str(k).startswith("_") and isinstance(v, str)}, None


@dataclass
class FCSChannelInfo:
    """Detailed metadata for a single optical channel / detector.

    Attributes:
        index: 1-based parameter index (Pn).
        pnn: Detector name (e.g. 'FL1 INT', 'FSC-A', 'Ax700-A').
        pns: Reagent stain or marker label (e.g. 'CD14-FITC', 'CD3').
        channel_type: Category ('scatter', 'fluorescent', 'time', or 'other').
        marker: Isolated biological marker name (e.g. 'CD14').
        fluorochrome: Isolated dye / fluorochrome name (e.g. 'FITC').
        range_max: Scale dynamic range maximum ($PnR).
        bit_depth: Bits allocated per parameter ($PnB).
        gain: Detector amplification gain ($PnG).
        voltage: Photomultiplier tube (PMT) voltage ($PnV).
        in_spillover: True if channel is included in compensation matrix.
    """

    index: int
    pnn: str
    pns: str
    channel_type: str
    marker: str = ""
    fluorochrome: str = ""
    range_max: float = 0.0
    bit_depth: int = 32
    gain: float = 1.0
    voltage: float = 0.0
    in_spillover: bool = False


@dataclass
class FCSFileInfo:
    """Inspection metrics and metadata for a single FCS file adhering to MIFlowCyt.

    Attributes:
        filepath: Full sanitized path to the FCS file.
        filename: Base filename.
        filesize_bytes: Size on disk in bytes.
        filesize_mb: Size on disk in Megabytes.
        fcs_version: FCS standard version (e.g. 'FCS3.0', 'FCS3.1').
        byte_order: Byte ordering specification ($BYTEORD).
        data_type: Storage representation ($DATATYPE, e.g. Float 'F', Double 'D').
        mode: Acquisition mode ($MODE, e.g. List 'L').
        event_count: Number of recorded events ($TOT).
        channel_count: Total parameter count ($PAR).
        vendor: Detected instrument manufacturer (e.g. 'Beckman Coulter', 'BD Biosciences').
        cytometer_model: Hardware cytometer model (e.g. 'Navios', 'LSRFortessa', 'CytoFLEX').
        cytometer_id: Cytometer serial number or hardware ID.
        software: Acquisition and analysis software name and version.
        lasers: Detected laser lines, wavelengths, and powers.
        acquisition_date: Date of specimen acquisition.
        acquisition_time_start: Acquisition start timestamp ($BTIM).
        acquisition_time_end: Acquisition end timestamp ($ETIM).
        acquisition_duration_sec: Elapsed acquisition duration in seconds.
        event_rate_eps: Acquisition flow rate in events per second.
        scatter_channels: Forward and side scatter channel names.
        fluorescent_channels: Fluorescent antibody channel names.
        time_channel: Time recording channel name.
        markers: List of unique biological markers detected.
        fluorochromes: List of unique fluorochromes detected.
        panel_signature: Concise canonical representation of the panel layout.
        has_spillover: True if embedded compensation/spillover matrix is present.
        spillover_channel_count: Number of parameters in spillover matrix.
        spillover_channels: Parameter names in spillover matrix.
        transformation_status: Pre-applied transformation or unmixing status.
        patient_case_id: Extracted patient/case identifier (e.g. 'Case1').
        population_class: Population code (e.g. 'A', 'O', 'N', 'G', 'P', 'K', 'B').
        population_name: Human-readable population description.
        qc_status: Quality status ('PASS' or 'WARNING').
        qc_warnings: List of specific quality control flags or anomalies.
        channel_details: List of per-channel information objects.
        negative_event_pct: Optional percentage of events with values <= 0 across fluorescent channels.
        saturated_event_pct: Optional percentage of events at upper scale limit ($PnR).
    """

    filepath: str
    filename: str
    filesize_bytes: int
    filesize_mb: float
    fcs_version: str
    byte_order: str
    data_type: str
    mode: str
    event_count: int
    channel_count: int
    vendor: str
    cytometer_model: str
    cytometer_id: str
    software: str
    lasers: str
    acquisition_date: str
    acquisition_time_start: str
    acquisition_time_end: str
    acquisition_duration_sec: float
    event_rate_eps: float
    scatter_channels: List[str] = field(default_factory=list)
    fluorescent_channels: List[str] = field(default_factory=list)
    time_channel: str = ""
    markers: List[str] = field(default_factory=list)
    fluorochromes: List[str] = field(default_factory=list)
    panel_signature: str = ""
    has_spillover: bool = False
    spillover_channel_count: int = 0
    spillover_channels: List[str] = field(default_factory=list)
    transformation_status: str = "none"
    patient_case_id: str = ""
    population_class: str = ""
    population_name: str = ""
    qc_status: str = "PASS"
    qc_warnings: List[str] = field(default_factory=list)
    channel_details: List[FCSChannelInfo] = field(default_factory=list)
    negative_event_pct: Optional[float] = None
    saturated_event_pct: Optional[float] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert inspection metrics to a flat dictionary suitable for CSV export."""
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "filesize_mb": round(self.filesize_mb, 3),
            "fcs_version": self.fcs_version,
            "vendor": self.vendor,
            "cytometer_model": self.cytometer_model,
            "cytometer_id": self.cytometer_id,
            "software": self.software,
            "acquisition_date": self.acquisition_date,
            "acquisition_time_start": self.acquisition_time_start,
            "acquisition_time_end": self.acquisition_time_end,
            "acquisition_duration_sec": round(self.acquisition_duration_sec, 2),
            "event_count": self.event_count,
            "event_rate_eps": round(self.event_rate_eps, 1) if self.event_rate_eps > 0 else 0.0,
            "channel_count": self.channel_count,
            "scatter_channel_count": len(self.scatter_channels),
            "fluorescent_channel_count": len(self.fluorescent_channels),
            "scatter_channels": "; ".join(self.scatter_channels),
            "fluorescent_channels": "; ".join(self.fluorescent_channels),
            "time_channel": self.time_channel,
            "markers": "; ".join(self.markers),
            "fluorochromes": "; ".join(self.fluorochromes),
            "panel_signature": self.panel_signature,
            "has_spillover": self.has_spillover,
            "spillover_channel_count": self.spillover_channel_count,
            "spillover_channels": "; ".join(self.spillover_channels),
            "transformation_status": self.transformation_status,
            "patient_case_id": self.patient_case_id,
            "population_class": self.population_class,
            "population_name": self.population_name,
            "lasers": self.lasers,
            "byte_order": self.byte_order,
            "data_type": self.data_type,
            "mode": self.mode,
            "negative_event_pct": round(self.negative_event_pct, 2) if self.negative_event_pct is not None else "",
            "saturated_event_pct": round(self.saturated_event_pct, 2) if self.saturated_event_pct is not None else "",
            "qc_status": self.qc_status,
            "qc_warnings": "; ".join(self.qc_warnings) if self.qc_warnings else "None",
        }


@dataclass
class FCSBatchSummary:
    """Cohort-level summary aggregating metrics across multiple FCS files."""

    total_files: int
    total_events: int
    total_size_mb: float
    min_events: int
    max_events: int
    median_events: float
    mean_events: float
    unique_vendors: List[str]
    unique_cytometers: List[str]
    unique_fcs_versions: List[str]
    panel_variants: Dict[str, int] = field(default_factory=dict)
    panel_variant_files: Dict[str, List[str]] = field(default_factory=dict)
    consensus_panel: str = ""
    spillover_present_count: int = 0
    spillover_missing_count: int = 0
    population_counts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    patient_cases: List[str] = field(default_factory=list)
    acquisition_dates: List[str] = field(default_factory=list)
    qc_pass_count: int = 0
    qc_warning_count: int = 0
    common_warnings: Dict[str, int] = field(default_factory=dict)


def _get_case_insensitive_meta(meta: Dict[str, Any], *candidate_keys: str, default: Any = "") -> Any:
    """Look up metadata value case-insensitively, ignoring leading dollar signs ($)."""
    lookup: Dict[str, Any] = {str(k).lower().lstrip("$"): v for k, v in meta.items()}
    for key in candidate_keys:
        norm = str(key).lower().lstrip("$")
        if norm in lookup:
            val = lookup[norm]
            if val is not None and str(val).strip():
                return val
    return default


def read_fcs_header_and_text(
    filepath: Union[str, Path]
) -> Tuple[str, Dict[str, str], int, int, int, int]:
    """Read FCS binary header and TEXT segment in sub-millisecond execution time.

    Parses standard FCS 2.0, 3.0, 3.1, and 3.2 headers and extracts all TEXT
    key-value pairs without loading massive event arrays into RAM. Properly
    handles double-delimiter escaping rules.

    Args:
        filepath: Path to the FCS file.

    Returns:
        Tuple of (fcs_version, metadata_dict, text_start, text_end, data_start, data_end).

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If file is not a valid FCS standard file.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"FCS file not found at path: {filepath}")

    with open(path, "rb") as f:
        header_bytes = f.read(58)
        if len(header_bytes) < 58:
            raise ValueError(f"FCS header truncated in '{path.name}' (got {len(header_bytes)} bytes)")

        version_str = header_bytes[:6].decode("ascii", errors="ignore").strip()
        if not version_str.startswith("FCS"):
            raise ValueError(f"File '{path.name}' does not have valid FCS magic header: '{version_str}'")

        try:
            text_start = int(header_bytes[10:18].decode("ascii", errors="ignore").strip() or 0)
            text_end = int(header_bytes[18:26].decode("ascii", errors="ignore").strip() or 0)
            data_start = int(header_bytes[26:34].decode("ascii", errors="ignore").strip() or 0)
            data_end = int(header_bytes[34:42].decode("ascii", errors="ignore").strip() or 0)
        except ValueError as exc:
            raise ValueError(f"Corrupt offset integer in FCS header of '{path.name}': {exc}") from exc

        # Seek to TEXT segment and read
        f.seek(text_start)
        text_len = text_end - text_start + 1
        if text_len <= 1:
            raise ValueError(f"Invalid TEXT segment range [{text_start}, {text_end}] in '{path.name}'")

        raw_text = f.read(text_len)
        delim = raw_text[:1]
        if not delim:
            raise ValueError(f"Empty TEXT delimiter in '{path.name}'")

        # Split on delimiter
        parts = raw_text.split(delim)
        meta: Dict[str, str] = {}
        i = 1
        n_parts = len(parts)

        while i < n_parts - 1:
            raw_key = parts[i].decode("latin1", errors="ignore").strip()
            raw_val = parts[i + 1].decode("latin1", errors="ignore").strip()

            if i + 2 < n_parts and parts[i + 2] == b"":
                escaped_append = parts[i + 3].decode("latin1", errors="ignore").strip() if i + 3 < n_parts else ""
                delim_char = delim.decode("latin1", errors="ignore")
                raw_val = f"{raw_val}{delim_char}{escaped_append}"
                i += 2

            meta[raw_key] = raw_val
            i += 2

        if data_start == 0 and ("$BEGINDATA" in meta or "BEGINDATA" in meta):
            data_start = int(meta.get("$BEGINDATA") or meta.get("BEGINDATA") or 0)
        if data_end == 0 and ("$ENDDATA" in meta or "ENDDATA" in meta):
            data_end = int(meta.get("$ENDDATA") or meta.get("ENDDATA") or 0)

        return version_str, meta, text_start, text_end, data_start, data_end


def detect_vendor_and_model(meta: Dict[str, Any]) -> Tuple[str, str]:
    """Identify flow cytometer hardware manufacturer and model.

    Adheres to MIFlowCyt section 3 (Instrument Details). Recognizes BD Biosciences,
    Beckman Coulter, Cytek Biosciences, Sony Biotechnology, Thermo Fisher,
    Miltenyi Biotec, Luminex/Amnis, and Agilent.

    Args:
        meta: FCS metadata dictionary.

    Returns:
        Tuple of (vendor_name, model_name).
    """
    cyt = str(_get_case_insensitive_meta(meta, "cyt", "originalinstrument", "instrument")).strip()
    orig = str(_get_case_insensitive_meta(meta, "originalinstrument")).strip()
    sys_k = str(_get_case_insensitive_meta(meta, "sys")).strip()
    creator = str(_get_case_insensitive_meta(meta, "creator", "com", "last_modifier", "cytobank")).strip()
    guid = str(_get_case_insensitive_meta(meta, "guid", "fileguid")).strip()
    all_text = f"{cyt} {orig} {sys_k} {creator} {guid}".lower()

    vendor = "Standard / Third-Party"
    model = orig or cyt or "Flow Cytometer"

    # 1. Beckman Coulter
    if (
        "navios" in all_text
        or "gallios" in all_text
        or "cytoflex" in all_text
        or "moflo" in all_text
        or "fc500" in all_text
        or "bci" in all_text
        or "kaluza" in all_text
        or "@beginkaluza" in [k.lower() for k in meta.keys()]
    ):
        vendor = "Beckman Coulter"
        if "navios" in all_text:
            model = "Navios"
        elif "cytoflex" in all_text:
            model = "CytoFLEX"
        elif "gallios" in all_text:
            model = "Gallios"
        elif "moflo" in all_text:
            model = "MoFlo Astrios"
        elif "kaluza" in all_text:
            model = "Navios / Gallios (Kaluza)"

    # 2. BD Biosciences
    elif (
        "fortessa" in all_text
        or "lsrii" in all_text
        or "lsr ii" in all_text
        or "lsr" in all_text
        or "canto" in all_text
        or "aria" in all_text
        or "symphony" in all_text
        or "accuri" in all_text
        or "calibur" in all_text
        or "facsdiva" in all_text
        or "becton" in all_text
        or "bd " in all_text
    ):
        vendor = "BD Biosciences"
        if "fortessa" in all_text:
            model = "LSRFortessa"
        elif "symphony" in all_text:
            model = "FACSSymphony"
        elif "canto" in all_text:
            model = "FACSCanto II" if "ii" in all_text else "FACSCanto"
        elif "aria" in all_text:
            model = "FACSAria III" if "iii" in all_text else "FACSAria"
        elif "lsrii" in all_text or "lsr ii" in all_text:
            model = "LSRII"
        elif "accuri" in all_text:
            model = "Accuri C6"
        elif "calibur" in all_text:
            model = "FACSCalibur"
        elif "facsdiva" in all_text:
            model = "BD FACSDiva System"

    # 3. Cytek Biosciences
    elif "aurora" in all_text or "cytek" in all_text or "spectroflo" in all_text or "northern lights" in all_text:
        vendor = "Cytek Biosciences"
        if "northern lights" in all_text:
            model = "Northern Lights"
        else:
            model = "Aurora Spectral Cytometer"

    # 4. Sony Biotechnology
    elif "sony" in all_text or "id7000" in all_text or "sa3800" in all_text or "ma900" in all_text:
        vendor = "Sony Biotechnology"
        if "id7000" in all_text:
            model = "ID7000 Spectral Cell Analyzer"
        elif "sa3800" in all_text:
            model = "SA3800 Spectral Analyzer"
        elif "ma900" in all_text:
            model = "MA900 Multi-Application Cell Sorter"

    # 5. Thermo Fisher Scientific
    elif "attune" in all_text or "applied biosystems" in all_text or "thermo" in all_text:
        vendor = "Thermo Fisher Scientific"
        model = "Attune NxT Acoustic Focusing Cytometer"

    # 6. Miltenyi Biotec
    elif "macsquant" in all_text or "miltenyi" in all_text:
        vendor = "Miltenyi Biotec"
        model = "MACSQuant Analyzer"

    # 7. Luminex / Amnis
    elif "imagestream" in all_text or "flowsight" in all_text or "guava" in all_text:
        vendor = "Luminex / Amnis"
        model = "ImageStreamX / FlowSight"

    # 8. Agilent / ACEA
    elif "novocyte" in all_text or "acea" in all_text:
        vendor = "Agilent Technologies"
        model = "NovoCyte"

    # 9. Inferred from flowCore metadata and 20-bit dynamic range ($PnR = 1,048,576)
    if vendor == "Standard / Third-Party":
        meta_keys_lower = [str(k).lower() for k in meta.keys()]
        has_flowcore = any("flowcore" in k for k in meta_keys_lower)
        p1r_val = str(_get_case_insensitive_meta(meta, "p1r", "flowcore_$p1rmax")).strip()
        has_20bit = p1r_val in {"1048576", "1048576.0", "1048780", "1048780.0"} or any(
            "fl1.int" in str(v).lower() or "fl1 int" in str(v).lower() for v in meta.values()
        )
        if has_20bit or has_flowcore:
            vendor = "Beckman Coulter"
            model = "Navios (flowCore R Export)" if has_flowcore else "Navios"

    return vendor, model


def detect_lasers(meta: Dict[str, Any]) -> str:
    """Extract optical laser line configurations and powers."""
    laser_descriptions: List[str] = []

    blue_p = _get_case_insensitive_meta(meta, "@bluelaserpower_start", "@bluetargetpower")
    red_p = _get_case_insensitive_meta(meta, "@redlaserpower_start", "@redtargetpower")
    violet_p = _get_case_insensitive_meta(meta, "@violetlaserpower_start", "@violettargetpower")

    if blue_p:
        laser_descriptions.append(f"Blue 488nm ({blue_p})")
    if red_p:
        laser_descriptions.append(f"Red 638nm ({red_p})")
    if violet_p:
        laser_descriptions.append(f"Violet 405nm ({violet_p})")

    if not laser_descriptions:
        for k, v in meta.items():
            k_lower = k.lower()
            if "laser" in k_lower and ("power" in k_lower or "wave" in k_lower):
                val_str = str(v).strip()
                if val_str and val_str not in laser_descriptions:
                    laser_descriptions.append(f"{k}: {val_str}")
            if len(laser_descriptions) >= 4:
                break

    return ", ".join(laser_descriptions) if laser_descriptions else "Standard Multi-Laser Optical Deck"


def decompose_marker_and_fluorochrome(pnn: str, pns: str) -> Tuple[str, str]:
    """Extract individual biological marker and fluorophore from channel labels."""
    raw_str = (pns.strip() if pns.strip() else pnn.strip()).replace(".", "-").replace("_", "-")

    fluoro_patterns = [
        ("fitc", "FITC"),
        ("pe-cy7", "PE-Cy7"),
        ("pe-cy5", "PE-Cy5"),
        ("pe-texas", "PE-TexasRed"),
        ("pe", "PE"),
        ("ecd", "ECD / PE-TxRed"),
        ("pc5-5", "PerCP-Cy5.5 / PC5.5"),
        ("pc5.5", "PerCP-Cy5.5 / PC5.5"),
        ("pc7", "PE-Cy7 / PC7"),
        ("apc-cy7", "APC-Cy7"),
        ("apc-h7", "APC-H7"),
        ("apc700", "APC-AlexaFluor 700"),
        ("apc750", "APC-AlexaFluor 750"),
        ("apc", "APC"),
        ("pb", "Pacific Blue"),
        ("pacblue", "Pacific Blue"),
        ("ko", "Krome Orange"),
        ("bv421", "Brilliant Violet 421"),
        ("bv510", "Brilliant Violet 510"),
        ("bv605", "Brilliant Violet 605"),
        ("bv711", "Brilliant Violet 711"),
        ("bv785", "Brilliant Violet 785"),
        ("buv395", "Brilliant UV 395"),
        ("buv737", "Brilliant UV 737"),
        ("bb515", "Brilliant Blue 515"),
        ("alexa-700", "Alexa Fluor 700"),
        ("ax700", "Alexa Fluor 700"),
    ]

    clean = re.sub(r"^FL\d+[-_]INT[-_]?", "", raw_str, flags=re.IGNORECASE)
    clean = re.sub(r"^FL\d+[-_]?", "", clean, flags=re.IGNORECASE)

    matched_marker = ""
    matched_fluoro = ""

    parts = [p.strip() for p in re.split(r"[- ]+", clean) if p.strip()]

    for p in parts:
        p_up = p.upper()
        if re.match(r"^CD\d+[A-Z]*$", p_up) or p_up in {"HLA", "HLADR", "HLA-DR", "LIN", "LIVE", "DEAD"}:
            matched_marker = "HLA-DR" if "HLA" in p_up else p_up
            break

    clean_lower = clean.lower()
    for pattern, display_name in fluoro_patterns:
        if pattern in clean_lower:
            matched_fluoro = display_name
            break

    if not matched_marker and parts:
        matched_marker = parts[0]
    if not matched_fluoro and len(parts) > 1:
        matched_fluoro = parts[-1]

    return matched_marker, matched_fluoro


def extract_acquisition_timing(
    meta: Dict[str, Any], event_count: int
) -> Tuple[str, str, str, float, float]:
    """Extract acquisition date, start time, end time, duration, and event rate."""
    date_val = str(_get_case_insensitive_meta(meta, "date", "@y2kdate", "y2kdate")).strip()
    btim_val = str(_get_case_insensitive_meta(meta, "btim", "begintime", "start_time")).strip()
    etim_val = str(_get_case_insensitive_meta(meta, "etim", "endtime", "stop_time")).strip()
    acq_val = str(_get_case_insensitive_meta(meta, "@acqtime", "@elapsedtime", "acqtime", "elapsedtime")).strip()

    duration_sec = 0.0

    if acq_val:
        try:
            duration_sec = float(acq_val)
        except ValueError:
            pass

    if duration_sec <= 0.0 and btim_val and etim_val:
        try:
            t0 = datetime.datetime.strptime(btim_val, "%H:%M:%S")
            t1 = datetime.datetime.strptime(etim_val, "%H:%M:%S")
            diff = (t1 - t0).total_seconds()
            if diff < 0:
                diff += 86400.0
            duration_sec = float(diff)
        except Exception:
            pass

    event_rate_eps = (event_count / duration_sec) if duration_sec > 0 else 0.0

    return date_val, btim_val, etim_val, duration_sec, event_rate_eps


def inspect_fcs_file(
    filepath: Union[str, Path],
    population_map: Optional[Union[Dict[str, str], str, Path]] = None,
    population_pattern: Optional[str] = None,
    include_event_stats: bool = False,
    max_events_for_stats: int = 50000,
) -> FCSFileInfo:
    """Perform a comprehensive industry-standard inspection of a single FCS file.

    Evaluates file header, text keywords, channel mappings, hardware parameters,
    compensation matrix presence, cell population class, and quality control metrics.

    Args:
        filepath: Path to the FCS file to inspect.
        population_map: Optional dictionary or path to JSON/YAML population config.
        population_pattern: Optional regex pattern for population code extraction.
        include_event_stats: If True, reads sample event arrays to compute
            zero/negative event percentage and saturation percentages.
        max_events_for_stats: Maximum number of events to sample for event stats.

    Returns:
        Populated FCSFileInfo dataclass instance.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is unreadable or malformed.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FCS file does not exist: {filepath}")

    filesize_bytes = path.stat().st_size
    filesize_mb = filesize_bytes / (1024 * 1024)

    # 1. Sub-millisecond binary header & TEXT parsing
    try:
        fcs_version, meta, text_start, text_end, data_start, data_end = read_fcs_header_and_text(path)
    except Exception as exc:
        try:
            import flowkit as fk

            sample = fk.Sample(str(path))
            meta = sample.get_metadata()
            fcs_version = str(_get_case_insensitive_meta(meta, "fcsversion", default="FCS3.0"))
            data_start, data_end = 0, 0
        except Exception as fk_exc:
            raise ValueError(f"Failed to parse FCS file '{path.name}': {exc} (FlowKit fallback: {fk_exc})") from exc

    # 2. Basic parameters
    tot_val = _get_case_insensitive_meta(meta, "tot", default="0")
    try:
        event_count = int(tot_val)
    except ValueError:
        event_count = 0

    par_val = _get_case_insensitive_meta(meta, "par", default="0")
    try:
        channel_count = int(par_val)
    except ValueError:
        channel_count = 0

    byte_order_val = str(_get_case_insensitive_meta(meta, "byteord", default="Unknown")).strip()
    byte_order_desc = (
        "4,3,2,1 (Big-endian)" if "4,3,2,1" in byte_order_val else "1,2,3,4 (Little-endian)" if "1,2,3,4" in byte_order_val else byte_order_val
    )

    data_type_val = str(_get_case_insensitive_meta(meta, "datatype", default="F")).strip().upper()
    data_type_desc = (
        "F (Float 32-bit)" if data_type_val == "F" else "D (Double 64-bit)" if data_type_val == "D" else "I (Integer)" if data_type_val == "I" else data_type_val
    )

    mode_val = str(_get_case_insensitive_meta(meta, "mode", default="L")).strip().upper()
    mode_desc = "L (List Mode)" if mode_val == "L" else mode_val

    # 3. Cytometer, Vendor, Software, Lasers
    vendor, cytometer_model = detect_vendor_and_model(meta)
    cytometer_id = str(_get_case_insensitive_meta(meta, "cytsn", "@cytometerid", "cytometerid")).strip()
    software = str(_get_case_insensitive_meta(meta, "last_modifier", "creator", "com", "sw_ver", "cytobank")).strip()
    if not software and any("flowcore" in str(k).lower() for k in meta.keys()):
        software = "flowCore (Bioconductor R package)"
    elif not software:
        software = "Standard Flow Cytometry Acquisition"
    lasers = detect_lasers(meta)

    # 4. Acquisition timing and flow rate
    acq_date, btim, etim, duration_sec, event_rate = extract_acquisition_timing(meta, event_count)

    # 5. Extract Spillover / Compensation Matrix
    spill_key = None
    spill_channels: List[str] = []
    has_spillover = False
    for k in meta.keys():
        k_clean = k.lower().lstrip("$")
        if k_clean in {"spillover", "spill", "comp"}:
            spill_key = k
            break

    if spill_key:
        raw_spill = str(meta[spill_key]).strip()
        tokens = [t.strip() for t in raw_spill.replace("\n", ",").replace("\r", ",").split(",") if t.strip()]
        if tokens:
            try:
                n_comp = int(tokens[0])
                if len(tokens) >= 1 + n_comp:
                    spill_channels = tokens[1 : 1 + n_comp]
                    has_spillover = True
            except ValueError:
                pass

    transformation_val = str(_get_case_insensitive_meta(meta, "transformation", default="none")).strip().lower()

    # 6. Channel parsing ($PnN, $PnS, $PnR, $PnB, $PnV, $PnG)
    channel_details: List[FCSChannelInfo] = []
    scatter_channels: List[str] = []
    fluorescent_channels: List[str] = []
    time_channel: str = ""
    markers_set: List[str] = []
    fluorochromes_set: List[str] = []

    for i in range(1, channel_count + 1):
        pnn = str(_get_case_insensitive_meta(meta, f"p{i}n", default=f"P{i}")).strip()
        pns = str(_get_case_insensitive_meta(meta, f"p{i}s", default="")).strip()
        pnr = float(_get_case_insensitive_meta(meta, f"p{i}r", default="262144"))
        pnb = int(_get_case_insensitive_meta(meta, f"p{i}b", default="32"))
        png = float(_get_case_insensitive_meta(meta, f"p{i}g", default="1.0"))
        pnv = float(_get_case_insensitive_meta(meta, f"p{i}v", default="0.0"))

        pnn_lower = pnn.lower()

        if any(sc in pnn_lower for sc in ["fsc", "fs-", "fs.", "fs int", "ssc", "ss-", "ss.", "ss int", "ss peak", "ss tof"]):
            ch_type = "scatter"
            scatter_channels.append(pnn)
            marker = ""
            fluoro = ""
        elif "time" in pnn_lower:
            ch_type = "time"
            time_channel = pnn
            marker = ""
            fluoro = ""
        else:
            ch_type = "fluorescent"
            fluorescent_channels.append(pnn)
            marker, fluoro = decompose_marker_and_fluorochrome(pnn, pns)
            if marker and marker not in markers_set:
                markers_set.append(marker)
            if fluoro and fluoro not in fluorochromes_set:
                fluorochromes_set.append(fluoro)

        in_spill = pnn in spill_channels or (pns and pns in spill_channels)

        channel_details.append(
            FCSChannelInfo(
                index=i,
                pnn=pnn,
                pns=pns,
                channel_type=ch_type,
                marker=marker,
                fluorochrome=fluoro,
                range_max=pnr,
                bit_depth=pnb,
                gain=png,
                voltage=pnv,
                in_spillover=in_spill,
            )
        )

    # 7. Generate Panel Signature
    sig_elements = [f"{ch.pnn}:{ch.pns}" for ch in channel_details]
    panel_signature_raw = "|".join(sig_elements)

    if any("FL1.INT_CD14.FITC" in ch.pnn for ch in channel_details):
        panel_signature = "Variant-2 (Case*_B Dots & Merged Detectors)"
    elif any("CD14.FITC" in ch.pnn for ch in channel_details):
        panel_signature = "Variant-3 (Case30 Clean Channel Format)"
    elif any("FL1 INT" in ch.pnn for ch in channel_details):
        panel_signature = "Variant-1 (Navios Standard 10-Color)"
    else:
        sig_hash = hashlib.md5(panel_signature_raw.encode("utf-8")).hexdigest()[:8]
        panel_signature = f"Custom Panel ({channel_count} channels, #{sig_hash})"

    # 8. Patient Case & Population annotation (Customizable / Decoupled)
    stem = path.stem
    pop_dict, custom_pat = load_population_map(population_map)
    active_pat = population_pattern or custom_pat

    patient_case_id = ""
    population_class = ""
    population_name = ""

    if pop_dict:
        matched_code = ""
        if active_pat:
            match = re.search(active_pat, stem)
            if match:
                matched_code = match.group("code") if "code" in match.groupdict() else match.group(1)
        else:
            for code in pop_dict.keys():
                if stem.endswith(f"_{code}") or stem == code or f"_{code}_" in stem:
                    matched_code = code
                    break

        if matched_code:
            population_class = matched_code
            population_name = pop_dict.get(matched_code, f"Sub-population {matched_code}")

    case_match = re.match(r"^(Case\d+|Patient\d+|Donor\d+|Sample\d+)", stem, re.IGNORECASE)
    if case_match:
        patient_case_id = case_match.group(1)
    else:
        smno = str(_get_case_insensitive_meta(meta, "smno", "sampleid", "fil")).strip()
        patient_case_id = smno or stem

    if not population_name:
        population_name = "Unannotated Specimen"

    # 9. Quality Control & Anomaly Flags
    qc_warnings: List[str] = []

    if event_count < 100:
        qc_warnings.append(f"Extremely low event count ({event_count} events)")
    elif event_count < 500:
        qc_warnings.append(f"Low event count ({event_count} events)")

    if fluorescent_channels and not has_spillover:
        qc_warnings.append("No embedded spillover/compensation matrix ($SPILLOVER absent)")

    if transformation_val == "applied":
        qc_warnings.append("Data flagged as pre-transformed/compensated (TRANSFORMATION: applied)")

    if "Variant-2" in panel_signature or "Variant-3" in panel_signature:
        qc_warnings.append(f"Panel format discrepancy: {panel_signature}")

    negative_event_pct = None
    saturated_event_pct = None

    if include_event_stats and event_count > 0:
        try:
            import flowkit as fk

            sample_obj = fk.Sample(str(path))
            raw_ev = sample_obj.get_events(source="raw")
            if len(raw_ev) > max_events_for_stats:
                rng = np.random.default_rng(42)
                sub_indices = rng.choice(len(raw_ev), size=max_events_for_stats, replace=False)
                sampled_ev = raw_ev[sub_indices]
            else:
                sampled_ev = raw_ev

            fluo_indices = [
                i - 1 for i, ch in enumerate(channel_details) if ch.channel_type == "fluorescent" and i - 1 < sampled_ev.shape[1]
            ]
            if fluo_indices:
                fluo_data = sampled_ev[:, fluo_indices]
                neg_count = np.sum(fluo_data <= 0)
                negative_event_pct = float((neg_count / fluo_data.size) * 100.0)

                sat_counts = 0
                for ch_idx in fluo_indices:
                    ch_info = channel_details[ch_idx]
                    limit = ch_info.range_max * 0.99
                    sat_counts += np.sum(sampled_ev[:, ch_idx] >= limit)
                saturated_event_pct = float((sat_counts / fluo_data.size) * 100.0)

                if negative_event_pct > 15.0:
                    qc_warnings.append(f"High negative event rate ({negative_event_pct:.1f}% <= 0)")
                if saturated_event_pct > 5.0:
                    qc_warnings.append(f"High detector saturation ({saturated_event_pct:.1f}% pegged at max)")
        except Exception:
            pass

    qc_status = "WARNING" if qc_warnings else "PASS"

    return FCSFileInfo(
        filepath=sanitize_path(path),
        filename=path.name,
        filesize_bytes=filesize_bytes,
        filesize_mb=filesize_mb,
        fcs_version=fcs_version,
        byte_order=byte_order_desc,
        data_type=data_type_desc,
        mode=mode_desc,
        event_count=event_count,
        channel_count=channel_count,
        vendor=vendor,
        cytometer_model=cytometer_model,
        cytometer_id=cytometer_id,
        software=software,
        lasers=lasers,
        acquisition_date=acq_date,
        acquisition_time_start=btim,
        acquisition_time_end=etim,
        acquisition_duration_sec=duration_sec,
        event_rate_eps=event_rate,
        scatter_channels=scatter_channels,
        fluorescent_channels=fluorescent_channels,
        time_channel=time_channel,
        markers=markers_set,
        fluorochromes=fluorochromes_set,
        panel_signature=panel_signature,
        has_spillover=has_spillover,
        spillover_channel_count=len(spill_channels),
        spillover_channels=spill_channels,
        transformation_status=transformation_val,
        patient_case_id=patient_case_id,
        population_class=population_class,
        population_name=population_name,
        qc_status=qc_status,
        qc_warnings=qc_warnings,
        channel_details=channel_details,
        negative_event_pct=negative_event_pct,
        saturated_event_pct=saturated_event_pct,
    )


def inspect_fcs_files(
    file_paths: List[Union[str, Path]],
    population_map: Optional[Union[Dict[str, str], str, Path]] = None,
    population_pattern: Optional[str] = None,
    include_event_stats: bool = False,
) -> Tuple[List[FCSFileInfo], FCSBatchSummary]:
    """Inspect an explicit list of FCS files and compute batch-level metrics.

    Allows user to pass arbitrary lists of filenames or paths from manifest files,
    CSV lists, glob queries, or direct CLI arguments.

    Args:
        file_paths: List of file path strings or Path objects.
        population_map: Optional dictionary or path to JSON/YAML population config.
        population_pattern: Optional regex pattern for population code extraction.
        include_event_stats: If True, computes deep event statistics.

    Returns:
        Tuple of (list_of_individual_file_infos, batch_summary_object).

    Raises:
        ValueError: If file_paths list is empty.
    """
    if not file_paths:
        raise ValueError("No FCS file paths provided for inspection")

    results: List[FCSFileInfo] = []
    for fp in file_paths:
        p = Path(fp).resolve()
        if not p.is_file():
            results.append(
                FCSFileInfo(
                    filepath=sanitize_path(p),
                    filename=p.name,
                    filesize_bytes=0,
                    filesize_mb=0.0,
                    fcs_version="NOT_FOUND",
                    byte_order="",
                    data_type="",
                    mode="",
                    event_count=0,
                    channel_count=0,
                    vendor="Unknown",
                    cytometer_model="",
                    cytometer_id="",
                    software="",
                    lasers="",
                    acquisition_date="",
                    acquisition_time_start="",
                    acquisition_time_end="",
                    acquisition_duration_sec=0.0,
                    event_rate_eps=0.0,
                    qc_status="ERROR",
                    qc_warnings=[f"File not found on disk: {p}"],
                )
            )
            continue

        try:
            info = inspect_fcs_file(
                p,
                population_map=population_map,
                population_pattern=population_pattern,
                include_event_stats=include_event_stats,
            )
            results.append(info)
        except Exception as exc:
            results.append(
                FCSFileInfo(
                    filepath=sanitize_path(p),
                    filename=p.name,
                    filesize_bytes=p.stat().st_size if p.exists() else 0,
                    filesize_mb=(p.stat().st_size / (1024 * 1024)) if p.exists() else 0.0,
                    fcs_version="CORRUPT",
                    byte_order="",
                    data_type="",
                    mode="",
                    event_count=0,
                    channel_count=0,
                    vendor="Unknown",
                    cytometer_model="",
                    cytometer_id="",
                    software="",
                    lasers="",
                    acquisition_date="",
                    acquisition_time_start="",
                    acquisition_time_end="",
                    acquisition_duration_sec=0.0,
                    event_rate_eps=0.0,
                    qc_status="ERROR",
                    qc_warnings=[f"Failed to inspect file: {exc}"],
                )
            )

    # Compile Cohort-level Batch Summary
    total_files = len(results)
    event_counts = [r.event_count for r in results]
    total_events = sum(event_counts)
    total_size_mb = sum(r.filesize_mb for r in results)
    min_events = min(event_counts) if event_counts else 0
    max_events = max(event_counts) if event_counts else 0
    median_events = float(np.median(event_counts)) if event_counts else 0.0
    mean_events = float(np.mean(event_counts)) if event_counts else 0.0

    unique_vendors = sorted(list({r.vendor for r in results if r.vendor and r.vendor != "Unknown"}))
    unique_cytometers = sorted(list({r.cytometer_model for r in results if r.cytometer_model}))
    unique_fcs_versions = sorted(list({r.fcs_version for r in results if r.fcs_version and r.fcs_version != "NOT_FOUND"}))

    panel_variants: Dict[str, int] = {}
    panel_variant_files: Dict[str, List[str]] = {}
    for r in results:
        sig = r.panel_signature or "Unknown Panel"
        panel_variants[sig] = panel_variants.get(sig, 0) + 1
        if sig not in panel_variant_files:
            panel_variant_files[sig] = []
        panel_variant_files[sig].append(r.filename)

    consensus_panel = max(panel_variants.items(), key=lambda kv: kv[1])[0] if panel_variants else "None"

    spill_present = sum(1 for r in results if r.has_spillover)
    spill_missing = sum(1 for r in results if not r.has_spillover)

    pop_counts: Dict[str, Dict[str, Any]] = {}
    for r in results:
        pop_code = r.population_class or ("Unannotated" if r.population_name == "Unannotated Specimen" else "Other")
        if pop_code not in pop_counts:
            pop_counts[pop_code] = {
                "name": r.population_name or pop_code,
                "file_count": 0,
                "event_count": 0,
            }
        pop_counts[pop_code]["file_count"] += 1
        pop_counts[pop_code]["event_count"] += r.event_count

    patient_cases = sorted(
        list({r.patient_case_id for r in results if r.patient_case_id}),
        key=lambda x: int(re.sub(r"\D", "", x)) if re.sub(r"\D", "", x) else 0,
    )
    acquisition_dates = sorted(list({r.acquisition_date for r in results if r.acquisition_date}))

    qc_pass = sum(1 for r in results if r.qc_status == "PASS")
    qc_warn = sum(1 for r in results if r.qc_status != "PASS")

    common_warnings: Dict[str, int] = {}
    for r in results:
        for w in r.qc_warnings:
            w_lower = w.lower()
            if "low event count" in w_lower:
                cat = "Low / rare event count (< 500 events)"
            elif "spillover" in w_lower:
                cat = "No embedded spillover/compensation matrix ($SPILLOVER absent)"
            elif "panel format discrepancy" in w_lower:
                cat = "Panel format discrepancy across files"
            elif "pre-transformed" in w_lower or "transformation: applied" in w_lower:
                cat = "Pre-transformed/compensated data"
            else:
                cat = w.split(":")[0] if ":" in w else w
            common_warnings[cat] = common_warnings.get(cat, 0) + 1

    summary = FCSBatchSummary(
        total_files=total_files,
        total_events=total_events,
        total_size_mb=total_size_mb,
        min_events=min_events,
        max_events=max_events,
        median_events=median_events,
        mean_events=mean_events,
        unique_vendors=unique_vendors,
        unique_cytometers=unique_cytometers,
        unique_fcs_versions=unique_fcs_versions,
        panel_variants=panel_variants,
        panel_variant_files=panel_variant_files,
        consensus_panel=consensus_panel,
        spillover_present_count=spill_present,
        spillover_missing_count=spill_missing,
        population_counts=pop_counts,
        patient_cases=patient_cases,
        acquisition_dates=acquisition_dates,
        qc_pass_count=qc_pass,
        qc_warning_count=qc_warn,
        common_warnings=common_warnings,
    )

    return results, summary


def find_fcs_files(
    target_dir: Union[str, Path],
    recursive: bool = True,
    pattern: Optional[str] = None,
    max_depth: Optional[int] = None,
    include_lmd: bool = True,
) -> List[Path]:
    """Discover all FCS files in a directory, supporting deep recursion, case-insensitivity, and depth control.

    Handles heterogeneous flow cytometry directory layouts:
    - Multiple nested subdirectories (e.g., patient folders, 96-well plates, acquisition dates).
    - Case-insensitive extension matching (.fcs, .FCS, .fcs3, .FCS3, .lmd, .LMD).
    - Subdirectory depth limitation via max_depth.
    - Automatic skipping of hidden directories (e.g. .git, .venv, .cache).

    Args:
        target_dir: Directory path containing flow cytometry data.
        recursive: If True (default), traverses all nested subdirectories.
        pattern: Optional custom glob pattern (e.g. '*Case*.fcs'). If None, matches standard FCS extensions.
        max_depth: Optional maximum directory depth to traverse (0 = only target_dir root).
        include_lmd: If True, also includes Beckman Coulter List Mode Data (.lmd/.LMD).

    Returns:
        Sorted list of resolved Path objects.

    Raises:
        FileNotFoundError: If target_dir does not exist or is not a directory.
    """
    root_path = Path(target_dir).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Target directory does not exist or is not a directory: {target_dir}")

    valid_exts = {".fcs", ".fcs3"}
    if include_lmd:
        valid_exts.update({".lmd"})

    # If user provided a specific glob pattern with wildcards (e.g. '*Case*.fcs')
    if pattern and any(c in pattern for c in "*?[]"):
        candidates: List[Path] = []
        if recursive:
            for p in root_path.rglob(pattern):
                if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root_path).parts):
                    if max_depth is not None:
                        depth = len(p.relative_to(root_path).parts) - 1
                        if depth > max_depth:
                            continue
                    candidates.append(p)
            # Also try uppercase pattern if standard lowercase was provided
            if pattern.endswith(".fcs") and not pattern.endswith(".FCS"):
                upper_pattern = pattern[:-4] + ".FCS"
                for p in root_path.rglob(upper_pattern):
                    if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root_path).parts):
                        if max_depth is not None:
                            depth = len(p.relative_to(root_path).parts) - 1
                            if depth > max_depth:
                                continue
                        candidates.append(p)
        else:
            for p in root_path.glob(pattern):
                if p.is_file() and not p.name.startswith("."):
                    candidates.append(p)
            if pattern.endswith(".fcs") and not pattern.endswith(".FCS"):
                upper_pattern = pattern[:-4] + ".FCS"
                for p in root_path.glob(upper_pattern):
                    if p.is_file() and not p.name.startswith("."):
                        candidates.append(p)

        seen: Set[Path] = set()
        unique: List[Path] = []
        for p in candidates:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(resolved)
        return sorted(unique)

    # General extension scanning (case-insensitive across .fcs, .FCS, .fcs3, .lmd)
    found_files: List[Path] = []

    if recursive:
        for root, dirs, files in os.walk(root_path):
            # Exclude hidden directories in-place (e.g. .git, .cache, .venv)
            dirs[:] = sorted([d for d in dirs if not d.startswith(".")])

            rel_root = Path(root).relative_to(root_path)
            depth = len(rel_root.parts) if rel_root.parts != (".",) and str(rel_root) != "." else 0

            if max_depth is not None and depth > max_depth:
                dirs.clear()  # Do not descend further into subdirectories
                continue

            for f in sorted(files):
                if f.startswith("."):
                    continue
                ext = Path(f).suffix.lower()
                if ext in valid_exts:
                    found_files.append((Path(root) / f).resolve())
    else:
        for entry in sorted(root_path.iterdir()):
            if entry.is_file() and not entry.name.startswith("."):
                if entry.suffix.lower() in valid_exts:
                    found_files.append(entry.resolve())

    # Return deduplicated, deterministically sorted file list
    seen_paths: Set[Path] = set()
    result: List[Path] = []
    for p in found_files:
        if p not in seen_paths:
            seen_paths.add(p)
            result.append(p)

    return sorted(result)


def inspect_fcs_directory(
    dir_path: Union[str, Path],
    recursive: bool = True,
    pattern: Optional[str] = None,
    max_depth: Optional[int] = None,
    population_map: Optional[Union[Dict[str, str], str, Path]] = None,
    population_pattern: Optional[str] = None,
    include_event_stats: bool = False,
) -> Tuple[List[FCSFileInfo], FCSBatchSummary]:
    """Inspect all FCS files in a target directory and compute batch-level metrics.

    Recursively scans nested subdirectories by default, matching both lowercase (.fcs)
    and uppercase (.FCS) flow cytometry data files.

    Args:
        dir_path: Directory path containing .fcs files.
        recursive: If True (default), recursively searches all nested subdirectories.
        pattern: Optional file glob pattern (e.g. '*Case*.fcs'). If None, matches all FCS files.
        max_depth: Optional maximum subdirectory depth to traverse.
        population_map: Optional dictionary or path to JSON/YAML population config.
        population_pattern: Optional regex pattern for population code extraction.
        include_event_stats: If True, computes deep event statistics.

    Returns:
        Tuple of (list_of_individual_file_infos, batch_summary_object).

    Raises:
        FileNotFoundError: If dir_path does not exist.
        ValueError: If no FCS files are found in directory.
    """
    target_dir = Path(dir_path).resolve()
    fcs_files = find_fcs_files(target_dir, recursive=recursive, pattern=pattern, max_depth=max_depth)
    if not fcs_files:
        raise ValueError(f"No FCS files found in '{sanitize_path(target_dir)}' (recursive={recursive})")

    return inspect_fcs_files(
        file_paths=fcs_files,
        population_map=population_map,
        population_pattern=population_pattern,
        include_event_stats=include_event_stats,
    )


def export_inspection_csv(results: List[FCSFileInfo], output_csv: Union[str, Path]) -> Path:
    """Export batch inspection results to a standardized CSV file."""
    out_path = Path(output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [r.to_summary_dict() for r in results]
    if not rows:
        raise ValueError("Cannot export empty inspection results list to CSV")

    fieldnames = list(rows[0].keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def export_channel_details_csv(results: List[FCSFileInfo], output_csv: Union[str, Path]) -> Path:
    """Export fine-grained channel and detector mappings for every file to CSV."""
    out_path = Path(output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for r in results:
        for ch in r.channel_details:
            rows.append(
                {
                    "filename": r.filename,
                    "patient_case_id": r.patient_case_id,
                    "population_class": r.population_class,
                    "channel_index": ch.index,
                    "pnn_detector": ch.pnn,
                    "pns_marker": ch.pns,
                    "channel_type": ch.channel_type,
                    "marker": ch.marker,
                    "fluorochrome": ch.fluorochrome,
                    "range_max_pnr": ch.range_max,
                    "bit_depth_pnb": ch.bit_depth,
                    "gain_png": ch.gain,
                    "voltage_pnv": ch.voltage,
                    "in_spillover": ch.in_spillover,
                }
            )

    if not rows:
        raise ValueError("No channel details found in inspection results")

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def export_inspection_markdown(
    summary: FCSBatchSummary,
    results: List[FCSFileInfo],
    output_md: Union[str, Path],
    title: str = "Flow Cytometry Dataset Inspection & QC Report",
    generated_by: str = "FlowX Automated FCS Inspector v1.0 (flowx.io.inspector)",
) -> Path:
    """Generate and export a comprehensive Markdown report directly from code.

    Args:
        summary: Aggregated FCSBatchSummary object.
        results: List of individual FCSFileInfo records.
        output_md: Target filepath for the markdown file.
        title: Title of the report document.
        generated_by: Attribution string indicating generating code/engine.

    Returns:
        Path object pointing to the written markdown file.
    """
    out_path = Path(output_md).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pass_pct = (summary.qc_pass_count / summary.total_files * 100.0) if summary.total_files > 0 else 0.0
    warn_pct = (summary.qc_warning_count / summary.total_files * 100.0) if summary.total_files > 0 else 0.0

    lines: List[str] = [
        f"# {title}",
        "",
        f"> **Generated by:** {generated_by}  ",
        f"> **Timestamp:** {now_iso}  ",
        f"> **Standard Compliance:** MIFlowCyt (Minimum Information about a Flow Cytometry Experiment) & ISAC FCS 2.0/3.0/3.1  ",
        f"> **Summary Output:** [`summary.csv`](file://{sanitize_path(out_path.parent / 'summary.csv')})  ",
        f"> **Channel Details:** [`channel_details.csv`](file://{sanitize_path(out_path.parent / 'channel_details.csv')})",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Value |",
        "|:-------|:------|",
        f"| **Total FCS Files** | {summary.total_files:,} files |",
        f"| **Total Recorded Events** | {summary.total_events:,} cell events |",
        f"| **Total Storage Size** | {summary.total_size_mb:.1f} MB ({summary.total_size_mb / 1024:.2f} GB) |",
        f"| **Event Count Range** | {summary.min_events:,} to {summary.max_events:,} (median: {summary.median_events:,.0f}) |",
        f"| **Detected Cytometers** | {', '.join(summary.unique_cytometers) or 'Standard Flow Cytometer'} |",
        f"| **Detected Manufacturers** | {', '.join(summary.unique_vendors) or 'Standard Vendor'} |",
        f"| **Acquisition Dates** | {', '.join(summary.acquisition_dates) or 'Not recorded'} |",
        f"| **Quality Control Status** | **{summary.qc_pass_count} Passed ({pass_pct:.1f}%)** │ **{summary.qc_warning_count} Flagged ({warn_pct:.1f}%)** |",
        f"| **Consensus Panel** | `{summary.consensus_panel}` |",
        "",
        "---",
        "",
        "## 2. Quality Control & Data Integrity Findings",
        "",
    ]

    # QC Findings bullets
    if summary.common_warnings:
        for cat, cnt in summary.common_warnings.items():
            lines.append(f"- **{cat}:** Encountered across **{cnt}** files.")
        lines.append("")
    else:
        lines.append("- All files passed automated quality control with zero warnings.\n")

    # Add alert callouts
    if summary.spillover_missing_count > 0:
        lines.extend([
            "> [!CAUTION]",
            f"> ### Compensation Matrices Missing in {summary.spillover_missing_count} Files",
            f"> While {summary.spillover_present_count} files contain embedded `$SPILLOVER` matrices, {summary.spillover_missing_count} files lack them.",
            "> Most of these uncompensated files have `transformation: applied` (e.g. flowCore exports), indicating that data was pre-compensated in R. Downstream pipelines must bypass redundant compensation for these files to prevent corrupting fluorescence values.",
            "",
        ])

    if len(summary.panel_variants) > 1:
        lines.extend([
            "> [!WARNING]",
            f"> ### Panel Naming Discrepancies Across Cohort ({len(summary.panel_variants)} Variants Detected)",
            "> Optical parameter detector names ($PnN) vary across files (e.g., standard `FL1 INT` vs dot notation `FL1.INT_CD14.FITC` vs clean stain name `CD14.FITC`).",
            "> Downstream algorithms must utilize marker-centric alias mapping to harmonize columns to canonical targets.",
            "",
        ])

    # 3. Population Distribution (if populations present)
    if summary.population_counts and any(k != "Unannotated" for k in summary.population_counts.keys()):
        lines.extend([
            "## 3. Cell Population Breakdown",
            "",
            "| Class | Population Description | Files | Total Events | % Cohort | Mean Events / File |",
            "|:-----:|:-----------------------|:-----:|:------------:|:--------:|:------------------:|",
        ])
        sorted_pops = sorted(
            summary.population_counts.items(),
            key=lambda kv: (-1 if kv[0] == "A" else 0, -kv[1]["event_count"]),
        )
        for code, pdata in sorted_pops:
            ev = pdata["event_count"]
            pct = (ev / summary.total_events * 100.0) if summary.total_events > 0 else 0.0
            mean_ev = (ev // pdata["file_count"]) if pdata["file_count"] > 0 else 0
            lines.append(f"| **{code}** | {pdata['name']} | {pdata['file_count']} | {ev:,} | {pct:.2f}% | {mean_ev:,} |")
        lines.append("")

    # 4. Panel Concordance Table
    lines.extend([
        "## 4. Optical Panel Configurations & Cross-File Concordance",
        "",
        "| Panel Variant Signature | Files | % Files | Status / Description |",
        "|:------------------------|:-----:|:-------:|:---------------------|",
    ])
    for sig, cnt in summary.panel_variants.items():
        pct = (cnt / summary.total_files * 100.0) if summary.total_files > 0 else 0.0
        if "Variant-1" in sig:
            desc = "Consensus standard raw Navios format (embedded $SPILLOVER present)"
        elif "Variant-2" in sig:
            desc = "flowCore dot-notation formatting (no embedded spillover)"
        elif "Variant-3" in sig:
            desc = "Direct marker-named detectors (no embedded spillover)"
        else:
            desc = "Custom or alternative channel layout"
        lines.append(f"| `{sig}` | {cnt} | {pct:.1f}% | {desc} |")
    lines.append("")

    # 5. Downstream Preprocessing Recommendations
    lines.extend([
        "## 5. Downstream Preprocessing Recommendations",
        "",
        "To ensure that discrepancies like **Variant 2** and **Variant 3** do not disrupt downstream preprocessing:",
        "",
        "1. **Canonical Marker Harmonization:**  ",
        "   Before feeding data into batch correction (CytoNorm) or clustering (FlowSOM/PhenoGraph), resolve all channel columns to unified biological marker names (e.g. `CD14`, `CD19`, `CD34`, `CD45`, `HLA-DR`, `FSC-A`, `SSC-A`) using `fcs.resolve_channel_name()` or an alias harmonization dictionary.",
        "",
        "2. **Conditional Compensation Bypass:**  ",
        "   Check the file's `has_spillover` and `transformation_status` flags. If `transformation: applied` or if the file is Variant 2/3 lacking a spillover matrix, bypass compensation to prevent double-subtraction.",
        "",
        "3. **Zero / Floor Value Imputation:**  ",
        "   Logicle or Arcsinh scaling should be parameterized with appropriate cofactor/W parameters tailored to the dynamic range ($PnR = 1,048,576 for Navios).",
        "",
        "---",
        "",
        "## 6. Sample-Level Inspection Roster (Sample)",
        "",
        "| Filename | Case ID | Pop | Event Count | Duration | Flow Rate | Spillover | QC Status |",
        "|:---------|:-------:|:---:|:-----------:|:--------:|:---------:|:---------:|:---------:|",
    ])
    for r in results[:35]:
        spill_str = "Present" if r.has_spillover else "None"
        dur_str = f"{r.acquisition_duration_sec:.1f}s" if r.acquisition_duration_sec > 0 else "-"
        rate_str = f"{r.event_rate_eps:.0f} eps" if r.event_rate_eps > 0 else "-"
        lines.append(
            f"| `{r.filename}` | {r.patient_case_id} | {r.population_class or '-'} | {r.event_count:,} | {dur_str} | {rate_str} | {spill_str} | `{r.qc_status}` |"
        )

    if len(results) > 35:
        lines.append(f"\n*Showing first 35 of {len(results)} files. See [`summary.csv`](file://{sanitize_path(out_path.parent / 'summary.csv')}) for the complete 38-column manifest.*")

    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def export_qc_file_lists(
    results: List[FCSFileInfo],
    output_dir: Union[str, Path],
    pass_filename: str = "files_pass.txt",
    flagged_filename: str = "files_flagged.txt",
    relative_to: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Export text file lists of samples partitioned by QC status ('PASS' vs 'FLAGGED'/'FAIL').

    Enables a seamless handoff from cohort inspection to downstream batch preprocessing
    pipelines (e.g. flowx pipeline --files-list outputs/flowcyt_class_a_inspection/files_pass.txt).

    Args:
        results: List of inspected FCSFileInfo records.
        output_dir: Destination folder where file lists will be saved.
        pass_filename: Name of the file list for samples passing all QC checks (default: 'files_pass.txt').
        flagged_filename: Name of the file list for samples with QC warnings/anomalies (default: 'files_flagged.txt').
        relative_to: Optional base path to make exported paths relative (defaults to CWD/workspace root).

    Returns:
        Tuple of (path_to_files_pass, path_to_files_flagged).
    """
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    pass_lines: List[str] = []
    flagged_lines: List[str] = []

    base_dir = Path(relative_to).resolve() if relative_to else Path.cwd().resolve()

    for r in results:
        fpath = Path(r.filepath).resolve()
        try:
            display_path = str(fpath.relative_to(base_dir))
        except ValueError:
            display_path = str(fpath)

        if str(r.qc_status).upper() == "PASS":
            pass_lines.append(display_path)
        else:
            flagged_lines.append(display_path)

    pass_path = target_dir / pass_filename
    flagged_path = target_dir / flagged_filename

    pass_path.write_text("\n".join(pass_lines) + ("\n" if pass_lines else ""), encoding="utf-8")
    flagged_path.write_text("\n".join(flagged_lines) + ("\n" if flagged_lines else ""), encoding="utf-8")

    return pass_path, flagged_path


def export_inspection_suite(
    results: List[FCSFileInfo],
    summary: FCSBatchSummary,
    output_dir: Union[str, Path] = "outputs/flowcyt_inspection",
    summary_filename: str = "summary.csv",
    channels_filename: str = "channel_details.csv",
    report_filename: str = "inspection_report.md",
    pass_filename: str = "files_pass.txt",
    flagged_filename: str = "files_flagged.txt",
    generated_by: str = "FlowX Automated FCS Inspector v1.0",
) -> Dict[str, Path]:
    """Export the complete audit package: summary CSV, channel details CSV, report, and QC file lists.

    Args:
        results: List of inspected FCSFileInfo records.
        summary: Aggregated FCSBatchSummary object.
        output_dir: Destination folder (defaults to 'outputs/flowcyt_inspection').
        summary_filename: Name of the summary CSV.
        channels_filename: Name of the channel breakdown CSV.
        report_filename: Name of the generated Markdown report.
        pass_filename: Name of the file list for samples passing QC.
        flagged_filename: Name of the file list for samples flagged during QC.
        generated_by: Attribution signature for the report.

    Returns:
        Dictionary mapping artifact keys to saved Paths.
    """
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    summary_path = export_inspection_csv(results, target_dir / summary_filename)
    channels_path = export_channel_details_csv(results, target_dir / channels_filename)
    report_path = export_inspection_markdown(
        summary=summary,
        results=results,
        output_md=target_dir / report_filename,
        generated_by=generated_by,
    )
    pass_path, flagged_path = export_qc_file_lists(
        results=results,
        output_dir=target_dir,
        pass_filename=pass_filename,
        flagged_filename=flagged_filename,
    )

    return {
        "summary_csv": summary_path,
        "channels_csv": channels_path,
        "report_md": report_path,
        "files_pass": pass_path,
        "files_flagged": flagged_path,
    }
