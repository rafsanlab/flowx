"""Sample metadata and cohort experimental design parser.

Provides loading and validation of sample metadata tables (CSV, TSV, Excel)
for cohort grouping, differential abundance, and differential state analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd

from flowx.io.paths import sanitize_path

NON_FLUORESCENCE_MARKER_CLASSES = {
    "scatter",
    "time",
    "unused",
    "index",
    "qc",
    "autofluorescence",
}
NON_CLUSTERING_MARKER_CLASSES = NON_FLUORESCENCE_MARKER_CLASSES | {"viability", "dump"}
VALID_MARKER_CLASSES = {
    "lineage",
    "functional",
    "viability",
    "dump",
    "scatter",
    "time",
    "unused",
    "index",
    "qc",
    "autofluorescence",
}


def load_sample_metadata(
    filepath: Union[str, Path],
    sample_id_column: Optional[str] = None,
) -> pd.DataFrame:
    """Load sample experimental metadata table for cohort statistical analysis.

    Supports CSV (.csv), TSV (.tsv), tab-delimited text (.txt), and Excel (.xlsx, .xls) files.
    Automatically identifies and cleans the sample ID column, stripping .fcs extensions
    to ensure 1:1 matching with cohort sample keys.

    Args:
        filepath: Path to metadata tabular file.
        sample_id_column: Explicit column name containing sample identifiers.
                          If None, auto-detected from common column names.

    Returns:
        pd.DataFrame indexed by cleaned sample_id.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Sample metadata file not found: {sanitize_path(path)}")

    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        try:
            df = pd.read_csv(path)
            # If only 1 column was parsed but text has tabs, try sep='\t'
            if len(df.columns) == 1:
                df_tab = pd.read_csv(path, sep="\t")
                if len(df_tab.columns) > 1:
                    df = df_tab
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Resolve sample ID column
    sample_col = sample_id_column
    if sample_col is None or sample_col not in df.columns:
        candidates = [
            "sample_id",
            "sample",
            "Sample_ID",
            "SampleID",
            "Sample",
            "sample_name",
            "filename",
            "file_name",
            "fcs_file",
            "file",
            "id",
            "ID",
        ]
        for cand in candidates:
            if cand in df.columns:
                sample_col = cand
                break

    if sample_col is not None and sample_col in df.columns:
        # Strip .fcs extension and directory names to match sample stems
        df[sample_col] = (
            df[sample_col]
            .astype(str)
            .apply(lambda s: Path(s).name)
            .str.replace(r"\.fcs$", "", case=False, regex=True)
        )
        df.set_index(sample_col, inplace=True)
        df.index.name = "sample_id"
    elif not df.index.name:
        # Fall back to first column if it contains strings or unique IDs
        first_col = df.columns[0]
        if df[first_col].dtype == object or df[first_col].is_unique:
            df[first_col] = (
                df[first_col]
                .astype(str)
                .apply(lambda s: Path(s).name)
                .str.replace(r"\.fcs$", "", case=False, regex=True)
            )
            df.set_index(first_col, inplace=True)
            df.index.name = "sample_id"

    return df


@dataclass
class MarkerMetadataResolution:
    """Resolved channel-to-marker ontology and provenance for a cohort/panel.

    Attributes:
        detector_to_marker: Dict mapping physical FCS detector ($PnN) to canonical biological marker name.
        marker_to_detector: Dict mapping biological marker name back to physical FCS detector ($PnN).
        marker_classes: Dict mapping physical detector ($PnN) to normalized list of roles
            (e.g. ['lineage', 'functional']).
        clustering_channels: Recommended fluorescence detectors for clustering based on marker_class
            (includes 'lineage' / 'functional'; excludes 'scatter', 'time', 'viability', 'dump').
        markers_df: Full loaded markers DataFrame if markers.csv was supplied, else None.
        provenance_source: Human-readable string documenting where mappings came from.
        has_verified_biological_markers: True if mappings came from curated markers.csv, explicit
            marker_mapping, or non-empty FCS $PnS fields; False if detectors had to be kept verbatim.
        warnings: Actionable diagnostics emitted during resolution.
    """

    detector_to_marker: Dict[str, str] = field(default_factory=dict)
    marker_to_detector: Dict[str, str] = field(default_factory=dict)
    detector_to_fluorophore: Dict[str, str] = field(default_factory=dict)
    marker_classes: Dict[str, List[str]] = field(default_factory=dict)
    clustering_channels: List[str] = field(default_factory=list)
    markers_df: Optional[pd.DataFrame] = None
    provenance_source: str = "unmapped_detectors"
    has_verified_biological_markers: bool = False
    warnings: List[str] = field(default_factory=list)


def _split_semicolon_field(val: Any) -> List[str]:
    """Parse semicolon-delimited multi-valued CSV fields cleanly."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    txt = str(val).strip()
    if not txt or txt.lower() in {"nan", "none", "na"}:
        return []
    return [part.strip() for part in txt.split(";") if part.strip()]


def _extract_antigen_from_pns(pns_label: str) -> str:
    """Extract canonical antigen symbol from a compound FCS $PnS label like 'CD4 PcpCy55' or 'HLA-DR V500'."""
    txt = str(pns_label).strip()
    if not txt:
        return ""
    tokens = txt.split()
    if len(tokens) == 1:
        return tokens[0]
    # Check if trailing token represents a common fluorophore tag paired after the antigen symbol
    fluoro_hints = (
        "fitc", "pe", "apc", "pcp", "percp", "bv", "buv", "bb", "sb", "v450", "v500", "v605",
        "pacblue", "pacorange", "ax", "af", "alexa", "cy5", "cy7", "cf594", "txred", "ecd",
        "e450", "efluor", "spark", "nova", "rb", "ry", "dazzle", "fire", "apch7", "pecy7",
    )
    tail_low = tokens[-1].lower().replace("-", "")
    if any(hint in tail_low for hint in fluoro_hints):
        return " ".join(tokens[:-1]).strip()
    return txt


def load_marker_metadata(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load and validate a curated panel marker metadata table (markers.csv / TSV).

    Enforces 1-row-per-channel keying via 'channel_name' ($PnN) and supports semicolon-delimited
    multi-valued roles in 'marker_class' and 'cell_type_association'.

    Args:
        filepath: Path to markers.csv or markers.tsv.

    Returns:
        Validated pandas DataFrame containing normalized columns and parsed 'marker_classes_list'.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Marker metadata file not found: {sanitize_path(path)}")

    ext = path.suffix.lower()
    if ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if "channel_name" not in df.columns:
        raise ValueError(
            f"Marker metadata file {sanitize_path(path)} missing required primary key column 'channel_name'."
        )
    if "marker_name" not in df.columns:
        raise ValueError(
            f"Marker metadata file {sanitize_path(path)} missing required column 'marker_name'."
        )

    df = df.copy()
    df["channel_name"] = df["channel_name"].astype(str).str.strip()
    df["marker_name"] = df["marker_name"].fillna("").astype(str).str.strip()

    if df["channel_name"].duplicated().any():
        dups = df.loc[df["channel_name"].duplicated(), "channel_name"].tolist()
        raise ValueError(
            f"Marker metadata {sanitize_path(path)} contains duplicate 'channel_name' keys: {dups}"
        )

    if "marker_class" in df.columns:
        parsed_classes: List[List[str]] = []
        for raw_cls in df["marker_class"]:
            items = [c.lower() for c in _split_semicolon_field(raw_cls)]
            parsed_classes.append(items)
        df["marker_classes_list"] = parsed_classes
    else:
        df["marker_classes_list"] = [[] for _ in range(len(df))]

    return df


def resolve_marker_metadata(
    markers_file: Optional[Union[str, Path]] = None,
    dataset_dir: Optional[Union[str, Path]] = None,
    fcs_pnn: Optional[Sequence[str]] = None,
    fcs_pns: Optional[Sequence[str]] = None,
    marker_mapping: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> MarkerMetadataResolution:
    """Resolve detector-to-marker ontology deterministically with zero silent guessing.

    Priority cascade:
    1. Explicit `markers_file` (markers.csv / markers.tsv).
    2. Auto-discovered `<dataset_dir>/metadata/markers.csv` (or `markers.tsv` / root `markers.csv`).
    3. Explicit user `marker_mapping` dictionary.
    4. Verified non-empty FCS `$PnS` fields (recorded provenance: 'FCS $PnS').
    5. Raw physical detector names (`$PnN`) verbatim with an explicit diagnostic warning.
    """
    res = MarkerMetadataResolution()
    pnn_list = [str(c) for c in (fcs_pnn or [])]
    pns_list = [str(c) if c is not None else "" for c in (fcs_pns or [])]

    # 1. Load explicit markers_file or check dataset_dir if explicitly provided
    target_path: Optional[Path] = None
    if markers_file:
        target_path = Path(markers_file)
        if not target_path.is_file():
            raise FileNotFoundError(f"Specified markers file not found: {sanitize_path(target_path)}")
    elif dataset_dir:
        ds_dir = Path(dataset_dir)
        if ds_dir.is_dir():
            search_dirs = [ds_dir]
            if ds_dir.name == "samples" and ds_dir.parent.name == "raw":
                search_dirs.append(ds_dir.parent.parent)
            elif ds_dir.name in ("samples", "raw"):
                search_dirs.append(ds_dir.parent)
            for sdir in search_dirs:
                for cand in (
                    sdir / "metadata" / "markers.csv",
                    sdir / "metadata" / "markers.tsv",
                    sdir / "markers.csv",
                    sdir / "markers.tsv",
                ):
                    if cand.is_file():
                        target_path = cand
                        break
                if target_path is not None:
                    break
    elif logger:
        logger.info("No explicit 'markers_file' specified; resolving panel parameters from FCS headers and explicit mappings.")

    if target_path is not None and target_path.is_file():
        mdf = load_marker_metadata(target_path)
        res.markers_df = mdf
        res.provenance_source = f"markers_csv:{sanitize_path(target_path)}"
        raw_clustering_candidates: List[Tuple[str, str]] = []
        for _, row in mdf.iterrows():
            ch = str(row["channel_name"]).strip()
            mk = str(row["marker_name"]).strip() or ch
            roles = list(row["marker_classes_list"])
            res.detector_to_marker[ch] = mk
            # Prefer Area (-A) detector over Height (-H) or Width (-W) when mapping marker_to_detector
            if mk not in res.marker_to_detector or ch.upper().endswith("-A"):
                res.marker_to_detector[mk] = ch
            if "fluorophore" in mdf.columns and pd.notna(row.get("fluorophore")):
                fl_str = str(row.get("fluorophore")).strip()
                if fl_str and fl_str.lower() not in ({"nan", "none"} | NON_FLUORESCENCE_MARKER_CLASSES):
                    res.detector_to_fluorophore[ch] = fl_str
            res.marker_classes[ch] = roles

            # Select clustering channels cleanly: exclude scatter, time, viability, dump, unused, index, qc, autofluorescence
            has_exclusion_role = bool(set(roles) & NON_CLUSTERING_MARKER_CLASSES)
            if not has_exclusion_role:
                if not roles or ("lineage" in roles or "functional" in roles):
                    raw_clustering_candidates.append((ch, mk))
                if mk.lower() not in {"fsc-a", "fsc-h", "fsc-w", "ssc-a", "ssc-h", "ssc-w", "time"} and mk != ch:
                    res.has_verified_biological_markers = True

        # Deduplicate pulse-height (-H) and pulse-width (-W) channels when the corresponding
        # Area (-A) detector or identical biological antigen is already included in clustering_channels
        area_Write_bases = {
            ch[:-2].upper() for ch, _ in raw_clustering_candidates if ch.upper().endswith("-A")
        }
        seen_markers: set[str] = set()
        for ch, mk in raw_clustering_candidates:
            ch_up = ch.upper()
            if (ch_up.endswith("-H") or ch_up.endswith("-W")) and ch_up[:-2] in area_Write_bases:
                continue
            mk_norm = mk.strip().lower()
            if (ch_up.endswith("-H") or ch_up.endswith("-W")) and mk_norm in seen_markers:
                continue
            res.clustering_channels.append(ch)
            if mk_norm:
                seen_markers.add(mk_norm)

        # Ensure any explicit user marker_mapping overrides cleanly if both supplied
        if marker_mapping:
            for k, v in marker_mapping.items():
                if k in res.detector_to_marker or k in pnn_list:
                    res.detector_to_marker[k] = v
                    res.marker_to_detector[v] = k
                else:
                    # Handle marker->detector convention ('CD3': 'V450-A')
                    res.marker_to_detector[k] = v
                    res.detector_to_marker[v] = k
            res.has_verified_biological_markers = True
        return res

    # No markers.csv found -> emit clear warning
    warn_msg = (
        "No 'markers.csv' panel dictionary found; falling back to explicit marker_mapping "
        "or FCS $PnS headers. Provide 'metadata/markers.csv' for deterministic marker roles and provenance."
    )
    res.warnings.append(warn_msg)
    if logger:
        logger.warning(warn_msg)

    # 3. Apply explicit marker_mapping dictionary if supplied
    if marker_mapping:
        res.provenance_source = "user_marker_mapping"
        for k, v in marker_mapping.items():
            if k in pnn_list:
                res.detector_to_marker[k] = v
                res.marker_to_detector[v] = k
            elif v in pnn_list:
                res.marker_to_detector[k] = v
                res.detector_to_marker[v] = k
            else:
                res.marker_to_detector[k] = v
                res.detector_to_marker[v] = k
        res.has_verified_biological_markers = True

    # 4. Populate any remaining optical channels from verified non-empty FCS $PnS
    pns_hits = 0
    for idx, pnn in enumerate(pnn_list):
        if pnn in res.detector_to_marker:
            continue
        pns_raw = pns_list[idx].strip() if idx < len(pns_list) else ""
        if pns_raw and pns_raw != pnn:
            antigen = _extract_antigen_from_pns(pns_raw)
            res.detector_to_marker[pnn] = antigen
            if antigen not in res.marker_to_detector:
                res.marker_to_detector[antigen] = pnn
            pns_hits += 1
        else:
            res.detector_to_marker[pnn] = pnn

    if pns_hits > 0:
        res.has_verified_biological_markers = True
        if res.provenance_source == "unmapped_detectors":
            res.provenance_source = "fcs_pns_headers"

    if not res.has_verified_biological_markers and pnn_list:
        empty_warn = (
            "Both 'markers.csv' and FCS $PnS marker annotations are absent/blank. "
            "Cluster annotation will strictly retain physical detector IDs ($PnN) without inferring biological lineages."
        )
        res.warnings.append(empty_warn)
        if logger:
            logger.warning(empty_warn)

    return res


def resolve_scatter_and_role_channels(
    marker_metadata: Optional[MarkerMetadataResolution],
    available_columns: Sequence[str],
) -> Dict[str, Optional[str]]:
    """Resolve scatter, viability, and dump channels cleanly from curated markers metadata.

    Scans `markers.csv` roles (`scatter`, `viability`, `dump`) intersected with `available_columns`
    so configs can declare `'auto'` instead of hardcoding detector names.
    """
    cols = list(available_columns)
    cols_set = set(cols)
    resolved: Dict[str, Optional[str]] = {
        "fsc_channel": None,
        "ssc_channel": None,
        "fsc_height_channel": None,
        "ssc_height_channel": None,
        "viability_channel": None,
        "dump_channel": None,
    }
    if marker_metadata is None or not getattr(marker_metadata, "marker_classes", None):
        return resolved

    for ch, roles in marker_metadata.marker_classes.items():
        if ch not in cols_set:
            continue
        roles_low = {r.lower() for r in roles}
        ch_low = ch.lower()
        mk_low = str(marker_metadata.detector_to_marker.get(ch, ch)).lower()
        combined = f"{ch_low} {mk_low}"

        if "scatter" in roles_low:
            if resolved["fsc_channel"] is None and "fsc" in combined and ("-a" in ch_low or ("-h" not in ch_low and "-w" not in ch_low)):
                resolved["fsc_channel"] = ch
            elif resolved["fsc_height_channel"] is None and "fsc" in combined and "-h" in ch_low:
                resolved["fsc_height_channel"] = ch
            elif resolved["ssc_channel"] is None and "ssc" in combined and ("-a" in ch_low or ("-h" not in ch_low and "-w" not in ch_low)):
                resolved["ssc_channel"] = ch
            elif resolved["ssc_height_channel"] is None and "ssc" in combined and "-h" in ch_low:
                resolved["ssc_height_channel"] = ch
        if "viability" in roles_low and resolved["viability_channel"] is None:
            resolved["viability_channel"] = ch
        if "dump" in roles_low and resolved["dump_channel"] is None:
            resolved["dump_channel"] = ch

    return resolved


def select_markers_from_metadata(
    marker_metadata: Optional[MarkerMetadataResolution],
    available_columns: Sequence[str],
    explicit_markers: Optional[Union[str, Sequence[str]]] = None,
    marker_roles: Optional[Sequence[str]] = None,
    include_markers: Optional[Sequence[str]] = None,
    exclude_markers: Optional[Sequence[str]] = None,
) -> Optional[List[str]]:
    """Resolve clustering / lineage detector channels using roles + include/exclude overrides.

    Supports:
    - `explicit_markers="auto"` -> select channels whose `marker_class` matches `marker_roles`
      (defaults to `['lineage']`, falling back to `clustering_channels` if no rows match `'lineage'`).
    - `explicit_markers=[...]` -> resolve detector or marker symbols explicitly.
    - `marker_roles=[...]` -> filter `markers.csv` rows matching any listed role (excluding scatter/time/viability/dump
      unless explicitly included via `include_markers`).
    - `include_markers` / `exclude_markers` -> additive and subtractive detector/antigen overrides.
    """
    cols = list(available_columns)
    cols_set = set(cols)

    def _to_detector(token: str) -> Optional[str]:
        tok = str(token).strip()
        if not tok:
            return None
        if tok in cols_set:
            return tok
        if marker_metadata is not None:
            det = marker_metadata.marker_to_detector.get(tok)
            if det and det in cols_set:
                return det
            # Case-insensitive fallback match on marker_name / detector
            tok_low = tok.lower()
            for d, m in marker_metadata.detector_to_marker.items():
                if d in cols_set and (d.lower() == tok_low or str(m).lower() == tok_low):
                    return d
        return None

    is_auto = isinstance(explicit_markers, str) and explicit_markers.strip().lower() == "auto"
    explicit_list = (
        [str(x) for x in explicit_markers if x]
        if isinstance(explicit_markers, (list, tuple))
        else []
    )

    selected: List[str] = []
    if explicit_list and not (len(explicit_list) == 1 and explicit_list[0].lower() == "auto"):
        for item in explicit_list:
            det = _to_detector(item)
            if det and det not in selected:
                selected.append(det)
    elif marker_metadata is not None and (is_auto or marker_roles is not None):
        target_roles = {str(r).strip().lower() for r in (marker_roles or ["lineage"]) if str(r).strip()}
        raw_cands: List[Tuple[str, str]] = []
        for ch, roles in marker_metadata.marker_classes.items():
            if ch not in cols_set:
                continue
            roles_low = {r.lower() for r in roles}
            if roles_low & NON_CLUSTERING_MARKER_CLASSES:
                continue
            if roles_low & target_roles:
                mk = marker_metadata.detector_to_marker.get(ch, ch)
                raw_cands.append((ch, mk))

        # Fallback to full clustering_channels if strict role (e.g. 'lineage') yielded empty panel
        if not raw_cands and marker_metadata.clustering_channels:
            selected = [c for c in marker_metadata.clustering_channels if c in cols_set]
        else:
            area_bases = {ch[:-2].upper() for ch, _ in raw_cands if ch.upper().endswith("-A")}
            seen_mk: set[str] = set()
            for ch, mk in raw_cands:
                ch_up = ch.upper()
                if (ch_up.endswith("-H") or ch_up.endswith("-W")) and ch_up[:-2] in area_bases:
                    continue
                mk_low = mk.strip().lower()
                if (ch_up.endswith("-H") or ch_up.endswith("-W")) and mk_low in seen_mk:
                    continue
                if ch not in selected:
                    selected.append(ch)
                if mk_low:
                    seen_mk.add(mk_low)
    elif marker_metadata is not None and marker_metadata.clustering_channels:
        selected = [c for c in marker_metadata.clustering_channels if c in cols_set]
    elif not is_auto and not include_markers and not exclude_markers:
        return None

    # Additive override: include_markers
    if include_markers:
        for item in include_markers:
            det = _to_detector(item)
            if det and det not in selected:
                selected.append(det)

    # Subtractive override: exclude_markers
    if exclude_markers:
        excl_dets: set[str] = set()
        for item in exclude_markers:
            det = _to_detector(item)
            if det:
                excl_dets.add(det)
            excl_dets.add(str(item).strip())
        selected = [c for c in selected if c not in excl_dets]

    return selected if selected else None


