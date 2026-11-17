"""FCS file parsing and data container module.

Provides robust reading of FCS 2.0, 3.0, and 3.1 files, metadata extraction,
channel mapping (detectors and markers), and spillover matrix parsing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import flowkit as fk
import numpy as np
import pandas as pd

from flowx.io.paths import sanitize_path


@dataclass
class FCSData:
    """Container for parsed flow cytometry sample data and metadata.

    Attributes:
        filepath: Full path to the FCS file.
        filename: Base name of the FCS file.
        event_count: Number of recorded cell events.
        channel_count: Number of measured parameters/channels.
        pnn_labels: List of parameter detector names (e.g. FSC-A, Ax700-A).
        pns_labels: List of parameter stain/marker names (e.g. CD3, Dead).
        detector_to_marker: Mapping from detector name to stain label.
        marker_to_detector: Mapping from stain label to detector name.
        metadata: Dictionary containing all FCS header and text keywords.
        raw_events: 2D NumPy array of shape (events, channels).
        events_df: Pandas DataFrame of event data with detector columns.
        spillover_matrix: Pandas DataFrame representation of compensation matrix if present.
        spillover_channels: List of channel detector names associated with the spillover matrix.
        sample_obj: Optional FlowKit Sample object reference.
    """

    filepath: str
    filename: str
    event_count: int
    channel_count: int
    pnn_labels: List[str]
    pns_labels: List[str]
    detector_to_marker: Dict[str, str] = field(default_factory=dict)
    marker_to_detector: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_events: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    events_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    spillover_matrix: Optional[pd.DataFrame] = None
    spillover_channels: List[str] = field(default_factory=list)
    cytometer: str = ""
    channel_ranges: Dict[str, float] = field(default_factory=dict)
    sample_obj: Optional[Any] = None
    channel_resolutions: Dict[str, str] = field(default_factory=dict)
    resolution_audit_log: List[Dict[str, Any]] = field(default_factory=list)

    def get_resolution_audit(self) -> pd.DataFrame:
        """Return a structured DataFrame of all resolved channel queries and strategies for auditing.

        Returns:
            Pandas DataFrame containing resolution audit records with columns:
            ['query', 'resolved_detector', 'strategy', 'marker_label', 'sample', 'timestamp'].
        """
        if not self.resolution_audit_log:
            return pd.DataFrame(
                columns=["query", "resolved_detector", "strategy", "marker_label", "sample", "timestamp"]
            )
        return pd.DataFrame(self.resolution_audit_log)

    def export_resolution_audit(self, output_path: Union[str, Path]) -> Path:
        """Export channel resolution audit trail to CSV file for reproducibility and MIFlowCyt compliance.

        Args:
            output_path: Target CSV filepath.

        Returns:
            Resolved Path object.
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df = self.get_resolution_audit()
        df.to_csv(out_p, index=False)
        return out_p

    @property
    def sample_name(self) -> str:
        """Sample identifier derived from file stem or $FIL header keyword."""
        if hasattr(self, "filename") and self.filename:
            return Path(self.filename).stem
        if hasattr(self, "filepath") and self.filepath:
            return Path(self.filepath).stem
        return self.metadata.get("$FIL", "sample")

    def export_fcs(
        self,
        output_path: Union[str, Path],
        events_df: Optional[pd.DataFrame] = None,
        is_compensated: bool = False,
    ) -> Path:
        """Export current or filtered events to a standardized FCS 3.0 / 3.1 file.

        Preserves channel labels ($PnN), marker annotations ($PnS), cytometer metadata,
        and embedded spillover matrices. If is_compensated=True, embeds an identity
        spillover matrix to prevent downstream software from double-compensating.

        Args:
            output_path: Target .fcs file path.
            events_df: Optional subset DataFrame to export (defaults to self.events_df).
            is_compensated: Whether exported events are already compensated.

        Returns:
            Resolved Path to the written FCS file.
        """
        target_df = events_df if events_df is not None else self.events_df
        spill = self.spillover_matrix
        if is_compensated and spill is not None:
            # Compensated data has identity spillover between fluorescent channels
            spill = pd.DataFrame(
                np.eye(len(spill.columns), dtype=np.float64),
                index=list(spill.index),
                columns=list(spill.columns),
            )
        return write_fcs(
            output_path=output_path,
            events_df=target_df,
            pnn_labels=self.pnn_labels,
            pns_labels=self.pns_labels,
            metadata=self.metadata,
            spillover_matrix=spill,
            channel_ranges=self.channel_ranges,
            is_compensated=is_compensated,
        )

    def harmonize(
        self,
        harmonizer_or_config: Any,
        inplace: bool = False,
    ) -> Any:
        """Harmonize this FCSData's event channels against a panel harmonization specification.

        Args:
            harmonizer_or_config: A PanelHarmonizer instance or path to panel_harmonization.yaml.
            inplace: If True, replaces events_df column names in place.

        Returns:
            HarmonizationResult object.
        """
        from flowx.panels.harmonization import PanelHarmonizer

        if isinstance(harmonizer_or_config, PanelHarmonizer):
            harmonizer = harmonizer_or_config
        else:
            harmonizer = PanelHarmonizer.load(harmonizer_or_config)

        return harmonizer.harmonize_sample(self, inplace=inplace)

    def get_harmonized_df(
        self,
        harmonizer_or_config: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Retrieve events DataFrame with column headers harmonized to canonical marker names.

        Args:
            harmonizer_or_config: Optional PanelHarmonizer or path to panel_harmonization.yaml.
                If None and 'harmonized_channels' metadata exists, uses the cached mapping.

        Returns:
            Pandas DataFrame with canonical marker columns.
        """
        if harmonizer_or_config is not None:
            res = self.harmonize(harmonizer_or_config, inplace=False)
            return res.harmonized_df

        if "harmonized_channels" in self.metadata and isinstance(self.metadata["harmonized_channels"], dict):
            mapping = self.metadata["harmonized_channels"]
            rev = {v: k for k, v in mapping.items()}
            return self.events_df.rename(columns=rev)

        # Default fallback: resolve each column to marker label if available
        rename_map = {}
        for pnn in self.pnn_labels:
            marker = self.detector_to_marker.get(pnn)
            if marker and marker != pnn:
                rename_map[pnn] = marker
        return self.events_df.rename(columns=rename_map)

    def get_channel_label(self, channel: str) -> str:
        """Return a formatted display name for a channel (e.g. 'CD3 (Ax700-A)').

        Args:
            channel: Detector name or marker name.

        Returns:
            Human-readable display string.
        """
        try:
            resolved = self.resolve_channel_name(channel)
        except KeyError:
            resolved = str(channel)
        marker = self.detector_to_marker.get(resolved, "")
        if marker and marker.strip() and marker.strip() != resolved:
            return f"{marker} ({resolved})"
        return resolved

    def get_fsc_channel(self) -> str:
        """Return the primary forward scatter detector channel name."""
        candidates = [
            "fsc-a", "fs int", "fs.int", "fsc", "fs", "fs-int", "fs_int", "fsc_a", "forward scatter", "fsc-area"
        ]
        res = self.find_channel_by_alias(candidates)
        if res:
            return res
        for pnn in self.pnn_labels:
            p_low = pnn.lower()
            if "fsc" in p_low or p_low.startswith("fs"):
                return pnn
        return self.pnn_labels[0]

    def get_ssc_channel(self) -> str:
        """Return the primary side scatter detector channel name."""
        candidates = [
            "ssc-a", "ss int", "ss.int", "ssc", "ss", "ss-int", "ss_int", "ssc_a", "side scatter", "ssc-area"
        ]
        res = self.find_channel_by_alias(candidates)
        if res:
            return res
        for pnn in self.pnn_labels:
            p_low = pnn.lower()
            if ("ssc" in p_low or p_low.startswith("ss")) and "peak" not in p_low and "tof" not in p_low:
                return pnn
        return self.pnn_labels[1] if len(self.pnn_labels) > 1 else self.pnn_labels[0]

    def get_fsc_h_channel(self) -> Optional[str]:
        """Return forward scatter height channel name if present, else None."""
        candidates = [
            "fsc-h", "fs peak", "fs.peak", "fs-h", "fs_h", "fsc_h", "forward scatter height"
        ]
        return self.find_channel_by_alias(candidates)

    def get_ssc_h_channel(self) -> Optional[str]:
        """Return side scatter height channel name if present, else None."""
        candidates = [
            "ssc-h", "ss peak", "ss.peak", "ss-h", "ss_h", "ssc_h", "side scatter height"
        ]
        return self.find_channel_by_alias(candidates)

    def get_fsc_w_channel(self) -> Optional[str]:
        """Return forward scatter width / time-of-flight channel name if present, else None."""
        candidates = [
            "fsc-w", "fs tof", "fs.tof", "fs-w", "fs_w", "fsc_w", "forward scatter width"
        ]
        return self.find_channel_by_alias(candidates)

    def get_ssc_w_channel(self) -> Optional[str]:
        """Return side scatter width / time-of-flight channel name if present, else None."""
        candidates = [
            "ssc-w", "ss tof", "ss.tof", "ss-w", "ss_w", "ssc_w", "side scatter width"
        ]
        return self.find_channel_by_alias(candidates)

    def get_time_channel(self) -> Optional[str]:
        """Return the time parameter channel name if present, else None."""
        candidates = ["time", "time ms", "time.ms", "time-a"]
        return self.find_channel_by_alias(candidates)

    def resolve_channel_name(self, channel: str) -> str:
        """Resolve a channel name (either detector or marker) to its primary detector name.

        Robustly resolves standard scatter names ('FSC-A' -> 'FS INT' / 'FS.INT'),
        marker names ('CD14' -> 'CD14-FITC' / 'FL1 INT'), punctuation variants,
        and case variations. Tracks resolution decisions in resolution_audit_log and
        logs them via flowx logger.

        Args:
            channel: Detector name (e.g. 'Ax700-A') or marker name (e.g. 'CD3', 'FSC-A').

        Returns:
            Primary detector name (PnN label).

        Raises:
            KeyError: If channel name cannot be found in detectors or markers.
        """
        import datetime
        import logging
        import re

        logger = logging.getLogger("flow_cytometry")

        # Fast path: check existing resolution cache
        if hasattr(self, "channel_resolutions") and self.channel_resolutions and channel in self.channel_resolutions:
            return self.channel_resolutions[channel]

        resolved: Optional[str] = None
        strategy: str = ""

        if channel in self.pnn_labels:
            resolved = channel
            strategy = "exact_detector"
        elif channel in self.marker_to_detector:
            resolved = self.marker_to_detector[channel]
            strategy = "exact_marker"
        else:
            channel_lower = channel.lower().strip()

            # Check standard scatter aliases
            fsc_aliases = {"fsc", "fsc-a", "fsc_a", "fs int", "fs.int", "fs", "fs-int", "fs_int", "fsc-area", "forward scatter"}
            if channel_lower in fsc_aliases:
                resolved = self.get_fsc_channel()
                strategy = "scatter_alias (fsc)"

            if resolved is None:
                ssc_aliases = {"ssc", "ssc-a", "ssc_a", "ss int", "ss.int", "ss", "ss-int", "ss_int", "ssc-area", "side scatter"}
                if channel_lower in ssc_aliases:
                    resolved = self.get_ssc_channel()
                    strategy = "scatter_alias (ssc)"

            if resolved is None:
                fsc_h_aliases = {"fsc-h", "fsc_h", "fs peak", "fs.peak", "fs-h", "fs_h"}
                if channel_lower in fsc_h_aliases:
                    fsc_h = self.get_fsc_h_channel()
                    if fsc_h:
                        resolved = fsc_h
                        strategy = "scatter_height_alias (fsc-h)"

            if resolved is None:
                ssc_h_aliases = {"ssc-h", "ssc_h", "ss peak", "ss.peak", "ss-h", "ss_h"}
                if channel_lower in ssc_h_aliases:
                    ssc_h = self.get_ssc_h_channel()
                    if ssc_h:
                        resolved = ssc_h
                        strategy = "scatter_height_alias (ssc-h)"

            if resolved is None:
                time_aliases = {"time", "time ms", "time.ms", "time-a"}
                if channel_lower in time_aliases:
                    time_ch = self.get_time_channel()
                    if time_ch:
                        resolved = time_ch
                        strategy = "time_alias"

            # Case-insensitive marker and detector dictionary check
            if resolved is None:
                for marker, det in self.marker_to_detector.items():
                    if marker.lower() == channel_lower:
                        resolved = det
                        strategy = "case_insensitive_marker"
                        break

            if resolved is None:
                for pnn in self.pnn_labels:
                    if pnn.lower() == channel_lower:
                        resolved = pnn
                        strategy = "case_insensitive_detector"
                        break

            # Punctuation / separator normalized check (e.g. HLA-DR vs HLA.DR vs HLADR)
            if resolved is None:
                norm_input = re.sub(r"[^a-zA-Z0-9]", "", channel_lower)
                if norm_input:
                    for marker, det in self.marker_to_detector.items():
                        if re.sub(r"[^a-zA-Z0-9]", "", marker.lower()) == norm_input:
                            resolved = det
                            strategy = "punctuation_normalized_marker"
                            break
                    if resolved is None:
                        for pnn in self.pnn_labels:
                            if re.sub(r"[^a-zA-Z0-9]", "", pnn.lower()) == norm_input:
                                resolved = pnn
                                strategy = "punctuation_normalized_detector"
                                break

            # Marker synonym clusters (e.g. HLA-DR <-> HLA in 10-color acute leukemia panels)
            if resolved is None:
                marker_synonym_clusters = [
                    {"hla-dr", "hla_dr", "hla.dr", "hladr", "hla"},
                    {"cd3", "cd3e", "cd3eps", "cd3-a"},
                    {"cd4", "cd4a", "cd4-a"},
                    {"cd8", "cd8a", "cd8b", "cd8-a"},
                    {"cd16", "cd16a", "cd16b", "fcgr3"},
                    {"cd19", "b4"},
                    {"cd20", "b1"},
                    {"cd45", "lca", "ptprc"},
                    {"cd34", "gp105"},
                    {"cd14", "cd14st"},
                    {"cd133", "prom1"},
                    {"cd117", "c-kit", "ckit"},
                    {"cd33", "siglec-3"},
                    {"dead", "viability", "livedead", "zombie", "aard", "dapi", "7-aad", "pi"},
                ]
                for cluster in marker_synonym_clusters:
                    cluster_norms = {re.sub(r"[^a-zA-Z0-9]", "", s) for s in cluster}
                    if channel_lower in cluster or (norm_input and norm_input in cluster_norms):
                        cluster_name = next(iter(cluster))
                        for syn in cluster:
                            if syn in self.marker_to_detector:
                                resolved = self.marker_to_detector[syn]
                                strategy = f"synonym_cluster ({cluster_name})"
                                break
                            syn_norm = re.sub(r"[^a-zA-Z0-9]", "", syn)
                            for marker, det in self.marker_to_detector.items():
                                if marker.lower() == syn or re.sub(r"[^a-zA-Z0-9]", "", marker.lower()) == syn_norm:
                                    resolved = det
                                    strategy = f"synonym_cluster ({cluster_name})"
                                    break
                            if resolved is not None:
                                break
                            for pnn in self.pnn_labels:
                                if pnn.lower() == syn or re.sub(r"[^a-zA-Z0-9]", "", pnn.lower()) == syn_norm:
                                    resolved = pnn
                                    strategy = f"synonym_cluster ({cluster_name})"
                                    break
                            if resolved is not None:
                                break
                    if resolved is not None:
                        break

            # Synonym alias search fallback
            if resolved is None:
                alias_res = self.find_channel_by_alias([channel])
                if alias_res:
                    resolved = alias_res
                    strategy = "alias_search"

        if resolved is not None:
            if not hasattr(self, "channel_resolutions") or self.channel_resolutions is None:
                self.channel_resolutions = {}
            if not hasattr(self, "resolution_audit_log") or self.resolution_audit_log is None:
                self.resolution_audit_log = []

            self.channel_resolutions[channel] = resolved
            marker_lbl = self.detector_to_marker.get(resolved, "")
            sample_name = getattr(self, "filename", "FCSData")
            audit_entry = {
                "query": channel,
                "resolved_detector": resolved,
                "strategy": strategy,
                "marker_label": marker_lbl,
                "sample": sample_name,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.resolution_audit_log.append(audit_entry)
            logger.debug(
                f"[{sample_name}] Resolved channel query '{channel}' -> '{resolved}' "
                f"(strategy: {strategy}, marker: '{marker_lbl}')"
            )
            return resolved

        logger.debug(
            f"[{getattr(self, 'filename', 'FCSData')}] Channel '{channel}' not found in available detectors: "
            f"{self.pnn_labels} or markers: {list(self.marker_to_detector.keys())}"
        )
        raise KeyError(
            f"Channel '{channel}' not found. Available detectors: {self.pnn_labels}, "
            f"markers: {list(self.marker_to_detector.keys())}"
        )

    def detect_max_range(self) -> float:
        """Detect the instrument scale dynamic range across fluorescent channels.

        Inspects $PnR hardware ranges and event maxima to return the upper scale limit T.
        Recognizes standard cytometer ranges (BD 18-bit: 262,144; Cytek/Beckman 20-bit: 1,048,576 / 1,000,000;
        Sony/Attune 24-bit: 16,777,216; or analog 10-bit: 1,024).

        Returns:
            Estimated maximum value for scaling / Logicle linearization.
        """
        fluoro_ranges = []
        for ch, r in self.channel_ranges.items():
            ch_clean = ch.lower().replace(".", " ").replace("-", " ").replace("_", " ")
            tokens = set(ch_clean.split())
            is_scatter_or_time = bool(tokens.intersection({"fsc", "ssc", "fs", "ss", "time"}))
            if not is_scatter_or_time and r > 100:
                fluoro_ranges.append(r)

        if fluoro_ranges:
            max_r = max(fluoro_ranges)
            if 200000 <= max_r <= 300000:
                return 262144.0
            elif 900000 <= max_r <= 1100000:
                return float(max_r)
            elif 15000000 <= max_r <= 18000000:
                return 16777216.0
            elif 1000 <= max_r <= 1050:
                return 1024.0
            return float(max_r)
        return 262144.0

    def find_channel_by_alias(self, candidates: List[str]) -> Optional[str]:
        """Search detectors and markers against a list of synonym aliases.

        Args:
            candidates: List of alias strings (e.g. ['cd3', 'cd3e', 'cd3-a']).

        Returns:
            Resolved detector name if found, else None.
        """
        cand_lower = [c.lower().strip() for c in candidates]

        import re

        # 1. Exact marker matches
        for m, det in self.marker_to_detector.items():
            if m.lower().strip() in cand_lower:
                return det

        # 2. Exact detector matches
        for pnn in self.pnn_labels:
            if pnn.lower().strip() in cand_lower:
                return pnn

        # 3. Whole-word / token boundary match on markers (e.g. 'cd4' in 'CD4 BUV661', but NOT 'CD45RA')
        for c in cand_lower:
            pattern = re.compile(rf"(?<![a-zA-Z0-9]){re.escape(c)}(?![a-zA-Z0-9])", re.IGNORECASE)
            for m, det in self.marker_to_detector.items():
                if pattern.search(m):
                    return det

        # 4. Whole-word / token boundary match on detectors
        for c in cand_lower:
            pattern = re.compile(rf"(?<![a-zA-Z0-9]){re.escape(c)}(?![a-zA-Z0-9])", re.IGNORECASE)
            for pnn in self.pnn_labels:
                if pattern.search(pnn):
                    return pnn

        # 5. Fallback substring match on markers
        for c in cand_lower:
            for m, det in self.marker_to_detector.items():
                if c in m.lower():
                    return det

        # 6. Fallback substring match on detectors
        for c in cand_lower:
            for pnn in self.pnn_labels:
                if c in pnn.lower():
                    return pnn

        return None


    def get_channel_index(self, channel: str) -> int:
        """Get 0-based column index of a channel.

        Args:
            channel: Detector or marker name.

        Returns:
            Integer column index.
        """
        det = self.resolve_channel_name(channel)
        return self.pnn_labels.index(det)

    def get_channel_data(self, channel: str) -> np.ndarray:
        """Get 1D array of raw values for the specified channel.

        Args:
            channel: Detector or marker name.

        Returns:
            1D NumPy array of event values.
        """
        idx = self.get_channel_index(channel)
        return self.raw_events[:, idx]


def parse_spillover_string(spill_text: str) -> Tuple[List[str], np.ndarray]:
    """Parse FCS spillover text string into detector list and matrix.

    Handles standard FCS $SPILL / $SPILLOVER formats:
    'n, fluoro1, fluoro2, ..., val11, val12, ...'

    Args:
        spill_text: Raw string value from FCS metadata.

    Returns:
        Tuple of (detector_names, spillover_matrix_2d_array).

    Raises:
        ValueError: If format cannot be parsed.
    """
    cleaned = spill_text.strip().replace("\n", ",").replace("\r", ",")
    tokens = [t.strip() for t in cleaned.split(",") if t.strip()]

    if not tokens:
        raise ValueError("Spillover text is empty")

    try:
        n_detectors = int(tokens[0])
    except ValueError as exc:
        raise ValueError(f"Invalid detector count in spillover header: '{tokens[0]}'") from exc

    detectors = tokens[1 : 1 + n_detectors]
    values_raw = tokens[1 + n_detectors : 1 + n_detectors + (n_detectors * n_detectors)]

    if len(values_raw) != n_detectors * n_detectors:
        raise ValueError(
            f"Expected {n_detectors * n_detectors} matrix values for {n_detectors} detectors, "
            f"found {len(values_raw)}"
        )

    matrix_values = np.array([float(x) for x in values_raw], dtype=np.float64)
    matrix = matrix_values.reshape((n_detectors, n_detectors))
    return detectors, matrix


def extract_spillover_matrix(
    metadata: Dict[str, Any], pnn_labels: List[str]
) -> Optional[Tuple[List[str], pd.DataFrame]]:
    """Extract and validate spillover matrix from metadata dictionary.

    Args:
        metadata: Metadata dictionary from FCS file.
        pnn_labels: List of detector labels in sample.

    Returns:
        Tuple of (detectors, spillover_dataframe) or None if no spillover found.
    """
    candidate_keys = ["$spillover", "spillover", "$spill", "spill", "comp", "$comp"]
    spill_key = None
    for k in metadata.keys():
        if k.lower() in candidate_keys:
            spill_key = k
            break

    if not spill_key:
        return None

    raw_val = metadata[spill_key]
    if isinstance(raw_val, str) and "," in raw_val:
        try:
            detectors, matrix = parse_spillover_string(raw_val)
            df = pd.DataFrame(matrix, index=detectors, columns=detectors)
            return detectors, df
        except Exception:
            return None

    return None


def decompose_marker_name(text: str) -> Optional[Tuple[str, str]]:
    """Decompose concatenated detector/stain strings like 'CD14-FITC' or 'CD14.FITC' into marker and dye.

    Args:
        text: Raw detector or stain label.

    Returns:
        Tuple of (marker_name, fluorophore_name) or None if cannot be decomposed.
    """
    import re

    clean = text.strip()
    if not clean:
        return None

    pattern = (
        r"^([A-Za-z0-9_\-\/]+?)[.\-_]"
        r"(FITC|PE|ECD|PC5|PC5\.5|PC7|APC|APC700|APC750|KO|PB|BV\d+|BUV\d+|PerCP.*|Cy\d+|AF\d+|Ax\d+|BB\d+|eFluor.*|V\d+)$"
    )
    m = re.match(pattern, clean, re.IGNORECASE)
    if m:
        marker, fluoro = m.group(1), m.group(2)
        canonical_marker = marker.replace(".", "-")
        return canonical_marker, fluoro
    return None


def parse_fcs(
    filepath: Union[str, Path],
    null_channels: Optional[List[str]] = None,
    marker_mapping: Optional[Dict[str, str]] = None,
) -> FCSData:
    """Parse a single FCS file into an FCSData container.

    Reads raw data without modifying the source file. Extracts events,
    PnN detector labels, PnS marker labels, hardware cytometer specifications,
    scale dynamic ranges ($PnR), full metadata dictionary, and embedded
    spillover compensation matrices.

    Args:
        filepath: Path to the FCS file to parse.
        null_channels: Optional list of channel names to mark as null/unused.
        marker_mapping: Optional manual mapping overriding or supplementing
            marker-to-detector assignments (e.g. {'CD3': 'FL1-A', 'CD4': 'FL2-A'}).

    Returns:
        Populated FCSData container.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file cannot be parsed or contains no event data.
        RuntimeError: If FlowKit fails to read the file.
    """
    path_obj = Path(filepath)
    if not path_obj.is_file():
        raise FileNotFoundError(f"FCS file not found at path: {filepath}")

    try:
        sample = fk.Sample(str(path_obj), null_channel_list=null_channels)
    except Exception as exc:
        raise RuntimeError(f"Error parsing FCS file '{path_obj.name}': {exc}") from exc

    try:
        pnn_labels = list(sample.pnn_labels)
        pns_labels = list(sample.pns_labels)
    except Exception:
        # Fallback if channel attributes differ
        channel_df = sample.channels
        pnn_labels = list(channel_df["pnn"])
        pns_labels = list(channel_df["pns"])

    # Build bidirectional mapping with automated decomposition guardrails
    detector_to_marker: Dict[str, str] = {}
    marker_to_detector: Dict[str, str] = {}
    for i, (pnn, pns) in enumerate(zip(pnn_labels, pns_labels)):
        pns_clean = pns.strip() if isinstance(pns, str) else ""

        decomp = None
        if pns_clean and pns_clean != pnn:
            decomp = decompose_marker_name(pns_clean)
        if decomp is None:
            decomp = decompose_marker_name(pnn)

        if decomp is not None:
            marker_clean, fluoro = decomp
            marker_to_detector[marker_clean] = pnn
            marker_to_detector[marker_clean.replace("-", ".")] = pnn
            marker_to_detector[marker_clean.replace(".", "-")] = pnn
            if pns_clean:
                marker_to_detector[pns_clean] = pnn
            marker_to_detector[pnn] = pnn
            detector_to_marker[pnn] = marker_clean
            if not pns_clean or pns_clean == pnn:
                pns_labels[i] = marker_clean
        else:
            detector_to_marker[pnn] = pns_clean
            if pns_clean:
                marker_to_detector[pns_clean] = pnn

    # Apply manual marker mapping overrides if provided
    if marker_mapping:
        for marker, det_name in marker_mapping.items():
            for pnn in pnn_labels:
                if pnn.lower() == det_name.lower():
                    marker_to_detector[marker] = pnn
                    detector_to_marker[pnn] = marker
                    break

    metadata = sample.get_metadata()
    raw_events = np.array(sample.get_events(source="raw"), copy=True)
    event_count = int(raw_events.shape[0])
    channel_count = int(raw_events.shape[1])

    # Extract cytometer model if present
    cytometer = str(
        metadata.get("cyt", metadata.get("$cyt", metadata.get("cytometer", metadata.get("$CYT", ""))))
    ).strip()

    # Extract channel ranges ($PnR) adjusted for gain ($PnG)
    channel_ranges: Dict[str, float] = {}
    for i, pnn in enumerate(pnn_labels):
        pnr_val = (
            metadata.get(f"p{i+1}r")
            or metadata.get(f"$p{i+1}r")
            or metadata.get(f"$P{i+1}R")
            or metadata.get(f"P{i+1}R")
        )
        png_val = (
            metadata.get(f"p{i+1}g")
            or metadata.get(f"$p{i+1}g")
            or metadata.get(f"$P{i+1}G")
            or metadata.get(f"P{i+1}G")
        )
        gain = 1.0
        if png_val is not None:
            try:
                g_f = float(png_val)
                if g_f > 0.0:
                    gain = g_f
            except (ValueError, TypeError):
                pass
        if pnr_val is not None:
            try:
                channel_ranges[pnn] = float(pnr_val) / gain
            except (ValueError, TypeError):
                pass
        if pnn not in channel_ranges and len(raw_events) > 0:
            col_99 = float(np.nanpercentile(raw_events[:, i], 99.95))
            channel_ranges[pnn] = col_99 if col_99 > 0 else 262144.0

    # Construct DataFrame with detector column names
    events_df = pd.DataFrame(raw_events, columns=pnn_labels)

    # Extract spillover matrix if present
    spill_info = extract_spillover_matrix(metadata, pnn_labels)
    spill_df: Optional[pd.DataFrame] = None
    spill_channels: List[str] = []
    if spill_info is not None:
        spill_channels, spill_df = spill_info

    return FCSData(
        filepath=sanitize_path(path_obj),
        filename=path_obj.name,
        event_count=event_count,
        channel_count=channel_count,
        pnn_labels=pnn_labels,
        pns_labels=pns_labels,
        detector_to_marker=detector_to_marker,
        marker_to_detector=marker_to_detector,
        metadata=metadata,
        raw_events=raw_events,
        events_df=events_df,
        spillover_matrix=spill_df,
        spillover_channels=spill_channels,
        cytometer=cytometer,
        channel_ranges=channel_ranges,
        sample_obj=sample,
    )


def write_fcs(
    output_path: Union[str, Path],
    events_df: pd.DataFrame,
    pnn_labels: Optional[List[str]] = None,
    pns_labels: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    spillover_matrix: Optional[pd.DataFrame] = None,
    channel_ranges: Optional[Dict[str, float]] = None,
    is_compensated: bool = False,
) -> Path:
    """Write an events DataFrame to a standardized FCS file.

    Preserves channel labels ($PnN), marker annotations ($PnS), cytometer metadata,
    and embedded compensation spillover matrix ($SPILLOVER / SPILL) for seamless
    handoffs across multi-step computational pipelines.

    Crucially sanitizes per-channel gain ($PnG) by fixing gain to 1.0, ensuring
    downstream readers like FlowKit do not re-divide already gain-calibrated events.

    Args:
        output_path: Target .fcs file path.
        events_df: DataFrame containing events with column names corresponding to pnn_labels.
        pnn_labels: Channel names ($PnN). Defaults to list(events_df.columns).
        pns_labels: Optional stain/marker names ($PnS).
        metadata: Optional dictionary of TEXT metadata to preserve.
        spillover_matrix: Optional DataFrame representing spillover matrix to embed.
        channel_ranges: Optional dictionary mapping channel name to calibrated range ($PnR).

    Returns:
        Resolved Path to the written FCS file.
    """
    import flowio

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    channels = list(pnn_labels) if pnn_labels is not None else list(events_df.columns)
    stains = (
        list(pns_labels)
        if pns_labels is not None
        else ["" for _ in range(len(channels))]
    )
    if len(stains) < len(channels):
        stains.extend(["" for _ in range(len(channels) - len(stains))])

    # Validate channel columns in DataFrame
    missing_cols = [c for c in channels if c not in events_df.columns]
    if missing_cols:
        raise ValueError(
            f"Events DataFrame missing required channel columns: {missing_cols[:5]}"
        )

    events_array = events_df[channels].to_numpy(dtype=np.float32).flatten()

    # Filter internal FCS keywords recalculated by flowio
    skip_keys = {
        "$tot", "$par", "$datatype", "$byteord", "begindata", "enddata",
        "beginanalysis", "endanalysis", "beginstext", "endstext",
        "tot", "par", "datatype", "byteord", "mode", "$mode",
    }
    meta_dict: Dict[str, str] = {}
    if metadata:
        for k, v in metadata.items():
            k_str = str(k)
            k_norm = k_str.lower().lstrip("$")
            if k_norm in skip_keys or k_str.lower() in skip_keys:
                continue
            # Filter out legacy per-channel metadata ($Pn... / Pn...) and flowcore tags
            # so flowio generates fresh, uncorrupted channel descriptors.
            # Crucially, old per-channel gain ($PnG / PnG) must NEVER be passed through,
            # as events_df already holds calibrated, gain-divided floating-point values.
            # Passing old gain would cause FlowKit to divide by gain AGAIN upon re-reading.
            if (k_norm.startswith("p") and any(c.isdigit() for c in k_norm[:4])) or k_norm.startswith("flowcore_p"):
                continue
            if v is not None:
                meta_dict[k_str] = str(v)

    # Set explicit unit gain ($PnG = 1.0) and accurate channel range ($PnR)
    for i, ch in enumerate(channels, 1):
        meta_dict[f"$P{i}G"] = "1.0"
        cr: Optional[float] = None
        if channel_ranges and ch in channel_ranges:
            cr = float(channel_ranges[ch])
        if cr is None or cr <= 0:
            if ch in events_df.columns and len(events_df[ch]) > 0:
                max_val = float(np.nanmax(events_df[ch]))
                cr = max(262144.0, max_val * 1.05) if max_val > 0 else 262144.0
            else:
                cr = 262144.0
        meta_dict[f"$P{i}R"] = str(int(np.ceil(cr)))

    # Embed spillover matrix if provided or if present in metadata
    if spillover_matrix is not None:
        dets = list(spillover_matrix.index)
        n_dets = len(dets)
        spill_vals = [str(n_dets)] + dets
        for i in range(n_dets):
            for j in range(n_dets):
                spill_vals.append(str(float(spillover_matrix.iloc[i, j])))
        spill_str = ",".join(spill_vals)
        meta_dict["$SPILLOVER"] = spill_str
        meta_dict["SPILL"] = spill_str

    if is_compensated:
        meta_dict["$COMP"] = "COMPENSATED"
        meta_dict["SPILLOVER_APPLIED"] = "TRUE"
        meta_dict["FLOWX_CLEANUP_STAGE"] = "STEP4_CLEANUP"

    with open(out_file, "wb") as fh:
        flowio.create_fcs(
            fh,
            events_array,
            channels,
            opt_channel_names=stains,
            metadata_dict=meta_dict,
        )

    return out_file

