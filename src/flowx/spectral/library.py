"""Reference Spectral Library builder for full-spectrum flow cytometry.

Extracts continuous spectral emission fingerprints across multi-laser optical detector arrays
from single-stained bead or cell controls, normalizes spectral shapes, audits tandem dye integrity,
and assembles overdetermined deconvolution libraries.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from flowx.io.parser import FCSData, parse_fcs
from flowx.math.compensation import detect_control_carrier
from flowx.spectral.models import ReferenceSpectralLibrary, SpectralReference

logger = logging.getLogger(__name__)


def is_optical_fluorescence_channel(channel_name: str) -> bool:
    """Determine whether a channel represents an optical fluorescence detector.

    Excludes forward scatter, side scatter, and time channels across all cytometer vendors.

    Args:
        channel_name: Channel name string (e.g. 'UV1-A', 'B525-A', 'FSC-A', 'Time').

    Returns:
        True if the channel is an optical fluorescence detector, False otherwise.
    """
    ch_low = channel_name.lower().strip()
    non_optical = [
        "fsc",
        "ssc",
        "time",
        "width",
        "fs int",
        "ss int",
        "fs peak",
        "ss peak",
        "fs tof",
        "ss tof",
    ]
    for non_opt in non_optical:
        if ch_low == non_opt or ch_low.startswith(non_opt + "-") or ch_low.startswith(non_opt + "_"):
            return False
        # Catch specific formats like FSC-A, SSC-H, FSC_W
        if re.match(rf"^{non_opt}[\-_.]?[ahw]?$", ch_low):
            return False

    # Violet Side Scatter (V-SSC / SSC-B / VSSC) should be treated as scatter rather than fluorescence
    if any(k in ch_low for k in ["ssc-b", "v-ssc", "vssc", "ssc 405", "ssc-v", "ssc_b"]):
        return False

    return True


def filter_optical_channels(all_channels: List[str]) -> List[str]:
    """Filter channel list to include only optical fluorescence detector channels."""
    return [ch for ch in all_channels if is_optical_fluorescence_channel(ch)]


def extract_single_stain_spectrum(
    fcs_input: Union[str, Path, FCSData],
    fluorophore_name: Optional[str] = None,
    optical_channels: Optional[List[str]] = None,
    carrier: str = "auto",
    target_detector: Optional[str] = None,
    min_events: int = 200,
    random_seed: int = 42,
) -> SpectralReference:
    """Extract normalized reference emission spectrum from a single-stained control.

    Args:
        fcs_input: Filepath to FCS file or parsed FCSData instance.
        fluorophore_name: Optional explicit name of the fluorophore.
        optical_channels: Optional list of optical detector channels. If None, auto-detected.
        carrier: 'auto', 'beads', or 'cells'.
        target_detector: Optional target primary detector channel.
        min_events: Minimum clean events required.
        random_seed: Random seed for GMM clustering.

    Returns:
        SpectralReference instance containing normalized spectral fingerprint and metadata.
    """
    if isinstance(fcs_input, (str, Path)):
        fcs_data = parse_fcs(fcs_input)
        fpath_str = str(fcs_input)
    else:
        fcs_data = fcs_input
        fpath_str = getattr(fcs_data, "filepath", "")

    if optical_channels is None:
        optical_channels = filter_optical_channels(fcs_data.pnn_labels)

    if not optical_channels:
        raise ValueError(f"No optical fluorescence channels identified in {fpath_str}")

    df_events = fcs_data.events_df
    if len(df_events) < min_events:
        raise ValueError(
            f"Control '{Path(fpath_str).name}' has insufficient events ({len(df_events)} < {min_events})"
        )

    # Carrier detection
    resolved_carrier = carrier
    if carrier == "auto":
        fname_lower = Path(fpath_str).name.lower()
        if "bead" in fname_lower:
            resolved_carrier = "beads"
        elif "cell" in fname_lower:
            resolved_carrier = "cells"
        else:
            resolved_carrier = detect_control_carrier(df_events, optical_channels[0])

    # Debris, doublet, and saturation pre-gating conforming to MIFlowCyt standards
    clean_df = df_events
    try:
        from flowx.math.compensation import _pregate_control_scatter_and_singlets
        channel_ranges = getattr(fcs_data, "channel_ranges", None)
        non_sat, debris_free, singlets, _ = _pregate_control_scatter_and_singlets(
            df_events,
            resolved_carrier=resolved_carrier,
            all_channels=optical_channels,
            channel_ranges=channel_ranges,
            random_seed=random_seed,
        )
        if np.sum(singlets) >= min_events:
            clean_df = df_events.loc[singlets]
        elif np.sum(debris_free) >= min_events:
            clean_df = df_events.loc[debris_free]
        elif np.sum(non_sat) >= min_events:
            clean_df = df_events.loc[non_sat]
    except Exception as exc:
        logger.debug("Falling back to scatter percentile filtering: %s", exc)
        fsc_candidates = [c for c in df_events.columns if c.upper().startswith("FSC")]
        if fsc_candidates:
            fsc_col = fsc_candidates[0]
            fsc_vals = df_events[fsc_col].to_numpy(dtype=np.float64)
            p05 = float(np.percentile(fsc_vals, 5.0))
            p99 = float(np.percentile(fsc_vals, 99.5))
            debris_mask = (fsc_vals >= p05) & (fsc_vals <= p99)
            if np.sum(debris_mask) >= min_events:
                clean_df = df_events.loc[debris_mask]

    # Resolve fluorophore name if not provided
    inferred_name = fluorophore_name
    if not inferred_name:
        stem = Path(fpath_str).stem
        # Strip generic prefixes like 'Compensation Specimen' or 'Reference Group'
        clean_stem = re.sub(r"(?i)compensation\s*specimen_?", "", stem)
        clean_stem = re.sub(r"(?i)reference\s*group[-_]?", "", clean_stem)
        clean_stem = re.sub(r"(?i)sample[-_]?", "", clean_stem)

        # Look for standard fluorophore tokens with strict word boundaries
        pattern = re.compile(
            r"(?<![A-Za-z])("
            r"BUV\d+|BV\d+|BB\d+|FITC|"
            r"PE[-_\s]?(?:Cy[57]|Dazzle(?:\s*594)?|AF\d+|Fire\s*\d+|eFluor\s*\d+|TexasRed)?|PE|"
            r"PerCP[-_\s]?(?:eFluor\s*\d+|Cy5\.?5)?|"
            r"APC[-_\s]?(?:Cy7|Fire\s*\d+|H7|R700|eFluor\s*\d+)?|APC|"
            r"AF\d+|Alexa\s*Fluor\s*\d+|"
            r"Real\s*Blue(?:\s*RB)?\s*\d+|RB\d+|"
            r"Real\s*Yellow(?:\s*RY)?\s*\d+|RY\d+|"
            r"Spark\s*(?:UV|NIR|Blue|Violet)?\s*\d+|"
            r"StarBright\s*(?:Violet|Blue|Yellow)?\s*\d+|"
            r"Live[-_\s]?(?:or[-_\s]?dye(?:_TM)?|DEAD|Dead)?|"
            r"NFB\s*[A-Za-z0-9_]+|JF\s*\d+|V500|Qdot\d+"
            r")(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        m = pattern.search(clean_stem)
        if m:
            inferred_name = m.group(1).strip().replace(" ", "_")
        else:
            inferred_name = clean_stem
            get_logger().warning(
                "Could not identify a standard fluorophore from filename '%s'. "
                "Defaulting to full sample stem '%s'. To ensure exact channel mapping, provide an explicit control manifest (comp_manifest).",
                clean_stem,
                clean_stem,
            )

    # Extract optical event matrix
    opt_matrix = clean_df[optical_channels].to_numpy(dtype=np.float64)

    # Find peak detector channel
    if target_detector and target_detector in optical_channels:
        peak_idx = optical_channels.index(target_detector)
    else:
        # Use robust spread (P85 - P15) to prevent outlier saturation spikes from capturing the peak
        p85s = np.percentile(opt_matrix, 85.0, axis=0)
        p15s = np.percentile(opt_matrix, 15.0, axis=0)
        spread = p85s - p15s
        peak_idx = int(np.argmax(spread))

    peak_channel = optical_channels[peak_idx]
    peak_vals = opt_matrix[:, peak_idx]

    # Split positive vs negative populations using GMM or robust percentile bisection
    pos_mask = np.zeros(len(clean_df), dtype=bool)
    neg_mask = np.zeros(len(clean_df), dtype=bool)

    finite_mask = np.isfinite(peak_vals)
    valid_vals = peak_vals[finite_mask]

    try:
        gmm = GaussianMixture(n_components=2, random_state=random_seed)
        labels = gmm.fit_predict(valid_vals.reshape(-1, 1))
        means = gmm.means_.ravel()
        high_component = int(np.argmax(means))
        low_component = int(np.argmin(means))

        full_labels = np.full(len(peak_vals), -1, dtype=int)
        full_labels[finite_mask] = labels

        pos_mask = full_labels == high_component
        neg_mask = full_labels == low_component

        # Quality check: positive population must have higher median than negative
        if np.sum(pos_mask) < 20 or np.sum(neg_mask) < 20:
            raise ValueError("GMM clusters too small")
    except Exception:
        # Fallback: top 25% vs bottom 25% on peak channel
        p75 = float(np.percentile(valid_vals, 75.0))
        p25 = float(np.percentile(valid_vals, 25.0))
        pos_mask = peak_vals >= p75
        neg_mask = peak_vals <= p25

    pos_mfi = np.median(opt_matrix[pos_mask], axis=0)
    neg_mfi = np.median(opt_matrix[neg_mask], axis=0)

    delta_mfi = pos_mfi - neg_mfi
    # Clamp negative baseline noise in secondary channels
    delta_mfi = np.maximum(delta_mfi, 0.0)

    max_delta = float(np.max(delta_mfi))
    if max_delta <= 1e-6:
        # Avoid zero division
        normalized = np.zeros_like(delta_mfi)
        normalized[peak_idx] = 1.0
    else:
        normalized = delta_mfi / max_delta

    # Audit tandem dye ratio if tandem fluorophore
    tandem_ratio = None
    if any(td in inferred_name.lower() for td in ["cy5", "cy7", "dazzle", "cf594", "fire"]):
        # Find second highest peak distinct from primary
        sorted_indices = np.argsort(normalized)[::-1]
        for s_idx in sorted_indices:
            if s_idx != peak_idx and normalized[s_idx] > 0.15:
                tandem_ratio = float(normalized[s_idx])
                break

    return SpectralReference(
        fluorophore=inferred_name,
        carrier=resolved_carrier,
        channel_names=optical_channels,
        normalized_spectrum=normalized,
        raw_positive_mfi=pos_mfi,
        raw_negative_mfi=neg_mfi,
        peak_channel=peak_channel,
        is_autofluorescence=False,
        tandem_ratio=tandem_ratio,
        file_path=fpath_str,
    )


class ReferenceSpectralLibraryBuilder:
    """Builder for constructing multi-parameter ReferenceSpectralLibrary instances."""

    def __init__(self, optical_channels: Optional[List[str]] = None) -> None:
        """Initialize library builder."""
        self.optical_channels = list(optical_channels) if optical_channels else []
        self.references: Dict[str, SpectralReference] = {}

    def add_reference(self, reference: SpectralReference) -> ReferenceSpectralLibraryBuilder:
        """Add an extracted SpectralReference object to the library."""
        if not self.optical_channels:
            self.optical_channels = list(reference.channel_names)
        elif self.optical_channels != reference.channel_names:
            raise ValueError(
                f"Detector channel mismatch for reference '{reference.fluorophore}'. "
                f"Expected {len(self.optical_channels)} channels, got {len(reference.channel_names)}"
            )

        fluoro_key = reference.fluorophore
        if fluoro_key in self.references:
            idx = 2
            while f"{fluoro_key}_{idx}" in self.references:
                idx += 1
            fluoro_key = f"{fluoro_key}_{idx}"
            reference.fluorophore = fluoro_key

        self.references[fluoro_key] = reference
        return self

    def add_control_file(
        self,
        fcs_file: Union[str, Path, FCSData],
        fluorophore_name: Optional[str] = None,
        carrier: str = "auto",
        target_detector: Optional[str] = None,
    ) -> ReferenceSpectralLibraryBuilder:
        """Extract and add a spectral reference directly from a control FCS file."""
        ref = extract_single_stain_spectrum(
            fcs_input=fcs_file,
            fluorophore_name=fluorophore_name,
            optical_channels=self.optical_channels or None,
            carrier=carrier,
            target_detector=target_detector,
        )
        return self.add_reference(ref)

    def add_controls_directory(
        self,
        directory: Union[str, Path],
        file_mapping: Optional[Dict[str, str]] = None,
    ) -> ReferenceSpectralLibraryBuilder:
        """Batch-extract references from a directory of single-stained control FCS files."""
        dir_p = Path(directory)
        fcs_files = sorted(list(dir_p.glob("*.fcs")))
        if not fcs_files:
            fcs_files = sorted(list(dir_p.rglob("*.fcs")))

        for f_path in fcs_files:
            # Skip unstained or negative controls (handled by autofluorescence module)
            if any(w in f_path.name.lower() for w in ["unstained", "negative", "blank"]):
                continue

            fluoro_name = None
            if file_mapping and f_path.name in file_mapping:
                fluoro_name = file_mapping[f_path.name]

            try:
                self.add_control_file(f_path, fluorophore_name=fluoro_name)
            except Exception as exc:
                logger.warning("Could not extract reference from %s: %s", f_path.name, exc)

        return self

    def build(self) -> ReferenceSpectralLibrary:
        """Assemble all accumulated references into a ReferenceSpectralLibrary."""
        if not self.references:
            raise ValueError("Cannot build an empty ReferenceSpectralLibrary")

        fluorophores = list(self.references.keys())
        detectors = self.optical_channels

        matrix = np.zeros((len(detectors), len(fluorophores)), dtype=np.float64)
        for j, fluoro in enumerate(fluorophores):
            matrix[:, j] = self.references[fluoro].normalized_spectrum

        df = pd.DataFrame(matrix, index=detectors, columns=fluorophores)

        return ReferenceSpectralLibrary(
            detectors=detectors,
            fluorophores=fluorophores,
            matrix=matrix,
            dataframe=df,
            references=dict(self.references),
        )


def extract_embedded_unmixing_matrix(fcs_data: FCSData) -> Optional[Any]:
    """Extract embedded unmixing/spillover matrix from FCS metadata if present.

    Parses $SPILLOVER, SPILL, or $COMP keywords and builds a CompensationMatrix.

    Args:
        fcs_data: Parsed FCSData object.

    Returns:
        CompensationMatrix if matrix is found in metadata, None otherwise.
    """
    if fcs_data.spillover_matrix is not None and fcs_data.spillover_channels:
        from flowx.math.compensation import CompensationMatrix
        return CompensationMatrix(
            detectors=fcs_data.spillover_channels,
            matrix=fcs_data.spillover_matrix,
        )
    return None

