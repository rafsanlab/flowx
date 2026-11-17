"""Cellular Autofluorescence (AF) Profiler and Extractor.

Extracts single-AF baseline profiles and multi-AF subpopulation signatures from unstained
biological specimens to separate natural cellular autofluorescence from antibody fluorophores
during spectral deconvolution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from flowx.io.parser import FCSData, parse_fcs
from flowx.spectral.library import filter_optical_channels
from flowx.spectral.models import ReferenceSpectralLibrary, SpectralReference

logger = logging.getLogger(__name__)


def extract_autofluorescence_spectrum(
    unstained_input: Union[str, Path, FCSData],
    label: str = "Autofluorescence",
    optical_channels: Optional[List[str]] = None,
    min_events: int = 200,
) -> SpectralReference:
    """Extract a single baseline cellular autofluorescence spectrum from an unstained control.

    Args:
        unstained_input: Path to unstained FCS file or parsed FCSData instance.
        label: Channel name for the extracted autofluorescence (default: 'Autofluorescence').
        optical_channels: List of optical detector channels. If None, auto-detected.
        min_events: Minimum clean events required.

    Returns:
        SpectralReference representing the continuous cellular autofluorescence profile.
    """
    if isinstance(unstained_input, (str, Path)):
        fcs_data = parse_fcs(unstained_input)
        fpath_str = str(unstained_input)
    else:
        fcs_data = fcs_input = unstained_input
        fpath_str = getattr(fcs_data, "filepath", "")

    if optical_channels is None:
        optical_channels = filter_optical_channels(fcs_data.pnn_labels)

    df_events = fcs_data.events_df
    if len(df_events) < min_events:
        raise ValueError(
            f"Unstained control '{Path(fpath_str).name}' has too few events ({len(df_events)} < {min_events})"
        )

    # Debris removal on FSC if available
    fsc_candidates = [c for c in df_events.columns if c.upper().startswith("FSC")]
    clean_df = df_events
    if fsc_candidates:
        fsc_col = fsc_candidates[0]
        fsc_vals = df_events[fsc_col].to_numpy(dtype=np.float64)
        debris_mask = fsc_vals >= float(np.percentile(fsc_vals, 5.0))
        if np.sum(debris_mask) >= min_events:
            clean_df = df_events.loc[debris_mask]

    opt_matrix = clean_df[optical_channels].to_numpy(dtype=np.float64)
    med_spectrum = np.median(opt_matrix, axis=0)

    # Clamp baseline negative noise
    med_spectrum = np.maximum(med_spectrum, 0.0)

    peak_idx = int(np.argmax(med_spectrum))
    peak_channel = optical_channels[peak_idx]
    max_val = float(med_spectrum[peak_idx])

    if max_val <= 1e-6:
        normalized = np.ones(len(optical_channels), dtype=np.float64) / len(optical_channels)
    else:
        normalized = med_spectrum / max_val

    return SpectralReference(
        fluorophore=label,
        carrier="cells",
        channel_names=optical_channels,
        normalized_spectrum=normalized,
        raw_positive_mfi=med_spectrum,
        raw_negative_mfi=np.zeros_like(med_spectrum),
        peak_channel=peak_channel,
        is_autofluorescence=True,
        tandem_ratio=None,
        file_path=fpath_str,
    )


def extract_multi_autofluorescence_spectra(
    unstained_input: Union[str, Path, FCSData],
    n_components: int = 2,
    prefix: str = "AF",
    optical_channels: Optional[List[str]] = None,
    min_events: int = 500,
    random_seed: int = 42,
) -> List[SpectralReference]:
    """Extract multiple distinct cellular autofluorescence spectra from heterogeneous tissue samples.

    Uses Gaussian Mixture Modeling on optical scatter and autofluorescent channels to segment
    distinct biological lineages (e.g. lymphocytes vs. myeloid/macrophages) and derive their
    unique emission fingerprints.

    Args:
        unstained_input: Path to unstained FCS file or parsed FCSData instance.
        n_components: Number of distinct AF signatures to deconvolve (typically 2 or 3).
        prefix: Prefix for generated AF channel names (e.g. 'AF_1', 'AF_2').
        optical_channels: List of optical detector channels. If None, auto-detected.
        min_events: Minimum clean events required.
        random_seed: Random seed for GMM clustering.

    Returns:
        List of SpectralReference objects corresponding to each autofluorescent subpopulation.
    """
    if isinstance(unstained_input, (str, Path)):
        fcs_data = parse_fcs(unstained_input)
        fpath_str = str(unstained_input)
    else:
        fcs_data = unstained_input
        fpath_str = getattr(fcs_data, "filepath", "")

    if optical_channels is None:
        optical_channels = filter_optical_channels(fcs_data.pnn_labels)

    df_events = fcs_data.events_df
    if len(df_events) < max(min_events, n_components * 100):
        # Fall back to single AF if too few events for multi-component clustering
        logger.warning(
            "Insufficient events (%d) for %d-component multi-AF clustering; falling back to single AF",
            len(df_events),
            n_components,
        )
        return [extract_autofluorescence_spectrum(fcs_data, label=f"{prefix}_1", optical_channels=optical_channels)]

    opt_matrix = df_events[optical_channels].to_numpy(dtype=np.float64)

    # Use PCA or top variance channels for fast GMM clustering
    std_per_ch = np.std(opt_matrix, axis=0)
    top_indices = np.argsort(std_per_ch)[::-1][:min(5, len(optical_channels))]
    features = opt_matrix[:, top_indices]

    gmm = GaussianMixture(n_components=n_components, random_state=random_seed, max_iter=100)
    labels = gmm.fit_predict(features)

    af_references: List[SpectralReference] = []

    for comp_idx in range(n_components):
        comp_mask = labels == comp_idx
        if np.sum(comp_mask) < 50:
            continue

        comp_matrix = opt_matrix[comp_mask]
        med_spectrum = np.median(comp_matrix, axis=0)
        med_spectrum = np.maximum(med_spectrum, 0.0)

        peak_idx = int(np.argmax(med_spectrum))
        peak_channel = optical_channels[peak_idx]
        max_val = float(med_spectrum[peak_idx])

        if max_val <= 1e-6:
            normalized = np.ones(len(optical_channels), dtype=np.float64) / len(optical_channels)
        else:
            normalized = med_spectrum / max_val

        label = f"{prefix}_{comp_idx + 1}"
        ref = SpectralReference(
            fluorophore=label,
            carrier="cells",
            channel_names=optical_channels,
            normalized_spectrum=normalized,
            raw_positive_mfi=med_spectrum,
            raw_negative_mfi=np.zeros_like(med_spectrum),
            peak_channel=peak_channel,
            is_autofluorescence=True,
            tandem_ratio=None,
            file_path=fpath_str,
        )
        af_references.append(ref)

    if not af_references:
        af_references = [
            extract_autofluorescence_spectrum(
                fcs_data, label=f"{prefix}_1", optical_channels=optical_channels
            )
        ]

    return af_references


def attach_autofluorescence_to_library(
    library: ReferenceSpectralLibrary,
    af_references: Union[SpectralReference, List[SpectralReference]],
) -> ReferenceSpectralLibrary:
    """Attach one or more autofluorescence reference vectors to an existing spectral library.

    Args:
        library: Existing ReferenceSpectralLibrary.
        af_references: Single SpectralReference or list of SpectralReferences.

    Returns:
        New updated ReferenceSpectralLibrary including the autofluorescence columns.
    """
    if isinstance(af_references, SpectralReference):
        af_list = [af_references]
    else:
        af_list = list(af_references)

    new_fluorophores = list(library.fluorophores)
    new_refs = dict(library.references)

    for af_ref in af_list:
        if af_ref.fluorophore not in new_fluorophores:
            new_fluorophores.append(af_ref.fluorophore)
        new_refs[af_ref.fluorophore] = af_ref

    detectors = library.detectors
    matrix = np.zeros((len(detectors), len(new_fluorophores)), dtype=np.float64)

    for j, fluoro in enumerate(new_fluorophores):
        matrix[:, j] = new_refs[fluoro].normalized_spectrum

    df = pd.DataFrame(matrix, index=detectors, columns=new_fluorophores)

    return ReferenceSpectralLibrary(
        detectors=detectors,
        fluorophores=new_fluorophores,
        matrix=matrix,
        dataframe=df,
        references=new_refs,
    )

