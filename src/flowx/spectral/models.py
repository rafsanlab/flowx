"""Data models and configuration containers for spectral flow cytometry unmixing.

Provides structured dataclasses for reference spectral signatures, spectral libraries,
unmixing configurations, deconvolution results, and panel quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class SpectralReference:
    """Represents the continuous emission spectrum of a single fluorophore or autofluorescence.

    Attributes:
        fluorophore: Name of the fluorophore or autofluorescence profile (e.g. 'BV421', 'AF_Lymphocytes').
        carrier: Carrier type used ('beads', 'cells', or 'biological_sample').
        channel_names: List of detector channels spanning the measured optical spectrum.
        normalized_spectrum: 1D NumPy array of shape (D,) normalized to peak intensity = 1.0.
        raw_positive_mfi: 1D NumPy array of shape (D,) of raw median intensities in the positive population.
        raw_negative_mfi: 1D NumPy array of shape (D,) of raw median intensities in the negative population.
        peak_channel: Name of the detector channel exhibiting maximum emission intensity.
        is_autofluorescence: True if this reference represents unstained cellular autofluorescence.
        tandem_ratio: Optional ratio of secondary/acceptor peak to primary emission for tandem dyes.
        file_path: Optional path to the source control FCS file.
    """

    fluorophore: str
    carrier: str
    channel_names: List[str]
    normalized_spectrum: np.ndarray
    raw_positive_mfi: np.ndarray
    raw_negative_mfi: np.ndarray
    peak_channel: str
    is_autofluorescence: bool = False
    tandem_ratio: Optional[float] = None
    file_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate spectrum dimensions."""
        if len(self.channel_names) != len(self.normalized_spectrum):
            raise ValueError(
                f"Channel count ({len(self.channel_names)}) does not match "
                f"normalized spectrum length ({len(self.normalized_spectrum)})"
            )


@dataclass
class ReferenceSpectralLibrary:
    """Overdetermined spectral reference library (D detectors x (F fluorophores + K autofluorescence)).

    Attributes:
        detectors: List of D detector channel names (rows).
        fluorophores: List of P fluorophore and autofluorescence labels (columns).
        matrix: 2D NumPy array of shape (D, P) with normalized emission profiles.
        dataframe: Pandas DataFrame representation with index=detectors, columns=fluorophores.
        references: Mapping from fluorophore name to individual SpectralReference objects.
    """

    detectors: List[str]
    fluorophores: List[str]
    matrix: np.ndarray
    dataframe: pd.DataFrame
    references: Dict[str, SpectralReference] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate matrix shape matches detectors and fluorophores."""
        mat_arr = np.asarray(self.matrix, dtype=np.float64)
        if mat_arr.ndim != 2:
            raise ValueError(f"Spectral library matrix must be 2D, got ndim={mat_arr.ndim}")
        if mat_arr.shape[0] != len(self.detectors):
            raise ValueError(
                f"Matrix rows ({mat_arr.shape[0]}) do not match detector count ({len(self.detectors)})"
            )
        if mat_arr.shape[1] != len(self.fluorophores):
            raise ValueError(
                f"Matrix columns ({mat_arr.shape[1]}) do not match fluorophore count ({len(self.fluorophores)})"
            )
        self.matrix = mat_arr
        if self.dataframe is None or self.dataframe.empty:
            self.dataframe = pd.DataFrame(
                self.matrix, index=self.detectors, columns=self.fluorophores
            )

    @property
    def condition_number(self) -> float:
        """Condition number kappa(M) of the spectral reference matrix."""
        if self.matrix.size == 0:
            return 1.0
        return float(np.linalg.cond(self.matrix))

    @property
    def detectors_count(self) -> int:
        """Number of optical detector channels D."""
        return len(self.detectors)

    @property
    def fluorophores_count(self) -> int:
        """Number of unmixed fluorophores and autofluorescence components P."""
        return len(self.fluorophores)

    def get_spectrum(self, fluorophore: str) -> np.ndarray:
        """Retrieve 1D normalized spectrum vector for a fluorophore."""
        if fluorophore not in self.dataframe.columns:
            raise KeyError(f"Fluorophore '{fluorophore}' not in spectral library")
        return self.dataframe[fluorophore].to_numpy(dtype=np.float64)

    def get_peak_detector_map(self) -> Dict[str, str]:
        """Return a mapping from fluorophore/AF name to primary peak optical detector channel."""
        mapping: Dict[str, str] = {}
        for f_name, ref in self.references.items():
            if getattr(ref, "peak_channel", None):
                mapping[f_name] = ref.peak_channel
        for f_name in self.fluorophores:
            if f_name not in mapping and f_name in self.dataframe.columns:
                peak_det = self.dataframe[f_name].idxmax()
                mapping[f_name] = str(peak_det)
        return mapping

    def compute_unmixing_dataframe(self) -> pd.DataFrame:
        """Compute linear pseudo-inverse unmixing matrix U = (M^T M)^(-1) M^T.

        Returns:
            DataFrame of shape (P_fluorophores, D_detectors) mapping detector signals
            to unmixed fluorophore and autofluorescence abundances.
        """
        mat = self.matrix
        if mat.size == 0:
            return pd.DataFrame(index=self.fluorophores, columns=self.detectors, dtype=np.float64)
        try:
            gram = mat.T @ mat
            unmix_mat = np.linalg.solve(gram, mat.T)
        except np.linalg.LinAlgError:
            unmix_mat = np.linalg.pinv(mat)
        return pd.DataFrame(unmix_mat, index=self.fluorophores, columns=self.detectors)

    def compute_detector_spillover_dataframe(self) -> Optional[pd.DataFrame]:
        """Build a square detector-indexed spillover matrix for biological fluorophores.

        Maps each non-autofluorescence reference to its peak optical detector channel
        when peak detectors are unique, aligning spectral single-stain controls with
        conventional FCS $SPILLOVER matrices for downstream comparison.

        Returns:
            Square DataFrame (detectors x detectors) with 1.0 on diagonal, or None if
            peak detectors are not unique or fewer than 2 biological markers exist.
        """
        bio_refs = [
            (f_name, self.references[f_name])
            for f_name in self.fluorophores
            if f_name in self.references and not getattr(self.references[f_name], "is_autofluorescence", False)
        ]
        if len(bio_refs) < 2:
            return None

        mapped_detectors: List[str] = []
        for f_name, ref in bio_refs:
            det_match: Optional[str] = None
            if getattr(ref, "file_path", None):
                fname_low = Path(ref.file_path).name.lower()
                for det in self.detectors:
                    det_stem = det.lower().replace("-a", "").replace("_a", "")
                    if det.lower() in fname_low or (len(det_stem) >= 3 and det_stem in fname_low):
                        det_match = det
                        break
            mapped_detectors.append(det_match or ref.peak_channel)

        if len(set(mapped_detectors)) != len(mapped_detectors):
            return None
        if not all(det in self.dataframe.index for det in mapped_detectors):
            return None

        # Columns are fluorophores (one per nominal detector), rows are nominal detectors
        bio_names = [f_name for f_name, _ in bio_refs]
        sub = self.dataframe.loc[mapped_detectors, bio_names].to_numpy(dtype=np.float64)
        # Normalize each column by its diagonal (nominal detector) so diagonal == 1.0
        diag = np.diag(sub)
        if np.any(np.abs(diag) < 1e-12):
            return None
        spill = (sub / diag[np.newaxis, :]).T  # FCS convention: rows=source, cols=detector
        return pd.DataFrame(spill, index=mapped_detectors, columns=mapped_detectors)


@dataclass
class UnmixingConfig:
    """User-customizable configuration parameters for spectral deconvolution.

    Attributes:
        method: Unmixing solver:
            - 'wls' (default): Weighted Least Squares with Poisson photon counting noise model.
            - 'ols': Ordinary Least Squares deconvolution (M^T M)^(-1) M^T y.
            - 'nnls': Non-Negative Least Squares / regularized bounded solver.
        alpha_poisson: Poisson variance scaling coefficient (sigma_d^2 = alpha * y_d + sigma_noise^2).
        electronic_noise: Baseline electronic/instrument read noise variance floor.
        allow_negative_slack: If True, preserves bounded negative fluctuations around zero.
        negative_slack_floor: Minimum allowable abundance for negative background noise (default: -1000.0).
        compute_residuals: If True, computes per-event unmixing residual norm ||y - M x||_2.
        residual_outlier_sigma: Number of robust standard deviations above median residual to flag outliers.
        autofluorescence_mode: Cellular AF extraction mode ('single', 'multi', or 'none').
        multi_af_components: Number of distinct cellular AF subpopulations to extract if mode='multi' (default: 2).
        min_control_events: Minimum clean events required in single-stain control files.
        random_seed: Random seed for reproducible GMM subpopulation clustering.
    """

    method: str = "wls"
    alpha_poisson: float = 1.0
    electronic_noise: float = 50.0
    allow_negative_slack: bool = True
    negative_slack_floor: float = -1000.0
    compute_residuals: bool = True
    residual_outlier_sigma: float = 3.5
    autofluorescence_mode: str = "single"
    multi_af_components: int = 2
    min_control_events: int = 300
    random_seed: int = 42


@dataclass
class UnmixingResult:
    """Results and diagnostics of spectral unmixing deconvolution.

    Attributes:
        unmixed_df: DataFrame of unmixed fluorophore abundances and AF channels.
        residuals_norm: 1D NumPy array of per-event residual norms ||y_i - M x_i||_2.
        residual_channel_name: Name of the added residual quality channel (e.g. 'Unmixing_Residual').
        spectral_library: The ReferenceSpectralLibrary instance used for deconvolution.
        method_used: The mathematical solver executed ('wls', 'ols', 'nnls').
        fluorophore_channels: List of unmixed biological fluorophore/marker channel names.
        af_channels: List of extracted cellular autofluorescence channel names.
        scatter_and_time_channels: List of preserved non-fluorescent channels (FSC, SSC, Time).
        mean_residual_norm: Mean unmixing residual across all analyzed events.
        median_residual_norm: Median unmixing residual across all analyzed events.
        residual_outliers_count: Number of events exceeding the residual outlier threshold.
        execution_time_seconds: Runtime elapsed during deconvolution in seconds.
    """

    unmixed_df: pd.DataFrame
    residuals_norm: np.ndarray
    spectral_library: ReferenceSpectralLibrary
    method_used: str = "wls"
    residual_channel_name: str = "Unmixing_Residual"
    fluorophore_channels: List[str] = field(default_factory=list)
    af_channels: List[str] = field(default_factory=list)
    scatter_and_time_channels: List[str] = field(default_factory=list)
    mean_residual_norm: float = 0.0
    median_residual_norm: float = 0.0
    residual_outliers_count: int = 0
    execution_time_seconds: float = 0.0


@dataclass
class SpectralMetricsResult:
    """Panel health and quality metrics for spectral flow cytometry.

    Attributes:
        similarity_matrix: DataFrame of shape (P, P) with pairwise Cosine Spectral Similarities.
        complexity_index: Total panel complexity index Tr((M^T M)^(-1)).
        condition_number: Matrix condition number kappa(M).
        high_similarity_pairs: List of (dye1, dye2, similarity_value) exceeding warning threshold (>= 0.85).
        ssm_dataframe: Post-unmixing Spillover Spreading Matrix (SSM) DataFrame.
    """

    similarity_matrix: pd.DataFrame
    complexity_index: float
    condition_number: float
    high_similarity_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    ssm_dataframe: Optional[pd.DataFrame] = None

