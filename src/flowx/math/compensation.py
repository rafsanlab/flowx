"""Flow cytometry fluorescence compensation module.

Provides functions and classes to extract spillover matrices, calculate spillover
from single-stained bead or cell controls (via Median Delta-MFI, AutoSpill
linear-space iterative regression, or Hybrid estimation), integrate with Time QC,
compute the Spillover Spreading Matrix (SSM), verify post-compensation residual
alignment across all N x (N-1) channel pairs, and apply linear algebra
compensation to multi-parameter event arrays.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import flowutils.compensate as comp_utils
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


@dataclass
class CompensationConfig:
    """User-customizable configuration parameters for fluorescence compensation.

    Attributes:
        source: Matrix source ('auto', 'controls', 'embedded', 'csv', 'none').
        estimation_method: Spillover estimator for single-stained controls:
            - 'hybrid' (default): Uses Median Delta-MFI for bimodal bead controls and
              AutoSpill robust linear regression for cell/skewed controls, followed by
              iterative linear residual refinement.
            - 'median_delta_mfi': Traditional (Pos_MFI - Neg_MFI) ratio with automatic
              bead vs. cell carrier adaptation and scatter-matched negative gating.
            - 'autospill_regression': Iterative robust linear-space regression slope
              estimation (Roca et al., Nature Communications 2021).
        apply_time_qc_to_controls: If True, executes Time QC (flow rate + margin saturation)
            on every single-stained control before estimating spillover.
        apply_time_qc_to_sample: If True, executes Time QC on the biological sample before
            applying compensation and rendering verification diagnostics.
        auto_scatter_singlet_gate: If True, gates singlets (FSC-A vs FSC-H) and homogeneous
            scatter (FSC-A vs SSC-A) on controls prior to fluorescence gating.
        control_carrier_type: 'auto' (auto-detect 'beads' vs 'cells' per control),
            'beads', 'cells', or a Dict mapping channel name to 'beads'/'cells'.
        negative_reference: 'auto' (internal scatter-matched negative unless unstained_control
            provided), 'internal', or 'unstained_file'.
        clamp_negative: If True, clamps negative spillover noise (< 0.0) to 0.0.
        autospill_iterations: Number of iterative linear-space slope refinement passes (default: 2).
        fallback_policy: Action when a control is missing or fails QC ('embedded_row',
            'identity_row', or 'strict_error').
        manual_overrides: Optional dict of {(primary_channel, secondary_channel): spillover_coeff}
            or {"primary->secondary": spillover_coeff} for interactive fine-tuning.
        max_condition_number_warn: Threshold on matrix condition number kappa(S) to flag collinearity.
        residual_z_threshold: Threshold on post-compensation |Delta_MFI_comp / rSD_neg| (in rSD units)
            to flag under- or over-compensation (default: 0.5).
        compute_ssm: If True, calculates the Nguyen et al. (2013) Spillover Spreading Matrix (SSM).
        align_sample_tandem_mismatch: If True, audits the biological sample's live singlet lymphocytes
            for extreme bead-to-cell tandem decoupling over-compensation (S > 1.0, dy < -1500, Z < -3.0)
            and aligns the decoupled tandem spillover coefficient on the biological sample.
        min_control_events: Minimum clean events required for a control file after Time QC.
        sample_nxn_gating_mode: Gating mode for N x N sample validation matrices:
            'auto' (default: Intact + Singlets + Viability if dye present),
            'viable' (strict viability gating), 'singlet' (intact singlets),
            'intact' (intact morphology), or 'all' (raw ungated).
        random_seed: Random seed for reproducible clustering and subsampling.
    """

    source: str = "auto"
    estimation_method: str = "hybrid"
    apply_time_qc_to_controls: bool = True
    apply_time_qc_to_sample: bool = True
    auto_scatter_singlet_gate: bool = True
    control_carrier_type: Union[str, Dict[str, str]] = "auto"
    negative_reference: str = "auto"
    clamp_negative: bool = True
    autospill_iterations: int = 2
    fallback_policy: str = "embedded_row"
    manual_overrides: Optional[Dict[Union[Tuple[str, str], str], float]] = None
    max_condition_number_warn: float = 100.0
    residual_z_threshold: float = 0.5
    compute_ssm: bool = True
    align_sample_tandem_mismatch: bool = False
    min_control_events: int = 500
    sample_nxn_gating_mode: str = "viable"
    random_seed: int = 42
    voltage_mismatch_policy: str = "warn"
    voltage_tolerance_v: float = 5.0
    pmt_gain_exponent_alpha: float = 7.2


@dataclass
class PMTVoltageAuditReport:
    """Audit report comparing PMT voltages/gains between controls and sample.

    Attributes:
        sample_name: Name of biological sample audited.
        has_mismatch: True if any fluorescence detector voltage delta exceeds tolerance.
        mismatched_channels: List of fluorescent detector names with voltage drift.
        audit_df: Full DataFrame of all detectors, voltages, deltas, and statuses.
        summary_message: Human-readable concordance summary.
        action_taken: Description of policy action executed ('warn', 'fallback_embedded', 'scale_power_law').
    """
    sample_name: str
    has_mismatch: bool = False
    mismatched_channels: List[str] = field(default_factory=list)
    audit_df: Optional[pd.DataFrame] = None
    summary_message: str = ""
    action_taken: str = "warn"


@dataclass
class CompensationControlResult:
    """Represents the results and diagnostics of calculating a spillover matrix from controls.

    Attributes:
        matrix: Calculated CompensationMatrix instance.
        dataframe: Pandas DataFrame representation of the active spillover matrix.
        control_files_mapped: Dict mapping channel name to matched control FCS file path.
        unmatched_channels: List of channels without a matching control file.
        population_stats: Statistics per channel (carrier type, Time QC, Pos/Neg MFIs, SNR, etc.).
        condition_number: Condition number kappa(S) of the resulting spillover matrix.
        embedded_correlation: Pearson correlation r with the sample's embedded matrix (if available).
        embedded_mae: Mean Absolute Error compared to the sample's embedded matrix (if available).
        estimation_method_used: Estimator used ('hybrid', 'median_delta_mfi', 'autospill_regression').
        mfi_matrix_df: Spillover matrix estimated purely via upgraded Median Delta-MFI.
        regression_matrix_df: Spillover matrix estimated via AutoSpill robust linear regression.
        ssm_dataframe: Spillover Spreading Matrix (SSM) DataFrame (Nguyen et al. 2013).
        residual_z_dataframe: Post-compensation residual median alignment Z-score matrix
            (Delta_MFI_comp / rSD_neg) across all N x N primary-vs-secondary pairs.
        residual_pct_dataframe: Post-compensation residual error as % of primary Delta-MFI.
        diff_from_embedded_dataframe: Difference matrix (Calculated - Embedded) in percentage points.
        control_qc_summary_df: Tabular summary of Time QC, carrier detection, gating, and residual alignment per control.
        control_audit_records: Cached subsampled event snapshots per control for fast diagnostic plotting.
        voltage_audit_report: Optional PMTVoltageAuditReport comparing control vs sample voltages.
        voltage_audit_df: Tabular DataFrame of channel-level voltage concordance.
        has_voltage_mismatch: True if any fluorescence detector voltage delta exceeds tolerance.
        voltage_mismatch_channels: List of fluorescent detector names with voltage drift.
    """

    matrix: CompensationMatrix
    dataframe: pd.DataFrame
    control_files_mapped: Dict[str, str] = field(default_factory=dict)
    unmatched_channels: List[str] = field(default_factory=list)
    population_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    condition_number: float = 1.0
    embedded_correlation: Optional[float] = None
    embedded_mae: Optional[float] = None
    estimation_method_used: str = "hybrid"
    mfi_matrix_df: Optional[pd.DataFrame] = None
    regression_matrix_df: Optional[pd.DataFrame] = None
    ssm_dataframe: Optional[pd.DataFrame] = None
    residual_z_dataframe: Optional[pd.DataFrame] = None
    residual_pct_dataframe: Optional[pd.DataFrame] = None
    diff_from_embedded_dataframe: Optional[pd.DataFrame] = None
    control_qc_summary_df: Optional[pd.DataFrame] = None
    control_audit_records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gated_controls: Optional[Dict[str, Any]] = None
    voltage_audit_report: Optional[PMTVoltageAuditReport] = None
    voltage_audit_df: Optional[pd.DataFrame] = None
    has_voltage_mismatch: bool = False
    voltage_mismatch_channels: List[str] = field(default_factory=list)


class CompensationMatrix:
    """Represents a fluorescence spillover/compensation matrix.

    Attributes:
        detectors: List of channel detector names.
        matrix: 2D NumPy array of shape (N, N) containing spillover coefficients.
        dataframe: Pandas DataFrame representation of the matrix.
        inverse_matrix: 2D NumPy array containing inverted matrix (compensation coefficients).
    """

    def __init__(
        self,
        detectors: List[str],
        matrix: Union[np.ndarray, List[List[float]]],
    ) -> None:
        """Initialize compensation matrix.

        Args:
            detectors: List of detector channel names.
            matrix: Square matrix of spillover values.

        Raises:
            ValueError: If matrix dimensions do not match detectors or matrix is singular.
        """
        self.detectors = list(detectors)
        mat_arr = np.asarray(matrix, dtype=np.float64)

        if mat_arr.ndim != 2 or mat_arr.shape[0] != mat_arr.shape[1]:
            raise ValueError(
                f"Compensation matrix must be square 2D array, got shape {mat_arr.shape}"
            )

        if mat_arr.shape[0] != len(self.detectors):
            raise ValueError(
                f"Matrix dimension ({mat_arr.shape[0]}) does not match number of "
                f"detectors ({len(self.detectors)})"
            )

        self.matrix = mat_arr
        self.dataframe = pd.DataFrame(
            self.matrix, index=self.detectors, columns=self.detectors
        )
        try:
            self.inverse_matrix = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Spillover matrix is singular and cannot be inverted") from exc

    @property
    def condition_number(self) -> float:
        """Compute condition number of the spillover matrix."""
        if self.matrix.size == 0 or len(self.matrix) == 0:
            return 1.0
        return float(np.linalg.cond(self.matrix))

    def apply(
        self,
        events: np.ndarray,
        channel_indices: List[int],
    ) -> np.ndarray:
        """Apply compensation to an event array.

        Args:
            events: 2D NumPy array of shape (n_events, n_channels).
            channel_indices: Column indices in `events` corresponding to `self.detectors`.

        Returns:
            New 2D NumPy array with compensated channels.
        """
        compensated = np.array(events, copy=True, dtype=np.float64)
        sub_events = compensated[:, channel_indices]

        # In flow cytometry: Y = X * S^T -> X = Y * (S^T)^(-1)
        # Using np.linalg.solve(S.T, Y.T).T is mathematically equivalent to and more stable than inv(S)
        compensated[:, channel_indices] = np.linalg.solve(self.matrix.T, sub_events.T).T
        return compensated

    def apply_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply compensation to a Pandas DataFrame of events.

        Args:
            df: DataFrame containing columns matching `self.detectors`.

        Returns:
            New DataFrame with compensated values in specified detector columns.

        Raises:
            KeyError: If any detector is missing from DataFrame columns.
        """
        missing = [d for d in self.detectors if d not in df.columns]
        if missing:
            raise KeyError(f"Columns missing from DataFrame for compensation: {missing}")

        res_df = df.copy()
        sub_matrix = res_df[self.detectors].to_numpy(dtype=np.float64)
        comp_sub = np.linalg.solve(self.matrix.T, sub_matrix.T).T

        for i, det in enumerate(self.detectors):
            res_df[det] = comp_sub[:, i]

        return res_df

    def with_overrides(
        self,
        overrides: Dict[Union[Tuple[str, str], str], float],
    ) -> CompensationMatrix:
        """Return a new CompensationMatrix with manual spillover overrides applied.

        Args:
            overrides: Mapping of (primary_channel, secondary_channel) or "primary->secondary"
                to new fractional spillover value (e.g. 0.05 for 5%).

        Returns:
            New CompensationMatrix instance.
        """
        new_df = self.dataframe.copy()
        for key, val in overrides.items():
            if isinstance(key, tuple) and len(key) == 2:
                p_ch, s_ch = key
            elif isinstance(key, str) and "->" in key:
                parts = key.split("->", 1)
                p_ch, s_ch = parts[0].strip(), parts[1].strip()
            else:
                continue
            if p_ch in new_df.index and s_ch in new_df.columns and p_ch != s_ch:
                new_df.loc[p_ch, s_ch] = float(val)
        return CompensationMatrix(detectors=self.detectors, matrix=new_df.values)

    def to_dict(self) -> dict:
        """Serialize compensation matrix to a dictionary."""
        return {
            "detectors": self.detectors,
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CompensationMatrix:
        """Create CompensationMatrix from dictionary."""
        return cls(detectors=data["detectors"], matrix=np.array(data["matrix"]))

    def __repr__(self) -> str:
        return f"CompensationMatrix(detectors={len(self.detectors)}, cond={self.condition_number:.2f})"


def compensate_events(
    events: np.ndarray,
    spillover: Union[np.ndarray, CompensationMatrix],
    channel_indices: List[int],
) -> np.ndarray:
    """Compensate raw event values using spillover matrix.

    Args:
        events: 2D NumPy array of shape (n_events, n_channels).
        spillover: 2D spillover matrix or CompensationMatrix instance.
        channel_indices: Channel indices corresponding to rows/cols of spillover.

    Returns:
        Compensated 2D NumPy array.
    """
    if isinstance(spillover, CompensationMatrix):
        return spillover.apply(events, channel_indices)

    spill_arr = np.asarray(spillover, dtype=np.float64)
    comp_obj = CompensationMatrix(
        detectors=[f"ch_{i}" for i in range(spill_arr.shape[0])], matrix=spill_arr
    )
    return comp_obj.apply(events, channel_indices)


def _robust_sd(vals: np.ndarray) -> float:
    """Calculate robust standard deviation (rSD = MAD * 1.4826) of a 1D array."""
    finite = vals[np.isfinite(vals)]
    if len(finite) < 2:
        return 1.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    rsd = mad * 1.4826
    if rsd <= 1e-6:
        iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25))
        rsd = iqr / 1.349
    return max(rsd, 1.0)


def _find_col_by_alias(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find a column in df matching candidates (case-insensitive and normalized)."""
    cols = list(df.columns)
    col_map = {c.lower(): c for c in cols}
    for cand in candidates:
        cand_low = cand.lower()
        if cand_low in col_map:
            return col_map[cand_low]
    clean_map = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in cols}
    for cand in candidates:
        clean_cand = re.sub(r"[^a-z0-9]", "", cand.lower())
        if clean_cand in clean_map:
            return clean_map[clean_cand]
    return None


def _find_scatter_channels_in_df(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Resolve scatter and height/width parameter column names from any flow cytometer vendor."""
    fsc = _find_col_by_alias(df, ["fsc-a", "fs int", "fs.int", "fs-int", "fs_int", "fsc_a", "fsc", "fs", "forward scatter"])
    ssc = _find_col_by_alias(df, ["ssc-a", "ss int", "ss.int", "ss-int", "ss_int", "ssc_a", "ssc", "ss", "side scatter"])
    fsc_h = _find_col_by_alias(df, ["fsc-h", "fs peak", "fs.peak", "fs-peak", "fs_h", "fsc_h", "forward scatter height"])
    ssc_h = _find_col_by_alias(df, ["ssc-h", "ss peak", "ss.peak", "ss-peak", "ss_h", "ssc_h", "side scatter height"])
    fsc_w = _find_col_by_alias(df, ["fsc-w", "fs tof", "fs.tof", "fs-tof", "fs_w", "fsc_w", "forward scatter width"])
    return {"fsc": fsc, "ssc": ssc, "fsc_h": fsc_h, "ssc_h": ssc_h, "fsc_w": fsc_w}


def detect_control_carrier(
    df: pd.DataFrame,
    target_channel: str,
    channel_ranges: Optional[Dict[str, float]] = None,
) -> str:
    """Automatically detect whether a single-stained control uses antibody-capture beads or cells.

    Excludes low-scatter sub-cellular debris prior to evaluating physical size dispersion so
    bead controls containing debris fragments are never misclassified as cells. Antibody-capture
    beads exhibit uniform physical size (low non-debris FSC robust CV <= 0.45), whereas cell
    controls (such as PBMCs/WBCs stained with viability dyes) exhibit broad multi-lineage scatter
    or a large lysed-RBC debris fraction paired with high-FSC leukocytes.

    Args:
        df: Events DataFrame of the control sample.
        target_channel: Primary stained detector channel name.
        channel_ranges: Optional dictionary mapping channel name to maximum range.

    Returns:
        Carrier type string: 'beads' or 'cells'.
    """
    sc_cols = _find_scatter_channels_in_df(df)
    fsc_col = sc_cols["fsc"]
    if fsc_col and fsc_col in df.columns and len(df) >= 50:
        fsc = df[fsc_col].to_numpy(dtype=np.float64)
        finite_fsc = fsc[np.isfinite(fsc)]
        if len(finite_fsc) >= 50:
            if channel_ranges and fsc_col in channel_ranges:
                max_f = float(channel_ranges[fsc_col])
            else:
                p99 = float(np.percentile(finite_fsc, 99.0))
                max_f = 1048576.0 if p99 > 350000.0 else 262144.0

            # Relative scaling: whole blood / lysed WBC cell controls typically have >35% RBC lysis debris
            debris_cut = 0.08 * max_f
            p95_cut = 0.32 * max_f
            debris_frac = float(np.mean(finite_fsc < debris_cut))
            p95_fsc = float(np.percentile(finite_fsc, 95.0))
            if debris_frac > 0.35 and p95_fsc > p95_cut:
                return "cells"

            # Strip sub-cellular debris before evaluating scatter CV
            valid_fsc = finite_fsc[
                (finite_fsc >= (0.06 * max_f)) & (finite_fsc <= np.percentile(finite_fsc, 99.0))
            ]

            if len(valid_fsc) >= 50:
                med_fsc = float(np.median(valid_fsc))
                iqr_fsc = float(np.percentile(valid_fsc, 75) - np.percentile(valid_fsc, 25))
                rcv_fsc = iqr_fsc / max(1.0, med_fsc)
                if rcv_fsc > 0.45:
                    return "cells"
    return "beads"


def load_control_manifest(
    manifest_path: Union[str, Path],
) -> Tuple[Dict[str, str], Optional[str]]:
    """Load an explicit Sample-Control Manifest from a CSV or YAML file.

    Enables reproducible batch compensation workflows by explicitly mapping target fluorescence
    channels / detector names to specific single-stained control FCS file paths, with optional
    unstained control designation.

    Args:
        manifest_path: Path to the CSV (.csv) or YAML (.yaml, .yml) manifest file.

    Returns:
        Tuple of (channel_to_filepath_dict, optional_unstained_control_filepath).

    Raises:
        FileNotFoundError: If manifest_path does not exist.
        ValueError: If manifest format is unsupported or missing required columns.
    """
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Compensation control manifest file not found: {path}")

    base_dir = path.parent
    mapping: Dict[str, str] = {}
    unstained_file: Optional[str] = None

    def resolve_fpath(p_str: str) -> str:
        p_cand = Path(p_str)
        if p_cand.is_absolute() and p_cand.is_file():
            return str(p_cand)
        rel_cand = base_dir / p_cand
        if rel_cand.is_file():
            return str(rel_cand)
        cwd_cand = Path.cwd() / p_cand
        if cwd_cand.is_file():
            return str(cwd_cand)
        return str(rel_cand)

    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to parse YAML control manifests. Install with: pip install pyyaml"
            ) from exc

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(
                f"YAML manifest must contain a mapping/dictionary, got {type(data).__name__}"
            )

        if "unstained" in data and data["unstained"]:
            unstained_file = resolve_fpath(str(data["unstained"]))

        raw_controls = data.get("controls", data)
        if isinstance(raw_controls, dict):
            for k, v in raw_controls.items():
                if k.lower() in ("unstained", "negative", "blank"):
                    unstained_file = resolve_fpath(str(v))
                elif k != "controls":
                    mapping[str(k).strip()] = resolve_fpath(str(v))
        elif isinstance(raw_controls, list):
            for item in raw_controls:
                if isinstance(item, dict):
                    ch = item.get("channel") or item.get("detector") or item.get("name")
                    fp = (
                        item.get("file")
                        or item.get("path")
                        or item.get("control_file")
                        or item.get("filepath")
                    )
                    if ch and fp:
                        if str(ch).lower().strip() in ("unstained", "negative", "blank"):
                            unstained_file = resolve_fpath(str(fp))
                        else:
                            mapping[str(ch).strip()] = resolve_fpath(str(fp))

    elif ext == ".csv":
        df = pd.read_csv(path)
        col_map = {c.lower().strip(): c for c in df.columns}
        ch_col = None
        for cand in ["channel", "detector", "pnn", "name", "target_channel", "marker"]:
            if cand in col_map:
                ch_col = col_map[cand]
                break

        fp_col = None
        for cand in [
            "control_file",
            "file",
            "path",
            "filepath",
            "control_path",
            "filename",
            "fcs_file",
        ]:
            if cand in col_map:
                fp_col = col_map[cand]
                break

        if not ch_col or not fp_col:
            raise ValueError(
                f"CSV manifest '{path.name}' must have channel/detector and file/path columns. "
                f"Found columns: {list(df.columns)}"
            )

        for _, row in df.iterrows():
            ch = str(row[ch_col]).strip()
            fp = str(row[fp_col]).strip()
            if not ch or not fp or ch.lower() == "nan" or fp.lower() == "nan":
                continue
            if ch.lower() in ("unstained", "negative", "blank"):
                unstained_file = resolve_fpath(fp)
            else:
                mapping[ch] = resolve_fpath(fp)

    else:
        raise ValueError(
            f"Unsupported manifest format: '{ext}'. Please provide a .csv or .yaml file."
        )

    return mapping, unstained_file


def match_control_files(
    controls_dir_or_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    channels: Optional[List[str]] = None,
    fcs_sample: Optional[Any] = None,
    channel_mapping: Optional[Dict[str, str]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Match single-stained compensation control FCS files to target channels.

    Heuristics:
    1. Explicit Sample-Control Manifest (`manifest_path`).
    2. Direct override via `channel_mapping`.
    3. Excitation and emission wavelength match (e.g. 'Ex 488: Em 610-A' <-> '...Ex 488...Em 610...').
    4. Normalized channel name match.
    5. Marker name match from sample metadata ($PnS).

    Args:
        controls_dir_or_files: Directory path or list of FCS control file paths.
        channels: Target detector channel names to match.
        fcs_sample: Optional parsed FCSData instance for marker lookup.
        channel_mapping: Optional user-provided channel-to-filepath mapping.
        manifest_path: Optional path to CSV or YAML control manifest file.

    Returns:
        Tuple of (mapped_dict, unmatched_channels_list).
    """
    effective_mapping: Dict[str, str] = {}
    if manifest_path is not None:
        man_map, _ = load_control_manifest(manifest_path)
        effective_mapping.update(man_map)

    if channel_mapping:
        effective_mapping.update(channel_mapping)

    if channels is None:
        if fcs_sample is not None and getattr(fcs_sample, "spillover_channels", None):
            channels = list(fcs_sample.spillover_channels)
        elif effective_mapping:
            channels = list(effective_mapping.keys())
        else:
            channels = []

    candidates: List[Path] = []
    if controls_dir_or_files is not None:
        if isinstance(controls_dir_or_files, (str, Path)):
            p = Path(controls_dir_or_files)
            if p.is_dir():
                sub_comp = p / "compensation_controls"
                sub_ctrl = p / "controls"
                sub_raw = p / "raw"
                target_dir = (
                    sub_raw if sub_raw.is_dir() else (sub_comp if sub_comp.is_dir() else (sub_ctrl if sub_ctrl.is_dir() else p))
                )
                candidates = sorted([f for f in target_dir.glob("*.fcs") if f.is_file()])
                if not candidates:
                    candidates = sorted([f for f in target_dir.rglob("*.fcs") if f.is_file()])
                if not candidates:
                    candidates = sorted(
                        [
                            f
                            for f in target_dir.iterdir()
                            if f.is_file() and f.suffix.lower() == ".fcs"
                        ]
                    )
            elif p.is_file():
                candidates = [p]
        else:
            candidates = [Path(f) for f in controls_dir_or_files if Path(f).is_file()]
    elif effective_mapping:
        candidates = [Path(f) for f in effective_mapping.values() if Path(f).is_file()]

    mapped: Dict[str, str] = {}
    unmatched: List[str] = []

    for ch in channels:
        if ch in effective_mapping:
            mapped[ch] = str(effective_mapping[ch])
            continue

        ch_clean = ch.lower().replace(",2f,", "/")
        ch_base_no_suffix = re.sub(r"[-_.\s]*(a|h|w|area|height|width)$", "", ch_clean).strip()
        ch_norm = re.sub(r"[^a-zA-Z0-9]", "", ch_base_no_suffix)
        if not ch_norm:
            ch_norm = re.sub(r"[^a-zA-Z0-9]", "", ch_clean)
        ch_ex_m = re.search(r"Ex\s*([\d/]+)", ch, re.IGNORECASE)
        ch_em_m = re.search(r"Em\s*(\d+)", ch, re.IGNORECASE)
        ch_laser_wave_m = re.match(r"^\s*([buvryg]|uv|yg)\s*[-_]?\s*(\d{3})\b", ch_clean)

        marker_name = ""
        fluoro_name = ""
        if fcs_sample is not None:
            if hasattr(fcs_sample, "detector_to_marker") and fcs_sample.detector_to_marker:
                marker_name = fcs_sample.detector_to_marker.get(ch, "").lower().strip()
            if hasattr(fcs_sample, "detector_to_fluorophore") and fcs_sample.detector_to_fluorophore:
                fluoro_name = fcs_sample.detector_to_fluorophore.get(ch, "").lower().strip()
            elif hasattr(fcs_sample, "pnn_labels") and hasattr(fcs_sample, "pns_labels"):
                try:
                    idx_ch = list(fcs_sample.pnn_labels).index(ch)
                    if idx_ch < len(fcs_sample.pns_labels) and fcs_sample.pns_labels[idx_ch]:
                        fluoro_name = str(fcs_sample.pns_labels[idx_ch]).lower().strip()
                except ValueError:
                    pass

        stop_words = {"control", "controls", "compensation", "stained", "a", "h", "w", "fl", "channel", "int", "peak", "bd", "horizon", "brilliant"}
        detector_tokens = {
            t for t in re.split(r"[\s\-_.,/]+", ch_base_no_suffix)
            if len(t) >= 2 and t not in stop_words
        }
        if ch_laser_wave_m:
            detector_tokens.add(f"{ch_laser_wave_m.group(1)}{ch_laser_wave_m.group(2)}")
        antigen_tokens = {
            t for t in re.split(r"[\s\-_.,/]+", marker_name)
            if len(t) >= 2 and t not in stop_words
        } if marker_name else set()
        fluoro_tokens = {
            t for t in re.split(r"[\s\-_.,/]+", fluoro_name)
            if len(t) >= 2 and t not in stop_words
        } if fluoro_name else set()

        antigen_norm = re.sub(r"[^a-zA-Z0-9]", "", marker_name)
        fluoro_norm = re.sub(r"[^a-zA-Z0-9]", "", fluoro_name)

        matched_path: Optional[Path] = None

        # Strategy A: Wavelength pair matching (Ex / Em)
        if ch_ex_m and ch_em_m:
            target_em = ch_em_m.group(1)
            target_ex_list = re.findall(r"\d{3}", ch_ex_m.group(1))
            for cand in candidates:
                cand_name = cand.name
                cand_ex_m = re.search(r"Ex\s*([\d/,]+)", cand_name, re.IGNORECASE)
                cand_em_m = re.search(r"Em\s*(\d+)", cand_name, re.IGNORECASE)
                if cand_ex_m and cand_em_m and cand_em_m.group(1) == target_em:
                    cand_ex_list = re.findall(r"\d{3}", cand_ex_m.group(1))
                    if set(cand_ex_list).intersection(set(target_ex_list)):
                        matched_path = cand
                        break

        # Strategy B & C: Best token & normalized overlap match across detector, fluorophore, and antigen
        if matched_path is None:
            best_cand: Optional[Path] = None
            best_score: float = -1.0
            candidate_target_tokens = detector_tokens | antigen_tokens | fluoro_tokens

            for cand in candidates:
                cand_stem_lower = cand.stem.lower().replace(",2f,", "/")
                if "unstained" in cand_stem_lower and "unstained" not in ch_clean:
                    continue
                if "fmo" in cand_stem_lower and "fmo" not in ch_clean:
                    continue

                # Guard against cross-laser false matches when both channel and control name specify a laser + wavelength
                if ch_laser_wave_m:
                    cand_lw_m = re.search(r"(?:^|[\s\-_.,/])([buvryg]|uv|yg)\s*[-_]?\s*(\d{3})\b", cand_stem_lower)
                    if cand_lw_m and (
                        cand_lw_m.group(1) != ch_laser_wave_m.group(1)
                        or cand_lw_m.group(2) != ch_laser_wave_m.group(2)
                    ):
                        continue

                cand_tokens = {t for t in re.split(r"[\s\-_.,/]+", cand_stem_lower) if len(t) >= 2 and t not in stop_words}
                cand_norm = re.sub(r"[^a-zA-Z0-9]", "", cand_stem_lower)

                score = 0.0
                # Exact normalized match on detector, fluorophore, or antigen gets highest priority
                if ch_norm and len(ch_norm) >= 3 and ch_norm in cand_norm:
                    score += 60.0
                    if ch_norm == cand_norm or cand_norm.endswith(ch_norm):
                        score += 25.0
                if fluoro_norm and len(fluoro_norm) >= 3 and fluoro_norm in cand_norm:
                    score += 55.0
                    if fluoro_norm in cand_tokens:
                        score += 20.0
                if antigen_norm and len(antigen_norm) >= 3 and antigen_norm in cand_norm:
                    score += 45.0
                    if antigen_norm in cand_tokens:
                        score += 15.0

                if candidate_target_tokens:
                    common = candidate_target_tokens & cand_tokens
                    if common:
                        score += len(common) * 15.0
                        score += (len(common) / len(candidate_target_tokens)) * 10.0
                        # Penalty for extraneous fluorophore modifiers in candidate (e.g. cf594, cy7 when searching for pe)
                        extra_tokens = cand_tokens - candidate_target_tokens
                        score -= len(extra_tokens) * 2.0

                if score > best_score and score > 0:
                    best_score = score
                    best_cand = cand

            if best_cand is not None:
                matched_path = best_cand

        if matched_path is not None:
            mapped[ch] = str(matched_path)
        else:
            unmatched.append(ch)

    return mapped, unmatched


def _pregate_control_scatter_and_singlets(
    df: pd.DataFrame,
    resolved_carrier: str,
    target_channel: Optional[str] = None,
    max_range: float = 262144.0,
    all_channels: Optional[List[str]] = None,
    channel_ranges: Optional[Dict[str, float]] = None,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Execute strict sequential pre-gating on a control sample to remove debris and doublets.

    Follows gold-standard cytometry order of operations:
      Stage 0 (Detector Saturation Exclusion): Removes clipped events on the primary channel,
          scatter channels, and secondary fluorescence channels using channel-specific ranges.
      Stage 1 (Debris Removal & Morphology Gate on FSC vs SSC):
          - For 'beads': Strips low-scatter debris, locates 2D monomer bead density peak,
            and applies a 2D elliptical monomer gate (excluding debris & dimers/aggregates).
          - For 'cells': Fits a 3-component log-scatter Gaussian Mixture Model on
            (log10(FSC), log10(SSC)) paired with a relative leukocyte floor to
            exclude sub-cellular RBC lysis debris and platelets prior to singlet gating.
      Stage 2 (Dual Singlet Discrimination on FSC-A vs FSC-H, SSC-A vs SSC-H, and FSC-W):
          - Computes pulse height-to-area ratio medians and MADs strictly on debris-free cells/beads.

    Args:
        df: Events DataFrame of the control sample (post-Time QC if enabled).
        resolved_carrier: 'beads' or 'cells'.
        target_channel: Primary stained detector channel name (if applicable).
        max_range: Default hardware saturation limit ($PnR / $PnG).
        all_channels: Optional list of all panel fluorescence detectors.
        channel_ranges: Optional dictionary mapping each detector name to its specific range.
        random_seed: Random seed for reproducible GMM fitting.

    Returns:
        Tuple of (non_sat_mask, debris_free_mask, clean_singlet_mask, gate_geometry_dict).
    """
    n_total = len(df)
    if n_total == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty, {"carrier_type": resolved_carrier}

    # Stage 0: Detector saturation exclusion (primary + secondary channels, C5)
    non_sat = np.ones(n_total, dtype=bool)
    t_pnr = channel_ranges.get(target_channel, max_range) if (channel_ranges and target_channel) else max_range
    if target_channel and target_channel in df.columns and t_pnr > 1000.0:
        t_raw = df[target_channel].to_numpy(dtype=np.float64)
        cand_t = np.isfinite(t_raw) & (t_raw < 0.985 * t_pnr)
        if int(np.sum(cand_t)) >= 50:
            non_sat &= cand_t

    if all_channels:
        cand_sec = non_sat.copy()
        for ch_sec in all_channels:
            if ch_sec != target_channel and ch_sec in df.columns:
                s_pnr = channel_ranges.get(ch_sec, max_range) if channel_ranges else max_range
                if s_pnr > 1000.0:
                    s_raw = df[ch_sec].to_numpy(dtype=np.float64)
                    cand_sec &= np.isfinite(s_raw) & (s_raw < 0.992 * s_pnr)
        if int(np.sum(cand_sec)) >= 50:
            non_sat = cand_sec

    sc_cols = _find_scatter_channels_in_df(df)
    fsc_col = sc_cols["fsc"]
    ssc_col = sc_cols["ssc"]
    fsc_h_col = sc_cols["fsc_h"]
    ssc_h_col = sc_cols["ssc_h"]
    fsc_w_col = sc_cols["fsc_w"]

    debris_free = non_sat.copy()
    morph_kept = non_sat.copy()
    gate_geom: Dict[str, Any] = {
        "carrier_type": resolved_carrier,
        "gate_type": "none",
        "fsc_channel": fsc_col or "FSC-A",
        "ssc_channel": ssc_col or "SSC-A",
    }

    # Stage 1: Debris Removal & Carrier-Specific FSC vs SSC Morphology Gate (FIRST)
    if fsc_col and fsc_col in df.columns and int(np.sum(non_sat)) >= 60:
        fsc = df[fsc_col].to_numpy(dtype=np.float64)
        has_ssc = ssc_col is not None and ssc_col in df.columns
        ssc = df[ssc_col].to_numpy(dtype=np.float64) if has_ssc else np.zeros_like(fsc)

        if channel_ranges and fsc_col in channel_ranges:
            fsc_max = float(channel_ranges[fsc_col])
        else:
            p99_f = float(np.percentile(fsc[non_sat], 99.0)) if np.any(non_sat) else 262144.0
            fsc_max = 1048576.0 if p99_f > 350000.0 else 262144.0

        if channel_ranges and ssc_col in channel_ranges:
            ssc_max = float(channel_ranges[ssc_col])
        else:
            p99_s = float(np.percentile(ssc[non_sat], 99.0)) if (has_ssc and np.any(non_sat)) else 262144.0
            ssc_max = 1048576.0 if p99_s > 350000.0 else 262144.0

        is_high_scale = fsc_max > 10000.0

        if resolved_carrier == "beads":
            fsc_lo = 0.055 * fsc_max if is_high_scale else 50.0
            fsc_hi = min(0.98 * fsc_max, float(np.nanpercentile(fsc[non_sat], 99.5))) if is_high_scale else float(np.nanpercentile(fsc[non_sat], 99.5))
            rough_valid = non_sat & np.isfinite(fsc) & (fsc >= fsc_lo) & (fsc <= fsc_hi)
            if has_ssc:
                ssc_lo = 0.03 * ssc_max if (is_high_scale and ssc_max > 15000.0) else 50.0
                ssc_hi = float(np.nanpercentile(ssc[non_sat], 99.5))
                rough_valid &= np.isfinite(ssc) & (ssc >= ssc_lo) & (ssc <= ssc_hi)

            if has_ssc and int(np.sum(rough_valid)) >= 60:
                fsc_sub = fsc[rough_valid]
                ssc_sub = ssc[rough_valid]
                # Cap at 88th percentile when locating 2D mode so bead dimers never win the peak
                f_cap = float(np.nanpercentile(fsc_sub, 88.0))
                s_cap = float(np.nanpercentile(ssc_sub, 88.0))
                core_sub = (fsc_sub <= f_cap) & (ssc_sub <= s_cap)
                if int(np.sum(core_sub)) >= 40:
                    H, xedges, yedges = np.histogram2d(
                        fsc_sub[core_sub], ssc_sub[core_sub], bins=32
                    )
                    im = np.unravel_index(np.argmax(H), H.shape)
                    mode_fsc = float(0.5 * (xedges[im[0]] + xedges[im[0] + 1]))
                    mode_ssc = float(0.5 * (yedges[im[1]] + yedges[im[1] + 1]))
                else:
                    mode_fsc = float(np.nanmedian(fsc_sub))
                    mode_ssc = float(np.nanmedian(ssc_sub))

                near_mode = (
                    rough_valid
                    & (np.abs(fsc - mode_fsc) <= 0.35 * mode_fsc)
                    & (np.abs(ssc - mode_ssc) <= 0.35 * mode_ssc)
                )
                if int(np.sum(near_mode)) >= 30:
                    mad_f = max(
                        float(np.nanmedian(np.abs(fsc[near_mode] - mode_fsc))),
                        0.045 * mode_fsc,
                    )
                    mad_s = max(
                        float(np.nanmedian(np.abs(ssc[near_mode] - mode_ssc))),
                        0.055 * mode_ssc,
                    )
                else:
                    mad_f = max(
                        float(np.nanmedian(np.abs(fsc_sub - mode_fsc))), 0.05 * mode_fsc
                    )
                    mad_s = max(
                        float(np.nanmedian(np.abs(ssc_sub - mode_ssc))), 0.06 * mode_ssc
                    )

                rx = float(4.2 * mad_f)
                ry = float(4.2 * mad_s)
                cand_deb_free = (
                    non_sat
                    & np.isfinite(fsc)
                    & np.isfinite(ssc)
                    & (fsc >= max(fsc_lo, mode_fsc - rx))
                    & (ssc >= max(ssc_lo, mode_ssc - ry))
                )
                cand_ell = cand_deb_free & (
                    ((fsc - mode_fsc) / rx) ** 2 + ((ssc - mode_ssc) / ry) ** 2 <= 1.0
                )
                if int(np.sum(cand_ell)) >= 50:
                    debris_free = cand_deb_free
                    morph_kept = cand_ell
                    gate_geom = {
                        "carrier_type": "beads",
                        "gate_type": "ellipse",
                        "center_fsc": mode_fsc,
                        "center_ssc": mode_ssc,
                        "radius_fsc": rx,
                        "radius_ssc": ry,
                    }
            elif int(np.sum(rough_valid)) >= 50:
                fsc_med = float(np.nanmedian(fsc[rough_valid]))
                fsc_mad = max(
                    float(np.nanmedian(np.abs(fsc[rough_valid] - fsc_med))),
                    0.05 * fsc_med,
                )
                rx = float(4.0 * fsc_mad)
                cand_deb_free = non_sat & np.isfinite(fsc) & (fsc >= max(fsc_lo, fsc_med - rx))
                cand_1d = cand_deb_free & (np.abs(fsc - fsc_med) <= rx)
                if int(np.sum(cand_1d)) >= 50:
                    debris_free = cand_deb_free
                    morph_kept = cand_1d
                    gate_geom = {
                        "carrier_type": "beads",
                        "gate_type": "box_1d",
                        "center_fsc": fsc_med,
                        "radius_fsc": rx,
                    }
        else:
            # Cell-based control: exclude sub-cellular RBC lysis debris and platelets FIRST
            fsc_lo = 0.065 * fsc_max if is_high_scale else 100.0
            fsc_hi = min(0.98 * fsc_max, float(np.nanpercentile(fsc[non_sat], 99.5))) if is_high_scale else float(np.nanpercentile(fsc[non_sat], 99.5))
            ssc_lo = 0.03 * ssc_max if (has_ssc and is_high_scale and ssc_max > 15000.0) else 50.0
            ssc_hi = min(0.95 * ssc_max, float(np.nanpercentile(ssc[non_sat], 99.5))) if (has_ssc and is_high_scale) else (float(np.nanpercentile(ssc[non_sat], 99.5)) if has_ssc else 1e9)

            base_cells = non_sat & np.isfinite(fsc) & (fsc >= fsc_lo) & (fsc <= fsc_hi)
            if has_ssc:
                base_cells &= np.isfinite(ssc) & (ssc >= ssc_lo) & (ssc <= ssc_hi)

            if has_ssc and is_high_scale and int(np.sum(non_sat)) >= 200:
                try:
                    log_fsc = np.log10(np.maximum(fsc, 100.0))
                    log_ssc = np.log10(np.maximum(ssc, 100.0))
                    X_all = np.column_stack([log_fsc, log_ssc])
                    valid_idx = np.where(non_sat & np.isfinite(log_fsc) & np.isfinite(log_ssc))[0]
                    rng_gmm = np.random.RandomState(random_seed)
                    fit_idx = (
                        rng_gmm.choice(valid_idx, size=min(len(valid_idx), 15000), replace=False)
                        if len(valid_idx) > 15000
                        else valid_idx
                    )
                    gmm_morph = GaussianMixture(n_components=3, random_state=random_seed).fit(
                        X_all[fit_idx]
                    )
                    lbls = gmm_morph.predict(X_all)
                    cell_clusters = [
                        k
                        for k in range(3)
                        if not (
                            float(np.nanmedian(fsc[lbls == k])) < (0.075 * fsc_max)
                            and float(np.nanmedian(ssc[lbls == k])) < (0.045 * ssc_max)
                        )
                    ]
                    if cell_clusters:
                        cand_cells = base_cells & np.isin(lbls, cell_clusters)
                        if int(np.sum(cand_cells)) >= 50:
                            base_cells = cand_cells
                except Exception:
                    pass

            if int(np.sum(base_cells)) >= 50:
                debris_free = base_cells
                morph_kept = base_cells
                gate_geom = {
                    "carrier_type": "cells",
                    "gate_type": "cell_box",
                    "fsc_channel": fsc_col or "FSC-A",
                    "ssc_channel": ssc_col or "SSC-A",
                    "fsc_lo": float(np.nanmin(fsc[base_cells])),
                    "fsc_hi": float(np.nanmax(fsc[base_cells])),
                    "ssc_lo": float(np.nanmin(ssc[base_cells])) if has_ssc else 0.0,
                    "ssc_hi": float(np.nanmax(ssc[base_cells])) if has_ssc else 0.0,
                }

    # Stage 2: Dual Singlet Gate (FSC-A vs FSC-H and SSC-A vs SSC-H + Pulse Width)
    singlet_kept = morph_kept.copy()
    if fsc_col and fsc_h_col and fsc_h_col in df.columns and int(np.sum(singlet_kept)) >= 50:
        fsc_a = df[fsc_col].to_numpy(dtype=np.float64)
        fsc_h = df[fsc_h_col].to_numpy(dtype=np.float64)
        ratio_f = fsc_h / np.maximum(fsc_a, 1.0)
        med_rf = float(np.nanmedian(ratio_f[morph_kept]))
        mad_rf = max(
            float(np.nanmedian(np.abs(ratio_f[morph_kept] - med_rf))),
            0.025,
        )
        tol_f = 3.5 * mad_rf
        cand_f = singlet_kept & np.isfinite(ratio_f) & (np.abs(ratio_f - med_rf) <= tol_f)
        if int(np.sum(cand_f)) >= 50:
            singlet_kept = cand_f
            gate_geom["fsc_singlet_ratio_median"] = round(med_rf, 4)
            gate_geom["fsc_singlet_ratio_tol"] = round(tol_f, 4)

    if ssc_col and ssc_h_col and ssc_h_col in df.columns and int(np.sum(singlet_kept)) >= 50:
        ssc_a = df[ssc_col].to_numpy(dtype=np.float64)
        ssc_h = df[ssc_h_col].to_numpy(dtype=np.float64)
        ratio_s = ssc_h / np.maximum(ssc_a, 1.0)
        med_rs = float(np.nanmedian(ratio_s[singlet_kept]))
        mad_rs = max(
            float(np.nanmedian(np.abs(ratio_s[singlet_kept] - med_rs))),
            0.028,
        )
        tol_s = 3.8 * mad_rs
        cand_s = singlet_kept & np.isfinite(ratio_s) & (np.abs(ratio_s - med_rs) <= tol_s)
        if int(np.sum(cand_s)) >= 50:
            singlet_kept = cand_s
            gate_geom["ssc_singlet_ratio_median"] = round(med_rs, 4)
            gate_geom["ssc_singlet_ratio_tol"] = round(tol_s, 4)

    if fsc_w_col and fsc_w_col in df.columns and int(np.sum(singlet_kept)) >= 50:
        fsc_w = df[fsc_w_col].to_numpy(dtype=np.float64)
        med_fw = float(np.nanmedian(fsc_w[singlet_kept]))
        mad_fw = max(
            float(np.nanmedian(np.abs(fsc_w[singlet_kept] - med_fw))),
            0.03 * med_fw,
        )
        cand_w = (
            singlet_kept
            & np.isfinite(fsc_w)
            & (fsc_w >= med_fw - 4.5 * mad_fw)
            & (fsc_w <= med_fw + 4.5 * mad_fw)
        )
        if int(np.sum(cand_w)) >= 50:
            singlet_kept = cand_w

    return non_sat, debris_free, singlet_kept, gate_geom


def gate_control_populations(
    df: pd.DataFrame,
    target_channel: str,
    method: str = "auto",
    unstained_df: Optional[pd.DataFrame] = None,
    random_seed: int = 42,
    carrier_type: str = "auto",
    auto_scatter_singlet_gate: bool = True,
    max_range: float = 262144.0,
    all_channels: Optional[List[str]] = None,
    channel_ranges: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Identify high-quality positive and negative bead or cell populations in a single-stained control.

    Executes strict sequential pre-gating:
      1. Multi-channel detector saturation exclusion using vendor-specific PnR/PnG ranges.
      2. Carrier-aware Debris Removal & Morphology Gate on FSC vs SSC (2D monomer bead
         density peak + elliptical gate for beads; log-scatter GMM + leukocyte floor for cells).
      3. Dual Singlet Discrimination (FSC height/area, SSC height/area, pulse width).
      4. High-purity Positive and Negative fluorescence gating (posterior-filtered GMM separation
         for bimodal distributions; bright tail + scatter-matched untruncated mode for viability dyes;
         or matched external unstained reference control).

    Args:
        df: DataFrame of control events (post-Time QC if enabled).
        target_channel: Primary stained detector channel.
        method: Gating algorithm ('auto', 'gmm', 'tail_scatter_matched', or 'quantile').
        unstained_df: Optional DataFrame of unstained control events.
        random_seed: Seed for reproducibility in clustering.
        carrier_type: 'auto', 'beads', or 'cells'.
        auto_scatter_singlet_gate: If True, filters debris, homogeneous scatter, and singlets.
        max_range: Upper hardware saturation limit ($PnR / $PnG).
        all_channels: Optional list of all fluorescence channels.
        channel_ranges: Optional mapping from channel name to specific range limit.

    Returns:
        Tuple of (clean_df, pos_mask, neg_mask, population_stats_dict).
    """
    raw_df = df.copy()
    n_initial = len(raw_df)

    resolved_carrier = (
        detect_control_carrier(raw_df, target_channel, channel_ranges=channel_ranges)
        if carrier_type == "auto"
        else carrier_type.lower()
    )

    n_debris_removed = 0
    n_doublets_removed = 0
    debris_removed_pct = 0.0
    doublets_removed_pct = 0.0
    scatter_kept_pct = 100.0
    gate_geom: Dict[str, Any] = {"carrier_type": resolved_carrier}

    if auto_scatter_singlet_gate and n_initial >= 60:
        non_sat_mask, debris_free_mask, clean_singlet_mask, gate_geom = (
            _pregate_control_scatter_and_singlets(
                df=raw_df,
                resolved_carrier=resolved_carrier,
                target_channel=target_channel,
                max_range=max_range,
                all_channels=all_channels,
                channel_ranges=channel_ranges,
                random_seed=random_seed,
            )
        )
        if int(np.sum(clean_singlet_mask)) >= 50:
            clean_df = raw_df.loc[clean_singlet_mask].copy()
            n_debris_removed = int(n_initial - np.sum(debris_free_mask))
            n_doublets_removed = int(np.sum(debris_free_mask) - np.sum(clean_singlet_mask))
            debris_removed_pct = float(n_debris_removed / max(1, n_initial) * 100.0)
            doublets_removed_pct = float(n_doublets_removed / max(1, n_initial) * 100.0)
            scatter_kept_pct = float(len(clean_df) / max(1, n_initial) * 100.0)
            gate_geom["non_sat_mask"] = non_sat_mask
            gate_geom["debris_free_mask"] = debris_free_mask
            gate_geom["clean_singlet_mask"] = clean_singlet_mask
        else:
            clean_df = raw_df
    else:
        t_pnr = channel_ranges.get(target_channel, max_range) if (channel_ranges and target_channel) else max_range
        if target_channel in raw_df.columns and t_pnr > 1000.0:
            t_raw = raw_df[target_channel].to_numpy(dtype=np.float64)
            non_sat = np.isfinite(t_raw) & (t_raw < 0.985 * t_pnr)
            clean_df = raw_df.loc[non_sat].copy() if int(np.sum(non_sat)) >= 50 else raw_df
        else:
            clean_df = raw_df

    target_vals = clean_df[target_channel].to_numpy(dtype=np.float64)
    n_events = len(target_vals)

    used_method = "quantile"
    pos_mask = np.zeros(n_events, dtype=bool)
    neg_mask = np.zeros(n_events, dtype=bool)

    # C3: An external unstained cell control must only be used for cell-based controls.
    # Antibody-capture beads always use internal negative beads in the same tube (Roederer Rule #2).
    use_external_unstained = (
        unstained_df is not None
        and not unstained_df.empty
        and resolved_carrier == "cells"
    )

    clean_un_df: Optional[pd.DataFrame] = None
    if use_external_unstained:
        un_carrier = detect_control_carrier(unstained_df, target_channel)
        if auto_scatter_singlet_gate and len(unstained_df) >= 60:
            _, _, un_clean_mask, _ = _pregate_control_scatter_and_singlets(
                df=unstained_df,
                resolved_carrier=un_carrier,
                target_channel=target_channel,
                max_range=max_range,
                all_channels=all_channels,
                channel_ranges=channel_ranges,
                random_seed=random_seed,
            )
            clean_un_df = (
                unstained_df.loc[un_clean_mask].copy()
                if int(np.sum(un_clean_mask)) >= 30
                else unstained_df.copy()
            )
        else:
            clean_un_df = unstained_df.copy()

    # Step 4: Fluorescence Gating on clean_df
    # Try GMM separation on arcsinh-transformed values first (effective for both beads and bimodal cells)
    if method in ("auto", "gmm") and n_events >= 40:
        try:
            asinh_vals = np.arcsinh(np.clip(target_vals / 150.0, -10.0, None)).reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=random_seed)
            labels = gmm.fit_predict(asinh_vals)
            probs = gmm.predict_proba(asinh_vals)
            means = gmm.means_.flatten()

            pos_idx = int(np.argmax(means))
            neg_idx = int(np.argmin(means))

            cand_pos = (labels == pos_idx) & (probs[:, pos_idx] >= 0.80)
            cand_neg = (labels == neg_idx) & (probs[:, neg_idx] >= 0.80)
            if int(np.sum(cand_neg)) >= 30:
                neg_hi = float(np.nanpercentile(target_vals[cand_neg], 98.5))
                neg_lo = float(np.nanpercentile(target_vals[cand_neg], 0.5))
                cand_neg &= (target_vals >= neg_lo) & (target_vals <= neg_hi)
            if int(np.sum(cand_pos)) >= 30:
                pos_lo = float(np.nanpercentile(target_vals[cand_pos], 2.0))
                cand_pos &= target_vals >= pos_lo

            if (
                np.sum(cand_pos) >= 15
                and np.sum(cand_neg) >= 15
                and (means[pos_idx] - means[neg_idx]) >= 0.5
            ):
                pos_mask = cand_pos
                neg_mask = cand_neg
                used_method = "gmm"
        except Exception:
            pass

    # If GMM did not resolve distinct clusters and carrier is cells (e.g. viability dye with tiny dead tail)
    if np.sum(pos_mask) == 0 and (resolved_carrier == "cells" or method == "tail_scatter_matched"):
        used_method = "tail_scatter_matched"
        p965 = float(np.nanpercentile(target_vals, 96.5))
        p50 = float(np.nanpercentile(target_vals, 50.0))
        if p965 > max(2500.0, p50 * 5.0) and int(np.sum(target_vals >= p965)) >= 30:
            pos_mask = target_vals >= p965
        else:
            p90 = float(np.nanpercentile(target_vals, 90.0))
            pos_mask = target_vals >= p90

        # C6: Live negative population should be centered around the main live mode (not truncated at bottom 45%)
        low_vals = target_vals[target_vals <= float(np.nanpercentile(target_vals, 90.0))]
        live_mode = float(np.median(low_vals))
        live_mad = float(np.median(np.abs(low_vals - live_mode)))
        live_sigma = max(50.0, 2.5 * 1.4826 * live_mad)

        # Match negative cells to the scatter morphology profile of the positive cells
        sc_cols = _find_scatter_channels_in_df(clean_df)
        fsc_col = sc_cols["fsc"]
        ssc_col = sc_cols["ssc"]
        if fsc_col and ssc_col and int(np.sum(pos_mask)) >= 15:
            fsc_arr = clean_df[fsc_col].to_numpy(dtype=np.float64)
            ssc_arr = clean_df[ssc_col].to_numpy(dtype=np.float64)
            fsc_lo_pos, fsc_hi_pos = np.nanpercentile(fsc_arr[pos_mask], [5, 95])
            ssc_lo_pos, ssc_hi_pos = np.nanpercentile(ssc_arr[pos_mask], [5, 95])
            scatter_matched = (
                (fsc_arr >= fsc_lo_pos)
                & (fsc_arr <= fsc_hi_pos)
                & (ssc_arr >= ssc_lo_pos)
                & (ssc_arr <= ssc_hi_pos)
            )
            cand_neg = scatter_matched & (np.abs(target_vals - live_mode) <= live_sigma)
            if int(np.sum(cand_neg)) >= 30:
                neg_mask = cand_neg
            else:
                neg_mask = np.abs(target_vals - live_mode) <= live_sigma
        else:
            neg_mask = np.abs(target_vals - live_mode) <= live_sigma

    # Fallback quantile gating if neither GMM nor tail gate resolved
    if np.sum(pos_mask) == 0:
        used_method = "quantile"
        pos_mask = target_vals > np.nanpercentile(target_vals, 75)
    if np.sum(neg_mask) == 0:
        neg_mask = target_vals < np.nanpercentile(target_vals, 35)

    pos_mfi = float(np.nanmedian(target_vals[pos_mask])) if np.sum(pos_mask) > 0 else 0.0

    # C1: If external unstained control is used, neg_mfi comes from clean_un_df,
    # while pos_mask and neg_mask remain boolean masks of clean_df (same length!)
    if clean_un_df is not None and target_channel in clean_un_df.columns:
        neg_mfi = float(np.nanmedian(clean_un_df[target_channel]))
        neg_rsd = _robust_sd(clean_un_df[target_channel].to_numpy(dtype=np.float64))
        used_method = "unstained_control"
    else:
        neg_mfi = float(np.nanmedian(target_vals[neg_mask])) if np.sum(neg_mask) > 0 else 0.0
        neg_rsd = _robust_sd(target_vals[neg_mask]) if np.sum(neg_mask) > 0 else 1.0

    delta_mfi = pos_mfi - neg_mfi
    snr = delta_mfi / max(1.0, neg_rsd)
    stain_index = delta_mfi / (2.0 * max(1.0, neg_rsd))

    sc_cols_clean = _find_scatter_channels_in_df(clean_df)
    fsc_col_clean = sc_cols_clean["fsc"]
    pos_fsc_med = (
        float(np.nanmedian(clean_df.loc[pos_mask, fsc_col_clean]))
        if (fsc_col_clean and fsc_col_clean in clean_df.columns and int(np.sum(pos_mask)) > 0)
        else 0.0
    )
    neg_fsc_med = (
        float(np.nanmedian(clean_df.loc[neg_mask, fsc_col_clean]))
        if (
            fsc_col_clean
            and fsc_col_clean in clean_df.columns
            and len(neg_mask) == len(clean_df)
            and int(np.sum(neg_mask)) > 0
        )
        else 0.0
    )

    stats = {
        "channel": target_channel,
        "carrier_type": resolved_carrier,
        "method": used_method,
        "n_initial": n_initial,
        "debris_removed_events": n_debris_removed,
        "debris_removed_pct": round(debris_removed_pct, 2),
        "doublets_removed_events": n_doublets_removed,
        "doublets_removed_pct": round(doublets_removed_pct, 2),
        "n_events": n_events,
        "n_clean_events": n_events,
        "scatter_singlet_kept_pct": round(scatter_kept_pct, 2),
        "n_pos": int(np.sum(pos_mask)),
        "n_neg": int(np.sum(neg_mask)),
        "pos_mfi": pos_mfi,
        "neg_mfi": neg_mfi,
        "neg_rsd": neg_rsd,
        "delta_mfi": delta_mfi,
        "snr": round(snr, 2),
        "stain_index": round(stain_index, 2),
        "pos_fsc_median": round(pos_fsc_med, 1),
        "neg_fsc_median": round(neg_fsc_med, 1),
        "gate_geometry": {
            k: v
            for k, v in gate_geom.items()
            if k not in ("non_sat_mask", "debris_free_mask", "clean_singlet_mask")
        },
        "_pregate_masks": {
            k: gate_geom[k]
            for k in ("non_sat_mask", "debris_free_mask", "clean_singlet_mask")
            if k in gate_geom
        },
    }
    return clean_df, pos_mask, neg_mask, stats


def _estimate_autospill_row(
    clean_df: pd.DataFrame,
    target_channel: str,
    channels: List[str],
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    unstained_df: Optional[pd.DataFrame] = None,
    clamp_negative: bool = True,
) -> Dict[str, float]:
    """Estimate spillover coefficients for one single-stained control via anchored robust linear regression.

    Implements the linear-space regression principle of AutoSpill (Roca et al., Nature Comms 2021):
    anchors the intercept at the robust negative baseline and fits a weighted linear slope
    across the negative and positive event continuum, downweighting extreme leverage outliers.
    """
    valid_mask = pos_mask | neg_mask
    if int(np.sum(valid_mask)) < 20:
        valid_mask = np.ones(len(clean_df), dtype=bool)

    x_all = clean_df.loc[valid_mask, target_channel].to_numpy(dtype=np.float64)
    if unstained_df is not None and target_channel in unstained_df.columns:
        x_base = float(np.nanmedian(unstained_df[target_channel]))
    else:
        x_base = float(np.nanmedian(clean_df.loc[neg_mask, target_channel]))

    dx = x_all - x_base
    # Trim extreme top 0.5% leverage points to guard against pulse pileup
    dx_hi = float(np.nanpercentile(dx, 99.5))
    inlier = np.isfinite(dx) & (dx <= dx_hi)
    dx_in = dx[inlier]
    denom = float(np.dot(dx_in, dx_in))

    row_dict: Dict[str, float] = {}
    for ch_det in channels:
        if ch_det == target_channel:
            row_dict[ch_det] = 1.0
            continue
        if ch_det not in clean_df.columns or denom <= 1e-6:
            row_dict[ch_det] = 0.0
            continue

        y_all = clean_df.loc[valid_mask, ch_det].to_numpy(dtype=np.float64)[inlier]
        if unstained_df is not None and ch_det in unstained_df.columns:
            y_base = float(np.nanmedian(unstained_df[ch_det]))
        else:
            y_base = float(np.nanmedian(clean_df.loc[neg_mask, ch_det]))

        dy = y_all - y_base
        # Initial OLS anchored slope
        slope = float(np.dot(dx_in, dy) / denom)

        # 1-step Huber / IRLS residual reweighting to suppress secondary channel outliers
        resid = dy - slope * dx_in
        mad_res = max(_robust_sd(resid), 1.0)
        huber_k = 2.5 * mad_res
        abs_res = np.abs(resid)
        weights = np.where(abs_res <= huber_k, 1.0, huber_k / np.maximum(abs_res, 1e-6))
        w_denom = float(np.sum(weights * dx_in * dx_in))
        if w_denom > 1e-6:
            slope = float(np.sum(weights * dx_in * dy) / w_denom)

        if clamp_negative:
            slope = max(0.0, slope)
        row_dict[ch_det] = slope

    return row_dict


def compute_spillover_spreading_matrix(
    channels: List[str],
    comp_matrix: CompensationMatrix,
    gated_controls: Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]],
) -> pd.DataFrame:
    """Compute the Spillover Spreading Matrix (SSM) per Nguyen et al. (Cytometry Part A, 2013).

    For each single-stained control i (primary channel) and each secondary detector j:
        SSM[i, j] = sqrt(max(0, rSD(CompPos_j)^2 - rSD(CompNeg_j)^2)) / sqrt(Delta_MFI_i)

    Quantifies the intrinsic photon-counting spread introduced by fluorophore i into detector j
    after mathematically accurate compensation.

    Args:
        channels: Ordered list of detector channels.
        comp_matrix: Active CompensationMatrix used to compensate the single-stain controls.
        gated_controls: Mapping from primary channel name to (clean_df, pos_mask, neg_mask, stats).

    Returns:
        Square DataFrame of shape (N, N) containing SSM coefficients.
    """
    n_ch = len(channels)
    ssm = np.zeros((n_ch, n_ch), dtype=np.float64)

    for i, ch_i in enumerate(channels):
        if ch_i not in gated_controls:
            continue
        clean_df, pos_mask, neg_mask, stats = gated_controls[ch_i]
        if int(np.sum(pos_mask)) < 10 or int(np.sum(neg_mask)) < 10:
            continue

        try:
            comp_df = comp_matrix.apply_dataframe(clean_df)
        except Exception:
            continue

        pos_i = float(np.nanmedian(comp_df.loc[pos_mask, ch_i]))
        neg_i = float(np.nanmedian(comp_df.loc[neg_mask, ch_i]))
        delta_i = pos_i - neg_i
        if delta_i <= 10.0:
            continue

        sqrt_delta_i = np.sqrt(delta_i)
        for j, ch_j in enumerate(channels):
            if i == j or ch_j not in comp_df.columns:
                ssm[i, j] = 0.0
                continue
            sigma_pos = _robust_sd(comp_df.loc[pos_mask, ch_j].to_numpy())
            sigma_neg = _robust_sd(comp_df.loc[neg_mask, ch_j].to_numpy())
            var_diff = max(0.0, sigma_pos * sigma_pos - sigma_neg * sigma_neg)
            ssm[i, j] = float(np.sqrt(var_diff) / sqrt_delta_i)

    return pd.DataFrame(ssm, index=channels, columns=channels)


def verify_compensation_residuals(
    channels: List[str],
    comp_matrix: CompensationMatrix,
    gated_controls: Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Verify post-compensation median alignment across all N x (N-1) primary-secondary pairs.

    For every single-stained control i (after applying `comp_matrix`) and every detector j != i:
      - Residual Z-Score = (Median(CompPos_j) - Median(CompNeg_j)) / rSD(CompNeg_j)
        Measures how many negative-population standard deviations the compensated positive
        median deviates from the negative median. |Z| < 0.5 indicates tight alignment.
      - Residual % of Primary = (Median(CompPos_j) - Median(CompNeg_j)) / Delta_MFI_i * 100%

    Args:
        channels: Ordered list of detector channels.
        comp_matrix: Active CompensationMatrix to verify.
        gated_controls: Mapping from primary channel name to (clean_df, pos_mask, neg_mask, stats).

    Returns:
        Tuple of (residual_z_df, residual_pct_df).
    """
    n_ch = len(channels)
    res_z = np.zeros((n_ch, n_ch), dtype=np.float64)
    res_pct = np.zeros((n_ch, n_ch), dtype=np.float64)

    for i, ch_i in enumerate(channels):
        if ch_i not in gated_controls:
            continue
        clean_df, pos_mask, neg_mask, _ = gated_controls[ch_i]
        if int(np.sum(pos_mask)) < 10 or int(np.sum(neg_mask)) < 10:
            continue

        try:
            comp_df = comp_matrix.apply_dataframe(clean_df)
        except Exception:
            continue

        delta_i = float(
            np.nanmedian(comp_df.loc[pos_mask, ch_i]) - np.nanmedian(comp_df.loc[neg_mask, ch_i])
        )
        if delta_i <= 10.0:
            continue

        for j, ch_j in enumerate(channels):
            if i == j or ch_j not in comp_df.columns:
                continue
            pos_med_j = float(np.nanmedian(comp_df.loc[pos_mask, ch_j]))
            neg_med_j = float(np.nanmedian(comp_df.loc[neg_mask, ch_j]))
            rsd_neg_j = _robust_sd(comp_df.loc[neg_mask, ch_j].to_numpy())
            diff_j = pos_med_j - neg_med_j

            res_z[i, j] = float(diff_j / max(rsd_neg_j, 5.0))
            res_pct[i, j] = float((diff_j / delta_i) * 100.0)

    return (
        pd.DataFrame(res_z, index=channels, columns=channels),
        pd.DataFrame(res_pct, index=channels, columns=channels),
    )


def extract_detector_voltages(fcs: Any) -> Dict[str, Optional[float]]:
    """Extract PMT detector voltages and gains for all channels in FCSData.

    Inspects FCS metadata keywords case-insensitively, supporting FCS 3.0/3.1 conventions
    ($PnV, PnV, $PnG, PnG).

    Args:
        fcs: Parsed FCSData object or compatible metadata container.

    Returns:
        Dict mapping channel detector name (e.g. 'PE-A') to voltage in Volts (float),
        or None if not recorded in FCS metadata.
    """
    voltages: Dict[str, Optional[float]] = {}
    if not hasattr(fcs, "pnn_labels") or not hasattr(fcs, "metadata"):
        return voltages

    meta = {str(k).lower().lstrip("$"): v for k, v in fcs.metadata.items()}
    for i, ch in enumerate(fcs.pnn_labels, 1):
        v_str = meta.get(f"p{i}v") or meta.get(f"p{i}g")
        if v_str is not None:
            try:
                voltages[ch] = float(v_str)
            except (ValueError, TypeError):
                voltages[ch] = None
        else:
            voltages[ch] = None
    return voltages


def audit_pmt_voltage_concordance(
    sample_fcs: Any,
    control_files_mapped: Dict[str, Union[str, Path, Any]],
    channels: List[str],
    tolerance_v: float = 5.0,
    marker_map: Optional[Dict[str, str]] = None,
) -> PMTVoltageAuditReport:
    """Audit PMT detector voltages/gains between single-stained controls and biological sample.

    Determines if single-stained reference controls were acquired under identical optical
    gain settings to the biological sample, per MIFlowCyt guidelines.

    Args:
        sample_fcs: Biological sample FCSData.
        control_files_mapped: Mapping of {channel_name: control_fcs_or_path}.
        channels: Target fluorescent channels to audit.
        tolerance_v: Maximum acceptable absolute voltage discrepancy (|Delta V|) in Volts.
        marker_map: Optional dict mapping channel name to biological marker label.

    Returns:
        PMTVoltageAuditReport with concordance status, per-channel audit table, and diagnostics.
    """
    from flowx.io.parser import parse_fcs

    samp_voltages = extract_detector_voltages(sample_fcs)
    marker_map = marker_map or {}
    records: List[Dict[str, Any]] = []
    mismatched_channels: List[str] = []

    ctrl_voltages_cache: Dict[str, Dict[str, Optional[float]]] = {}

    for ch in channels:
        is_fluo = not any(s in ch.upper() for s in ["FSC", "SSC", "TIME"])
        m_label = marker_map.get(ch, "") or getattr(sample_fcs, "detector_to_marker", {}).get(ch, "")
        s_volt = samp_voltages.get(ch)

        ctrl_ref = control_files_mapped.get(ch)
        ctrl_volt: Optional[float] = None
        ctrl_filename = ""

        if ctrl_ref is not None:
            if isinstance(ctrl_ref, (str, Path)):
                c_path_str = str(ctrl_ref)
                ctrl_filename = Path(c_path_str).name
                if c_path_str not in ctrl_voltages_cache:
                    try:
                        c_fcs = parse_fcs(c_path_str)
                        ctrl_voltages_cache[c_path_str] = extract_detector_voltages(c_fcs)
                    except Exception:
                        ctrl_voltages_cache[c_path_str] = {}
                ctrl_volt = ctrl_voltages_cache[c_path_str].get(ch)
            elif hasattr(ctrl_ref, "metadata"):
                ctrl_filename = getattr(ctrl_ref, "filename", "control")
                if ctrl_filename not in ctrl_voltages_cache:
                    ctrl_voltages_cache[ctrl_filename] = extract_detector_voltages(ctrl_ref)
                ctrl_volt = ctrl_voltages_cache[ctrl_filename].get(ch)

        delta_v: Optional[float] = None
        status = "MATCH"
        if ctrl_volt is None or s_volt is None:
            status = "UNRECORDED"
        else:
            delta_v = round(s_volt - ctrl_volt, 2)
            if abs(delta_v) > tolerance_v:
                if is_fluo:
                    status = "MISMATCH"
                    mismatched_channels.append(ch)
                else:
                    status = "SCATTER_ADJUST"

        records.append({
            "channel": ch,
            "marker": m_label or "-",
            "channel_type": "Fluorescence" if is_fluo else "Scatter",
            "control_file": ctrl_filename or "-",
            "control_voltage": ctrl_volt,
            "sample_voltage": s_volt,
            "voltage_delta": delta_v,
            "status": status,
        })

    audit_df = pd.DataFrame(records)
    has_mismatch = len(mismatched_channels) > 0
    s_name = getattr(sample_fcs, "filename", getattr(sample_fcs, "sample_name", "sample"))

    if has_mismatch:
        summary_msg = (
            f"PMT voltage mismatch detected on {len(mismatched_channels)} fluorescent channel(s) "
            f"({', '.join(mismatched_channels)}) between controls and biological sample '{s_name}'. "
            f"Voltage deltas exceed tolerance threshold (+/-{tolerance_v:.1f} V)."
        )
    else:
        summary_msg = (
            f"PMT voltages are concordant across all {len(channels)} channels between "
            f"controls and biological sample '{s_name}' (within +/-{tolerance_v:.1f} V)."
        )

    return PMTVoltageAuditReport(
        sample_name=s_name,
        has_mismatch=has_mismatch,
        mismatched_channels=mismatched_channels,
        audit_df=audit_df,
        summary_message=summary_msg,
        action_taken="warn",
    )


def adjust_spillover_by_pmt_voltages(
    base_matrix_df: pd.DataFrame,
    control_voltages: Dict[str, Optional[float]],
    sample_voltages: Dict[str, Optional[float]],
    alpha: float = 7.2,
) -> pd.DataFrame:
    """Analytically adjust a spillover matrix from control PMT voltages to sample PMT voltages.

    Based on the photomultiplier tube gain power-law relation G proportional to V^alpha:
        S_{i->j}(sample) = S_{i->j}(control) * (V_{j,sample} / V_{j,ctrl})^alpha * (V_{i,ctrl} / V_{i,sample})^alpha

    Args:
        base_matrix_df: Base spillover matrix DataFrame derived from single-stained controls.
        control_voltages: Dict mapping channel name to control PMT voltage.
        sample_voltages: Dict mapping channel name to target sample PMT voltage.
        alpha: PMT gain power-law exponent (typically 6.8 - 7.5; Hamamatsu / BD default 7.2).

    Returns:
        Adjusted spillover matrix DataFrame matching the target sample's detector gains.
    """
    channels = list(base_matrix_df.index)
    adj_df = base_matrix_df.copy()

    for i_ch in channels:  # Primary detector
        v_i_ctrl = control_voltages.get(i_ch)
        v_i_samp = sample_voltages.get(i_ch)
        for j_ch in channels:  # Secondary detector
            if i_ch == j_ch:
                continue
            v_j_ctrl = control_voltages.get(j_ch)
            v_j_samp = sample_voltages.get(j_ch)

            if (
                v_i_ctrl is not None and v_i_ctrl > 0
                and v_i_samp is not None and v_i_samp > 0
                and v_j_ctrl is not None and v_j_ctrl > 0
                and v_j_samp is not None and v_j_samp > 0
            ):
                gain_ratio_primary = (v_i_ctrl / v_i_samp) ** alpha
                gain_ratio_secondary = (v_j_samp / v_j_ctrl) ** alpha
                scaling = gain_ratio_secondary * gain_ratio_primary
                adj_df.loc[i_ch, j_ch] = float(base_matrix_df.loc[i_ch, j_ch] * scaling)

    return adj_df


def calculate_spillover_from_controls(
    controls_dir_or_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    channels: Optional[List[str]] = None,
    fcs_sample: Optional[Any] = None,
    unstained_control: Optional[Union[str, Path]] = None,
    channel_mapping: Optional[Dict[str, str]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    clamp_negative: bool = True,
    fallback_matrix: Optional[pd.DataFrame] = None,
    random_seed: int = 42,
    qc_manifest: Optional[Union[str, Path, pd.DataFrame]] = None,
    config: Optional[CompensationConfig] = None,
) -> CompensationControlResult:
    """Calculate fluorescence spillover matrix from single-stained compensation controls.

    Automates the flow cytometrist's complete single-stain control workflow:
    1. Discovers and maps single-stained control FCS files to target channels (via manifest or heuristics).
    2. Applies Time QC (flow rate stability & margin saturation removal) to each control file.
    3. Automatically detects Bead vs. Cell carriers and applies scatter/singlet pre-gating.
    4. Computes spillover via both upgraded Median Delta-MFI and AutoSpill robust linear regression,
       selecting the optimal estimator per `config.estimation_method` ('hybrid', 'median_delta_mfi',
       or 'autospill_regression') with iterative linear-space residual refinement.
    5. Computes the Spillover Spreading Matrix (SSM) and verifies N x (N-1) post-compensation
       residual median alignment across all single-stained controls.

    Args:
        controls_dir_or_files: Directory containing control FCS files, or explicit list of files.
        channels: List of fluorescent detector channel names.
        fcs_sample: Optional FCSData object for metadata lookup.
        unstained_control: Optional path to universal unstained / negative control FCS file.
        channel_mapping: Optional manual mapping dictionary of {channel_name: control_filepath}.
        manifest_path: Optional path to CSV or YAML control manifest file.
        clamp_negative: If True, clamps negative spillover noise values (< 0.0) to 0.0.
        fallback_matrix: Optional DataFrame to use for any unmatched channels.
        random_seed: Random seed for clustering.
        qc_manifest: Optional path to or DataFrame of existing Time QC summary CSV (qc_cohort_summary.csv).
        config: Optional CompensationConfig instance for fine-grained cytometry control.

    Returns:
        CompensationControlResult containing matrix, dataframe, SSM, residual matrices, and audit records.
    """
    from flowx.io.parser import parse_fcs
    from flowx.qc.engine import apply_time_qc
    from flowx.qc.models import TimeQCConfig

    cfg = config or CompensationConfig(clamp_negative=clamp_negative, random_seed=random_seed)
    est_method = cfg.estimation_method.lower().strip()

    qc_manifest_df: Optional[pd.DataFrame] = None
    if qc_manifest is not None:
        if isinstance(qc_manifest, (str, Path)) and Path(qc_manifest).is_file():
            qc_manifest_df = pd.read_csv(qc_manifest)
        elif isinstance(qc_manifest, pd.DataFrame):
            qc_manifest_df = qc_manifest

    if manifest_path is not None:
        man_map, man_un = load_control_manifest(manifest_path)
        if unstained_control is None and man_un is not None:
            unstained_control = man_un
        channel_mapping = {**man_map, **(channel_mapping or {})}
        if controls_dir_or_files is None:
            controls_dir_or_files = list(channel_mapping.values())

    if channels is None:
        if fcs_sample is not None and getattr(fcs_sample, "spillover_channels", None):
            channels = list(fcs_sample.spillover_channels)
        elif channel_mapping:
            channels = list(channel_mapping.keys())
        elif fcs_sample is not None and getattr(fcs_sample, "pnn_labels", None):
            channels = [
                ch for ch in fcs_sample.pnn_labels
                if not any(
                    s in ch.lower()
                    for s in (
                        "time",
                        "fsc",
                        "ssc",
                        "fs int",
                        "ss int",
                        "fs peak",
                        "ss peak",
                        "ss tof",
                        "fs.int",
                        "ss.int",
                    )
                )
            ]
        else:
            channels = []

    mapped_files, unmatched = match_control_files(
        controls_dir_or_files=controls_dir_or_files,
        channels=channels,
        fcs_sample=fcs_sample,
        channel_mapping=channel_mapping,
        manifest_path=manifest_path,
    )

    def _align_control_channels_to_sample(ctrl_obj: Any, expected_channels: List[str]) -> None:
        """Align control FCS detector column names (e.g. '_' vs '/') to sample channel names."""
        if not expected_channels or getattr(ctrl_obj, "events_df", None) is None:
            return
        norm_lookup = {
            re.sub(r"[^a-z0-9]", "", str(col).lower()): col
            for col in ctrl_obj.events_df.columns
        }
        rename_map: Dict[str, str] = {}
        for ch in expected_channels:
            if ch in ctrl_obj.events_df.columns:
                continue
            cand = norm_lookup.get(re.sub(r"[^a-z0-9]", "", str(ch).lower()))
            if cand and cand not in rename_map and cand not in expected_channels:
                rename_map[cand] = ch
        if rename_map:
            ctrl_obj.events_df = ctrl_obj.events_df.rename(columns=rename_map)
            if getattr(ctrl_obj, "channel_ranges", None):
                for old_ch, new_ch in rename_map.items():
                    if old_ch in ctrl_obj.channel_ranges and new_ch not in ctrl_obj.channel_ranges:
                        ctrl_obj.channel_ranges[new_ch] = ctrl_obj.channel_ranges[old_ch]

    unstained_df: Optional[pd.DataFrame] = None
    if unstained_control and Path(unstained_control).is_file():
        un_fcs = parse_fcs(str(unstained_control))
        _align_control_channels_to_sample(un_fcs, channels)
        un_raw = un_fcs.events_df
        un_time = un_fcs.get_time_channel()
        if cfg.apply_time_qc_to_controls and un_time and un_time in un_raw.columns and len(un_raw) >= 100:
            try:
                un_tqc = apply_time_qc(un_fcs, config=TimeQCConfig.for_controls())
                un_dec = un_tqc.evaluate_workflow(
                    un_fcs, min_control_events=cfg.min_control_events
                )
                if un_dec.can_proceed and un_dec.clean_events_df is not None:
                    un_raw = un_dec.clean_events_df
            except Exception:
                pass
        if cfg.auto_scatter_singlet_gate and len(un_raw) >= 60:
            un_carrier = detect_control_carrier(
                un_raw, channels[0] if channels else un_fcs.pnn_labels[0]
            )
            _, _, un_clean_mask, _ = _pregate_control_scatter_and_singlets(
                df=un_raw,
                resolved_carrier=un_carrier,
                target_channel=None,
                all_channels=channels,
                channel_ranges=un_fcs.channel_ranges,
                random_seed=cfg.random_seed,
            )
            if int(np.sum(un_clean_mask)) >= 40:
                un_raw = un_raw.loc[un_clean_mask].copy()
        unstained_df = un_raw

    n_ch = len(channels)
    mfi_matrix_df = pd.DataFrame(np.eye(n_ch, dtype=np.float64), index=channels, columns=channels)
    reg_matrix_df = pd.DataFrame(np.eye(n_ch, dtype=np.float64), index=channels, columns=channels)
    active_matrix_df = pd.DataFrame(np.eye(n_ch, dtype=np.float64), index=channels, columns=channels)

    population_stats: Dict[str, Dict[str, Any]] = {}
    gated_controls: Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]] = {}
    control_audit_records: Dict[str, Dict[str, Any]] = {}
    qc_summary_rows: List[Dict[str, Any]] = []

    comp_ref = fallback_matrix
    if (
        comp_ref is None
        and fcs_sample is not None
        and getattr(fcs_sample, "spillover_matrix", None) is not None
    ):
        comp_ref = fcs_sample.spillover_matrix

    for ch_target in channels:
        ctrl_path = mapped_files.get(ch_target)
        marker_lbl = ""
        if fcs_sample is not None and hasattr(fcs_sample, "detector_to_marker"):
            marker_lbl = fcs_sample.detector_to_marker.get(ch_target, "")

        if not ctrl_path or not Path(ctrl_path).is_file():
            if (
                cfg.fallback_policy == "embedded_row"
                and comp_ref is not None
                and ch_target in comp_ref.index
            ):
                for ch_det in channels:
                    if ch_det in comp_ref.columns:
                        val = float(comp_ref.loc[ch_target, ch_det])
                        mfi_matrix_df.loc[ch_target, ch_det] = val
                        reg_matrix_df.loc[ch_target, ch_det] = val
                        active_matrix_df.loc[ch_target, ch_det] = val
            elif cfg.fallback_policy == "strict_error":
                raise FileNotFoundError(
                    f"Missing compensation control file for channel '{ch_target}'"
                )
            continue

        ctrl_fcs = parse_fcs(ctrl_path)
        _align_control_channels_to_sample(ctrl_fcs, channels)
        raw_df = ctrl_fcs.events_df
        total_raw_events = len(raw_df)

        # Step 1: Time QC integration on control file
        time_qc_passed_pct = 100.0
        time_qc_status = "PASS"
        time_qc_action = "PROCEED_CLEAN"
        working_df = raw_df

        ctrl_time = ctrl_fcs.get_time_channel()
        if cfg.apply_time_qc_to_controls and ctrl_time and ctrl_time in raw_df.columns and total_raw_events >= 100:
            try:
                tqc_cfg = TimeQCConfig.for_controls()
                tqc_res = apply_time_qc(ctrl_fcs, config=tqc_cfg)
                decision = tqc_res.evaluate_workflow(
                    ctrl_fcs, min_control_events=cfg.min_control_events
                )
                time_qc_passed_pct = tqc_res.passed_pct
                time_qc_status = tqc_res.qc_status
                time_qc_action = decision.action
                if decision.can_proceed and decision.clean_events_df is not None:
                    working_df = decision.clean_events_df
                elif (
                    not decision.can_proceed
                    and cfg.fallback_policy == "embedded_row"
                    and comp_ref is not None
                    and ch_target in comp_ref.index
                ):
                    for ch_det in channels:
                        if ch_det in comp_ref.columns:
                            val = float(comp_ref.loc[ch_target, ch_det])
                            mfi_matrix_df.loc[ch_target, ch_det] = val
                            reg_matrix_df.loc[ch_target, ch_det] = val
                            active_matrix_df.loc[ch_target, ch_det] = val
                    continue
            except Exception as exc:
                logger.debug(f"Control Time QC skipped for {ctrl_fcs.filename}: {exc}")

        # Determine carrier override if dictionary provided
        if isinstance(cfg.control_carrier_type, dict):
            carrier_arg = cfg.control_carrier_type.get(ch_target, "auto")
        else:
            carrier_arg = cfg.control_carrier_type

        max_rng = ctrl_fcs.channel_ranges.get(ch_target, 262144.0)

        # C3: Ensure unstained cell control is only used for cells, while beads use internal negative
        ctrl_carrier = (
            detect_control_carrier(working_df, ch_target, channel_ranges=ctrl_fcs.channel_ranges)
            if carrier_arg == "auto"
            else carrier_arg.lower()
        )
        use_un_for_ctrl = unstained_df if (cfg.negative_reference != "internal" and ctrl_carrier == "cells") else None

        clean_ctrl_df, pos_mask, neg_mask, stats = gate_control_populations(
            df=working_df,
            target_channel=ch_target,
            method="auto",
            unstained_df=use_un_for_ctrl,
            random_seed=cfg.random_seed,
            carrier_type=carrier_arg,
            auto_scatter_singlet_gate=cfg.auto_scatter_singlet_gate,
            max_range=max_rng,
            all_channels=channels,
            channel_ranges=ctrl_fcs.channel_ranges,
        )

        pregate_masks = stats.pop("_pregate_masks", {})
        gate_geom = stats.pop("gate_geometry", {"carrier_type": stats["carrier_type"]})

        stats["control_file"] = Path(ctrl_path).name
        stats["marker"] = marker_lbl
        stats["raw_events"] = total_raw_events
        stats["time_qc_passed_events"] = len(working_df)
        stats["time_qc_passed_pct"] = round(time_qc_passed_pct, 2)
        stats["time_qc_status"] = time_qc_status
        stats["time_qc_action"] = time_qc_action

        gated_controls[ch_target] = (clean_ctrl_df, pos_mask, neg_mask, stats)

        delta_target = stats["delta_mfi"]
        if delta_target <= 0.0:
            population_stats[ch_target] = stats
            continue

        # Method 1: Upgraded Median Delta-MFI row
        for ch_det in channels:
            if ch_det == ch_target:
                mfi_matrix_df.loc[ch_target, ch_det] = 1.0
            elif ch_det in clean_ctrl_df.columns:
                pos_det = float(np.nanmedian(clean_ctrl_df.loc[pos_mask, ch_det]))
                if (
                    unstained_df is not None
                    and cfg.negative_reference != "internal"
                    and stats["carrier_type"] == "cells"
                    and ch_det in unstained_df.columns
                ):
                    neg_det = float(np.nanmedian(unstained_df[ch_det]))
                else:
                    neg_det = float(np.nanmedian(clean_ctrl_df.loc[neg_mask, ch_det]))

                delta_det = pos_det - neg_det
                spill = delta_det / delta_target
                if cfg.clamp_negative:
                    spill = max(0.0, spill)
                mfi_matrix_df.loc[ch_target, ch_det] = float(spill)
            else:
                mfi_matrix_df.loc[ch_target, ch_det] = 0.0

        # Method 2: AutoSpill Robust Linear Regression row
        reg_row = _estimate_autospill_row(
            clean_df=clean_ctrl_df,
            target_channel=ch_target,
            channels=channels,
            pos_mask=pos_mask,
            neg_mask=neg_mask,
            unstained_df=use_un_for_ctrl,
            clamp_negative=cfg.clamp_negative,
        )
        for ch_det, val in reg_row.items():
            reg_matrix_df.loc[ch_target, ch_det] = float(val)

        # Select row for active matrix based on estimation_method
        if est_method == "median_delta_mfi":
            selected_estimator = "median_delta_mfi"
            active_matrix_df.loc[ch_target] = mfi_matrix_df.loc[ch_target]
        elif est_method == "autospill_regression":
            selected_estimator = "autospill_regression"
            active_matrix_df.loc[ch_target] = reg_matrix_df.loc[ch_target]
        else:
            # Hybrid: Use median_delta_mfi for clean bimodal beads, and autospill_regression
            # for cell controls or skewed distributions where regression across intensity gradient excels
            if stats["carrier_type"] == "cells" or stats["method"] != "gmm":
                selected_estimator = "autospill_regression"
                active_matrix_df.loc[ch_target] = reg_matrix_df.loc[ch_target]
            else:
                selected_estimator = "median_delta_mfi"
                active_matrix_df.loc[ch_target] = mfi_matrix_df.loc[ch_target]

        stats["estimator_used"] = selected_estimator
        population_stats[ch_target] = stats

        # Cache compact snapshot for control gating and scatter/singlet audit plots
        rng = np.random.RandomState(cfg.random_seed)
        n_clean = len(clean_ctrl_df)
        sub_idx = rng.choice(n_clean, size=min(n_clean, 1500), replace=False)

        n_work = len(working_df)
        raw_sub_idx = rng.choice(n_work, size=min(n_work, 1800), replace=False)
        deb_free_full = pregate_masks.get("debris_free_mask", np.ones(n_work, dtype=bool))
        clean_sing_full = pregate_masks.get("clean_singlet_mask", np.ones(n_work, dtype=bool))

        control_audit_records[ch_target] = {
            "channel": ch_target,
            "marker": marker_lbl,
            "control_file": Path(ctrl_path).name,
            "carrier_type": stats["carrier_type"],
            "estimator_used": selected_estimator,
            "fsc_vals": (
                clean_ctrl_df["FSC-A"].to_numpy()[sub_idx]
                if "FSC-A" in clean_ctrl_df.columns
                else np.zeros(len(sub_idx))
            ),
            "ssc_vals": (
                clean_ctrl_df["SSC-A"].to_numpy()[sub_idx]
                if "SSC-A" in clean_ctrl_df.columns
                else np.zeros(len(sub_idx))
            ),
            "raw_fsc_vals": (
                working_df["FSC-A"].to_numpy()[raw_sub_idx]
                if "FSC-A" in working_df.columns
                else np.zeros(len(raw_sub_idx))
            ),
            "raw_ssc_vals": (
                working_df["SSC-A"].to_numpy()[raw_sub_idx]
                if "SSC-A" in working_df.columns
                else np.zeros(len(raw_sub_idx))
            ),
            "raw_debris_mask": ~deb_free_full[raw_sub_idx],
            "raw_doublet_mask": deb_free_full[raw_sub_idx] & ~clean_sing_full[raw_sub_idx],
            "raw_clean_mask": clean_sing_full[raw_sub_idx],
            "gate_geometry": gate_geom,
            "target_vals": clean_ctrl_df[ch_target].to_numpy(),
            "pos_mask": pos_mask,
            "neg_mask": neg_mask,
            "stats": stats,
        }

    # Iterative linear-space residual refinement for AutoSpill / Hybrid (left-multiplication S_new = (I + R) @ S)
    if est_method in ("autospill_regression", "hybrid") and cfg.autospill_iterations >= 1:
        for _ in range(cfg.autospill_iterations):
            try:
                cur_comp = CompensationMatrix(detectors=channels, matrix=active_matrix_df.values)
                cur_z_df, cur_pct_df = verify_compensation_residuals(
                    channels, cur_comp, gated_controls
                )
                cur_z_norm = float(np.mean(np.abs(cur_z_df.values)))

                residual_mat = cur_pct_df.values / 100.0
                updated = (np.eye(n_ch) + 0.5 * residual_mat) @ active_matrix_df.values
                for i in range(n_ch):
                    if updated[i, i] > 0:
                        updated[i, :] /= updated[i, i]
                if cfg.clamp_negative:
                    off_diag = ~np.eye(n_ch, dtype=bool)
                    updated[off_diag] = np.maximum(0.0, updated[off_diag])

                cand_comp = CompensationMatrix(detectors=channels, matrix=updated)
                cand_z_df, _ = verify_compensation_residuals(channels, cand_comp, gated_controls)
                cand_z_norm = float(np.mean(np.abs(cand_z_df.values)))
                if cand_z_norm <= cur_z_norm:
                    active_matrix_df = pd.DataFrame(updated, index=channels, columns=channels)
                else:
                    break
            except Exception:
                break

    # Apply any manual user overrides
    if cfg.manual_overrides:
        temp_comp = CompensationMatrix(detectors=channels, matrix=active_matrix_df.values)
        temp_comp = temp_comp.with_overrides(cfg.manual_overrides)
        active_matrix_df = temp_comp.dataframe

    comp_obj = CompensationMatrix(detectors=channels, matrix=active_matrix_df.values)

    # Compute Spillover Spreading Matrix (SSM) and Residual Verification Matrices
    ssm_df = (
        compute_spillover_spreading_matrix(channels, comp_obj, gated_controls)
        if cfg.compute_ssm
        else pd.DataFrame(np.zeros((n_ch, n_ch)), index=channels, columns=channels)
    )
    res_z_df, res_pct_df = verify_compensation_residuals(channels, comp_obj, gated_controls)

    # Summarize per-control QC & Residual Alignment metrics
    for ch_target in channels:
        st = population_stats.get(ch_target, {})
        if not st:
            continue
        row_z = res_z_df.loc[ch_target].drop(labels=[ch_target]).abs()
        row_pct = res_pct_df.loc[ch_target].drop(labels=[ch_target]).abs()
        max_abs_z = float(row_z.max()) if len(row_z) > 0 else 0.0
        max_z_ch = str(row_z.idxmax()) if len(row_z) > 0 else "N/A"
        max_abs_pct = float(row_pct.max()) if len(row_pct) > 0 else 0.0

        st["max_residual_abs_z"] = round(max_abs_z, 3)
        st["max_residual_channel"] = max_z_ch
        st["max_residual_abs_pct"] = round(max_abs_pct, 3)
        st["residual_status"] = (
            "ALIGNED" if max_abs_z <= cfg.residual_z_threshold else "MINOR_SPREAD_OR_TILT"
        )
        qc_summary_rows.append(st)

    control_qc_summary_df = pd.DataFrame(qc_summary_rows)

    # Compare with embedded cytometer matrix if available
    emb_corr: Optional[float] = None
    emb_mae: Optional[float] = None
    diff_emb_df: Optional[pd.DataFrame] = None
    if comp_ref is not None:
        try:
            ref_sub = comp_ref.reindex(index=channels, columns=channels).fillna(0.0).values
            calc_sub = active_matrix_df.values
            off_diag = ~np.eye(n_ch, dtype=bool)
            emb_mae = float(np.mean(np.abs(calc_sub[off_diag] - ref_sub[off_diag])))
            corr_mat = np.corrcoef(calc_sub[off_diag], ref_sub[off_diag])
            emb_corr = float(corr_mat[0, 1])
            diff_emb_df = pd.DataFrame(
                (calc_sub - ref_sub) * 100.0, index=channels, columns=channels
            )
        except Exception:
            pass

    v_audit: Optional[PMTVoltageAuditReport] = None
    if fcs_sample is not None and mapped_files:
        try:
            v_audit = audit_pmt_voltage_concordance(
                sample_fcs=fcs_sample,
                control_files_mapped=mapped_files,
                channels=channels,
                tolerance_v=cfg.voltage_tolerance_v,
                marker_map=getattr(fcs_sample, "detector_to_marker", None),
            )
        except Exception as exc:
            logger.debug(f"PMT voltage concordance check failed: {exc}")

    return CompensationControlResult(
        matrix=comp_obj,
        dataframe=active_matrix_df,
        control_files_mapped=mapped_files,
        unmatched_channels=unmatched,
        population_stats=population_stats,
        condition_number=comp_obj.condition_number,
        embedded_correlation=emb_corr,
        embedded_mae=emb_mae,
        estimation_method_used=est_method,
        mfi_matrix_df=mfi_matrix_df,
        regression_matrix_df=reg_matrix_df,
        ssm_dataframe=ssm_df,
        residual_z_dataframe=res_z_df,
        residual_pct_dataframe=res_pct_df,
        diff_from_embedded_dataframe=diff_emb_df,
        control_qc_summary_df=control_qc_summary_df,
        control_audit_records=control_audit_records,
        gated_controls=gated_controls,
        voltage_audit_report=v_audit,
        voltage_audit_df=v_audit.audit_df if v_audit else None,
        has_voltage_mismatch=v_audit.has_mismatch if v_audit else False,
        voltage_mismatch_channels=v_audit.mismatched_channels if v_audit else [],
    )


def _evaluate_estimator_matrix(
    base_res: CompensationControlResult,
    matrix_df: pd.DataFrame,
    method_name: str,
    comp_ref: Optional[pd.DataFrame] = None,
) -> CompensationControlResult:
    """Evaluate residual alignment, SSM, and embedded correlation for an alternative estimator matrix.

    Reuses cached gated controls from a primary CompensationControlResult, avoiding redundant file I/O.
    """
    channels = list(matrix_df.index)
    n_ch = len(channels)
    comp_obj = CompensationMatrix(detectors=channels, matrix=matrix_df.values)

    res_z_df = pd.DataFrame(np.zeros((n_ch, n_ch)), index=channels, columns=channels)
    res_pct_df = pd.DataFrame(np.zeros((n_ch, n_ch)), index=channels, columns=channels)
    ssm_df = pd.DataFrame(np.zeros((n_ch, n_ch)), index=channels, columns=channels)

    if base_res.gated_controls:
        try:
            res_z_df, res_pct_df = verify_compensation_residuals(
                channels, comp_obj, base_res.gated_controls
            )
            ssm_df = compute_spillover_spreading_matrix(
                channels, comp_obj, base_res.gated_controls
            )
        except Exception as exc:
            logger.debug(f"Residual evaluation failed for {method_name}: {exc}")

    emb_corr: Optional[float] = None
    emb_mae: Optional[float] = None
    diff_emb_df: Optional[pd.DataFrame] = None
    if comp_ref is not None:
        try:
            ref_sub = comp_ref.reindex(index=channels, columns=channels).fillna(0.0).values
            calc_sub = matrix_df.values
            off_diag = ~np.eye(n_ch, dtype=bool)
            emb_mae = float(np.mean(np.abs(calc_sub[off_diag] - ref_sub[off_diag])))
            corr_mat = np.corrcoef(calc_sub[off_diag], ref_sub[off_diag])
            emb_corr = float(corr_mat[0, 1])
            diff_emb_df = pd.DataFrame(
                (calc_sub - ref_sub) * 100.0, index=channels, columns=channels
            )
        except Exception:
            pass

    return CompensationControlResult(
        matrix=comp_obj,
        dataframe=matrix_df,
        control_files_mapped=base_res.control_files_mapped,
        unmatched_channels=base_res.unmatched_channels,
        population_stats=base_res.population_stats,
        condition_number=comp_obj.condition_number,
        embedded_correlation=emb_corr,
        embedded_mae=emb_mae,
        estimation_method_used=method_name,
        mfi_matrix_df=base_res.mfi_matrix_df,
        regression_matrix_df=base_res.regression_matrix_df,
        ssm_dataframe=ssm_df,
        residual_z_dataframe=res_z_df,
        residual_pct_dataframe=res_pct_df,
        diff_from_embedded_dataframe=diff_emb_df,
        control_qc_summary_df=base_res.control_qc_summary_df,
        control_audit_records=base_res.control_audit_records,
        gated_controls=base_res.gated_controls,
        voltage_audit_report=base_res.voltage_audit_report,
        voltage_audit_df=base_res.voltage_audit_df,
        has_voltage_mismatch=base_res.has_voltage_mismatch,
        voltage_mismatch_channels=base_res.voltage_mismatch_channels,
    )



def run_dataset_compensation(
    dataset_dir_or_sample: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    controls_dir: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    unstained_control: Optional[Union[str, Path]] = None,
    qc_manifest_path: Optional[Union[str, Path]] = None,
    config: Optional[CompensationConfig] = None,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end Time QC + Fluorescence Compensation + Verification Suite on a dataset.

    Integrates Step 1.1/1.2 (Time QC on controls and biological sample) with Step 1.3
    (Spillover calculation via Median Delta-MFI, AutoSpill Regression, and Hybrid estimators,
    Spillover Spreading Matrix calculation, N x (N-1) Residual Alignment verification,
    and publication-grade visual diagnostics).

    Args:
        dataset_dir_or_sample: Path to dataset directory (e.g. 'data/conventional/somasundaram_2020_frfcmz2zh')
            or specific biological sample FCS file.
        output_dir: Target directory for tables, diagnostic plots, and MIFlowCyt report.
            If None, defaults to 'outputs/<dataset_name>_compensation'.
        controls_dir: Optional explicit path to single-stained controls directory.
        manifest_path: Optional path to CSV/YAML control manifest.
        unstained_control: Optional path to unstained control FCS file.
        qc_manifest_path: Optional path to existing Time QC summary CSV (qc_cohort_summary.csv)
            to reuse pre-computed QC decisions and skip cohort time binning.
        config: Optional CompensationConfig configuration override.
        generate_plots: If True, renders the complete 6-part cytometrist verification gallery.

    Returns:
        Dictionary containing CompensationControlResult objects, file paths, and summary metrics.
    """
    import time
    from datetime import datetime
    from flowx.io.parser import parse_fcs
    from flowx.qc.engine import apply_time_qc
    from flowx.qc.models import TimeQCConfig
    from flowx.qc.runner import detect_sample_type_from_path, run_cohort_time_qc
    from flowx.visualization.comp_plots import (
        plot_all_active_spillover_pairs_atlas,
        plot_control_gating_audit_grid,
        plot_control_scatter_singlet_audit_grid,
        plot_estimator_comparison,
        plot_nxn_compensation_matrix,
        plot_residual_alignment_diagnostics,
        plot_sample_top_pairs_gallery,
        plot_spillover_and_diff_matrix,
        plot_spillover_spreading_matrix,
    )

    t0 = time.perf_counter()
    cfg = config or CompensationConfig()
    target_path = Path(dataset_dir_or_sample).resolve()

    if target_path.is_dir():
        dataset_name = target_path.name
        all_fcs = sorted(target_path.glob("*.fcs"))
        if not all_fcs:
            all_fcs = sorted(target_path.rglob("*.fcs"))
        sample_files = [f for f in all_fcs if detect_sample_type_from_path(f) == "sample"]
        ctrl_files = [f for f in all_fcs if detect_sample_type_from_path(f) == "control"]
        if not sample_files and all_fcs:
            sample_files = [all_fcs[-1]]
        sample_fcs_path = sample_files[0]
        effective_controls: Union[str, Path, List[Union[str, Path]]] = (
            controls_dir if controls_dir is not None else (ctrl_files if ctrl_files else target_path)
        )
    elif target_path.is_file():
        sample_fcs_path = target_path
        dataset_name = target_path.parent.name
        effective_controls = controls_dir if controls_dir is not None else target_path.parent
        all_fcs = sorted(target_path.parent.glob("*.fcs"))
    else:
        raise FileNotFoundError(f"Dataset or sample path not found: {target_path}")

    out_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else Path(f"outputs/{dataset_name}_compensation").resolve()
    )
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    qc_dir = out_dir / "time_qc"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run or Ingest Cohort Time QC
    if qc_manifest_path and Path(qc_manifest_path).is_file():
        logger.info(f"Step 1.1/1.2: Ingesting existing Time QC manifest from '{qc_manifest_path}' (skipping re-binning)...")
        qc_summary_df = pd.read_csv(qc_manifest_path)
        shutil_df_path = tables_dir / "time_qc_cohort_summary.csv"
        qc_summary_df.to_csv(shutil_df_path, index=False)
        qc_csv_path = shutil_df_path
        qc_report_path = qc_dir / "qc_time_report.md"

        # Check if clean_fcs_path was exported in manifest
        if "clean_fcs_path" in qc_summary_df.columns:
            samp_matches = qc_summary_df[qc_summary_df["sample_type"] == "sample"]
            if not samp_matches.empty:
                first_clean = str(samp_matches.iloc[0].get("clean_fcs_path", "")).strip()
                if first_clean and first_clean != "nan" and Path(first_clean).is_file():
                    sample_fcs_path = Path(first_clean)
                    logger.info(f"Loaded clean, time-gated biological sample from '{sample_fcs_path}'")
    else:
        qc_summary_df, qc_csv_path, qc_report_path = run_cohort_time_qc(
            file_list_or_paths=all_fcs,
            output_dir=qc_dir,
            generate_plots=generate_plots,
            min_control_events=cfg.min_control_events,
        )
        shutil_df_path = tables_dir / "time_qc_cohort_summary.csv"
        qc_summary_df.to_csv(shutil_df_path, index=False)

    # 2. Parse biological sample and apply sample Time QC
    sample_fcs = parse_fcs(sample_fcs_path)
    sample_tqc = apply_time_qc(sample_fcs, config=TimeQCConfig.for_sample())
    sample_decision = sample_tqc.evaluate_workflow(sample_fcs)
    sample_passed_pct = sample_tqc.passed_pct

    if qc_manifest_path and "clean_fcs_path" in qc_summary_df.columns:
        clean_sample_df = sample_fcs.events_df.copy()
        samp_match = qc_summary_df[qc_summary_df["sample_type"] == "sample"]
        if not samp_match.empty and "passed_pct" in samp_match.columns:
            sample_passed_pct = float(samp_match.iloc[0]["passed_pct"])
    else:
        clean_sample_df = (
            sample_decision.clean_events_df
            if (cfg.apply_time_qc_to_sample and sample_decision.clean_events_df is not None)
            else sample_fcs.events_df.copy()
        )

    # 3. Compute Active Spillover Matrix (calculating MFI & AutoSpill matrices concurrently)
    active_res = calculate_spillover_from_controls(
        controls_dir_or_files=effective_controls,
        fcs_sample=sample_fcs,
        unstained_control=unstained_control,
        manifest_path=manifest_path,
        qc_manifest=qc_summary_df,
        config=cfg,
    )

    comp_ref = sample_fcs.spillover_matrix
    active_method = cfg.estimation_method.lower().strip()
    if active_method == "median_delta_mfi":
        mfi_res = active_res
        reg_res = _evaluate_estimator_matrix(
            active_res, active_res.regression_matrix_df, "autospill_regression", comp_ref
        )
        hyb_res = _evaluate_estimator_matrix(
            active_res, active_res.dataframe, "hybrid", comp_ref
        )
    elif active_method == "autospill_regression":
        reg_res = active_res
        mfi_res = _evaluate_estimator_matrix(
            active_res, active_res.mfi_matrix_df, "median_delta_mfi", comp_ref
        )
        hyb_res = _evaluate_estimator_matrix(
            active_res, active_res.dataframe, "hybrid", comp_ref
        )
    else:  # hybrid
        hyb_res = active_res
        mfi_res = _evaluate_estimator_matrix(
            active_res, active_res.mfi_matrix_df, "median_delta_mfi", comp_ref
        )
        reg_res = _evaluate_estimator_matrix(
            active_res, active_res.regression_matrix_df, "autospill_regression", comp_ref
        )

    # 4. Export CSV Tables
    exported_tables: Dict[str, str] = {
        "time_qc_cohort_summary": str(shutil_df_path),
    }
    p_active = tables_dir / "spillover_matrix_calculated.csv"
    active_res.dataframe.to_csv(p_active)
    exported_tables["spillover_matrix_calculated"] = str(p_active)

    p_hyb = tables_dir / "spillover_matrix_hybrid.csv"
    hyb_res.dataframe.to_csv(p_hyb)
    exported_tables["spillover_matrix_hybrid"] = str(p_hyb)

    p_mfi = tables_dir / "spillover_matrix_median_delta_mfi.csv"
    mfi_res.dataframe.to_csv(p_mfi)
    exported_tables["spillover_matrix_median_delta_mfi"] = str(p_mfi)

    p_reg = tables_dir / "spillover_matrix_autospill_regression.csv"
    reg_res.dataframe.to_csv(p_reg)
    exported_tables["spillover_matrix_autospill_regression"] = str(p_reg)

    if sample_fcs.spillover_matrix is not None:
        p_emb = tables_dir / "spillover_matrix_embedded.csv"
        sample_fcs.spillover_matrix.to_csv(p_emb)
        exported_tables["spillover_matrix_embedded"] = str(p_emb)

    if active_res.diff_from_embedded_dataframe is not None:
        p_diff = tables_dir / "spillover_difference_vs_embedded_pct.csv"
        active_res.diff_from_embedded_dataframe.to_csv(p_diff)
        exported_tables["spillover_difference_vs_embedded_pct"] = str(p_diff)

    if active_res.ssm_dataframe is not None:
        p_ssm = tables_dir / "spillover_spreading_matrix_ssm.csv"
        active_res.ssm_dataframe.to_csv(p_ssm)
        exported_tables["spillover_spreading_matrix_ssm"] = str(p_ssm)

    if active_res.residual_z_dataframe is not None:
        p_res_z = tables_dir / "post_comp_residual_z_scores.csv"
        active_res.residual_z_dataframe.to_csv(p_res_z)
        exported_tables["post_comp_residual_z_scores"] = str(p_res_z)

    if active_res.residual_pct_dataframe is not None:
        p_res_pct = tables_dir / "post_comp_residual_pct_primary.csv"
        active_res.residual_pct_dataframe.to_csv(p_res_pct)
        exported_tables["post_comp_residual_pct_primary"] = str(p_res_pct)

    if active_res.control_qc_summary_df is not None:
        p_ctrl_qc = tables_dir / "control_gating_and_qc_audit.csv"
        active_res.control_qc_summary_df.to_csv(p_ctrl_qc, index=False)
        exported_tables["control_gating_and_qc_audit"] = str(p_ctrl_qc)

    # Export method benchmark table
    n_ch = len(active_res.dataframe)
    off_diag = ~np.eye(n_ch, dtype=bool)
    bench_rows = []
    for m_name, r_obj in [
        ("median_delta_mfi", mfi_res),
        ("autospill_regression", reg_res),
        ("hybrid", hyb_res),
    ]:
        z_arr = np.abs(r_obj.residual_z_dataframe.values[off_diag])
        pct_arr = np.abs(r_obj.residual_pct_dataframe.values[off_diag])
        diff_arr = (
            np.abs(r_obj.diff_from_embedded_dataframe.values[off_diag])
            if r_obj.diff_from_embedded_dataframe is not None
            else np.zeros_like(z_arr)
        )
        bench_rows.append(
            {
                "estimation_method": m_name,
                "condition_number": round(r_obj.condition_number, 3),
                "embedded_pearson_r": round(r_obj.embedded_correlation or 0.0, 6),
                "embedded_mae_pct": round((r_obj.embedded_mae or 0.0) * 100.0, 4),
                "embedded_max_diff_pct": round(float(np.max(diff_arr)), 3),
                "mean_residual_abs_z": round(float(np.mean(z_arr)), 4),
                "p95_residual_abs_z": round(float(np.percentile(z_arr, 95)), 4),
                "pairs_within_0_5_rsd_pct": round(float(np.mean(z_arr <= 0.5) * 100.0), 2),
                "mean_residual_abs_pct_primary": round(float(np.mean(pct_arr)), 4),
            }
        )
    bench_df = pd.DataFrame(bench_rows)
    p_bench = tables_dir / "estimator_benchmark_comparison.csv"
    bench_df.to_csv(p_bench, index=False)
    exported_tables["estimator_benchmark_comparison"] = str(p_bench)

    # 4.5 PMT Detector Voltage & Gain Concordance Audit
    marker_map = getattr(sample_fcs, "detector_to_marker", {}) or {}
    voltage_audit = audit_pmt_voltage_concordance(
        sample_fcs=sample_fcs,
        control_files_mapped=active_res.control_files_mapped,
        channels=list(active_res.dataframe.index),
        tolerance_v=cfg.voltage_tolerance_v,
        marker_map=marker_map,
    )
    if voltage_audit.audit_df is not None:
        p_vaudit = tables_dir / "pmt_voltage_concordance_audit.csv"
        voltage_audit.audit_df.to_csv(p_vaudit, index=False)
        exported_tables["pmt_voltage_concordance_audit"] = str(p_vaudit)

    # 5. Apply Active Compensation Matrix to Biological Sample (with Voltage Mismatch Handling)
    sample_comp_matrix = active_res.matrix
    policy = (cfg.voltage_mismatch_policy or "warn").lower().strip()

    ctrl_volts_cache: Dict[str, Optional[float]] = {}
    for ch_name, c_path in active_res.control_files_mapped.items():
        try:
            c_fcs = parse_fcs(c_path)
            ctrl_volts_cache.update({k: v for k, v in extract_detector_voltages(c_fcs).items() if v is not None})
        except Exception:
            pass

    if voltage_audit.has_mismatch:
        logger.warning(
            f"PMT voltage mismatch detected between single-stained controls and sample "
            f"'{sample_fcs.sample_name}' across {len(voltage_audit.mismatched_channels)} channel(s): "
            f"{voltage_audit.mismatched_channels}. Policy: '{policy}'."
        )
        if policy == "strict_error":
            raise ValueError(
                f"PMT voltage mismatch detected on channels {voltage_audit.mismatched_channels} "
                f"under strict policy 'strict_error'."
            )
        elif policy == "fallback_embedded" and sample_fcs.spillover_matrix is not None:
            logger.info(
                f"Applying sample embedded spillover matrix for '{sample_fcs.sample_name}' "
                f"due to PMT voltage mismatch with single-stained controls."
            )
            emb_sub = sample_fcs.spillover_matrix.reindex(
                index=active_res.dataframe.index,
                columns=active_res.dataframe.index,
            ).fillna(0.0)
            sample_comp_matrix = CompensationMatrix(
                detectors=list(active_res.dataframe.index),
                matrix=emb_sub.values,
            )
            voltage_audit.action_taken = "fallback_embedded (applied sample's embedded spillover matrix)"
        elif policy == "scale_power_law":
            samp_volts = extract_detector_voltages(sample_fcs)
            scaled_matrix_df = adjust_spillover_by_pmt_voltages(
                base_matrix_df=active_res.dataframe,
                control_voltages=ctrl_volts_cache,
                sample_voltages=samp_volts,
                alpha=cfg.pmt_gain_exponent_alpha,
            )
            sample_comp_matrix = CompensationMatrix(
                detectors=list(active_res.dataframe.index),
                matrix=scaled_matrix_df.values,
            )
            p_scaled_mat = tables_dir / "spillover_matrix_voltage_adjusted.csv"
            scaled_matrix_df.to_csv(p_scaled_mat)
            exported_tables["spillover_matrix_voltage_adjusted"] = str(p_scaled_mat)
            voltage_audit.action_taken = f"scale_power_law (scaled control matrix via PMT alpha={cfg.pmt_gain_exponent_alpha})"
        else:
            voltage_audit.action_taken = "warn (proceeded with calculated control matrix)"
    else:
        voltage_audit.action_taken = "concordant (proceeded with calculated control matrix)"

    comp_sample_df = sample_comp_matrix.apply_dataframe(clean_sample_df)
    tandem_alignment_rows: List[Dict[str, Any]] = []

    if cfg.align_sample_tandem_mismatch:
        try:
            from flowx.visualization.comp_plots import _pregate_biological_sample_for_comp_check

            gated_u, gated_c, _ = _pregate_biological_sample_for_comp_check(
                clean_sample_df, comp_sample_df, marker_map=sample_fcs.detector_to_marker
            )
            sing_u, _, _ = _pregate_biological_sample_for_comp_check(
                clean_sample_df, comp_sample_df, marker_map=None
            )
            if len(gated_u) >= 2000:
                adj_df = active_res.dataframe.copy()
                ch_list = list(adj_df.index)
                for _pass in range(2):
                    cur_c_live = CompensationMatrix(
                        detectors=ch_list, matrix=adj_df.values
                    ).apply_dataframe(gated_u)
                    cur_c_sing = CompensationMatrix(
                        detectors=ch_list, matrix=adj_df.values
                    ).apply_dataframe(sing_u)
                    for p_ch in ch_list:
                        for s_ch in ch_list:
                            if p_ch == s_ch or float(adj_df.loc[p_ch, s_ch]) < 0.35:
                                continue
                            is_viab = ("355: Em 450" in p_ch) or ("355: Em 450" in s_ch)
                            ref_u = sing_u if is_viab else gated_u
                            cur_c_df = cur_c_sing if is_viab else cur_c_live
                            x_v = cur_c_df[p_ch].to_numpy()
                            y_v = cur_c_df[s_ch].to_numpy()
                            y_med = float(np.nanmedian(y_v))
                            y_rsd = float(np.nanmedian(np.abs(y_v - y_med))) * 1.4826
                            y_gate = y_v <= (y_med + 2.5 * max(y_rsd, 150.0))
                            neg_m = (
                                y_gate
                                & (x_v >= float(np.nanpercentile(x_v, 10)))
                                & (x_v <= float(np.nanpercentile(x_v, 55)))
                            )
                            pos_m = (
                                y_gate
                                & (x_v >= float(np.nanpercentile(x_v, 90)))
                                & (x_v <= float(np.nanpercentile(x_v, 99.2)))
                            )
                            if np.sum(neg_m) < 50 or np.sum(pos_m) < 30:
                                continue
                            dx = float(np.nanmedian(x_v[pos_m]) - np.nanmedian(x_v[neg_m]))
                            dy = float(np.nanmedian(y_v[pos_m]) - np.nanmedian(y_v[neg_m]))
                            rsd_neg = float(
                                np.nanmedian(np.abs(y_v[neg_m] - np.nanmedian(y_v[neg_m]))) * 1.4826
                            )
                            z_pre = dy / max(rsd_neg, 50.0)
                            if dy <= -450.0 and z_pre <= -2.5 and dx > 250.0:
                                zero_df = adj_df.copy()
                                zero_df.loc[p_ch, s_ch] = 0.0
                                c_zero = CompensationMatrix(
                                    detectors=ch_list, matrix=zero_df.values
                                ).apply_dataframe(ref_u)
                                xz = c_zero[p_ch].to_numpy()
                                yz = c_zero[s_ch].to_numpy()
                                yz_med = float(np.nanmedian(yz))
                                yz_rsd = float(np.nanmedian(np.abs(yz - yz_med))) * 1.4826
                                yz_gate = yz <= (yz_med + 2.5 * max(yz_rsd, 150.0))
                                neg_mz = (
                                    yz_gate
                                    & (xz >= float(np.nanpercentile(xz, 10)))
                                    & (xz <= float(np.nanpercentile(xz, 55)))
                                )
                                pos_mz = (
                                    yz_gate
                                    & (xz >= float(np.nanpercentile(xz, 90)))
                                    & (xz <= float(np.nanpercentile(xz, 99.2)))
                                )
                                if np.sum(neg_mz) < 30 or np.sum(pos_mz) < 20:
                                    continue
                                dx_z = max(
                                    1.0, float(np.nanmedian(xz[pos_mz]) - np.nanmedian(xz[neg_mz]))
                                )
                                dy_z = max(
                                    0.0, float(np.nanmedian(yz[pos_mz]) - np.nanmedian(yz[neg_mz]))
                                )
                                old_s = float(active_res.dataframe.loc[p_ch, s_ch])
                                new_s = float(np.clip(dy_z / dx_z, 0.0, old_s))
                                adj_df.loc[p_ch, s_ch] = new_s
                                tandem_alignment_rows.append(
                                    {
                                        "primary_channel": p_ch,
                                        "primary_marker": sample_fcs.detector_to_marker.get(
                                            p_ch, p_ch
                                        ),
                                        "secondary_channel": s_ch,
                                        "secondary_marker": sample_fcs.detector_to_marker.get(
                                            s_ch, s_ch
                                        ),
                                        "bead_control_spillover_pct": round(old_s * 100.0, 2),
                                        "cell_aligned_spillover_pct": round(new_s * 100.0, 2),
                                        "pre_alignment_delta_mfi": round(dy, 1),
                                        "pre_alignment_z_score": round(z_pre, 2),
                                        "reason": "Tandem dye decoupling / bead-to-cell lot mismatch",
                                    }
                                )
                if tandem_alignment_rows:
                    sample_comp_matrix = CompensationMatrix(detectors=ch_list, matrix=adj_df.values)
                    comp_sample_df = sample_comp_matrix.apply_dataframe(clean_sample_df)
                    p_tandem = tables_dir / "sample_tandem_alignment_audit.csv"
                    pd.DataFrame(tandem_alignment_rows).to_csv(p_tandem, index=False)
                    exported_tables["sample_tandem_alignment_audit"] = str(p_tandem)
                    p_aligned_mat = tables_dir / "spillover_matrix_sample_aligned.csv"
                    adj_df.to_csv(p_aligned_mat)
                    exported_tables["spillover_matrix_sample_aligned"] = str(p_aligned_mat)
        except Exception as exc:
            logger.debug(f"Sample tandem mismatch alignment skipped: {exc}")

    # 6. Render Cytometrist Verification Plots
    exported_plots: Dict[str, str] = {}
    if generate_plots:
        marker_map = sample_fcs.detector_to_marker

        p0 = plots_dir / "comp_00_control_scatter_singlet_audit.png"
        plot_control_scatter_singlet_audit_grid(active_res, marker_map=marker_map, output_path=p0)
        exported_plots["comp_00_control_scatter_singlet_audit"] = str(p0)

        p1 = plots_dir / "comp_01_control_gating_audit.png"
        plot_control_gating_audit_grid(active_res, marker_map=marker_map, output_path=p1)
        exported_plots["comp_01_control_gating_audit"] = str(p1)

        p2 = plots_dir / "comp_02_residual_alignment_summary.png"
        plot_residual_alignment_diagnostics(active_res, marker_map=marker_map, output_path=p2)
        exported_plots["comp_02_residual_alignment_summary"] = str(p2)

        p3 = plots_dir / "comp_03_spillover_and_diff_matrix.png"
        plot_spillover_and_diff_matrix(active_res, marker_map=marker_map, output_path=p3)
        exported_plots["comp_03_spillover_and_diff_matrix"] = str(p3)

        p4 = plots_dir / "comp_04_spillover_spreading_matrix.png"
        plot_spillover_spreading_matrix(active_res, marker_map=marker_map, output_path=p4)
        exported_plots["comp_04_spillover_spreading_matrix"] = str(p4)

        p5 = plots_dir / "comp_05_sample_top_pairs_before_after.png"
        plot_sample_top_pairs_gallery(
            uncomp_df=clean_sample_df,
            comp_df=comp_sample_df,
            comp_matrix=sample_comp_matrix,
            marker_map=marker_map,
            output_path=p5,
            n_pairs=None,
            reference_control_matrix=active_res.dataframe,
        )
        exported_plots["comp_05_sample_top_pairs_before_after"] = str(p5)

        p6 = plots_dir / "comp_06_estimator_comparison.png"
        plot_estimator_comparison(
            mfi_result=mfi_res,
            reg_result=reg_res,
            hybrid_result=hyb_res,
            embedded_df=sample_fcs.spillover_matrix,
            output_path=p6,
        )
        exported_plots["comp_06_estimator_comparison"] = str(p6)

        p7 = plots_dir / "comp_07_all_active_spillover_pairs_atlas.png"
        plot_all_active_spillover_pairs_atlas(
            uncomp_df=clean_sample_df,
            comp_df=comp_sample_df,
            comp_matrix=sample_comp_matrix,
            marker_map=marker_map,
            output_path=p7,
            min_spillover=0.25,
            reference_control_matrix=active_res.dataframe,
            export_laser_sheets=True,
        )
        exported_plots["comp_07_all_active_spillover_pairs_atlas"] = str(p7)
        for s_file in sorted(plots_dir.glob("comp_07_atlas_laser_*.png")):
            exported_plots[s_file.stem] = str(s_file)

        # 8. N x N Compensation Validation Matrices (organized in nxn_matrices/)
        nxn_dir = out_dir / "nxn_matrices"
        nxn_controls_dir = nxn_dir / "controls"
        nxn_samples_dir = nxn_dir / "samples"
        nxn_controls_dir.mkdir(parents=True, exist_ok=True)
        nxn_samples_dir.mkdir(parents=True, exist_ok=True)

        # Also support plots/nxn_matrices
        plots_nxn_controls_dir = plots_dir / "nxn_matrices" / "controls"
        plots_nxn_samples_dir = plots_dir / "nxn_matrices" / "samples"
        plots_nxn_controls_dir.mkdir(parents=True, exist_ok=True)
        plots_nxn_samples_dir.mkdir(parents=True, exist_ok=True)

        # Primary biological sample NxN matrix
        p_primary_sample_nxn = nxn_samples_dir / f"sample_{sample_fcs.sample_name}_nxn_matrix.png"
        plot_nxn_compensation_matrix(
            uncomp_df=clean_sample_df,
            comp_df=comp_sample_df,
            channels=list(active_res.dataframe.index),
            marker_map=marker_map,
            sample_name=sample_fcs.sample_name,
            output_path=p_primary_sample_nxn,
            title=f"Sample Compensation Validation Matrix ({n_ch}×{n_ch}) — {sample_fcs.sample_name}",
            lower_triangle_only=True,
            auto_gate_sample=True,
            gating_mode=cfg.sample_nxn_gating_mode,
            viable_only=True,
            channel_ranges=sample_fcs.channel_ranges,
        )
        shutil.copy2(p_primary_sample_nxn, plots_nxn_samples_dir / p_primary_sample_nxn.name)
        exported_plots[f"sample_{sample_fcs.sample_name}_nxn_matrix"] = str(p_primary_sample_nxn)
        exported_plots["nxn_matrices"] = str(nxn_dir)

        # Generate N x N validation matrix for EACH single-stained control
        if active_res.gated_controls:
            for ctrl_ch, (ctrl_clean_df, _, _, _) in active_res.gated_controls.items():
                safe_ch = re.sub(r"[^\w\-_\.]", "_", ctrl_ch)
                m_name = marker_map.get(ctrl_ch, "") if marker_map else ""
                m_tag = f" ({m_name})" if m_name else ""
                c_ctrl_df = sample_comp_matrix.apply_dataframe(ctrl_clean_df)
                p_ctrl_nxn = nxn_controls_dir / f"control_{safe_ch}_nxn_matrix.png"
                plot_nxn_compensation_matrix(
                    uncomp_df=ctrl_clean_df,
                    comp_df=c_ctrl_df,
                    channels=list(active_res.dataframe.index),
                    marker_map=marker_map,
                    sample_name=f"Control: {ctrl_ch}{m_tag}",
                    output_path=p_ctrl_nxn,
                    title=f"Single-Stain Control Compensation Validation Matrix ({n_ch}×{n_ch}) — {ctrl_ch}{m_tag}",
                    lower_triangle_only=True,
                    auto_gate_sample=False,
                    gating_mode="all",
                )
                shutil.copy2(p_ctrl_nxn, plots_nxn_controls_dir / p_ctrl_nxn.name)
                exported_plots[f"control_{safe_ch}_nxn_matrix"] = str(p_ctrl_nxn)

    # Compensate all biological samples in dataset if multiple are present
    all_compensated_samples: Dict[str, str] = {}
    if target_path.is_dir():
        other_samples = [f for f in sample_files if f != sample_fcs_path]
        if other_samples:
            comp_samples_dir = out_dir / "compensated_samples"
            comp_samples_dir.mkdir(parents=True, exist_ok=True)
            for s_path in sample_files:
                try:
                    s_fcs = parse_fcs(s_path)
                    s_df = s_fcs.events_df
                    if cfg.apply_time_qc_to_sample:
                        s_tqc = apply_time_qc(s_fcs, config=TimeQCConfig.for_sample())
                        s_dec = s_tqc.evaluate_workflow(s_fcs)
                        if s_dec.clean_events_df is not None:
                            s_df = s_dec.clean_events_df
                    s_comp_mat = sample_comp_matrix
                    if policy in ("fallback_embedded", "scale_power_law") and s_path != sample_fcs_path:
                        s_v_audit = audit_pmt_voltage_concordance(
                            sample_fcs=s_fcs,
                            control_files_mapped=active_res.control_files_mapped,
                            channels=list(active_res.dataframe.index),
                            tolerance_v=cfg.voltage_tolerance_v,
                            marker_map=marker_map,
                        )
                        if s_v_audit.has_mismatch:
                            if policy == "fallback_embedded" and s_fcs.spillover_matrix is not None:
                                s_emb = s_fcs.spillover_matrix.reindex(
                                    index=active_res.dataframe.index,
                                    columns=active_res.dataframe.index,
                                ).fillna(0.0)
                                s_comp_mat = CompensationMatrix(
                                    detectors=list(active_res.dataframe.index),
                                    matrix=s_emb.values,
                                )
                            elif policy == "scale_power_law":
                                s_volts = extract_detector_voltages(s_fcs)
                                s_scaled_df = adjust_spillover_by_pmt_voltages(
                                    base_matrix_df=active_res.dataframe,
                                    control_voltages=ctrl_volts_cache,
                                    sample_voltages=s_volts,
                                    alpha=cfg.pmt_gain_exponent_alpha,
                                )
                                s_comp_mat = CompensationMatrix(
                                    detectors=list(active_res.dataframe.index),
                                    matrix=s_scaled_df.values,
                                )
                        else:
                            s_comp_mat = active_res.matrix

                    c_df = s_comp_mat.apply_dataframe(s_df)
                    c_out = comp_samples_dir / f"{s_path.stem}_compensated.csv"
                    c_df.to_csv(c_out, index=False)
                    all_compensated_samples[s_path.name] = str(c_out)
                    exported_tables[f"compensated_sample_{s_path.stem}"] = str(c_out)

                    if generate_plots and s_path != sample_fcs_path:
                        p_s_nxn = nxn_samples_dir / f"sample_{s_path.stem}_nxn_matrix.png"
                        plot_nxn_compensation_matrix(
                            uncomp_df=s_df,
                            comp_df=c_df,
                            channels=list(active_res.dataframe.index),
                            marker_map=marker_map,
                            sample_name=s_path.stem,
                            output_path=p_s_nxn,
                            title=f"Sample Compensation Validation Matrix ({n_ch}×{n_ch}) — {s_path.stem}",
                            lower_triangle_only=True,
                            auto_gate_sample=True,
                            gating_mode=cfg.sample_nxn_gating_mode,
                            viable_only=True,
                            channel_ranges=s_fcs.channel_ranges,
                        )
                        shutil.copy2(p_s_nxn, plots_nxn_samples_dir / p_s_nxn.name)
                        exported_plots[f"sample_{s_path.stem}_nxn_matrix"] = str(p_s_nxn)
                except Exception as exc:
                    logger.warning(f"Could not batch-compensate sample {s_path.name}: {exc}")

    elapsed = time.perf_counter() - t0
    report_md_path = out_dir / "compensation_report.md"

    # 7. Generate Comprehensive MIFlowCyt Compensation Report
    n_mapped = len(active_res.control_files_mapped)
    total_ctrl_channels = (
        len(sample_fcs.spillover_channels)
        if getattr(sample_fcs, "spillover_channels", None)
        else n_ch
    )
    n_pairs = n_ch * (n_ch - 1)

    lines: List[str] = [
        f"# Flow Cytometry Compensation & Verification Report — `{dataset_name}`",
        "",
        f"> **Pipeline:** FlowX Preprocessing Step 1.3 (Time-QC Integrated Compensation & Verification)  ",
        f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Biological Sample:** `{sample_fcs.filename}` ({sample_fcs.event_count:,} raw events → {len(clean_sample_df):,} Time-QC passed events, {sample_passed_pct:.2f}%)  ",
        f"> **Single-Stained Controls:** {n_mapped}/{total_ctrl_channels} channels mapped  ",
        f"> **Active Estimator:** `{active_res.estimation_method_used.upper()}` (Combines Upgraded Median ΔMFI for bimodal beads + AutoSpill Linear Regression for cell controls + Iterative Residual Refinement)  ",
        f"> **PMT Voltage Concordance:** {'MISMATCH DETECTED' if voltage_audit.has_mismatch else 'CONCORDANT (PASS)'}  ",
        f"> **Execution Time:** {elapsed:.2f} seconds  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Estimator Benchmark",
        "",
        "| Estimator Method | Condition Number κ(S) | Instrument Matrix Correlation (r) | Off-Diagonal MAE (%) | Max Diff vs Instrument (%) | Mean Post-Comp Residual \\|Z\\| (rSD) | 95th Pct Residual \\|Z\\| (rSD) | Pairs Aligned (\\|Z\\| ≤ 0.5 rSD) |",
        "|:-----------------|:---------------------:|:---------------------------------:|:--------------------:|:--------------------------:|:-----------------------------------:|:-----------------------------:|:-------------------------------:|",
    ]

    for _, brow in bench_df.iterrows():
        m_lbl = f"**`{brow['estimation_method']}`**" if brow["estimation_method"] == active_method else f"`{brow['estimation_method']}`"
        lines.append(
            f"| {m_lbl} | {brow['condition_number']:.2f} | **{brow['embedded_pearson_r']:.5f}** | "
            f"**{brow['embedded_mae_pct']:.3f}%** | {brow['embedded_max_diff_pct']:.2f}% | "
            f"**{brow['mean_residual_abs_z']:.3f}** | {brow['p95_residual_abs_z']:.3f} | "
            f"**{brow['pairs_within_0_5_rsd_pct']:.1f}%** |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Instrument PMT Voltage & Gain Concordance Audit",
            "",
        ]
    )

    if voltage_audit.has_mismatch:
        lines.extend(
            [
                "> [!WARNING]",
                f"> **PMT Voltage Discrepancy Detected:** Single-stained reference controls and biological sample `{sample_fcs.filename}` were acquired with differing PMT detector gains across **{len(voltage_audit.mismatched_channels)}** fluorescence channel(s): `{', '.join(voltage_audit.mismatched_channels)}`.  ",
                "> Applying a single-stain matrix acquired at different PMT voltages directly to a mismatched sample will introduce systematic under- or over-compensation.  ",
                f"> **Configured Policy:** `{cfg.voltage_mismatch_policy}` | **Action Taken:** `{voltage_audit.action_taken}`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "> [!NOTE]",
                f"> **PMT Voltage Concordance Confirmed:** Single-stained controls and biological sample `{sample_fcs.filename}` have matching PMT detector gains across all fluorescence channels (within $\\pm {cfg.voltage_tolerance_v:.1f}\\text{{ V}}$).",
                "",
            ]
        )

    lines.extend(
        [
            "| Detector Channel | Marker | Channel Type | Matched Control | Control Voltage | Sample Voltage | ΔV (Sample - Control) | Concordance Status |",
            "|:-----------------|:-------|:------------:|:----------------|----------------:|---------------:|----------------------:|:------------------:|",
        ]
    )

    if voltage_audit.audit_df is not None:
        for _, vr in voltage_audit.audit_df.iterrows():
            c_v_str = f"{vr['control_voltage']:.1f} V" if pd.notna(vr["control_voltage"]) and vr["control_voltage"] is not None else "-"
            s_v_str = f"{vr['sample_voltage']:.1f} V" if pd.notna(vr["sample_voltage"]) and vr["sample_voltage"] is not None else "-"
            d_v_str = f"{vr['voltage_delta']:+.1f} V" if pd.notna(vr["voltage_delta"]) and vr["voltage_delta"] is not None else "-"
            stat_lbl = f"`{vr['status']}`"
            if vr['status'] == "MISMATCH":
                stat_lbl = "**`MISMATCH`** ⚠️"
            elif vr['status'] == "MATCH":
                stat_lbl = "`MATCH` ✅"
            lines.append(
                f"| `{vr['channel']}` | **{vr['marker']}** | {vr['channel_type']} | "
                f"`{vr['control_file']}` | {c_v_str} | {s_v_str} | {d_v_str} | {stat_lbl} |"
            )

    lines.extend(
        [
            "",
            f"- **PMT Voltage Drift Tolerance:** $\\pm {cfg.voltage_tolerance_v:.1f}\\text{{ V}}$",
            f"- **Policy Applied:** `{voltage_audit.action_taken}`",
            "",
            "---",
            "",
            "## 3. Diagnostic Visual Verification Suite",
            "",
            "> [!NOTE]",
            "> **MIFlowCyt / ISAC Compliance Note on Residual Verification:**  ",
            "> Post-compensation residual median alignment (ΔMFI within ±0.5 rSD_neg) evaluated on single-stained controls ",
            "> serves as an internal self-consistency and quality control diagnostic to detect spectral unmixing errors, ",
            "> over-/under-compensation, or carrier autofluorescence mismatches. Per MIFlowCyt guidelines, residual checks ",
            "> on single-stained controls do not substitute for biological multi-stain verification (e.g., FMO controls or ",
            "> known double-negative/mutually-exclusive populations in the biological sample). Both single-stain internal ",
            "> self-consistency and biological multi-stain visualizations (Plots 00–07) are provided below.",
            "",
            "0. **Single-Stain Control Scatter, Debris Exclusion & Singlet Pre-Gating Audit**: [`plots/comp_00_control_scatter_singlet_audit.png`](plots/comp_00_control_scatter_singlet_audit.png)  ",
            "   Verifies sub-cellular debris removal (`FSC-A vs SSC-A`), 2D monomer bead density mode gating, and dual singlet discrimination (`FSC-H/FSC-A` & `SSC-H/SSC-A`) prior to fluorescence gating.",
            f"1. **Single-Stain Control Fluorescence Gating & Carrier Audit Grid**: [`plots/comp_01_control_gating_audit.png`](plots/comp_01_control_gating_audit.png)  ",
            f"   Verifies automated `[BEAD]` vs `[CELL]` carrier classification, Time QC pass rate, scatter/singlet clean retention, and Positive/Negative separation across all {n_mapped} controls.",
            f"2. **Post-Compensation Residual Median Alignment & Binned Trajectory Check**: [`plots/comp_02_residual_alignment_summary.png`](plots/comp_02_residual_alignment_summary.png)  ",
            f"   Verifies post-compensation positive and negative medians alignment within $\\pm 0.5\\,\\text{{rSD}}_{{\\text{{neg}}}}$ across all ${n_ch} \\times {n_ch - 1} = {n_pairs}$ channel pairs on single-stained controls.",
            f"3. **Laser-Ordered Spillover Matrix & Instrument Difference Heatmap**: [`plots/comp_03_spillover_and_diff_matrix.png`](plots/comp_03_spillover_and_diff_matrix.png)  ",
            f"   Displays the {n_ch}-channel spillover matrix grouped by excitation laser alongside $\\Delta\\% = S_{{\\text{{calc}}}} - S_{{\\text{{embedded}}}}$.",
            "4. **Spillover Spreading Matrix (SSM — Nguyen et al. 2013)**: [`plots/comp_04_spillover_spreading_matrix.png`](plots/comp_04_spillover_spreading_matrix.png)  ",
            "   Quantifies detector photon-counting spread introduced by each fluorophore into secondary detectors after mathematically exact compensation.",
            "5. **Complete All-Stains Biological Compensation Verification (All Stains Before vs. After)**: [`plots/comp_05_sample_top_pairs_before_after.png`](plots/comp_05_sample_top_pairs_before_after.png)  ",
            "   Auto-scaled grid displaying the #1 highest-spillover pair for every single stain in the panel before and after compensation on gated Live Singlet Lymphocytes.",
            "6. **Estimator Benchmark Comparison**: [`plots/comp_06_estimator_comparison.png`](plots/comp_06_estimator_comparison.png)  ",
            "   Compares Upgraded Median $\\Delta\\text{MFI}$, AutoSpill Regression, and Hybrid estimators.",
            f"7. **Compensated Spillover Verification Atlas (All Active Spillover Pairs $\\ge 25\\%$)**: [`plots/comp_07_all_active_spillover_pairs_atlas.png`](plots/comp_07_all_active_spillover_pairs_atlas.png)  ",
            "   High-density atlas organized by excitation lasers. Individual per-laser figure sheets are also available for focused inspection:  ",
        ]
    )

    laser_sheets = sorted(plots_dir.glob("comp_07_atlas_laser_*.png"))
    if laser_sheets:
        for s_file in laser_sheets:
            sheet_lbl = s_file.stem.replace("comp_07_atlas_laser_", "Laser ").replace("_", " ")
            lines.append(f"   - **{sheet_lbl}**: [`plots/{s_file.name}`](plots/{s_file.name})  ")
    else:
        lines.append("   - *(No multi-detector laser groupings required for this panel)*  ")

    lines.extend(
        [
            f"8. **N×N Compensation Validation Matrices**: [`nxn_matrices/`](nxn_matrices/)  ",
            f"   Lower-triangular {n_ch}×{n_ch} matrices overlaying uncompensated (red) vs. compensated (blue) events with orthogonal zero guidelines across all unique detector pairs. Segregated into dedicated subdirectories for biological cohort samples and single-stained controls:",
            "",
            "### Individual N×N Validation Matrices (Controls & Cohort Samples)",
            f"Comprehensive unique-pair {n_ch}×{n_ch} matrix plots for all biological cohort samples and single-stained controls are archived in [`nxn_matrices/`](nxn_matrices/):",
        ]
    )

    nxn_sample_files = sorted((out_dir / "nxn_matrices" / "samples").glob("*.png")) or sorted(plots_dir.glob("nxn_matrices/samples/*.png"))
    nxn_ctrl_files = sorted((out_dir / "nxn_matrices" / "controls").glob("*.png")) or sorted(plots_dir.glob("nxn_matrices/controls/*.png"))
    if not nxn_sample_files and not nxn_ctrl_files:
        flat_files = sorted(plots_dir.glob("comp_08_nxn_matrices/*.png")) or sorted(plots_dir.glob("nxn_matrices/*.png"))
        nxn_ctrl_files = [f for f in flat_files if f.name.startswith("control_")]
        nxn_sample_files = [f for f in flat_files if f.name.startswith("sample_")]

    if nxn_sample_files or nxn_ctrl_files:
        if nxn_sample_files:
            lines.append("")
            lines.append("#### Biological Cohort Samples (`nxn_matrices/samples/`)")
            for s_file in nxn_sample_files:
                rel_path = f"nxn_matrices/samples/{s_file.name}"
                clean_name = s_file.stem.replace("_nxn_matrix", "").replace("sample_", "")
                lines.append(f"- **Sample: {clean_name}**: [`{rel_path}`]({rel_path})  ")

        if nxn_ctrl_files:
            lines.append("")
            lines.append("#### Single-Stained Controls (`nxn_matrices/controls/`)")
            for c_file in nxn_ctrl_files:
                rel_path = f"nxn_matrices/controls/{c_file.name}"
                clean_name = c_file.stem.replace("_nxn_matrix", "").replace("control_", "")
                lines.append(f"- **Control: {clean_name}**: [`{rel_path}`]({rel_path})  ")
    else:
        lines.append("- *(N×N matrix plots not generated or skipped)*  ")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Single-Stained Control Audit Roster (Time QC + Debris/Singlet Pre-Gating + Fluorescence Gating + Residual Check)",
            "",
            "| Detector Channel | Marker | Control File | Carrier | Estimator | Time QC Pass | Debris Excl | Doublets Excl | Clean Singlets Kept | Gated Pos / Neg | Pos MFI | Neg MFI | SNR | Max Residual \\|Z\\| | Status |",
            "|:-----------------|:-------|:-------------|:-------:|:---------:|:------------:|:-----------:|:-------------:|:-------------------:|:---------------:|--------:|--------:|----:|:------------------:|:------:|",
        ]
    )

    if active_res.control_qc_summary_df is not None:
        for _, r in active_res.control_qc_summary_df.iterrows():
            lines.append(
                f"| `{r['channel']}` | **{r.get('marker') or '-'}** | `{r.get('control_file', '')}` | "
                f"`{str(r.get('carrier_type', 'beads')).upper()}` | `{r.get('estimator_used', '')}` | "
                f"{r.get('time_qc_passed_pct', 100.0):.1f}% (`{r.get('time_qc_status', 'PASS')}`) | "
                f"{r.get('debris_removed_pct', 0.0):.1f}% ({int(r.get('debris_removed_events', 0)):,}) | "
                f"{r.get('doublets_removed_pct', 0.0):.1f}% ({int(r.get('doublets_removed_events', 0)):,}) | "
                f"{int(r.get('n_clean_events', r.get('n_events', 0))):,} ({r.get('scatter_singlet_kept_pct', 100.0):.1f}%) | "
                f"{int(r.get('n_pos', 0)):,} / {int(r.get('n_neg', 0)):,} | "
                f"{r.get('pos_mfi', 0.0):,.1f} | {r.get('neg_mfi', 0.0):,.1f} | "
                f"{r.get('snr', 0.0):,.1f} | {r.get('max_residual_abs_z', 0.0):.3f} | "
                f"`{r.get('residual_status', 'ALIGNED')}` |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Exported Data Artifacts & Matrices",
            "",
        ]
    )
    from flowx.io.paths import sanitize_path
    for k, v in exported_tables.items():
        rel_v = sanitize_path(v, base_dir=out_dir)
        lines.append(f"- **`{k}`**: [`{rel_v}`]({rel_v})")
    rel_qc_rep = sanitize_path(qc_report_path, base_dir=out_dir)
    lines.append(f"- **`time_qc_cohort_report`**: [`{rel_qc_rep}`]({rel_qc_rep})")

    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "report_md_path": str(report_md_path),
        "time_qc_report_path": str(qc_report_path),
        "active_result": active_res,
        "mfi_result": mfi_res,
        "regression_result": reg_res,
        "hybrid_result": hyb_res,
        "benchmark_df": bench_df,
        "plots": exported_plots,
        "tables": exported_tables,
        "sample_time_qc": sample_tqc,
        "compensated_samples": all_compensated_samples,
        "voltage_audit": voltage_audit,
    }


@dataclass
class CompensationStepResult:
    """Container holding results from an independent single-sample compensation/unmixing step."""

    compensated_df: pd.DataFrame
    comp_matrix: Optional[CompensationMatrix] = None
    comp_ctrl_result: Optional[CompensationControlResult] = None
    comp_source_used: str = "none"
    target_channels: List[str] = field(default_factory=list)
    spectral_pipeline_res: Optional[Any] = None
    spectral_library: Optional[Any] = None
    spill_matrix_csv_path: Optional[str] = None
    tables: Dict[str, str] = field(default_factory=dict)
    plots: Dict[str, str] = field(default_factory=dict)


def resolve_compensation_controls(
    fcs_data: Any,
    comp_controls_dir: Optional[Union[str, Path]] = None,
    unstained_control: Optional[Union[str, Path]] = None,
    comp_enabled: bool = True,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Auto-discover compensation controls directory and unstained control when not explicitly supplied."""
    effective_controls_dir = Path(comp_controls_dir) if (comp_enabled and comp_controls_dir) else None
    effective_unstained = Path(unstained_control) if (comp_enabled and unstained_control) else None

    if effective_unstained is not None:
        ust_name = effective_unstained.name.lower()
        if ("viability" in ust_name or "live-or-dye" in ust_name) and "unstained" not in ust_name:
            effective_unstained = None

    if comp_enabled and effective_controls_dir is None and getattr(fcs_data, "filepath", None):
        fcs_dir = Path(fcs_data.filepath).parent
        candidates = [
            fcs_dir.parent / "controls",
            fcs_dir.parent / "raw" / "controls",
            fcs_dir / "controls",
        ]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.fcs")):
                effective_controls_dir = candidate
                break

    if comp_enabled and effective_unstained is None and effective_controls_dir is not None and effective_controls_dir.is_dir():
        for f in effective_controls_dir.glob("*.fcs"):
            f_low = f.name.lower()
            if any(w in f_low for w in ["unstained", "blank"]) and "fmo" not in f_low:
                effective_unstained = f
                break

    return effective_controls_dir, effective_unstained


def export_compensation_tables(
    tables_dir: Optional[Union[str, Path]],
    named_dataframes: Dict[str, Optional[pd.DataFrame]],
    index_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, str]:
    """Persist compensation or spectral unmixing matrices and audit tables to CSV.

    Provides a unified artifact export contract for both conventional spillover
    compensation and spectral unmixing workflows.

    Args:
        tables_dir: Target directory for CSV tables (e.g. ``02_compensation/tables``).
        named_dataframes: Mapping from artifact key (file stem) to DataFrame.
        index_flags: Optional per-key override for ``DataFrame.to_csv(index=...)``.

    Returns:
        Dictionary mapping each exported artifact key to its saved CSV filepath.
    """
    if tables_dir is None:
        return {}
    t_dir = Path(tables_dir)
    t_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, str] = {}
    flags = index_flags or {}
    for key, df in named_dataframes.items():
        if df is None:
            continue
        out_path = t_dir / f"{key}.csv"
        df.to_csv(out_path, index=flags.get(key, True))
        exported[key] = str(out_path)
    return exported


def run_compensation_step(
    fcs_data: Any,
    working_df: pd.DataFrame,
    target_channels: Optional[List[str]] = None,
    enable_compensation: bool = True,
    compensation_source: str = "auto",
    technology: str = "auto",
    comp_controls_dir: Optional[Union[str, Path]] = None,
    comp_manifest: Optional[Union[str, Path]] = None,
    unstained_control: Optional[Union[str, Path]] = None,
    comp_channel_mapping: Optional[Dict[str, str]] = None,
    comp_config: Optional[CompensationConfig] = None,
    spectral_config: Optional[Any] = None,
    cached_comp_matrix: Optional[CompensationMatrix] = None,
    cached_spectral_library: Optional[Any] = None,
    plots_dir: Optional[Union[str, Path]] = None,
    tables_dir: Optional[Union[str, Path]] = None,
    plot_name_fn: Optional[Any] = None,
    sample_stem: Optional[str] = None,
    random_seed: int = 42,
    generate_plots: bool = True,
    generate_nxn: bool = True,
) -> CompensationStepResult:
    """Execute independent compensation or spectral unmixing step on a sample DataFrame.

    Shared by both single-sample/cohort pipeline runs and standalone step execution.
    """
    import copy as _copy
    from flowx.math.transforms import is_empirically_pre_transformed

    _pname = plot_name_fn or (lambda name: name)
    p_dir = Path(plots_dir) if plots_dir else None
    t_dir = Path(tables_dir) if tables_dir else None

    df_comp = working_df.copy()
    ch_list = list(target_channels or getattr(fcs_data, "spillover_channels", []) or [])
    if not ch_list:
        exclude = {
            "fsc-a", "fsc-h", "fsc-w", "fsc-width",
            "ssc-a", "ssc-h", "ssc-w", "ssc-width",
            "time", "width", "original_id", "event_id",
            "sample_id", "index", "cluster_id", "lineage_id",
            "umap_1", "umap_2", "fs int", "ss int", "ss peak", "ss tof",
        }
        ch_list = [
            ch
            for ch in getattr(fcs_data, "pnn_labels", [])
            if ch.lower() not in exclude
            and not any(ch.lower().startswith(p) for p in ["fs ", "fs.", "ss ", "ss.", "time", "original_id", "event_id"])
        ]

    active_tech = (technology or "auto").lower()
    if active_tech == "auto":
        if compensation_source == "spectral" or spectral_config is not None:
            active_tech = "spectral"
        elif getattr(fcs_data, "cytometer", None) and any(
            w in str(fcs_data.cytometer).lower() for w in ["aurora", "id7000", "bigfoot", "spectral"]
        ):
            active_tech = "spectral"
        else:
            active_tech = "conventional"

    is_spectral = active_tech == "spectral" or compensation_source == "spectral"
    emp_pre_transformed = is_empirically_pre_transformed(fcs_data)
    comp_enabled = bool(enable_compensation and compensation_source != "none" and not emp_pre_transformed)

    effective_controls_dir, effective_unstained = resolve_compensation_controls(
        fcs_data,
        comp_controls_dir=comp_controls_dir,
        unstained_control=unstained_control,
        comp_enabled=comp_enabled,
    )

    should_use_controls = comp_enabled and (
        compensation_source == "controls"
        or (compensation_source == "auto" and (effective_controls_dir is not None or comp_manifest is not None))
    )

    meta = getattr(fcs_data, "metadata", {}) or {}
    is_pre_transformed = (
        emp_pre_transformed
        or str(meta.get("transformation", "")).strip().lower() == "applied"
        or str(meta.get("transformation:applied", "")).strip().lower() == "true"
        or any("transformation" in k.lower() and "applied" in str(v).lower() for k, v in meta.items())
    )

    raw_spectral_pattern = re.compile(r"^(UV|V|B|YG|R)\d+-A$", re.IGNORECASE)
    has_raw_spectral_array = (
        len(ch_list) >= 20
        and sum(1 for c in ch_list if raw_spectral_pattern.match(c)) >= int(0.6 * len(ch_list))
    )

    comp_obj: Optional[CompensationMatrix] = None
    comp_ctrl_result: Optional[CompensationControlResult] = None
    comp_source_used = "none"
    spectral_pipeline_res: Optional[Any] = None
    spectral_library_out = cached_spectral_library
    spill_matrix_csv_path: Optional[str] = None
    out_tables: Dict[str, str] = {}
    out_plots: Dict[str, str] = {}

    if not comp_enabled:
        comp_source_used = "pre_transformed" if emp_pre_transformed else "none"
    elif is_spectral and has_raw_spectral_array:
        if cached_spectral_library is not None:
            from flowx.spectral.models import UnmixingConfig
            from flowx.spectral.unmixing import unmix_dataframe

            unmix_res = unmix_dataframe(
                working_df,
                cached_spectral_library,
                config=spectral_config or UnmixingConfig(),
            )
            df_comp = unmix_res.unmixed_df
            ch_list = list(unmix_res.fluorophore_channels)
            comp_source_used = "spectral_unmixing"
        else:
            from flowx.spectral.runner import run_spectral_unmixing_pipeline

            fcs_work = _copy.copy(fcs_data)
            fcs_work.events_df = working_df
            fcs_work.event_count = len(working_df)
            unmix_pipe_res = run_spectral_unmixing_pipeline(
                sample_input=fcs_work,
                controls_dir_or_files=effective_controls_dir,
                unstained_control=effective_unstained,
                output_dir=(p_dir.parent if p_dir is not None else Path("outputs/spectral_unmixing")),
                tables_dir=t_dir,
                plots_dir=p_dir,
                config=spectral_config,
            )
            df_comp = unmix_pipe_res.unmixing_result.unmixed_df
            ch_list = list(unmix_pipe_res.unmixing_result.fluorophore_channels)
            comp_source_used = "spectral_unmixing"
            spectral_pipeline_res = unmix_pipe_res
            spectral_library_out = unmix_pipe_res.spectral_library
            out_tables.update(unmix_pipe_res.generated_tables)
            if unmix_pipe_res.spill_matrix_csv_path:
                spill_matrix_csv_path = unmix_pipe_res.spill_matrix_csv_path
            if getattr(fcs_data, "spillover_matrix", None) is not None and t_dir is not None:
                emb_tables = export_compensation_tables(
                    t_dir,
                    {"spillover_matrix_embedded": fcs_data.spillover_matrix},
                )
                out_tables.update(emb_tables)
            out_plots.update(unmix_pipe_res.generated_plots)
    elif is_pre_transformed and getattr(fcs_data, "spillover_matrix", None) is None and not should_use_controls:
        comp_source_used = "pre_compensated"
    elif cached_comp_matrix is not None and set(cached_comp_matrix.detectors).issubset(set(df_comp.columns)):
        comp_obj = cached_comp_matrix
        comp_source_used = "controls"
        df_comp = comp_obj.apply_dataframe(df_comp)
    elif should_use_controls and (effective_controls_dir is not None or comp_manifest is not None) and not (
        compensation_source != "controls"
        and is_spectral
        and not has_raw_spectral_array
        and getattr(fcs_data, "spillover_matrix", None) is not None
    ):
        try:
            comp_ctrl_result = calculate_spillover_from_controls(
                controls_dir_or_files=effective_controls_dir,
                channels=ch_list,
                fcs_sample=fcs_data,
                unstained_control=effective_unstained,
                channel_mapping=comp_channel_mapping,
                manifest_path=comp_manifest,
                fallback_matrix=getattr(fcs_data, "spillover_matrix", None),
                random_seed=random_seed,
                config=comp_config,
            )
            comp_obj = comp_ctrl_result.matrix
            comp_source_used = "controls"
            df_comp = comp_obj.apply_dataframe(df_comp)

            if t_dir is not None:
                exported_ctrl_tables = export_compensation_tables(
                    t_dir,
                    {
                        "spillover_matrix_calculated": comp_ctrl_result.dataframe,
                        "control_gating_and_qc_audit": comp_ctrl_result.control_qc_summary_df,
                        "spillover_spreading_matrix_ssm": comp_ctrl_result.ssm_dataframe,
                        "post_comp_residual_z_scores": comp_ctrl_result.residual_z_dataframe,
                    },
                    index_flags={"control_gating_and_qc_audit": False},
                )
                out_tables.update(exported_ctrl_tables)
                spill_matrix_csv_path = exported_ctrl_tables.get("spillover_matrix_calculated")

            if generate_plots and p_dir is not None:
                from flowx.visualization.comp_plots import (
                    plot_control_gating_audit_grid,
                    plot_control_scatter_singlet_audit_grid,
                )
                from flowx.visualization.plotting import plot_spillover_matrix

                p_dir.mkdir(parents=True, exist_ok=True)
                spill_plot_path = p_dir / _pname("step2_spillover_matrix_calculated.png")
                plot_spillover_matrix(
                    spillover_df=comp_ctrl_result.dataframe,
                    title="Calculated Spillover Matrix (Single-Stained Controls)",
                    output_path=spill_plot_path,
                    condition_number=comp_obj.condition_number,
                    correlation_with_embedded=comp_ctrl_result.embedded_correlation,
                )
                out_plots["step2_spillover_matrix"] = str(spill_plot_path)
                try:
                    scatter_audit_path = p_dir / _pname("step2_control_scatter_singlet_audit.png")
                    plot_control_scatter_singlet_audit_grid(
                        comp_ctrl_result,
                        marker_map=getattr(fcs_data, "detector_to_marker", None),
                        output_path=scatter_audit_path,
                    )
                    out_plots["step2_control_scatter_singlet_audit"] = str(scatter_audit_path)

                    gating_audit_path = p_dir / _pname("step2_control_gating_audit.png")
                    plot_control_gating_audit_grid(
                        comp_ctrl_result,
                        marker_map=getattr(fcs_data, "detector_to_marker", None),
                        output_path=gating_audit_path,
                    )
                    out_plots["step2_control_gating_audit"] = str(gating_audit_path)
                except Exception as plot_exc:
                    logger.debug(f"Optional control audit plots skipped: {plot_exc}")
        except Exception as exc:
            logger.error(f"Failed to calculate spillover from controls: {exc}. Falling back to embedded matrix if available.")
            if getattr(fcs_data, "spillover_matrix", None) is not None:
                comp_obj = CompensationMatrix(
                    detectors=fcs_data.spillover_channels,
                    matrix=fcs_data.spillover_matrix.values,
                )
                comp_source_used = "embedded"
                df_comp = comp_obj.apply_dataframe(df_comp)
    elif getattr(fcs_data, "spillover_matrix", None) is not None and getattr(fcs_data, "spillover_channels", None):
        emb_mat = CompensationMatrix(
            detectors=fcs_data.spillover_channels,
            matrix=fcs_data.spillover_matrix.values,
        )
        off_diag = emb_mat.matrix.copy()
        np.fill_diagonal(off_diag, 0.0)
        is_non_trivial = float(np.max(np.abs(off_diag))) > 1e-4 if off_diag.size > 0 else False
        if not is_spectral or is_non_trivial:
            comp_obj = emb_mat
            comp_source_used = "embedded"
            df_comp = comp_obj.apply_dataframe(df_comp)
            if t_dir is not None:
                exported_emb = export_compensation_tables(
                    t_dir,
                    {"spillover_matrix_embedded": fcs_data.spillover_matrix},
                )
                out_tables.update(exported_emb)
                spill_matrix_csv_path = exported_emb.get("spillover_matrix_embedded")
            if generate_plots and p_dir is not None:
                from flowx.visualization.plotting import plot_spillover_matrix

                p_dir.mkdir(parents=True, exist_ok=True)
                spill_plot_path = p_dir / _pname("step2_spillover_matrix_embedded.png")
                plot_spillover_matrix(
                    spillover_df=fcs_data.spillover_matrix,
                    title="Embedded Spillover Matrix (FCS Header)",
                    output_path=spill_plot_path,
                    condition_number=comp_obj.condition_number,
                )
                out_plots["step2_spillover_matrix"] = str(spill_plot_path)
        else:
            comp_source_used = "pre_unmixed"

    if generate_plots and p_dir is not None and comp_obj is not None and len(comp_obj.detectors) >= 2:
        from flowx.visualization.plotting import plot_compensation_comparison

        p_dir.mkdir(parents=True, exist_ok=True)
        spill_mat = comp_obj.dataframe.values.copy()
        np.fill_diagonal(spill_mat, 0.0)
        if np.max(spill_mat) > 0.0:
            max_idx = np.unravel_index(np.argmax(spill_mat), spill_mat.shape)
            ch1 = comp_obj.detectors[max_idx[0]]
            ch2 = comp_obj.detectors[max_idx[1]]
        else:
            ch1 = comp_obj.detectors[0]
            ch2 = comp_obj.detectors[1]
        diag_name = f"{sample_stem}_compensation_diagnostic.png" if sample_stem else _pname("step2_compensation_diagnostic.png")
        diag_path = p_dir / diag_name
        plot_compensation_comparison(
            uncomp_df=working_df,
            comp_df=df_comp,
            x_channel=ch1,
            y_channel=ch2,
            x_label=fcs_data.get_channel_label(ch1),
            y_label=fcs_data.get_channel_label(ch2),
            output_path=diag_path,
        )
        out_plots["step2_compensation_diagnostic"] = str(diag_path)

    if generate_plots and generate_nxn and p_dir is not None and (comp_source_used == "spectral_unmixing" or (comp_obj is not None and len(comp_obj.detectors) >= 2)):
        try:
            from flowx.visualization.comp_plots import plot_nxn_compensation_matrix

            plots_nxn_dir = p_dir / "nxn_matrices"
            plots_nxn_samples = plots_nxn_dir / "samples"
            (plots_nxn_dir / "controls").mkdir(parents=True, exist_ok=True)
            plots_nxn_samples.mkdir(parents=True, exist_ok=True)

            clean_stem = sample_stem or Path(getattr(fcs_data, "filename", "sample")).stem
            step2_nxn_path = plots_nxn_samples / f"sample_{clean_stem}_nxn_matrix.png"
            if comp_source_used == "spectral_unmixing":
                plot_nxn_compensation_matrix(
                    uncomp_df=working_df,
                    comp_df=df_comp,
                    channels=ch_list,
                    sample_name=clean_stem,
                    output_path=step2_nxn_path,
                    lower_triangle_only=True,
                    auto_gate_sample=True,
                    viable_only=True,
                )
                shutil.copy2(step2_nxn_path, p_dir / _pname("step2_nxn_spectral_matrix.png"))
                out_plots["step2_nxn_matrix"] = str(step2_nxn_path)
            elif comp_obj is not None:
                plot_nxn_compensation_matrix(
                    uncomp_df=working_df,
                    comp_df=df_comp,
                    comp_matrix=comp_obj,
                    channels=list(comp_obj.detectors),
                    marker_map=getattr(fcs_data, "detector_to_marker", None),
                    sample_name=clean_stem,
                    output_path=step2_nxn_path,
                    lower_triangle_only=True,
                    auto_gate_sample=True,
                    viable_only=True,
                )
                shutil.copy2(step2_nxn_path, p_dir / _pname("step2_nxn_compensation_matrix.png"))
                out_plots["step2_nxn_matrix"] = str(step2_nxn_path)
        except Exception as nxn_exc:
            logger.debug(f"Optional Step 2 NxN matrix plot skipped: {nxn_exc}")

    return CompensationStepResult(
        compensated_df=df_comp,
        comp_matrix=comp_obj,
        comp_ctrl_result=comp_ctrl_result,
        comp_source_used=comp_source_used,
        target_channels=ch_list,
        spectral_pipeline_res=spectral_pipeline_res,
        spectral_library=spectral_library_out,
        spill_matrix_csv_path=spill_matrix_csv_path,
        tables=out_tables,
        plots=out_plots,
    )
