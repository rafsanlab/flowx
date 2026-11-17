"""Core Clean-up Gating Pipeline (Debris, Dual Singlets, Viability, and Dump Channel).

Provides a modular, scalable, user-customizable sequential gating cascade conforming to
MIFlowCyt standards to isolate viable, intact single target cells prior to downstream
batch normalization, high-dimensional reduction, or unsupervised clustering:
1. Debris Removal: FSC-A vs SSC-A to exclude sub-cellular debris and optical electronic noise.
2. Dual Singlet Discrimination: FSC-A vs FSC-H and SSC-A vs SSC-H (or Width/TOF) to exclude doublets.
3. Viability Gating: Live/Dead dye exclusion (AARD, DAPI, 7-AAD, Zombie, etc.) to discard dead cells.
4. Dump Channel Gating: Optional lineage exclusion to filter out non-target cell lineages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture

from flowx.gating.density_gating import DensityCutoffResult, find_density_minimum
from flowx.gating.gate import Gate, GatingHierarchy, GMMClusterGate, PolygonGate, RangeGate
from flowx.io.parser import FCSData, parse_fcs

logger = logging.getLogger(__name__)

# Keywords for viability dye auto-detection across manufacturers and assays
VIABILITY_KEYWORDS = [
    "dead",
    "viability",
    "aard",
    "7-aad",
    "7aad",
    "dapi",
    "pi",
    "propidium",
    "zombie",
    "live/dead",
    "livedead",
    "live-dead",
    "fvs",
    "efluor",
    "draq7",
    "ethd",
    "sytox",
    "vitality",
    "ghost",
]

# Keywords for dump / lineage exclusion channel auto-detection
DUMP_KEYWORDS = [
    "dump",
    "lineage",
    "lin-",
    "lin_",
    "exclusion",
    "dump channel",
]


def matches_viability_dye(text: str) -> bool:
    """Robust pattern matching for viability dyes avoiding substring false positives.

    Prevents false matches in words containing short acronyms (e.g. 'pi' in 'lipid', 'spike',
    'pi3k', 'capillary') and distinguishes Fixable Viability Dye eFluor from standard
    fluorophore conjugates like 'CD3 eFluor450'.
    """
    if not text:
        return False
    t_clean = text.lower().strip()
    tokens = set(re.split(r"\W+", t_clean))
    if any(tok in {"pi", "fvs", "dapi", "7aad", "aad", "aard", "dead", "zombie", "draq7", "sytox", "ethd", "aamine"} for tok in tokens):
        return True

    if any(term in t_clean for term in [
        "7-aad", "7aad", "live/dead", "live-dead", "livedead", "live-or-dye", "live or dye",
        "propidium", "viability", "vitality", "ghost dye", "fixable viability", "fvd",
        "zombie", "7-aminoactinomycin", "aqua amine", "aamine", "l/d "
    ]):
        return True

    if "efluor" in t_clean and any(k in t_clean for k in ["fvd", "viability", "fixable", "dye", "506", "780"]):
        return True

    return False


def matches_dump_channel(text: str) -> bool:
    """Pattern matching for dump / lineage exclusion channels avoiding false positives like 'linker'."""
    if not text:
        return False
    t_clean = text.lower().strip()
    tokens = set(re.split(r"\W+", t_clean))
    if any(tok in {"dump", "lineage", "lin", "exclusion"} for tok in tokens):
        return True
    if any(term in t_clean for term in ["dump channel", "dump-channel", "lin-", "lin_"]):
        return True
    return False


@dataclass
class CleanUpGateConfig:
    """User-customizable configuration parameters for Core Clean-up Gating.

    Attributes:
        debris_method: Method for debris exclusion ('gmm' [default automated clustering], 'adaptive', or 'manual').
        debris_vertices: Custom polygon vertices [[x1, y1], [x2, y2], ...] if debris_method='manual'.
        debris_sd_scale: Boundary looseness multiplier for debris exclusion (default: 1.0; >1.0 loosens via
            Mahalanobis rescue/wider envelope, <1.0 tightens around intact cell cluster cores).
        debris_min_probability: Optional minimum GMM posterior probability across intact cell clusters
            (e.g. 0.15-0.30 loosens boundary retention; 0.75-0.90 tightens; None keeps argmax MAP boundary).
        fsc_channel: Override for forward scatter area channel (auto-detected if None).
        ssc_channel: Override for side scatter area channel (auto-detected if None).
        singlet_mode: Singlet discrimination mode: 'dual' (FSC then SSC, default),
            'single' or 'one_time' (single active singlet gate), 'fsc_only', 'ssc_only', or 'none'.
        singlet_fsc_enabled: Whether to apply forward scatter doublet discrimination (default: True).
        singlet_ssc_enabled: Whether to apply side scatter doublet discrimination (default: True).
        singlet_tolerance: Relative boundary width around diagonal pulse slope (default: 0.25 = +/- 25%).
        singlet_method: Singlet discrimination algorithm ('gmm_ratio' [default dynamic GMM pulse-ratio] or 'envelope').
        fsc_height_channel: Override for FSC height channel (auto-detected if None).
        ssc_height_channel: Override for SSC height channel (auto-detected if None).
        viability_enabled: Whether to perform viability exclusion gating (default: True).
        viability_channel: Explicit viability channel name (auto-detected if None).
        viability_cutoff: Explicit viability dye upper cutoff threshold in transformed space (auto if None).
        viability_looseness: Looseness multiplier for automated viability threshold (default: 1.0; >1.0 raises
            cutoff to retain marginal dim events, <1.0 tightens; ignored when viability_cutoff is explicitly set).
        viability_gate_range: Search interval (min, max) for automated density valley cutoff (default: (0.25, 0.85),
            accommodating low live autofluorescence to high dead dye intensity).
        dump_enabled: Whether to apply optional dump/lineage exclusion gating (default: False).
        dump_channel: Explicit dump channel name (auto-detected if None).
        dump_cutoff: Explicit dump upper cutoff threshold in transformed space (auto if None).
        dump_looseness: Looseness multiplier for automated dump channel threshold (default: 1.0; ignored when
            dump_cutoff is explicitly set).
        dump_gate_range: Search interval (min, max) for automated dump density cutoff.
        min_events_debris: Minimum intact events required to proceed to singlet gating (default: 100).
        min_events_singlet: Minimum singlets required to proceed to viability gating (default: 50).
        min_events_viability: Minimum viable events required to consider sample valid (default: 50).
    """

    debris_method: str = "gmm"
    debris_vertices: Optional[List[List[float]]] = None
    debris_sd_scale: float = 1.0
    debris_min_probability: Optional[float] = None
    fsc_channel: Optional[str] = None
    ssc_channel: Optional[str] = None
    singlet_mode: str = "dual"
    singlet_fsc_enabled: bool = True
    singlet_ssc_enabled: bool = True
    singlet_tolerance: float = 0.25
    singlet_method: str = "gmm_ratio"
    fsc_height_channel: Optional[str] = None
    ssc_height_channel: Optional[str] = None
    viability_enabled: bool = True
    viability_channel: Optional[str] = None
    viability_cutoff: Optional[float] = None
    viability_looseness: float = 1.0
    viability_gate_range: Tuple[float, float] = (0.25, 0.85)
    dump_enabled: bool = False
    dump_channel: Optional[str] = None
    dump_cutoff: Optional[float] = None
    dump_looseness: float = 1.0
    dump_gate_range: Tuple[float, float] = (0.15, 0.65)
    min_events_debris: int = 100
    min_events_singlet: int = 50
    min_events_viability: int = 50
    debris_n_components: int = 3
    show_cluster_shades: bool = True


@dataclass
class CleanUpGatingResult:
    """Container holding results, gates, masks, and statistics from Clean-up Gating.

    Attributes:
        sample_name: Identifier name of analyzed FCS sample.
        total_events: Total input event count prior to clean-up.
        hierarchy: Configured GatingHierarchy tree containing all clean-up gates.
        masks: Dictionary mapping gate names ('Cells', 'Singlets_FSC', 'Singlets_SSC', 'Viable', 'Target')
            to boolean event membership arrays.
        final_clean_mask: Boolean array indicating events that passed the entire clean-up cascade.
        clean_event_count: Number of events passing all active clean-up filters.
        clean_event_pct: Percentage of total events passing all active clean-up filters.
        statistics_df: MIFlowCyt-standard population statistics DataFrame.
        cutoffs: Dictionary of calculated or applied cutoff thresholds.
        channels_used: Dictionary mapping clean-up roles to resolved instrument channel names.
        config: CleanUpGateConfig used during execution.
        execution_time_seconds: Processing runtime in seconds.
    """

    sample_name: str
    total_events: int
    hierarchy: GatingHierarchy
    masks: Dict[str, np.ndarray]
    final_clean_mask: np.ndarray
    clean_event_count: int
    clean_event_pct: float
    statistics_df: pd.DataFrame
    cutoffs: Dict[str, float] = field(default_factory=dict)
    channels_used: Dict[str, str] = field(default_factory=dict)
    config: CleanUpGateConfig = field(default_factory=CleanUpGateConfig)
    execution_time_seconds: float = 0.0
    cluster_info: Optional[Dict[str, Any]] = None

    def get_clean_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only clean, viable, single target cell events from an events DataFrame.

        Args:
            df: DataFrame containing events corresponding to the sample.

        Returns:
            Filtered DataFrame containing only events matching final_clean_mask.
        """
        return df.loc[self.final_clean_mask].copy()

    def export_fcs(
        self,
        output_path: Union[str, Path],
        fcs_data: FCSData,
        events_df: Optional[pd.DataFrame] = None,
        is_compensated: bool = True,
    ) -> Path:
        """Export clean, gated events to a standardized FCS 3.0 / 3.1 file.

        Args:
            output_path: Destination path for clean .fcs file.
            fcs_data: Original parsed FCSData instance for metadata and spillover preservation.
            events_df: Optional source DataFrame (defaults to fcs_data.events_df).
            is_compensated: Whether exported events are already compensated (default: True).
                If True, embeds an Identity spillover matrix ($SPILLOVER = I) so downstream
                readers do not erroneously apply double-compensation.

        Returns:
            Resolved Path to exported clean FCS file.
        """
        source = events_df if events_df is not None else fcs_data.events_df
        clean_subset = self.get_clean_events(source)
        return fcs_data.export_fcs(output_path, events_df=clean_subset, is_compensated=is_compensated)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert result metrics to a flat summary dictionary for cohort logging."""
        stats_map: Dict[str, Any] = {
            "sample_name": self.sample_name,
            "total_events": self.total_events,
            "clean_events": self.clean_event_count,
            "clean_pct_total": round(self.clean_event_pct, 2),
            "execution_time_sec": round(self.execution_time_seconds, 3),
        }
        for name, mask in self.masks.items():
            if name != "root":
                stats_map[f"{name.lower()}_count"] = int(np.sum(mask))
                stats_map[f"{name.lower()}_pct"] = round(float(np.sum(mask) / self.total_events * 100.0), 2) if self.total_events > 0 else 0.0

        # Margin saturation and debris breakdown (D4)
        cluster_info = self.cluster_info
        if cluster_info is not None and "is_sat" in cluster_info:
            sat_count = int(np.sum(cluster_info["is_sat"]))
            sat_pct = round(float(sat_count / self.total_events * 100.0), 2) if self.total_events > 0 else 0.0
            cells_cnt = int(np.sum(self.masks.get("Cells", [])))
            debris_count = max(0, self.total_events - cells_cnt - sat_count)
            debris_pct = round(float(debris_count / self.total_events * 100.0), 2) if self.total_events > 0 else 0.0
        else:
            cells_cnt = int(np.sum(self.masks.get("Cells", [])))
            sat_count = 0
            sat_pct = 0.0
            debris_count = max(0, self.total_events - cells_cnt)
            debris_pct = round(float(debris_count / self.total_events * 100.0), 2) if self.total_events > 0 else 0.0

        stats_map["debris_count"] = debris_count
        stats_map["debris_pct"] = debris_pct
        stats_map["margin_saturation_count"] = sat_count
        stats_map["margin_saturation_pct"] = sat_pct

        for k, v in self.channels_used.items():
            stats_map[f"channel_{k}"] = v
        for k, v in self.cutoffs.items():
            stats_map[f"cutoff_{k}"] = round(float(v), 4)
        return stats_map


def detect_viability_channel(fcs_data: FCSData) -> Optional[str]:
    """Automatically detect the viability dye channel in an FCS dataset.

    Inspects detector labels (PnN), marker annotations (PnS), and resolved aliases
    against a comprehensive catalog of viability dyes (e.g. AARD, DAPI, 7-AAD,
    Propidium Iodide, Zombie, Live/Dead Fixable, FVS, eFluor, DRAQ7) using robust
    word boundary matching to eliminate false positive matches.

    Args:
        fcs_data: Parsed FCSData object.

    Returns:
        Matched detector channel name if detected, else None.
    """
    # 1. Search marker_to_detector mapping
    for marker, detector in fcs_data.marker_to_detector.items():
        if matches_viability_dye(marker):
            return detector

    # 2. Search stain / marker labels (pns_labels)
    for idx, pns in enumerate(fcs_data.pns_labels):
        if matches_viability_dye(pns):
            return fcs_data.pnn_labels[idx]

    # 3. Search detector labels (pnn_labels)
    for pnn in fcs_data.pnn_labels:
        if matches_viability_dye(pnn):
            return pnn

    return None


def detect_dump_channel(fcs_data: FCSData) -> Optional[str]:
    """Automatically detect optional dump / lineage exclusion channel in an FCS dataset.

    Args:
        fcs_data: Parsed FCSData object.

    Returns:
        Matched detector channel name if detected, else None.
    """
    for marker, detector in fcs_data.marker_to_detector.items():
        if matches_dump_channel(marker):
            return detector

    for idx, pns in enumerate(fcs_data.pns_labels):
        if matches_dump_channel(pns):
            return fcs_data.pnn_labels[idx]

    for pnn in fcs_data.pnn_labels:
        if matches_dump_channel(pnn):
            return pnn

    return None


def resolve_scatter_channels(
    fcs_data: FCSData,
    df_events: pd.DataFrame,
    fsc_override: Optional[str] = None,
    ssc_override: Optional[str] = None,
    fsc_h_override: Optional[str] = None,
    ssc_h_override: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Resolve forward and side scatter area, height, and width channels.

    Handles standard cytometer detector naming schemes (BD LSRII / Fortessa / Symphony,
    Beckman Coulter CytoFLEX / Navios / Kaluza, Cytek Aurora).

    Args:
        fcs_data: Parsed FCSData object.
        df_events: Events DataFrame.
        fsc_override: Optional explicit FSC-A channel override.
        ssc_override: Optional explicit SSC-A channel override.
        fsc_h_override: Optional explicit FSC-H channel override.
        ssc_h_override: Optional explicit SSC-H channel override.

    Returns:
        Dictionary mapping channel roles to resolved column names.
    """
    fsc_a = fsc_override or (fcs_data.get_fsc_channel() if hasattr(fcs_data, "get_fsc_channel") else None)
    if not fsc_a or fsc_a not in df_events.columns:
        fsc_a = next((c for c in ["FSC-A", "FS INT", "FS.INT", "FS-INT", "FSC_A", "FSC", "FS"] if c in df_events.columns), df_events.columns[0])

    ssc_a = ssc_override or (fcs_data.get_ssc_channel() if hasattr(fcs_data, "get_ssc_channel") else None)
    if not ssc_a or ssc_a not in df_events.columns:
        ssc_a = next((c for c in ["SSC-A", "SSC-B-A", "V-SSC-A", "SS INT", "SS.INT", "SS-INT", "SSC_A", "SSC", "SS"] if c in df_events.columns), (df_events.columns[1] if len(df_events.columns) > 1 else df_events.columns[0]))

    fsc_h = fsc_h_override or (fcs_data.get_fsc_h_channel() if hasattr(fcs_data, "get_fsc_h_channel") else None)
    if not fsc_h or fsc_h not in df_events.columns:
        fsc_h = next((c for c in ["FSC-H", "FS PEAK", "FS.PEAK", "FS-PEAK", "FS-H", "FSC_H", "FSC.H", "FS_H", "FS_PEAK"] if c in df_events.columns), None)

    ssc_h = ssc_h_override or (fcs_data.get_ssc_h_channel() if hasattr(fcs_data, "get_ssc_h_channel") else None)
    if not ssc_h or ssc_h not in df_events.columns:
        ssc_h = next((c for c in ["SSC-H", "SSC-B-H", "V-SSC-H", "SS PEAK", "SS.PEAK", "SS-PEAK", "SS-H", "SSC_H", "SSC.H", "SS_H", "SS_PEAK"] if c in df_events.columns), None)

    fsc_w = fcs_data.get_fsc_w_channel() if hasattr(fcs_data, "get_fsc_w_channel") else None
    if not fsc_w or fsc_w not in df_events.columns:
        fsc_w = next((c for c in ["FSC-W", "FS TOF", "FS.TOF", "FS-TOF", "FS-W", "FSC_W", "FSC.W", "FS_W", "FS_TOF", "FS WIDTH"] if c in df_events.columns), None)

    ssc_w = fcs_data.get_ssc_w_channel() if hasattr(fcs_data, "get_ssc_w_channel") else None
    if not ssc_w or ssc_w not in df_events.columns:
        ssc_w = next((c for c in ["SSC-W", "SS TOF", "SS.TOF", "SS-TOF", "SS-W", "SSC_W", "SSC.W", "SS_W", "SS_TOF", "SS WIDTH"] if c in df_events.columns), None)

    return {
        "fsc_area": fsc_a,
        "ssc_area": ssc_a,
        "fsc_height": fsc_h,
        "ssc_height": ssc_h,
        "fsc_width": fsc_w,
        "ssc_width": ssc_w,
    }


def build_debris_gate(
    fcs_data: FCSData,
    df_events: pd.DataFrame,
    config: Optional[CleanUpGateConfig] = None,
) -> PolygonGate:
    """Construct a Debris Exclusion Polygon Gate for FSC-A vs SSC-A.

    Isolates intact, living cell bodies and excludes low-scatter sub-cellular fragments,
    erythrocyte ghosts, and electronic/optical baseline noise.

    Args:
        fcs_data: Parsed FCS sample data.
        df_events: Events DataFrame in linear space.
        config: Optional CleanUpGateConfig configuration.

    Returns:
        Configured PolygonGate for 'Cells'.
    """
    cfg = config or CleanUpGateConfig()
    channels = resolve_scatter_channels(
        fcs_data,
        df_events,
        fsc_override=cfg.fsc_channel,
        ssc_override=cfg.ssc_channel,
    )
    fsc_col = channels["fsc_area"]
    ssc_col = channels["ssc_area"]

    if cfg.debris_method == "manual" and cfg.debris_vertices:
        return PolygonGate(
            name="Cells",
            channels=[fsc_col, ssc_col],
            vertices=cfg.debris_vertices,
            parent_gate=None,
        )

    fsc_vals = df_events[fsc_col].to_numpy(dtype=np.float64)
    ssc_vals = df_events[ssc_col].to_numpy(dtype=np.float64)

    fsc_max_scale = float(fcs_data.channel_ranges.get(fsc_col, np.max(fsc_vals)))
    ssc_max_scale = float(fcs_data.channel_ranges.get(ssc_col, np.max(ssc_vals)))
    if fsc_max_scale <= 0:
        fsc_max_scale = 262144.0
    if ssc_max_scale <= 0:
        ssc_max_scale = 262144.0

    # Defensive dynamic scale adjustment:
    # If the empirical upper ceiling of scatter values is drastically smaller than
    # fsc_max_scale (e.g. detector saturation rail or legacy gain mismatch where
    # thousands of events saturate at a level much below fsc_max_scale), adjust the
    # normalization scale to the true physical ceiling.
    if len(fsc_vals) > 100:
        fsc_max = float(np.max(fsc_vals))
        if fsc_max > 0 and fsc_max_scale > fsc_max * 1.5:
            sat_cnt = int(np.sum(fsc_vals >= fsc_max * 0.995))
            if sat_cnt >= 50 or (sat_cnt / len(fsc_vals)) > 0.0005:
                fsc_max_scale = fsc_max

    if len(ssc_vals) > 100:
        ssc_max = float(np.max(ssc_vals))
        if ssc_max > 0 and ssc_max_scale > ssc_max * 1.5:
            sat_cnt = int(np.sum(ssc_vals >= ssc_max * 0.995))
            if sat_cnt >= 50 or (sat_cnt / len(ssc_vals)) > 0.0005:
                ssc_max_scale = ssc_max

    if cfg.debris_method == "gmm" and len(df_events) > 200:
        # 1. Identify true hardware detector ceiling saturation and baseline zero noise spikes first
        ceil_fsc = fsc_max_scale
        ceil_ssc = ssc_max_scale
        is_sat = (fsc_vals >= ceil_fsc * 0.98) | (ssc_vals >= ceil_ssc * 0.98)

        # 2. Compute empirical 99.5th percentile of non-saturated events so that ultra-wide
        # dynamic range cytometers (24-bit CytoFLEX, 30-bit NovoCyte Penteon, Cytek Aurora)
        # do not compress intact cells into the bottom <15% of [0, 1] scatter space.
        non_sat_fsc = fsc_vals[(~is_sat) & np.isfinite(fsc_vals) & (fsc_vals > 0)]
        non_sat_ssc = ssc_vals[(~is_sat) & np.isfinite(ssc_vals) & (ssc_vals > 0)]
        if len(non_sat_fsc) > 100:
            p995_fsc = float(np.percentile(non_sat_fsc, 99.5))
            if p995_fsc > 0 and fsc_max_scale > p995_fsc * 1.4:
                fsc_max_scale = p995_fsc * 1.05
        if len(non_sat_ssc) > 100:
            p995_ssc = float(np.percentile(non_sat_ssc, 99.5))
            if p995_ssc > 0 and ssc_max_scale > p995_ssc * 1.4:
                ssc_max_scale = p995_ssc * 1.05

        # GMM clustering on normalized scatter space to separate cells from debris & saturation
        norm_fsc = np.clip(fsc_vals / fsc_max_scale, -0.05, 1.04)
        norm_ssc = np.clip(ssc_vals / ssc_max_scale, -0.05, 1.04)

        is_noise = (norm_fsc <= 0.02) | (norm_ssc <= 0.01)

        valid_idx = (
            (~is_sat)
            & (~is_noise)
            & (norm_fsc < 1.05)
            & (norm_ssc < 1.05)
            & np.isfinite(norm_fsc)
            & np.isfinite(norm_ssc)
        )
        sub_idx = np.where(valid_idx)[0]

        if len(sub_idx) > 500:
            rng = np.random.RandomState(42)
            train_idx = rng.choice(sub_idx, size=min(25000, len(sub_idx)), replace=False)
            X = np.column_stack([norm_fsc[train_idx], norm_ssc[train_idx]])
            n_comp = max(2, cfg.debris_n_components)
            gmm = GaussianMixture(n_components=n_comp, random_state=42, max_iter=100)
            gmm.fit(X)

            cell_clusters: List[int] = []
            debris_clusters: List[int] = []
            sat_clusters: List[int] = []

            for k in range(n_comp):
                mean_fsc = gmm.means_[k, 0]
                mean_ssc = gmm.means_[k, 1]
                comp_weight = float(gmm.weights_[k])
                if mean_fsc >= 0.96 or mean_ssc >= 0.96:
                    sat_clusters.append(k)
                elif (
                    mean_fsc < 0.08
                    or (mean_fsc < 0.12 and mean_ssc < 0.08)
                    or (mean_fsc < 0.15 and mean_ssc < 0.10 and comp_weight < 0.30)
                    or (mean_ssc < 0.025 and mean_fsc < 0.15)
                ):
                    debris_clusters.append(k)
                else:
                    cell_clusters.append(k)

            # Fallback: ensure at least one cluster is designated as cells if all fell into debris/sat
            if not cell_clusters:
                cands = [k for k in range(n_comp) if k not in sat_clusters]
                if cands:
                    best_k = int(cands[np.argmax([gmm.means_[k, 0] + gmm.means_[k, 1] for k in cands])])
                    cell_clusters = [best_k]
                else:
                    cell_clusters = [int(np.argmax(gmm.weights_))]

            # Sort cell clusters by SSC mean to differentiate biological leukocyte subsets (e.g. Lymphocytes vs Granulocytes)
            cell_clusters = sorted(cell_clusters, key=lambda k: gmm.means_[k, 1])

            norm_fsc_safe = np.nan_to_num(norm_fsc, nan=0.0, posinf=1.0, neginf=0.0)
            norm_ssc_safe = np.nan_to_num(norm_ssc, nan=0.0, posinf=1.0, neginf=0.0)
            pred_all = gmm.predict(np.column_stack([norm_fsc_safe, norm_ssc_safe]))

            # Assign noise and saturation events to dedicated outlier IDs
            if debris_clusters:
                pred_all[is_noise] = debris_clusters[0]
            else:
                pred_all[is_noise] = -2

            if np.any(is_sat):
                pred_all[is_sat] = -1

            # Strict intact cell mask: belongs to an intact cell cluster AND not saturation AND not noise
            cell_mask = np.isin(pred_all, cell_clusters) & (~is_sat) & (~is_noise)

            # Build universal population names and styling palette: Intact Cells, Debris, Margin Saturation
            cluster_names: Dict[int, str] = {}
            cluster_colors: Dict[int, str] = {}

            for k in cell_clusters:
                cluster_names[k] = "Intact Cells"
                cluster_colors[k] = "#2ca02c"  # Nature green for viable intact cells

            for k in debris_clusters:
                cluster_names[k] = "Debris / Low Scatter"
                cluster_colors[k] = "#7f7f7f"  # Slate gray

            if -2 in pred_all:
                cluster_names[-2] = "Debris / Low Scatter"
                cluster_colors[-2] = "#7f7f7f"  # Slate gray

            if -1 in pred_all:
                cluster_names[-1] = "Margin Saturation"
                cluster_colors[-1] = "#e53935"  # Warning red

            # Compute Gaussian component means and covariance matrices in unnormalized scatter coordinates
            scale_diag = np.array([fsc_max_scale, ssc_max_scale], dtype=np.float64)
            means_raw: List[np.ndarray] = []
            covs_raw: List[np.ndarray] = []
            for k in range(n_comp):
                means_raw.append(gmm.means_[k] * scale_diag)
                covs_raw.append(gmm.covariances_[k] * np.outer(scale_diag, scale_diag))

            cluster_info: Dict[str, Any] = {
                "method": "gmm",
                "n_components": n_comp,
                "cell_clusters": cell_clusters,
                "sat_clusters": sat_clusters,
                "debris_clusters": debris_clusters,
                "pred_labels": pred_all,
                "cell_mask": cell_mask,
                "is_sat": is_sat,
                "cluster_names": cluster_names,
                "cluster_colors": cluster_colors,
                "means": means_raw,
                "covariances": covs_raw,
                "weights": gmm.weights_,
            }

            cell_pts = np.column_stack([fsc_vals[cell_mask], ssc_vals[cell_mask]])
            vertices: List[List[float]] = []
            if len(cell_pts) > 50:
                # To prevent outlier points from drawing the hull across the debris region,
                # clip the points used for representative visual hull to above the 1.5th percentile of SSC
                ssc_floor = float(np.percentile(cell_pts[:, 1], 1.5))
                clean_cell_pts = cell_pts[cell_pts[:, 1] >= ssc_floor]
                if len(clean_cell_pts) > 30:
                    hull_size = min(5000, len(clean_cell_pts))
                    h_idx = rng.choice(len(clean_cell_pts), size=hull_size, replace=False)
                    hull = ConvexHull(clean_cell_pts[h_idx])
                    vertices = clean_cell_pts[h_idx][hull.vertices].tolist()
                else:
                    hull = ConvexHull(cell_pts)
                    vertices = cell_pts[hull.vertices].tolist()

            gmm_gate = GMMClusterGate(
                name="Cells",
                channels=[fsc_col, ssc_col],
                means=gmm.means_,
                covariances=gmm.covariances_,
                weights=gmm.weights_,
                cell_clusters=cell_clusters,
                scale_diag=[fsc_max_scale, ssc_max_scale],
                fsc_sat_thresh=float((ceil_fsc * 0.98) / fsc_max_scale),
                ssc_sat_thresh=float((ceil_ssc * 0.98) / ssc_max_scale),
                fsc_noise_thresh=0.02,
                ssc_noise_thresh=0.01,
                debris_sd_scale=float(cfg.debris_sd_scale),
                debris_min_probability=cfg.debris_min_probability,
                vertices=vertices,
                parent_gate=None,
                metadata={"cluster_info": cluster_info},
            )
            if abs(float(cfg.debris_sd_scale) - 1.0) > 1e-6 or cfg.debris_min_probability is not None:
                try:
                    eff_mask = gmm_gate.apply(np.column_stack([fsc_vals, ssc_vals]))
                    eff_pts = np.column_stack([fsc_vals[eff_mask], ssc_vals[eff_mask]])
                    if len(eff_pts) >= 10:
                        hull_n = min(5000, len(eff_pts))
                        h_idx = rng.choice(len(eff_pts), size=hull_n, replace=False) if len(eff_pts) > hull_n else np.arange(len(eff_pts))
                        eff_hull = ConvexHull(eff_pts[h_idx])
                        gmm_gate.vertices = np.asarray(eff_pts[h_idx][eff_hull.vertices], dtype=np.float64)
                except Exception:
                    pass
            return gmm_gate

    # General adaptive scatter envelope
    fsc_min = float(max(fsc_max_scale * 0.15, np.percentile(fsc_vals, 6.0)))
    fsc_max = float(min(fsc_max_scale * 0.96, np.percentile(fsc_vals, 99.0)))
    ssc_min = float(max(ssc_max_scale * 0.04, np.percentile(ssc_vals, 2.5)))
    ssc_max = float(min(ssc_max_scale * 0.92, np.percentile(ssc_vals, 98.0)))

    sd_scale = max(0.25, float(cfg.debris_sd_scale))
    if abs(sd_scale - 1.0) > 1e-6:
        fsc_mid = 0.5 * (fsc_min + fsc_max)
        ssc_mid = 0.5 * (ssc_min + ssc_max)
        fsc_half = max(1e-6, 0.5 * (fsc_max - fsc_min)) * sd_scale
        ssc_half = max(1e-6, 0.5 * (ssc_max - ssc_min)) * sd_scale
        fsc_min = float(max(0.0, fsc_mid - fsc_half))
        fsc_max = float(min(fsc_max_scale * 0.98, fsc_mid + fsc_half))
        ssc_min = float(max(0.0, ssc_mid - ssc_half))
        ssc_max = float(min(ssc_max_scale * 0.96, ssc_mid + ssc_half))

    vertices = [
        [fsc_min, ssc_min],
        [fsc_max, ssc_min * 1.3],
        [fsc_max, ssc_max],
        [fsc_min * 1.1, ssc_max],
        [fsc_min, ssc_max * 0.5],
    ]

    return PolygonGate(
        name="Cells",
        channels=[fsc_col, ssc_col],
        vertices=vertices,
        parent_gate=None,
    )


def build_singlet_gate(
    fcs_data: FCSData,
    df_events: pd.DataFrame,
    scatter_axis: str = "fsc",
    parent_mask: Optional[np.ndarray] = None,
    parent_gate: Optional[str] = "Cells",
    config: Optional[CleanUpGateConfig] = None,
) -> Tuple[Gate, float]:
    """Construct an adaptive Singlet Discrimination Gate for FSC or SSC pulse geometry.

    Evaluates the empirical linear area-to-height pulse ratio (H = m * A) or area-to-width
    ratio and constructs a diagonal tolerance envelope to filter out doublets and clumps.

    Args:
        fcs_data: Parsed FCS sample data.
        df_events: Events DataFrame in linear space.
        scatter_axis: 'fsc' for Forward Scatter, 'ssc' for Side Scatter.
        parent_mask: Boolean mask of valid parent intact cells.
        parent_gate: Identifier name of parent gate in hierarchy.
        config: Optional CleanUpGateConfig.

    Returns:
        Tuple containing:
            - Constructed Gate (PolygonGate).
            - Estimated pulse ratio slope m (float).
    """
    cfg = config or CleanUpGateConfig()
    channels = resolve_scatter_channels(
        fcs_data,
        df_events,
        fsc_override=cfg.fsc_channel,
        ssc_override=cfg.ssc_channel,
        fsc_h_override=cfg.fsc_height_channel,
        ssc_h_override=cfg.ssc_height_channel,
    )

    axis_clean = scatter_axis.lower().strip()
    gate_name = "Singlets_FSC" if axis_clean == "fsc" else "Singlets_SSC"

    if axis_clean == "fsc":
        area_ch = channels["fsc_area"]
        height_ch = channels["fsc_height"] or channels["fsc_width"]
    else:
        area_ch = channels["ssc_area"]
        height_ch = channels["ssc_height"] or channels["ssc_width"]

    area_max = float(fcs_data.channel_ranges.get(area_ch, np.max(df_events[area_ch])))
    if area_max <= 0:
        area_max = 262144.0
    if len(df_events) > 100 and area_ch in df_events.columns:
        a_max = float(np.max(df_events[area_ch]))
        if a_max > 0 and area_max > a_max * 1.5:
            sat_cnt = int(np.sum(df_events[area_ch] >= a_max * 0.995))
            if sat_cnt >= 50 or (sat_cnt / len(df_events)) > 0.0005:
                area_max = a_max

    # Fallback if height/width channel is not recorded on this cytometer
    if not height_ch or height_ch not in df_events.columns:
        logger.warning(
            f"Singlet gating for {scatter_axis.upper()} skipped: no corresponding Height or Width channel found. "
            f"Creating pass-through gate for '{gate_name}'."
        )
        return (
            PolygonGate(
                name=gate_name,
                channels=[area_ch, area_ch],
                vertices=[
                    [-area_max * 10.0, -area_max * 10.0],
                    [area_max * 50.0, -area_max * 10.0],
                    [area_max * 50.0, area_max * 50.0],
                    [-area_max * 10.0, area_max * 50.0],
                ],
                parent_gate=parent_gate,
            ),
            1.0,
        )

    height_max = float(fcs_data.channel_ranges.get(height_ch, np.max(df_events[height_ch]))) if height_ch in df_events.columns else area_max
    if height_max <= 0:
        height_max = area_max
    if len(df_events) > 100 and height_ch in df_events.columns:
        h_max = float(np.max(df_events[height_ch]))
        if h_max > 0 and height_max > h_max * 1.5:
            sat_cnt = int(np.sum(df_events[height_ch] >= h_max * 0.995))
            if sat_cnt >= 50 or (sat_cnt / len(df_events)) > 0.0005:
                height_max = h_max

    # Expected ratio based on instrument digitizer bit depths (e.g. 1.0 for BD, ~0.125 for Beckman Coulter)
    expected_slope = float(height_max / area_max) if area_max > 0 else 1.0

    # Extract parent subset
    if parent_mask is not None and np.sum(parent_mask) > 50:
        sub_a = df_events.loc[parent_mask, area_ch].to_numpy(dtype=np.float64)
        sub_h = df_events.loc[parent_mask, height_ch].to_numpy(dtype=np.float64)
    else:
        sub_a = df_events[area_ch].to_numpy(dtype=np.float64)
        sub_h = df_events[height_ch].to_numpy(dtype=np.float64)

    # Filter out baseline noise and zeros
    noise_floor_a = float(area_max * 0.015) if area_max > 0 else 0.0
    noise_floor_h = float(height_max * 0.015) if height_max > 0 else 0.0
    valid_pulse = (sub_a > noise_floor_a) & (sub_h > noise_floor_h) & np.isfinite(sub_a) & np.isfinite(sub_h)

    clean_ratios = np.empty(0, dtype=np.float64)
    if np.sum(valid_pulse) > 30:
        ratios = sub_h[valid_pulse] / sub_a[valid_pulse]
        # Restrict to realistic pulse ratio domain accommodating asymmetrical ADCs (0.02 - 10.0)
        clean_ratios = ratios[(ratios >= 0.02) & (ratios <= 10.0)]
        slope = float(np.median(clean_ratios)) if len(clean_ratios) > 30 else expected_slope
    else:
        slope = expected_slope

    # Enclose the full legitimate scatter area domain from baseline noise floor
    # to instrument dynamic ceiling without arbitrary percentile truncation
    x_min = float(max(0.0, noise_floor_a))
    x_max = float(max(area_max * 1.02, float(np.max(sub_a)) * 1.01 if len(sub_a) > 0 else area_max))

    # Diagonal envelope bounds estimation
    tol = max(0.05, min(0.85, cfg.singlet_tolerance))
    lower_factor = max(0.20, 1.0 - tol)
    upper_factor = min(2.00, 1.0 + tol)
    gmm_singlet_meta: Dict[str, Any] = {"singlet_method": "envelope", "singlet_tolerance": tol}

    # Dynamic 1D GMM pulse-ratio singlet estimation
    if cfg.singlet_method == "gmm_ratio" and len(clean_ratios) > 50:
        w = clean_ratios / slope
        rng = np.random.RandomState(42)
        train_w = rng.choice(w, size=min(25000, len(w)), replace=False)
        try:
            gmm_s = GaussianMixture(n_components=2, random_state=42, max_iter=100)
            gmm_s.fit(train_w.reshape(-1, 1))

            # Singlet component is the one whose normalized mean is closest to 1.0
            s_idx = int(np.argmin(np.abs(gmm_s.means_.flatten() - 1.0)))
            mu_norm = float(gmm_s.means_[s_idx, 0])
            var_norm = float(gmm_s.covariances_[s_idx, 0, 0]) if gmm_s.covariance_type == "full" else float(gmm_s.covariances_[s_idx])
            sigma_norm = float(np.sqrt(max(1e-6, var_norm)))
            # Restrict empirical sigma to realistic cytometer bounds
            sigma_norm = max(0.020, min(0.20, sigma_norm))

            # Scale k_sigma according to user-configured singlet_tolerance (default: 0.25 -> k = 2.5)
            k_sigma = 2.5 * (tol / 0.25)
            lower_factor = max(0.20, mu_norm - k_sigma * sigma_norm)
            upper_factor = min(2.00, mu_norm + k_sigma * sigma_norm)

            gmm_singlet_meta = {
                "singlet_method": "gmm_ratio",
                "singlet_mu": mu_norm * slope,
                "singlet_sigma": sigma_norm * slope,
                "singlet_k_sigma": k_sigma,
                "singlet_tolerance": tol,
                "singlet_lower_factor": lower_factor,
                "singlet_upper_factor": upper_factor,
            }
        except Exception as e:
            logger.debug(f"1D GMM pulse-ratio singlet estimation fallback to tolerance envelope: {e}")

    y_low_start = float(max(0.0, x_min * slope * lower_factor))
    y_low_end = float(x_max * slope * lower_factor)
    y_high_end = float(x_max * slope * upper_factor)
    y_high_start = float(x_min * slope * upper_factor)

    vertices = [
        [x_min, y_low_start],
        [x_max, y_low_end],
        [x_max, y_high_end],
        [x_min, y_high_start],
    ]

    gate = PolygonGate(
        name=gate_name,
        channels=[area_ch, height_ch],
        vertices=vertices,
        parent_gate=parent_gate,
        metadata=gmm_singlet_meta,
    )
    return gate, slope


def _adjust_cutoff_looseness(
    vals: np.ndarray,
    base_cutoff: float,
    looseness: float,
    min_bound: float,
    max_bound: float,
) -> float:
    """Shift an automated upper cutoff by (looseness - 1.0) * 1.5 * robust_sd of negative events."""
    eff_loose = max(0.25, float(looseness))
    if abs(eff_loose - 1.0) <= 1e-6 or len(vals) == 0:
        return float(base_cutoff)
    left = vals[vals <= base_cutoff]
    if len(left) > 10:
        med_left = float(np.median(left))
        mad_left = float(np.median(np.abs(left - med_left)))
        robust_sd = 1.4826 * mad_left if mad_left > 0 else 0.035
    else:
        robust_sd = 0.035
    shifted = float(base_cutoff) + (eff_loose - 1.0) * 1.5 * robust_sd
    return float(np.clip(shifted, min_bound, max_bound))


def build_viability_gate(
    fcs_data: FCSData,
    df_transformed: pd.DataFrame,
    parent_mask: Optional[np.ndarray] = None,
    parent_gate: str = "Singlets",
    config: Optional[CleanUpGateConfig] = None,
) -> Tuple[Optional[RangeGate], Optional[float]]:
    """Construct an automated Viability Gate to exclude dead and dying cells.

    Operates on compensated and transformed fluorescence data. Live cells exclude the
    viability dye (remaining at baseline / negative values), whereas dead cells with
    compromised membranes stain brightly positive.

    Args:
        fcs_data: Parsed FCS sample data.
        df_transformed: Transformed DataFrame.
        parent_mask: Boolean mask of valid parent singlet events.
        parent_gate: Parent gate name in hierarchy.
        config: Optional CleanUpGateConfig.

    Returns:
        Tuple containing:
            - Configured RangeGate (name='Viable') or None if dye not available.
            - Calculated or applied cutoff threshold (float).
    """
    cfg = config or CleanUpGateConfig()
    if not cfg.viability_enabled:
        return None, None

    viab_ch = cfg.viability_channel or detect_viability_channel(fcs_data)
    if not viab_ch or viab_ch not in df_transformed.columns:
        logger.info("Viability gating skipped: no viability dye channel detected or configured.")
        return None, None

    # If explicit cutoff provided, construct directly
    if cfg.viability_cutoff is not None:
        cutoff = float(cfg.viability_cutoff)
        gate = RangeGate(
            name="Viable",
            channel=viab_ch,
            max_val=cutoff,
            parent_gate=parent_gate,
        )
        return gate, cutoff

    # Automated density thresholding on transformed parent singlet events
    if parent_mask is not None and np.sum(parent_mask) > cfg.min_events_singlet:
        vals = df_transformed.loc[parent_mask, viab_ch].dropna().to_numpy()
    else:
        vals = df_transformed[viab_ch].dropna().to_numpy()

    if len(vals) < cfg.min_events_viability:
        logger.warning(
            f"Insufficient events ({len(vals)}) for automated viability gating on '{viab_ch}'. Using default cutoff."
        )
        cutoff = 0.55
        gate = RangeGate(
            name="Viable",
            channel=viab_ch,
            max_val=cutoff,
            parent_gate=parent_gate,
        )
        return gate, cutoff

    res: DensityCutoffResult = find_density_minimum(
        values=vals,
        channel_name=viab_ch,
        gate_range=cfg.viability_gate_range,
        default_cutoff=0.55,
        prefer_first_valley=True,
    )

    v_loose = max(0.25, float(cfg.viability_looseness))
    val_max = float(np.percentile(vals, 99.5)) if len(vals) > 0 else 1.0
    val_min = float(np.percentile(vals, 0.5)) if len(vals) > 0 else 0.0
    val_span = val_max - val_min
    is_normalized_unit = (val_span <= 2.5 and val_max <= 2.0 and val_min >= -1.0)

    if res.is_bimodal and res.cutoff is not None:
        cutoff = float(res.cutoff)
        if abs(v_loose - 1.0) > 1e-6:
            min_v = cfg.viability_gate_range[0] if (is_normalized_unit and cfg.viability_gate_range) else val_min
            max_v = (
                min(0.98, (cfg.viability_gate_range[1] if cfg.viability_gate_range else 0.85) + max(0.0, v_loose - 1.0) * 0.13)
                if is_normalized_unit
                else val_max
            )
            cutoff = _adjust_cutoff_looseness(vals, cutoff, v_loose, min_v, max_v)
        method_desc = f"bimodal valley at {cutoff:.4f} (looseness={v_loose:.2f})"
    else:
        # Option C: Robust Tail Statistics for Unimodal Samples (default for high-viability cohorts)
        # When sample viability is very high (>95%), dead cells exist only as a low-frequency
        # continuum without an independent secondary Gaussian peak.
        # Estimate the spread of the live/negative population using the uncontaminated
        # left side (values <= mode) and set cutoff at mode + (4.5 * viability_looseness) * robust_sd.
        hist_counts, hist_edges = np.histogram(vals, bins=200)
        mode_idx = int(np.argmax(hist_counts))
        mode_val = float((hist_edges[mode_idx] + hist_edges[mode_idx + 1]) / 2.0)

        left_side = vals[vals <= mode_val]
        if len(left_side) > 10:
            mad_left = float(np.median(np.abs(left_side - mode_val)))
            robust_sd = 1.4826 * mad_left if mad_left > 0 else 0.035
        else:
            robust_sd = 0.035

        tail_cutoff = mode_val + (4.5 * v_loose) * robust_sd

        if is_normalized_unit:
            min_b = cfg.viability_gate_range[0] if cfg.viability_gate_range else 0.25
            base_max_b = cfg.viability_gate_range[1] if cfg.viability_gate_range else 0.85
            max_b = min(0.98, base_max_b + 0.13 * min(1.0, v_loose - 1.0)) if v_loose > 1.0 else base_max_b
            cutoff = float(np.clip(tail_cutoff, min_b, max_b))
        else:
            min_b = mode_val + 1.0 * robust_sd
            max_b = val_max
            cutoff = float(np.clip(tail_cutoff, min_b, max_b))

        method_desc = f"unimodal robust tail (mode={mode_val:.3f}, sd={robust_sd:.3f}, looseness={v_loose:.2f} -> cutoff={cutoff:.4f})"

    gate = RangeGate(
        name="Viable",
        channel=viab_ch,
        max_val=round(cutoff, 4),
        parent_gate=parent_gate,
    )
    logger.debug(f"Automated viability cutoff on '{viab_ch}': {cutoff:.4f} via {method_desc}")
    return gate, round(cutoff, 4)


def build_dump_gate(
    fcs_data: FCSData,
    df_transformed: pd.DataFrame,
    parent_mask: Optional[np.ndarray] = None,
    parent_gate: str = "Viable",
    config: Optional[CleanUpGateConfig] = None,
) -> Tuple[Optional[RangeGate], Optional[float]]:
    """Construct an optional Dump / Lineage Exclusion Gate to isolate target cells.

    Filters out unwanted cell lineages stained with a cocktail of lineage markers
    sharing a single fluorophore detector channel.

    Args:
        fcs_data: Parsed FCS sample data.
        df_transformed: Transformed DataFrame.
        parent_mask: Boolean mask of valid parent viable events.
        parent_gate: Parent gate name in hierarchy.
        config: Optional CleanUpGateConfig.

    Returns:
        Tuple containing:
            - Configured RangeGate (name='Target') or None if dump channel not present.
            - Calculated or applied cutoff threshold (float).
    """
    cfg = config or CleanUpGateConfig()
    if not cfg.dump_enabled:
        return None, None

    dump_ch = cfg.dump_channel or detect_dump_channel(fcs_data)
    if not dump_ch or dump_ch not in df_transformed.columns:
        logger.info("Dump gating skipped: no dump channel detected or configured.")
        return None, None

    if cfg.dump_cutoff is not None:
        cutoff = float(cfg.dump_cutoff)
    else:
        if parent_mask is not None and np.sum(parent_mask) > cfg.min_events_viability:
            vals = df_transformed.loc[parent_mask, dump_ch].dropna().to_numpy()
        else:
            vals = df_transformed[dump_ch].dropna().to_numpy()

        res: DensityCutoffResult = find_density_minimum(
            values=vals,
            channel_name=dump_ch,
            gate_range=cfg.dump_gate_range,
            default_cutoff=0.35,
            prefer_first_valley=True,
        )
        cutoff = float(res.cutoff) if res.cutoff is not None else 0.35
        d_loose = max(0.25, float(cfg.dump_looseness))
        if abs(d_loose - 1.0) > 1e-6 and len(vals) > 0:
            val_min = float(np.percentile(vals, 0.5))
            val_max = float(np.percentile(vals, 99.5))
            cutoff = _adjust_cutoff_looseness(vals, cutoff, d_loose, val_min, val_max)

    gate = RangeGate(
        name="Target",
        channel=dump_ch,
        max_val=round(cutoff, 4) if cutoff is not None else 0.35,
        parent_gate=parent_gate,
    )
    return gate, round(cutoff, 4) if cutoff is not None else 0.35


def run_cleanup_gating(
    fcs_data: FCSData,
    df_events: Optional[pd.DataFrame] = None,
    df_transformed: Optional[pd.DataFrame] = None,
    config: Optional[CleanUpGateConfig] = None,
) -> CleanUpGatingResult:
    """Execute complete end-to-end Core Clean-up Gating cascade on an FCS dataset.

    Executes sequential hierarchical filtering:
    1. All Events -> Cells (Debris Out)
    2. Cells -> Singlets_FSC (FSC Doublets Out)
    3. Singlets_FSC -> Singlets_SSC (SSC Doublets Out, optional)
    4. Singlets -> Viable (Dead Out, optional)
    5. Viable -> Target (Dump Out, optional)

    Args:
        fcs_data: Parsed FCSData object.
        df_events: Linear events DataFrame (defaults to fcs_data.events_df).
        df_transformed: Transformed events DataFrame (defaults to df_events if not provided).
        config: Optional CleanUpGateConfig configuration.

    Returns:
        Populated CleanUpGatingResult instance with masks, hierarchy, and statistics.
    """
    import time
    t0 = time.perf_counter()

    cfg = config or CleanUpGateConfig()
    events = df_events if df_events is not None else fcs_data.events_df
    if df_transformed is not None:
        xformed = df_transformed
    else:
        # If viability or dump gating is enabled and fluorescence channels exist,
        # apply Logicle transform so thresholding operates on standardized Logicle distributions
        if (cfg.viability_enabled or cfg.dump_enabled) and len(events) > 0:
            opt_cols = [
                c for c in events.columns
                if not c.startswith("FSC") and not c.startswith("SSC") and c != "Time"
            ]
            if opt_cols:
                try:
                    from flowx.math.transforms import apply_transform_to_dataframe, LogicleTransform
                    logicle = LogicleTransform()
                    xformed = apply_transform_to_dataframe(events, opt_cols, logicle)
                except Exception:
                    xformed = events
            else:
                xformed = events
        else:
            xformed = events
    n_total = len(events)

    hierarchy = GatingHierarchy()
    cutoffs: Dict[str, float] = {}
    channels_used: Dict[str, str] = {}

    # Step 1: Debris Exclusion (FSC vs SSC)
    cells_gate = build_debris_gate(fcs_data, events, config=cfg)
    hierarchy.add_gate(cells_gate)
    channels_used["debris_fsc"] = cells_gate.channels[0]
    channels_used["debris_ssc"] = cells_gate.channels[1]
    cluster_info = getattr(cells_gate, "metadata", {}).get("cluster_info", None)

    # Evaluate step 1 to pass parent mask to singlets
    interim_masks = hierarchy.evaluate(events, transformed_df=xformed)
    cells_mask = interim_masks["Cells"]

    # Minimum event validation after debris exclusion
    n_intact = int(np.sum(cells_mask))
    if n_intact < cfg.min_events_debris:
        logger.warning(
            f"Intact cell count ({n_intact}) is below min_events_debris ({cfg.min_events_debris}). "
            f"Sample may contain insufficient intact cellular events."
        )

    # Margin saturation check and warning (D4)
    if cluster_info is not None and "is_sat" in cluster_info:
        sat_cnt = int(np.sum(cluster_info["is_sat"]))
        sat_pct = (sat_cnt / n_total * 100.0) if n_total > 0 else 0.0
        if sat_pct > 5.0:
            logger.warning(
                f"High margin saturation detected in '{getattr(fcs_data, 'sample_name', 'sample')}': "
                f"{sat_cnt:,} events ({sat_pct:.1f}% >= 0.98 dynamic range). "
                "Detector PMT voltages/gains may be saturated or set too high."
            )

    # Resolve singlet execution based on singlet_mode and explicit booleans
    s_mode = (cfg.singlet_mode or "dual").lower().strip()
    do_fsc = cfg.singlet_fsc_enabled
    do_ssc = cfg.singlet_ssc_enabled

    if s_mode in ("none", "disabled", "off"):
        do_fsc = False
        do_ssc = False
    elif s_mode in ("fsc", "fsc_only"):
        do_fsc = True
        do_ssc = False
    elif s_mode in ("ssc", "ssc_only"):
        do_fsc = False
        do_ssc = True
    elif s_mode in ("single", "one_time", "once", "single_step"):
        # Single singlet gating: prefer scatter channel with active height/peak channel recorded
        scatter_ch = resolve_scatter_channels(
            fcs_data,
            events,
            fsc_override=cfg.fsc_channel,
            ssc_override=cfg.ssc_channel,
            fsc_h_override=cfg.fsc_height_channel,
            ssc_h_override=cfg.ssc_height_channel,
        )
        fsc_pulse = scatter_ch.get("fsc_height") or scatter_ch.get("fsc_width")
        ssc_pulse = scatter_ch.get("ssc_height") or scatter_ch.get("ssc_width")
        fsc_has_height = bool(fsc_pulse and fsc_pulse != scatter_ch.get("fsc_area"))
        ssc_has_height = bool(ssc_pulse and ssc_pulse != scatter_ch.get("ssc_area"))

        if fsc_has_height:
            do_fsc = True
            do_ssc = False
        elif ssc_has_height:
            do_fsc = False
            do_ssc = True
        else:
            # Fallback to FSC pass-through
            do_fsc = True
            do_ssc = False

    channels_used["singlet_mode"] = s_mode

    # Step 2: Forward Scatter Singlets
    current_parent = "Cells"
    if do_fsc:
        singlet_fsc_gate, fsc_slope = build_singlet_gate(
            fcs_data,
            events,
            scatter_axis="fsc",
            parent_mask=cells_mask,
            parent_gate=current_parent,
            config=cfg,
        )
        hierarchy.add_gate(singlet_fsc_gate)
        channels_used["singlet_fsc_area"] = singlet_fsc_gate.channels[0]
        channels_used["singlet_fsc_height"] = singlet_fsc_gate.channels[1]
        is_fsc_active = singlet_fsc_gate.channels[0] != singlet_fsc_gate.channels[1]
        channels_used["singlet_fsc_status"] = "active" if is_fsc_active else "pass_through"
        if is_fsc_active:
            cutoffs["singlet_fsc_slope"] = fsc_slope
        current_parent = "Singlets_FSC"

    # Step 3: Side Scatter Singlets
    if do_ssc:
        interim_masks = hierarchy.evaluate(events, transformed_df=xformed)
        parent_singlet_mask = interim_masks[current_parent]
        singlet_ssc_gate, ssc_slope = build_singlet_gate(
            fcs_data,
            events,
            scatter_axis="ssc",
            parent_mask=parent_singlet_mask,
            parent_gate=current_parent,
            config=cfg,
        )
        hierarchy.add_gate(singlet_ssc_gate)
        channels_used["singlet_ssc_area"] = singlet_ssc_gate.channels[0]
        channels_used["singlet_ssc_height"] = singlet_ssc_gate.channels[1]
        is_ssc_active = singlet_ssc_gate.channels[0] != singlet_ssc_gate.channels[1]
        channels_used["singlet_ssc_status"] = "active" if is_ssc_active else "pass_through"
        if is_ssc_active:
            cutoffs["singlet_ssc_slope"] = ssc_slope
        current_parent = "Singlets_SSC"

    # Step 4: Viability Gating (Live/Dead Dye Exclusion)
    if cfg.viability_enabled:
        interim_masks = hierarchy.evaluate(events, transformed_df=xformed)
        parent_viab_mask = interim_masks[current_parent]
        viab_gate, viab_cutoff = build_viability_gate(
            fcs_data,
            xformed,
            parent_mask=parent_viab_mask,
            parent_gate=current_parent,
            config=cfg,
        )
        if viab_gate is not None and viab_cutoff is not None:
            hierarchy.add_gate(viab_gate)
            channels_used["viability"] = viab_gate.channels[0]
            cutoffs["viability_cutoff"] = viab_cutoff
            current_parent = "Viable"

    # Step 5: Optional Dump Channel Gating (Lineage Exclusion)
    if cfg.dump_enabled:
        interim_masks = hierarchy.evaluate(events, transformed_df=xformed)
        parent_dump_mask = interim_masks[current_parent]
        dump_gate, dump_cutoff = build_dump_gate(
            fcs_data,
            xformed,
            parent_mask=parent_dump_mask,
            parent_gate=current_parent,
            config=cfg,
        )
        if dump_gate is not None and dump_cutoff is not None:
            hierarchy.add_gate(dump_gate)
            channels_used["dump"] = dump_gate.channels[0]
            cutoffs["dump_cutoff"] = dump_cutoff
            current_parent = "Target"

    # Evaluate final hierarchy on appropriate dataframes
    final_masks = hierarchy.evaluate(events, transformed_df=xformed)
    final_clean_mask = final_masks[current_parent]
    clean_count = int(np.sum(final_clean_mask))
    clean_pct = float(clean_count / n_total * 100.0) if n_total > 0 else 0.0

    # Minimum event validation on singlets and viability
    n_singlets = int(np.sum(final_masks.get("Singlets_SSC", final_masks.get("Singlets_FSC", cells_mask))))
    if n_singlets < cfg.min_events_singlet:
        logger.warning(
            f"Singlet cell count ({n_singlets}) is below min_events_singlet ({cfg.min_events_singlet})."
        )
    if "Viable" in final_masks:
        n_viable = int(np.sum(final_masks["Viable"]))
        if n_viable < cfg.min_events_viability:
            logger.warning(
                f"Viable cell count ({n_viable}) is below min_events_viability ({cfg.min_events_viability})."
            )
    clean_count = int(np.sum(final_clean_mask))
    clean_pct = float(clean_count / n_total * 100.0) if n_total > 0 else 0.0

    # Build MIFlowCyt summary statistics
    stats_df = hierarchy.get_population_statistics(
        events,
        report_channels=[channels_used.get("debris_fsc", "FSC-A"), channels_used.get("debris_ssc", "SSC-A")],
        masks=final_masks,
    )

    elapsed = time.perf_counter() - t0
    sample_name = getattr(fcs_data, "sample_name", Path(getattr(fcs_data, "file_path", "sample.fcs")).name)

    if n_total >= 500 and clean_pct < 25.0:
        logger.warning(
            f"Low clean-up retention yield in '{sample_name}': {clean_count:,}/{n_total:,} events ({clean_pct:.1f}% < 25.0%). "
            "For fragile, necrotic, or high-debris tissues (e.g., thymocytes, solid tumors, cryopreserved PBMCs), "
            "inspect 03_cleanup_gating/plots/ and consider relaxing 'debris_sd_scale', 'singlet_tolerance', or 'viability_looseness'."
        )

    return CleanUpGatingResult(
        sample_name=sample_name,
        total_events=n_total,
        hierarchy=hierarchy,
        masks=final_masks,
        final_clean_mask=final_clean_mask,
        clean_event_count=clean_count,
        clean_event_pct=clean_pct,
        statistics_df=stats_df,
        cutoffs=cutoffs,
        channels_used=channels_used,
        config=cfg,
        execution_time_seconds=elapsed,
        cluster_info=cluster_info,
    )
