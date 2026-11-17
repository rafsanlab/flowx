"""Adaptive density-based gating module inspired by openCyto and flowDensity.

Implements automated 1D kernel density estimation (KDE), peak detection, and
local density minimum (valley / saddle point) identification to dynamically
position gate thresholds between bimodal cell populations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from flowx.gating.gate import (
    GatingHierarchy,
    PolygonGate,
    QuadrantGate,
    RangeGate,
    RectangleGate,
)
from flowx.io.parser import FCSData


@dataclass
class QuadrantCutoffResult:
    """Result container for 2D quadrant cutoff detection.

    Attributes:
        x_channel: Name of horizontal marker/channel.
        y_channel: Name of vertical marker/channel.
        x_cutoff: Determined cutoff value along x-axis.
        y_cutoff: Determined cutoff value along y-axis.
        x_density_result: 1D DensityCutoffResult for x-axis.
        y_density_result: 1D DensityCutoffResult for y-axis.
        quadrant_counts: Event count in each quadrant {"Q1", "Q2", "Q3", "Q4"}.
        quadrant_frequencies: Percentage frequency in each quadrant {"Q1", "Q2", "Q3", "Q4"}.
        gate: Instantiated QuadrantGate object.
    """

    x_channel: str
    y_channel: str
    x_cutoff: float
    y_cutoff: float
    x_density_result: Optional[DensityCutoffResult] = None
    y_density_result: Optional[DensityCutoffResult] = None
    quadrant_counts: Dict[str, int] = field(default_factory=dict)
    quadrant_frequencies: Dict[str, float] = field(default_factory=dict)
    gate: Optional[QuadrantGate] = None


@dataclass
class DensityCutoffResult:
    """Result container for density-based threshold detection.

    Attributes:
        channel: Channel detector or marker name analyzed.
        cutoff: Selected threshold value (e.g. local density minimum).
        peaks: Estimated peak locations (modes) along the channel.
        valley: Local density minimum value between the two dominant peaks, or None.
        confidence: Ratio indicating depth of valley relative to adjacent peak heights.
        is_bimodal: Boolean indicating whether two distinct modes were detected.
        x_grid: Grid points where density was evaluated.
        density: Evaluated kernel density values along x_grid.
    """

    channel: str
    cutoff: float
    peaks: List[float] = field(default_factory=list)
    valley: Optional[float] = None
    confidence: float = 1.0
    is_bimodal: bool = False
    x_grid: np.ndarray = field(default_factory=lambda: np.empty(0))
    density: np.ndarray = field(default_factory=lambda: np.empty(0))


def find_density_minimum(
    values: Union[np.ndarray, pd.Series],
    channel_name: str = "Channel",
    gate_range: Optional[Tuple[float, float]] = None,
    default_cutoff: float = 0.35,
    bw_method: Union[str, float] = 0.08,
    grid_points: int = 500,
    min_peak_height_ratio: float = 0.02,
    min_prominence_ratio: float = 0.01,
    min_confidence: float = 0.15,
    min_peak_separation: float = 0.08,
    peak_distance: int = 20,
    subsample_max: int = 25000,
    random_state: int = 42,
    prefer_first_valley: bool = False,
) -> DensityCutoffResult:
    """Identify the local density minimum (valley) between two peaks using 1D KDE.

    Follows the algorithmic principle of openCyto's `gate_mindensity` and
    flowDensity's `deGate(..., method='mindensity')`:
    1. Computes smoothed 1D Gaussian Kernel Density Estimation (KDE).
    2. Identifies prominent local modes (peaks) using scipy signal processing
       with height, distance, and prominence filtering (eliminating noise ripples).
    3. Finds distinct separated peaks (distance >= min_peak_separation).
    4. If bimodal with a significant trough (confidence >= min_confidence),
       locates the local density minimum between the negative and positive peaks.
    5. Falls back to default_cutoff if the distribution is unimodal.

    Args:
        values: 1D array or Series of transformed marker values.
        channel_name: Name of channel for logging/diagnostics.
        gate_range: Optional (min, max) range bounding where the cutoff can be placed.
        default_cutoff: Fallback cutoff if bimodal separation is not resolved.
        bw_method: Bandwidth method or scalar for Gaussian KDE smoothing.
        grid_points: Number of evaluation points along the density grid.
        min_peak_height_ratio: Minimum relative peak height to consider as a valid mode.
        min_prominence_ratio: Minimum prominence ratio to filter out local noise ripples.
        min_confidence: Minimum valley depth ratio relative to adjacent peaks.
        min_peak_separation: Minimum numerical distance between candidate peak modes.
        peak_distance: Minimum index distance between neighboring peaks.
        subsample_max: Maximum points used for fast KDE evaluation.
        random_state: Seed for deterministic subsampling.

    Returns:
        DensityCutoffResult containing the calculated cutoff, peaks, and density curve.
    """
    clean_vals = np.asarray(values, dtype=np.float64)
    clean_vals = clean_vals[np.isfinite(clean_vals)]

    if len(clean_vals) < 50:
        return DensityCutoffResult(
            channel=channel_name,
            cutoff=default_cutoff,
            confidence=0.0,
            is_bimodal=False,
        )

    # Subsample if necessary for fast execution
    if len(clean_vals) > subsample_max:
        rng = np.random.RandomState(random_state)
        eval_vals = rng.choice(clean_vals, size=subsample_max, replace=False)
    else:
        eval_vals = clean_vals

    val_min = float(np.percentile(clean_vals, 0.5))
    val_max = float(np.percentile(clean_vals, 99.5))
    if val_max <= val_min:
        val_max = val_min + 1.0

    x_grid = np.linspace(val_min, val_max, grid_points)

    try:
        kde = gaussian_kde(eval_vals, bw_method=bw_method)
        density = kde(x_grid)
    except Exception:
        return DensityCutoffResult(
            channel=channel_name,
            cutoff=default_cutoff,
            confidence=0.0,
            is_bimodal=False,
        )

    max_dens = float(np.max(density)) if len(density) > 0 else 1.0
    min_height = max_dens * min_peak_height_ratio
    min_prominence = max_dens * min_prominence_ratio

    peak_indices, _ = find_peaks(
        density,
        height=min_height,
        distance=peak_distance,
        prominence=min_prominence,
    )
    peak_locations = [float(x_grid[idx]) for idx in peak_indices]

    # Evaluate bimodal separation across candidate peak pairs
    if len(peak_indices) >= 2:
        sorted_by_height = sorted(peak_indices, key=lambda idx: density[idx], reverse=True)
        best_candidate: Optional[Tuple[int, int, int, float, float]] = None  # (p1, p2, valley_idx, valley, conf)
        best_confidence: float = -1.0

        for i in range(len(sorted_by_height)):
            for j in range(i + 1, len(sorted_by_height)):
                idx_a, idx_b = sorted([sorted_by_height[i], sorted_by_height[j]])
                sep = abs(float(x_grid[idx_b]) - float(x_grid[idx_a]))
                if sep < min_peak_separation:
                    continue

                search_slice = density[idx_a : idx_b + 1]
                trough_rel_idx = int(np.argmin(search_slice))
                valley_idx = idx_a + trough_rel_idx
                candidate_valley = float(x_grid[valley_idx])

                min_peak_height = min(float(density[idx_a]), float(density[idx_b]))
                valley_height = float(density[valley_idx])
                cand_conf = float(max(0.0, 1.0 - (valley_height / (min_peak_height + 1e-9))))

                if cand_conf >= min_confidence:
                    # Prefer pair whose valley falls within or adjacent to gate_range if provided
                    in_range = True
                    if gate_range is not None:
                        g_min, g_max = gate_range
                        in_range = (g_min - 0.05 <= candidate_valley <= g_max + 0.05)

                    # Score candidate: prioritize valid range and higher confidence
                    effective_conf = cand_conf if in_range else (cand_conf * 0.5)
                    if prefer_first_valley and i == 0:
                        # Prioritize the valley immediately bounding the negative/live population
                        effective_conf *= 1.4
                    if effective_conf > best_confidence:
                        best_confidence = effective_conf
                        best_candidate = (idx_a, idx_b, valley_idx, candidate_valley, cand_conf)

        if best_candidate is not None:
            _, _, valley_idx, candidate_valley, confidence = best_candidate
            final_cutoff = candidate_valley
            if gate_range is not None:
                g_min, g_max = gate_range
                final_cutoff = float(np.clip(final_cutoff, g_min, g_max))

            return DensityCutoffResult(
                channel=channel_name,
                cutoff=round(final_cutoff, 3),
                peaks=[round(p, 3) for p in peak_locations],
                valley=round(candidate_valley, 3),
                confidence=round(confidence, 3),
                is_bimodal=True,
                x_grid=x_grid,
                density=density,
            )

    # Fallback for unimodal distribution or insufficient trough depth
    final_cutoff = default_cutoff
    if gate_range is not None:
        final_cutoff = float(np.clip(final_cutoff, gate_range[0], gate_range[1]))

    return DensityCutoffResult(
        channel=channel_name,
        cutoff=round(final_cutoff, 3),
        peaks=[round(p, 3) for p in peak_locations],
        valley=None,
        confidence=0.0,
        is_bimodal=False,
        x_grid=x_grid,
        density=density,
    )


def build_adaptive_density_hierarchy(
    fcs_data: FCSData,
    df_transformed: pd.DataFrame,
    default_hierarchy: Optional[GatingHierarchy] = None,
    bw_method: Union[str, float] = 0.08,
) -> Tuple[GatingHierarchy, Dict[str, DensityCutoffResult]]:
    """Build an adaptive gating hierarchy with data-driven density valleys.

    Adapts 1D thresholds (CD3, Viability, CD4, CD8) to match each sample's
    local density minima (openCyto/flowDensity approach), replacing rigid
    hardcoded boundaries with data-fitted troughs while preserving the
    hierarchical parent-child gating tree.

    Args:
        fcs_data: Parsed FCS sample data containing channel/marker mapping.
        df_transformed: Transformed event DataFrame.
        default_hierarchy: Optional reference hierarchy to inherit Cells and Singlets gates.
        bw_method: Bandwidth smoothing parameter for Gaussian KDE.

    Returns:
        Tuple of (adaptive_hierarchy, cutoffs_dict).
    """
    hierarchy = GatingHierarchy()
    cutoffs: Dict[str, DensityCutoffResult] = {}

    # Step 1: Cells Gate (FSC-A vs SSC-A)
    if default_hierarchy and "Cells" in default_hierarchy.order:
        cells_gate = default_hierarchy.get_gate("Cells")
    else:
        cells_gate = PolygonGate(
            name="Cells",
            channels=["FSC-A", "SSC-A"],
            vertices=[
                [60000.0, 15000.0],
                [180000.0, 25000.0],
                [220000.0, 120000.0],
                [140000.0, 160000.0],
                [60000.0, 60000.0],
            ],
            parent_gate=None,
        )
    hierarchy.add_gate(cells_gate)

    # Evaluate Cells to obtain parent mask
    cells_mask = cells_gate.apply(df_transformed)

    # Step 2: Singlets Gate (FSC-A vs FSC-H)
    if default_hierarchy and "Singlets" in default_hierarchy.order:
        singlets_gate = default_hierarchy.get_gate("Singlets")
    else:
        singlets_gate = PolygonGate(
            name="Singlets",
            channels=["FSC-A", "FSC-H"],
            vertices=[
                [60000.0, 45000.0],
                [175000.0, 135000.0],
                [175000.0, 165000.0],
                [60000.0, 62000.0],
            ],
            parent_gate="Cells",
        )
    hierarchy.add_gate(singlets_gate)

    singlets_mask = cells_mask & singlets_gate.apply(df_transformed)

    # Step 3: Viability Gate (AARD-A / Dead)
    dead_ch = fcs_data.resolve_channel_name("Dead") if "Dead" in fcs_data.marker_to_detector else "AARD-A"
    if dead_ch in fcs_data.pnn_labels and dead_ch in df_transformed.columns:
        dead_vals = df_transformed.loc[singlets_mask, dead_ch].dropna().to_numpy()
        dead_res = find_density_minimum(
            values=dead_vals,
            channel_name=dead_ch,
            gate_range=(0.35, 0.55),
            default_cutoff=0.40,
            bw_method=bw_method,
        )
        cutoffs["Viable"] = dead_res
        viable_cutoff = dead_res.cutoff

        viable_gate = RangeGate(
            name="Viable",
            channel=dead_ch,
            max_val=viable_cutoff,
            parent_gate="Singlets",
        )
        hierarchy.add_gate(viable_gate)
        viable_mask = singlets_mask & viable_gate.apply(df_transformed)
        parent_for_cd3 = "Viable"
    else:
        viable_mask = singlets_mask
        parent_for_cd3 = "Singlets"

    # Step 4: CD3+ T Cells Gate (Ax700-A / CD3)
    cd3_ch = fcs_data.resolve_channel_name("CD3") if "CD3" in fcs_data.marker_to_detector else "Ax700-A"
    if cd3_ch in fcs_data.pnn_labels and cd3_ch in df_transformed.columns:
        cd3_vals = df_transformed.loc[viable_mask, cd3_ch].dropna().to_numpy()
        cd3_res = find_density_minimum(
            values=cd3_vals,
            channel_name=cd3_ch,
            gate_range=(0.20, 0.35),
            default_cutoff=0.25,
            bw_method=bw_method,
        )
        cutoffs["CD3+"] = cd3_res
        cd3_cutoff = cd3_res.cutoff

        cd3_gate = RangeGate(
            name="CD3+",
            channel=cd3_ch,
            min_val=cd3_cutoff,
            parent_gate=parent_for_cd3,
        )
        hierarchy.add_gate(cd3_gate)
        cd3_mask = viable_mask & cd3_gate.apply(df_transformed)
        parent_for_sub = "CD3+"
    else:
        cd3_mask = viable_mask
        parent_for_sub = parent_for_cd3

    # Step 5: CD4+ and CD8+ T Cell Subsets (PE-TxRed-A vs PacBlue-A)
    cd4_ch = fcs_data.resolve_channel_name("CD4") if "CD4" in fcs_data.marker_to_detector else "PE-TxRed-A"
    cd8_ch = fcs_data.resolve_channel_name("CD8") if "CD8" in fcs_data.marker_to_detector else "PacBlue-A"

    if (
        cd4_ch in fcs_data.pnn_labels
        and cd8_ch in fcs_data.pnn_labels
        and cd4_ch in df_transformed.columns
        and cd8_ch in df_transformed.columns
    ):
        cd4_vals = df_transformed.loc[cd3_mask, cd4_ch].dropna().to_numpy()
        cd8_vals = df_transformed.loc[cd3_mask, cd8_ch].dropna().to_numpy()

        cd4_res = find_density_minimum(
            values=cd4_vals,
            channel_name=cd4_ch,
            gate_range=(0.25, 0.42),
            default_cutoff=0.34,
            bw_method=bw_method,
        )
        cd8_res = find_density_minimum(
            values=cd8_vals,
            channel_name=cd8_ch,
            gate_range=(0.25, 0.42),
            default_cutoff=0.35,
            bw_method=bw_method,
        )
        cutoffs["CD4"] = cd4_res
        cutoffs["CD8"] = cd8_res

        cd4_cutoff = cd4_res.cutoff
        cd8_cutoff = cd8_res.cutoff

        cd4_gate = RectangleGate(
            name="CD4+ T Cells",
            channels=[cd4_ch, cd8_ch],
            x_min=cd4_cutoff,
            y_max=cd8_cutoff,
            parent_gate=parent_for_sub,
        )
        hierarchy.add_gate(cd4_gate)

        cd8_gate = RectangleGate(
            name="CD8+ T Cells",
            channels=[cd4_ch, cd8_ch],
            x_max=cd4_cutoff,
            y_min=cd8_cutoff,
            parent_gate=parent_for_sub,
        )
        hierarchy.add_gate(cd8_gate)

    return hierarchy, cutoffs


def find_quadrant_cutoffs(
    data: pd.DataFrame,
    x_channel: str,
    y_channel: str,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    default_x_cutoff: float = 0.50,
    default_y_cutoff: float = 0.50,
    bw_method: Union[str, float] = 0.08,
    gate_name: str = "QuadrantGate",
    parent_gate: Optional[str] = None,
    refine_2d: bool = True,
    subsample_max: int = 25000,
    random_state: int = 42,
) -> QuadrantCutoffResult:
    """Identify optimal 2D quadrant cutoffs using marginal density minima and bivariate refinement.

    Follows the ISAC flow cytometry standard for 2D quadrant gating:
    1. Evaluates 1D kernel density estimation and valley detection along the x-axis.
    2. Evaluates 1D kernel density estimation and valley detection along the y-axis.
    3. If refine_2d is True, checks local bivariate KDE density around candidate intersection
       to ensure crosshairs avoid local density modes and sit in the true density saddle.
    4. Computes event counts and percentage frequencies for Q1, Q2, Q3, Q4.
    5. Returns an instantiated QuadrantGate ready for hierarchical execution.

    Args:
        data: DataFrame containing transformed event intensities.
        x_channel: Horizontal axis channel name.
        y_channel: Vertical axis channel name.
        x_range: Optional bounding search interval for x cutoff (min, max).
        y_range: Optional bounding search interval for y cutoff (min, max).
        default_x_cutoff: Fallback cutoff if x distribution is unimodal.
        default_y_cutoff: Fallback cutoff if y distribution is unimodal.
        bw_method: Bandwidth method for KDE smoothing.
        gate_name: Name of the resulting QuadrantGate.
        parent_gate: Name of parent population gate.
        refine_2d: Whether to perform local 2D saddle-point refinement.
        subsample_max: Maximum points for fast bivariate estimation.
        random_state: Seed for deterministic evaluation.

    Returns:
        QuadrantCutoffResult containing cutoffs, quadrant counts, frequencies, and QuadrantGate.
    """
    clean_df = data[[x_channel, y_channel]].dropna()
    if len(clean_df) > subsample_max:
        eval_df = clean_df.sample(n=subsample_max, random_state=random_state)
    else:
        eval_df = clean_df

    x_vals = eval_df[x_channel].to_numpy()
    y_vals = eval_df[y_channel].to_numpy()

    # 1. Marginal density minimum for X
    x_res = find_density_minimum(
        values=x_vals,
        channel_name=x_channel,
        gate_range=x_range,
        default_cutoff=default_x_cutoff,
        bw_method=bw_method,
        subsample_max=subsample_max,
        random_state=random_state,
    )
    x_cut = float(x_res.cutoff)

    # 2. Marginal density minimum for Y
    y_res = find_density_minimum(
        values=y_vals,
        channel_name=y_channel,
        gate_range=y_range,
        default_cutoff=default_y_cutoff,
        bw_method=bw_method,
        subsample_max=subsample_max,
        random_state=random_state,
    )
    y_cut = float(y_res.cutoff)

    # 3. 2D Bivariate Saddle-Point Refinement
    if refine_2d and len(eval_df) >= 100:
        try:
            dx = (x_range[1] - x_range[0]) * 0.15 if x_range else 0.10
            dy = (y_range[1] - y_range[0]) * 0.15 if y_range else 0.10
            grid_x = np.linspace(x_cut - dx, x_cut + dx, 15)
            grid_y = np.linspace(y_cut - dy, y_cut + dy, 15)
            gx, gy = np.meshgrid(grid_x, grid_y)
            grid_pts = np.vstack([gx.ravel(), gy.ravel()])

            kde_2d = gaussian_kde(np.vstack([x_vals, y_vals]), bw_method=bw_method)
            dens = kde_2d(grid_pts)
            min_idx = int(np.argmin(dens))
            # Shift towards 2D minimum if within plausible bounds
            refined_x = float(grid_pts[0, min_idx])
            refined_y = float(grid_pts[1, min_idx])

            # Apply gentle convex blend to prevent boundary overshoot
            x_cut = 0.7 * x_cut + 0.3 * refined_x
            y_cut = 0.7 * y_cut + 0.3 * refined_y
        except Exception:
            pass  # Keep marginal cutoffs on singular matrix

    # 4. Create QuadrantGate and compute population stats
    quad_gate = QuadrantGate(
        name=gate_name,
        channels=[x_channel, y_channel],
        x_cutoff=x_cut,
        y_cutoff=y_cut,
        quadrant="all",
        parent_gate=parent_gate,
    )
    counts = quad_gate.get_quadrant_counts(clean_df)
    freqs = quad_gate.get_quadrant_frequencies(clean_df)

    return QuadrantCutoffResult(
        x_channel=x_channel,
        y_channel=y_channel,
        x_cutoff=x_cut,
        y_cutoff=y_cut,
        x_density_result=x_res,
        y_density_result=y_res,
        quadrant_counts=counts,
        quadrant_frequencies=freqs,
        gate=quad_gate,
    )

