"""Flow cytometry gating module.

Provides polygon, rectangle, range, and quadrant gates, as well as a hierarchical
gating tree that manages parent-child event filtering and population statistics.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.path as mpath
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


class Gate(abc.ABC):
    """Abstract base class for gates.

    Attributes:
        name: Unique identifier for the gate.
        channels: List of channel names on which the gate operates.
        parent_gate: Name of the parent gate in the hierarchy, or None if root.
    """

    def __init__(
        self,
        name: str,
        channels: List[str],
        parent_gate: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize gate.

        Args:
            name: Gate identifier name.
            channels: Channel names used by this gate.
            parent_gate: Identifier of parent gate, or None.
            metadata: Optional dictionary of arbitrary metadata.
        """
        self.name = name
        self.channels = list(channels)
        self.parent_gate = parent_gate
        self.metadata = dict(metadata) if metadata else {}

    @abc.abstractmethod
    def apply(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Evaluate gate membership.

        Args:
            data: DataFrame containing columns for `self.channels` or 2D array.

        Returns:
            1D boolean NumPy array where True indicates event is inside the gate.
        """
        pass

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize gate definition to dictionary."""
        pass


class RectangleGate(Gate):
    """Rectangular gate defined by bounding box intervals.

    Attributes:
        x_min: Minimum x coordinate (inclusive). None means -inf.
        x_max: Maximum x coordinate (inclusive). None means +inf.
        y_min: Minimum y coordinate (inclusive). None means -inf.
        y_max: Maximum y coordinate (inclusive). None means +inf.
    """

    def __init__(
        self,
        name: str,
        channels: List[str],
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        parent_gate: Optional[str] = None,
    ) -> None:
        """Initialize rectangular gate.

        Args:
            name: Gate name.
            channels: [x_channel, y_channel] or [x_channel].
            x_min: Left boundary.
            x_max: Right boundary.
            y_min: Bottom boundary.
            y_max: Top boundary.
            parent_gate: Optional parent gate name.
        """
        super().__init__(name, channels, parent_gate)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def apply(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Evaluate events inside the rectangular region."""
        if isinstance(data, pd.DataFrame):
            x_vals = data[self.channels[0]].to_numpy()
            y_vals = data[self.channels[1]].to_numpy() if len(self.channels) > 1 else None
        else:
            x_vals = data[:, 0]
            y_vals = data[:, 1] if data.shape[1] > 1 else None

        mask = np.ones(len(x_vals), dtype=bool)

        if self.x_min is not None:
            mask &= x_vals >= self.x_min
        if self.x_max is not None:
            mask &= x_vals <= self.x_max

        if y_vals is not None:
            if self.y_min is not None:
                mask &= y_vals >= self.y_min
            if self.y_max is not None:
                mask &= y_vals <= self.y_max

        return mask

    def to_dict(self) -> Dict[str, Any]:
        """Serialize RectangleGate."""
        return {
            "type": "rectangle",
            "name": self.name,
            "channels": self.channels,
            "parent_gate": self.parent_gate,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RectangleGate:
        """Deserialize RectangleGate."""
        return cls(
            name=d["name"],
            channels=d["channels"],
            x_min=d.get("x_min"),
            x_max=d.get("x_max"),
            y_min=d.get("y_min"),
            y_max=d.get("y_max"),
            parent_gate=d.get("parent_gate"),
        )


class PolygonGate(Gate):
    """2D Polygonal gate defined by an ordered list of boundary vertices.

    Attributes:
        vertices: 2D array of coordinates with shape (N, 2).
    """

    def __init__(
        self,
        name: str,
        channels: List[str],
        vertices: Union[List[List[float]], np.ndarray],
        parent_gate: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize PolygonGate.

        Args:
            name: Gate identifier name.
            channels: Exactly two channel names: [x_channel, y_channel].
            vertices: Ordered list of 2D vertex coordinates [[x1, y1], [x2, y2], ...].
            parent_gate: Optional parent gate name.
            metadata: Optional dictionary of arbitrary metadata.

        Raises:
            ValueError: If fewer than 3 vertices or channels != 2.
        """
        if len(channels) != 2:
            raise ValueError(f"PolygonGate requires exactly 2 channels, got {channels}")

        vert_arr = np.asarray(vertices, dtype=np.float64)
        if vert_arr.ndim != 2 or vert_arr.shape[1] != 2 or len(vert_arr) < 3:
            raise ValueError(
                f"PolygonGate requires at least 3 (x, y) vertices, got shape {vert_arr.shape}"
            )

        super().__init__(name, channels, parent_gate, metadata=metadata)
        self.vertices = vert_arr
        self._path = mpath.Path(self.vertices)

    def apply(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Determine points located within the polygon boundary."""
        if isinstance(data, pd.DataFrame):
            pts = data[self.channels].to_numpy()
        else:
            pts = data[:, :2]

        return self._path.contains_points(pts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PolygonGate."""
        return {
            "type": "polygon",
            "name": self.name,
            "channels": self.channels,
            "parent_gate": self.parent_gate,
            "vertices": self.vertices.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PolygonGate:
        """Deserialize PolygonGate."""
        return cls(
            name=d["name"],
            channels=d["channels"],
            vertices=d["vertices"],
            parent_gate=d.get("parent_gate"),
        )


class RangeGate(Gate):
    """1D Interval or Threshold gate on a single channel.

    Attributes:
        min_val: Lower threshold bound.
        max_val: Upper threshold bound.
    """

    def __init__(
        self,
        name: str,
        channel: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        parent_gate: Optional[str] = None,
    ) -> None:
        """Initialize 1D RangeGate.

        Args:
            name: Gate name.
            channel: Single channel name.
            min_val: Optional lower bound.
            max_val: Optional upper bound.
            parent_gate: Optional parent gate.
        """
        super().__init__(name, [channel], parent_gate)
        self.min_val = min_val
        self.max_val = max_val

    @property
    def channel(self) -> str:
        """Primary channel name for this 1D range gate."""
        return self.channels[0]

    def apply(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Filter events within 1D interval."""
        if isinstance(data, pd.DataFrame):
            vals = data[self.channels[0]].to_numpy()
        else:
            vals = data[:, 0]

        mask = np.ones(len(vals), dtype=bool)
        if self.min_val is not None:
            mask &= vals >= self.min_val
        if self.max_val is not None:
            mask &= vals <= self.max_val
        return mask

    def to_dict(self) -> Dict[str, Any]:
        """Serialize RangeGate."""
        return {
            "type": "range",
            "name": self.name,
            "channels": self.channels,
            "parent_gate": self.parent_gate,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RangeGate:
        """Deserialize RangeGate."""
        return cls(
            name=d["name"],
            channel=d["channels"][0],
            min_val=d.get("min_val"),
            max_val=d.get("max_val"),
            parent_gate=d.get("parent_gate"),
        )


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize NumPy structures and arrays for standard JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {
            str(k): _sanitize_for_json(v)
            for k, v in obj.items()
            if not (isinstance(v, np.ndarray) and v.size > 100)
        }
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    return obj


class GMMClusterGate(PolygonGate):
    """Gate defining event membership via trained Gaussian Mixture Model clusters.

    Operates in normalized coordinate space, filtering out hardware margin saturation
    and baseline noise artifacts, and classifying events to Gaussian components.
    Inherits from PolygonGate to expose boundary polygon vertices for standard
    FCS visualization and plotting while evaluating exact clustering membership.

    Attributes:
        means: Gaussian component means in normalized space.
        covariances: Gaussian component covariances in normalized space.
        weights: Component mixture weights.
        cell_clusters: List of component indices corresponding to intact target cells.
        scale_diag: Scaling factors [scale_x, scale_y] for coordinate normalization.
        fsc_sat_thresh: Normalized threshold for forward scatter saturation.
        ssc_sat_thresh: Normalized threshold for side scatter saturation.
        fsc_noise_thresh: Normalized threshold for forward scatter zero/noise spikes.
        ssc_noise_thresh: Normalized threshold for side scatter zero/noise spikes.
        vertices: Representative polygon boundary coordinates for visualization.
    """

    def __init__(
        self,
        name: str,
        channels: List[str],
        means: Union[List[List[float]], np.ndarray],
        covariances: Union[List[List[List[float]]], np.ndarray],
        weights: Union[List[float], np.ndarray],
        cell_clusters: List[int],
        scale_diag: Union[List[float], np.ndarray],
        fsc_sat_thresh: float = 0.98,
        ssc_sat_thresh: float = 0.98,
        fsc_noise_thresh: float = 0.03,
        ssc_noise_thresh: float = 0.015,
        debris_sd_scale: float = 1.0,
        debris_min_probability: Optional[float] = None,
        vertices: Optional[Union[List[List[float]], np.ndarray]] = None,
        parent_gate: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        vert_arr = (
            np.asarray(vertices, dtype=np.float64)
            if vertices is not None and len(vertices) >= 3
            else np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        )
        super().__init__(name, channels, vert_arr, parent_gate=parent_gate, metadata=metadata)
        self.means = np.asarray(means, dtype=np.float64)
        self.covariances = np.asarray(covariances, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.cell_clusters = [int(c) for c in cell_clusters]
        self.scale_diag = np.asarray(scale_diag, dtype=np.float64)
        self.fsc_sat_thresh = float(fsc_sat_thresh)
        self.ssc_sat_thresh = float(ssc_sat_thresh)
        self.fsc_noise_thresh = float(fsc_noise_thresh)
        self.ssc_noise_thresh = float(ssc_noise_thresh)
        self.debris_sd_scale = float(debris_sd_scale)
        self.debris_min_probability = (
            float(debris_min_probability) if debris_min_probability is not None else None
        )

    def apply(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Evaluate GMM cluster membership for input events."""
        if isinstance(data, pd.DataFrame):
            fsc_vals = data[self.channels[0]].to_numpy(dtype=np.float64)
            ssc_vals = data[self.channels[1]].to_numpy(dtype=np.float64)
        else:
            fsc_vals = data[:, 0].astype(np.float64)
            ssc_vals = data[:, 1].astype(np.float64)

        norm_fsc = fsc_vals / self.scale_diag[0]
        norm_ssc = ssc_vals / self.scale_diag[1]

        is_sat = (norm_fsc >= self.fsc_sat_thresh) | (norm_ssc >= self.ssc_sat_thresh)
        is_noise = (norm_fsc <= self.fsc_noise_thresh) | (norm_ssc <= self.ssc_noise_thresh)

        X = np.column_stack([
            np.clip(norm_fsc, -0.05, 1.04),
            np.clip(norm_ssc, -0.05, 1.04),
        ])
        X_safe = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        n_comp = len(self.weights)
        n_pts = len(X_safe)
        log_probs = np.full((n_pts, n_comp), -1e12, dtype=np.float64)

        for k in range(n_comp):
            w_k = max(1e-12, float(self.weights[k]))
            try:
                log_probs[:, k] = np.log(w_k) + multivariate_normal.logpdf(
                    X_safe, mean=self.means[k], cov=self.covariances[k]
                )
            except Exception:
                pass

        pred = np.argmax(log_probs, axis=1)
        in_cell = np.isin(pred, self.cell_clusters)

        if self.debris_min_probability is not None:
            log_norm = logsumexp(log_probs, axis=1, keepdims=True)
            probs = np.exp(log_probs - log_norm)
            if self.cell_clusters:
                cell_prob = np.sum(probs[:, self.cell_clusters], axis=1)
            else:
                cell_prob = np.zeros(n_pts, dtype=np.float64)
            in_cell = cell_prob >= self.debris_min_probability
        elif abs(self.debris_sd_scale - 1.0) > 1e-6 and self.cell_clusters:
            min_d2 = np.full(n_pts, np.inf, dtype=np.float64)
            for k in self.cell_clusters:
                try:
                    diff = X_safe - self.means[k]
                    inv_cov = np.linalg.pinv(self.covariances[k])
                    d2_k = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
                    min_d2 = np.minimum(min_d2, d2_k)
                except Exception:
                    pass
            r_thresh2 = (2.5 * max(0.1, self.debris_sd_scale)) ** 2
            if self.debris_sd_scale > 1.0:
                in_cell = in_cell | (min_d2 <= r_thresh2)
            else:
                in_cell = in_cell & (min_d2 <= r_thresh2)

        return in_cell & (~is_sat) & (~is_noise)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize GMMClusterGate."""
        return {
            "type": "gmm_cluster",
            "name": self.name,
            "channels": self.channels,
            "parent_gate": self.parent_gate,
            "means": self.means.tolist(),
            "covariances": self.covariances.tolist(),
            "weights": self.weights.tolist(),
            "cell_clusters": list(self.cell_clusters),
            "scale_diag": self.scale_diag.tolist(),
            "fsc_sat_thresh": self.fsc_sat_thresh,
            "ssc_sat_thresh": self.ssc_sat_thresh,
            "fsc_noise_thresh": self.fsc_noise_thresh,
            "ssc_noise_thresh": self.ssc_noise_thresh,
            "debris_sd_scale": self.debris_sd_scale,
            "debris_min_probability": self.debris_min_probability,
            "vertices": self.vertices.tolist() if len(self.vertices) > 0 else [],
            "metadata": _sanitize_for_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GMMClusterGate:
        """Deserialize GMMClusterGate."""
        return cls(
            name=d["name"],
            channels=d["channels"],
            means=d["means"],
            covariances=d["covariances"],
            weights=d["weights"],
            cell_clusters=d["cell_clusters"],
            scale_diag=d["scale_diag"],
            fsc_sat_thresh=d.get("fsc_sat_thresh", 0.98),
            ssc_sat_thresh=d.get("ssc_sat_thresh", 0.98),
            fsc_noise_thresh=d.get("fsc_noise_thresh", 0.03),
            ssc_noise_thresh=d.get("ssc_noise_thresh", 0.015),
            debris_sd_scale=d.get("debris_sd_scale", 1.0),
            debris_min_probability=d.get("debris_min_probability"),
            vertices=d.get("vertices"),
            parent_gate=d.get("parent_gate"),
            metadata=d.get("metadata"),
        )


class QuadrantGate(Gate):
    """Quadrant gate dividing a 2D parameter space into 4 quadrants.

    Standard quadrant definitions:
        - Q1 (Upper Left): x <= x_cutoff and y > y_cutoff (X- Y+)
        - Q2 (Upper Right): x > x_cutoff and y > y_cutoff (X+ Y+, double positive)
        - Q3 (Lower Left): x <= x_cutoff and y <= y_cutoff (X- Y-, double negative)
        - Q4 (Lower Right): x > x_cutoff and y <= y_cutoff (X+ Y-)

    Attributes:
        x_cutoff: Threshold along x-axis.
        y_cutoff: Threshold along y-axis.
        quadrant: Target quadrant evaluated by apply() ("Q1", "Q2", "Q3", "Q4", or "all").
    """

    def __init__(
        self,
        name: str,
        channels: List[str],
        x_cutoff: float,
        y_cutoff: float,
        quadrant: str = "all",
        parent_gate: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(channels) != 2:
            raise ValueError(f"QuadrantGate requires exactly 2 channels, got {len(channels)}: {channels}")
        super().__init__(name=name, channels=channels, parent_gate=parent_gate, metadata=metadata)
        self.x_cutoff = float(x_cutoff)
        self.y_cutoff = float(y_cutoff)
        valid_quadrants = {"Q1", "Q2", "Q3", "Q4", "all"}
        if quadrant not in valid_quadrants:
            raise ValueError(f"Invalid quadrant '{quadrant}'. Must be one of {valid_quadrants}")
        self.quadrant = quadrant

    @property
    def x_channel(self) -> str:
        return self.channels[0]

    @property
    def y_channel(self) -> str:
        return self.channels[1]

    def _extract_xy(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(data, pd.DataFrame):
            x = data[self.x_channel].to_numpy()
            y = data[self.y_channel].to_numpy()
        else:
            arr = np.asarray(data)
            x = arr[:, 0]
            y = arr[:, 1]
        return x, y

    def get_quadrant_masks(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute Boolean masks for all 4 quadrants.

        Returns:
            Dictionary mapping 'Q1', 'Q2', 'Q3', 'Q4' to 1D boolean NumPy arrays.
        """
        x, y = self._extract_xy(data)
        x_pos = x > self.x_cutoff
        y_pos = y > self.y_cutoff

        return {
            "Q1": (~x_pos) & y_pos,   # UL: X- Y+
            "Q2": x_pos & y_pos,       # UR: X+ Y+ (double positive)
            "Q3": (~x_pos) & (~y_pos), # LL: X- Y- (double negative)
            "Q4": x_pos & (~y_pos),    # LR: X+ Y-
        }

    def get_quadrant_counts(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, int]:
        """Compute event counts in each quadrant."""
        masks = self.get_quadrant_masks(data)
        return {q: int(np.sum(m)) for q, m in masks.items()}

    def get_quadrant_frequencies(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, float]:
        """Compute percentage frequencies (0-100%) in each quadrant."""
        x, _ = self._extract_xy(data)
        total = len(x)
        if total == 0:
            return {q: 0.0 for q in ["Q1", "Q2", "Q3", "Q4"]}
        counts = self.get_quadrant_counts(data)
        return {q: round(c / total * 100.0, 3) for q, c in counts.items()}

    def apply(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Evaluate gate membership.

        If `self.quadrant == 'all'`, returns True for all valid events.
        Otherwise, returns boolean mask for the target quadrant.
        """
        masks = self.get_quadrant_masks(data)
        if self.quadrant == "all":
            x, _ = self._extract_xy(data)
            return np.ones(len(x), dtype=bool)
        return masks[self.quadrant]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize QuadrantGate definition to dictionary."""
        return {
            "type": "quadrant",
            "name": self.name,
            "channels": list(self.channels),
            "x_cutoff": self.x_cutoff,
            "y_cutoff": self.y_cutoff,
            "quadrant": self.quadrant,
            "parent_gate": self.parent_gate,
            "metadata": _sanitize_for_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> QuadrantGate:
        """Deserialize QuadrantGate."""
        return cls(
            name=d["name"],
            channels=d["channels"],
            x_cutoff=d["x_cutoff"],
            y_cutoff=d["y_cutoff"],
            quadrant=d.get("quadrant", "all"),
            parent_gate=d.get("parent_gate"),
            metadata=d.get("metadata"),
        )


def create_gate_from_dict(d: Dict[str, Any]) -> Gate:
    """Instantiate Gate subclass from dictionary."""
    gate_type = d.get("type", "").lower()
    if gate_type == "polygon":
        return PolygonGate.from_dict(d)
    elif gate_type == "rectangle":
        return RectangleGate.from_dict(d)
    elif gate_type == "range":
        return RangeGate.from_dict(d)
    elif gate_type == "gmm_cluster":
        return GMMClusterGate.from_dict(d)
    elif gate_type == "quadrant":
        return QuadrantGate.from_dict(d)
    else:
        raise ValueError(f"Unknown gate type: '{gate_type}'")


class GatingHierarchy:
    """Manages an ordered hierarchy/tree of gates and evaluates subpopulation masks."""

    def __init__(self) -> None:
        """Initialize empty hierarchy."""
        self.gates: Dict[str, Gate] = {}
        self.order: List[str] = []

    def add_gate(self, gate: Gate) -> None:
        """Add a gate to the hierarchy.

        Args:
            gate: Gate instance to add.

        Raises:
            ValueError: If gate name already exists or parent gate does not exist.
        """
        if gate.name in self.gates:
            raise ValueError(f"Gate named '{gate.name}' already exists in hierarchy")

        if gate.parent_gate is not None and gate.parent_gate not in self.gates:
            raise ValueError(
                f"Parent gate '{gate.parent_gate}' does not exist in hierarchy"
            )

        self.gates[gate.name] = gate
        self.order.append(gate.name)

    def get_gate(self, name: str) -> Gate:
        """Retrieve gate by name."""
        return self.gates[name]

    def get_children(self, parent_name: Optional[str] = None) -> List[Gate]:
        """Get all direct child gates of a given parent."""
        return [g for g in self.gates.values() if g.parent_gate == parent_name]

    def evaluate(
        self,
        df: pd.DataFrame,
        transformed_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Evaluate boolean event membership masks for all gates in the hierarchy.

        Each gate mask accounts for its parent's membership:
        gate_mask = parent_mask & gate.apply(df)

        Args:
            df: Event DataFrame containing linear scatter / raw channel columns.
            transformed_df: Optional DataFrame with transformed fluorescence channels.
                If provided, gates operating on fluorescent channels ('Viable', 'Target')
                evaluate on transformed_df.

        Returns:
            Dictionary mapping gate name to boolean mask array of length len(df).
        """
        n_events = len(df)
        masks: Dict[str, np.ndarray] = {
            "root": np.ones(n_events, dtype=bool),
        }

        for name in self.order:
            gate = self.gates[name]
            parent_key = gate.parent_gate if gate.parent_gate else "root"
            parent_mask = masks[parent_key]

            # Evaluate gate definition
            eval_source = transformed_df if (transformed_df is not None and name in ("Viable", "Target")) else df
            local_mask = gate.apply(eval_source)
            masks[name] = parent_mask & local_mask

        return masks

    def get_population_statistics(
        self,
        df: pd.DataFrame,
        report_channels: Optional[List[str]] = None,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Calculate event counts, percentages, and MFI for each gate.

        Args:
            df: Transformed/processed DataFrame.
            report_channels: Optional list of channels to report median fluorescence intensity for.
            masks: Optional pre-evaluated dictionary of boolean masks per gate.

        Returns:
            Summary pandas DataFrame.
        """
        eval_masks = masks if masks is not None else self.evaluate(df)
        total_events = len(df)
        rows: List[Dict[str, Any]] = []

        # Root statistics
        root_row: Dict[str, Any] = {
            "Gate": "root",
            "Parent": None,
            "Count": total_events,
            "% Parent": 100.0,
            "% Total": 100.0,
            "Parent_Pct": 100.0,
            "Total_Pct": 100.0,
        }
        if report_channels:
            for ch in report_channels:
                if ch in df.columns:
                    root_row[f"{ch} Median"] = float(df[ch].median())
        rows.append(root_row)

        for name in self.order:
            gate = self.gates[name]
            parent_name = gate.parent_gate if gate.parent_gate else "root"
            mask = eval_masks[name]
            parent_mask = eval_masks[parent_key if (parent_key := parent_name) else "root"]

            count = int(np.sum(mask))
            parent_count = int(np.sum(parent_mask))
            pct_parent = (count / parent_count * 100.0) if parent_count > 0 else 0.0
            pct_total = (count / total_events * 100.0) if total_events > 0 else 0.0

            row: Dict[str, Any] = {
                "Gate": name,
                "Parent": parent_name,
                "Count": count,
                "% Parent": round(pct_parent, 2),
                "% Total": round(pct_total, 2),
                "Parent_Pct": round(pct_parent, 2),
                "Total_Pct": round(pct_total, 2),
            }

            if report_channels and count > 0:
                sub_df = df.loc[mask]
                for ch in report_channels:
                    if ch in sub_df.columns:
                        row[f"{ch} Median"] = round(float(sub_df[ch].median()), 3)

            rows.append(row)

        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hierarchy to dictionary."""
        return {
            "order": self.order,
            "gates": [self.gates[name].to_dict() for name in self.order],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GatingHierarchy:
        """Deserialize hierarchy from dictionary."""
        hierarchy = cls()
        for g_dict in d.get("gates", []):
            gate = create_gate_from_dict(g_dict)
            hierarchy.add_gate(gate)
        return hierarchy
