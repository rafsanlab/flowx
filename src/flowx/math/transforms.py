"""Flow cytometry data transformations module.

Provides standard Gating-ML 2.0 compliant Logicle, Hyperbolic Arcsine (asinh),
Biexponential, and Logarithmic scale transformations with forward and inverse operations.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import flowutils.transforms as fu_transforms
import numpy as np
import pandas as pd


class BaseTransform(abc.ABC):
    """Abstract base class for channel value transformations."""

    @abc.abstractmethod
    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply the forward transformation to event values.

        Args:
            values: 1D or 2D array of raw or compensated event values.

        Returns:
            Transformed values array of identical shape.
        """
        pass

    @abc.abstractmethod
    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Apply the inverse transformation to return to unscaled space.

        Args:
            values: 1D or 2D array of transformed values.

        Returns:
            Inverse transformed values array of identical shape.
        """
        pass

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize transformation parameters to dictionary."""
        pass


class LogicleTransform(BaseTransform):
    """Logicle transformation conforming to Gating-ML 2.0 specification.

    Logicle smoothly transitions from a linear scale around zero (to handle negative
    fluorescence fluctuations from compensation) to a logarithmic scale at high intensities.

    Parameters:
        param_t: Top of scale value (maximum expected data value, e.g. 262144.0).
        param_w: Number of decades in the linear region (typically 0.5 - 1.0).
        param_m: Number of decades the logarithmic scale spans (typically 4.5).
        param_a: Additional decades of negative data to display (typically 0.0).
    """

    def __init__(
        self,
        param_t: float = 262144.0,
        param_w: float = 0.5,
        param_m: float = 4.5,
        param_a: float = 0.0,
    ) -> None:
        """Initialize Logicle transform parameters."""
        self.param_t = float(param_t)
        self.param_w = float(param_w)
        self.param_m = float(param_m)
        self.param_a = float(param_a)

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply forward Logicle transformation."""
        arr = np.asarray(values, dtype=np.float64)
        flat = arr.ravel()
        transformed = fu_transforms.logicle(
            flat,
            None,
            self.param_t,
            self.param_m,
            self.param_w,
            self.param_a,
        )
        return transformed.reshape(arr.shape)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Apply inverse Logicle transformation."""
        arr = np.asarray(values, dtype=np.float64)
        flat = arr.ravel()
        inv_vals = fu_transforms.logicle_inverse(
            flat,
            None,
            self.param_t,
            self.param_m,
            self.param_w,
            self.param_a,
        )
        return inv_vals.reshape(arr.shape)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters."""
        return {
            "type": "logicle",
            "param_t": self.param_t,
            "param_w": self.param_w,
            "param_m": self.param_m,
            "param_a": self.param_a,
        }

    def __repr__(self) -> str:
        return (
            f"LogicleTransform(t={self.param_t}, w={self.param_w}, "
            f"m={self.param_m}, a={self.param_a})"
        )


class BiexponentialTransform(BaseTransform):
    """FlowJo-compatible Biexponential transformation.

    Generates the exact look-up table (LUT) used by TreeStar / FlowJo 10 and BD FACSDiva
    for biexponential scaling, smoothly displaying negative compensated values while
    providing logarithmic expansion for positive signals.

    Parameters:
        max_value: Top of data range (e.g. 262144.0 for BD 18-bit instruments).
        positive: Number of display decades (typically 4.418540, standard FlowJo default).
        width_basis: Controls the compression of values into linear zero space (typically -10.0).
        negative: Number of additional negative decades (typically 0.0).
        channel_range: Output display range / number of LUT divisions (default: 4096).
        normalized: If True (default), scales output to [0.0, 1.0] for direct compatibility
            with standard Gating-ML 2.0 downstream gates, density estimators, and clustering algorithms.
            If False, scales to [0.0, channel_range] (0 to 4096) matching raw FlowJo display bins.
    """

    def __init__(
        self,
        max_value: float = 262144.0,
        positive: float = 4.418540,
        width_basis: float = -10.0,
        negative: float = 0.0,
        channel_range: int = 4096,
        normalized: bool = True,
    ) -> None:
        """Initialize FlowJo Biexponential transform parameters."""
        self.max_value = float(max_value)
        self.positive = float(positive)
        self.width_basis = float(width_basis)
        self.negative = float(negative)
        self.channel_range = int(channel_range)
        self.normalized = bool(normalized)

        # Build lookup table and interpolation splines
        x_lut, y_lut = self._generate_lut(
            channel_range=self.channel_range,
            pos=self.positive,
            neg=self.negative,
            width_basis=self.width_basis,
            max_value=self.max_value,
        )

        from scipy import interpolate

        self._lut_func = interpolate.interp1d(
            x_lut, y_lut, kind="linear", bounds_error=False, fill_value=(float(np.min(y_lut)), float(np.max(y_lut)))
        )
        self._inverse_lut_func = interpolate.interp1d(
            y_lut, x_lut, kind="linear", bounds_error=False, fill_value=(float(np.min(x_lut)), float(np.max(x_lut)))
        )

    @property
    def param_t(self) -> float:
        """Top of scale alias for consistency with LogicleTransform."""
        return self.max_value

    @property
    def param_m(self) -> float:
        """Decades alias for consistency with LogicleTransform."""
        return self.positive

    @property
    def param_w(self) -> float:
        """Width alias for consistency with LogicleTransform."""
        return self.width_basis

    @property
    def param_a(self) -> float:
        """Negative decades alias for consistency with LogicleTransform."""
        return self.negative

    @staticmethod
    def _log_root(b: float, w: float) -> float:
        """Find root for biexponential negative decade scaling."""
        x_lo = 0.0
        x_hi = float(b)
        d = (x_lo + x_hi) / 2.0
        dx = abs(int(x_lo - x_hi))
        dx_last = dx
        fb = -2.0 * np.log(b) + w * b
        f = 2.0 * np.log(d) + w * b + fb
        df = 2.0 / d + w

        if w == 0:
            return float(b)

        for _ in range(100):
            if (((d - x_hi) * df - f) - ((d - x_lo) * df - f)) > 0 or abs(2.0 * f) > abs(dx_last * df):
                dx = (x_hi - x_lo) / 2.0
                d = x_lo + dx
                if d == x_lo:
                    return float(d)
            else:
                dx = f / df
                t = d
                d -= dx
                if d == t:
                    return float(d)

            if abs(dx) < 1.0e-12:
                return float(d)

            dx_last = dx
            f = 2.0 * np.log(d) + w * d + fb
            df = 2.0 / d + w
            if f < 0:
                x_lo = d
            else:
                x_hi = d

        return float(d)

    @classmethod
    def _generate_lut(
        cls,
        channel_range: int = 4096,
        pos: float = 4.418540,
        neg: float = 0.0,
        width_basis: float = -10.0,
        max_value: float = 262144.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a FlowJo-compatible biex lookup table (LUT)."""
        ln10 = np.log(10.0)
        decades = float(pos)
        low_scale = float(width_basis)
        width = np.log10(-low_scale)

        decades = decades - (width / 2.0)
        extra = max(0.0, float(neg)) + (width / 2.0)

        zero_point = int((extra * channel_range) / (extra + decades))
        zero_point = int(np.min([zero_point, channel_range / 2]))

        if zero_point > 0:
            decades = extra * channel_range / zero_point

        width = width / (2.0 * decades)

        maximum = float(max_value)
        positive_range = ln10 * decades
        minimum = maximum / np.exp(positive_range)

        negative_range = cls._log_root(positive_range, width)

        max_channel_value = channel_range + 1
        n_points = max_channel_value

        step = (max_channel_value - 1) / (n_points - 1)
        values = np.arange(n_points, dtype=np.float64)
        positive = np.exp(values / float(n_points) * positive_range)
        negative = np.exp(values / float(n_points) * -negative_range)

        values = values * step
        s = np.exp((positive_range + negative_range) * (width + extra / decades))
        negative *= s
        s = positive[zero_point] - negative[zero_point]

        positive[zero_point:n_points] = positive[zero_point:n_points] - negative[zero_point:n_points]
        positive[zero_point:n_points] = minimum * (positive[zero_point:n_points] - s)

        neg_range = np.arange(zero_point)
        m = 2 * zero_point - neg_range
        positive[neg_range] = -positive[m]

        return positive, values

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply forward FlowJo biexponential transformation."""
        arr = np.asarray(values, dtype=np.float64)
        flat = arr.ravel()
        out = self._lut_func(flat)
        if self.normalized:
            out = out / float(self.channel_range)
        return out.reshape(arr.shape)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Apply inverse FlowJo biexponential transformation."""
        arr = np.asarray(values, dtype=np.float64)
        flat = arr.ravel()
        if self.normalized:
            flat = flat * float(self.channel_range)
        out = self._inverse_lut_func(flat)
        return out.reshape(arr.shape)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transformation parameters to dictionary."""
        return {
            "type": "biexponential",
            "max_value": self.max_value,
            "positive": self.positive,
            "width_basis": self.width_basis,
            "negative": self.negative,
            "channel_range": self.channel_range,
            "normalized": self.normalized,
        }

    def __repr__(self) -> str:
        return (
            f"BiexponentialTransform(max_value={self.max_value}, pos={self.positive}, "
            f"width={self.width_basis}, neg={self.negative}, norm={self.normalized})"
        )


class AsinhTransform(BaseTransform):
    """Hyperbolic arcsine (arcsinh) transformation.

    Formula: f(x) = asinh(x / cofactor)
    Inverse: f^-1(y) = cofactor * sinh(y)

    Parameters:
        cofactor: Scaling denominator (typically 150 for flow cytometry, 5 for CyTOF).
        max_value: Top of scale value used when normalized=True.
        normalized: If True, scales transformed values onto [0, 1] for compatibility
            with standardized downstream gating and clustering thresholds.
    """

    def __init__(
        self,
        cofactor: float = 150.0,
        max_value: float = 262144.0,
        normalized: bool = False,
    ) -> None:
        """Initialize Asinh transform.

        Args:
            cofactor: Scale factor. Must be greater than 0.
            max_value: Maximum detector range when normalized=True.
            normalized: Whether to normalize output to [0, 1].
        """
        if cofactor <= 0:
            raise ValueError(f"Cofactor must be positive, got {cofactor}")
        self.cofactor = float(cofactor)
        self.max_value = float(max(max_value, cofactor * 10.0))
        self.normalized = bool(normalized)
        self._low_asinh = float(np.arcsinh(-5.0))
        self._high_asinh = float(np.arcsinh(self.max_value / self.cofactor))

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply forward arcsinh transformation."""
        arr = np.asarray(values, dtype=np.float64)
        out = np.arcsinh(arr / self.cofactor)
        if self.normalized:
            span = max(self._high_asinh - self._low_asinh, 1e-6)
            out = np.clip((out - self._low_asinh) / span, 0.0, 1.0)
        return out

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Apply inverse arcsinh transformation."""
        arr = np.asarray(values, dtype=np.float64)
        if self.normalized:
            span = max(self._high_asinh - self._low_asinh, 1e-6)
            arr = arr * span + self._low_asinh
        return self.cofactor * np.sinh(arr)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters."""
        return {
            "type": "asinh",
            "cofactor": self.cofactor,
            "max_value": self.max_value,
            "normalized": self.normalized,
        }

    def __repr__(self) -> str:
        return f"AsinhTransform(cofactor={self.cofactor}, normalized={self.normalized})"


class PreTransformedScaleTransform(BaseTransform):
    """Linearly maps already-transformed (e.g. pre-arcsinh or pre-logicle) FCS data onto [0, 1].

    Prevents double-transformation (such as applying Logicle(T=262144) to values in [-1, 5])
    while placing negative baseline populations around ~0.18 and positive populations in [0.35, 0.98].
    """

    def __init__(self, low_val: float = -1.0, high_val: float = 4.5) -> None:
        self.low_val = float(low_val)
        self.high_val = float(max(high_val, low_val + 1.0))

    def apply(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        span = max(self.high_val - self.low_val, 1e-6)
        return np.clip((arr - self.low_val) / span, 0.0, 1.0)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        span = max(self.high_val - self.low_val, 1e-6)
        return arr * span + self.low_val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "pre_transformed_scale",
            "low_val": self.low_val,
            "high_val": self.high_val,
        }

    def __repr__(self) -> str:
        return f"PreTransformedScaleTransform(low_val={self.low_val}, high_val={self.high_val})"


def is_empirically_pre_transformed(
    fcs_data: Any,
    df: Optional[pd.DataFrame] = None,
    channels: Optional[List[str]] = None,
) -> bool:
    """Detect whether an FCS file already contains logicle/arcsinh-transformed event values.

    Some benchmark datasets (e.g. CytoNorm pre-processed cohorts) export already-transformed
    values in [-2, +10] while retaining raw cytometer $PnR headers (e.g. 262144). Applying
    compensation or Logicle(T=262144) to such data collapses the dynamic range.
    """
    events_df = df if df is not None else getattr(fcs_data, "events_df", None)
    if events_df is None or events_df.empty:
        return False

    max_range = float(fcs_data.detect_max_range()) if hasattr(fcs_data, "detect_max_range") else 262144.0
    if max_range < 500.0:
        return False

    _non_fl_tokens = ("TIME", "FSC", "SSC", "ORIGINAL_ID", "EVENT_ID", "SAMPLE_ID", "INDEX", "CLUSTER", "UMAP", "TSNE", "WIDTH")
    check_cols = [
        c for c in (channels or getattr(fcs_data, "fluoro_channels", []) or list(events_df.columns))
        if c in events_df.columns and not any(k in c.upper() for k in _non_fl_tokens)
    ]
    if not check_cols:
        return False

    sub = events_df[check_cols].to_numpy(dtype=np.float64)
    finite = sub[np.isfinite(sub)]
    if len(finite) == 0:
        return False

    max_abs = float(np.nanpercentile(np.abs(finite), 99.9))
    val_span = float(np.nanmax(finite) - np.nanmin(finite))
    return bool(0.1 < max_abs <= 25.0 and val_span <= 30.0)


class LogTransform(BaseTransform):
    """Logarithmic transformation with threshold floor for negative values.

    Parameters:
        floor: Minimal value threshold before taking logarithm.
        base: Logarithmic base (default: 10.0).
    """

    def __init__(self, floor: float = 1.0, base: float = 10.0) -> None:
        """Initialize log transform."""
        if floor <= 0:
            raise ValueError("Floor must be positive")
        self.floor = float(floor)
        self.base = float(base)
        self._log_base = np.log(self.base)

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply forward log transformation."""
        arr = np.asarray(values, dtype=np.float64)
        clipped = np.clip(arr, a_min=self.floor, a_max=None)
        return np.log(clipped) / self._log_base

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Apply inverse log transformation."""
        arr = np.asarray(values, dtype=np.float64)
        return np.power(self.base, arr)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters."""
        return {
            "type": "log",
            "floor": self.floor,
            "base": self.base,
        }

    def __repr__(self) -> str:
        return f"LogTransform(floor={self.floor}, base={self.base})"


def create_transform(config: Dict[str, Any]) -> BaseTransform:
    """Factory function to instantiate a transform from a configuration dictionary.

    Args:
        config: Dictionary containing 'type' and corresponding parameters.

    Returns:
        Instance of BaseTransform.

    Raises:
        ValueError: If transform type is unknown.
    """
    xform_type = config.get("type", "").lower()
    if xform_type == "logicle":
        return LogicleTransform(
            param_t=config.get("param_t", 262144.0),
            param_w=config.get("param_w", 0.5),
            param_m=config.get("param_m", 4.5),
            param_a=config.get("param_a", 0.0),
        )
    elif xform_type in ("biex", "biexponential", "wsp_biex"):
        norm_val = config.get("normalized", True)
        if isinstance(norm_val, str):
            norm_val = norm_val.lower() in ("true", "1", "yes")
        return BiexponentialTransform(
            max_value=float(config.get("max_value", config.get("param_t", 262144.0))),
            positive=float(config.get("positive", config.get("param_m", 4.418540))),
            width_basis=float(config.get("width_basis", config.get("width", -10.0))),
            negative=float(config.get("negative", config.get("neg", 0.0))),
            channel_range=int(config.get("channel_range", 4096)),
            normalized=bool(norm_val),
        )
    elif xform_type in ("asinh", "arcsinh"):
        return AsinhTransform(
            cofactor=float(config.get("cofactor", 150.0)),
            max_value=float(config.get("max_value", 262144.0)),
            normalized=bool(config.get("normalized", False)),
        )
    elif xform_type == "pre_transformed_scale":
        return PreTransformedScaleTransform(
            low_val=float(config.get("low_val", -1.0)),
            high_val=float(config.get("high_val", 4.5)),
        )
    elif xform_type == "log":
        return LogTransform(
            floor=config.get("floor", 1.0),
            base=config.get("base", 10.0),
        )
    else:
        raise ValueError(f"Unknown transform type: '{xform_type}'")


def apply_transform_to_dataframe(
    df: pd.DataFrame,
    channels: List[str],
    transform: Union[BaseTransform, Dict[str, BaseTransform]],
) -> pd.DataFrame:
    """Apply transformation to designated channels in a DataFrame.

    Non-designated channels remain unmodified in their original (e.g. linear) scale.

    Args:
        df: Input DataFrame with flow cytometry events.
        channels: List of channel names to transform.
        transform: BaseTransform instance or dict mapping channel names to BaseTransform.

    Returns:
        New DataFrame with transformed values in specified columns.
    """
    result_df = df.copy()
    for ch in channels:
        if ch in result_df.columns:
            if isinstance(transform, dict):
                xform = transform.get(ch) or transform.get("default")
            else:
                xform = transform

            if xform is not None:
                vals = result_df[ch].to_numpy(dtype=np.float64)
                result_df[ch] = xform.apply(vals)
    return result_df


def get_optimal_transform(
    fcs_data: Any,
    transform_type: str = "logicle",
    param_w: float = 0.5,
    param_a: float = 0.0,
    width_basis: float = -10.0,
    negative: float = 0.0,
    normalized: bool = True,
    cofactor: Optional[float] = None,
) -> BaseTransform:
    """Auto-detect optimal transform parameters based on cytometer dynamic range.

    Adapts the upper scale limit T and decades M based on the instrument's hardware
    range (e.g. T=262,144 for BD 18-bit, T=1,000,000 for Beckman/Cytek 20-bit,
    T=16,777,216 for 24-bit).

    Args:
        fcs_data: Parsed FCSData container or object with detect_max_range().
        transform_type: Type of transformation ('logicle', 'biexponential', 'asinh', 'log').
        param_w: Linearization width parameter (for Logicle).
        param_a: Negative decades parameter (for Logicle).
        width_basis: Width basis parameter (for Biexponential, typically -10.0).
        negative: Negative decades parameter (for Biexponential, typically 0.0).
        normalized: Whether Biexponential/Asinh outputs are scaled to [0, 1].
        cofactor: Optional arcsinh cofactor override.

    Returns:
        Configured BaseTransform tailored to the instrument and selected mode.
    """
    if is_empirically_pre_transformed(fcs_data):
        events_df = getattr(fcs_data, "events_df", None)
        low_v, high_v = -1.0, 4.5
        if events_df is not None and not events_df.empty:
            _non_fl_tokens = ("TIME", "FSC", "SSC", "ORIGINAL_ID", "EVENT_ID", "SAMPLE_ID", "INDEX", "CLUSTER", "UMAP", "TSNE", "WIDTH")
            fl_cols = [
                c for c in (getattr(fcs_data, "fluoro_channels", []) or list(events_df.columns))
                if c in events_df.columns and not any(k in c.upper() for k in _non_fl_tokens)
            ]
            if fl_cols:
                vals = events_df[fl_cols].to_numpy(dtype=np.float64)
                finite = vals[np.isfinite(vals)]
                if len(finite) > 0:
                    low_v = float(min(-0.8, np.nanpercentile(finite, 0.2) - 0.1))
                    high_v = float(max(4.2, np.nanpercentile(finite, 99.9) + 0.15))
        return PreTransformedScaleTransform(low_val=low_v, high_val=high_v)

    if hasattr(fcs_data, "detect_max_range"):
        max_range = float(fcs_data.detect_max_range())
    else:
        max_range = 262144.0

    mode = (transform_type or "logicle").lower()

    if mode in ("biex", "biexponential", "wsp_biex"):
        if max_range > 5000000.0:
            pos = 5.5
        elif max_range > 500000.0:
            pos = 5.0
        elif max_range < 5000.0:
            pos = 3.5
        else:
            pos = 4.418540

        return BiexponentialTransform(
            max_value=max_range,
            positive=pos,
            width_basis=width_basis,
            negative=negative,
            channel_range=4096,
            normalized=normalized,
        )
    elif mode in ("asinh", "arcsinh"):
        eff_cofactor = float(cofactor) if (cofactor is not None and cofactor > 0) else 150.0
        # Prevent CyTOF cofactor (e.g. 5) from splitting negative populations on fluorescence cytometers
        if max_range >= 100000.0 and eff_cofactor < 100.0:
            eff_cofactor = 150.0 if max_range <= 1000000.0 else 400.0
        return AsinhTransform(
            cofactor=eff_cofactor,
            max_value=max_range,
            normalized=normalized,
        )
    elif mode == "log":
        return LogTransform(floor=1.0, base=10.0)
    else:
        # Default to standard Gating-ML 2.0 Logicle
        if max_range > 5000000.0:
            param_m = 5.5
        elif max_range > 500000.0:
            param_m = 5.0
        elif max_range < 5000.0:
            param_m = 3.5
        else:
            param_m = 4.5

        return LogicleTransform(
            param_t=max_range,
            param_w=param_w,
            param_m=param_m,
            param_a=param_a,
        )


@dataclass
class TransformConfig:
    """Configuration container for Step 3 / Step 4 scale transformation.

    Attributes:
        enabled: Whether scale transformation is active (default: True).
        method: Transformation scale ('logicle', 'biexponential', 'asinh', 'log').
        cofactor: Cofactor for arcsinh transformation (default: 5.0; auto-scaled for high-bit instruments).
        channels: 'auto', None (auto-resolve via markers.csv), or explicit list of channels to transform.
        exclude_channels: Optional list of detector/marker names to exclude when resolving 'auto' channels.
        param_w: Linearization width parameter for Logicle (default: 0.5).
        param_a: Negative decades parameter for Logicle (default: 0.0).
        normalized: Whether Biexponential/Asinh outputs are scaled to [0, 1] (default: True).
    """

    enabled: bool = True
    method: str = "logicle"
    cofactor: float = 5.0
    channels: Optional[Union[str, List[str]]] = None
    exclude_channels: Optional[List[str]] = None
    param_w: float = 0.5
    param_a: float = 0.0
    normalized: bool = True


@dataclass
class TransformStepResult:
    """Container holding results from an independent scale transformation step."""

    transformed_df: pd.DataFrame
    transform: Optional[BaseTransform] = None
    channels_transformed: List[str] = field(default_factory=list)
    plot_path: Optional[str] = None


def resolve_fluorescence_channels(
    fcs_data: Any,
    explicit_channels: Optional[Union[str, List[str]]] = None,
    exclude_channels: Optional[List[str]] = None,
    df: Optional[pd.DataFrame] = None,
    marker_metadata: Optional[Any] = None,
) -> List[str]:
    """Resolve target fluorescence channels for compensation or scale transformation.

    Priority:
    1. Explicit channel list (when non-empty and not `'auto'`).
    2. Marker metadata (`markers.csv`) excluding `scatter` and `time` roles.
    3. FCS `spillover_channels` if available.
    4. All FCS `$PnN` labels excluding scatter, time, and auxiliary metadata columns.
    Applies optional `exclude_channels` (matching physical detector or marker symbol).
    """
    available_cols = set(df.columns) if df is not None else set(getattr(fcs_data, "pnn_labels", []) or [])

    excl_set: set[str] = set()
    if exclude_channels:
        m2d = getattr(marker_metadata, "marker_to_detector", {}) or {}
        d2m = getattr(marker_metadata, "detector_to_marker", {}) or {}
        for item in exclude_channels:
            tok = str(item).strip()
            if not tok:
                continue
            excl_set.add(tok)
            if tok in m2d:
                excl_set.add(m2d[tok])
            tok_low = tok.lower()
            for det, mk in d2m.items():
                if det.lower() == tok_low or str(mk).lower() == tok_low:
                    excl_set.add(det)

    def _apply_exclusions(chs: List[str]) -> List[str]:
        return [c for c in chs if c not in excl_set] if excl_set else chs

    is_auto = (
        explicit_channels is None
        or (isinstance(explicit_channels, str) and explicit_channels.strip().lower() == "auto")
        or (isinstance(explicit_channels, list) and len(explicit_channels) == 1 and str(explicit_channels[0]).strip().lower() == "auto")
    )

    if not is_auto and isinstance(explicit_channels, list) and explicit_channels:
        resolved = [c for c in explicit_channels if not available_cols or c in available_cols]
        if resolved:
            return _apply_exclusions(resolved)

    if marker_metadata is not None and getattr(marker_metadata, "marker_classes", None):
        from flowx.io.metadata import NON_FLUORESCENCE_MARKER_CLASSES

        meta_channels = [
            ch
            for ch, roles in marker_metadata.marker_classes.items()
            if not (set(roles) & NON_FLUORESCENCE_MARKER_CLASSES) and (not available_cols or ch in available_cols)
        ]
        if meta_channels:
            return _apply_exclusions(meta_channels)

    spill_chs = list(getattr(fcs_data, "spillover_channels", []) or [])
    if spill_chs:
        filtered_spill = [c for c in spill_chs if not available_cols or c in available_cols]
        if filtered_spill:
            return _apply_exclusions(filtered_spill)

    exclude = {
        "fsc-a", "fsc-h", "fsc-w", "fsc-width",
        "ssc-a", "ssc-h", "ssc-w", "ssc-width",
        "time", "width", "original_id", "event_id",
        "sample_id", "index", "cluster_id", "lineage_id",
        "umap_1", "umap_2", "fs int", "ss int", "ss peak", "ss tof",
    }
    pnn_labels = list(getattr(fcs_data, "pnn_labels", []) or list(available_cols))
    return _apply_exclusions([
        ch
        for ch in pnn_labels
        if ch.lower() not in exclude
        and not any(ch.lower().startswith(p) for p in ["fs ", "fs.", "ss ", "ss.", "time", "original_id", "event_id"])
    ])


def run_transformation_step(
    fcs_data: Any,
    df_events: pd.DataFrame,
    config: Optional[TransformConfig] = None,
    target_channels: Optional[List[str]] = None,
    custom_transform: Optional[BaseTransform] = None,
    marker_metadata: Optional[Any] = None,
    output_plot_path: Optional[Union[str, Path]] = None,
    generate_plot: bool = True,
) -> TransformStepResult:
    """Execute independent scale transformation step on a single event DataFrame.

    Resolves channels, builds or applies the optimal scale transform, optionally
    renders the pre/post transformation diagnostic plot, and returns a structured result.
    """
    cfg = config or TransformConfig()
    explicit = target_channels if target_channels is not None else cfg.channels
    fluoro_channels = resolve_fluorescence_channels(
        fcs_data,
        explicit_channels=explicit,
        exclude_channels=cfg.exclude_channels,
        df=df_events,
        marker_metadata=marker_metadata,
    )

    if not cfg.enabled and custom_transform is None:
        return TransformStepResult(
            transformed_df=df_events.copy(),
            transform=None,
            channels_transformed=[],
            plot_path=None,
        )

    transform_obj = custom_transform or get_optimal_transform(
        fcs_data,
        transform_type=(cfg.method or "logicle").lower(),
        param_w=cfg.param_w,
        param_a=cfg.param_a,
        normalized=cfg.normalized,
        cofactor=cfg.cofactor,
    )

    df_transformed = apply_transform_to_dataframe(
        df=df_events,
        channels=fluoro_channels,
        transform=transform_obj,
    )

    plot_out_str: Optional[str] = None
    if generate_plot and output_plot_path is not None:
        from flowx.visualization.plotting import plot_transformation_comparison

        m_to_d = getattr(fcs_data, "marker_to_detector", {}) or {}
        cd4_det = fcs_data.resolve_channel_name("CD4") if "CD4" in m_to_d else None
        cd3_det = fcs_data.resolve_channel_name("CD3") if "CD3" in m_to_d else None
        if cd4_det and cd4_det in df_transformed.columns:
            test_fluoro = cd4_det
        elif cd3_det and cd3_det in df_transformed.columns:
            test_fluoro = cd3_det
        elif fluoro_channels:
            test_fluoro = fluoro_channels[0]
        else:
            test_fluoro = fcs_data.get_fsc_channel()

        if test_fluoro and test_fluoro in df_events.columns and test_fluoro in df_transformed.columns:
            is_biex = isinstance(transform_obj, BiexponentialTransform)
            xform_label = (
                "Biexponential"
                if is_biex
                else ("Logicle" if isinstance(transform_obj, LogicleTransform) else transform_obj.__class__.__name__)
            )
            out_p = Path(output_plot_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plot_transformation_comparison(
                raw_vals=df_events[test_fluoro].to_numpy(),
                transformed_vals=df_transformed[test_fluoro].to_numpy(),
                channel_name=fcs_data.get_channel_label(test_fluoro),
                transform_name=xform_label,
                output_path=out_p,
            )
            plot_out_str = str(out_p)

    return TransformStepResult(
        transformed_df=df_transformed,
        transform=transform_obj,
        channels_transformed=fluoro_channels,
        plot_path=plot_out_str,
    )


