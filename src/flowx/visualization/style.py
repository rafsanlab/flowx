"""Nature publication styling module for flow cytometry figures.

Provides publication-ready aesthetics conforming to Nature / Nature Methods guidelines:
- Compact subplot dimensions designed for single-column (89 mm / ~3.6 in) and
  double-column (180 mm / ~7.2 in) publication figures.
- Neutral, high-legibility sans-serif typography (Inter font with seamless OS fallbacks
  to Arial, Helvetica, DejaVu Sans).
- Minimalist axes lines (0.6 pt spines, outward ticks, subtle gridlines).
- TrueType (Type 42) font embedding for publication vector graphics (PDF/EPS).
- Full customizability: functions respect user-specified `figsize`, `ax`, and `rcParams`.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Standard publication figure dimensions (in inches)
# Calibrated for Nature column specifications:
# - Single column width: ~89 mm (3.5 - 3.6 inches)
# - Double column width: ~180 mm (7.1 - 7.2 inches)
DEFAULT_FIGSIZE_SINGLE: Tuple[float, float] = (3.6, 3.6)
DEFAULT_FIGSIZE_DUAL: Tuple[float, float] = (7.2, 3.6)
DEFAULT_FIGSIZE_TRIPLE: Tuple[float, float] = (10.8, 3.6)
DEFAULT_FIGSIZE_HEATMAP: Tuple[float, float] = (7.5, 5.0)
DEFAULT_FIGSIZE_WIDE: Tuple[float, float] = (5.0, 3.6)
DEFAULT_FIGSIZE_QC: Tuple[float, float] = (7.6, 4.4)

# Cross-platform prioritized sans-serif font stack
# Inter is prioritized; Arial/Helvetica/DejaVu Sans provide 100% OS fallback reliability
SANS_SERIF_FONT_STACK: List[str] = [
    "Inter",
    "Arial",
    "Helvetica Neue",
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
    "sans-serif",
]


def register_inter_font() -> bool:
    """Register bundled Inter TrueType fonts with matplotlib font manager.

    Ensures that Inter font families and weights (Regular, Medium, SemiBold, Bold)
    are available across all operating systems (Linux, macOS, Windows) even if not
    installed in the system font directory.

    Returns:
        bool: True if at least one Inter font weight was registered, False otherwise.
    """
    registered_any = False
    try:
        fonts_dir = Path(__file__).resolve().parent / "fonts"
        if fonts_dir.is_dir():
            for font_file in sorted(fonts_dir.glob("*.ttf")):
                try:
                    fm.fontManager.addfont(str(font_file))
                    registered_any = True
                except Exception as inner_exc:
                    logger.debug("Failed registering font %s: %s", font_file.name, inner_exc)
    except Exception as exc:
        logger.debug("Could not register bundled Inter fonts: %s", exc)
    return registered_any


def get_nature_rcparams() -> Dict[str, Union[str, float, int, bool, List[str]]]:
    """Return dictionary of matplotlib rcParams configured for Nature publication standards.

    Returns:
        Dict of matplotlib configuration parameters.
    """
    return {
        # Typography: clean sans-serif stack
        "font.family": "sans-serif",
        "font.sans-serif": list(SANS_SERIF_FONT_STACK),
        # Font sizes scaled for 3.6 in subplot panels (7 - 9 pt publication standard)
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.titlepad": 6.0,
        "axes.labelsize": 8.5,
        "axes.labelweight": "medium",
        "axes.labelpad": 4.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.title_fontsize": 8.0,
        "figure.titlesize": 10.0,
        "figure.titleweight": "bold",
        # Axes & Spines: 0.6 pt minimalist lines
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        # Ticks: pointing outwards, fine width
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        # Gridlines: very subtle light dashed lines
        "grid.color": "#e2e2e2",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        # Vector and bitmap export settings (Type 42 TrueType embedding for editable text)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }


def set_nature_style(reset_first: bool = False) -> None:
    """Apply Nature publication style globally to matplotlib rcParams.

    Registers bundled Inter font (with cross-OS fallbacks to Arial, Helvetica,
    and DejaVu Sans) and applies publication-grade typography, line weights,
    and figure parameters.

    Args:
        reset_first: If True, resets matplotlib to default configuration before applying.
    """
    if reset_first:
        matplotlib.rcdefaults()

    register_inter_font()
    params = get_nature_rcparams()
    matplotlib.rcParams.update(params)


@contextlib.contextmanager
def nature_style() -> Generator[None, None, None]:
    """Context manager for temporarily applying Nature publication style.

    Example:
        >>> with nature_style():
        >>>     fig, ax = plot_scatter_2d(df, "FSC-A", "SSC-A")
        >>>     fig.savefig("figure1a.pdf")
    """
    register_inter_font()
    params = get_nature_rcparams()
    with plt.rc_context(params):
        yield


def apply_nature_axis_style(
    ax: plt.Axes,
    show_grid: bool = True,
    despine: bool = False,
) -> None:
    """Apply Nature publication styling directly to a specific Matplotlib Axes.

    Ensures consistent line weights, tick directions, and label sizing even when
    users create custom subplots in Jupyter notebooks.

    Args:
        ax: Target Matplotlib Axes.
        show_grid: Whether to display subtle gridlines.
        despine: If True, removes top and right spines.
    """
    for spine_name, spine in ax.spines.items():
        spine.set_linewidth(0.6)
        spine.set_color("#333333")
        if despine and spine_name in ("top", "right"):
            spine.set_visible(False)

    ax.tick_params(
        direction="out",
        length=3.0,
        width=0.6,
        colors="#333333",
        labelsize=7.5,
    )
    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.5, color="#e2e2e2", alpha=0.7)
    else:
        ax.grid(False)


def format_multiline_title(title: Optional[str], max_chars: int = 42) -> Optional[str]:
    """Format and wrap plot or subplot titles into multi-line strings if they exceed max_chars.

    Preserves existing line breaks, splits lines cleanly on word boundaries, and avoids
    splitting hyphenated compound marker names where possible.

    Args:
        title: Input title string (or None).
        max_chars: Maximum character width per line before wrapping.

    Returns:
        Formatted multi-line title string, or None if title is None.
    """
    if not title:
        return title

    import textwrap

    lines = str(title).split("\n")
    wrapped_lines = []
    for line in lines:
        if len(line) <= max_chars:
            wrapped_lines.append(line)
        else:
            # Wrap on space boundaries without tearing words or hyphens apart
            wrapped = textwrap.fill(line, width=max_chars, break_long_words=False, break_on_hyphens=False)
            wrapped_lines.append(wrapped)
    return "\n".join(wrapped_lines)


# Automatically apply Nature style upon importing the visualization system
set_nature_style()

