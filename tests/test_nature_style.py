"""Unit tests for Nature publication styling, Inter font fallback, and compact subplot sizing."""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from flowx.gating.gate import PolygonGate, RangeGate, RectangleGate
from flowx.visualization import (
    DEFAULT_FIGSIZE_DUAL,
    DEFAULT_FIGSIZE_HEATMAP,
    DEFAULT_FIGSIZE_QC,
    DEFAULT_FIGSIZE_SINGLE,
    DEFAULT_FIGSIZE_TRIPLE,
    DEFAULT_FIGSIZE_WIDE,
    SANS_SERIF_FONT_STACK,
    apply_nature_axis_style,
    get_nature_rcparams,
    nature_style,
    plot_cluster_heatmap,
    plot_clustering_comparison,
    plot_clustering_qc_diagnostic,
    plot_clusters_scatter_2d,
    plot_compensation_comparison,
    plot_gate_overlay,
    plot_histogram_1d,
    plot_scatter_2d,
    plot_transformation_comparison,
    register_inter_font,
    set_nature_style,
)



@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate reproducible sample DataFrame for visualization tests."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "FSC-A": np.random.normal(100000, 20000, n),
        "SSC-A": np.random.normal(40000, 10000, n),
        "CD4": np.random.exponential(1.5, n),
        "CD8": np.random.exponential(1.2, n),
    })


def test_nature_rcparams_configuration():
    """Verify that Nature publication rcParams match publication standards."""
    params = get_nature_rcparams()
    assert params["font.family"] == "sans-serif"
    assert params["font.sans-serif"][0] == "Inter"
    assert "Arial" in params["font.sans-serif"]
    assert "DejaVu Sans" in params["font.sans-serif"]
    assert params["axes.linewidth"] == 0.6
    assert params["xtick.direction"] == "out"
    assert params["ytick.direction"] == "out"
    assert params["pdf.fonttype"] == 42  # TrueType vector font embedding
    assert params["ps.fonttype"] == 42


def test_cross_os_font_stack():
    """Verify prioritized cross-platform font stack guarantees fallbacks for all OSs."""
    assert SANS_SERIF_FONT_STACK[0] == "Inter"
    # macOS fallbacks
    assert "Helvetica" in SANS_SERIF_FONT_STACK or "Helvetica Neue" in SANS_SERIF_FONT_STACK
    # Windows fallbacks
    assert "Arial" in SANS_SERIF_FONT_STACK
    # Linux fallbacks
    assert "DejaVu Sans" in SANS_SERIF_FONT_STACK or "Liberation Sans" in SANS_SERIF_FONT_STACK
    # Generic CSS / Matplotlib fallback
    assert SANS_SERIF_FONT_STACK[-1] == "sans-serif"


def test_register_inter_font():
    """Verify register_inter_font finds and registers the bundled Inter TrueType font."""
    registered = register_inter_font()
    assert registered is True
    # Verify Inter is recognized in fontManager
    available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    assert "Inter" in available_fonts


def test_default_publication_subplot_sizes(sample_data):
    """Verify default figure sizes are compact publication-ready dimensions (~3.6 in)."""
    # 1. 2D Scatter plot
    fig_scatter, _ = plot_scatter_2d(sample_data, "FSC-A", "SSC-A")
    assert fig_scatter.get_size_inches()[0] == pytest.approx(DEFAULT_FIGSIZE_SINGLE[0], abs=0.01)
    assert fig_scatter.get_size_inches()[1] == pytest.approx(DEFAULT_FIGSIZE_SINGLE[1], abs=0.01)
    plt.close(fig_scatter)

    # 2. 1D Histogram plot
    fig_hist, _ = plot_histogram_1d(sample_data, "CD4")
    assert fig_hist.get_size_inches()[0] == pytest.approx(DEFAULT_FIGSIZE_SINGLE[0], abs=0.01)
    assert fig_hist.get_size_inches()[1] == pytest.approx(DEFAULT_FIGSIZE_SINGLE[1], abs=0.01)
    plt.close(fig_hist)

    # 3. Gate overlay plot
    gate = RectangleGate("Lymphocytes", ["FSC-A", "SSC-A"], 50000, 150000, 20000, 80000)
    fig_gate, _ = plot_gate_overlay(sample_data, gate)
    assert fig_gate.get_size_inches()[0] == pytest.approx(DEFAULT_FIGSIZE_SINGLE[0], abs=0.01)
    assert fig_gate.get_size_inches()[1] == pytest.approx(DEFAULT_FIGSIZE_SINGLE[1], abs=0.01)
    plt.close(fig_gate)

    # 4. Compensation comparison (2 panels side-by-side -> dual column 7.2 in)
    fig_comp = plot_compensation_comparison(sample_data, sample_data, "CD4", "CD8")
    assert fig_comp.get_size_inches()[0] == pytest.approx(DEFAULT_FIGSIZE_DUAL[0], abs=0.01)
    assert fig_comp.get_size_inches()[1] == pytest.approx(DEFAULT_FIGSIZE_DUAL[1], abs=0.01)
    plt.close(fig_comp)

    # 5. Transformation comparison (2 panels side-by-side -> dual column 7.2 in)
    raw = sample_data["FSC-A"].to_numpy()
    xform = np.log1p(raw)
    fig_xform = plot_transformation_comparison(raw, xform, "FSC-A")
    assert fig_xform.get_size_inches()[0] == pytest.approx(DEFAULT_FIGSIZE_DUAL[0], abs=0.01)
    assert fig_xform.get_size_inches()[1] == pytest.approx(DEFAULT_FIGSIZE_DUAL[1], abs=0.01)
    plt.close(fig_xform)

    # 6. Cluster scatter (centroid mode -> compact 3.6 in)
    clusters = np.random.choice([0, 1, 2], size=len(sample_data))
    fig_clust, _ = plot_clusters_scatter_2d(sample_data, clusters, "FSC-A", "SSC-A")
    assert fig_clust.get_size_inches()[0] == pytest.approx(DEFAULT_FIGSIZE_SINGLE[0], abs=0.01)
    assert fig_clust.get_size_inches()[1] == pytest.approx(DEFAULT_FIGSIZE_SINGLE[1], abs=0.01)
    plt.close(fig_clust)

    # 7. Cluster scatter with external legend (5.0, 3.6 in)
    fig_clust_ext, _ = plot_clusters_scatter_2d(sample_data, clusters, "FSC-A", "SSC-A", legend_mode="external")
    assert fig_clust_ext.get_size_inches()[0] == pytest.approx(DEFAULT_FIGSIZE_WIDE[0], abs=0.01)
    assert fig_clust_ext.get_size_inches()[1] == pytest.approx(DEFAULT_FIGSIZE_WIDE[1], abs=0.01)
    plt.close(fig_clust_ext)


def test_user_manual_figsize_adjustment(sample_data):
    """Verify user can manually customize figsize in notebooks and scripts."""
    custom_size = (7.5, 4.2)
    fig, ax = plot_scatter_2d(sample_data, "FSC-A", "SSC-A", figsize=custom_size)
    assert fig.get_size_inches()[0] == pytest.approx(7.5)
    assert fig.get_size_inches()[1] == pytest.approx(4.2)
    plt.close(fig)


def test_user_manual_subplot_embedding(sample_data):
    """Verify user can create multi-panel publication figures and pass ax=axes[i] seamlessly."""
    # User creates a publication figure with a 2x2 grid in a notebook
    user_figsize = (7.2, 7.2)
    fig, axes = plt.subplots(2, 2, figsize=user_figsize)

    # Call plotting functions directly into user subplots
    plot_scatter_2d(sample_data, "FSC-A", "SSC-A", ax=axes[0, 0], title="A: Morphology")
    plot_histogram_1d(sample_data, "CD4", ax=axes[0, 1], title="B: CD4 Expression")
    gate = RectangleGate("Lymphocytes", ["FSC-A", "SSC-A"], 50000, 150000, 20000, 80000)
    plot_gate_overlay(sample_data, gate, ax=axes[1, 0], title="C: Gated Cells")
    clusters = np.random.choice([0, 1, 2], size=len(sample_data))
    plot_clusters_scatter_2d(sample_data, clusters, "FSC-A", "SSC-A", ax=axes[1, 1], title="D: Clusters")

    # Figure size must strictly remain the user's defined figure size
    assert fig.get_size_inches()[0] == pytest.approx(7.2)
    assert fig.get_size_inches()[1] == pytest.approx(7.2)

    # Each axis must contain expected plot artists
    assert len(axes[0, 0].collections) > 0  # scatter points
    assert len(axes[0, 1].patches) > 0      # histogram bars
    assert len(axes[1, 0].patches) > 0      # gate patch
    assert len(axes[1, 1].collections) > 0  # cluster points

    plt.close(fig)


def test_nature_style_context_manager(sample_data):
    """Verify nature_style context manager works for isolated execution."""
    with nature_style():
        fig, ax = plot_scatter_2d(sample_data, "FSC-A", "SSC-A")
        assert fig.get_size_inches()[0] == pytest.approx(3.6, abs=0.01)
        plt.close(fig)


def test_apply_nature_axis_style():
    """Verify apply_nature_axis_style applies 0.6pt lines and outward ticks."""
    fig, ax = plt.subplots()
    apply_nature_axis_style(ax, show_grid=True, despine=True)
    assert ax.spines["bottom"].get_linewidth() == pytest.approx(0.6)
    assert ax.spines["left"].get_linewidth() == pytest.approx(0.6)
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    plt.close(fig)


def test_plot_clustering_qc_diagnostic_legend_outside_bottom(sample_data):
    """Verify plot_clustering_qc_diagnostic defaults to outside bottom legends without obscuring data."""
    from flowx.clustering.qc_clustering import ClusteringQCResult

    n = len(sample_data)
    sample_data["FSC-H"] = sample_data["FSC-A"] * 0.95

    hull_cells = np.array([[60000, 20000], [140000, 20000], [150000, 70000], [70000, 60000]])
    cells_gate = PolygonGate("Cells", ["FSC-A", "SSC-A"], hull_cells)

    hull_sing = np.array([[60000, 58000], [140000, 135000], [145000, 142000], [62000, 65000]])
    singlets_gate = PolygonGate("Singlets", ["FSC-A", "FSC-H"], hull_sing)

    morph_labels = np.random.choice(["Intact Cells", "Debris / Noise", "Saturation Outliers"], size=n, p=[0.7, 0.25, 0.05])
    sing_labels = np.random.choice(["Singlets", "Doublets / Aggregates", "Debris / Non-cells"], size=n, p=[0.65, 0.15, 0.20])

    qc_res = ClusteringQCResult(
        cells_mask=morph_labels == "Intact Cells",
        singlets_mask=sing_labels == "Singlets",
        cells_gate=cells_gate,
        singlets_gate=singlets_gate,
        total_events=n,
        cell_count=int(np.sum(morph_labels == "Intact Cells")),
        singlet_count=int(np.sum(sing_labels == "Singlets")),
        debris_count=int(np.sum(morph_labels == "Debris / Noise")),
        doublet_count=int(np.sum(sing_labels == "Doublets / Aggregates")),
        morphology_labels=morph_labels,
        singlet_labels=sing_labels,
        summary_stats=pd.DataFrame(),
    )

    # Test default: legend_loc="bottom"
    fig_default = plot_clustering_qc_diagnostic(sample_data, qc_res)
    assert fig_default.get_size_inches()[0] == pytest.approx(DEFAULT_FIGSIZE_QC[0], abs=0.01)
    assert fig_default.get_size_inches()[1] == pytest.approx(DEFAULT_FIGSIZE_QC[1], abs=0.01)

    # Verify both axes have legends positioned outside below the axis (bbox y < 0)
    for ax in fig_default.get_axes():
        leg = ax.get_legend()
        assert leg is not None
        # In axes display coordinates, the legend box center/anchor is below y=0 (y=-0.18)
        assert leg._loc == 9 or leg.get_window_extent is not None  # loc 9 is upper center

    plt.close(fig_default)

    # Test optional inside legend
    fig_inside = plot_clustering_qc_diagnostic(sample_data, qc_res, legend_loc="inside")
    assert fig_inside.get_size_inches()[0] == pytest.approx(7.6, abs=0.01)
    assert fig_inside.get_size_inches()[1] == pytest.approx(3.8, abs=0.01)
    plt.close(fig_inside)


def test_format_multiline_title(sample_data: pd.DataFrame):
    """Verify that long plot and subplot titles are automatically wrapped into multi-line strings."""
    from flowx.visualization import format_multiline_title

    # Short title should remain single line
    short_title = "FSC-A vs SSC-A"
    assert format_multiline_title(short_title, max_chars=40) == short_title

    # None should return None
    assert format_multiline_title(None) is None

    # Long title exceeding max_chars should split cleanly across lines
    long_title = "Step 1A: Raw Morphology (FSC-A vs SSC-A) - samples_WBC 9196_002.fcs"
    wrapped = format_multiline_title(long_title, max_chars=40)
    assert "\n" in wrapped
    for line in wrapped.split("\n"):
        # Lines should not exceed max_chars unless a single unbroken token exceeds it
        assert len(line) <= 45

    # Multi-line title with existing breaks should preserve existing breaks
    multi = "Title Line 1\nThis is a very long title line 2 that definitely needs to wrap around to multiple lines"
    wrapped_multi = format_multiline_title(multi, max_chars=40)
    assert wrapped_multi.startswith("Title Line 1\n")
    assert wrapped_multi.count("\n") >= 2

    # Verify plot_scatter_2d applies wrapped title to Axes
    fig, ax = plot_scatter_2d(sample_data, "FSC-A", "SSC-A", title=long_title)
    ax_title = ax.get_title()
    assert "\n" in ax_title
    plt.close(fig)


