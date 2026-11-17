"""Unit tests for square plot area aspect ratio standardization in flow cytometry plotting."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from flowx.gating.gate import RangeGate, RectangleGate
from flowx.visualization.plotting import (
    plot_clusters_scatter_2d,
    plot_compensation_comparison,
    plot_gate_overlay,
    plot_histogram_1d,
    plot_pipeline_cascade,
    plot_scatter_2d,
    plot_transformation_comparison,
)


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    """Generate synthetic event data for plotting validation."""
    np.random.seed(42)
    n = 2000
    fsc_a = np.random.normal(120000, 25000, n)
    ssc_a = np.random.normal(50000, 15000, n)
    cd4 = np.random.exponential(1.5, n)
    cd8 = np.random.exponential(1.2, n)
    return pd.DataFrame({
        "FSC-A": fsc_a,
        "SSC-A": ssc_a,
        "CD4": cd4,
        "CD8": cd8,
    })


def test_plot_scatter_2d_default_square_plot_area(sample_events_df):
    """Verify plot_scatter_2d defaults to a 1:1 square plot box aspect ratio."""
    fig, ax = plot_scatter_2d(
        df=sample_events_df,
        x_channel="FSC-A",
        y_channel="SSC-A",
    )
    assert ax.get_box_aspect() == 1.0
    plt.close(fig)


def test_plot_scatter_2d_custom_figsize_respected(sample_events_df):
    """Verify specifying custom non-square figsize leaves layout as-is without forcing square aspect."""
    custom_size = (11.0, 4.5)
    fig, ax = plot_scatter_2d(
        df=sample_events_df,
        x_channel="FSC-A",
        y_channel="SSC-A",
        figsize=custom_size,
    )
    # Custom non-square figsize must not have box_aspect forced
    assert ax.get_box_aspect() is None
    assert fig.get_size_inches()[0] == pytest.approx(11.0)
    assert fig.get_size_inches()[1] == pytest.approx(4.5)
    plt.close(fig)


def test_plot_scatter_2d_custom_square_figsize(sample_events_df):
    """Verify custom square figsize (e.g. 8x8) maintains square box aspect."""
    fig, ax = plot_scatter_2d(
        df=sample_events_df,
        x_channel="FSC-A",
        y_channel="SSC-A",
        figsize=(8.0, 8.0),
    )
    assert ax.get_box_aspect() == 1.0
    assert fig.get_size_inches()[0] == pytest.approx(8.0)
    assert fig.get_size_inches()[1] == pytest.approx(8.0)
    plt.close(fig)


def test_plot_histogram_1d_default_square_plot_area(sample_events_df):
    """Verify plot_histogram_1d defaults to a 1:1 square plot box aspect ratio."""
    fig, ax = plot_histogram_1d(
        df=sample_events_df,
        channel="CD4",
        cutoff=2.0,
    )
    assert ax.get_box_aspect() == 1.0
    plt.close(fig)


def test_plot_histogram_1d_custom_figsize(sample_events_df):
    """Verify custom figsize on plot_histogram_1d disables forced square box aspect."""
    fig, ax = plot_histogram_1d(
        df=sample_events_df,
        channel="CD4",
        figsize=(10.0, 3.5),
    )
    assert ax.get_box_aspect() is None
    plt.close(fig)


def test_plot_gate_overlay_2d_and_1d_square_plot_area(sample_events_df):
    """Verify both 2D scatter gates and 1D histogram gates have square plot areas by default."""
    # 2D Rectangle Gate (scatter)
    rect_gate = RectangleGate(name="Cells", channels=["FSC-A", "SSC-A"], x_min=50000, x_max=180000)
    fig2d, ax2d = plot_gate_overlay(sample_events_df, rect_gate)
    assert ax2d.get_box_aspect() == 1.0
    plt.close(fig2d)

    # 1D Range Gate (histogram)
    range_gate = RangeGate(name="CD4+", channel="CD4", min_val=1.0)
    fig1d, ax1d = plot_gate_overlay(sample_events_df, range_gate)
    assert ax1d.get_box_aspect() == 1.0
    plt.close(fig1d)


def test_plot_gate_overlay_custom_figsize(sample_events_df):
    """Verify custom figsize on plot_gate_overlay disables forced square aspect."""
    rect_gate = RectangleGate(name="Cells", channels=["FSC-A", "SSC-A"], x_min=50000, x_max=180000)
    fig, ax = plot_gate_overlay(sample_events_df, rect_gate, figsize=(12.0, 5.0))
    assert ax.get_box_aspect() is None
    plt.close(fig)


def test_plot_compensation_comparison_square_panels(sample_events_df):
    """Verify both subplots in plot_compensation_comparison have square plot areas by default."""
    fig = plot_compensation_comparison(
        uncomp_df=sample_events_df,
        comp_df=sample_events_df,
        x_channel="CD4",
        y_channel="CD8",
    )
    axes = fig.get_axes()
    assert len(axes) == 2
    assert axes[0].get_box_aspect() == 1.0
    assert axes[1].get_box_aspect() == 1.0
    plt.close(fig)


def test_plot_transformation_comparison_square_panels(sample_events_df):
    """Verify both histogram subplots in plot_transformation_comparison have square plot areas by default."""
    raw = sample_events_df["FSC-A"].to_numpy()
    xform = np.log1p(raw)
    fig = plot_transformation_comparison(
        raw_vals=raw,
        transformed_vals=xform,
        channel_name="FSC-A",
    )
    axes = fig.get_axes()
    assert len(axes) == 2
    assert axes[0].get_box_aspect() == 1.0
    assert axes[1].get_box_aspect() == 1.0
    plt.close(fig)


def test_plot_clusters_scatter_2d_square_plot_area(sample_events_df):
    """Verify cluster scatter plot has square plot area by default."""
    clusters = np.random.choice([0, 1, 2], size=len(sample_events_df))
    fig, ax = plot_clusters_scatter_2d(
        df=sample_events_df,
        cluster_labels=clusters,
        x_channel="FSC-A",
        y_channel="SSC-A",
    )
    assert ax.get_box_aspect() == 1.0
    plt.close(fig)


def test_plot_cluster_heatmap_with_marker_name_map():
    """Verify plot_cluster_heatmap formats x-axis ticks with biological marker names when marker_name_map is provided."""
    from flowx.visualization.plotting import plot_cluster_heatmap

    stats_df = pd.DataFrame({
        "Cluster ID": [0, 1],
        "Count": [100, 200],
        "% Total": [33.3, 66.7],
        "PE-TxRed-A Median": [0.5, 0.2],
        "PacBlue-A Median": [0.1, 0.8],
    })
    marker_map = {
        "PE-TxRed-A": "CD4",
        "PacBlue-A": "CD8",
    }
    fig = plot_cluster_heatmap(
        cluster_stats_df=stats_df,
        marker_cols=["PE-TxRed-A", "PacBlue-A"],
        marker_name_map=marker_map,
    )
    ax = fig.axes[0]
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "CD4 (PE-TxRed-A)" in tick_labels
    assert "CD8 (PacBlue-A)" in tick_labels
    plt.close(fig)


def test_plot_pipeline_cascade_square_panels(sample_events_df):
    """Verify all panels in plot_pipeline_cascade have 1:1 square plot areas by default."""
    from flowx.gating.gate import RectangleGate, GatingHierarchy
    from flowx.visualization.plotting import plot_pipeline_cascade

    gate1 = RectangleGate("Cells", channels=["FSC-A", "SSC-A"], x_min=50, x_max=200, y_min=20, y_max=150)
    gate2 = RectangleGate("Singlets", channels=["FSC-A", "SSC-A"], x_min=60, x_max=180, y_min=30, y_max=140, parent_gate="Cells")
    hierarchy = GatingHierarchy()
    hierarchy.add_gate(gate1)
    hierarchy.add_gate(gate2)
    masks = hierarchy.evaluate(sample_events_df)

    fig = plot_pipeline_cascade(
        df=sample_events_df,
        gates=[gate1, gate2],
        hierarchy_masks=masks,
    )
    for ax in fig.axes:
        assert ax.get_box_aspect() == 1.0
    plt.close(fig)


def test_plot_clustering_comparison_square_panels(sample_events_df):
    """Verify all 3 panels in plot_clustering_comparison have 1:1 square plot areas."""
    from flowx.visualization.plotting import plot_clustering_comparison

    n = len(sample_events_df)
    template_labels = np.array(["CD4 T Cells"] * (n // 2) + ["CD8 T Cells"] * (n - n // 2))
    som_labels = np.random.choice([0, 1, 2], size=n)
    phenograph_labels = np.random.choice([0, 1, 2], size=n)

    fig = plot_clustering_comparison(
        df=sample_events_df,
        template_labels=template_labels,
        som_labels=som_labels,
        phenograph_labels=phenograph_labels,
        x_channel="FSC-A",
        y_channel="SSC-A",
    )
    for ax in fig.axes:
        assert ax.get_box_aspect() == 1.0
    plt.close(fig)


def test_plot_cluster_heatmap_square_cells_and_inner_lines():
    """Verify plot_cluster_heatmap has aspect='equal' and inner white grid lines."""
    from flowx.visualization.plotting import plot_cluster_heatmap

    summary_df = pd.DataFrame({
        "Cluster ID": [0, 1, 2, 3],
        "% Total": [40.0, 30.0, 20.0, 10.0],
        "CD3": [100, 2000, 150, 2500],
        "CD4": [50, 1500, 80, 200],
        "CD8": [80, 100, 2000, 150],
    })
    fig = plot_cluster_heatmap(summary_df, ["CD3", "CD4", "CD8"])
    ax = fig.axes[0]
    assert ax.get_aspect() in ("equal", 1.0)
    # Verify inner grid lines exist (axhline and axvline) and are thin (<= 0.5px)
    hlines = [line for line in ax.lines if line.get_color() == "white" or (isinstance(line.get_color(), tuple) and line.get_color()[:3] == (1.0, 1.0, 1.0))]
    assert len(hlines) >= 3  # (n_clusters - 1) inner lines
    assert hlines[0].get_linewidth() <= 0.5
    plt.close(fig)


def test_plot_cohort_heatmap_exact_square_cells():
    """Verify plot_cohort_heatmap mathematically maintains 1:1 square cells across shapes."""
    from flowx.stats.models import CohortFeatureSet
    from flowx.visualization.stats_plots import plot_cohort_heatmap

    n_samples, n_pops = 5, 8
    data_df = pd.DataFrame(
        np.random.rand(n_samples, n_pops),
        index=[f"Sample_{i}" for i in range(n_samples)],
        columns=[f"Pop_{j}" for j in range(n_pops)],
    )
    meta = pd.DataFrame({
        "condition": ["Control", "Control", "Treated", "Treated", "Treated"]
    }, index=data_df.index)

    cfs = CohortFeatureSet(
        sample_ids=list(data_df.index),
        population_names=list(data_df.columns),
        marker_names=[],
        count_matrix=data_df,
        parent_freq_matrix=data_df,
        viable_freq_matrix=data_df,
        sample_metadata=meta,
    )
    fig = plot_cohort_heatmap(cfs, condition_column="condition")

    for ax in fig.axes:
        labels = [t.get_text() for t in ax.get_xticklabels()]
        if set(labels) == set(data_df.columns):
            renderer = fig.canvas.get_renderer()
            bbox = ax.get_window_extent(renderer)
            cell_w = bbox.width / n_pops
            cell_h = bbox.height / n_samples
            aspect = cell_w / cell_h
            assert aspect == pytest.approx(1.0, abs=0.01)

    # Verify colorbar (Tier 1) and condition legend (Tier 2) are separated vertically with no collision
    renderer = fig.canvas.get_renderer()
    cbar_ax = fig.axes[-1]
    cbar_bbox = cbar_ax.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    leg_bbox = fig.legends[0].get_window_extent(renderer).transformed(fig.transFigure.inverted())
    assert cbar_bbox.y0 > leg_bbox.y1  # Strictly positive vertical clearance between tiers
    plt.close(fig)


def test_plot_volcano_and_forest_square_aspect():
    """Verify plot_volcano_da and plot_forest_da enforce 1:1 square plot box aspect ratio."""
    from flowx.stats.differential_abundance import DAResult, DAConfig
    from flowx.visualization.stats_plots import plot_volcano_da, plot_forest_da

    summary_df = pd.DataFrame({
        "population": ["CD4+ T cells", "CD8+ T cells", "B cells"],
        "log2fc": [1.5, -0.8, 0.2],
        "p_value": [0.001, 0.02, 0.5],
        "q_value": [0.003, 0.03, 0.5],
    })
    da_res = DAResult(
        comparison_name="Treated_vs_Control",
        target_group="Treated",
        reference_group="Control",
        summary_df=summary_df,
        significant_populations=["CD4+ T cells"],
        config=DAConfig(),
    )

    fig_volc = plot_volcano_da(da_res)
    ax_volc = fig_volc.axes[0]
    assert ax_volc.get_box_aspect() == 1.0
    plt.close(fig_volc)

    fig_forest = plot_forest_da(da_res)
    ax_forest = fig_forest.axes[0]
    assert ax_forest.get_box_aspect() == 1.0
    plt.close(fig_forest)


def test_plot_goodness_of_fit_square_aspect():
    """Verify plot_deconvolution_goodness_of_fit enforces 1:1 square plot box aspect ratio for both panels."""
    from flowx.spectral.visualization import plot_deconvolution_goodness_of_fit
    from flowx.spectral.unmixing import UnmixingResult
    from flowx.spectral.library import ReferenceSpectralLibrary, SpectralReference

    ref = SpectralReference(
        fluorophore="CD4",
        carrier="beads",
        channel_names=["Det1", "Det2", "Det3"],
        normalized_spectrum=np.array([0.2, 1.0, 0.1]),
        raw_positive_mfi=np.array([20.0, 100.0, 10.0]),
        raw_negative_mfi=np.array([0.0, 0.0, 0.0]),
        peak_channel="Det2",
    )
    matrix = np.array([[0.2, 1.0, 0.1]]).T
    df = pd.DataFrame(matrix, index=["Det1", "Det2", "Det3"], columns=["CD4"])
    lib = ReferenceSpectralLibrary(
        detectors=["Det1", "Det2", "Det3"],
        fluorophores=["CD4"],
        matrix=matrix,
        dataframe=df,
        references={"CD4": ref},
    )

    unmixed_df = pd.DataFrame({
        "CD4": [10.0, 20.0, 30.0],
    })
    residuals = np.array([0.1, 0.2, 0.15])
    unmixing_res = UnmixingResult(
        unmixed_df=unmixed_df,
        residuals_norm=residuals,
        spectral_library=lib,
        method_used="wls",
        fluorophore_channels=["CD4"],
        mean_residual_norm=float(np.mean(residuals)),
        median_residual_norm=float(np.median(residuals)),
        residual_outliers_count=0,
    )

    fig_gof = plot_deconvolution_goodness_of_fit(unmixing_res)
    assert len(fig_gof.axes) >= 2
    assert fig_gof.axes[0].get_box_aspect() == 1.0
    assert fig_gof.axes[1].get_box_aspect() == 1.0
    # Verify subplots are placed compactly side-by-side with tight spacing
    fig_w, fig_h = fig_gof.get_size_inches()
    assert fig_w <= 8.5
    plt.close(fig_gof)


def test_plot_umap_cluster_color_matching():
    """Verify plot_umap_projection and plot_clusters_scatter_2d match cluster badge color to cluster scatter color."""
    from flowx.visualization.plotting import plot_umap_projection, plot_clusters_scatter_2d
    import pandas as pd

    rng = np.random.RandomState(42)
    coords = rng.randn(100, 2)
    labels = np.array([0, 1, 2, 3] * 25)

    # 1. Test UMAP with match_cluster_color=True
    fig, ax = plot_umap_projection(
        embedding=coords,
        labels=labels,
        legend_mode="centroid",
        match_cluster_color=True,
    )
    # Check that annotations exist and have colors matching colormap (not black)
    texts = [c for c in ax.get_children() if isinstance(c, plt.Annotation)]
    assert len(texts) == 4
    for annot in texts:
        color = annot.get_color()
        assert color != "black"
    plt.close(fig)

    # 2. Test UMAP with match_cluster_color=False
    fig_black, ax_black = plot_umap_projection(
        embedding=coords,
        labels=labels,
        legend_mode="centroid",
        match_cluster_color=False,
    )
    texts_black = [c for c in ax_black.get_children() if isinstance(c, plt.Annotation)]
    assert len(texts_black) == 4
    for annot in texts_black:
        assert annot.get_color() == "black"
    plt.close(fig_black)

    # 3. Test plot_clusters_scatter_2d with match_cluster_color=True
    df = pd.DataFrame({"FSC-A": coords[:, 0], "SSC-A": coords[:, 1]})
    fig_2d, ax_2d = plot_clusters_scatter_2d(
        df=df,
        cluster_labels=labels,
        x_channel="FSC-A",
        y_channel="SSC-A",
        legend_mode="centroid",
        match_cluster_color=True,
    )
    texts_2d = [c for c in ax_2d.texts]
    assert len(texts_2d) == 4
    for txt in texts_2d:
        assert txt.get_color() != "black"
    plt.close(fig_2d)


def test_plot_cluster_heatmap_seamless_strip_concise_labels_and_legend():
    """Verify plot_cluster_heatmap renders concise labels, seamless strip, and discrete population legend."""
    from flowx.visualization.plotting import plot_cluster_heatmap, get_cluster_palette

    stats_df = pd.DataFrame({
        "Cluster ID": [0, 1, 2],
        "Count": [100, 200, 300],
        "% Total": [16.7, 33.3, 50.0],
        "CD3 Median": [10, 500, 2000],
        "CD4 Median": [20, 1000, 100],
        "CD8 Median": [5, 50, 1500],
    })
    pop_labels = {
        0: "CD4- CD8- Double Negative",
        1: "CD4+ Helper T Cells",
        2: "CD8+ Cytotoxic T Cells",
    }
    fig = plot_cluster_heatmap(
        cluster_stats_df=stats_df,
        marker_cols=["CD3", "CD4", "CD8"],
        cluster_to_label=pop_labels,
        cluster_palette="Spectral",
    )
    ax = fig.axes[0]
    y_tick_texts = [t.get_text() for t in ax.get_yticklabels()]
    # Row labels must be concise (e.g. C0 (16.7%), C1 (33.3%)), NOT full phenotype names
    for ytxt in y_tick_texts:
        assert ytxt.startswith("C")
        assert "%" in ytxt
        assert "Double Negative" not in ytxt
        assert "Cytotoxic" not in ytxt

    # Verify legend exists and contains population labels
    leg = ax.get_legend()
    assert leg is not None
    leg_texts = [t.get_text() for t in leg.get_texts()]
    assert any("CD4+ Helper T Cells" in t for t in leg_texts)
    assert any("CD8+ Cytotoxic T Cells" in t for t in leg_texts)

    # Verify legend does not overlap y-axis title
    renderer = fig.canvas.get_renderer()
    y_bb = ax.yaxis.label.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    leg_bb = leg.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    assert leg_bb.x0 > y_bb.x1, f"Legend left edge ({leg_bb.x0}) must be strictly to the right of y-axis label ({y_bb.x1})"
    assert leg_bb.x1 <= 1.0, f"Legend right edge ({leg_bb.x1}) must not exceed figure canvas boundary"

    # Verify gap between row color strip and heatmap matrix is preserved
    rc_ax = [a for a in fig.axes if a != ax and len(a.collections) > 0 and a.collections[0].get_linewidths()[0] == 0][0]
    w_fig, _ = fig.get_size_inches()
    gap_in = (ax.get_position().x0 - rc_ax.get_position().x1) * w_fig
    assert 0.075 <= gap_in <= 0.085, f"Row color strip gap must be ~0.08 inches, got {gap_in:.3f}"

    # Verify cluster palette helper generates distinct colors
    colors = get_cluster_palette([0, 1, 2], cmap_name="Spectral")
    assert len(colors) == 3
    assert colors["0"] != colors["1"]

    plt.close(fig)





