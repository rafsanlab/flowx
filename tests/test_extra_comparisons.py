"""Unit tests for auxiliary external comparison package (flowx.extra.comparisons)."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from flowx.extra.comparisons import (
    compute_cohort_cleanup_concordance,
    compute_cohort_frequency_concordance,
    compute_replicate_concordance,
    compute_spillover_concordance,
    ellipse_vertices_to_polygon,
    evaluate_cluster_manual_alignment,
    evaluate_flowjo_ellipse_mask,
    export_per_sample_cleanup_plots,
    extract_diva_spillover_matrix,
    match_diva_tubes,
    match_flowjo_sample_gates,
    parse_all_diva_tubes,
    parse_all_flowjo_samples,
    parse_diva_xml_gates,
    parse_flowjo_xml_gates,
    plot_cleanup_gates_comparison,
    plot_cohort_population_heatmap_comparison,
    plot_cohort_retention_concordance,
    plot_compensation_concordance,
    plot_grouped_benchmark_bars,
    plot_lineage_cluster_comparison,
    plot_multi_sample_gate_grid,
    plot_replicate_scatter_concordance,
)
from flowx.gating.gate import PolygonGate


def test_parse_flowjo_xml_and_ellipse_mask(tmp_path: Path):
    """Verify FlowJo XML parsing across PolyRect, Polygon, and Ellipse plus geometric containment."""
    xml_file = tmp_path / "workspace.xml"
    xml_file.write_text(
        """<Workspace>
  <SampleList>
    <SampleNode name="Ctrl_1.fcs" sampleID="19">
      <Subpopulations>
        <Population name="not debris" count="500">
          <PolygonGate>
            <PolyRect>
              <Vertex x="0" y="0" />
              <Vertex x="50000" y="0" />
              <Vertex x="50000" y="250000" />
              <Vertex x="0" y="250000" />
            </PolyRect>
          </PolygonGate>
        </Population>
        <Population name="singlets" count="450">
          <PolygonGate>
            <Polygon>
              <Vertex x="60000" y="55000" />
              <Vertex x="200000" y="185000" />
              <Vertex x="200000" y="210000" />
              <Vertex x="60000" y="75000" />
            </Polygon>
          </PolygonGate>
        </Population>
        <Population name="CD4" count="200">
          <PolygonGate>
            <Ellipse>
              <Vertex x="0.0" y="2.0" />
              <Vertex x="4.0" y="2.0" />
              <Vertex x="2.0" y="1.0" />
              <Vertex x="2.0" y="3.0" />
            </Ellipse>
          </PolygonGate>
        </Population>
      </Subpopulations>
    </SampleNode>
  </SampleList>
</Workspace>
""",
        encoding="utf-8",
    )

    gates = parse_flowjo_xml_gates(xml_file, sample_id="19")
    assert set(gates.keys()) == {"not debris", "singlets", "CD4"}
    assert gates["not debris"].gate_type == "polyrect"
    assert gates["not debris"].count == 500
    assert gates["singlets"].gate_type == "polygon"
    assert gates["CD4"].gate_type == "ellipse"
    assert len(gates["CD4"].vertices) == 4

    by_name = parse_flowjo_xml_gates(xml_file, sample_name="Ctrl_1.fcs")
    assert "CD4" in by_name

    # Ellipse centered at (2.0, 2.0) with semi-axes (2.0, 1.0)
    events = pd.DataFrame({"X": [2.0, 0.5, 4.5, 2.0], "Y": [2.0, 2.0, 2.0, 3.5]})
    mask, poly = evaluate_flowjo_ellipse_mask(
        events,
        x_col="X",
        y_col="Y",
        vertices=gates["CD4"].vertices,
        transform=None,
    )
    assert list(mask) == [True, True, False, False]
    assert poly.shape == (120, 2)

    with pytest.raises(FileNotFoundError):
        parse_flowjo_xml_gates(tmp_path / "missing.xml")
    with pytest.raises(KeyError):
        parse_flowjo_xml_gates(xml_file, sample_id="999")
    with pytest.raises(ValueError):
        ellipse_vertices_to_polygon(np.zeros((2, 2)))


def test_parse_diva_xml_gates_and_spillover_matrix(tmp_path: Path):
    """Verify BD FACSDiva XML region extraction and hardware compensation matrix inversion."""
    # Construct known donor-row spillover matrix S and store comp_mat = inv(S.T)
    expected_spill = np.array([[1.0, 0.25], [0.05, 1.0]], dtype=np.float64)
    diva_comp = np.linalg.inv(expected_spill.T)

    coeff_tags = "\n".join(
        f"        <compensation_coefficient>{val}</compensation_coefficient>"
        for val in diva_comp.flatten()
    )
    diva_xml = tmp_path / "diva.xml"
    diva_xml.write_text(
        f"""<bdfacs>
  <tube name="01gx 11b">
    <regions>
      <region name="P1" xparm="FSC-A" yparm="SSC-A">
        <points>
          <point x="50000" y="30000" />
          <point x="180000" y="30000" />
          <point x="180000" y="160000" />
          <point x="50000" y="160000" />
        </points>
      </region>
    </regions>
    <compensation>
{coeff_tags}
    </compensation>
  </tube>
</bdfacs>
""",
        encoding="utf-8",
    )

    regions = parse_diva_xml_gates(diva_xml, tube_name="01gx 11b")
    assert "P1" in regions
    assert regions["P1"].x_param == "FSC-A"
    assert len(regions["P1"].vertices) == 4

    spill_df = extract_diva_spillover_matrix(diva_xml, tube_name="01gx 11b", channel_names=["BB515-A", "PE-A"])
    assert list(spill_df.columns) == ["BB515-A", "PE-A"]
    np.testing.assert_allclose(spill_df.to_numpy(), expected_spill, atol=1e-6)

    with pytest.raises(KeyError):
        parse_diva_xml_gates(diva_xml, tube_name="nonexistent")
    with pytest.raises(ValueError):
        extract_diva_spillover_matrix(diva_xml, tube_name="01gx 11b", channel_names=["A", "B", "C"])


def test_concordance_and_alignment_metrics():
    """Verify spillover concordance, replicate consistency, and manual-cluster alignment metrics."""
    ch = ["A", "B", "C"]
    ref = pd.DataFrame([[1.0, 0.2, 0.0], [0.05, 1.0, 0.3], [0.01, 0.1, 1.0]], index=ch, columns=ch)
    calc = pd.DataFrame([[1.0, 0.22, 0.0], [0.05, 1.0, 0.28], [0.01, 0.1, 1.0]], index=ch, columns=ch)

    res = compute_spillover_concordance(calc, ref)
    assert res.pearson_r > 0.999
    assert res.r_squared > 0.998
    assert res.off_diag_mae == pytest.approx((0.02 + 0.02) / 6.0, abs=1e-9)
    assert res.off_diag_max_abs_error == pytest.approx(0.02, abs=1e-9)
    assert res.diff_pct_dataframe.loc["A", "B"] == pytest.approx(2.0, abs=1e-9)

    rep = compute_replicate_concordance([10.0, 25.0, 65.0], [10.5, 24.0, 65.5])
    assert rep["pearson_r"] > 0.999
    assert rep["max_abs_delta"] == pytest.approx(1.0)

    align = evaluate_cluster_manual_alignment([0, 0, 1, 1, 2, 2], [2, 2, 0, 0, 1, 1])
    assert align["ari"] == pytest.approx(1.0)
    assert align["purity"] == pytest.approx(1.0)
    assert align["contingency_df"].shape == (3, 3)


def test_comparison_plotting_functions_export_pngs(tmp_path: Path):
    """Smoke-test all reusable comparison plots and verify PNG outputs are generated cleanly."""
    rng = np.random.default_rng(123)
    n = 300
    events_df = pd.DataFrame({
        "FSC-A": rng.uniform(50000, 180000, n),
        "SSC-A": rng.uniform(30000, 140000, n),
        "FSC-H": rng.uniform(45000, 175000, n),
        "CD4": np.concatenate([rng.normal(8.0, 0.5, 150), rng.normal(2.0, 0.5, 150)]),
        "CD8": np.concatenate([rng.normal(2.0, 0.5, 150), rng.normal(8.0, 0.5, 150)]),
        "UMAP_1": np.concatenate([rng.normal(-3.0, 0.5, 150), rng.normal(3.0, 0.5, 150)]),
        "UMAP_2": rng.normal(0.0, 0.5, n),
    })

    cells_gate = PolygonGate(
        name="Cells",
        channels=["FSC-A", "SSC-A"],
        vertices=[(60000, 35000), (170000, 35000), (170000, 130000), (60000, 130000)],
    )
    singlet_gate = PolygonGate(
        name="Singlets",
        channels=["FSC-A", "FSC-H"],
        vertices=[(60000, 50000), (170000, 150000), (170000, 170000), (60000, 70000)],
    )
    cells_mask = cells_gate.apply(events_df)

    cleanup_png = tmp_path / "cleanup.png"
    plot_cleanup_gates_comparison(
        ref_raw_df=events_df,
        flowx_working_df=events_df,
        flowx_cells_gate=cells_gate,
        flowx_singlet_gate=singlet_gate,
        flowx_cells_mask=cells_mask,
        ref_singlet_gate=singlet_gate,
        ref_cells_parent_mask=cells_mask,
        ref_negated_box_vertices=[(0, 0), (50000, 0), (50000, 50000), (0, 50000)],
        output_path=cleanup_png,
    )
    assert cleanup_png.is_file() and cleanup_png.stat().st_size > 1000

    man_labels = np.array([0] * 150 + [1] * 150)
    cl_labels = np.array([0] * 150 + [1] * 150)
    lineage_png = tmp_path / "lineage.png"
    plot_lineage_cluster_comparison(
        events_df=events_df,
        reference_labels=man_labels,
        cluster_labels=cl_labels,
        bivariate_x="CD4",
        bivariate_y="CD8",
        output_path=lineage_png,
    )
    assert lineage_png.is_file() and lineage_png.stat().st_size > 1000

    ch = ["B515-A", "PE-A", "APC-A"]
    calc_s = pd.DataFrame([[1.0, 0.25, 0.02], [0.05, 1.0, 0.15], [0.01, 0.08, 1.0]], index=ch, columns=ch)
    ref_s = pd.DataFrame([[1.0, 0.24, 0.02], [0.04, 1.0, 0.16], [0.01, 0.09, 1.0]], index=ch, columns=ch)
    comp_png = tmp_path / "comp.png"
    plot_compensation_concordance(
        calculated_df=calc_s,
        reference_df=ref_s,
        marker_map={"B515-A": "CD33", "PE-A": "CD80", "APC-A": "CD4"},
        reference_label="BD FACSDiva Matrix",
        output_path=comp_png,
    )
    assert comp_png.is_file() and comp_png.stat().st_size > 1000

    bars_png = tmp_path / "bars.png"
    plot_grouped_benchmark_bars(
        categories=["Pop A", "Pop B"],
        series=[
            {"label": "Manual", "values": [60.0, 25.0], "errors": [2.0, 1.5]},
            {"label": "FlowX", "values": [61.0, 24.5]},
        ],
        output_path=bars_png,
    )
    assert bars_png.is_file() and bars_png.stat().st_size > 1000

    rep_png = tmp_path / "rep.png"
    plot_replicate_scatter_concordance(
        values_x=[12.0, 38.0, 50.0],
        values_y=[12.3, 37.5, 50.2],
        point_labels=["C0", "C1", "C2"],
        output_path=rep_png,
    )
    assert rep_png.is_file() and rep_png.stat().st_size > 1000
    plt.close("all")


def test_multi_sample_workspace_matching_and_cohort_concordance(tmp_path: Path):
    """Verify multi-sample FlowJo disambiguation, FACSDiva specimen matching, and cohort metrics/plots."""
    # 1. Multi-sample FlowJo XML with duplicate filenames across centers
    fj_xml = tmp_path / "multi_flowjo.xml"
    fj_xml.write_text(
        """<Workspace>
  <SampleList>
    <SampleNode name="Rep_1.fcs" sampleID="1" count="50000">
      <Subpopulations>
        <Population name="singlets" count="41000">
          <PolygonGate><Polygon><Vertex x="10" y="10"/><Vertex x="20" y="20"/><Vertex x="10" y="20"/></Polygon></PolygonGate>
        </Population>
      </Subpopulations>
    </SampleNode>
    <SampleNode name="Rep_1.fcs" sampleID="19" count="119531">
      <Subpopulations>
        <Population name="singlets" count="87033">
          <PolygonGate><Polygon><Vertex x="15" y="15"/><Vertex x="30" y="30"/><Vertex x="15" y="30"/></Polygon></PolygonGate>
        </Population>
      </Subpopulations>
    </SampleNode>
    <SampleNode name="Rep_2.fcs" sampleID="20" count="115728">
      <Subpopulations>
        <Population name="singlets" count="86802">
          <PolygonGate><Polygon><Vertex x="15" y="15"/><Vertex x="30" y="30"/><Vertex x="15" y="30"/></Polygon></PolygonGate>
        </Population>
      </Subpopulations>
    </SampleNode>
  </SampleList>
</Workspace>""",
        encoding="utf-8",
    )

    all_fj = parse_all_flowjo_samples(fj_xml)
    assert len(all_fj) == 3

    matched_fj = match_flowjo_sample_gates(
        fj_xml,
        sample_event_counts={"Rep_1": 119531, "Rep_2": 115728},
    )
    assert matched_fj["Rep_1"].sample_id == "19"
    assert matched_fj["Rep_2"].sample_id == "20"
    assert matched_fj["Rep_1"].gates["singlets"].count == 87033

    # 2. Multi-tube FACSDiva XML under <specimen>
    diva_xml = tmp_path / "multi_diva.xml"
    diva_xml.write_text(
        """<bdfacs>
  <specimen name="Specimen_001">
    <tube name="01gx 11b">
      <region name="P1" xparm="FSC-A" yparm="SSC-A">
        <points><point x="50" y="50"/><point x="100" y="50"/><point x="100" y="100"/></points>
      </region>
    </tube>
    <tube name="01gx 33">
      <region name="P1" xparm="FSC-A" yparm="SSC-A">
        <points><point x="60" y="60"/><point x="110" y="60"/><point x="110" y="110"/></points>
      </region>
    </tube>
  </specimen>
</bdfacs>""",
        encoding="utf-8",
    )
    all_tubes = parse_all_diva_tubes(diva_xml)
    assert len(all_tubes) == 2
    matched_diva = match_diva_tubes(diva_xml, ["Specimen_001_01gx 11b", "Specimen_001_01gx 33"])
    assert set(matched_diva.keys()) == {"Specimen_001_01gx 11b", "Specimen_001_01gx 33"}
    assert matched_diva["Specimen_001_01gx 33"].tube_name == "01gx 33"

    # 3. Cohort retention concordance (including n=1 fallback and n=3 multi-condition)
    single_res = compute_cohort_cleanup_concordance(
        pd.DataFrame({"sample_id": ["S1"], "flowx_final_pct": [70.0], "ref_final_pct": [72.0]})
    )
    assert single_res["n_samples"] == 1
    assert single_res["pearson_r"] is None
    assert single_res["mae"] == pytest.approx(2.0)

    cohort_df = pd.DataFrame({
        "sample_id": ["C1", "C2", "T1", "T2"],
        "condition": ["Control", "Control", "Treated", "Treated"],
        "ref_final_pct": [72.0, 74.0, 68.0, 65.0],
        "flowx_final_pct": [70.5, 73.1, 66.8, 63.9],
    })
    cohort_metrics = compute_cohort_cleanup_concordance(cohort_df)
    assert cohort_metrics["n_samples"] == 4
    assert cohort_metrics["pearson_r"] > 0.99

    ret_png = tmp_path / "cohort_retention.png"
    plot_cohort_retention_concordance(cohort_df, output_path=ret_png)
    assert ret_png.is_file() and ret_png.stat().st_size > 1000

    # 4. Cohort frequency concordance and heatmap
    fx_freq = pd.DataFrame(
        {"CD4": [40.0, 42.0, 31.0], "CD8": [20.0, 19.0, 28.0]},
        index=["S1", "S2", "S3"],
    )
    ref_freq = pd.DataFrame(
        {"CD4": [39.5, 41.0, 30.2], "CD8": [20.5, 19.5, 27.5]},
        index=["S1", "S2", "S3"],
    )
    freq_metrics = compute_cohort_frequency_concordance(fx_freq, ref_freq)
    assert freq_metrics["n_samples"] == 3
    assert freq_metrics["pearson_r"] > 0.99

    heatmap_png = tmp_path / "cohort_heatmap.png"
    plot_cohort_population_heatmap_comparison(
        fx_freq,
        ref_freq_df=ref_freq,
        sample_conditions={"S1": "Control", "S2": "Control", "S3": "Treated"},
        output_path=heatmap_png,
    )
    assert heatmap_png.is_file() and heatmap_png.stat().st_size > 1000

    # 5. Per-sample export dir and multi-sample grid
    rng = np.random.default_rng(77)
    ev = pd.DataFrame({
        "FSC-A": rng.uniform(50000, 180000, 150),
        "SSC-A": rng.uniform(30000, 140000, 150),
        "FSC-H": rng.uniform(45000, 175000, 150),
    })
    c_gate = PolygonGate("Cells", ["FSC-A", "SSC-A"], [(60000, 35000), (170000, 35000), (170000, 130000), (60000, 130000)])
    s_gate = PolygonGate("Singlets", ["FSC-A", "FSC-H"], [(60000, 50000), (170000, 150000), (170000, 170000), (60000, 70000)])
    c_mask = c_gate.apply(ev)

    per_sample_dir = tmp_path / "plots" / "per_sample"
    saved = export_per_sample_cleanup_plots(
        [
            {
                "sample_id": sid,
                "ref_raw_df": ev,
                "flowx_working_df": ev,
                "flowx_cells_gate": c_gate,
                "flowx_singlet_gate": s_gate,
                "flowx_cells_mask": c_mask,
                "ref_singlet_gate": s_gate,
                "ref_cells_parent_mask": c_mask,
                "ref_cells_gate": c_gate,
            }
            for sid in ("Sample_A", "Sample_B")
        ],
        per_sample_dir=per_sample_dir,
    )
    assert len(saved) == 2
    assert all(p.is_file() and p.stat().st_size > 1000 for p in saved)

    grid_png = tmp_path / "grid.png"
    plot_multi_sample_gate_grid(
        [
            {"df": ev, "gate": c_gate, "title": "Sample A"},
            {"df": ev, "gate": c_gate, "title": "Sample B"},
            {"df": ev, "gate": c_gate, "title": "Sample C"},
        ],
        max_cols=2,
        output_path=grid_png,
    )
    assert grid_png.is_file() and grid_png.stat().st_size > 1000
    plt.close("all")

