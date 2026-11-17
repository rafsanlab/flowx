"""Unit tests for Gating-ML 2.0 and FlowJo .wsp workspace export."""

from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

from flowx.gating.gate import GatingHierarchy, PolygonGate, QuadrantGate, RangeGate, RectangleGate
from flowx.workspace.gating_ml_export import GATING_ML_NS, export_flowjo_wsp, export_gating_ml
from flowx.workspace.session import WorkspaceSession


def test_export_gating_ml_standard(tmp_path: Path):
    """Verify export_gating_ml writes standard Gating-ML 2.0 XML with all gate types."""
    hierarchy = GatingHierarchy()
    poly = PolygonGate(
        name="Debris",
        channels=["FSC-A", "SSC-A"],
        vertices=[[50000, 20000], [150000, 20000], [150000, 100000], [50000, 100000]],
    )
    hierarchy.add_gate(poly)

    rect = RectangleGate(name="Singlets", channels=["FSC-A", "FSC-H"], x_min=50000, x_max=150000, y_min=40000, y_max=140000, parent_gate="Debris")
    hierarchy.add_gate(rect)

    rng_gate = RangeGate(name="Viable", channel="7AAD-A", max_val=0.45, parent_gate="Singlets")
    hierarchy.add_gate(rng_gate)

    quad = QuadrantGate(name="Subsets", channels=["CD4", "CD8"], x_cutoff=0.35, y_cutoff=0.40, parent_gate="Viable")
    hierarchy.add_gate(quad)

    out_file = tmp_path / "gating_ml_test.xml"
    res_path = export_gating_ml(hierarchy, out_file, experiment_name="UnitTest_Cohort")

    assert res_path.exists()
    content = res_path.read_text(encoding="utf-8")
    assert "gating:Gating-ML" in content
    assert "PolygonGate" in content
    assert "RectangleGate" in content
    assert "QuadrantGate" in content
    assert "CD4" in content
    assert "CD8" in content

    # Parse and verify XML validity
    root = ET.fromstring(content)
    assert root.tag == f"{{{GATING_ML_NS}}}Gating-ML"


def test_export_flowjo_workspace_session(tmp_path: Path):
    """Verify WorkspaceSession exports FlowJo-compatible .wsp XML."""
    session = WorkspaceSession(name="FlowJoTestWorkspace", sample_filename="Case1.fcs", total_events=50000)
    session.hierarchy.add_gate(RangeGate(name="CD3+", channel="CD3-A", min_val=0.25))

    out_wsp = tmp_path / "workspace.wsp"
    res_path = session.export_flowjo_wsp(out_wsp)

    assert res_path.exists()
    content = res_path.read_text(encoding="utf-8")
    assert "<Workspace" in content
    assert "Case1.fcs" in content
    assert "RangeGate" in content

    root = ET.fromstring(content)
    assert root.tag == "Workspace"
    assert root.attrib["flowJoVersion"] == "10.8.1"


def test_export_gating_ml_with_comp_and_transforms_parses_in_flowkit(tmp_path: Path):
    """Verify Gating-ML 2.0 export with CompensationMatrix & LogicleTransform parses in FlowKit."""
    import flowkit as fk
    import numpy as np
    from flowx.math.compensation import CompensationMatrix
    from flowx.math.transforms import LogicleTransform
    from flowx.workspace.gating_ml_export import export_preprocessing_gating_ml

    comp = CompensationMatrix(
        detectors=["FITC-A", "PE-A", "7AAD-A"],
        matrix=np.array([
            [1.0, 0.05, 0.0],
            [0.02, 1.0, 0.0],
            [0.0, 0.01, 1.0],
        ]),
    )
    lt = LogicleTransform(param_t=262144.0, param_w=0.5, param_m=4.5, param_a=0.0)
    xforms = {"FITC-A": lt, "PE-A": lt, "7AAD-A": lt}

    hierarchy = GatingHierarchy()
    hierarchy.add_gate(PolygonGate(
        name="Cells",
        channels=["FSC-A", "SSC-A"],
        vertices=[[40000, 15000], [180000, 15000], [180000, 140000], [40000, 140000]],
    ))
    hierarchy.add_gate(RangeGate(name="Viable", channel="7AAD-A", max_val=0.45, parent_gate="Cells"))
    hierarchy.add_gate(RectangleGate(
        name="CD4_CD8",
        channels=["FITC-A", "PE-A"],
        x_min=0.4,
        x_max=0.95,
        y_min=0.1,
        y_max=0.5,
        parent_gate="Viable",
    ))
    hierarchy.add_gate(QuadrantGate(
        name="Quads",
        channels=["FITC-A", "PE-A"],
        x_cutoff=0.4,
        y_cutoff=0.4,
        parent_gate="Viable",
    ))

    out_gml = tmp_path / "preprocessing_gating_ml_2.0.xml"
    export_preprocessing_gating_ml(
        hierarchy=hierarchy,
        filepath=out_gml,
        compensation=comp,
        transforms=xforms,
    )

    gs = fk.parse_gating_xml(str(out_gml))
    gate_names = {gid[0] for gid in gs.get_gate_ids()}
    assert {"Cells", "Viable", "CD4_CD8", "Quads"}.issubset(gate_names)
    assert "FlowX_Comp" in gs.comp_matrices
    assert len(gs.transformations) >= 1


def test_export_preprocessing_wsp_exact_gate_parity_with_flowkit(tmp_path: Path):
    """Verify preprocessing_gating.wsp & Gating-ML 2.0 reproduce exact FlowX gate counts in FlowKit."""
    import flowkit as fk
    import numpy as np
    import pandas as pd
    from flowx.io.parser import FCSData, write_fcs
    from flowx.math.compensation import CompensationMatrix
    from flowx.math.transforms import LogicleTransform, apply_transform_to_dataframe
    from flowx.workspace.gating_ml_export import (
        SampleWorkspaceSpec,
        export_preprocessing_flowjo_wsp,
        export_preprocessing_gating_ml,
    )

    rng = np.random.default_rng(7)
    n = 400
    fsc_a = np.concatenate([rng.normal(100000, 12000, 320), rng.normal(15000, 3000, 80)])
    ssc_a = np.concatenate([rng.normal(50000, 8000, 320), rng.normal(8000, 2000, 80)])
    fsc_h = fsc_a * 0.98 + rng.normal(0, 1500, n)
    fitc_true = np.concatenate([rng.normal(25000, 4000, 180), rng.normal(100, 50, 220)])
    viab_true = np.concatenate([rng.normal(80, 30, 300), rng.normal(30000, 4000, 100)])

    spill = np.array([[1.0, 0.08], [0.02, 1.0]], dtype=np.float64)
    raw_fl = np.column_stack([fitc_true, viab_true]) @ spill
    raw_df = pd.DataFrame({
        "FSC-A": np.clip(fsc_a, 100.0, 250000.0),
        "FSC-H": np.clip(fsc_h, 100.0, 250000.0),
        "SSC-A": np.clip(ssc_a, 100.0, 250000.0),
        "FITC-A": raw_fl[:, 0],
        "7AAD-A": raw_fl[:, 1],
    })

    raw_fcs_path = tmp_path / "Sample_A.fcs"
    write_fcs(raw_fcs_path, raw_df, pnn_labels=list(raw_df.columns))

    comp = CompensationMatrix(detectors=["FITC-A", "7AAD-A"], matrix=spill)
    df_comp = comp.apply_dataframe(raw_df)
    lt = LogicleTransform(param_t=262144.0, param_w=0.5, param_m=4.5, param_a=0.0)
    df_xf = apply_transform_to_dataframe(df_comp, ["FITC-A", "7AAD-A"], lt)

    hierarchy = GatingHierarchy()
    hierarchy.add_gate(PolygonGate(
        name="Cells",
        channels=["FSC-A", "SSC-A"],
        vertices=[[50000, 20000], [160000, 20000], [160000, 90000], [50000, 90000]],
    ))
    hierarchy.add_gate(PolygonGate(
        name="Singlets",
        channels=["FSC-A", "FSC-H"],
        vertices=[[50000, 40000], [160000, 145000], [160000, 165000], [50000, 58000]],
        parent_gate="Cells",
    ))
    hierarchy.add_gate(RangeGate(
        name="Viable",
        channel="7AAD-A",
        max_val=0.45,
        parent_gate="Singlets",
    ))
    hierarchy.add_gate(RangeGate(
        name="CD3+",
        channel="FITC-A",
        min_val=0.50,
        parent_gate="Viable",
    ))

    masks = hierarchy.evaluate(df_xf)
    counts = {k: int(np.sum(m)) for k, m in masks.items() if k != "root"}
    assert counts["Cells"] > 200
    assert counts["CD3+"] > 50

    wsp_path = tmp_path / "preprocessing_gating.wsp"
    gml_path = tmp_path / "preprocessing_gating_ml_2.0.xml"
    xforms = {"FITC-A": lt, "7AAD-A": lt}

    export_preprocessing_flowjo_wsp(
        samples=[
            SampleWorkspaceSpec(
                sample_filename="Sample_A.fcs",
                sample_path="Sample_A.fcs",
                total_events=n,
                channels=list(raw_df.columns),
                max_range=262144.0,
                compensation=comp,
                transforms=xforms,
                hierarchy=hierarchy,
                gate_counts=counts,
            )
        ],
        filepath=wsp_path,
    )
    export_preprocessing_gating_ml(
        hierarchy=hierarchy,
        filepath=gml_path,
        compensation=comp,
        transforms=xforms,
    )

    # 1. Verify FlowKit parses the .wsp cleanly
    wsp_parsed = fk.parse_wsp(str(wsp_path))
    assert "Sample_A.fcs" in wsp_parsed["samples"]
    s_entry = wsp_parsed["samples"]["Sample_A.fcs"]
    assert s_entry["compensation"] is not None
    assert "FITC-A" in s_entry["transforms"]

    # 2. Evaluate exported Gating-ML 2.0 on the raw FCS via FlowKit GatingStrategy
    fk_sample = fk.Sample(str(raw_fcs_path))
    fk_gs = fk.parse_gating_xml(str(gml_path))
    fk_res = fk_gs.gate_sample(fk_sample)
    for g_name, expected_cnt in counts.items():
        gml_gate_id = "CD3_pos" if g_name == "CD3+" else g_name
        fk_cnt = fk_res.get_gate_count(gml_gate_id)
        assert abs(fk_cnt - expected_cnt) <= 1, f"Gate {g_name}: FlowKit={fk_cnt} vs FlowX={expected_cnt}"

    # 3. Evaluate exported FlowJo .wsp on the raw FCS via FlowKit Workspace
    fk_ws = fk.Workspace(str(wsp_path), fcs_samples=[fk_sample])
    fk_ws.analyze_samples(sample_id="Sample_A.fcs")
    ws_res = fk_ws.get_gating_results("Sample_A.fcs")
    for g_name, expected_cnt in counts.items():
        ws_cnt = ws_res.get_gate_count(g_name)
        assert abs(ws_cnt - expected_cnt) <= 1, f"WSP Gate {g_name}: FlowKit={ws_cnt} vs FlowX={expected_cnt}"


@pytest.mark.parametrize("xform_kind", ["asinh", "biex"])
def test_export_preprocessing_non_logicle_transforms_parity(tmp_path: Path, xform_kind: str):
    """Verify Asinh & Biexponential inverse-coordinate export matches FlowKit on raw FCS."""
    import flowkit as fk
    import numpy as np
    import pandas as pd
    from flowx.io.parser import write_fcs
    from flowx.math.compensation import CompensationMatrix
    from flowx.math.transforms import (
        AsinhTransform,
        BiexponentialTransform,
        apply_transform_to_dataframe,
    )
    from flowx.workspace.gating_ml_export import (
        SampleWorkspaceSpec,
        export_preprocessing_flowjo_wsp,
        export_preprocessing_gating_ml,
    )

    rng = np.random.default_rng(19)
    n = 250
    viab_true = np.concatenate([rng.normal(100, 40, 180), rng.normal(25000, 3000, 70)])
    spill = np.array([[1.0, 0.04], [0.01, 1.0]], dtype=np.float64)
    fitc_true = rng.normal(5000, 1000, n)
    raw_fl = np.column_stack([fitc_true, viab_true]) @ spill
    raw_df = pd.DataFrame({
        "FSC-A": rng.uniform(50000, 150000, n),
        "SSC-A": rng.uniform(25000, 80000, n),
        "FITC-A": raw_fl[:, 0],
        "7AAD-A": raw_fl[:, 1],
    })
    raw_fcs_path = tmp_path / f"sample_{xform_kind}.fcs"
    write_fcs(raw_fcs_path, raw_df, pnn_labels=list(raw_df.columns))

    comp = CompensationMatrix(detectors=["FITC-A", "7AAD-A"], matrix=spill)
    df_comp = comp.apply_dataframe(raw_df)
    xf = (
        AsinhTransform(cofactor=150.0, normalized=True)
        if xform_kind == "asinh"
        else BiexponentialTransform(normalized=True)
    )
    df_xf = apply_transform_to_dataframe(df_comp, ["FITC-A", "7AAD-A"], xf)

    hierarchy = GatingHierarchy()
    hierarchy.add_gate(RangeGate(name="Viable", channel="7AAD-A", max_val=0.45))
    masks = hierarchy.evaluate(df_xf)
    counts = {"Viable": int(np.sum(masks["Viable"]))}

    wsp_path = tmp_path / f"{xform_kind}.wsp"
    gml_path = tmp_path / f"{xform_kind}_gml.xml"
    xforms = {"FITC-A": xf, "7AAD-A": xf}

    export_preprocessing_flowjo_wsp(
        samples=[
            SampleWorkspaceSpec(
                sample_filename=raw_fcs_path.name,
                sample_path=raw_fcs_path.name,
                total_events=n,
                channels=list(raw_df.columns),
                compensation=comp,
                transforms=xforms,
                hierarchy=hierarchy,
                gate_counts=counts,
            )
        ],
        filepath=wsp_path,
    )
    export_preprocessing_gating_ml(
        hierarchy=hierarchy,
        filepath=gml_path,
        compensation=comp,
        transforms=xforms,
    )

    fk_sample = fk.Sample(str(raw_fcs_path))
    fk_gs = fk.parse_gating_xml(str(gml_path))
    fk_cnt = fk_gs.gate_sample(fk_sample).get_gate_count("Viable")
    assert abs(fk_cnt - counts["Viable"]) <= 2

    fk_ws = fk.Workspace(str(wsp_path), fcs_samples=[fk_sample])
    fk_ws.analyze_samples(sample_id=raw_fcs_path.name)
    ws_cnt = fk_ws.get_gating_results(raw_fcs_path.name).get_gate_count("Viable")
    assert abs(ws_cnt - counts["Viable"]) <= 2


def test_pipeline_runner_exports_preprocessing_workspaces(tmp_path: Path):
    """Verify FlowCytometryPipeline exports preprocessing .wsp and Gating-ML 2.0 in cohort mode."""
    import flowkit as fk
    import numpy as np
    import pandas as pd
    from flowx.io.parser import write_fcs
    from flowx.pipeline import FlowCytometryPipeline

    rng = np.random.default_rng(123)
    sample_paths = []
    for s_idx, name in enumerate(["Ctrl_1.fcs", "Treat_1.fcs"]):
        n = 300
        fsc_a = rng.normal(100000, 12000, n)
        fsc_h = fsc_a * 0.98 + rng.normal(0, 1000, n)
        ssc_a = rng.normal(50000, 8000, n)
        df = pd.DataFrame({
            "FSC-A": np.clip(fsc_a, 1000.0, 240000.0),
            "FSC-H": np.clip(fsc_h, 1000.0, 240000.0),
            "SSC-A": np.clip(ssc_a, 1000.0, 240000.0),
            "CD3-A": rng.normal(25000 if s_idx == 0 else 30000, 4000, n),
            "CD4-A": rng.normal(15000, 3000, n),
            "7AAD-A": np.concatenate([rng.normal(80, 30, 240), rng.normal(25000, 3000, 60)]),
            "Time": np.linspace(0.0, 30.0, n),
        })
        fp = tmp_path / name
        write_fcs(fp, df, pnn_labels=list(df.columns))
        sample_paths.append(fp)

    out_dir = tmp_path / "run_out"
    runner = FlowCytometryPipeline(
        output_dir=out_dir,
        timestamp_dir=False,
        overwrite=True,
        enable_qc=True,
        enable_time_qc=False,
        enable_cleanup_gating=True,
        enable_compensation=False,
        enable_discovery=False,
    )
    res = runner.run(sample_paths)

    prep_wsp = Path(res.preprocessing_wsp_path)
    prep_gml = Path(res.preprocessing_gating_ml_path)
    assert prep_wsp.is_file()
    assert prep_gml.is_file()
    assert prep_wsp.name == "preprocessing_gating.wsp"
    assert prep_gml.name == "preprocessing_gating_ml_2.0.xml"

    wsp_parsed = fk.parse_wsp(str(prep_wsp))
    assert set(wsp_parsed["samples"].keys()) == {"Ctrl_1.fcs", "Treat_1.fcs"}
    assert set(wsp_parsed["groups"]["All Samples"]["samples"]) == {"Ctrl_1.fcs", "Treat_1.fcs"}
    fk_gs = fk.parse_gating_xml(str(prep_gml))
    assert any(g[0] == "Cells" for g in fk_gs.get_gate_ids())


def test_preprocessing_wsp_relative_uri_loads_in_flowkit_without_explicit_fcs(tmp_path: Path):
    """Verify preprocessing_gating.wsp resolves relative DataSet URIs and excludes non-raw unmixed transforms."""
    import flowkit as fk
    import numpy as np
    import pandas as pd
    from flowx.io.parser import write_fcs
    from flowx.math.transforms import AsinhTransform
    from flowx.workspace.gating_ml_export import SampleWorkspaceSpec, export_preprocessing_flowjo_wsp

    rng = np.random.default_rng(7)
    n = 150
    df = pd.DataFrame({
        "FSC-A": rng.uniform(40000, 160000, n),
        "SSC-A": rng.uniform(15000, 90000, n),
    })
    fcs_path = tmp_path / "data" / "raw_sample.fcs"
    fcs_path.parent.mkdir(parents=True, exist_ok=True)
    write_fcs(fcs_path, df, pnn_labels=["FSC-A", "SSC-A"])

    wsp_path = tmp_path / "exports" / "workspaces" / "preprocessing_gating.wsp"
    hierarchy = GatingHierarchy()
    hierarchy.add_gate(PolygonGate(
        name="Cells",
        channels=["FSC-A", "SSC-A"],
        vertices=[[50000, 20000], [150000, 20000], [150000, 80000], [50000, 80000]],
    ))

    # Include an unmixed channel transform that does not exist in raw FCS channels
    xform = AsinhTransform(cofactor=150.0)
    spec = SampleWorkspaceSpec(
        sample_filename="raw_sample.fcs",
        sample_path=str(fcs_path),
        total_events=n,
        channels=["FSC-A", "SSC-A"],
        transforms={"Unmixed_AF488": xform},
        hierarchy=hierarchy,
        gate_counts={"Cells": 10},
    )
    export_preprocessing_flowjo_wsp([spec], wsp_path)

    wsp_text = wsp_path.read_text(encoding="utf-8")
    assert 'uri="../../data/raw_sample.fcs"' in wsp_text
    assert "Unmixed_AF488" not in wsp_text

    parsed = fk.parse_wsp(str(wsp_path))
    assert parsed["groups"]["All Samples"]["samples"] == ["raw_sample.fcs"]

    fk_ws = fk.Workspace(str(wsp_path), find_fcs_files_from_wsp=True)
    assert fk_ws.get_sample_ids() == ["raw_sample.fcs"]
    fk_ws.analyze_samples(sample_id="raw_sample.fcs")
    res = fk_ws.get_gating_results(sample_id="raw_sample.fcs")
    assert res.get_gate_count("Cells") > 0


def test_cohort_augmented_fcs_workspace_gating_parity(tmp_path: Path):
    """Verify augmented FCS FlowJo .wsp and Gating-ML 2.0 isolate each cluster with exact counts."""
    import flowkit as fk
    import numpy as np
    import pandas as pd
    from flowx.clustering.cohort_runner import run_cohort_population_analysis

    rng = np.random.default_rng(11)
    n_per_sample = 120
    dfs = {}
    for s_id in ["S1", "S2"]:
        c1 = rng.normal(loc=1.0, scale=0.2, size=(n_per_sample // 2, 2))
        c2 = rng.normal(loc=5.0, scale=0.2, size=(n_per_sample // 2, 2))
        arr = np.vstack([c1, c2])
        dfs[s_id] = pd.DataFrame({"MarkerA": arr[:, 0], "MarkerB": arr[:, 1]})
    meta_df = pd.DataFrame({"condition": ["Ctrl", "Treat"]}, index=["S1", "S2"])

    res = run_cohort_population_analysis(
        cohort_dfs=dfs,
        sample_metadata=meta_df,
        markers=["MarkerA", "MarkerB"],
        clustering_method="som",
        n_clusters=3,
        export_augmented_fcs=True,
        export_flowjo_wsp=True,
        export_gating_ml=True,
        generate_miflowcyt_report=False,
        output_dir=tmp_path,
        numbered_steps=False,
        generate_plots=False,
        random_seed=42,
    )

    wsp_path = Path(res.generated_tables["flowjo_workspace"])
    gml_path = Path(res.generated_tables["gating_ml"])
    assert wsp_path.is_file()
    assert gml_path.is_file()

    fk_ws = fk.Workspace(str(wsp_path), find_fcs_files_from_wsp=True)
    assert set(fk_ws.get_sample_ids()) == {"augmented_S1.fcs", "augmented_S2.fcs"}

    fk_ws.analyze_samples()
    # Build expected cid -> gate_name mapping matching cohort_runner
    cid_to_lbl = {}
    for df in res.labeled_dfs.values():
        for raw_cid in df["cluster_id"].unique():
            cid = int(raw_cid)
            if cid not in cid_to_lbl:
                cid_to_lbl[cid] = str(df.loc[df["cluster_id"] == raw_cid, "lineage_id"].mode().iloc[0])
    lbl_counts = {}
    for lbl in cid_to_lbl.values():
        lbl_counts[lbl] = lbl_counts.get(lbl, 0) + 1
    cid_to_gate = {cid: (f"{lbl}_C{cid}" if lbl_counts[lbl] > 1 else lbl) for cid, lbl in cid_to_lbl.items()}

    for s_id, df in res.labeled_dfs.items():
        aug_name = f"augmented_{s_id}.fcs"
        g_res = fk_ws.get_gating_results(sample_id=aug_name)
        for cid in sorted(df["cluster_id"].unique()):
            expected = int((df["cluster_id"] == cid).sum())
            gate_name = cid_to_gate[int(cid)]
            assert g_res.get_gate_count(gate_name) == expected

    fk_gs = fk.parse_gating_xml(str(gml_path))
    assert len(fk_gs.get_gate_ids()) == len(cid_to_gate)




