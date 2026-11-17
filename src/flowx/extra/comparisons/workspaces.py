"""Parsers and geometric helpers for external cytometry workspace formats.

Supports extracting polygon/ellipse gates and compensation matrices from
external Ground Truth exports (FlowJo ``.xml`` / ``.wsp`` and BD FACSDiva
``.xml``) as well as loading FlowX augmented ``.fcs`` outputs.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import flowkit as fk
import numpy as np
import pandas as pd

from flowx.math.transforms import apply_transform_to_dataframe

logger = logging.getLogger(__name__)


@dataclass
class ExternalGateDefinition:
    """Normalized representation of a gate parsed from an external workspace.

    Attributes:
        name: Population or region name inside the external workspace.
        gate_type: Geometry type (``'polygon'``, ``'polyrect'``, ``'ellipse'``, ``'diva_region'``).
        vertices: Ordered list of ``(x, y)`` polygon or control vertices.
        x_param: Optional detector/channel name for the horizontal axis.
        y_param: Optional detector/channel name for the vertical axis.
        count: Optional reported event count stored in the workspace XML.
    """

    name: str
    gate_type: str
    vertices: List[Tuple[float, float]]
    x_param: Optional[str] = None
    y_param: Optional[str] = None
    count: Optional[int] = None


@dataclass
class FlowJoSampleGates:
    """All gates belonging to a single ``<SampleNode>`` in a FlowJo workspace.

    Attributes:
        sample_id: Workspace ``sampleID`` attribute.
        sample_name: Original FCS filename stored on ``SampleNode``.
        root_count: Total ungated event count stored on ``SampleNode`` (if present).
        gates: Dictionary mapping population name to :class:`ExternalGateDefinition`.
    """

    sample_id: str
    sample_name: str
    root_count: Optional[int]
    gates: Dict[str, ExternalGateDefinition]


@dataclass
class DivaTubeGates:
    """All regions belonging to a single ``<tube>`` in a BD FACSDiva workspace.

    Attributes:
        specimen_name: Enclosing ``<specimen name="...">`` attribute (empty if absent).
        tube_name: Target ``<tube name="...">`` attribute.
        full_name: Canonical ``"{specimen_name}_{tube_name}"`` key when specimen exists.
        gates: Dictionary mapping region name (e.g., ``'P1'``) to :class:`ExternalGateDefinition`.
    """

    specimen_name: str
    tube_name: str
    full_name: str
    gates: Dict[str, ExternalGateDefinition]


def _parse_xml_root(xml_path: Union[str, Path], format_label: str) -> ET.Element:
    """Safely load and parse an XML workspace root element."""
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"{format_label} file not found: {path}")
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse {format_label} '{path}': {exc}") from exc


def load_augmented_fcs(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load an exported FlowX augmented FCS file preserving clean channel names.

    Args:
        filepath: Path to an ``augmented_*.fcs`` file.

    Returns:
        DataFrame indexed sequentially with flattened detector/metadata channel names.

    Raises:
        FileNotFoundError: If the FCS file does not exist.
        RuntimeError: If parsing via FlowKit fails.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Augmented FCS file not found: {path}")

    try:
        sample = fk.Sample(str(path))
        df = sample.as_dataframe(source="raw")
        df.columns = [col[0] if isinstance(col, tuple) else str(col) for col in df.columns]
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to load augmented FCS file '{path}': {exc}") from exc


def ellipse_vertices_to_polygon(pts: np.ndarray, n_pts: int = 120) -> np.ndarray:
    """Reconstruct a smooth closed polygon boundary from FlowJo 4-vertex ellipses.

    Args:
        pts: Array of shape ``(4, 2)`` containing opposite axis endpoint pairs
            ``(pts[0], pts[1])`` and ``(pts[2], pts[3])``.
        n_pts: Number of angular segments along the reconstructed perimeter.

    Returns:
        Array of shape ``(n_pts, 2)`` tracing the closed ellipse boundary.

    Raises:
        ValueError: If fewer than 4 vertices are provided.
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    if pts_arr.shape[0] < 4 or pts_arr.shape[1] != 2:
        raise ValueError(f"Expected (4, 2) ellipse vertices, got shape {pts_arr.shape}")

    center = np.mean(pts_arr[:4], axis=0)
    a_vec = (pts_arr[1] - pts_arr[0]) / 2.0
    b_vec = (pts_arr[3] - pts_arr[2]) / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=True)
    return center + np.outer(np.cos(theta), a_vec) + np.outer(np.sin(theta), b_vec)


def _locate_flowjo_sample_node(
    root: ET.Element,
    sample_id: Optional[str] = None,
    sample_name: Optional[str] = None,
) -> ET.Element:
    """Resolve a target ``SampleNode`` element in a parsed FlowJo XML tree."""
    if sample_id is not None:
        node = root.find(f'.//SampleNode[@sampleID="{sample_id}"]')
        if node is not None:
            return node
        raise KeyError(f"FlowJo SampleNode with sampleID='{sample_id}' not found.")

    if sample_name is not None:
        for node in root.findall(".//SampleNode"):
            if node.attrib.get("name") == sample_name:
                return node
        raise KeyError(f"FlowJo SampleNode with name='{sample_name}' not found.")

    node = root.find(".//SampleNode")
    if node is None:
        raise KeyError("No SampleNode elements found in FlowJo XML workspace.")
    return node


def _local_tag(tag: str) -> str:
    """Strip XML namespace prefix from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _attr_ci(elem: ET.Element, key: str) -> Optional[str]:
    """Lookup an attribute by exact name or local name (ignoring XML namespace)."""
    if key in elem.attrib:
        return elem.attrib[key]
    for k, v in elem.attrib.items():
        if _local_tag(k) == key:
            return v
    return None


def _extract_flowjo_gates_from_node(sample_node: ET.Element) -> Dict[str, ExternalGateDefinition]:
    """Extract PolygonGate and RectangleGate definitions from a single FlowJo ``SampleNode``."""
    gates: Dict[str, ExternalGateDefinition] = {}
    for pop in sample_node.findall(".//Population"):
        pop_name = pop.attrib.get("name")
        if not pop_name:
            continue

        gate_container = pop.find("./Gate")
        search_root = gate_container if gate_container is not None else pop

        gate_elem = None
        gate_kind = None
        for child in search_root:
            ltag = _local_tag(child.tag)
            if ltag in ("PolygonGate", "RectangleGate"):
                gate_elem = child
                gate_kind = ltag
                break

        if gate_elem is None:
            continue

        verts: List[Tuple[float, float]] = []
        geom_type = "polygon"
        x_param: Optional[str] = None
        y_param: Optional[str] = None

        if gate_kind == "PolygonGate":
            # Check legacy FlowJo child tags first (PolyRect, Polygon, Ellipse)
            geom = None
            for tag_name, mapped_type in (("PolyRect", "polyrect"), ("Ellipse", "ellipse"), ("Polygon", "polygon")):
                candidate = gate_elem.find(f".//{tag_name}")
                if candidate is not None:
                    geom = candidate
                    geom_type = mapped_type
                    break

            if geom is not None:
                for vertex in geom.findall(".//Vertex"):
                    x_val = vertex.attrib.get("x")
                    y_val = vertex.attrib.get("y")
                    if x_val is not None and y_val is not None:
                        verts.append((float(x_val), float(y_val)))
                dims = gate_elem.findall(".//Dimension")
                x_param = dims[0].attrib.get("name") if len(dims) >= 1 else None
                y_param = dims[1].attrib.get("name") if len(dims) >= 2 else None
                if (x_param is None or y_param is None) and geom is not None:
                    raw_x = geom.attrib.get("xAxisName")
                    raw_y = geom.attrib.get("yAxisName")
                    if x_param is None and raw_x:
                        x_param = raw_x.strip("<>")
                    if y_param is None and raw_y:
                        y_param = raw_y.strip("<>")
            else:
                # Gating-ML 2.0 namespaced PolygonGate (<gating:dimension>, <gating:vertex>)
                dim_names: List[str] = []
                for elem in gate_elem:
                    ltag = _local_tag(elem.tag)
                    if ltag == "dimension":
                        for sub in elem:
                            if _local_tag(sub.tag) == "fcs-dimension":
                                dname = _attr_ci(sub, "name")
                                if dname:
                                    dim_names.append(dname)
                    elif ltag == "vertex":
                        coords: List[float] = []
                        for sub in elem:
                            if _local_tag(sub.tag) == "coordinate":
                                val = _attr_ci(sub, "value")
                                if val is not None:
                                    coords.append(float(val))
                        if len(coords) >= 2:
                            verts.append((coords[0], coords[1]))
                x_param = dim_names[0] if len(dim_names) >= 1 else None
                y_param = dim_names[1] if len(dim_names) >= 2 else None
                geom_type = "polygon"
        elif gate_kind == "RectangleGate":
            geom_type = "rectangle"
            dim_bounds: List[Tuple[Optional[str], float, float]] = []
            for elem in gate_elem:
                if _local_tag(elem.tag) == "dimension":
                    dname: Optional[str] = None
                    for sub in elem:
                        if _local_tag(sub.tag) == "fcs-dimension":
                            dname = _attr_ci(sub, "name")
                    dmin_raw = _attr_ci(elem, "min")
                    dmax_raw = _attr_ci(elem, "max")
                    dmin = float(dmin_raw) if dmin_raw is not None else float("-inf")
                    dmax = float(dmax_raw) if dmax_raw is not None else float("inf")
                    dim_bounds.append((dname, dmin, dmax))
            if len(dim_bounds) >= 2:
                x_param, x0, x1 = dim_bounds[0]
                y_param, y0, y1 = dim_bounds[1]
                verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            elif len(dim_bounds) == 1:
                x_param, x0, x1 = dim_bounds[0]
                verts = [(x0, 0.0), (x1, 0.0)]

        if not verts:
            continue

        count_attr = pop.attrib.get("count")
        count_val = int(count_attr) if (count_attr is not None and count_attr.lstrip("-").isdigit()) else None

        key = pop_name
        if key in gates:
            suffix = f" ({x_param} vs {y_param})" if (x_param and y_param) else f" #{len(gates)}"
            key = f"{pop_name}{suffix}"
            idx = 2
            while key in gates:
                key = f"{pop_name}{suffix}_{idx}"
                idx += 1

        gates[key] = ExternalGateDefinition(
            name=pop_name,
            gate_type=geom_type,
            vertices=verts,
            x_param=x_param,
            y_param=y_param,
            count=count_val,
        )

    return gates


def parse_flowjo_xml_gates(
    xml_path: Union[str, Path],
    sample_id: Optional[str] = None,
    sample_name: Optional[str] = None,
) -> Dict[str, ExternalGateDefinition]:
    """Parse population gates from a single ``SampleNode`` in a FlowJo ``.xml``/``.wsp`` workspace."""
    root = _parse_xml_root(xml_path, "FlowJo workspace file")
    sample_node = _locate_flowjo_sample_node(root, sample_id=sample_id, sample_name=sample_name)
    return _extract_flowjo_gates_from_node(sample_node)


def parse_all_flowjo_samples(xml_path: Union[str, Path]) -> List[FlowJoSampleGates]:
    """Extract gating hierarchies for every ``<SampleNode>`` in a FlowJo workspace.

    Args:
        xml_path: Path to FlowJo ``.xml`` or ``.wsp`` workspace file.

    Returns:
        Ordered list of :class:`FlowJoSampleGates` covering all samples in the file.
    """
    root = _parse_xml_root(xml_path, "FlowJo workspace file")
    records: List[FlowJoSampleGates] = []
    for idx, node in enumerate(root.findall(".//SampleNode")):
        sid = str(node.attrib.get("sampleID") or f"idx_{idx}")
        sname = str(node.attrib.get("name") or sid)
        cnt_raw = node.attrib.get("count")
        root_cnt = int(cnt_raw) if (cnt_raw is not None and cnt_raw.lstrip("-").isdigit()) else None
        records.append(
            FlowJoSampleGates(
                sample_id=sid,
                sample_name=sname,
                root_count=root_cnt,
                gates=_extract_flowjo_gates_from_node(node),
            )
        )
    return records


def match_flowjo_sample_gates(
    xml_path: Union[str, Path],
    sample_event_counts: Dict[str, int],
    explicit_sample_ids: Optional[Dict[str, str]] = None,
) -> Dict[str, FlowJoSampleGates]:
    """Deterministically match cohort sample IDs to FlowJo ``SampleNode`` records.

    Handles workspaces containing replicate batch entries sharing the same FCS filename
    by selecting the node whose stored ``root_count`` matches ``sample_event_counts``.

    Args:
        xml_path: Path to FlowJo ``.xml`` or ``.wsp`` workspace file.
        sample_event_counts: Mapping of sample stem (e.g. ``'CytoTrol_CytoTrol_1'``) to raw event count.
        explicit_sample_ids: Optional overrides mapping sample stem -> explicit XML ``sampleID``.

    Returns:
        Dictionary mapping each matched sample stem to its :class:`FlowJoSampleGates`.
    """
    all_nodes = parse_all_flowjo_samples(xml_path)
    by_id = {rec.sample_id: rec for rec in all_nodes}
    matched: Dict[str, FlowJoSampleGates] = {}

    for stem, raw_cnt in sample_event_counts.items():
        if explicit_sample_ids and stem in explicit_sample_ids:
            override_id = str(explicit_sample_ids[stem])
            if override_id in by_id:
                matched[stem] = by_id[override_id]
                continue

        candidates = [
            rec
            for rec in all_nodes
            if Path(rec.sample_name).stem == stem or rec.sample_name == stem
        ]
        if not candidates:
            logger.warning("No FlowJo SampleNode matched sample stem '%s'.", stem)
            continue

        if len(candidates) == 1:
            matched[stem] = candidates[0]
            continue

        with_cnt = [c for c in candidates if c.root_count is not None]
        if with_cnt:
            best = min(with_cnt, key=lambda c: abs(int(c.root_count or 0) - raw_cnt))
            matched[stem] = best
            logger.info(
                "Disambiguated FlowJo sample '%s' (%d candidates) -> sampleID=%s (count=%s vs raw=%d).",
                stem,
                len(candidates),
                best.sample_id,
                best.root_count,
                raw_cnt,
            )
        else:
            matched[stem] = candidates[0]

    return matched


def evaluate_flowjo_ellipse_mask(
    df_transformed: pd.DataFrame,
    x_col: str,
    y_col: str,
    vertices: Sequence[Tuple[float, float]],
    transform: Optional[Any] = None,
    linear_channels: Sequence[str] = ("FSC-A", "SSC-A", "FSC-H", "SSC-H", "FSC-W", "SSC-W"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate exact FlowJo 4-vertex ellipse membership in transformed coordinate space.

    Args:
        df_transformed: Event DataFrame whose fluorescence channels have already been
            transformed consistently with ``transform``.
        x_col: Horizontal channel column name.
        y_col: Vertical channel column name.
        vertices: Raw (linear-space) 4-vertex FlowJo ellipse coordinates.
        transform: Optional transform object passed to ``apply_transform_to_dataframe``
            for non-scatter axes of ``vertices``.
        linear_channels: Channel names that should NOT undergo logicle/biex transformation.

    Returns:
        Tuple of ``(boolean_mask, polygon_vertices_transformed)``.
    """
    if len(vertices) < 4:
        raise ValueError(f"FlowJo ellipse requires 4 axis vertices; got {len(vertices)}.")

    vdf = pd.DataFrame({
        x_col: [float(v[0]) for v in vertices[:4]],
        y_col: [float(v[1]) for v in vertices[:4]],
    })
    cols_to_transform = [c for c in (x_col, y_col) if c not in linear_channels]
    if transform is not None and cols_to_transform:
        vtr = apply_transform_to_dataframe(vdf, channels=cols_to_transform, transform=transform)
    else:
        vtr = vdf

    pts_v = vtr[[x_col, y_col]].to_numpy(dtype=np.float64)
    center = np.mean(pts_v, axis=0)
    axes_mat = np.column_stack([(pts_v[1] - pts_v[0]) / 2.0, (pts_v[3] - pts_v[2]) / 2.0])
    centered_events = df_transformed[[x_col, y_col]].to_numpy(dtype=np.float64) - center
    unit_coords = centered_events @ np.linalg.inv(axes_mat).T
    mask = (unit_coords[:, 0] ** 2 + unit_coords[:, 1] ** 2) <= 1.0
    return mask, ellipse_vertices_to_polygon(pts_v)


def _locate_diva_tube(root: ET.Element, tube_name: str) -> ET.Element:
    """Locate a target ``<tube name="...">`` node inside a BD FACSDiva XML export."""
    tube = root.find(f'.//tube[@name="{tube_name}"]')
    if tube is None:
        raise KeyError(f"BD FACSDiva tube '{tube_name}' not found in XML.")
    return tube


def _extract_diva_gates_from_tube(tube: ET.Element) -> Dict[str, ExternalGateDefinition]:
    """Extract all region polygon definitions from a BD FACSDiva ``<tube>`` element."""
    regions: Dict[str, ExternalGateDefinition] = {}
    for reg in tube.findall(".//region"):
        reg_name = reg.attrib.get("name")
        if not reg_name:
            continue
        pts = [
            (float(pt.attrib["x"]), float(pt.attrib["y"]))
            for pt in reg.findall("./points/point")
            if "x" in pt.attrib and "y" in pt.attrib
        ]
        if not pts:
            continue
        regions[reg_name] = ExternalGateDefinition(
            name=reg_name,
            gate_type="diva_region",
            vertices=pts,
            x_param=reg.attrib.get("xparm"),
            y_param=reg.attrib.get("yparm"),
        )
    return regions


def parse_diva_xml_gates(
    xml_path: Union[str, Path],
    tube_name: str,
) -> Dict[str, ExternalGateDefinition]:
    """Parse BD FACSDiva XML region polygons for a specified tube."""
    root = _parse_xml_root(xml_path, "BD FACSDiva XML file")
    tube = _locate_diva_tube(root, tube_name)
    return _extract_diva_gates_from_tube(tube)


def parse_all_diva_tubes(xml_path: Union[str, Path]) -> List[DivaTubeGates]:
    """Extract region gates for every ``<tube>`` across all specimens in a FACSDiva XML."""
    root = _parse_xml_root(xml_path, "BD FACSDiva XML file")
    results: List[DivaTubeGates] = []
    seen_ids: set[int] = set()

    for specimen in root.findall(".//specimen"):
        sp_name = str(specimen.attrib.get("name") or "")
        for tube in specimen.findall(".//tube"):
            seen_ids.add(id(tube))
            tname = str(tube.attrib.get("name") or "")
            if not tname:
                continue
            full_name = f"{sp_name}_{tname}" if sp_name else tname
            results.append(
                DivaTubeGates(
                    specimen_name=sp_name,
                    tube_name=tname,
                    full_name=full_name,
                    gates=_extract_diva_gates_from_tube(tube),
                )
            )

    for tube in root.findall(".//tube"):
        if id(tube) in seen_ids:
            continue
        tname = str(tube.attrib.get("name") or "")
        if not tname:
            continue
        results.append(
            DivaTubeGates(
                specimen_name="",
                tube_name=tname,
                full_name=tname,
                gates=_extract_diva_gates_from_tube(tube),
            )
        )

    return results


def match_diva_tubes(
    xml_path: Union[str, Path],
    sample_stems: Sequence[str],
) -> Dict[str, DivaTubeGates]:
    """Match biological FCS stems (e.g. ``Specimen_001_01gx 11b``) to FACSDiva tubes."""
    all_tubes = parse_all_diva_tubes(xml_path)
    matched: Dict[str, DivaTubeGates] = {}

    for stem in sample_stems:
        clean_stem = Path(stem).stem if "." in stem else stem
        exact = [t for t in all_tubes if t.full_name == clean_stem or t.tube_name == clean_stem]
        if not exact:
            exact = [t for t in all_tubes if clean_stem.endswith(f"_{t.tube_name}")]
        if exact:
            matched[stem] = exact[0]
        else:
            logger.warning("No FACSDiva tube matched sample stem '%s'.", stem)
    return matched


def extract_diva_spillover_matrix(
    xml_path: Union[str, Path],
    tube_name: str,
    channel_names: Sequence[str],
) -> pd.DataFrame:
    """Extract and convert BD FACSDiva compensation matrix to donor-row spillover format."""
    root = _parse_xml_root(xml_path, "BD FACSDiva XML file")
    tube = _locate_diva_tube(root, tube_name)
    coeffs = [
        float(coef.text)
        for comp in tube.findall(".//compensation")
        for coef in comp.findall("compensation_coefficient")
        if coef.text is not None and coef.text.strip() != ""
    ]

    n_ch = len(channel_names)
    expected = n_ch * n_ch
    if len(coeffs) != expected:
        raise ValueError(
            f"Expected {expected} compensation coefficients for {n_ch} channels in tube '{tube_name}', "
            f"found {len(coeffs)}."
        )

    comp_mat = np.array(coeffs, dtype=np.float64).reshape((n_ch, n_ch))
    try:
        spill_mat = np.linalg.inv(comp_mat).T
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"Singular FACSDiva compensation matrix in tube '{tube_name}': {exc}") from exc

    labels = list(channel_names)
    return pd.DataFrame(spill_mat, index=labels, columns=labels)


def extract_flowjo_spillover_matrix(
    xml_path: Union[str, Path],
    matrix_name: Optional[str] = None,
) -> pd.DataFrame:
    """Extract a Gating-ML 2.0 ``spilloverMatrix`` from a FlowJo ``.wsp``/``.xml`` workspace.

    Args:
        xml_path: Path to FlowJo ``.wsp`` or ``.xml`` workspace file.
        matrix_name: Optional matrix ``name`` attribute filter (defaults to first matrix).

    Returns:
        Square donor-row spillover ``pd.DataFrame`` indexed and columned by parameter name.
    """
    root = _parse_xml_root(xml_path, "FlowJo workspace file")
    candidates = [
        elem
        for elem in root.iter()
        if _local_tag(elem.tag) == "spilloverMatrix"
    ]
    if not candidates:
        legacy_candidates = [
            elem
            for elem in root.iter()
            if _local_tag(elem.tag) == "CompensationMatrix"
        ]
        if not legacy_candidates:
            raise ValueError(f"No spilloverMatrix or CompensationMatrix element found in FlowJo workspace '{xml_path}'.")
        target = legacy_candidates[0]
        if matrix_name is not None:
            named = [
                m for m in legacy_candidates
                if matrix_name.lower() in (m.attrib.get("name") or "").lower()
            ]
            if not named:
                raise ValueError(
                    f"CompensationMatrix matching '{matrix_name}' not found in FlowJo workspace '{xml_path}'."
                )
            target = named[0]

        params: List[str] = []
        rows: Dict[str, Dict[str, float]] = {}
        for ch in target.findall("Channel"):
            raw_donor = ch.attrib.get("name", "")
            donor = raw_donor.strip("<>")
            if donor and donor not in params:
                params.append(donor)
            row_vals: Dict[str, float] = {}
            for cv in ch.findall("ChannelValue"):
                raw_rec = cv.attrib.get("name", "")
                rec = raw_rec.strip("<>")
                val = cv.attrib.get("value")
                if rec and val is not None:
                    row_vals[rec] = float(val)
            rows[donor] = row_vals

        if not params:
            params = list(rows.keys())
        clean_params = [p.replace("_", "/") if "/" not in p and "_" in p else p for p in params]
        mat = np.zeros((len(params), len(params)), dtype=np.float64)
        for i, donor in enumerate(params):
            row = rows.get(donor, {})
            for j, rec in enumerate(params):
                mat[i, j] = row.get(rec, 1.0 if i == j else 0.0)
        return pd.DataFrame(mat, index=clean_params, columns=clean_params)

    target = candidates[0]
    if matrix_name is not None:
        named = [m for m in candidates if m.attrib.get("name") == matrix_name]
        if not named:
            raise ValueError(
                f"Spillover matrix '{matrix_name}' not found in FlowJo workspace '{xml_path}'."
            )
        target = named[0]

    params: List[str] = []
    rows: Dict[str, Dict[str, float]] = {}
    for child in target:
        ltag = _local_tag(child.tag)
        if ltag == "parameters":
            for p in child:
                if _local_tag(p.tag) == "parameter":
                    pname = _attr_ci(p, "name")
                    if pname:
                        params.append(pname)
        elif ltag == "spillover":
            donor = _attr_ci(child, "parameter")
            if not donor:
                continue
            row_vals: Dict[str, float] = {}
            for coeff in child:
                if _local_tag(coeff.tag) == "coefficient":
                    rec = _attr_ci(coeff, "parameter")
                    val = _attr_ci(coeff, "value")
                    if rec and val is not None:
                        row_vals[rec] = float(val)
            rows[donor] = row_vals

    if not params:
        params = list(rows.keys())
    if not params:
        raise ValueError(f"Empty spilloverMatrix in FlowJo workspace '{xml_path}'.")

    # Normalize FlowJo underscore-substituted detector names (e.g. '730_45 Red-A' -> '730/45 Red-A')
    clean_params = [p.replace("_", "/") if "/" not in p and "_" in p else p for p in params]
    mat = np.zeros((len(params), len(params)), dtype=np.float64)
    for i, donor in enumerate(params):
        row = rows.get(donor, {})
        for j, rec in enumerate(params):
            mat[i, j] = row.get(rec, 1.0 if i == j else 0.0)

    return pd.DataFrame(mat, index=clean_params, columns=clean_params)
