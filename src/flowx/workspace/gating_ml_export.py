"""Gating-ML 2.0 and FlowJo (.wsp) XML workspace exporter.

Conforms to ISAC Gating-ML 2.0 standard specifications and provides
interoperable export for external verification in desktop FlowJo, Cytobank,
and FlowKit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import quote
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from flowx.gating.gate import (
    Gate,
    GatingHierarchy,
    PolygonGate,
    QuadrantGate,
    RangeGate,
    RectangleGate,
)
from flowx.io.paths import sanitize_path
from flowx.math.transforms import (
    AsinhTransform,
    BaseTransform,
    BiexponentialTransform,
    LogTransform,
    LogicleTransform,
    PreTransformedScaleTransform,
    create_transform,
)


GATING_ML_NS = "http://www.isac-net.org/std/Gating-ML/v2.0/gating"
DATATYPE_NS = "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"
TRANSFORMS_NS = "http://www.isac-net.org/std/Gating-ML/v2.0/transformations"


@dataclass
class SampleWorkspaceSpec:
    """Per-sample configuration used to build multi-sample FlowJo .wsp exports.

    Attributes:
        sample_filename: Base FCS filename (e.g. 'sample_001.fcs').
        sample_path: Sanitized relative path or URI to the FCS file.
        total_events: Total raw event count in the FCS file.
        channels: Ordered list of physical channel ($PnN) names.
        max_range: Detector dynamic range upper bound for linear scatter axes.
        compensation: Optional square CompensationMatrix.
        spectral_library: Optional SpectralLibrary for spectral unmixing runs.
        transforms: Channel name -> BaseTransform (or parameter dict) mapping.
        hierarchy: GatingHierarchy containing cleanup and/or targeted gates.
        gate_counts: Optional gate_name -> surviving event count mapping.
    """

    sample_filename: str
    sample_path: str = ""
    total_events: int = 0
    channels: List[str] = field(default_factory=list)
    max_range: float = 262144.0
    compensation: Optional[Any] = None
    spectral_library: Optional[Any] = None
    transforms: Dict[str, Any] = field(default_factory=dict)
    hierarchy: GatingHierarchy = field(default_factory=GatingHierarchy)
    gate_counts: Dict[str, int] = field(default_factory=dict)


def _register_gml_namespaces() -> None:
    """Register standard ISAC Gating-ML 2.0 XML namespace prefixes."""
    ET.register_namespace("gating", GATING_ML_NS)
    ET.register_namespace("data-type", DATATYPE_NS)
    ET.register_namespace("transforms", TRANSFORMS_NS)


def _prettify_xml(elem: ET.Element) -> str:
    """Return pretty-printed XML string with indentation."""
    rough = ET.tostring(elem, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ")


def _sanitize_xml_id(name: str) -> str:
    """Convert arbitrary population names (e.g. 'CD3+', 'B cells') to valid Gating-ML 2.0 xs:ID."""
    s = str(name).replace("+", "_pos").replace("*", "_star")
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or not re.match(r"^[A-Za-z_]", s):
        s = f"Gate_{s}" if s else "Gate"
    return s


def _normalize_transforms(
    transforms: Optional[Dict[str, Any]],
) -> Dict[str, BaseTransform]:
    """Convert raw transform dicts or BaseTransform objects into BaseTransform instances."""
    out: Dict[str, BaseTransform] = {}
    if not transforms:
        return out
    for ch, xf in transforms.items():
        if isinstance(xf, BaseTransform):
            out[ch] = xf
        elif isinstance(xf, dict):
            try:
                out[ch] = create_transform(xf)
            except Exception:
                continue
    return out


def _inverse_scalar(
    val: Optional[float],
    channel: str,
    transforms: Dict[str, BaseTransform],
) -> Optional[float]:
    """Inverse-transform a single scalar gate coordinate back to linear space."""
    if val is None or not math.isfinite(float(val)):
        return None
    fval = float(val)
    xf = transforms.get(channel)
    if xf is None:
        return fval
    try:
        inv = float(xf.inverse(np.array([fval], dtype=np.float64))[0])
        return inv if math.isfinite(inv) else fval
    except Exception:
        return fval


def _transform_scalar_for_gml(
    val: Optional[float],
    channel: str,
    transforms: Dict[str, BaseTransform],
    inline_xform_channels: Dict[str, str],
) -> Optional[float]:
    """Prepare scalar coordinate for Gating-ML 2.0 export.

    LogicleTransform matches ISAC Gating-ML 2.0 <transforms:logicle> directly so
    transformed coordinates are kept verbatim with a transformation-ref. Other
    transforms (normalized Biex/Asinh/Log) are inverse-transformed to linear
    compensated units without a transformation-ref so Gating-ML tools agree.
    """
    if val is None or not math.isfinite(float(val)):
        return None
    if channel in inline_xform_channels:
        return float(val)
    return _inverse_scalar(val, channel, transforms)


def _format_num(val: float, decimals: int = 6) -> str:
    """Format floating-point numbers cleanly without excessive trailing noise."""
    return str(round(float(val), decimals))


def _resolve_matrix_metadata(
    compensation: Optional[Any] = None,
    spectral_library: Optional[Any] = None,
) -> Tuple[List[str], List[str], Optional[np.ndarray], str]:
    """Extract fluorochrome labels, detector labels, coefficient matrix, and name."""
    if compensation is not None and getattr(compensation, "matrix", None) is not None:
        detectors = [str(d) for d in compensation.detectors]
        mat = np.asarray(compensation.matrix, dtype=np.float64)
        name = getattr(compensation, "name", None) or "FlowX_Comp"
        return detectors, detectors, mat, name

    if spectral_library is not None and getattr(spectral_library, "matrix", None) is not None:
        detectors = [str(d) for d in spectral_library.detectors]
        fluoros = [str(f) for f in spectral_library.fluorophores]
        # SpectralLibrary.matrix has shape (D, F); spectrumMatrix expects F rows x D cols
        mat = np.asarray(spectral_library.matrix, dtype=np.float64).T
        return fluoros, detectors, mat, "FlowX_Spectral"

    return [], [], None, ""


def _add_gml_matrix(
    root: ET.Element,
    matrix_id: str,
    fluoros: List[str],
    detectors: List[str],
    matrix: np.ndarray,
) -> None:
    """Append an ISAC Gating-ML 2.0 <transforms:spectrumMatrix> element."""
    sm_el = ET.SubElement(
        root,
        f"{{{TRANSFORMS_NS}}}spectrumMatrix",
        attrib={f"{{{TRANSFORMS_NS}}}id": matrix_id},
    )
    fl_el = ET.SubElement(sm_el, f"{{{TRANSFORMS_NS}}}fluorochromes")
    for fl in fluoros:
        ET.SubElement(
            fl_el,
            f"{{{DATATYPE_NS}}}fcs-dimension",
            attrib={f"{{{DATATYPE_NS}}}name": fl},
        )

    det_el = ET.SubElement(sm_el, f"{{{TRANSFORMS_NS}}}detectors")
    for det in detectors:
        ET.SubElement(
            det_el,
            f"{{{DATATYPE_NS}}}fcs-dimension",
            attrib={f"{{{DATATYPE_NS}}}name": det},
        )

    for row in matrix:
        spec_el = ET.SubElement(sm_el, f"{{{TRANSFORMS_NS}}}spectrum")
        for val in row:
            ET.SubElement(
                spec_el,
                f"{{{TRANSFORMS_NS}}}coefficient",
                attrib={f"{{{TRANSFORMS_NS}}}value": f"{float(val):.8g}"},
            )


def _add_gml_logicle_transforms(
    root: ET.Element,
    transforms: Dict[str, BaseTransform],
) -> Dict[str, str]:
    """Emit deduplicated Gating-ML 2.0 <transforms:logicle> definitions.

    Returns:
        Mapping of channel_name -> transformation XML id for channels using Logicle.
    """
    sig_to_id: Dict[Tuple[float, float, float, float], str] = {}
    ch_to_id: Dict[str, str] = {}

    for ch, xf in transforms.items():
        if not isinstance(xf, LogicleTransform):
            continue
        sig = (xf.param_t, xf.param_w, xf.param_m, xf.param_a)
        if sig not in sig_to_id:
            xid = f"FlowX_Logicle_{len(sig_to_id) + 1}"
            sig_to_id[sig] = xid
            t_el = ET.SubElement(
                root,
                f"{{{TRANSFORMS_NS}}}transformation",
                attrib={f"{{{TRANSFORMS_NS}}}id": xid},
            )
            ET.SubElement(
                t_el,
                f"{{{TRANSFORMS_NS}}}logicle",
                attrib={
                    f"{{{TRANSFORMS_NS}}}T": str(xf.param_t),
                    f"{{{TRANSFORMS_NS}}}W": str(xf.param_w),
                    f"{{{TRANSFORMS_NS}}}M": str(xf.param_m),
                    f"{{{TRANSFORMS_NS}}}A": str(xf.param_a),
                },
            )
        ch_to_id[ch] = sig_to_id[sig]

    return ch_to_id


def _append_gml_dimension(
    parent_el: ET.Element,
    channel: str,
    comp_id: Optional[str] = None,
    comp_channels: Optional[set[str]] = None,
    xform_ids: Optional[Dict[str, str]] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    use_comp_dim: bool = False,
    require_comp_ref: bool = True,
) -> ET.Element:
    """Create a Gating-ML 2.0 <gating:dimension> element with optional refs and bounds."""
    attrib: Dict[str, str] = {}
    if comp_id and comp_channels and channel in comp_channels:
        attrib[f"{{{GATING_ML_NS}}}compensation-ref"] = comp_id
    elif require_comp_ref:
        attrib[f"{{{GATING_ML_NS}}}compensation-ref"] = "uncompensated"
    if xform_ids and channel in xform_ids:
        attrib[f"{{{GATING_ML_NS}}}transformation-ref"] = xform_ids[channel]
    if min_val is not None:
        attrib[f"{{{GATING_ML_NS}}}min"] = _format_num(min_val)
    if max_val is not None:
        attrib[f"{{{GATING_ML_NS}}}max"] = _format_num(max_val)

    dim_el = ET.SubElement(parent_el, f"{{{GATING_ML_NS}}}dimension", attrib=attrib)
    dim_name = f"Comp-{channel}" if (use_comp_dim and comp_channels and channel in comp_channels) else channel
    ET.SubElement(
        dim_el,
        f"{{{DATATYPE_NS}}}fcs-dimension",
        attrib={f"{{{DATATYPE_NS}}}name": dim_name},
    )
    return dim_el


def _append_quadrant_positions(
    gate_el: ET.Element,
    gate_name: str,
    div_x_id: str,
    div_y_id: str,
    x_cut: float,
    y_cut: float,
    max_range: float = 262144.0,
) -> None:
    """Write the 4 <gating:Quadrant> elements with <gating:position> tags."""
    step_x = max(abs(x_cut) * 0.5, 1.0 if abs(x_cut) > 10.0 else 0.25)
    step_y = max(abs(y_cut) * 0.5, 1.0 if abs(y_cut) > 10.0 else 0.25)
    quad_locs = {
        "Q1": (x_cut + step_x, y_cut + step_y),  # (+, +)
        "Q2": (x_cut - step_x, y_cut + step_y),  # (-, +)
        "Q3": (x_cut - step_x, y_cut - step_y),  # (-, -)
        "Q4": (x_cut + step_x, y_cut - step_y),  # (+, -)
    }
    for quad, (lx, ly) in quad_locs.items():
        q_el = ET.SubElement(
            gate_el,
            f"{{{GATING_ML_NS}}}Quadrant",
            attrib={f"{{{GATING_ML_NS}}}id": f"{gate_name}_{quad}"},
        )
        ET.SubElement(
            q_el,
            f"{{{GATING_ML_NS}}}position",
            attrib={
                f"{{{GATING_ML_NS}}}divider_ref": div_x_id,
                f"{{{GATING_ML_NS}}}location": _format_num(min(lx, max_range)),
            },
        )
        ET.SubElement(
            q_el,
            f"{{{GATING_ML_NS}}}position",
            attrib={
                f"{{{GATING_ML_NS}}}divider_ref": div_y_id,
                f"{{{GATING_ML_NS}}}location": _format_num(min(ly, max_range)),
            },
        )


def _append_gml_gate_element(
    parent_el: ET.Element,
    gate: Gate,
    transforms: Dict[str, BaseTransform],
    inline_xform_ids: Dict[str, str],
    comp_id: Optional[str],
    comp_channels: set[str],
    gate_id_override: Optional[str] = None,
    parent_id_override: Optional[str] = None,
    use_comp_dim: bool = False,
    require_comp_ref: bool = True,
    wsp_mode: bool = False,
) -> ET.Element:
    """Serialize a FlowX Gate to an ISAC Gating-ML 2.0 XML element."""
    g_id = gate_id_override or _sanitize_xml_id(gate.name)
    raw_parent = parent_id_override if parent_id_override is not None else (gate.parent_gate or None)
    p_id = _sanitize_xml_id(raw_parent) if (raw_parent and not gate_id_override) else raw_parent

    common_attrib: Dict[str, str] = {f"{{{GATING_ML_NS}}}id": g_id}
    if p_id:
        common_attrib[f"{{{GATING_ML_NS}}}parent_id"] = p_id
    if wsp_mode:
        common_attrib["eventsInside"] = "1"
        common_attrib["userDefined"] = "1"

    if isinstance(gate, PolygonGate):
        g_el = ET.SubElement(parent_el, f"{{{GATING_ML_NS}}}PolygonGate", attrib=common_attrib)
        for ch in gate.channels:
            _append_gml_dimension(
                g_el,
                channel=ch,
                comp_id=comp_id,
                comp_channels=comp_channels,
                xform_ids=inline_xform_ids,
                use_comp_dim=use_comp_dim,
                require_comp_ref=require_comp_ref,
            )
        ch_x, ch_y = gate.channels[0], gate.channels[1]
        for vert in gate.vertices:
            vx = _transform_scalar_for_gml(float(vert[0]), ch_x, transforms, inline_xform_ids)
            vy = _transform_scalar_for_gml(float(vert[1]), ch_y, transforms, inline_xform_ids)
            v_el = ET.SubElement(g_el, f"{{{GATING_ML_NS}}}vertex")
            ET.SubElement(
                v_el,
                f"{{{GATING_ML_NS}}}coordinate",
                attrib={f"{{{DATATYPE_NS}}}value": _format_num(vx if vx is not None else float(vert[0]), 4)},
            )
            ET.SubElement(
                v_el,
                f"{{{GATING_ML_NS}}}coordinate",
                attrib={f"{{{DATATYPE_NS}}}value": _format_num(vy if vy is not None else float(vert[1]), 4)},
            )
        return g_el

    if isinstance(gate, RectangleGate):
        g_el = ET.SubElement(parent_el, f"{{{GATING_ML_NS}}}RectangleGate", attrib=common_attrib)
        ch_x = gate.channels[0]
        _append_gml_dimension(
            g_el,
            channel=ch_x,
            comp_id=comp_id,
            comp_channels=comp_channels,
            xform_ids=inline_xform_ids,
            min_val=_transform_scalar_for_gml(gate.x_min, ch_x, transforms, inline_xform_ids),
            max_val=_transform_scalar_for_gml(gate.x_max, ch_x, transforms, inline_xform_ids),
            use_comp_dim=use_comp_dim,
            require_comp_ref=require_comp_ref,
        )
        if len(gate.channels) > 1:
            ch_y = gate.channels[1]
            _append_gml_dimension(
                g_el,
                channel=ch_y,
                comp_id=comp_id,
                comp_channels=comp_channels,
                xform_ids=inline_xform_ids,
                min_val=_transform_scalar_for_gml(gate.y_min, ch_y, transforms, inline_xform_ids),
                max_val=_transform_scalar_for_gml(gate.y_max, ch_y, transforms, inline_xform_ids),
                use_comp_dim=use_comp_dim,
                require_comp_ref=require_comp_ref,
            )
        return g_el

    if isinstance(gate, RangeGate):
        g_el = ET.SubElement(parent_el, f"{{{GATING_ML_NS}}}RectangleGate", attrib=common_attrib)
        _append_gml_dimension(
            g_el,
            channel=gate.channel,
            comp_id=comp_id,
            comp_channels=comp_channels,
            xform_ids=inline_xform_ids,
            min_val=_transform_scalar_for_gml(gate.min_val, gate.channel, transforms, inline_xform_ids),
            max_val=_transform_scalar_for_gml(gate.max_val, gate.channel, transforms, inline_xform_ids),
            use_comp_dim=use_comp_dim,
            require_comp_ref=require_comp_ref,
        )
        return g_el

    if isinstance(gate, QuadrantGate):
        g_el = ET.SubElement(parent_el, f"{{{GATING_ML_NS}}}QuadrantGate", attrib=common_attrib)
        div_x_id = f"{g_id}_X_div"
        div_y_id = f"{g_id}_Y_div"
        x_cut = _transform_scalar_for_gml(gate.x_cutoff, gate.x_channel, transforms, inline_xform_ids)
        y_cut = _transform_scalar_for_gml(gate.y_cutoff, gate.y_channel, transforms, inline_xform_ids)
        x_val = x_cut if x_cut is not None else float(gate.x_cutoff)
        y_val = y_cut if y_cut is not None else float(gate.y_cutoff)

        for div_id, ch, cut_v in ((div_x_id, gate.x_channel, x_val), (div_y_id, gate.y_channel, y_val)):
            div_attrib: Dict[str, str] = {f"{{{GATING_ML_NS}}}id": div_id}
            if comp_id and ch in comp_channels:
                div_attrib[f"{{{GATING_ML_NS}}}compensation-ref"] = comp_id
            elif require_comp_ref:
                div_attrib[f"{{{GATING_ML_NS}}}compensation-ref"] = "uncompensated"
            if ch in inline_xform_ids:
                div_attrib[f"{{{GATING_ML_NS}}}transformation-ref"] = inline_xform_ids[ch]
            div_el = ET.SubElement(g_el, f"{{{GATING_ML_NS}}}divider", attrib=div_attrib)
            dim_name = f"Comp-{ch}" if (use_comp_dim and ch in comp_channels) else ch
            ET.SubElement(div_el, f"{{{DATATYPE_NS}}}fcs-dimension", attrib={f"{{{DATATYPE_NS}}}name": dim_name})
            v_el = ET.SubElement(div_el, f"{{{GATING_ML_NS}}}value")
            v_el.text = _format_num(cut_v)

        _append_quadrant_positions(g_el, g_id, div_x_id, div_y_id, x_val, y_val)
        return g_el

    raise ValueError(f"Unsupported gate type for Gating-ML export: {type(gate).__name__}")


def export_gating_ml(
    hierarchy: GatingHierarchy,
    filepath: Union[str, Path],
    experiment_name: str = "FlowX_Analysis",
    compensation: Optional[Any] = None,
    transforms: Optional[Dict[str, Any]] = None,
    spectral_library: Optional[Any] = None,
) -> Path:
    """Export a GatingHierarchy to a standardized Gating-ML 2.0 XML document.

    When ``compensation`` and/or ``transforms`` are supplied, includes ISAC
    Gating-ML 2.0 compensation matrices and transformations so external tools
    (Cytobank, FlowKit, R/flowUtils) can apply the hierarchy directly to raw FCS files.

    Args:
        hierarchy: FlowX GatingHierarchy to serialize.
        filepath: Destination XML path.
        experiment_name: Document metadata label.
        compensation: Optional CompensationMatrix applied to raw events.
        transforms: Optional mapping of channel name -> BaseTransform or dict.
        spectral_library: Optional SpectralLibrary for spectral unmixing runs.

    Returns:
        Path to written Gating-ML 2.0 file.
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _register_gml_namespaces()

    root = ET.Element(f"{{{GATING_ML_NS}}}Gating-ML")
    if experiment_name:
        info_el = ET.SubElement(root, f"{{{DATATYPE_NS}}}custom_info")
        info_el.text = str(experiment_name)

    xform_map = _normalize_transforms(transforms)
    inline_xform_ids = _add_gml_logicle_transforms(root, xform_map)

    fluoros, detectors, comp_mat, _ = _resolve_matrix_metadata(
        compensation=compensation,
        spectral_library=spectral_library,
    )
    comp_id: Optional[str] = None
    comp_channels: set[str] = set()
    if comp_mat is not None and fluoros and detectors:
        comp_id = "FlowX_Comp"
        comp_channels = set(fluoros)
        _add_gml_matrix(root, comp_id, fluoros, detectors, comp_mat)

    for gate_name in hierarchy.order:
        gate = hierarchy.get_gate(gate_name)
        _append_gml_gate_element(
            root,
            gate=gate,
            transforms=xform_map,
            inline_xform_ids=inline_xform_ids,
            comp_id=comp_id,
            comp_channels=comp_channels,
        )

    out_path.write_text(_prettify_xml(root), encoding="utf-8")
    return out_path


def export_preprocessing_gating_ml(
    hierarchy: GatingHierarchy,
    filepath: Union[str, Path],
    experiment_name: str = "FlowX_Preprocessing_Gating",
    compensation: Optional[Any] = None,
    transforms: Optional[Dict[str, Any]] = None,
    spectral_library: Optional[Any] = None,
) -> Path:
    """Export FlowX preprocessing (compensation + transforms) and cleanup/targeted gates as Gating-ML 2.0 XML."""
    return export_gating_ml(
        hierarchy=hierarchy,
        filepath=filepath,
        experiment_name=experiment_name,
        compensation=compensation,
        transforms=transforms,
        spectral_library=spectral_library,
    )


def _resolve_wsp_dataset_uri(
    sample_path: str,
    sample_filename: str,
    wsp_filepath: Path,
) -> str:
    """Compute a FlowJo-compatible DataSet URI relative to the `.wsp` file's directory.

    Avoids exposing local machine absolute paths while ensuring FlowJo and FlowKit
    can locate the referenced `.fcs` file from the `.wsp` directory.
    """
    raw = str(sample_path or "").strip()
    if not raw:
        return sample_filename or ""

    try:
        ws_dir = wsp_filepath.resolve().parent
        target = Path(raw)
        if not target.is_absolute():
            cwd_target = (Path.cwd() / target).resolve()
            ws_target = (ws_dir / target).resolve()
            if cwd_target.exists():
                target = cwd_target
            elif ws_target.exists():
                target = ws_target
            else:
                target = cwd_target
        else:
            target = target.resolve()

        rel = os.path.relpath(target, start=ws_dir).replace("\\", "/")
        return quote(rel, safe="/:")
    except Exception:
        fallback = sanitize_path(raw) or sample_filename or ""
        return quote(fallback.replace("\\", "/"), safe="/:")


def _append_wsp_transform(
    transforms_el: ET.Element,
    channel: str,
    xf: Optional[BaseTransform],
    max_range: float,
) -> None:
    """Append a FlowJo v10 <Transformations> child element for one channel."""
    if isinstance(xf, LogicleTransform):
        el = ET.SubElement(
            transforms_el,
            f"{{{TRANSFORMS_NS}}}logicle",
            attrib={
                f"{{{TRANSFORMS_NS}}}length": "256",
                f"{{{TRANSFORMS_NS}}}T": str(xf.param_t),
                f"{{{TRANSFORMS_NS}}}W": str(xf.param_w),
                f"{{{TRANSFORMS_NS}}}M": str(xf.param_m),
                f"{{{TRANSFORMS_NS}}}A": str(xf.param_a),
            },
        )
    elif isinstance(xf, BiexponentialTransform):
        el = ET.SubElement(
            transforms_el,
            f"{{{TRANSFORMS_NS}}}biex",
            attrib={
                f"{{{TRANSFORMS_NS}}}length": "256",
                f"{{{TRANSFORMS_NS}}}maxRange": str(xf.max_value),
                f"{{{TRANSFORMS_NS}}}neg": str(xf.negative),
                f"{{{TRANSFORMS_NS}}}pos": str(xf.positive),
                f"{{{TRANSFORMS_NS}}}width": str(xf.width_basis),
            },
        )
    elif isinstance(xf, AsinhTransform):
        m_dec = 4.5
        t_equiv = float(xf.cofactor) * float(np.sinh(m_dec * np.log(10.0)))
        el = ET.SubElement(
            transforms_el,
            f"{{{TRANSFORMS_NS}}}fasinh",
            attrib={
                f"{{{TRANSFORMS_NS}}}length": "256",
                f"{{{TRANSFORMS_NS}}}maxRange": str(xf.max_value),
                f"{{{TRANSFORMS_NS}}}T": _format_num(t_equiv, 4),
                f"{{{TRANSFORMS_NS}}}M": str(m_dec),
                f"{{{TRANSFORMS_NS}}}A": "0.0",
            },
        )
    elif isinstance(xf, LogTransform):
        el = ET.SubElement(
            transforms_el,
            f"{{{TRANSFORMS_NS}}}log",
            attrib={
                f"{{{TRANSFORMS_NS}}}offset": str(xf.floor),
                f"{{{TRANSFORMS_NS}}}decades": "4.5",
            },
        )
    else:
        el = ET.SubElement(
            transforms_el,
            f"{{{TRANSFORMS_NS}}}linear",
            attrib={
                f"{{{TRANSFORMS_NS}}}minRange": "0",
                f"{{{TRANSFORMS_NS}}}maxRange": _format_num(max_range, 2),
                "gain": "1",
            },
        )
    ET.SubElement(el, f"{{{DATATYPE_NS}}}parameter", attrib={f"{{{DATATYPE_NS}}}name": channel})


def _append_wsp_spillover_matrix(
    parent_el: ET.Element,
    comp: Any,
    matrix_xml_id: str,
) -> Optional[set[str]]:
    """Append a FlowJo <transforms:spilloverMatrix> element if comp is square and valid."""
    if comp is None or getattr(comp, "matrix", None) is None:
        return None
    detectors = [str(d) for d in getattr(comp, "detectors", [])]
    mat = np.asarray(comp.matrix, dtype=np.float64)
    if not detectors or mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return None

    name = getattr(comp, "name", None) or "FlowX_Comp"
    sm_el = ET.SubElement(
        parent_el,
        f"{{{TRANSFORMS_NS}}}spilloverMatrix",
        attrib={
            "spectral": "0",
            "prefix": "Comp-",
            "suffix": "",
            "name": name,
            "editable": "0",
            "status": "FINALIZED",
            f"{{{TRANSFORMS_NS}}}id": matrix_xml_id,
        },
    )
    params_el = ET.SubElement(sm_el, f"{{{DATATYPE_NS}}}parameters")
    for det in detectors:
        ET.SubElement(
            params_el,
            f"{{{DATATYPE_NS}}}parameter",
            attrib={f"{{{DATATYPE_NS}}}name": det, "userProvidedCompInfix": f"Comp-{det}"},
        )

    for i, det_i in enumerate(detectors):
        sp_el = ET.SubElement(
            sm_el,
            f"{{{TRANSFORMS_NS}}}spillover",
            attrib={f"{{{DATATYPE_NS}}}parameter": det_i, "userProvidedCompInfix": f"Comp-{det_i}"},
        )
        for j, det_j in enumerate(detectors):
            ET.SubElement(
                sp_el,
                f"{{{TRANSFORMS_NS}}}coefficient",
                attrib={
                    f"{{{DATATYPE_NS}}}parameter": det_j,
                    f"{{{TRANSFORMS_NS}}}value": f"{float(mat[i, j]):.8g}",
                },
            )
    return set(detectors)


def _append_wsp_graph(
    parent_el: ET.Element,
    x_axis: str,
    y_axis: str = "",
) -> None:
    """Append a FlowJo <Graph> element so double-clicking a node opens the plot window."""
    is_hist = not y_axis
    g_el = ET.SubElement(
        parent_el,
        "Graph",
        attrib={
            "smoothing": "0",
            "backColor": "#ffffff",
            "foreColor": "#000000",
            "type": "Histogram" if is_hist else "Pseudocolor",
            "fast": "1",
        },
    )
    ET.SubElement(g_el, "Axis", attrib={"dimension": "x", "name": x_axis, "label": "", "auto": "auto"})
    ET.SubElement(g_el, "Axis", attrib={"dimension": "y", "name": y_axis, "label": "", "auto": "auto"})
    ET.SubElement(
        g_el,
        "GraphSettings",
        attrib={
            "level": "5%",
            "smoothingHighResolution": "1",
            "contourHighResolution": "1",
            "histogramSmoothingCount": "0",
            "graphResolution": "256",
            "showOutliers": "0",
            "drawLargeDots": "0",
            "dotsToDraw": "0" if is_hist else "8000",
            "tint": "le.chartfill.tinted.40",
            "lineWeight": "le.lineweight.normal",
            "lineStyle": "le.linestyle.solid",
        },
    )
    ET.SubElement(
        g_el,
        "GraphEnvironment",
        attrib={
            "showGrid": "0",
            "showAxes": "H" if is_hist else "tnlTNL",
            "showGates": "1",
            "showFreqOnPlots": "1",
            "showGateNameOnPlots": "1",
            "showMedians": "0",
            "showUncomped": "0",
            "addEventParam": "0",
            "lastYAxisName": "",
        },
    )


def _wsp_axes_for_children(
    gates: List[Gate],
    comp_channels: set[str],
    default_axes: Tuple[str, str],
) -> Tuple[str, str]:
    """Choose X/Y plot axes from the first child gate, or fall back to defaults."""
    if gates and gates[0].channels:
        chs = gates[0].channels
        x_ch = f"Comp-{chs[0]}" if chs[0] in comp_channels else chs[0]
        if len(chs) > 1:
            y_ch = f"Comp-{chs[1]}" if chs[1] in comp_channels else chs[1]
            return x_ch, y_ch
        return x_ch, ""
    return default_axes


def _append_wsp_population_tree(
    subpops_el: ET.Element,
    hierarchy: GatingHierarchy,
    transforms: Dict[str, BaseTransform],
    comp_channels: set[str],
    gate_counts: Dict[str, int],
    parent_gate_name: Optional[str],
    parent_xml_id: Optional[str],
    sample_idx: int,
    id_counter: List[int],
    children_map: Dict[Optional[str], List[Gate]],
    default_axes: Tuple[str, str] = ("FSC-A", "SSC-A"),
) -> None:
    """Recursively emit FlowJo <Population> -> <Gate> -> <Subpopulations> tree."""
    for gate in children_map.get(parent_gate_name, []):
        id_counter[0] += 1
        gate_xml_id = f"FX_{sample_idx}_{id_counter[0]}"
        pop_count = gate_counts.get(gate.name, 0)

        pop_el = ET.SubElement(
            subpops_el,
            "Population",
            attrib={
                "name": gate.name,
                "annotation": "",
                "owningGroup": "",
                "expanded": "1",
                "sortPriority": "10",
                "count": str(int(pop_count)),
            },
        )
        child_gates = children_map.get(gate.name, [])
        gx, gy = _wsp_axes_for_children(child_gates, comp_channels, default_axes)
        _append_wsp_graph(pop_el, gx, gy)
        gate_attrib: Dict[str, str] = {f"{{{GATING_ML_NS}}}id": gate_xml_id}
        if parent_xml_id:
            gate_attrib[f"{{{GATING_ML_NS}}}parent_id"] = parent_xml_id
        gate_el = ET.SubElement(pop_el, "Gate", attrib=gate_attrib)

        # FlowJo .wsp stores gate coordinates in untransformed compensated linear space
        # without inline Gating-ML transformation-ref attributes.
        _append_gml_gate_element(
            gate_el,
            gate=gate,
            transforms=transforms,
            inline_xform_ids={},
            comp_id=None,
            comp_channels=comp_channels,
            gate_id_override=gate_xml_id,
            parent_id_override=parent_xml_id,
            use_comp_dim=bool(comp_channels),
            require_comp_ref=False,
            wsp_mode=True,
        )

        if gate.name in children_map:
            child_sub_el = ET.SubElement(pop_el, "Subpopulations")
            _append_wsp_population_tree(
                child_sub_el,
                hierarchy=hierarchy,
                transforms=transforms,
                comp_channels=comp_channels,
                gate_counts=gate_counts,
                parent_gate_name=gate.name,
                parent_xml_id=gate_xml_id,
                sample_idx=sample_idx,
                id_counter=id_counter,
                children_map=children_map,
                default_axes=default_axes,
            )


def _collect_hierarchy_children_map(hierarchy: GatingHierarchy) -> Dict[Optional[str], List[Gate]]:
    """Group gates in topological order by effective parent (orphans attach to root)."""
    known_names = set(hierarchy.gates.keys())
    children_map: Dict[Optional[str], List[Gate]] = {}
    for gname in hierarchy.order:
        gate = hierarchy.get_gate(gname)
        parent = gate.parent_gate if (gate.parent_gate and gate.parent_gate in known_names) else None
        children_map.setdefault(parent, []).append(gate)
    return children_map


def export_preprocessing_flowjo_wsp(
    samples: Iterable[SampleWorkspaceSpec],
    filepath: Union[str, Path],
    experiment_name: str = "FlowX_Preprocessing_Workspace",
) -> Path:
    """Export per-sample preprocessing (comp + transforms) and cleanup/targeted gating to FlowJo v10 .wsp XML.

    Every gate coordinate on a transformed fluorescence channel is inverse-transformed
    to compensated linear units (`Comp-<detector>`), and every sample's full transform
    table and spillover matrix is embedded so FlowJo / FlowKit can load the workspace
    directly against raw `.fcs` files.

    Args:
        samples: Iterable of SampleWorkspaceSpec entries (one per raw FCS sample).
        filepath: Destination `.wsp` XML path.
        experiment_name: Workspace title attribute.

    Returns:
        Path to the written `.wsp` file.
    """
    sample_list_specs = list(samples)
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _register_gml_namespaces()

    root = ET.Element(
        "Workspace",
        attrib={
            "version": "20.0",
            "flowJoVersion": "10.8.1",
            "curGroup": "All Samples",
            "name": experiment_name,
        },
    )

    # Top-level Matrices & Groups
    matrices_el = ET.SubElement(root, "Matrices")
    for idx, spec in enumerate(sample_list_specs, start=1):
        _append_wsp_spillover_matrix(matrices_el, spec.compensation, matrix_xml_id=f"comp_{idx}")

    groups_el = ET.SubElement(root, "Groups")
    all_group_node_el = ET.SubElement(
        groups_el,
        "GroupNode",
        attrib={
            "name": "All Samples",
            "annotation": "",
            "owningGroup": "All Samples",
            "expanded": "1",
            "sortPriority": "10",
            "count": "-1",
        },
    )
    all_group_el = ET.SubElement(
        all_group_node_el,
        "Group",
        attrib={
            "name": "All Samples",
            "live": "1",
            "role": "ws.group.dlog.test",
            "key": "",
            "synchronized": "0",
            "foreground": "#000000",
            "fontStyle": "bold",
        },
    )
    ET.SubElement(all_group_el, "Criteria")
    sample_refs_el = ET.SubElement(all_group_el, "SampleRefs")
    for idx in range(1, len(sample_list_specs) + 1):
        ET.SubElement(sample_refs_el, "SampleRef", attrib={"sampleID": str(idx)})
    ET.SubElement(all_group_el, "Keywords")

    # Per-sample compensation, transforms, keywords, and gating trees
    sample_list_el = ET.SubElement(root, "SampleList")
    for idx, spec in enumerate(sample_list_specs, start=1):
        sid = str(idx)
        sample_el = ET.SubElement(sample_list_el, "Sample")
        uri = _resolve_wsp_dataset_uri(spec.sample_path, spec.sample_filename, out_path)
        ET.SubElement(sample_el, "DataSet", attrib={"uri": uri or spec.sample_filename, "sampleID": sid})

        comp_channels = _append_wsp_spillover_matrix(sample_el, spec.compensation, matrix_xml_id=f"comp_{idx}") or set()

        xform_map = _normalize_transforms(spec.transforms)
        ordered_channels: List[str] = list(spec.channels)
        if not ordered_channels:
            for ch in list(xform_map.keys()):
                if ch not in ordered_channels:
                    ordered_channels.append(ch)
        for gname in spec.hierarchy.order:
            for ch in spec.hierarchy.get_gate(gname).channels:
                if ch not in ordered_channels:
                    ordered_channels.append(ch)

        transforms_el = ET.SubElement(sample_el, "Transformations")
        for ch in ordered_channels:
            _append_wsp_transform(transforms_el, ch, xform_map.get(ch), max_range=spec.max_range)

        keywords_el = ET.SubElement(sample_el, "Keywords")
        ET.SubElement(keywords_el, "Keyword", name="$FIL", value=spec.sample_filename or "")
        ET.SubElement(keywords_el, "Keyword", name="$TOT", value=str(int(spec.total_events)))

        sample_node_el = ET.SubElement(
            sample_el,
            "SampleNode",
            attrib={
                "name": spec.sample_filename or f"sample_{idx}.fcs",
                "annotation": "",
                "owningGroup": "",
                "expanded": "1",
                "sortPriority": "10",
                "count": str(int(spec.total_events)),
                "sampleID": sid,
            },
        )
        default_x = "FSC-A" if "FSC-A" in ordered_channels else (ordered_channels[0] if ordered_channels else "FSC-A")
        default_y = "SSC-A" if "SSC-A" in ordered_channels else (ordered_channels[1] if len(ordered_channels) > 1 else "")
        children_map = _collect_hierarchy_children_map(spec.hierarchy)
        sx, sy = _wsp_axes_for_children(children_map.get(None, []), comp_channels, (default_x, default_y))
        _append_wsp_graph(sample_node_el, sx, sy)
        subpops_el = ET.SubElement(sample_node_el, "Subpopulations")
        _append_wsp_population_tree(
            subpops_el,
            hierarchy=spec.hierarchy,
            transforms=xform_map,
            comp_channels=comp_channels,
            gate_counts=spec.gate_counts or {},
            parent_gate_name=None,
            parent_xml_id=None,
            sample_idx=idx,
            id_counter=[0],
            children_map=children_map,
            default_axes=(default_x, default_y),
        )

    out_path.write_text(_prettify_xml(root), encoding="utf-8")
    return out_path


def export_flowjo_wsp(
    session: Any,
    filepath: Union[str, Path],
) -> Path:
    """Export WorkspaceSession into a FlowJo .wsp XML workspace file.

    Args:
        session: WorkspaceSession instance containing samples, compensation, and hierarchy.
        filepath: Destination .wsp file path.

    Returns:
        Path to written FlowJo .wsp workspace.
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element(
        "Workspace",
        attrib={
            "version": "20.0",
            "flowJoVersion": "10.8.1",
            "name": getattr(session, "name", "FlowX_Workspace"),
        },
    )

    # 1. SampleList
    sample_list = ET.SubElement(root, "SampleList")
    sample_el = ET.SubElement(
        sample_list,
        "Sample",
        attrib={
            "sampleID": "1",
            "name": getattr(session, "sample_filename", "sample.fcs") or "sample.fcs",
            "count": str(getattr(session, "total_events", 0)),
        },
    )

    # Keywords & Parameters
    keywords_el = ET.SubElement(sample_el, "Keywords")
    ET.SubElement(keywords_el, "Keyword", name="$FIL", value=getattr(session, "sample_filename", "") or "")
    ET.SubElement(keywords_el, "Keyword", name="$TOT", value=str(getattr(session, "total_events", 0)))

    # 2. Compensation Matrix
    comp = getattr(session, "compensation", None)
    if comp is not None and comp.matrix is not None:
        comp_name = getattr(comp, "name", None) or "CompMatrix"
        comp_el = ET.SubElement(root, "CompensationMatrix", attrib={"name": comp_name})
        detectors = comp.detectors
        for det in detectors:
            ET.SubElement(comp_el, "Channel", name=det)
        matrix_el = ET.SubElement(comp_el, "MatrixData")
        matrix_el.text = " ".join(f"{val:.6f}" for val in comp.matrix.ravel())

    # 3. Gating Hierarchy
    hierarchy = getattr(session, "hierarchy", None)
    if hierarchy is not None:
        gates_root = ET.SubElement(sample_el, "GatingHierarchy")
        for gate_name in hierarchy.order:
            gate = hierarchy.get_gate(gate_name)
            gate_el = ET.SubElement(
                gates_root,
                "Gate",
                attrib={
                    "name": gate.name,
                    "parent": gate.parent_gate or "",
                },
            )

            if isinstance(gate, PolygonGate):
                p_el = ET.SubElement(gate_el, "PolygonGate", attrib={"xAxis": gate.channels[0], "yAxis": gate.channels[1]})
                for x, y in gate.vertices:
                    ET.SubElement(p_el, "Vertex", x=str(round(float(x), 4)), y=str(round(float(y), 4)))

            elif isinstance(gate, RectangleGate):
                rect_attrib = {
                    "xAxis": gate.channels[0],
                    "xMin": str(gate.x_min) if gate.x_min is not None else "",
                    "xMax": str(gate.x_max) if gate.x_max is not None else "",
                }
                if len(gate.channels) > 1:
                    rect_attrib.update({
                        "yAxis": gate.channels[1],
                        "yMin": str(gate.y_min) if gate.y_min is not None else "",
                        "yMax": str(gate.y_max) if gate.y_max is not None else "",
                    })
                ET.SubElement(gate_el, "RectangleGate", attrib=rect_attrib)

            elif isinstance(gate, RangeGate):
                ET.SubElement(
                    gate_el,
                    "RangeGate",
                    attrib={
                        "channel": gate.channel,
                        "min": str(gate.min_val) if gate.min_val is not None else "",
                        "max": str(gate.max_val) if gate.max_val is not None else "",
                    },
                )

            elif isinstance(gate, QuadrantGate):
                ET.SubElement(
                    gate_el,
                    "QuadrantGate",
                    attrib={
                        "xAxis": gate.x_channel,
                        "yAxis": gate.y_channel,
                        "xCutoff": str(gate.x_cutoff),
                        "yCutoff": str(gate.y_cutoff),
                        "quadrant": gate.quadrant,
                    },
                )

    out_path.write_text(_prettify_xml(root), encoding="utf-8")
    return out_path
