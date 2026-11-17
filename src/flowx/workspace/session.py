"""Flow cytometry workspace session module.

Provides workspace session management, complete pipeline replay capability,
and serialization/deserialization to and from JSON and XML format (similar to FlowJo .wsp).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import pandas as pd

from flowx.gating.gate import Gate, GatingHierarchy, PolygonGate, RangeGate, RectangleGate, create_gate_from_dict
from flowx.io.paths import sanitize_path
from flowx.io.parser import FCSData, parse_fcs
from flowx.math.compensation import CompensationMatrix
from flowx.math.transforms import BaseTransform, create_transform


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer for NumPy types and custom objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


class WorkspaceSession:
    """Manages an entire flow cytometry analysis workspace session.

    Attributes:
        name: Name of workspace.
        sample_path: Path to the analyzed FCS file.
        sample_filename: Base filename of the sample.
        total_events: Original event count.
        channels: List of channel descriptions.
        compensation: Optional CompensationMatrix.
        transforms: Mapping of channel name to transform configuration dictionary.
        hierarchy: GatingHierarchy containing all defined gates and relations.
        plot_configs: List of configured plot specifications.
    """

    def __init__(
        self,
        name: str = "FlowSession",
        sample_path: Optional[str] = None,
        sample_filename: Optional[str] = None,
        total_events: int = 0,
        channels: Optional[List[Dict[str, str]]] = None,
        compensation: Optional[CompensationMatrix] = None,
        transforms: Optional[Dict[str, Dict[str, Any]]] = None,
        hierarchy: Optional[GatingHierarchy] = None,
        plot_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Initialize workspace session."""
        self.name = name
        self.sample_path = sanitize_path(sample_path) if sample_path else ""
        self.sample_filename = sample_filename or ""
        self.total_events = total_events
        self.channels = channels or []
        self.compensation = compensation
        self.transforms = transforms or {}
        self.hierarchy = hierarchy or GatingHierarchy()
        self.plot_configs = plot_configs or []

    def set_sample(self, fcs: FCSData) -> None:
        """Populate sample details from parsed FCSData."""
        self.sample_path = sanitize_path(fcs.filepath)
        self.sample_filename = fcs.filename
        self.total_events = fcs.event_count
        self.channels = [
            {
                "pnn": pnn,
                "pns": fcs.detector_to_marker.get(pnn, ""),
                "label": fcs.get_channel_label(pnn),
            }
            for pnn in fcs.pnn_labels
        ]

    def set_compensation(self, comp: CompensationMatrix) -> None:
        """Register compensation matrix in workspace."""
        self.compensation = comp

    def add_transform(self, channel: str, transform: BaseTransform) -> None:
        """Register a transformation for a specific channel."""
        self.transforms[channel] = transform.to_dict()

    def add_gate(self, gate: Gate) -> None:
        """Add a gate to the gating hierarchy."""
        self.hierarchy.add_gate(gate)

    def add_plot_config(
        self,
        plot_type: str,
        x_channel: str,
        y_channel: Optional[str] = None,
        gate_name: Optional[str] = None,
        title: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> None:
        """Register a plot specification to be reproduced."""
        self.plot_configs.append(
            {
                "type": plot_type,
                "x_channel": x_channel,
                "y_channel": y_channel,
                "gate_name": gate_name,
                "title": title,
                "output_file": sanitize_path(output_file) if output_file else None,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full workspace state to a dictionary."""
        return {
            "name": self.name,
            "version": "1.0",
            "sample": {
                "path": sanitize_path(self.sample_path),
                "filename": self.sample_filename,
                "total_events": self.total_events,
                "channels": self.channels,
            },
            "compensation": self.compensation.to_dict() if self.compensation else None,
            "transforms": self.transforms,
            "hierarchy": self.hierarchy.to_dict(),
            "plot_configs": self.plot_configs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkspaceSession:
        """Construct WorkspaceSession from a dictionary."""
        sample_info = data.get("sample", {})
        comp_dict = data.get("compensation")
        comp = CompensationMatrix.from_dict(comp_dict) if comp_dict else None

        hierarchy_dict = data.get("hierarchy", {})
        hierarchy = GatingHierarchy.from_dict(hierarchy_dict)

        return cls(
            name=data.get("name", "FlowSession"),
            sample_path=sample_info.get("path", ""),
            sample_filename=sample_info.get("filename", ""),
            total_events=sample_info.get("total_events", 0),
            channels=sample_info.get("channels", []),
            compensation=comp,
            transforms=data.get("transforms", {}),
            hierarchy=hierarchy,
            plot_configs=data.get("plot_configs", []),
        )

    def save_json(self, filepath: Union[str, Path], indent: int = 2) -> None:
        """Save workspace session to a JSON file.

        Args:
            filepath: Destination file path.
            indent: JSON indentation formatting.
        """
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, default=_json_default)

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> WorkspaceSession:
        """Load workspace session from a JSON file.

        Args:
            filepath: Source file path.

        Returns:
            Instantiated WorkspaceSession.
        """
        in_path = Path(filepath)
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_xml(self, filepath: Union[str, Path]) -> None:
        """Save workspace session to an XML file (FlowJo/GatingML inspired format).

        Args:
            filepath: Destination file path.
        """
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("Workspace", name=self.name, version="1.0")

        # Sample Element
        sample_el = ET.SubElement(
            root,
            "Sample",
            path=sanitize_path(self.sample_path),
            filename=self.sample_filename,
            total_events=str(self.total_events),
        )
        for ch in self.channels:
            ET.SubElement(
                sample_el,
                "Channel",
                pnn=ch.get("pnn", ""),
                pns=ch.get("pns", ""),
                label=ch.get("label", ""),
            )

        # Compensation Element
        if self.compensation:
            comp_el = ET.SubElement(root, "CompensationMatrix")
            for det in self.compensation.detectors:
                ET.SubElement(comp_el, "Detector", name=det)
            matrix_el = ET.SubElement(comp_el, "MatrixData")
            matrix_el.text = json.dumps(self.compensation.matrix.tolist())

        # Transforms Element
        transforms_el = ET.SubElement(root, "Transforms")
        for ch, xform_cfg in self.transforms.items():
            t_el = ET.SubElement(transforms_el, "Transform", channel=ch, type=xform_cfg.get("type", ""))
            for k, v in xform_cfg.items():
                if k != "type":
                    t_el.set(k, str(v))

        # Gating Hierarchy Element
        hierarchy_el = ET.SubElement(root, "GatingHierarchy")
        for g_dict in self.hierarchy.to_dict().get("gates", []):
            gate_el = ET.SubElement(
                hierarchy_el,
                "Gate",
                type=g_dict.get("type", ""),
                name=g_dict.get("name", ""),
                parent=g_dict.get("parent_gate") or "",
            )
            # Channels
            for ch in g_dict.get("channels", []):
                ET.SubElement(gate_el, "Channel", name=ch)
            # Parameters
            if g_dict.get("type") == "polygon":
                verts_el = ET.SubElement(gate_el, "Vertices")
                verts_el.text = json.dumps(g_dict.get("vertices", []))
            elif g_dict.get("type") == "rectangle":
                rect_el = ET.SubElement(gate_el, "Bounds")
                for bound in ["x_min", "x_max", "y_min", "y_max"]:
                    if g_dict.get(bound) is not None:
                        rect_el.set(bound, str(g_dict[bound]))
            elif g_dict.get("type") == "range":
                range_el = ET.SubElement(gate_el, "Range")
                if g_dict.get("min_val") is not None:
                    range_el.set("min_val", str(g_dict["min_val"]))
                if g_dict.get("max_val") is not None:
                    range_el.set("max_val", str(g_dict["max_val"]))

        # Plot Configs Element
        plots_el = ET.SubElement(root, "Plots")
        for p in self.plot_configs:
            ET.SubElement(
                plots_el,
                "Plot",
                type=p.get("type", ""),
                x_channel=p.get("x_channel", ""),
                y_channel=p.get("y_channel") or "",
                gate_name=p.get("gate_name") or "",
                title=p.get("title") or "",
                output_file=p.get("output_file") or "",
            )

        rough_string = ET.tostring(root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

    def export_gating_ml(self, filepath: Union[str, Path]) -> Path:
        """Export session's gating hierarchy to standard ISAC Gating-ML 2.0 XML.

        Args:
            filepath: Destination file path.

        Returns:
            Path to written Gating-ML 2.0 XML document.
        """
        from flowx.workspace.gating_ml_export import export_gating_ml
        return export_gating_ml(
            self.hierarchy,
            filepath,
            experiment_name=self.name,
            compensation=self.compensation,
            transforms=self.transforms,
        )

    def export_flowjo_wsp(self, filepath: Union[str, Path]) -> Path:
        """Export session to a FlowJo .wsp XML workspace file.

        Args:
            filepath: Destination .wsp file path.

        Returns:
            Path to written FlowJo workspace XML.
        """
        from flowx.workspace.gating_ml_export import export_flowjo_wsp
        return export_flowjo_wsp(self, filepath)

    def export_preprocessing_wsp(
        self,
        filepath: Union[str, Path],
        gate_counts: Optional[Dict[str, int]] = None,
        max_range: float = 262144.0,
    ) -> Path:
        """Export preprocessing (compensation + transforms) and gating tree to FlowJo v10 .wsp XML.

        Args:
            filepath: Destination .wsp file path.
            gate_counts: Optional gate_name -> surviving event count mapping.
            max_range: Linear detector scale upper bound for scatter axes.

        Returns:
            Path to written FlowJo v10 .wsp workspace.
        """
        from flowx.workspace.gating_ml_export import (
            SampleWorkspaceSpec,
            export_preprocessing_flowjo_wsp,
        )

        ch_names = [c.get("pnn", "") for c in self.channels if c.get("pnn")]
        spec = SampleWorkspaceSpec(
            sample_filename=self.sample_filename or "sample.fcs",
            sample_path=self.sample_path or self.sample_filename or "sample.fcs",
            total_events=self.total_events,
            channels=ch_names,
            max_range=max_range,
            compensation=self.compensation,
            transforms=self.transforms,
            hierarchy=self.hierarchy,
            gate_counts=gate_counts or {},
        )
        return export_preprocessing_flowjo_wsp([spec], filepath, experiment_name=self.name)

    @classmethod
    def load_xml(cls, filepath: Union[str, Path]) -> WorkspaceSession:
        """Load workspace session from XML file.

        Args:
            filepath: Source file path.

        Returns:
            Instantiated WorkspaceSession.
        """
        in_path = Path(filepath)
        tree = ET.parse(in_path)
        root = tree.getroot()

        session_name = root.get("name", "FlowSession")

        # Parse Sample
        sample_el = root.find("Sample")
        sample_path = sample_el.get("path", "") if sample_el is not None else ""
        sample_filename = sample_el.get("filename", "") if sample_el is not None else ""
        total_events = int(sample_el.get("total_events", "0")) if sample_el is not None else 0
        channels = []
        if sample_el is not None:
            for ch_el in sample_el.findall("Channel"):
                channels.append(
                    {
                        "pnn": ch_el.get("pnn", ""),
                        "pns": ch_el.get("pns", ""),
                        "label": ch_el.get("label", ""),
                    }
                )

        # Parse Compensation
        comp = None
        comp_el = root.find("CompensationMatrix")
        if comp_el is not None:
            detectors = [d.get("name") for d in comp_el.findall("Detector")]
            matrix_data_el = comp_el.find("MatrixData")
            if matrix_data_el is not None and matrix_data_el.text:
                mat = json.loads(matrix_data_el.text)
                comp = CompensationMatrix(detectors=detectors, matrix=mat)

        # Parse Transforms
        transforms = {}
        transforms_el = root.find("Transforms")
        if transforms_el is not None:
            for t_el in transforms_el.findall("Transform"):
                ch = t_el.get("channel")
                t_type = t_el.get("type")
                cfg: Dict[str, Any] = {"type": t_type}
                for k, v in t_el.attrib.items():
                    if k not in ["channel", "type"]:
                        try:
                            cfg[k] = float(v)
                        except ValueError:
                            cfg[k] = v
                transforms[ch] = cfg

        # Parse Hierarchy
        hierarchy = GatingHierarchy()
        hierarchy_el = root.find("GatingHierarchy")
        if hierarchy_el is not None:
            for gate_el in hierarchy_el.findall("Gate"):
                g_type = gate_el.get("type")
                g_name = gate_el.get("name")
                g_parent = gate_el.get("parent") or None
                ch_list = [c.get("name") for c in gate_el.findall("Channel")]

                if g_type == "polygon":
                    verts_el = gate_el.find("Vertices")
                    verts = json.loads(verts_el.text) if verts_el is not None and verts_el.text else []
                    hierarchy.add_gate(
                        PolygonGate(name=g_name, channels=ch_list, vertices=verts, parent_gate=g_parent)
                    )
                elif g_type == "rectangle":
                    bounds_el = gate_el.find("Bounds")
                    x_min = float(bounds_el.get("x_min")) if bounds_el is not None and bounds_el.get("x_min") else None
                    x_max = float(bounds_el.get("x_max")) if bounds_el is not None and bounds_el.get("x_max") else None
                    y_min = float(bounds_el.get("y_min")) if bounds_el is not None and bounds_el.get("y_min") else None
                    y_max = float(bounds_el.get("y_max")) if bounds_el is not None and bounds_el.get("y_max") else None
                    hierarchy.add_gate(
                        RectangleGate(
                            name=g_name,
                            channels=ch_list,
                            x_min=x_min,
                            x_max=x_max,
                            y_min=y_min,
                            y_max=y_max,
                            parent_gate=g_parent,
                        )
                    )
                elif g_type == "range":
                    rng_el = gate_el.find("Range")
                    min_v = float(rng_el.get("min_val")) if rng_el is not None and rng_el.get("min_val") else None
                    max_v = float(rng_el.get("max_val")) if rng_el is not None and rng_el.get("max_val") else None
                    hierarchy.add_gate(
                        RangeGate(name=g_name, channel=ch_list[0], min_val=min_v, max_val=max_v, parent_gate=g_parent)
                    )

        # Parse Plots
        plots = []
        plots_el = root.find("Plots")
        if plots_el is not None:
            for p_el in plots_el.findall("Plot"):
                plots.append(
                    {
                        "type": p_el.get("type", ""),
                        "x_channel": p_el.get("x_channel", ""),
                        "y_channel": p_el.get("y_channel") or None,
                        "gate_name": p_el.get("gate_name") or None,
                        "title": p_el.get("title") or None,
                        "output_file": p_el.get("output_file") or None,
                    }
                )

        return cls(
            name=session_name,
            sample_path=sample_path,
            sample_filename=sample_filename,
            total_events=total_events,
            channels=channels,
            compensation=comp,
            transforms=transforms,
            hierarchy=hierarchy,
            plot_configs=plots,
        )

    def replay(
        self,
        fcs_data: Optional[FCSData] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Replay the recorded workspace pipeline identically from raw data.

        Args:
            fcs_data: Optional pre-loaded FCSData. If None, reads from `self.sample_path`.

        Returns:
            Tuple of (processed_events_dataframe, population_statistics_dataframe).
        """
        if fcs_data is None:
            fcs_data = parse_fcs(self.sample_path)

        events_df = fcs_data.events_df.copy()

        # Step 1: Compensation
        if self.compensation:
            events_df = self.compensation.apply_dataframe(events_df)

        # Step 2: Transforms
        for ch, xform_cfg in self.transforms.items():
            if ch in events_df.columns:
                xform_obj = create_transform(xform_cfg)
                events_df[ch] = xform_obj.apply(events_df[ch].to_numpy())

        # Step 3: Gating Hierarchy Statistics
        stats_df = self.hierarchy.get_population_statistics(events_df)
        return events_df, stats_df
