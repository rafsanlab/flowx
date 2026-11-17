"""Unit tests for workspace session persistence and replay."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from flowx.gating.gate import GatingHierarchy, RangeGate, RectangleGate
from flowx.math.compensation import CompensationMatrix
from flowx.math.transforms import LogicleTransform
from flowx.workspace.session import WorkspaceSession


def test_workspace_json_roundtrip(tmp_path: Path):
    """Verify saving and loading WorkspaceSession via JSON."""
    session = WorkspaceSession(
        name="TestSession",
        sample_path="sample.fcs",
        sample_filename="sample.fcs",
        total_events=1000,
    )
    session.set_compensation(CompensationMatrix(["CH1", "CH2"], np.eye(2)))
    session.add_transform("CH1", LogicleTransform())

    gate = RectangleGate(name="TestGate", channels=["CH1"], x_min=0.5)
    session.add_gate(gate)

    json_file = tmp_path / "test_workspace.json"
    session.save_json(json_file)

    loaded = WorkspaceSession.load_json(json_file)
    assert loaded.name == "TestSession"
    assert loaded.total_events == 1000
    assert loaded.compensation is not None
    assert loaded.compensation.detectors == ["CH1", "CH2"]
    assert "CH1" in loaded.transforms
    assert len(loaded.hierarchy.order) == 1
    assert loaded.hierarchy.order[0] == "TestGate"


def test_workspace_xml_roundtrip(tmp_path: Path):
    """Verify saving and loading WorkspaceSession via XML."""
    session = WorkspaceSession(
        name="XMLSession",
        sample_path="data.fcs",
        sample_filename="data.fcs",
        total_events=500,
    )
    session.set_compensation(CompensationMatrix(["A", "B"], np.eye(2)))
    gate = RangeGate(name="R_Gate", channel="A", min_val=10.0, max_val=20.0)
    session.add_gate(gate)

    xml_file = tmp_path / "test_workspace.xml"
    session.save_xml(xml_file)

    loaded = WorkspaceSession.load_xml(xml_file)
    assert loaded.name == "XMLSession"
    assert loaded.total_events == 500
    assert loaded.compensation is not None
    assert loaded.compensation.detectors == ["A", "B"]
    assert len(loaded.hierarchy.order) == 1
    assert loaded.hierarchy.order[0] == "R_Gate"
