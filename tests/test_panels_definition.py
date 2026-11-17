"""Unit tests for PanelDefinition and MarkerChannel mapping models."""

from pathlib import Path
import pytest

from flowx.panels import MarkerChannel, PanelDefinition


def test_panel_definition_mappings():
    """Verify PanelDefinition builds correct detector and role mappings."""
    panel = PanelDefinition(
        name="THP1_4Color_Macrophage",
        description="Hydrogel stiffness macrophage activation panel",
        cytometer="BD FACSCanto II",
        channels=[
            MarkerChannel(detector="Horizon Brilliant Blue 515-A", marker="CD11b", fluorophore="HBB515", role="lineage"),
            MarkerChannel(detector="BV605-A", marker="CD206", fluorophore="BV605", role="activation"),
            MarkerChannel(detector="PE-A", marker="CD33", fluorophore="PE", role="lineage"),
            MarkerChannel(detector="PE-CF594-A", marker="CD80", fluorophore="PE-CF594", role="activation"),
        ],
    )

    det_map = panel.get_marker_map()
    assert det_map["BV605-A"] == "CD206"
    assert det_map["PE-CF594-A"] == "CD80"

    rev_map = panel.get_reverse_marker_map()
    assert rev_map["CD11b"] == "Horizon Brilliant Blue 515-A"
    assert rev_map["CD33"] == "PE-A"

    act_markers = panel.get_markers_by_role("activation")
    assert "CD206" in act_markers
    assert "CD80" in act_markers

    lineage_all = panel.get_lineage_markers()
    assert len(lineage_all) == 4


def test_panel_definition_json_roundtrip(tmp_path: Path):
    """Verify PanelDefinition serializes to and deserializes from JSON and YAML."""
    panel = PanelDefinition(
        name="Placenta_6Color_HSPC",
        channels=[
            MarkerChannel(detector="730/45 Red-A", marker="cKit", role="lineage"),
            MarkerChannel(detector="710/50 Blue-A", marker="7AAD", role="viability"),
            MarkerChannel(detector="APC-Cy7-A", marker="LineageDump", role="dump"),
        ],
    )

    assert panel.get_viability_channel() == "710/50 Blue-A"
    assert panel.get_dump_channels() == ["APC-Cy7-A"]

    json_file = tmp_path / "placenta_panel.json"
    panel.save_json(json_file)
    assert json_file.exists()

    loaded = PanelDefinition.load_json(json_file)
    assert loaded.name == "Placenta_6Color_HSPC"
    assert len(loaded.channels) == 3
    assert loaded.get_viability_channel() == "710/50 Blue-A"

