"""Panel configuration and marker-to-detector dictionary models.

Enables standardized, human-readable mapping between optical detector channels,
canonical biological markers, fluorophores, and functional gating roles.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

MarkerRole = Literal["lineage", "activation", "viability", "dump", "scatter", "reporter", "other"]


@dataclass
class MarkerChannel:
    """Specification of an optical channel in a flow cytometry panel.

    Attributes:
        detector: Instrument detector/channel name (e.g. 'PE-CF594-A' or '730/45 Red-A').
        marker: Canonical antigen or marker name (e.g. 'CD80', 'c-Kit', '7-AAD').
        fluorophore: Optical fluorophore tag (e.g. 'PE-CF594', 'AF700', 'FITC').
        role: Functional role ('lineage', 'activation', 'viability', 'dump', 'scatter', 'reporter').
        gate_range: Optional recommended gating range (min, max).
        default_cutoff: Optional default threshold cutoff.
    """

    detector: str
    marker: str
    fluorophore: Optional[str] = None
    role: MarkerRole = "lineage"
    gate_range: Optional[Tuple[float, float]] = None
    default_cutoff: Optional[float] = None


@dataclass
class PanelDefinition:
    """Comprehensive staining panel specification.

    Attributes:
        name: Name or identifier of panel.
        description: Panel description, biological application, or citation.
        cytometer: Cytometer instrument model (e.g. 'BD FACSCanto II').
        channels: List of configured MarkerChannel instances.
    """

    name: str
    description: str = ""
    cytometer: str = ""
    channels: List[MarkerChannel] = field(default_factory=list)

    def get_marker_map(self) -> Dict[str, str]:
        """Return dictionary mapping detector name to canonical marker name."""
        return {ch.detector: ch.marker for ch in self.channels}

    def get_reverse_marker_map(self) -> Dict[str, str]:
        """Return dictionary mapping canonical marker name to detector name."""
        return {ch.marker: ch.detector for ch in self.channels}

    def get_channels_by_role(self, role: MarkerRole) -> List[str]:
        """Return list of detector names matching the specified role."""
        return [ch.detector for ch in self.channels if ch.role == role]

    def get_markers_by_role(self, role: MarkerRole) -> List[str]:
        """Return list of marker names matching the specified role."""
        return [ch.marker for ch in self.channels if ch.role == role]

    def get_viability_channel(self) -> Optional[str]:
        """Return detector name designated as the live/dead viability exclusion dye."""
        for ch in self.channels:
            if ch.role == "viability":
                return ch.detector
        return None

    def get_dump_channels(self) -> List[str]:
        """Return list of detector channels assigned to lineage dump exclusion."""
        return self.get_channels_by_role("dump")

    def get_lineage_markers(self) -> List[str]:
        """Return list of canonical marker names designated for phenotypic clustering."""
        return [ch.marker for ch in self.channels if ch.role in ("lineage", "activation")]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PanelDefinition to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PanelDefinition:
        """Instantiate PanelDefinition from a dictionary."""
        ch_list = [MarkerChannel(**ch) if isinstance(ch, dict) else ch for ch in data.get("channels", [])]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            cytometer=data.get("cytometer", ""),
            channels=ch_list,
        )

    def save_json(self, filepath: Union[str, Path], indent: int = 2) -> None:
        """Save panel definition to JSON file."""
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> PanelDefinition:
        """Load panel definition from JSON file."""
        p = Path(filepath)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save_yaml(self, filepath: Union[str, Path]) -> None:
        """Save panel definition to YAML file."""
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml
            out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        except ImportError:
            # Fallback to JSON if pyyaml not installed
            self.save_json(out)

    @classmethod
    def load_yaml(cls, filepath: Union[str, Path]) -> PanelDefinition:
        """Load panel definition from YAML or JSON file."""
        p = Path(filepath)
        try:
            import yaml
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except ImportError:
            return cls.load_json(p)

