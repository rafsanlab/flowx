"""Unit tests for data-driven adaptive morphological gating (Cells & Singlets)."""

from pathlib import Path
import numpy as np

from flowx.gating.adaptive_morphology import build_adaptive_cell_gate, build_adaptive_singlet_gate
from flowx.io.parser import parse_fcs


def test_adaptive_morphology_cell_and_singlet_gates(tqc_sample_fcs_path: Path):
    """Verify adaptive cell and singlet gate generation on raw events without tissue presets."""
    fcs = parse_fcs(tqc_sample_fcs_path)
    # 1. Cells
    cell_gate = build_adaptive_cell_gate(fcs, fcs.events_df)
    assert cell_gate.name == "Cells"
    assert len(cell_gate.vertices) >= 4
    mask_cells = cell_gate.apply(fcs.events_df)
    assert np.sum(mask_cells) > 0.5 * fcs.event_count

    # 2. Singlets
    singlet_gate = build_adaptive_singlet_gate(fcs, fcs.events_df, parent_mask=mask_cells)
    assert singlet_gate.name == "Singlets"
    assert len(singlet_gate.vertices) == 4
    mask_singlets = singlet_gate.apply(fcs.events_df)
    assert np.sum(mask_singlets) > 0.5 * fcs.event_count
