"""Unit tests for GraphPad Prism table formatting and exports."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from flowx.stats.models import CohortFeatureSet
from flowx.stats.prism import export_prism_tables, format_prism_grouped_csv


def test_format_prism_grouped_csv_with_conditions():
    """Verify clean two-header Grouped layout for Prism."""
    samples = ["S1", "S2", "S3", "S4"]
    pops = ["CD4 T", "CD8 T", "B cells"]
    matrix_df = pd.DataFrame(
        [[10.5, 20.0, 5.0], [12.0, 18.5, 4.2], [5.0, 30.0, 15.0], [4.5, 32.0, 16.2]],
        index=samples,
        columns=pops,
    )
    meta = pd.DataFrame({"condition": ["Ctrl", "Ctrl", "Treat", "Treat"]}, index=samples)

    csv_text = format_prism_grouped_csv(matrix_df, sample_metadata=meta, condition_column="condition")
    lines = [line.strip() for line in csv_text.strip().split("\n")]

    # Check Header 1
    assert lines[0] == "Population,Ctrl,Ctrl,Treat,Treat"
    # Check Header 2
    assert lines[1] == "Sample_ID,S1,S2,S3,S4"
    # Check Row 1
    assert lines[2] == "CD4 T,10.5000,12.0000,5.0000,4.5000"


def test_format_prism_grouped_csv_without_conditions():
    """Verify fallback to Prism Column Table when condition is absent."""
    samples = ["S1", "S2"]
    pops = ["P1", "P2"]
    matrix_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=samples, columns=pops)

    csv_text = format_prism_grouped_csv(matrix_df, sample_metadata=None, condition_column=None)
    assert "Sample_ID" in csv_text
    assert "P1" in csv_text
    assert "P2" in csv_text


def test_export_prism_tables(tmp_path: Path):
    """Verify full table export including viable frequencies, counts, and MFI."""
    samples = ["S1", "S2"]
    pops = ["Cluster_0", "Cluster_1"]
    df = pd.DataFrame([[10, 20], [15, 25]], index=samples, columns=pops)
    meta = pd.DataFrame({"group": ["A", "B"]}, index=samples)

    mfi_df = pd.DataFrame([[100.0, 200.0], [150.0, 250.0]], index=samples, columns=pops)
    cfs = CohortFeatureSet(
        sample_ids=samples,
        population_names=pops,
        marker_names=["CD3"],
        count_matrix=df,
        parent_freq_matrix=df,
        viable_freq_matrix=df,
        mfi_tensor={"CD3": mfi_df},
        sample_metadata=meta,
    )

    out_dir = tmp_path / "prism_out"
    exported = export_prism_tables(cfs, output_dir=out_dir, condition_column="group")

    assert "prism_frequencies_viable" in exported
    assert "prism_frequencies_parent" in exported
    assert "prism_counts" in exported
    assert "prism_mfi_CD3" in exported

    for p in exported.values():
        assert p.is_file()
        assert p.parent.name == "prism"
        content = p.read_text()
        assert "A" in content
        assert "B" in content

