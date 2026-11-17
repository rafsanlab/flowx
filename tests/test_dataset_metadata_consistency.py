"""Tests that dataset metadata and config files align with raw FCS headers.

Runs deterministically against the self-contained `synthetic_conventional_dataset_dir`
fixture so CI and package builds require zero external `data/` files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import flowkit as fk
import pandas as pd
import yaml

from flowx.io.metadata import VALID_MARKER_CLASSES, load_marker_metadata


def _norm_cyt(name: str) -> str:
    """Normalize cytometer names for vendor-prefix/whitespace-tolerant matching."""
    cleaned = re.sub(r"^(bd|cytek|agilent|beckman)\s*[\s_]*", "", name.strip().lower())
    return re.sub(r"[^a-z0-9]", "", cleaned)


def _sample_fcs_files(dataset_dir: Path) -> list[Path]:
    """Return raw sample FCS files for a dataset directory."""
    samples_dir = dataset_dir / "raw" / "samples"
    if samples_dir.is_dir():
        return sorted(samples_dir.glob("*.fcs"))
    return sorted((dataset_dir / "raw").glob("*.fcs"))


def test_dataset_cytometer_matches_raw_fcs(synthetic_conventional_dataset_dir: Path) -> None:
    """Verify study_metadata.json and samples.csv cytometers match raw FCS $CYT."""
    dataset_dir = synthetic_conventional_dataset_dir
    sample_files = _sample_fcs_files(dataset_dir)
    assert len(sample_files) > 0

    fcs_cyt = str(fk.Sample(str(sample_files[0])).metadata.get("cyt", "") or "").strip()
    assert fcs_cyt != ""
    fcs_norm = _norm_cyt(fcs_cyt)

    samples_csv = dataset_dir / "metadata" / "samples.csv"
    assert samples_csv.is_file()
    df = pd.read_csv(samples_csv)
    for val in df["cytometer"].dropna().astype(str):
        val_norm = _norm_cyt(val)
        assert fcs_norm in val_norm or val_norm in fcs_norm, (
            f"{dataset_dir}: samples.csv cytometer '{val}' != FCS $CYT '{fcs_cyt}'"
        )

    study_json = dataset_dir / "metadata" / "study_metadata.json"
    assert study_json.is_file()
    meta = json.loads(study_json.read_text(encoding="utf-8"))
    sm_cyt = str(meta.get("cytometer", "") or "").strip()
    sm_norm = _norm_cyt(sm_cyt)
    assert fcs_norm in sm_norm or sm_norm in fcs_norm


def test_samples_csv_matches_raw_fcs(synthetic_conventional_dataset_dir: Path) -> None:
    """Verify samples.csv matches raw sample FCS filenames, paths, and event counts."""
    dataset_dir = synthetic_conventional_dataset_dir
    sample_files = _sample_fcs_files(dataset_dir)
    samples_csv = dataset_dir / "metadata" / "samples.csv"
    assert samples_csv.is_file()

    df = pd.read_csv(samples_csv)
    req_cols = {"sample_id", "filename", "relative_path", "condition", "batch_id", "is_control", "provenance_source"}
    assert req_cols.issubset(df.columns)

    actual_names = {f.name for f in sample_files}
    csv_names = set(df["filename"].astype(str))
    assert actual_names == csv_names

    for _, row in df.iterrows():
        fname = str(row["filename"])
        assert str(row["sample_id"]) == Path(fname).stem
        fpath = dataset_dir / str(row["relative_path"])
        assert fpath.is_file()
        if "total_events" in df.columns and pd.notna(row["total_events"]):
            sample = fk.Sample(str(fpath))
            assert int(row["total_events"]) == sample.event_count


def test_markers_csv_valid_and_matches_fcs(synthetic_conventional_dataset_dir: Path) -> None:
    """Verify markers.csv roles are valid and channel_name matches non-control sample FCS $PnN."""
    dataset_dir = synthetic_conventional_dataset_dir
    markers_csv = dataset_dir / "metadata" / "markers.csv"
    assert markers_csv.is_file()

    mdf = load_marker_metadata(markers_csv)
    for idx, roles in enumerate(mdf["marker_classes_list"]):
        assert roles, f"{dataset_dir}: empty marker_class at row {idx}"
        invalid = [r for r in roles if r not in VALID_MARKER_CLASSES]
        assert not invalid, f"{dataset_dir}: invalid marker_class {invalid} at row {idx}"

    samples_df = pd.read_csv(dataset_dir / "metadata" / "samples.csv")
    active = samples_df[samples_df["is_control"].astype(str).str.lower() != "true"]
    csv_channels = set(mdf["channel_name"].astype(str))
    for _, row in active.iterrows():
        fpath = dataset_dir / str(row["relative_path"])
        fcs_channels = set(fk.Sample(str(fpath)).channels["pnn"])
        assert fcs_channels == csv_channels


def test_config_aligns_with_dataset_and_metadata(synthetic_conventional_dataset_dir: Path) -> None:
    """Verify config file paths, groups, and gating channels align with dataset metadata."""
    dataset_dir = synthetic_conventional_dataset_dir
    platform = dataset_dir.parent.name
    root_dir = dataset_dir.parent.parent.parent
    cfg_path = root_dir / "configs" / "config_run" / platform / f"{dataset_dir.name}.yaml"
    assert cfg_path.is_file()

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    inp = cfg.get("input", {})
    assert inp.get("technology") == platform
    for key in ("fcs_file", "metadata_file", "markers_file"):
        assert inp.get(key) and Path(str(inp[key])).exists()

    samples_df = pd.read_csv(Path(str(inp["metadata_file"])))
    sa = cfg.get("06_statistical_analysis", {})
    if sa.get("enabled"):
        cond_col = sa.get("condition_column", "condition")
        assert cond_col in samples_df.columns
        conds = set(samples_df[cond_col].dropna().astype(str))
        for gk in ("reference_group", "target_group"):
            gv = sa.get(gk)
            if gv:
                assert str(gv) in conds

    comp = cfg.get("02_compensation", {})
    if comp.get("enabled") and comp.get("source") == "controls":
        cdir = comp.get("controls_dir")
        assert cdir and Path(str(cdir)).is_dir() and list(Path(str(cdir)).glob("*.fcs"))

    mdf = load_marker_metadata(Path(str(inp["markers_file"])))
    known_names = set(mdf["channel_name"].astype(str)) | set(mdf["marker_name"].astype(str))
    cg = cfg.get("03_cleanup_gating", {})
    for enabled_key, ch_key in (("enable_viability", "viability_channel"), ("dump_enabled", "dump_channel")):
        if cg.get(enabled_key):
            ch_val = str(cg.get(ch_key) or "").strip()
            assert ch_val in known_names
