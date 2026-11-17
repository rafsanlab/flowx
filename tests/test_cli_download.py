"""Tests for FlowX dataset management and download CLI commands."""

import json
from pathlib import Path

import pytest

from flowx.cli import build_parser, main
from flowx.datasets import (
    BENCHMARK_DATASETS,
    DatasetInfo,
    get_dataset,
    list_datasets,
    resolve_datasets,
    verify_dataset,
)


def test_registry_catalogue_counts():
    """Verify registry contains 20 benchmark datasets (11 conv, 9 spectral)."""
    assert len(BENCHMARK_DATASETS) == 20
    conv = list_datasets("conventional")
    spec = list_datasets("spectral")
    assert len(conv) == 11
    assert len(spec) == 9


def test_registry_lookup_by_id_and_alias():
    """Verify lookups by ID, common aliases, and numeric index."""
    # Lookup by ID
    d1 = get_dataset("coburn_2021_thp1_stiffness")
    assert d1 is not None
    assert d1.category == "conventional"

    # Lookup by alias
    d2 = get_dataset("coburn")
    assert d2 is not None
    assert d2.id == "coburn_2021_thp1_stiffness"

    d3 = get_dataset("scenith")
    assert d3 is not None
    assert d3.id == "arguello_2020_scenith_metabolism"
    assert d3.category == "conventional"

    # Lookup by numeric selector
    d4 = get_dataset("3")
    assert d4 is not None

    # Invalid query
    assert get_dataset("non_existent_dataset_xyz") is None


def test_resolve_datasets_query():
    """Verify resolution of comma-separated queries and 'all' keyword."""
    resolved = resolve_datasets(["coburn,weber"])
    assert len(resolved) == 2
    ids = {d.id for d in resolved}
    assert "coburn_2021_thp1_stiffness" in ids
    assert "weber_2025_bk_polyomavirus" in ids

    # Filter by category
    conv_all = resolve_datasets(["all"], category="conventional")
    assert len(conv_all) == 11

    spec_all = resolve_datasets(["all"], category="spectral")
    assert len(spec_all) == 9


def test_verify_dataset_local():
    """Verify local verification on existing benchmark dataset."""
    data_dir = Path("data")
    coburn_info = get_dataset("coburn")
    assert coburn_info is not None

    res = verify_dataset(coburn_info, data_dir)
    assert res.exists is True
    assert res.status == "Ready"
    assert res.total_fcs == 9
    assert res.manifest_valid is True

    # Fake missing dataset
    fake_info = DatasetInfo(
        id="nonexistent_dataset_123",
        name="Fake",
        category="conventional",
    )
    fake_res = verify_dataset(fake_info, data_dir)
    assert fake_res.exists is False
    assert fake_res.status == "Missing"


def test_cli_parser_download_subcommand():
    """Test argument parser configurations for download and dataset alias."""
    parser = build_parser()

    # download --list
    args = parser.parse_args(["download", "--list"])
    assert args.command == "download"
    assert args.list is True

    # dataset alias with verify-only and category
    args2 = parser.parse_args(["dataset", "-c", "spectral", "--verify-only"])
    assert args2.command == "dataset"
    assert args2.category == "spectral"
    assert args2.verify_only is True

    # dataset with specific targets and info
    args3 = parser.parse_args(["download", "coburn", "scenith", "--info", "--json"])
    assert args3.datasets == ["coburn", "scenith"]
    assert args3.info is True
    assert args3.json is True
    assert args3.include_supplementary is False

    # download with --include-supplementary
    args4 = parser.parse_args(["download", "coburn", "--include-supplementary"])
    assert args4.include_supplementary is True


def test_cli_download_list_execution(capsys):
    """Test executing flowx download --list."""
    ret = main(["download", "--list", "-c", "conventional"])
    assert ret == 0


def test_cli_download_info_execution(capsys):
    """Test executing flowx download coburn --info."""
    ret = main(["download", "coburn", "--info"])
    assert ret == 0


def test_cli_download_json_execution(capsys):
    """Test executing flowx download coburn --info --json."""
    ret = main(["download", "coburn", "--info", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == "coburn_2021_thp1_stiffness"
    assert data[0]["verification"]["status"] == "Ready"


def test_cli_download_verify_only(capsys):
    """Test executing flowx download coburn --verify-only."""
    ret = main(["download", "coburn", "--verify-only"])
    assert ret == 0


def test_cli_download_invalid_dataset(capsys):
    """Test executing flowx download with invalid dataset query returns error code 1."""
    ret = main(["download", "completely_unknown_dataset_xyz"])
    assert ret == 1


def test_download_dataset_include_supplementary_default(monkeypatch, tmp_path):
    """Verify download_dataset defaults include_supplementary=False and forwards it."""
    from flowx.datasets import downloader

    calls = []

    def fake_handler(data_dir, verify_only=False, include_supplementary=False):
        calls.append((data_dir, verify_only, include_supplementary))

    monkeypatch.setattr(downloader, "_get_download_dispatcher", lambda *_args: fake_handler)

    info = DatasetInfo(id="coburn_2021_thp1_stiffness", name="THP1", category="conventional")
    assert downloader.download_dataset(info, tmp_path) is True
    assert downloader.download_dataset(info, tmp_path, include_supplementary=True) is True
    assert calls == [
        (tmp_path.resolve(), False, False),
        (tmp_path.resolve(), False, True),
    ]


def test_download_supplementary_file_disabled_by_default(monkeypatch, tmp_path):
    """Verify download_supplementary_file skips downloads unless include_supplementary=True."""
    from scripts.datasets import download_conventional_datasets as dconv
    from scripts.datasets import download_spectral_datasets as dspec

    def fail_get(*_args, **_kwargs):
        raise AssertionError("Network request should not be made when include_supplementary=False")

    monkeypatch.setattr(dconv.requests.Session, "get", fail_get)
    monkeypatch.setattr(dspec.requests.Session, "get", fail_get)

    conv_dest = tmp_path / "conv_supp.wsp"
    spec_dest = tmp_path / "spec_supp.wsp"

    assert dconv.download_supplementary_file("https://example.com/a.wsp", conv_dest) is False
    assert dspec.download_supplementary_file("https://example.com/b.wsp", spec_dest) is False
    assert not conv_dest.exists()
    assert not spec_dest.exists()

