"""Unit tests for path sanitization and privacy preservation utilities."""

from pathlib import Path
import pytest

from flowx.io.paths import get_workspace_root, sanitize_path


def test_sanitize_path_empty_and_none():
    """Verify None and empty strings produce empty string."""
    assert sanitize_path(None) == ""
    assert sanitize_path("") == ""
    assert sanitize_path("   ") == ""


def test_sanitize_path_workspace_relative():
    """Verify paths within workspace root are converted to clean relative paths."""
    ws_root = get_workspace_root()

    # Absolute path inside workspace
    abs_data_path = ws_root / "data" / "sample.fcs"
    assert sanitize_path(abs_data_path) == "data/sample.fcs"
    assert sanitize_path(str(abs_data_path)) == "data/sample.fcs"

    # Absolute path in output
    abs_output_path = ws_root / "outputs" / "run_20261012" / "report.md"
    assert sanitize_path(abs_output_path) == "outputs/run_20261012/report.md"

    # Relative paths preserved and normalized
    assert sanitize_path("data/sample.fcs") == "data/sample.fcs"
    assert sanitize_path("./data/sample.fcs") == "data/sample.fcs"
    assert sanitize_path(Path("outputs/run_1/pipeline.log")) == "outputs/run_1/pipeline.log"


def test_sanitize_path_custom_base_dir(tmp_path: Path):
    """Verify custom base_dir relativizes correctly."""
    target_file = tmp_path / "subfolder" / "manifest.json"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.touch()

    assert sanitize_path(target_file, base_dir=tmp_path) == "subfolder/manifest.json"


def test_sanitize_path_home_directory_anonymization(monkeypatch, tmp_path: Path):
    """Verify paths under $HOME outside workspace are anonymized to ~/."""
    fake_home = tmp_path / "fake_home"
    fake_ws = fake_home / "repos" / "project"
    fake_ws.mkdir(parents=True, exist_ok=True)
    outside_home_dir = fake_home / "documents" / "clinical"
    outside_home_dir.mkdir(parents=True, exist_ok=True)
    file_path = outside_home_dir / "patient_01.fcs"
    file_path.touch()

    # Test with custom base_dir and fake home
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    result = sanitize_path(file_path, base_dir=fake_ws, anonymize_home=True)
    assert result == "~/documents/clinical/patient_01.fcs"

    # With anonymize_home=False, it retains the absolute path
    result_raw = sanitize_path(file_path, base_dir=fake_ws, anonymize_home=False)
    assert result_raw == str(file_path)


def test_sanitize_path_external_non_home_path(tmp_path: Path):
    """Verify paths outside workspace and outside home remain valid paths."""
    ws_root = get_workspace_root()
    # tmp_path is under /tmp, which is outside workspace and outside /home
    temp_file = tmp_path / "isolated.fcs"
    temp_file.touch()

    res = sanitize_path(temp_file, base_dir=ws_root, anonymize_home=True)
    # Must preserve the full path so pytest tmp_path assertions remain valid
    assert res == str(temp_file)
    assert Path(res).is_file()


def test_get_workspace_root():
    """Verify workspace root locator detects repository root containing src/."""
    ws_root = get_workspace_root()
    assert (ws_root / "src").is_dir()
    assert ws_root.is_dir()
