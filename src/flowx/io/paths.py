"""Path sanitization and privacy preservation utilities.

Converts absolute filesystem paths to clean, workspace-relative or anonymized paths
to prevent accidental disclosure of system usernames, account names, and workstation
directory structures in logs, reports, workspaces, manifests, and CLI outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def get_workspace_root() -> Path:
    """Detect the active project workspace root directory.

    Searches upward from the current working directory for repository root indicators
    (such as .git, src, or pytest.ini). Defaults to Path.cwd() if not found.
    """
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").is_file() or (candidate / "src").is_dir() or (candidate / ".git").is_dir() or (candidate / "pytest.ini").is_file():
            return candidate
    return cwd


def sanitize_path(
    path: Union[str, Path, None],
    base_dir: Optional[Union[str, Path]] = None,
    anonymize_home: bool = True,
) -> str:
    """Anonymize and relativize absolute filesystem paths for privacy and portability.

    Features:
    1. If inside the workspace root (or provided base_dir), converts to a clean relative path
       (e.g., 'data/sample.fcs', 'outputs/run_1/pipeline.log').
    2. If inside the user's home directory ($HOME), replaces the absolute home prefix with '~/'
       (e.g., '~/data/sample.fcs').
    3. Preserves relative paths as-is without expanding them to absolute paths.
    4. Handles None and empty inputs gracefully.

    Args:
        path: Path string or Path object to sanitize.
        base_dir: Optional directory to relativize against (defaults to detected workspace root).
        anonymize_home: Whether to replace user home directory with '~/' if outside workspace.

    Returns:
        Clean, portable, and privacy-safe path string.
    """
    if path is None:
        return ""

    path_str = str(path).strip()
    if not path_str:
        return ""

    p = Path(path_str)

    base = Path(base_dir).resolve() if base_dir else get_workspace_root()

    try:
        resolved = p.expanduser().resolve()
    except Exception:
        resolved = p

    # 1. If inside workspace / repository root, return relative to workspace
    try:
        rel = resolved.relative_to(base)
        return str(rel)
    except (ValueError, RuntimeError):
        pass

    # 2. If it was already a relative path, keep it relative and normalized
    if not p.is_absolute() and not path_str.startswith("~"):
        return os.path.normpath(path_str)

    # 3. If inside user's home directory, replace home prefix with '~/'
    if anonymize_home:
        try:
            home = Path.home().resolve()
            if resolved == home:
                return "~"
            if home in resolved.parents:
                rel_home = resolved.relative_to(home)
                return f"~/{rel_home}"
        except (ValueError, RuntimeError):
            pass

    return str(p)

