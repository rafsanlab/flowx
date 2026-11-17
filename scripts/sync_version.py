"""Synchronize the package version from src/flowx/__init__.py into docs/myst.yml."""

from __future__ import annotations

import re
from pathlib import Path


def read_package_version(init_path: Path) -> str:
    """Extract __version__ string literal from src/flowx/__init__.py."""
    content = init_path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find __version__ in {init_path}")
    return match.group(1)


def sync_myst_version(myst_path: Path, version: str) -> None:
    """Update project.subtitle and site.options.logo_text in docs/myst.yml."""
    text = myst_path.read_text(encoding="utf-8")
    updated = re.sub(
        r"^(\s*subtitle:\s*).*$",
        rf"\g<1>Version {version}",
        text,
        flags=re.MULTILINE,
    )
    updated = re.sub(
        r"^(\s*logo_text:\s*).*$",
        rf"\g<1>FlowX Documentation v{version}",
        updated,
        flags=re.MULTILINE,
    )
    myst_path.write_text(updated, encoding="utf-8")


def main() -> None:
    """Run version synchronization from package init to MyST config."""
    root = Path(__file__).resolve().parent.parent
    version = read_package_version(root / "src" / "flowx" / "__init__.py")
    sync_myst_version(root / "docs" / "myst.yml", version)
    print(f"Synchronized docs/myst.yml to version {version}")


if __name__ == "__main__":
    main()
