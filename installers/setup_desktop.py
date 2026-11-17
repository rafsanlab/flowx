#!/usr/bin/env python3
"""Create OS-native desktop and application shortcuts for FlowX.

Creates two shortcuts with custom FlowX icons:
1. FlowX Terminal: opens a terminal with the `.venv` activated inside the FlowX directory.
2. FlowX Folder: opens the FlowX installation directory in the file manager.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _setup_linux(flowx_dir: Path) -> None:
    svg_icon = flowx_dir / "docs" / "images" / "flowx-logo.svg"
    png_icon = flowx_dir / "installers" / "assets" / "flowx.png"
    icon_path = svg_icon if svg_icon.exists() else png_icon

    # Store helper script outside the git repo so working tree stays clean
    helper_dir = Path.home() / ".local" / "share" / "flowx"
    helper_dir.mkdir(parents=True, exist_ok=True)
    launcher_sh = helper_dir / "launch-terminal.sh"

    quoted_dir = shlex.quote(str(flowx_dir))
    inner_cmd = (
        f"cd {quoted_dir} && "
        f"source {shlex.quote(str(flowx_dir / '.venv' / 'bin' / 'activate'))} && "
        f"echo 'FlowX environment ready. Run: flowx --help' && "
        f"exec \"${{SHELL:-/bin/bash}}\" -i"
    )
    quoted_inner = shlex.quote(inner_cmd)

    launcher_sh.write_text(
        f"""#!/usr/bin/env bash
set -e
if command -v gnome-terminal >/dev/null 2>&1; then
  exec gnome-terminal --working-directory={quoted_dir} -- bash -c {quoted_inner}
elif command -v konsole >/dev/null 2>&1; then
  exec konsole --workdir {quoted_dir} -e bash -c {quoted_inner}
elif command -v xfce4-terminal >/dev/null 2>&1; then
  exec xfce4-terminal --working-directory={quoted_dir} -e "bash -c {quoted_inner}"
elif command -v mate-terminal >/dev/null 2>&1; then
  exec mate-terminal --working-directory={quoted_dir} -e "bash -c {quoted_inner}"
elif command -v alacritty >/dev/null 2>&1; then
  exec alacritty --working-directory {quoted_dir} -e bash -c {quoted_inner}
elif command -v kitty >/dev/null 2>&1; then
  exec kitty --directory {quoted_dir} bash -c {quoted_inner}
elif command -v x-terminal-emulator >/dev/null 2>&1; then
  exec x-terminal-emulator -e bash -c {quoted_inner}
elif command -v xterm >/dev/null 2>&1; then
  exec xterm -e bash -c {quoted_inner}
else
  echo "No supported terminal emulator found." >&2
  exit 1
fi
""",
        encoding="utf-8",
    )
    _make_executable(launcher_sh)

    terminal_desktop = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=FlowX Terminal
Comment=Open a terminal with the FlowX virtual environment activated
Exec={shlex.quote(str(launcher_sh))}
Icon={icon_path}
Terminal=false
Categories=Science;Education;Development;
"""

    folder_desktop = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=FlowX Folder
Comment=Open the FlowX installation directory
Exec=xdg-open {quoted_dir}
Icon={icon_path}
Terminal=false
Categories=Science;Utility;
"""

    apps_dir = Path.home() / ".local" / "share" / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)

    desktop_dir = Path.home() / "Desktop"
    if shutil.which("xdg-user-dir"):
        res = subprocess.run(
            ["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            desktop_dir = Path(res.stdout.strip())

    entries = {
        "flowx-terminal.desktop": terminal_desktop,
        "flowx-folder.desktop": folder_desktop,
    }

    for filename, content in entries.items():
        app_file = apps_dir / filename
        app_file.write_text(content, encoding="utf-8")
        _make_executable(app_file)
        print(f"Created application entry: {app_file}")

        if desktop_dir.exists():
            desk_file = desktop_dir / filename
            desk_file.write_text(content, encoding="utf-8")
            _make_executable(desk_file)
            if shutil.which("gio"):
                subprocess.run(
                    ["gio", "set", str(desk_file), "metadata::trusted", "true"],
                    capture_output=True,
                    check=False,
                )
            print(f"Created desktop shortcut:  {desk_file}")


def _write_macos_app(
    app_path: Path,
    name: str,
    bundle_id: str,
    icns_src: Path,
    script_body: str,
) -> None:
    if app_path.exists():
        shutil.rmtree(app_path)

    contents = app_path / "Contents"
    macos_dir = contents / "MacOS"
    res_dir = contents / "Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    if icns_src.exists():
        shutil.copy2(icns_src, res_dir / "flowx.icns")

    plist_data = {
        "CFBundleExecutable": name,
        "CFBundleIconFile": "flowx.icns",
        "CFBundleIdentifier": bundle_id,
        "CFBundleName": name,
        "CFBundleDisplayName": name,
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "1.0",
        "CFBundleShortVersionString": "1.0",
    }
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(plist_data, f)

    exe_path = macos_dir / name
    exe_path.write_text(script_body, encoding="utf-8")
    _make_executable(exe_path)


def _setup_macos(flowx_dir: Path) -> None:
    icns_src = flowx_dir / "installers" / "assets" / "flowx.icns"
    apps_dir = Path.home() / "Applications"
    apps_dir.mkdir(parents=True, exist_ok=True)

    # Escape path for AppleScript double-quoted string inside shell
    escaped_dir = str(flowx_dir).replace("\\", "\\\\").replace('"', '\\"')
    term_script = f"""#!/usr/bin/env bash
osascript <<'EOF'
tell application "Terminal"
    activate
    do script "cd \\"{escaped_dir}\\" && source .venv/bin/activate && echo 'FlowX environment ready. Run: flowx --help'"
end tell
EOF
"""

    folder_script = f"""#!/usr/bin/env bash
open {shlex.quote(str(flowx_dir))}
"""

    term_app = apps_dir / "FlowX Terminal.app"
    folder_app = apps_dir / "FlowX Folder.app"

    _write_macos_app(
        term_app, "FlowX Terminal", "org.flowx.terminal", icns_src, term_script
    )
    _write_macos_app(
        folder_app, "FlowX Folder", "org.flowx.folder", icns_src, folder_script
    )
    print(f"Created app bundle: {term_app}")
    print(f"Created app bundle: {folder_app}")

    desktop_dir = Path.home() / "Desktop"
    if desktop_dir.exists():
        for app in (term_app, folder_app):
            link = desktop_dir / app.name
            if link.is_symlink() or link.exists():
                if link.is_dir() and not link.is_symlink():
                    shutil.rmtree(link)
                else:
                    link.unlink()
            link.symlink_to(app)
            print(f"Created desktop link: {link}")


def _setup_windows(flowx_dir: Path) -> None:
    ico_path = flowx_dir / "installers" / "assets" / "flowx.ico"
    activate_bat = flowx_dir / ".venv" / "Scripts" / "activate.bat"

    def ps_escape(val: str) -> str:
        return val.replace("'", "''")

    term_args = (
        f'/k ""{activate_bat}" && cd /d "{flowx_dir}" && '
        f'echo FlowX environment ready. Run: flowx --help"'
    )

    ps_script = f"""
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell

$desktop = [Environment]::GetFolderPath('Desktop')
$programs = Join-Path ([Environment]::GetFolderPath('Programs')) 'FlowX'
New-Item -ItemType Directory -Force -Path $programs | Out-Null

$targets = @($desktop, $programs)
foreach ($dir in $targets) {{
    $termLnk = Join-Path $dir 'FlowX Terminal.lnk'
    $s1 = $shell.CreateShortcut($termLnk)
    $s1.TargetPath = 'cmd.exe'
    $s1.Arguments = '{ps_escape(term_args)}'
    $s1.WorkingDirectory = '{ps_escape(str(flowx_dir))}'
    $s1.IconLocation = '{ps_escape(str(ico_path))}'
    $s1.Description = 'Open a terminal with the FlowX virtual environment activated'
    $s1.Save()

    $folderLnk = Join-Path $dir 'FlowX Folder.lnk'
    $s2 = $shell.CreateShortcut($folderLnk)
    $s2.TargetPath = 'explorer.exe'
    $s2.Arguments = '{ps_escape(str(flowx_dir))}'
    $s2.WorkingDirectory = '{ps_escape(str(flowx_dir))}'
    $s2.IconLocation = '{ps_escape(str(ico_path))}'
    $s2.Description = 'Open the FlowX installation directory'
    $s2.Save()
}}
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=True,
    )
    print("Created Windows Desktop and Start Menu shortcuts for FlowX Terminal and FlowX Folder.")


def main() -> None:
    flowx_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    system = sys.platform

    if system.startswith("linux"):
        _setup_linux(flowx_dir)
    elif system == "darwin":
        _setup_macos(flowx_dir)
    elif system in ("win32", "cygwin"):
        _setup_windows(flowx_dir)
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


if __name__ == "__main__":
    main()
