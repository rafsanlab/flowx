#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/rafsanlab/flowx.git"
ARCHIVE_URL="https://github.com/rafsanlab/flowx/archive/refs/heads/main.tar.gz"
DEFAULT_INSTALL_DIR="${FLOWX_INSTALL_DIR:-$HOME/flowx}"

echo "=== FlowX Installer (Linux) ==="

# 1. Ensure uv is installed
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[1/4] Installing uv..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Error: curl or wget is required to install uv." >&2
    exit 1
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
  echo "[1/4] uv is already installed: $(command -v uv)"
fi

# 2. Locate or download FlowX repository
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
if [ -f "$SCRIPT_DIR/../pyproject.toml" ] && grep -q 'name = "flowx"' "$SCRIPT_DIR/../pyproject.toml" 2>/dev/null; then
  FLOWX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
  echo "[2/4] Using local FlowX repository at: $FLOWX_DIR"
else
  FLOWX_DIR="$DEFAULT_INSTALL_DIR"
  if [ -d "$FLOWX_DIR" ] && [ -f "$FLOWX_DIR/pyproject.toml" ]; then
    echo "[2/4] FlowX directory already exists at: $FLOWX_DIR"
  elif command -v git >/dev/null 2>&1; then
    echo "[2/4] Cloning FlowX into $FLOWX_DIR..."
    git clone "$REPO_URL" "$FLOWX_DIR"
  else
    echo "[2/4] Downloading FlowX archive into $FLOWX_DIR..."
    TMP_TAR="$(mktemp)"
    curl -fsSL "$ARCHIVE_URL" -o "$TMP_TAR"
    mkdir -p "$FLOWX_DIR"
    tar -xzf "$TMP_TAR" --strip-components=1 -C "$FLOWX_DIR"
    rm -f "$TMP_TAR"
  fi
fi

# 3. Create .venv (uv downloads Python 3.11 automatically if needed) & install core flowx
cd "$FLOWX_DIR"
echo "[3/4] Setting up virtual environment and installing core FlowX..."
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .

# 4. Create desktop/application shortcuts with custom icons
echo "[4/4] Creating desktop and application shortcuts..."
.venv/bin/python installers/setup_desktop.py "$FLOWX_DIR"

echo ""
echo "FlowX installation complete!"
echo "  Directory: $FLOWX_DIR"
echo "  Shortcuts: 'FlowX Terminal' and 'FlowX Folder' added to your Desktop / Applications."
