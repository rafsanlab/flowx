<p align="center">
  <img src="docs/images/flowx-logo-dark.svg" alt="FlowX Logo" width="120" />
</p>

<h1 align="center">FlowX</h1>

<p align="center">
  Modular, Automated Flow &amp; Spectral Cytometry Analysis Framework
</p>

---

**FlowX** is a modular, Python-native framework for conventional and full-spectrum flow cytometry preprocessing, quality control, compensation/unmixing, batch correction, unsupervised population discovery, cohort statistics, and MIFlowCyt-compliant reporting.

## Installation

Requires **Python >= 3.10** (automatically managed if using `uv` or the one-click installers below).

### One-Click Installers (Windows, macOS, Linux)

These installers automatically install [`uv`](https://docs.astral.sh/uv/), set up a `.venv` with Python 3.11, install core **FlowX**, and create two custom-icon shortcuts (**FlowX Terminal** and **FlowX Folder**) on your Desktop / Applications menu:

- **Windows:** Download and double-click [`installers/install-flowx.bat`](installers/install-flowx.bat)
- **macOS:** Download and double-click [`installers/install-flowx.command`](installers/install-flowx.command) *(if Gatekeeper prompts, right-click → Open)*
- **Linux (Ubuntu, etc.):** Run:
  ```bash
  curl -fsSL https://raw.githubusercontent.com/rafsanlab/flowx/main/installers/install-flowx.sh | bash
  ```
  Or download [`installers/install-flowx.sh`](installers/install-flowx.sh) and run `bash install-flowx.sh`.

### Using `uv`

```bash
# Editable install from local clone (into active virtual environment)
uv pip install -e .

# Include dev or optional PhenoGraph dependencies
uv pip install -e ".[dev,phenograph]"

# Or install the `flowx` CLI globally from local clone or GitHub
uv tool install -e .
uv tool install "git+https://github.com/rafsanlab/flowx.git"
```

### Using `pip`

```bash
# Editable install from local clone
pip install -e .

# Include dev or optional PhenoGraph dependencies
pip install -e ".[dev,phenograph]"

# Or install directly from GitHub
pip install "git+https://github.com/rafsanlab/flowx.git"
```

> **Note on Optional Dependencies:** `phenograph` pulls in GPL-licensed dependencies (`leidenalg`, `igraph`) and is kept as an optional extra (`[phenograph]`) separate from the permissively licensed core dependencies.

## Quickstart

```bash
# View available CLI commands
flowx --help

# Inspect a dataset cohort before running
flowx inspect --config configs/config_run/<platform>/<dataset>.yaml

# Run the full end-to-end pipeline
flowx pipeline --config configs/config_run/<platform>/<dataset>.yaml
```

## Documentation

Full documentation (Tutorials, How-To Guides, Reference, and Mathematical Explanations) is available in [`docs/README.md`](docs/README.md).

## License

FlowX is licensed under the **Business Source License 1.1** ([`LICENSE.txt`](LICENSE.txt)).

- **Additional Use Grant:** Production use is permitted solely for non-commercial academic, educational, or non-profit scientific research purposes. All other production use requires a commercial license from the Licensor.
- **Change Date:** `2030-11-17`
- **Change License:** **Mozilla Public License 2.0 (MPL-2.0)**
