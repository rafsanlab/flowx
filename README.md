<p align="center">
  <img src="docs/images/flowx-logo-dark.svg" alt="FlowX Logo" width="120" />
</p>

<h1 align="center">FlowX</h1>

<p align="center">
  Modular, Automated Flow &amp; Spectral Cytometry Analysis Framework
</p>

---

**FlowX** is an open-source, Python-native framework for conventional and full-spectrum flow cytometry preprocessing, quality control, compensation/unmixing, batch correction, unsupervised population discovery, cohort statistics, and MIFlowCyt-compliant reporting.

## Installation

Requires **Python >= 3.10**.

### Using `uv`

```bash
# Editable install from local clone (into active virtual environment)
uv pip install -e .

# Include dev or optional PhenoGraph dependencies
uv pip install -e ".[dev,phenograph]"

# Or install the `flowx` CLI globally from local clone or GitHub
uv tool install -e .
uv tool install "git+https://github.com/rafsanlab/antigravity-test.git"
```

### Using `pip`

```bash
# Editable install from local clone
pip install -e .

# Include dev or optional PhenoGraph dependencies
pip install -e ".[dev,phenograph]"

# Or install directly from GitHub
pip install "git+https://github.com/rafsanlab/antigravity-test.git"
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
