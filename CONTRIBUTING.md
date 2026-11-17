# Contributing to FlowX

Thank you for helping improve **FlowX**! Whether you are fixing a bug, improving documentation, or adding a new bioinformatics feature, please follow these guidelines.

## 1. Contributor License Agreement (CLA)
* Before we can merge any Pull Request (PR), you must sign our [Contributor License Agreement](CLA.md).
* When you open a PR, our automated bot will post a link allowing you to sign it in one click with your GitHub account.

## 2. Dependency & License Hygiene
* **Discuss new dependencies first:** Please open an Issue before adding any new third-party Python packages to `pyproject.toml` / `requirements.txt`.
* **No copyleft dependencies:** Do not introduce libraries licensed under **GPL** or **AGPL**. All dependencies must use permissive licenses (such as MIT, BSD, or Apache 2.0) to remain compatible with FlowX's licensing.

## 3. Development & Testing
1. Fork the repository and create a new branch from `main` (`git checkout -b fix/your-feature`).
2. Install development dependencies and run the test suite:
   ```bash
   pip install -e ".[dev]"
   pytest
   ```
