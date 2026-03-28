<div align="center">

  <a href="https://github.com/earthai-tech/geoprior-v3/actions/workflows/python-package-conda.yml">
     <img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/earthai-tech/geoprior-v3/python-package-conda.yml?branch=main&style=flat-square">
  </a>
  <img src="https://raw.githubusercontent.com/earthai-tech/geoprior-v3/main/docs/source/_static/geoprior-svg.svg" 
     alt="GeoPrior logo" width="320"><br>
  <h1>Physics-guided AI for Geohazards</h1>

  <p align="center">
    <!-- Docs (add when ready) -->
    <a href="https://earthai-tech.github.io/geoprior-v3/">
      <img alt="Docs" src="https://img.shields.io/badge/docs-coming%20soon-blue?style=flat-square">
    </a>
    <a href="https://github.com/earthai-tech/geoprior-v3/blob/main/LICENSE">
      <img alt="License" src="https://img.shields.io/github/license/earthai-tech/geoprior-v3?style=plastic&logo=Apache&logoColor=0492C2&color=FBB040">
    </a>
    <a href="https://github.com/psf/black">
      <img alt="Black" src="https://img.shields.io/badge/code%20style-Black-000000.svg?style=flat-square">
    </a>
    <a href="https://github.com/earthai-tech/geoprior-v3/blob/main/CONTRIBUTING.md">
      <img alt="Contributions" src="https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=flat-square">
    </a>
  </p>

  <p>
    <em>GeoPrior-v3</em> is an open, reproducible research codebase for
    <strong>physics-guided machine learning</strong> in <strong>geohazard forecasting</strong>
    and <strong>risk assessment</strong>. Today, it focuses on land subsidence via
    <strong>GeoPriorSubsNet</strong>; next, it expands toward landslides and broader
    geohazard regimes.
  </p>

</div>

-----------------------------------------------------

## ✨ Why GeoPrior?

GeoPrior is built to answer practical scientific questions:

- **Physics-consistent forecasting** — Can predictions respect governing constraints (e.g., consolidation / groundwater dynamics)?
- **Generalization across cities/regions** — Can one workflow adapt to different regimes with a consistent packaging strategy?
- **Uncertainty that supports decisions** — How reliable are forecast intervals, and where do risks concentrate?
- **Reproducibility** — Can reviewers and researchers re-run results end-to-end (GitHub + Code Ocean)?

-----------------------------------------------------

## 🔭 Scope & Roadmap

**Current focus**
- Land subsidence forecasting with **GeoPriorSubsNet v3.2**
- Physics residual monitoring + learned closures (e.g., effective parameters)

**Planned**
- Landslide forecasting modules (susceptibility + triggers + dynamics)
- Multi-hazard risk analytics (composable workflows and diagnostics)

-----------------------------------------------------

## 📥 Installation

### From PyPI (recommended)

```bash
pip install geoprior-v3
````

This installs GeoPrior and the scientific Python stack it depends on.
Python **3.10+** is supported **(available soon)**.

### Development install (editable)

```bash
git clone https://github.com/earthai-tech/geoprior-v3.git
cd geoprior-v3
pip install -e .[dev]
```

The `[dev]` extra installs testing and tooling dependencies

(e.g., pytest, coverage, black, ruff).

---

## ⚡ Quick Start

### 1) Import + logging

```python
import geoprior as gp
from geoprior.logging import get_logger, initialize_logging

initialize_logging(verbose=False)
log = get_logger(__name__)
log.info("GeoPrior logging is ready.")
```

### 2) Initialize a project config

GeoPrior now exposes a **family-based CLI**.
The root entry point is `geoprior`, and dedicated shortcuts are also
available for the `run`, `build`, and `plot` families.

```bash
geoprior-init --help
geoprior --help
geoprior-run --help
geoprior-build --help
```

To create `nat.com/config.py` interactively:

```bash
geoprior-init
```

### 3) Use the CLI families

The canonical root form is:

```bash
geoprior run <command> [args]
geoprior build <command> [args]
geoprior plot <command> [args]
```

Family-specific entry points avoid repeating the family name:

```bash
geoprior-run <command> [args]
geoprior-build <command> [args]
geoprior-plot <command> [args]
```

Examples:

```bash
geoprior run stage1-preprocess
geoprior-run stage2-train --help
geoprior build full-inputs-npz --help
geoprior-build physics-payload-npz --help
geoprior-run sm3-suite --preset tau50 --help
```

#### Compact CLI command table

| Family | Command | Purpose |
|---|---|---|
| `run` | `stage1-preprocess` | Stage-1 preprocessing and artifact export. |
| `run` | `stage2-train` | Train the main GeoPrior model. |
| `run` | `stage3-tune` | Hyperparameter tuning workflow. |
| `run` | `stage4-infer` | Inference, evaluation, and export from trained runs. |
| `run` | `stage5-transfer` | Cross-city transfer evaluation. |
| `run` | `sensitivity` | Run physics or lambda sensitivity workflows. |
| `run` | `sm3-identifiability` | Run one SM3 synthetic identifiability experiment. |
| `run` | `sm3-suite` | Launch preset SM3 regime suites such as `tau50` or `both50`. |
| `run` | `offset-diagnostics` | Run SM3 log-offset diagnostics. |
| `build` | `full-inputs-npz` | Merge Stage-1 split input NPZ files into one union NPZ. |
| `build` | `physics-payload-npz` | Export a physics payload NPZ from a trained model and inputs. |
| `build` | `external-validation-metrics` | Compute borehole or external validation metrics. |
| `build` | `external-validation-fullcity` | Build the full-city validation workflow end to end. |
| `build` | `assign-boreholes` | Assign validation boreholes to the nearest city cloud. |
| `build` | `add-zsurf-from-coords` | Merge elevation lookup tables and add `z_surf` to tabular data. |
| `build` | `sm3-collect-summaries` | Combine per-regime SM3 summary CSV files into one table. |

Typical usage:

```bash
geoprior run stage1-preprocess
geoprior run stage2-train
geoprior build full-inputs-npz
geoprior-build external-validation-metrics --help
geoprior-run sm3-suite --preset tau50
```

### 4) Reproduce paper-style workflows

A typical end-to-end workflow is organized as:

* Stage-1 preprocessing
* Stage-2 training
* Stage-3 tuning
* Stage-4 inference
* Stage-5 transfer evaluation
* supplementary build and diagnostic commands

Example:

```bash
geoprior run stage1-preprocess
geoprior run stage2-train
geoprior run stage4-infer --help
geoprior build external-validation-metrics --help
geoprior build sm3-collect-summaries --help
```


A slightly more concise version is:

### 3) Backward-compatible legacy scripts

For reproducibility and legacy workflows, helper scripts are 
still available under `scripts/`.

A typical workflow includes:

- preprocessing
- training
- evaluation and figure generation

These script-based entry points remain supported. For example:

```bash
python -m scripts plot-physics-fields --help
python -m scripts plot-sm3-identifiability --help
python -m scripts make-exposure --help
```
> For Code Ocean users: see `codeocean/README.md` and `codeocean/run.sh`.

---

## 🧪 Reproducibility

GeoPrior-v3 is designed for **end-to-end reproducible science**.

* Stable configs via `nat.com/config.py` and CLI overrides
* Deterministic pipeline stages exposed through `geoprior run ...`
* Reusable artifact builders exposed through `geoprior build ...`
* Capsule-friendly execution via `codeocean/`

If you publish results, please pin:

* the GeoPrior version (tag / release)
* config files used
* CLI commands invoked
* environment definition (e.g., `environment.yml`)

---

## 📚 Documentation

Documentation is under active development.

* Landing page (coming soon): `https://earthai-tech.github.io/geoprior-v3/`
* Meanwhile, start with:

  * `docs/reproducibility.md`
  * `geoprior --help`
  * `geoprior-run --help`
  * `geoprior-build --help`
  * inline API docstrings

---

## 💻 GeoPrior 3.0 Forecaster App

The **GeoPrior research core** is openly available to support
reproducible scientific research.

In addition, GeoPrior includes the **GeoPrior 3.0 Forecaster App**,
which offers a complete GUI-based application workflow built on top of
the research framework. This application version is maintained
separately from the public repository, which focuses on the core
scientific and reproducible modules.

For access to the **GeoPrior 3.0 Forecaster App**, please
**contact the author** directly.

* App tutorial: [https://youtu.be/JtOpX5lv4iw](https://youtu.be/JtOpX5lv4iw)
* Example simulation run: [https://youtu.be/nCouLQQFpQg](https://youtu.be/nCouLQQFpQg)

---

## 🙌 Contributing

Contributions are welcome.

1. Check the **Issues** tracker for bugs or ideas.
2. Fork the repository.
3. Create a new branch for your feature/fix.
4. Add tests when applicable.
5. Open a Pull Request.

Please see `CONTRIBUTING.md` for details.

---

## 📜 License

GeoPrior-v3 is distributed under the **Apache License 2.0**.
See `LICENSE` and `NOTICE` for details.

Note: GeoPrior-v3 may include a small number of files adapted from
other earthai-tech repositories under their original licenses
(e.g., BSD-3-Clause). See `third_party/licenses/`.

---

## 📞 Contact & Support

* **Bug reports & feature requests:** GitHub Issues
* **Author:** Laurent Kouadio — [https://lkouadio.com/](https://lkouadio.com/)
* **Email:** [etanoyau@gmail.com](mailto:etanoyau@gmail.com)

