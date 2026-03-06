<div align="center">

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

### 2) Use the CLI

```bash
geoprior --help
```

### 3) Reproduce paper-style runs (example)

We ship reproducibility scripts under `scripts/`.
A typical end-to-end run is organized as:

* dataset / preprocessing
* training
* evaluation + figure generation

Example:

```bash
python -m scripts plot-physics-fields  --help
python -m scripts plot-sm3-identifiability --help
python -m scripts make-exposure --help
```

> For Code Ocean users: see `codeocean/README.md` and `codeocean/run.sh`.

---

## 🧪 Reproducibility

GeoPrior-v3 is designed for **end-to-end reproducible science**.

* Stable configs in `configs/`
* Deterministic/controlled pipelines in `scripts/`
* Capsule-friendly execution via `codeocean/`

If you publish results, please pin:

* the GeoPrior version (tag / release)
* config files used
* environment definition (e.g., `environment.yml`)

---

## 📚 Documentation

Documentation is under active development.

* Landing page (coming soon): `https://earthai-tech.github.io/geoprior-v3/`
* Meanwhile, start with:

  * `docs/reproducibility.md`
  * `scripts/` entrypoints
  * inline API docstrings

---

## 💻 Premium GUI (separate/private)

GeoPrior’s **research core** is open-source.
The **GUI/application layer** is developed separately (private),
so the public repository remains focused on reproducible science.

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

