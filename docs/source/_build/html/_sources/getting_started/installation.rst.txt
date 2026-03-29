Installation
============

GeoPrior-v3 is distributed as a Python package for
**physics-guided geohazard modeling, forecasting, and risk
analytics**. The current release targets **Python 3.10 and
above** and is designed for scientific workflows that rely
on a modern numerical Python stack.

The package currently focuses on **land subsidence** through
GeoPriorSubsNet v3.x while remaining structured for broader
geohazard extensions. See :doc:`overview` for the project
scope and :doc:`quickstart` for the fastest way to begin.

Requirements
------------

GeoPrior-v3 currently declares the following baseline
requirements:

- Python 3.10 or newer
- NumPy
- pandas
- SciPy
- matplotlib
- tqdm
- scikit-learn
- joblib
- TensorFlow
- Keras
- statsmodels
- PyYAML
- platformdirs
- lz4
- psutil

These dependencies are the scientific and runtime foundation
used by the current package metadata.

.. note::

   GeoPrior-v3 is not a minimal pure-Python package. Its
   runtime environment includes the scientific stack needed
   for model execution, numerical processing, and the current
   TensorFlow/Keras-based workflows. If you are working in a
   clean environment, it is strongly recommended to install
   inside a dedicated virtual environment or Conda
   environment.

Create a virtual environment
----------------------------

Using an isolated environment is strongly recommended.

.. code-block:: bash

   python -m venv .venv

Activate it.

On Linux or macOS:

.. code-block:: bash

   source .venv/bin/activate

On Windows PowerShell:

.. code-block:: powershell

   .venv\Scripts\Activate.ps1

Upgrade packaging tools before installing GeoPrior-v3:

.. code-block:: bash

   python -m pip install --upgrade pip setuptools wheel

Install from PyPI
-----------------

If GeoPrior-v3 is published to your target package index, a
standard installation is:

.. code-block:: bash

   pip install geoprior-v3

After installation, you should be able to import the package
and inspect the CLI entry points.

.. code-block:: bash

   python -c "import geoprior as gp; print(gp.__version__)"
   geoprior --help
   geoprior-init --help

Install from source
-------------------

For active development or local testing, install directly
from the repository root.

Standard editable install:

.. code-block:: bash

   pip install -e .

If you prefer a regular local install without editable mode:

.. code-block:: bash

   pip install .

This is usually the best route while the package structure,
CLI, and documentation are evolving together.

Optional extras
---------------

GeoPrior-v3 currently defines optional dependency groups for
specialized workflows.

Install the optional ``kdiagram`` extra if you want the
bridged ``geoprior.kdiagram`` integration:

.. code-block:: bash

   pip install -e ".[kdiagram]"

or

.. code-block:: bash

   pip install "geoprior-v3[kdiagram]"

Install the development toolchain:

.. code-block:: bash

   pip install -e ".[dev]"

Install the documentation toolchain:

.. code-block:: bash

   pip install -e ".[docs]"

Install both development and documentation extras together:

.. code-block:: bash

   pip install -e ".[dev,docs]"

The current docs extra includes the Sphinx toolchain used by
this documentation site, including PyData Sphinx Theme,
MyST, sphinx-design, sphinx-copybutton, sphinx-gallery,
numpydoc, and sphinxcontrib-bibtex.

Install for documentation work
------------------------------

If your goal is to build or maintain the documentation, the
recommended setup is:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   pip install -e ".[docs]"

If you are also editing code, tests, or examples, use:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Build the documentation
-----------------------

From the repository root, build the HTML documentation with:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

Then open:

.. code-block:: text

   docs/build/html/index.html

The current documentation configuration uses:

- the **PyData Sphinx Theme**
- ``sphinx_design`` for layout components
- ``numpydoc`` and Napoleon-style parsing
- ``myst_parser`` for Markdown support
- ``sphinxcontrib.bibtex`` with ``references.bib``

This means the docs environment should be installed with the
declared ``docs`` extra rather than a minimal Sphinx-only
setup.

Verify the installation
-----------------------

A minimal verification checklist is:

.. code-block:: bash

   python -c "import geoprior as gp; print(gp.__version__)"
   geoprior --help
   geoprior-run --help
   geoprior-build --help
   geoprior-plot --help
   geoprior-init --help

A minimal Python check is also useful:

.. code-block:: python

   import geoprior as gp

   print(gp.__version__)

If the package imports successfully and the CLI help commands
resolve, the installation is generally in good shape for the
next steps.

What happens on import
----------------------

GeoPrior-v3 keeps import-time noise relatively low and tries
to degrade gracefully if some optional or runtime pieces are
missing. The package checks a list of expected dependencies
at import time and may emit an ``ImportWarning`` if some are
not available, rather than failing immediately in all cases.

It also reduces some warning and TensorFlow log noise by
default. This behavior is helpful during interactive use,
but it should not be treated as a substitute for a correct
environment setup.

.. warning::

   A successful ``import geoprior`` does not always guarantee
   that every workflow is fully ready. For example, training,
   staged CLI execution, and documentation builds may each
   exercise different parts of the dependency stack.

Common installation paths
-------------------------

**For normal package use**

.. code-block:: bash

   pip install geoprior-v3

**For local development**

.. code-block:: bash

   pip install -e ".[dev]"

**For documentation work**

.. code-block:: bash

   pip install -e ".[docs]"

**For full contributor setup**

.. code-block:: bash

   pip install -e ".[dev,docs,kdiagram]"

Troubleshooting
---------------

If installation or import fails, check the following first.

**Python version**

GeoPrior-v3 requires Python 3.10 or newer. Confirm this with:

.. code-block:: bash

   python --version

**Virtual environment activation**

Make sure the environment where you installed GeoPrior-v3 is
the same one from which you run ``python``, ``pip``, and the
CLI commands.

**Heavy scientific dependencies**

TensorFlow and related compiled dependencies can make fresh
environment setup slower or more fragile than lightweight
packages. If installation stalls or partially succeeds,
upgrade packaging tools first and retry inside a clean
environment.

**Docs build errors**

If Sphinx cannot find extensions such as
``pydata_sphinx_theme`` or ``sphinx_design``, install the
declared docs extra instead of installing Sphinx alone.

Next steps
----------

Once GeoPrior-v3 is installed, the best next pages are:

.. seealso::

   - :doc:`quickstart`
   - :doc:`first_project_run`
   - :doc:`../user_guide/cli`
   - :doc:`../user_guide/configuration`
   - :doc:`../user_guide/workflow_overview`