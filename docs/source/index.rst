.. _home:

GeoPrior-v3
===========

.. rst-class:: hero-tagline

Physics-guided AI for geohazards and risk analytics.

GeoPrior-v3 is a scientific Python framework for building
physics-guided models for geohazard analysis, forecasting, and
risk-oriented interpretation. The current generation focuses on
**land subsidence** through **GeoPriorSubsNet v3.x**, while the broader
roadmap extends toward **landslides and other geohazard modeling
tasks**. The project combines scientific modeling,
configuration-driven workflows, staged CLI execution, and
reproducible figure generation in a single documentation space.

.. note::

   GeoPrior-v3 provides both a Python package interface and a
   command-line workflow. The project exposes dedicated CLI
   entry points for initialization, staged runs, builds, and
   plotting, including ``geoprior``, ``geoprior-run``,
   ``geoprior-build``, ``geoprior-plot``, and
   ``geoprior-init``. See :doc:`user_guide/cli` for the full
   command reference.

Start here
----------

.. grid:: 1 1 2 4
   :gutter: 3
   :class-container: cta-tiles

   .. grid-item-card:: Getting started
      :link: getting_started/index
      :link-type: doc
      :class-card: card--workflow

      Begin with the project overview, installation guidance,
      quickstart usage, and a first end-to-end run.

   .. grid-item-card:: Scientific foundations
      :link: scientific_foundations/index
      :link-type: doc
      :class-card: card--physics

      Understand the model family, the physical formulation,
      residual construction, scaling strategy, and the main
      scientific assumptions.

   .. grid-item-card:: User guide
      :link: user_guide/index
      :link-type: doc
      :class-card: card--configuration

      Follow the staged workflow, command-line logic,
      diagnostics, inference paths, and export-oriented use.

   .. grid-item-card:: Applications
      :link: applications/index
      :link-type: doc

      Explore forecasting workflows, tuning, calibration,
      uncertainty handling, and reproducibility scripts.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc
      :class-card: card--cli

      Browse the documented Python interfaces for parameters,
      CLI modules, subsidence components, tuners, utilities,
      and packaged resources.

   .. grid-item-card:: Developer notes
      :link: developer/index
      :link-type: doc

      See package layout, migration notes, and contribution
      guidance for maintaining and extending GeoPrior-v3.

   .. grid-item-card:: Release notes
      :link: release_notes
      :link-type: doc

      Track user-visible changes across versions, including
      workflow, API, scientific, and documentation updates.

Why GeoPrior-v3?
----------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Physics-guided by design
      :class-card: sd-border-1 sd-shadow-sm

      GeoPrior-v3 is built around physics-aware modeling
      rather than purely black-box forecasting. The platform
      emphasizes scientifically interpretable learning for
      subsidence analysis and broader geohazard settings.

   .. grid-item-card:: Workflow-oriented
      :class-card: sd-border-1 sd-shadow-sm

      The documentation is organized around how the project is
      actually used: initialization, staged execution,
      configuration, diagnostics, inference, plotting, and
      reproducible scientific outputs.

   .. grid-item-card:: Built for applications
      :class-card: sd-border-1 sd-shadow-sm

      The project is not only an API library. It is also an
      application framework with dedicated command-line entry
      points and figure-generation scripts for research and
      reporting pipelines.

   .. grid-item-card:: Structured for growth
      :class-card: sd-border-1 sd-shadow-sm

      The current flagship application is land subsidence,
      but the framework is positioned to expand toward broader
      geohazard workflows and future physics-guided hazard
      modeling tasks.

Quick start
-----------

Install the package and start from either Python or the CLI.

.. code-block:: bash

   pip install geoprior-v3

.. code-block:: python

   import geoprior as gp

.. code-block:: bash

   geoprior --help
   geoprior-init --help
   geoprior-run --help
   geoprior-build --help
   geoprior-plot --help

The package metadata presents GeoPrior-v3 as a Python package
for physics-guided geohazard modeling with land subsidence as
its current flagship application area. The command-line
workflow complements the Python API so the same project can be
used both programmatically and operationally.

.. seealso::

   - :doc:`getting_started/index`
   - :doc:`user_guide/index`
   - :doc:`applications/index`
   - :doc:`api/index`

Documentation map
-----------------

The documentation is organized into **seven main sections**
plus a shared bibliography.

1. **Getting started**
   introduces the project, installation, quickstart usage,
   and the first practical run.

2. **Scientific foundations**
   presents the model family, physical formulation,
   poroelastic background, residual construction, losses,
   scaling, mathematics, and identifiability assumptions.

3. **User guide**
   explains the staged workflow, command-line usage,
   configuration system, diagnostics, and export logic.

4. **Applications**
   focuses on subsidence forecasting, tuning workflows,
   uncertainty-oriented calibration, figure scripts, and
   reproducibility.

5. **API reference**
   documents the importable interfaces for parameters, the
   CLI, subsidence modules, tuner components, utilities, and
   packaged resources.

6. **Developer notes**
   collects package structure, migration notes, and
   contribution guidance.

7. **Release notes**
   record user-visible changes across versions, including
   workflow, API, scientific, and documentation changes.

The shared :doc:`references` page gathers the bibliography
used throughout the scientific and application-facing
documentation.

Navigate directly
-----------------

.. button-ref:: getting_started/index
   :ref-type: doc
   :color: primary
   :expand:

   Begin with getting started

.. button-ref:: scientific_foundations/index
   :ref-type: doc
   :color: secondary
   :expand:

   Explore the models and physics

.. button-ref:: user_guide/index
   :ref-type: doc
   :color: primary
   :expand:

   Open the user guide

.. button-ref:: applications/index
   :ref-type: doc
   :color: secondary
   :expand:

   See the application workflows

.. button-ref:: api/index
   :ref-type: doc
   :color: primary
   :expand:

   Browse the API reference

.. button-ref:: release_notes
   :ref-type: doc
   :color: secondary
   :expand:

   View release notes

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   scientific_foundations/index
   user_guide/index
   applications/index
   api/index
   developer/index
   release_notes
   references
