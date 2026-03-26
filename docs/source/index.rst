.. _home:

GeoPrior-v3
===========

Physics-guided AI for geohazards and risk analytics.

GeoPrior-v3 is a scientific Python framework for building
physics-guided models for geohazard analysis, forecasting,
and risk-oriented interpretation. The current generation
focuses on **land subsidence** through **GeoPriorSubsNet v3.x**,
while the broader roadmap extends toward **landslides and
other geohazard modeling tasks**. The project combines
scientific modeling, configuration-driven workflows, staged
CLI execution, and reproducible figure generation in a
single documentation space.

.. note::

   GeoPrior-v3 provides both a Python package interface and a
   command-line workflow. The project exposes dedicated CLI
   entry points for initialization, staged runs, builds, and
   plotting, including ``geoprior``, ``geoprior-run``,
   ``geoprior-build``, ``geoprior-plot``, and
   ``geoprior-init``. See :doc:`user_guide/cli` for the full
   command reference.

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started/overview
      :link-type: doc
      :class-card: sd-shadow-sm

      Start with the project overview, installation steps,
      and a minimal first run.

   .. grid-item-card:: Scientific foundations
      :link: scientific_foundations/models_overview
      :link-type: doc
      :class-card: sd-shadow-sm

      Understand the model family, the physics formulation,
      residual construction, scaling strategy, and
      identifiability assumptions.

   .. grid-item-card:: User guide
      :link: user_guide/workflow_overview
      :link-type: doc
      :class-card: sd-shadow-sm

      Follow the staged workflow, configuration logic,
      diagnostics, inference, and export paths.

   .. grid-item-card:: Applications
      :link: applications/subsidence_forecasting
      :link-type: doc
      :class-card: sd-shadow-sm

      Explore land subsidence forecasting, tuning workflows,
      uncertainty handling, and reproducibility scripts.

   .. grid-item-card:: API reference
      :link: api/cli
      :link-type: doc
      :class-card: sd-shadow-sm

      Browse the documented Python interfaces for CLI,
      subsidence models, tuner components, utilities,
      and resources.

   .. grid-item-card:: Developer notes
      :link: developer/package_structure
      :link-type: doc
      :class-card: sd-shadow-sm

      See package layout, migration notes, and contribution
      guidance for maintaining and extending GeoPrior-v3.

   .. grid-item-card:: Release notes
      :link: release_notes
      :link-type: doc
      :class-card: sd-shadow-sm

      Track user-visible changes across versions, including
      workflow, API, scientific, and documentation updates.

Why GeoPrior-v3?
----------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Physics-guided by design
      :class-card: sd-border-1 sd-shadow-sm

      GeoPrior-v3 is built around physics-aware modeling
      rather than purely black-box forecasting. The current
      platform emphasizes physically informed learning for
      subsidence analysis and related geohazard settings.

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
      geohazard workflows, including future landslide-related
      modeling directions.

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

The package metadata defines GeoPrior-v3 as a Python package
for physics-guided geohazard modeling and registers dedicated
console commands for the staged application workflow. The
package namespace also presents GeoPrior-v3 as
``Physics-guided AI for geohazards & risk analytics`` with
land subsidence as the current core application area.

.. seealso::

   - :doc:`getting_started/quickstart`
   - :doc:`user_guide/cli`
   - :doc:`user_guide/configuration`
   - :doc:`applications/reproducibility_scripts`

Documentation map
-----------------

The documentation is organized into **seven main sections**
plus a shared bibliography.

1. **Getting started**
   introduces the project, installation, quickstart usage,
   and the first practical run.

2. **User guide**
   explains the staged workflow, command-line usage,
   configuration system, diagnostics, and export logic.

3. **Scientific foundations**
   presents the model family, physical formulation,
   poroelastic background, residual construction, losses,
   scaling, mathematics, and identifiability assumptions.

4. **Applications**
   focuses on subsidence forecasting, tuning workflows,
   uncertainty-oriented calibration, figure scripts, and
   reproducibility.

5. **API reference**
   documents the importable interfaces for the CLI,
   subsidence modules, tuner components, utilities,
   and packaged resources.

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

.. button-ref:: getting_started/overview
   :ref-type: doc
   :color: primary
   :expand:

   Read the overview

.. button-ref:: scientific_foundations/models_overview
   :ref-type: doc
   :color: secondary
   :expand:

   Explore the models and physics

.. button-ref:: user_guide/cli
   :ref-type: doc
   :color: primary
   :expand:

   Go to the CLI guide

.. button-ref:: applications/subsidence_forecasting
   :ref-type: doc
   :color: secondary
   :expand:

   See the main application workflow

.. button-ref:: release_notes
   :ref-type: doc
   :color: primary
   :expand:

   View release notes

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   getting_started/overview
   getting_started/installation
   getting_started/quickstart
   getting_started/first_project_run

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User guide

   user_guide/workflow_overview
   user_guide/stage1
   user_guide/stage2
   user_guide/stage3
   user_guide/stage4
   user_guide/stage5
   user_guide/cli
   user_guide/configuration
   user_guide/diagnostics
   user_guide/inference_and_export
   user_guide/faq

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Scientific foundations

   scientific_foundations/models_overview
   scientific_foundations/geoprior_subsnet
   scientific_foundations/physics_formulation
   scientific_foundations/poroelastic_background
   scientific_foundations/residual_assembly
   scientific_foundations/losses_and_training
   scientific_foundations/data_and_units
   scientific_foundations/scaling
   scientific_foundations/maths
   scientific_foundations/identifiability
   scientific_foundations/stability_and_constraints

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Applications

   applications/subsidence_forecasting
   applications/tuner_workflow
   applications/calibration_and_uncertainty
   applications/reproducibility_scripts
   applications/figure_generation

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API reference

   api/cli
   api/subsidence
   api/tuner
   api/utils
   api/resources

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Developer notes

   developer/package_structure
   developer/migration_from_fusionlab
   developer/contributing

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Release notes

   release_notes

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: References

   references