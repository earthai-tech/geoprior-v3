.. _home:

GeoPrior-v3
===========

.. rst-class:: hero-tagline

Physics-guided AI for geohazards and risk analytics.

GeoPrior-v3 is a scientific Python framework for building
physics-guided models for geohazard analysis, forecasting, and
risk-oriented interpretation. The current generation focuses on
**land subsidence** through **GeoPriorSubsNet v3**, while the broader
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
   ``geoprior-init``. See :doc:`cli/index`  for the full
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
      
   .. grid-item-card:: User guide
      :link: user_guide/index
      :link-type: doc
      :class-card: card--configuration

      Follow the staged workflow, command-line logic,
      diagnostics, inference paths, and export-oriented use.

   .. grid-item-card:: CLI
      :link: cli/index
      :link-type: doc
      :class-card: card--cli

      Find run, build, and plot commands, shared conventions,
      command families, and the main command-line entry points.
      
   .. grid-item-card:: Scientific foundations
      :link: scientific_foundations/index
      :link-type: doc
      :class-card: card--physics

      Understand the model family, the physical formulation,
      residual construction, scaling strategy, and the main
      scientific assumptions.
      
   .. grid-item-card:: Gallery
      :link: examples/index
      :link-type: doc
      :class-card: card--workflow

      Browse lesson-style examples for forecasting, uncertainty,
      diagnostics, figure generation, table builders, and
      model-inspection utilities.
      
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

Roadmap
-------

.. image:: _static/previews/geoprior-documentation-roadmap.svg
   :alt: GeoPrior-v3 documentation roadmap
   :width: 100%
   :align: center
 
 
Featured applications
---------------------

.. grid:: 1 1 2 4
   :gutter: 3
   :class-container: see-also-tiles

   .. grid-item-card:: Applications
      :link: applications/index
      :link-type: doc
      :class-card: sd-shadow-sm seealso-card card--workflow

      Explore forecasting workflows, tuning, calibration,
      uncertainty handling, and reproducibility scripts.

   .. grid-item-card:: Core & ablation
      :link: auto_examples/figure_generation/plot_core_ablation
      :link-type: doc
      :class-card: sd-shadow-sm seealso-card card--core-ablation

      Compare the main with-physics and no-physics results in
      the core ablation figure.

   .. grid-item-card:: Bounds vs ridge summary
      :link: auto_examples/figure_generation/plot_sm3_bounds_ridge_summary
      :link-type: doc
      :class-card: sd-shadow-sm seealso-card card--ridge-bounds

      Inspect clipping, ridge behavior, and SM3 failure modes
      in one summary figure.

   .. grid-item-card:: Where to act first
      :link: auto_examples/figure_generation/plot_hotspot_analytics
      :link-type: doc
      :class-card: sd-shadow-sm seealso-card card--hotspot-analytics

      See the hotspot analytics figure that highlights where
      intervention should come first.
      
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


Navigate directly
-----------------

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item::
      .. button-ref:: getting_started/index
         :ref-type: doc
         :color: primary
         :expand:

         Begin with getting started

   .. grid-item::
      .. button-ref:: user_guide/index
         :ref-type: doc
         :color: secondary
         :expand:

         Open the user guide

   .. grid-item::
      .. button-ref:: cli/index
         :ref-type: doc
         :color: primary
         :expand:

         Explore the CLI reference

   .. grid-item::
      .. button-ref:: applications/index
         :ref-type: doc
         :color: secondary
         :expand:

         See the application workflows

   .. grid-item::
      .. button-ref:: examples/index
         :ref-type: doc
         :color: primary
         :expand:

         Browse gallery examples

   .. grid-item::
      .. button-ref:: scientific_foundations/index
         :ref-type: doc
         :color: secondary
         :expand:

         Explore the models and physics

   .. grid-item::
      .. button-ref:: api/index
         :ref-type: doc
         :color: primary
         :expand:

         Browse the API reference

   .. grid-item::
      .. button-ref:: developer/index
         :ref-type: doc
         :color: secondary
         :expand:

         Open developer notes

   .. grid-item::
      .. button-ref:: release_notes
         :ref-type: doc
         :color: primary
         :expand:

         View release notes

   .. grid-item::
      .. button-ref:: glossary/index
         :ref-type: doc
         :color: secondary
         :expand:

         Open the glossary

   .. grid-item::
      .. button-ref:: scientific_scope
         :ref-type: doc
         :color: primary
         :expand:

         Understand the scientific scope
         
   .. grid-item::
      .. button-ref:: references
         :ref-type: doc
         :color: secondary
         :expand:

         See shared references

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   getting_started/index
   user_guide/index
   cli/index
   applications/index
   examples/index
   scientific_foundations/index
   api/index
   developer/index
   release_notes
   glossary/index
   scientific_scope
   references