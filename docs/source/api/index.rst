API reference
=============

The API reference documents the public Python surfaces that support the
GeoPrior workflow, from configuration-driven CLI execution to model
construction, tuning, utilities, and parameter wrappers. This section is
intended for users who want a stable view of the documented import
surface, along with contextual notes explaining how the pieces fit
within the broader subsidence-forecasting stack.

Unlike a flat module dump, this reference is organized around the major
interfaces that users are expected to touch directly:

- command-line entry points and wrapper modules,
- the subsidence model family and its supporting components,
- tuning interfaces for search and experimentation,
- utility layers used for preprocessing, calibration, diagnostics,
  and data handling,
- resource and configuration modules that support reproducible runs,
- parameter wrappers used to represent physical constants and
  learnable scalar coefficients.

For a higher-level overview of the scientific workflow, start with the
:doc:`../scientific_foundations/geoprior_subsnet` page and the
:doc:`../user_guide/workflow_overview` guide. For end-to-end usage
examples, see the :doc:`../applications/subsidence_forecasting` and
:doc:`../applications/tuner_workflow` application pages.

.. note::

   The pages collected here focus on the **documented public surface** of
   GeoPrior. Internal compatibility shims or private implementation files
   may exist underneath the package, but the API reference is designed to
   emphasize the modules that users are expected to import, inspect, and
   extend in normal workflows.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: CLI and workflow surface
      :class-card: sd-shadow-sm

      Browse the documented command-line wrappers, staged execution
      modules, and resource pages that support initialization, runs,
      diagnostics, and exports.

      :doc:`cli`

   .. grid-item-card:: Subsidence models
      :class-card: sd-shadow-sm

      Explore the subsidence model family, supporting mathematical and
      diagnostic modules, and the interfaces that structure the main
      GeoPrior forecasting workflow.

      :doc:`subsidence`

   .. grid-item-card:: Tuning components
      :class-card: sd-shadow-sm

      Review the documented tuner interfaces used for calibration,
      configuration search, and structured experimentation.

      :doc:`tuner`

   .. grid-item-card:: Utility layers
      :class-card: sd-shadow-sm

      Understand the utility stack spanning preprocessing, calibration,
      geospatial handling, forecasting helpers, and supporting model
      utilities.

      :doc:`utils`

   .. grid-item-card:: Resources and packaged data
      :class-card: sd-shadow-sm

      Inspect packaged resources, templates, and supporting files that
      help keep GeoPrior workflows reproducible and configuration-driven.

      :doc:`resources`

   .. grid-item-card:: Physical parameter wrappers
      :class-card: sd-shadow-sm

      Learn how scalar physical constants and learnable coefficients are
      represented through the documented parameter helper layer.

      :doc:`params`

How to use this section
-----------------------

The API pages are designed to be read in two complementary ways.

First, users can treat them as a reference for concrete import paths,
public classes, and helper functions. This is especially useful when
building scripts, notebooks, or research pipelines around the documented
GeoPrior interfaces.

Second, the pages can be read as architectural notes. Several of the API
modules are tightly connected to the scientific model design, so the
reference also explains why a given layer exists, how it is positioned
in the workflow, and when it should be used directly versus indirectly
through the CLI or higher-level orchestration helpers.

.. toctree::
   :maxdepth: 1
   :hidden:

   cli
   subsidence
   tuner
   utils
   resources
   params
