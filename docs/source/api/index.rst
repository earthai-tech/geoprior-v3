API reference
=============

The API reference collects the documented public Python surface of
GeoPrior. It is designed for users who want to move from the narrative
workflow pages into the actual modules, classes, helper layers, and
configuration surfaces that support reproducible forecasting runs.

This section is not just a flat module listing. It is organized around
the main interfaces that users are most likely to touch directly during
normal work:

- command-line wrappers and staged workflow entry points,
- subsidence-model interfaces and supporting model components,
- tuning interfaces for calibration and structured experimentation,
- utility modules for preprocessing, diagnostics, geospatial handling,
  and forecast support,
- packaged resources and configuration-driven assets,
- parameter helpers used to represent physical constants and learnable
  scalar coefficients.

For a scientific overview of the modeling stack, start with
:doc:`../scientific_foundations/geoprior_subsnet` and
:doc:`../scientific_foundations/models_overview`. For workflow-oriented
usage, see :doc:`../user_guide/workflow_overview` and the examples
gallery under :doc:`../examples/index`.

.. note::

   The pages collected here emphasize the **documented public surface**
   of GeoPrior. Internal helpers and compatibility layers may exist
   underneath the package, but this reference focuses on the parts of
   the codebase that users are expected to import, inspect, extend, or
   call through normal workflows.

.. container:: cta-tiles

   .. grid:: 1 2 2 3
      :gutter: 3

      .. grid-item-card:: CLI and workflow surface
         :class-card: sd-shadow-sm
         :link: cli
         :link-type: doc

         Browse the documented command-line wrappers, staged execution
         modules, and resource pages that support initialization, runs,
         diagnostics, and exports.

      .. grid-item-card:: Subsidence models
         :class-card: sd-shadow-sm
         :link: subsidence
         :link-type: doc

         Explore the subsidence model family, supporting mathematical and
         diagnostic modules, and the interfaces that structure the main
         GeoPrior forecasting workflow.

      .. grid-item-card:: Tuning components
         :class-card: sd-shadow-sm
         :link: tuner
         :link-type: doc

         Review the documented tuner interfaces used for calibration,
         configuration search, and structured experimentation.

      .. grid-item-card:: Utility layers
         :class-card: sd-shadow-sm
         :link: utils
         :link-type: doc

         Understand the utility stack spanning preprocessing,
         calibration, geospatial handling, forecasting helpers, and
         supporting model utilities.

      .. grid-item-card:: Resources and packaged data
         :class-card: sd-shadow-sm
         :link: resources
         :link-type: doc

         Inspect packaged resources, templates, and supporting files
         that help keep GeoPrior workflows reproducible and
         configuration-driven.

      .. grid-item-card:: Physical parameter wrappers
         :class-card: sd-shadow-sm
         :link: params
         :link-type: doc

         Learn how scalar physical constants and learnable coefficients
         are represented through the documented parameter helper layer.

How to read this section
------------------------

This part of the documentation can be used in two complementary ways.

First, it can be read as a practical reference. When you need a stable
import path, a documented class, or a specific helper function, these
pages show the intended public surface and help you locate the right
module quickly.

Second, it can be read as an architectural map. Several API layers are
closely tied to the scientific workflow, so the reference pages also
explain why a module exists, how it fits into the broader forecasting
stack, and whether it is typically used directly or through higher-level
orchestration.

A good reading path is often:

#. start with the workflow-facing CLI surface,
#. move to the subsidence model pages,
#. review tuner and utility layers as needed,
#. consult resources and parameter wrappers for reproducibility and
   physical configuration details.

.. tip::

   If you are new to the package, begin with the workflow and model
   pages first. If you are writing scripts, extending internals, or
   building reproducible research pipelines, the utility, resource, and
   parameter sections will usually become important next.

.. toctree::
   :maxdepth: 1
   :hidden:

   cli
   subsidence
   tuner
   utils
   resources
   params