Applications
============

This section presents GeoPrior through concrete research and workflow
use cases. Instead of focusing only on isolated module interfaces, the
application pages show how the package is used to solve practical tasks
such as subsidence forecasting, hyperparameter tuning,
uncertainty-aware calibration, reproducibility support, and figure
production for scientific communication.

The goal is to connect three levels of understanding that often need to
be read together in real work:

- **workflow intent**, explaining what a page is trying to accomplish,
- **scientific context**, clarifying why the task matters in the land
  subsidence setting,
- **implementation path**, showing which package layers, stages, or
  scripts support the workflow in practice.

For readers coming from the command-line workflow, this section is a
natural continuation of the :doc:`../user_guide/workflow_overview`
and the stage-by-stage guides. For readers coming from the model side,
it also serves as an applied complement to the
:doc:`../scientific_foundations/models_overview` and
:doc:`../scientific_foundations/geoprior_subsnet` pages.

.. note::

   Use this section when you want to understand how GeoPrior behaves in
   realistic forecasting and analysis scenarios. These pages provide the
   most direct bridge between the scientific narrative, the workflow
   logic, and the practical outputs produced in applied studies.

.. container:: cta-tiles

   .. grid:: 1 2 2 3
      :gutter: 3

      .. grid-item-card:: Subsidence forecasting
         :class-card: sd-shadow-sm card--workflow
         :link: subsidence_forecasting
         :link-type: doc

         Follow the end-to-end forecasting narrative, including inputs,
         staged processing, model execution, and interpretation of outputs
         in the GeoPrior subsidence setting.

      .. grid-item-card:: Tuner workflow
         :class-card: sd-shadow-sm card--configuration
         :link: tuner_workflow
         :link-type: doc

         Understand how configuration search, experiment structure, and
         tuning logic fit into the wider modeling workflow.

      .. grid-item-card:: Calibration and uncertainty
         :class-card: sd-shadow-sm card--uncertainty
         :link: calibration_and_uncertainty
         :link-type: doc

         Explore how calibration logic, uncertainty-aware diagnostics, and
         evaluation support more credible model interpretation.

      .. grid-item-card:: Reproducibility scripts
         :class-card: sd-shadow-sm card--cli
         :link: reproducibility_scripts
         :link-type: doc

         Review the reproducibility-oriented scripts that help regenerate
         experiments, organize outputs, and keep results traceable.

      .. grid-item-card:: Figure generation
         :class-card: sd-shadow-sm card--physics
         :link: figure_generation
         :link-type: doc

         See how the project turns model outputs and diagnostics into
         publication-ready figures for interpretation and communication.

How to read this section
------------------------

These pages can be read in two complementary ways.

The first is as an applied walkthrough of the package. In that mode, a
reader starts from the forecasting workflow, then moves outward into
model tuning, calibration, uncertainty, reproducibility, and figure
production. This path is useful when the main goal is to understand how
GeoPrior is used in an end-to-end study rather than only how individual
modules are implemented.

The second is as a map of research-facing use cases. In that mode, each
page acts as a focused application note showing how a specific activity
fits into the broader GeoPrior ecosystem, what artifacts it depends on,
and which later outputs it supports.

A useful reading path is often:

#. begin with :doc:`subsidence_forecasting`,
#. continue to :doc:`tuner_workflow`,
#. then read :doc:`calibration_and_uncertainty`,
#. finish with :doc:`reproducibility_scripts` and
   :doc:`figure_generation`.

That sequence moves from the main applied workflow to experimentation,
then to credibility analysis, and finally to the reproducibility and
communication layers that support research-grade use.

.. tip::

   If you are new to GeoPrior, start with the forecasting page first.
   If you already know the workflow and mainly want to understand how
   results are organized, checked, regenerated, and presented, move
   directly to the calibration, reproducibility, and figure pages.

.. toctree::
   :maxdepth: 1
   :hidden:

   subsidence_forecasting
   tuner_workflow
   calibration_and_uncertainty
   reproducibility_scripts
   figure_generation
