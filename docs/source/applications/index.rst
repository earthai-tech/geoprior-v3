Applications
============

This section presents GeoPrior from the perspective of concrete research
and workflow use cases. Rather than focusing only on isolated module
interfaces, the application pages show how the package is used to solve
practical tasks such as subsidence forecasting, hyperparameter tuning,
uncertainty-aware calibration, reproducibility support, and figure
production for scientific communication.

The goal is to bridge three levels of understanding:

- **workflow intent**, explaining what a page is trying to accomplish;
- **scientific context**, clarifying why the task matters in the land
  subsidence setting;
- **implementation path**, showing which package layers, stages, or
  scripts support the workflow in practice.

For readers coming from the command-line workflow, this section is a
natural continuation of the :doc:`../user_guide/workflow_overview` and
stage-by-stage guides. For readers coming from the model side, it also
serves as an applied complement to the
:doc:`../scientific_foundations/models_overview` and
:doc:`../scientific_foundations/geoprior_subsnet` pages.

.. tip::

   If you are trying to understand how the package behaves in realistic
   forecasting and analysis scenarios, start here before diving into the
   lower-level API pages. The application notes provide the most direct
   bridge between the scientific narrative and the practical workflow.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Subsidence forecasting
      :class-card: sd-shadow-sm card--workflow

      Follow the end-to-end forecasting narrative, including inputs,
      staged processing, model execution, and interpretation of outputs
      in the GeoPrior subsidence setting.

      :doc:`subsidence_forecasting`

   .. grid-item-card:: Tuner workflow
      :class-card: sd-shadow-sm card--configuration

      Understand how configuration search, experiment structure, and
      tuning logic fit into the wider modeling workflow.

      :doc:`tuner_workflow`

   .. grid-item-card:: Calibration and uncertainty
      :class-card: sd-shadow-sm card--uncertainty

      Explore how calibration logic, uncertainty-aware diagnostics, and
      evaluation support more credible model interpretation.

      :doc:`calibration_and_uncertainty`

   .. grid-item-card:: Reproducibility scripts
      :class-card: sd-shadow-sm card--cli

      Review the reproducibility-oriented scripts that help regenerate
      experiments, organize outputs, and keep results traceable.

      :doc:`reproducibility_scripts`

   .. grid-item-card:: Figure generation
      :class-card: sd-shadow-sm card--physics

      See how the project turns model outputs and diagnostics into
      publication-ready figures for interpretation and communication.

      :doc:`figure_generation`

Reading path
------------

A sensible reading order depends on what you want from the package.

If you are new to GeoPrior, begin with
:doc:`subsidence_forecasting` to understand the full applied narrative.
Then move to :doc:`tuner_workflow` and
:doc:`calibration_and_uncertainty` to see how experimentation and
credibility analysis are layered on top of the base workflow.

If your primary interest is reproducibility and scientific reporting,
read :doc:`reproducibility_scripts` together with
:doc:`figure_generation`. Those pages make it easier to understand how
GeoPrior supports not only modeling, but also the regeneration and
presentation of research outputs.

.. toctree::
   :maxdepth: 1
   :hidden:

   subsidence_forecasting
   tuner_workflow
   calibration_and_uncertainty
   reproducibility_scripts
   figure_generation
