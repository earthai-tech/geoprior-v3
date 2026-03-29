Gallery
=======

This section collects the **example-driven lessons** of
GeoPrior-v3.

Unlike the API reference, these pages are organized around
practical workflows and interpretation tasks. Each subsection
shows executable examples, generated outputs, and guided
explanations that help you understand what the workflow is
doing and how to read the results.

Use this gallery to move from **running GeoPrior** to
**interpreting GeoPrior**.

Start here
----------

.. container:: cta-tiles

   .. grid:: 1 1 2 3
      :gutter: 3

      .. grid-item-card:: Forecasting
         :link: ../auto_examples/forecasting/index
         :link-type: doc

         Learn the basic forecasting workflow, future quantile
         mapping, and holdout-versus-forecast comparisons.

      .. grid-item-card:: Uncertainty
         :link: ../auto_examples/uncertainty/index
         :link-type: doc

         Explore interval calibration, coverage-versus-sharpness,
         and exceedance-oriented uncertainty diagnostics.

      .. grid-item-card:: Diagnostics
         :link: ../auto_examples/diagnostics/index
         :link-type: doc

         Inspect stage-oriented data checks, training curves,
         and tuning summaries for the core workflow.

      .. grid-item-card:: Figure generation
         :link: ../auto_examples/figure_generation/index
         :link-type: doc

         Build the paper-ready and analysis-ready figures used
         to communicate GeoPrior results.

      .. grid-item-card:: Tables and summaries
         :link: ../auto_examples/tables_and_summaries/index
         :link-type: doc

         Build tidy metric tables, hotspot summaries, extended
         forecasts, and lightweight spatial support layers.

      .. grid-item-card:: Model inspection
         :link: ../auto_examples/model_inspection/index
         :link-type: doc

         Inspect training histories, epsilon and physics-loss
         trends, payload values, coordinates, and learned
         parameters.


   
How this gallery is organized
-----------------------------

The gallery is split by purpose so you can navigate the
documentation according to the kind of task you want to do.

**Forecasting**, **uncertainty**, and **diagnostics**
focus on understanding workflow results and model behaviour.

**Figure generation** focuses on producing polished visual
outputs for analysis, reporting, and publication.

**Tables and summaries** focuses on reusable artifacts such
as metric tables, summaries, and lightweight support layers.

**Model inspection** focuses on deeper checks of training
behaviour, physics diagnostics, and learned quantities.

How to use these pages
----------------------

Each subsection is designed as a set of small lessons.

A typical page will help you:

- build or load a compact example input,
- run a real GeoPrior helper or plotting routine,
- inspect the resulting figure, table, or artifact,
- and understand how to interpret what you see.

That means the gallery is not only for copying code. It is
also a practical guide to reading outputs with confidence.

See also
--------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Gallery execution times
      :link: ../sg_execution_times
      :link-type: doc

      Review execution-time summaries for the generated
      gallery examples.
      
.. toctree::
   :hidden:

   ../auto_examples/forecasting/index
   ../auto_examples/uncertainty/index
   ../auto_examples/diagnostics/index
   ../auto_examples/figure_generation/index
   ../auto_examples/tables_and_summaries/index
   ../auto_examples/model_inspection/index
   ../sg_execution_times