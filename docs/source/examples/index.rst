Gallery
=======

This section collects the **example-driven lessons** of
GeoPrior-v3.

Unlike the API reference, these pages are organized around
practical workflows, interpretation tasks, and decision-making
questions. Each subsection shows executable examples,
generated outputs, and guided explanations that help users
understand not only **how to run** a helper, but also **how to
read what it produced**.

Use this gallery to move from **running GeoPrior** to
**interpreting GeoPrior with confidence**.

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

         Explore calibration, reliability, coverage-versus-
         sharpness, raw-versus-calibrated reliability, and
         exceedance-oriented uncertainty diagnostics.

      .. grid-item-card:: Evaluation
         :link: ../auto_examples/evaluation/index
         :link-type: doc

         Read forecast quality through horizon-wise metrics,
         stability, interval scores, calibration summaries,
         and multi-metric comparison plots.

      .. grid-item-card:: Diagnostics
         :link: ../auto_examples/diagnostics/index
         :link-type: doc

         Inspect stage-oriented data checks, training curves,
         tuning summaries, and regression-style fit diagnostics
         such as R² comparison views.

      .. grid-item-card:: Spatial
         :link: ../auto_examples/spatial/index
         :link-type: doc

         Learn how to read mapped outputs through full-domain
         scatter maps, ROI zooms, contours, hotspot views,
         Voronoi partitions, and heatmap-style summaries.

      .. grid-item-card:: Inspection
         :link: ../auto_examples/inspection/index
         :link-type: doc

         Read saved workflow artifacts as evidence: audits,
         manifests, scaling sidecars, training summaries,
         evaluation bundles, transfer results, and ablation logs.

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

**Forecasting** focuses on what was predicted and how forecast
outputs are structured.

**Uncertainty** focuses on probabilistic trust questions such as
reliability, calibration, coverage, sharpness, raw-versus-
calibrated comparison, and exceedance behavior.

**Evaluation** focuses on judging forecast quality once results
already exist: error over horizon, weighted metrics, stability,
interval scores, calibration summaries, and compact model
comparison views.

**Diagnostics** focuses on whether the workflow behaved cleanly:
stage checks, training health, tuning summaries, and
regression-style fit views such as R² diagnostics.

**Spatial** focuses on where patterns happen and how mapped
results should be read: sampled points, local regions of
interest, smoothed surfaces, hotspots, support partitions,
and gridded summaries.

**Inspection** focuses on reading the saved workflow artifacts
that connect those stages together: audits, manifests,
configuration sidecars, summaries, evaluation JSONs, transfer
bundles, and experiment logs. It is the best place to go when
you already have an artifact on disk and want to decide whether
to continue, recalibrate, compare, export, or re-run.

**Figure generation** focuses on producing polished visual
outputs for analysis, reporting, and publication.

**Tables and summaries** focuses on reusable artifacts such as
metric tables, summaries, and lightweight support layers.

**Model inspection** focuses on deeper checks of training
behavior, physics diagnostics, and learned quantities inside the
model itself.

A practical reading rule
------------------------

If you are not sure where to begin, use this guide:

- Go to **Forecasting** when your main question is *what was
  predicted?*
- Go to **Uncertainty** when your main question is *how reliable
  are the intervals, quantiles, or exceedance estimates?*
- Go to **Evaluation** when your main question is *how good is the
  forecast once I quantify it across horizons, metrics, and
  calibration views?*
- Go to **Diagnostics** when your main question is *did the
  workflow run cleanly and do the staged checks or fit summaries
  look healthy?*
- Go to **Spatial** when your main question is *where are the
  mapped patterns, support zones, hotspots, or local regional
  structures?*
- Go to **Inspection** when your main question is *what does this
  saved artifact mean, and is it trustworthy enough for the next
  decision?*
- Go to **Figure generation** when your main question is *how do I
  communicate the result clearly?*
- Go to **Tables and summaries** when your main question is *how do
  I export a reusable metric or spatial summary?*
- Go to **Model inspection** when your main question is *what did
  the model learn internally and how did the physics terms behave?*

How to use these pages
----------------------

Each subsection is designed as a set of small lessons.

A typical page will help you:

- build or load a compact example input,
- run a real GeoPrior helper or plotting routine,
- inspect the resulting figure, table, or artifact,
- understand what the output means,
- and decide what to do next.

That means the gallery is not only for copying code. It is also
a practical guide to reading outputs with confidence.

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
   ../auto_examples/evaluation/index
   ../auto_examples/diagnostics/index
   ../auto_examples/spatial/index
   ../auto_examples/inspection/index
   ../auto_examples/figure_generation/index
   ../auto_examples/tables_and_summaries/index
   ../auto_examples/model_inspection/index
   ../sg_execution_times
