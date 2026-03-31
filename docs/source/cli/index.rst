.. _cli-index:

CLI
======

GeoPrior provides a **family-based command-line interface** built
around one main dispatcher and a set of family-specific entry points.

This section is the best place to start when you want to understand
**how the CLI is organized**, **which command family fits your task**,
and **where to look next** for shared conventions or detailed command
pages.

The Command-line interface (CLI) is organized into three public families:

- **Run** commands execute staged workflows and research drivers.
- **Build** commands materialize reusable artifacts such as tables,
  summaries, payloads, and derived datasets.
- **Plot** commands render figures, maps, and publication-oriented
  diagnostics.

CLI families at a glance
------------------------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: cta-tiles cli-tiles

   .. grid-item-card:: Run family
      :link: run_family
      :link-type: doc
      :class-card: card--workflow

      Use the run family when you want to execute workflows.

      This includes staged preprocessing, training, tuning, inference,
      transfer evaluation, and supplementary run-side drivers such as
      sensitivity and SM3 workflows.

   .. grid-item-card:: Build family
      :link: build_family
      :link-type: doc
      :class-card: card--configuration

      Use the build family when you want to materialize artifacts
      that can be reused later.

      These commands create compact datasets, tables, payloads,
      validation products, summary records, and derived geospatial
      outputs.

   .. grid-item-card:: Plot family
      :link: plot_family
      :link-type: doc
      :class-card: card--cli

      Use the plot family when you want to render visual outputs.

      This includes forecasting figures, uncertainty panels, physics
      maps, transfer plots, hotspot analytics, and SM3 diagnostics.

Start here
----------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: cta-tiles cli-tiles

   .. grid-item-card:: Introduction
      :link: introduction
      :link-type: doc
      :class-card: card--workflow

      Start with the CLI mental model: the root dispatcher,
      family-specific entry points, aliases, and the overall public
      command surface.

   .. grid-item-card:: Shared conventions
      :link: shared_conventions
      :link-type: doc
      :class-card: card--configuration

      Learn the repeated CLI patterns once: config installation,
      runtime overrides, shared output handling, and common
      data-loading behavior.
      

How this section fits with the rest of the docs
-----------------------------------------------

A simple way to use the documentation is:

- start in the **CLI** section when you already know you want to run a
  command,
- move to the **User Guide** when you want workflow context and
  stage-by-stage reasoning,
- move to **Examples** when you want complete lessons with outputs,
  figures, and interpretation,
- move to the **API reference** when you want importable Python
  interfaces.

The CLI pages therefore answer a practical question:

**Which command family should I use, and where do I find the details
for that task?**

From here
---------

A good reading path is:

#. :doc:`introduction`
#. :doc:`shared_conventions`
#. one of :doc:`run_family`, :doc:`build_family`, or
   :doc:`plot_family` depending on your task

.. toctree::
   :maxdepth: 1
   :hidden:

   introduction
   shared_conventions
   run_family
   build_family
   plot_family