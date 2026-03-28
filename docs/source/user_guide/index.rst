User Guide
==========

The user guide is the operational companion to the scientific
foundations and API reference. It explains how GeoPrior-v3 is used in
practice: how configuration is organized, how the staged workflow is
executed, how diagnostics are interpreted, and how trained workflows
lead to inference products and exportable artifacts.

This section is intended for readers who already understand the general
project scope and now want to **run**, **inspect**, and **control** the
workflow in a structured way. It connects the command-line interface,
the configuration system, and the staged execution logic into one
narrative so that day-to-day use remains reproducible and transparent.

In broad terms, the user guide answers four practical questions:

- What is the overall execution logic of the project?
- What does each workflow stage do?
- How are configuration, diagnostics, and export paths controlled?
- Where should a user look when something does not behave as expected?

.. note::

   The staged workflow is designed to make the project easier to reason
   about. Rather than hiding all logic behind a single opaque command,
   GeoPrior-v3 exposes clearly named stages and companion guides for
   configuration, diagnostics, and output handling.

Reading paths
-------------

Different readers usually arrive here with different goals. The guide is
therefore organized so that it can be read either sequentially or by
topic.

- **New workflow users** should begin with
  :doc:`workflow_overview`, then read :doc:`configuration`, and finally
  move through :doc:`stage1`, :doc:`stage2`, :doc:`stage3`,
  :doc:`stage4`, and :doc:`stage5`.
- **CLI-oriented users** can pair :doc:`cli` with the stage-specific
  pages and use :doc:`diagnostics` when validating intermediate results.
- **Inference-focused users** will usually want
  :doc:`inference_and_export` after the stage pages.
- **Readers troubleshooting behavior** should keep :doc:`faq` and
  :doc:`diagnostics` close at hand.

Workflow guidance
-----------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: cta-tiles

   .. grid-item-card:: Workflow overview
      :link: workflow_overview
      :link-type: doc
      :class-card: card--workflow

      Start with the high-level execution logic, the relationship
      between stages, and the reading order for the operational
      workflow.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc
      :class-card: card--configuration

      Learn how project behavior is controlled through settings,
      defaults, and configuration-driven workflow design.

   .. grid-item-card:: Diagnostics
      :link: diagnostics
      :link-type: doc

      Understand how to inspect intermediate states, read warnings,
      verify outputs, and evaluate whether a run behaved as expected.

   .. grid-item-card:: Inference and export
      :link: inference_and_export
      :link-type: doc

      Follow the path from trained or tuned workflows to forecast
      outputs, artifacts, and downstream export products.

Stage-by-stage execution
------------------------

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: cta-tiles

   .. grid-item-card:: Stage 1
      :link: stage1
      :link-type: doc

      Data preparation, project initialization logic, and the early
      processing steps that establish the workflow foundation.

   .. grid-item-card:: Stage 2
      :link: stage2
      :link-type: doc

      Training-oriented execution and the transition from prepared
      inputs toward calibrated model behavior.

   .. grid-item-card:: Stage 3
      :link: stage3
      :link-type: doc

      Tuning and search procedures used to refine model configuration
      and improve workflow robustness.

   .. grid-item-card:: Stage 4
      :link: stage4
      :link-type: doc

      Inference-time execution, forecast generation, and the practical
      use of saved workflow outputs.

   .. grid-item-card:: Stage 5
      :link: stage5
      :link-type: doc

      Transfer, extension, and final workflow products that complete the
      staged operational path.

Reference and support
---------------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: cta-tiles

   .. grid-item-card:: CLI guide
      :link: cli
      :link-type: doc
      :class-card: card--cli

      Browse the command-line entry points, command groupings, and how
      they map onto the staged workflow.

   .. grid-item-card:: FAQ
      :link: faq
      :link-type: doc

      Find concise answers to recurring practical questions about
      running, configuring, diagnosing, and exporting workflows.

Section contents
----------------

.. toctree::
   :maxdepth: 1
   :hidden:

   workflow_overview
   stage1
   stage2
   stage3
   stage4
   stage5
   cli
   configuration
   diagnostics
   inference_and_export
   faq
