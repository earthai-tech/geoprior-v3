Diagnostics
===========

This gallery focuses on **workflow-stage diagnostics** in GeoPrior.

The examples in this section explain how to inspect the major staged
parts of the workflow before moving on to forecasting, uncertainty, or
final figure generation. They are designed to help readers understand:

- whether the inputs look sane,
- how training evolves,
- and how tuning results should be read.

In other words, these pages teach how to diagnose the GeoPrior
workflow across its main operational stages.

What this section teaches
-------------------------

The examples in this folder are designed as **diagnostic lessons**.

Each page usually does four things:

1. create or load a compact synthetic stage-oriented input,
2. call the relevant GeoPrior diagnostic plotting helper,
3. inspect the produced diagnostic output,
4. explain what a healthy or unhealthy result looks like.

The main goal is to make stage-level workflow behavior easy to check
before readers move on to deeper modeling interpretation.

How to read this gallery
------------------------

A good way to use this section is to move from:

- stage-1 data checks,
- to stage-2 training behavior,
- to stage-3 tuning summaries.

That order mirrors the normal operational sequence of the workflow and
helps users diagnose problems early rather than late.

Examples in this section
------------------------

``plot_stage1_data_checks.py``
    Inspect the early data-processing outputs and check that the main
    stage-1 artifacts look consistent before training begins.

``plot_stage2_training_curves.py``
    Examine stage-2 training curves to understand optimization
    behavior, stability, and possible training pathologies.

``plot_stage3_tuning_summary.py``
    Summarize stage-3 tuning outputs and help interpret which
    hyperparameter settings or trial families performed best.

Practical workflow
------------------

A common workflow through this section is:

1. inspect stage-1 data outputs,
2. inspect stage-2 training curves,
3. inspect stage-3 tuning summaries,
4. then proceed to forecasting, uncertainty, or final reporting.

This separation is deliberate. It helps keep:

- workflow validation,
- model interpretation,
- and final presentation

cleanly distinct.

Notes
-----

- These examples are intentionally compact and lesson-oriented.
- The pages in this section are stage-first: they explain how to
  validate the workflow as it runs.
- A good rule of thumb is:

  - ``diagnostics/`` checks workflow health,
  - ``forecasting/`` explains prediction outputs,
  - ``uncertainty/`` explains forecast reliability,
  - ``figure_generation/`` builds final communication figures.