Forecasting
===========

This gallery focuses on the **forecasting workflow** in GeoPrior.

The examples in this section explain how forecast outputs are built,
inspected, and interpreted across the core prediction pipeline. They
are designed to help readers move from the raw stage-4 outputs to a
clear understanding of:

- basic forecast structure,
- future quantile maps,
- and holdout-versus-forecast comparisons.

In other words, these pages teach how to read the main forecasting
products before moving to uncertainty, diagnostics, or final
figure-generation workflows.

What this section teaches
-------------------------

The examples in this folder are designed as **forecasting lessons**.

Each page usually does four things:

1. create or load a compact synthetic forecasting input,
2. call the relevant GeoPrior plotting or workflow helper,
3. inspect the produced forecast output,
4. explain how to interpret the result.

The main goal is to make the forecasting workflow readable and
reproducible, rather than to jump directly to a paper-ready figure.

How to read this gallery
------------------------

A good way to use this section is to move from:

- a basic stage-4 forecast output,
- to future uncertainty-aware maps,
- to a comparison between held-out observations and forecasts.

That order mirrors how most users first encounter GeoPrior forecast
artifacts in practice.

Examples in this section
------------------------

``plot_stage4_basic_forecast.py``
    Introduce the basic stage-4 forecasting output and show how to
    inspect the main predicted quantities.

``plot_future_quantiles_map.py``
    Visualize future quantile forecasts spatially and explain how to
    read lower, median, and upper forecast surfaces.

``plot_holdout_vs_forecast.py``
    Compare held-out targets against forecasts to understand how the
    model behaves on evaluation-style splits.

Practical workflow
------------------

A common workflow through this section is:

1. run or load stage-4 inference outputs,
2. inspect the basic forecast structure,
3. examine future quantile maps,
4. compare holdout behavior against the forecasted trajectories,
5. then move on to uncertainty and diagnostics.

This separation is deliberate. It helps keep:

- forecasting,
- uncertainty interpretation,
- and diagnostic inspection

cleanly distinct.

Notes
-----

- These examples are intentionally compact and lesson-oriented.
- The pages in this section are forecast-first: they help explain what
  the forecast outputs mean and how to interpret them.
- A good rule of thumb is:

  - ``forecasting/`` explains forecast products,
  - ``uncertainty/`` explains reliability and interval behavior,
  - ``diagnostics/`` explains workflow-stage checks,
  - ``figure_generation/`` builds final visual outputs.