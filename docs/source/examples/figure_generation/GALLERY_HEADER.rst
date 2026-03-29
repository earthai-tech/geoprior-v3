Figure generation
=================

This gallery focuses on **paper-ready and analysis-ready plotting
workflows** in GeoPrior.

Unlike the examples in ``tables_and_summaries/``, the pages collected
here begin from plotting scripts whose main purpose is to generate
visual outputs: main figures, supplementary figures, diagnostic panels,
sensitivity plots, transfer figures, and spatial analytics views.

The emphasis is therefore on **communication through figures**. These
examples show how GeoPrior turns model outputs, diagnostic signals, and
support artifacts into interpretable visual summaries that can be used
for analysis, reporting, and manuscript preparation.

Typical outputs in this section include:

- publication-style PNG, SVG, or EPS figures,
- multi-panel diagnostic layouts,
- sensitivity heatmaps and comparison plots,
- spatial forecast maps and hotspot analytics panels,
- transferability and impact figures,
- validation and cumulative-summary plots.

In other words, this gallery is about **building the figures that
communicate results**, not the intermediate artifacts that those figures
depend on.

What this gallery teaches
-------------------------

Most pages in this section follow the same broad pattern:

#. create or load a compact synthetic input,
#. call the real GeoPrior plotting script,
#. inspect the generated figure files,
#. explain how to read and interpret the main panels.

Even when a page prints small tables or intermediate values, the main
goal remains the same: to explain the **figure and its interpretation**.

Module guide
------------

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Module
     - Main output
     - Purpose
   * - ``plot_driver_response.py``
     - Relationship figure
     - Plot driver-response relationships to show how key forcing or
       context variables relate to predicted behavior.
   * - ``plot_core_ablation.py``
     - Ablation comparison figure
     - Plot the core ablation comparisons used to interpret the main
       model-design trade-offs.
   * - ``plot_uncertainty.py``
     - Uncertainty figure
     - Plot the main uncertainty diagnostics, including calibration and
       reliability-oriented views.
   * - ``plot_spatial_forecasts.py``
     - Spatial forecast maps
     - Plot the main spatial forecast maps and multi-panel forecast
       layouts.
   * - ``plot_physics_sanity.py``
     - Physics sanity figure
     - Plot sanity checks for the physics-informed components.
   * - ``plot_physics_maps.py``
     - Physics map panels
     - Plot map-based views of learned or inferred physics quantities.
   * - ``plot_physics_fields.py``
     - Field diagnostic figure
     - Plot field-level diagnostics for physical parameters and related
       outputs.
   * - ``plot_physics_profiles.py``
     - Profile diagnostics
     - Plot profile-style physics diagnostics, typically for appendix or
       supplementary interpretation.
   * - ``plot_litho_parity.py``
     - Parity / agreement figure
     - Plot lithology-related parity or agreement diagnostics.
   * - ``plot_ablations_sensitivity.py``
     - Sensitivity surface figure
     - Plot sensitivity surfaces and ablation-response summaries.
   * - ``plot_physics_sensitivity.py``
     - Physics sensitivity figure
     - Plot the response of physical metrics or residuals to
       sensitivity settings.
   * - ``plot_sm3_identifiability.py``
     - Identifiability figure
     - Plot SM3 identifiability diagnostics.
   * - ``plot_sm3_bounds_ridge_summary.py``
     - Bounds / ridge summary
     - Plot ridge and bounds summaries for SM3-style identifiability
       experiments.
   * - ``plot_sm3_log_offsets.py``
     - Log-offset diagnostics
     - Plot SM3 log-offset diagnostics and related parameter-behavior
       summaries.
   * - ``plot_xfer_transferability.py``
     - Transferability figure
     - Plot transferability performance across source-target settings.
   * - ``plot_xfer_impact.py``
     - Transfer impact figure
     - Plot transfer impact in terms of retention, hotspot overlap, and
       related measures.
   * - ``plot_external_validation.py``
     - Validation figure
     - Plot external point-support validation results.
   * - ``plot_geo_cumulative.py``
     - Cumulative summary figure
     - Plot cumulative geo-style summaries across years or scenarios.
   * - ``plot_hotspot_analytics.py``
     - Hotspot analytics figure
     - Plot hotspot analytics views that summarize spatial hotspot
       behavior over time or across settings.
   * - ``plot_uncertainty_extras.py``
     - Supplementary uncertainty panels
     - Plot additional uncertainty panels beyond the main uncertainty
       figure.

Reading path
------------

A useful way to move through this gallery is to follow the logic of a
complete visual analysis workflow:

#. begin with core model-behavior figures,
#. continue to uncertainty and sensitivity structure,
#. move into spatial and physics-aware visual diagnostics,
#. finish with transfer, impact, and validation views.

That is why the examples are grouped by **plotting purpose** rather than
only by command family.

Gallery organization
--------------------

Core result figures
~~~~~~~~~~~~~~~~~~~

These examples correspond most directly to the central narrative figures
of the GeoPrior workflow. They are often the best place to start when
you want a high-level visual understanding of model behavior and main
results.

The main pages in this group are:

- ``plot_driver_response.py``
- ``plot_core_ablation.py``
- ``plot_uncertainty.py``
- ``plot_spatial_forecasts.py``

Physics and mechanistic diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These examples focus on the physical structure of the model and on the
interpretation of learned or inferred physics fields. They are most
useful when you want to inspect whether the scientific behavior exposed
by the model remains interpretable and internally coherent.

The main pages in this group are:

- ``plot_physics_sanity.py``
- ``plot_physics_maps.py``
- ``plot_physics_fields.py``
- ``plot_physics_profiles.py``
- ``plot_litho_parity.py``

Sensitivity and identifiability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These examples focus on how results change under different weights,
priors, constraints, or identifiability settings. They are particularly
helpful when comparing training regimes or understanding the robustness
of the formulation.

The main pages in this group are:

- ``plot_ablations_sensitivity.py``
- ``plot_physics_sensitivity.py``
- ``plot_sm3_identifiability.py``
- ``plot_sm3_bounds_ridge_summary.py``
- ``plot_sm3_log_offsets.py``

Transfer, impact, and validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These examples focus on cross-city transfer, downstream impact, and
out-of-sample validation. They are especially useful when the goal is
to understand generalization, downstream relevance, or practical
forecast usefulness beyond a single training context.

The main pages in this group are:

- ``plot_xfer_transferability.py``
- ``plot_xfer_impact.py``
- ``plot_external_validation.py``
- ``plot_geo_cumulative.py``
- ``plot_hotspot_analytics.py``

Supplementary and extended visual analyses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These examples provide additional visual depth beyond the core figure
set and are useful when a workflow needs more detail than the main
results pages alone provide.

The main page in this group is:

- ``plot_uncertainty_extras.py``

Why this separation matters
---------------------------

This gallery deliberately keeps three concerns distinct:

- **artifact construction**,
- **plot generation**,
- **figure interpretation**.

That separation makes the workflow easier to understand. It also helps
users distinguish between commands that build reusable analysis
products, scripts that turn those products into figures, and pages that
focus on explaining what the resulting visuals actually mean.

Notes
-----

- These examples are intentionally compact and lesson-oriented.
- The scripts in this section are plot-first: they may print small
  summaries, but their main output is a figure file.
- A useful rule of thumb is:

  - ``tables_and_summaries/`` builds reusable artifacts,
  - ``model_inspection/`` explains helper-based diagnostics,
  - ``figure_generation/`` turns results into visual outputs.

- Many of these scripts are naturally suited to PNG, SVG, or EPS export
  for manuscripts, appendices, reports, and presentation material.