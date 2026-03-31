.. _cli-plot-family:

Plot family
===========

The **plot family** is where you render figures, maps, and visual
diagnostics from GeoPrior artifacts.

Use this page when your goal is to **visualize** model outputs, physics
payloads, transfer results, uncertainty behavior, or supplementary
diagnostics. In contrast to the run family, plot commands do not
primarily execute the full workflow. In contrast to the build family,
they do not primarily materialize reusable tabular or NPZ artifacts.
Their job is to turn existing artifacts into clear visual outputs.

You can invoke these commands either from the root dispatcher:

.. code-block:: bash

   geoprior plot <command> [args]

or from the family-specific entry point:

.. code-block:: bash

   geoprior-plot <command> [args]

Many plot commands also preserve a legacy reproducibility path through:

.. code-block:: bash

   python -m scripts <legacy-plot-command> [args]

This page is intentionally designed a little differently from the run
and build family pages.

Because many plot commands share the same figure-output habits, text
controls, and rendering conventions, this page introduces those shared
patterns **once** near the top. That lets the per-command sections stay
lighter later on.

How to choose a plot command
----------------------------

A practical way to choose a plot command is to first decide **what kind
of visual story** you want to tell:

- choose a **forecasting or uncertainty** plot when you want to show
  predictive behavior, forecast spread, or uncertainty structure,
- choose a **physics or sensitivity** plot when you want to inspect
  learned physical fields, diagnostics, or parameter sensitivity,
- choose a **transfer or validation** plot when you want to compare
  cross-city generalization or external site support,
- choose a **supporting or supplementary** plot when you want a figure
  focused on ablation, driver-response, lithology parity, or SM3
  diagnostics.

Plot commands at a glance
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 32 22 22

   * - Command
     - Use it when
     - Main output
     - Related page
   * - ``driver-response``
     - You want a figure linking drivers to subsidence response.
     - Driver-response figure.
     - :doc:`../auto_examples/figure_generation/plot_driver_response`
   * - ``core-ablation``
     - You want the main ablation-style summary figure.
     - Core ablation figure.
     - :doc:`../auto_examples/figure_generation/plot_core_ablation`
   * - ``litho-parity``
     - You want lithology parity checks or supporting comparison plots.
     - Lithology parity figure.
     - :doc:`../auto_examples/figure_generation/plot_litho_parity`
   * - ``uncertainty``
     - You want the main forecast uncertainty figure.
     - Uncertainty figure.
     - :doc:`../auto_examples/figure_generation/plot_uncertainty`
   * - ``spatial-forecasts``
     - You want forecast maps across space.
     - Spatial forecast maps.
     - :doc:`../auto_examples/figure_generation/plot_spatial_forecasts`
   * - ``physics-sanity``
     - You want compact physics sanity checks.
     - Physics sanity figure.
     - :doc:`../auto_examples/figure_generation/plot_physics_sanity`
   * - ``physics-maps``
     - You want mapped physics diagnostics.
     - Physics maps.
     - :doc:`../auto_examples/figure_generation/plot_physics_maps`
   * - ``physics-fields``
     - You want rendered learned physical fields.
     - Physics fields figure.
     - :doc:`../auto_examples/figure_generation/plot_physics_fields`
   * - ``physics-profiles``
     - You want appendix-style physical profiles.
     - Physics profiles figure.
     - :doc:`../auto_examples/figure_generation/plot_physics_profiles`
   * - ``uncertainty-extras``
     - You want supplementary uncertainty panels.
     - Extra uncertainty figure.
     - :doc:`../auto_examples/figure_generation/plot_uncertainty_extras`
   * - ``ablations-sensitivity``
     - You want sensitivity-style ablation visualization.
     - Ablation sensitivity figure.
     - :doc:`../auto_examples/figure_generation/index`
   * - ``physics-sensitivity``
     - You want visual summaries of physics sensitivity sweeps.
     - Physics sensitivity figure.
     - :doc:`../auto_examples/figure_generation/plot_physics_sensitivity`
   * - ``sm3-identifiability``
     - You want the figure view of SM3 identifiability experiments.
     - SM3 identifiability figure.
     - :doc:`../auto_examples/figure_generation/plot_sm3_identifiability`
   * - ``sm3-bounds-ridge-summary``
     - You want a summary figure relating SM3 bounds and ridge behavior.
     - SM3 bounds/ridge summary.
     - :doc:`../auto_examples/figure_generation/plot_sm3_bounds_ridge_summary`
   * - ``sm3-log-offsets``
     - You want the visual diagnostic for SM3 log offsets.
     - SM3 log-offset figure.
     - :doc:`../auto_examples/figure_generation/plot_sm3_log_offsets`
   * - ``xfer-transferability``
     - You want cross-city transferability comparison.
     - Transferability figure.
     - :doc:`../auto_examples/figure_generation/plot_xfer_transferability`
   * - ``xfer-impact``
     - You want transfer impact summaries.
     - Transfer impact figure.
     - :doc:`../auto_examples/figure_generation/plot_xfer_impact`
   * - ``transfer``
     - You want the short alias for the transferability plot.
     - Transferability figure.
     - :doc:`../auto_examples/figure_generation/plot_xfer_transferability`
   * - ``transfer-impact``
     - You want the short alias for the transfer impact plot.
     - Transfer impact figure.
     - :doc:`../auto_examples/figure_generation/plot_xfer_impact`
   * - ``geo-cumulative``
     - You want cumulative geoscience-style curves.
     - Cumulative geo figure.
     - :doc:`../auto_examples/figure_generation/plot_geo_cumulative`
   * - ``hotspot-analytics``
     - You want maps and timelines for hotspot behavior.
     - Hotspot analytics figure.
     - :doc:`../auto_examples/figure_generation/plot_hotspot_analytics`
   * - ``external-validation``
     - You want external point-support validation plots.
     - External validation figure.
     - :doc:`../auto_examples/figure_generation/plot_external_validation`

Shared plot conventions
-----------------------

Unlike many run and build commands, plot commands often share a strong
set of output and rendering conventions. Learning these once makes the
rest of the plot family much easier to use.

Common invocation patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

Most plotting workflows follow one of these forms:

.. code-block:: bash

   geoprior plot physics-fields --help
   geoprior-plot uncertainty
   python -m scripts plot-physics-fields --help

The modern public interface is the first one users should learn. The
legacy ``python -m scripts`` path remains useful for reproducibility,
older notes, and backward-compatible script runs.

Figure output paths
~~~~~~~~~~~~~~~~~~~

Many plot commands expose an ``--out`` argument.

GeoPrior resolves that output path with a simple rule set:

- an **absolute path** is used as given,
- a **relative path with an explicit parent** is respected as written,
- a **bare filename or stem** is routed to the user-facing
  ``scripts/figs/`` export area.

This is what makes commands like the following feel natural:

.. code-block:: bash

   geoprior-plot physics-fields --out phys_main
   geoprior-plot uncertainty --out exports/uncertainty_main
   geoprior-plot hotspot-analytics --out /tmp/hotspots_main

In practice, a short bare name is often the nicest choice when you want
the command to follow the standard artifact layout automatically.

Figure export formats
~~~~~~~~~~~~~~~~~~~~~

Plot helpers treat the output as a **figure stem**. If you provide a
suffix such as ``.png``, the shared helpers can strip it and treat the
value as a stem.

Many plot scripts save figures through the shared ``save_figure`` helper,
which supports multi-format export such as:

- PNG
- SVG
- PDF
- EPS

That design is especially convenient when you want one figure for
documentation, one for paper submission, and one editable vector version
for later refinement.

Text visibility and Illustrator-friendly output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several figure scripts reuse a shared text-control layer intended for
paper workflows and later editing. Common flags include:

- ``--show-legend``
- ``--show-labels``
- ``--show-ticklabels``
- ``--show-title``
- ``--show-panel-titles``
- ``--title``

These options make it easier to produce:

- one version with full labels for gallery use,
- another version with reduced text for Illustrator or layout editing.

A common pattern is:

.. code-block:: bash

   geoprior-plot physics-fields \
       --out phys_fields_main \
       --show-title false \
       --show-panel-titles false

Render styles for sensitivity surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several sensitivity-style figure scripts also share reusable rendering
controls. Common options include:

- ``--render`` with choices such as ``heatmap``, ``tricontour``, and
  ``pcolormesh``,
- ``--levels`` for contour density,
- ``--clip`` for percentile-based color scaling,
- ``--agg`` for duplicate-grid aggregation,
- ``--show-points`` for overlaying sampled points.

That means many sensitivity figures can move smoothly between a more
discrete grid-like rendering and a smoother contour-style rendering
without changing the underlying workflow.

Example:

.. code-block:: bash

   geoprior-plot physics-sensitivity \
       --out phys_sensitivity_main \
       --render tricontour \
       --levels 18 \
       --clip 2,98 \
       --show-points true

Artifact roots and environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared plotting layer also supports artifact-root environment
variables such as:

- ``GEOPRIOR_ARTIFACT_ROOT``
- ``GEOPRIOR_FIG_DIR``
- ``GEOPRIOR_OUT_DIR``

When these are not set explicitly, the default figure export area is
built under the project-level ``scripts/figs/`` directory. This helps
keep documentation figures, reproducibility figures, and quick manual
runs aligned to the same filesystem logic.

How this plot family is organized
---------------------------------

To keep the page easy to navigate, the detailed plot sections that
follow are grouped into a few practical categories.

The first group covers **forecasting and uncertainty** figures. These
commands focus on predictive spread, uncertainty structure, forecast
maps, and hotspot-style forecast summaries.

The second group covers **physics and physical diagnostics**. These
commands focus on learned fields, physics sanity checks, profiles,
sensitivity sweeps, and SM3-oriented diagnostics.

The third group covers **transfer and validation** figures. These
commands focus on cross-city transfer behavior and external site-based
validation.

The fourth group covers **supporting and supplementary figures**. These
commands include ablation, driver-response, parity, and additional
figure panels that support the broader GeoPrior workflow.

The rest of this page builds on that structure so that users can think
in terms of **what they want to visualize**, not only in terms of
individual command names.

Forecasting and uncertainty
---------------------------

``uncertainty``
~~~~~~~~~~~~~~~

Use ``uncertainty`` when you want the main **uncertainty calibration
figure** built from calibrated forecast CSVs, optionally enriched with
GeoPrior phys JSON for interval-calibration notes.

This command produces the main Fig. 5 style layout: reliability,
per-horizon mini panels, and radial sharpness summaries. It also writes
a compact metrics table to ``scripts/out/`` through ``--out-csv``.

Usage
^^^^^

Run the figure from auto-detected city run directories:

.. code-block:: bash

   geoprior plot uncertainty \
       --ns-src results/nansha_run \
       --zh-src results/zhongshan_run \
       --split auto

Use explicit forecast CSVs and phys JSON overrides:

.. code-block:: bash

   geoprior-plot uncertainty \
       --ns-forecast results/nansha_eval_calibrated.csv \
       --zh-forecast results/zhongshan_eval_calibrated.csv \
       --ns-phys-json results/nansha_eval_phys.json \
       --zh-phys-json results/zhongshan_eval_phys.json \
       --out fig5_uncertainty_main \
       --out-csv ext-table-fig5-uncertainty.csv \
       --radial-title city \
       --show-point-values true \
       --show-mini-titles true \
       --show-json-notes true

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--ns-src`` / ``--zh-src`` and ``--ns-forecast`` / ``--zh-forecast``
   Choose between auto-discovery from run directories and explicit
   calibrated forecast CSV overrides.

``--ns-phys-json`` / ``--zh-phys-json``
   Add GeoPrior JSON notes to the reliability panels when available.

``--split``
   Choose ``auto``, ``val``, or ``test`` when inputs are discovered from
   run directories.

``--out-csv``
   Write the exported uncertainty metrics table.

``--radial-title``
   Control whether radial panels use ``full``, ``city``, or ``none``.

``--show-point-values`` / ``--show-mini-titles`` / ``--show-mini-legend`` / ``--show-json-notes``
  Fine-tune the richer uncertainty annotations.

**Related examples:**

- :doc:`../auto_examples/figure_generation/plot_uncertainty`
- :doc:`../auto_examples/uncertainty/index`


``spatial-forecasts``
~~~~~~~~~~~~~~~~~~~~~

Use ``spatial-forecasts`` when you want the main **spatial validation
plus forecast maps** figure.

This command combines one observed validation-year map, one predicted
validation-year map, and one or more future forecast-year maps. It can
also mark hotspot contours and export hotspot points.

Usage
^^^^^

Run the figure from auto-detected run directories:

.. code-block:: bash

   geoprior plot spatial-forecasts \
       --ns-src results/nansha_run \
       --zh-src results/zhongshan_run \
       --split auto

Use explicit calibrated and future CSVs with forecast controls:

.. code-block:: bash

   geoprior-plot spatial-forecasts \
       --ns-calib results/nansha_eval_calibrated.csv \
       --zh-calib results/zhongshan_eval_calibrated.csv \
       --ns-future results/nansha_future.csv \
       --zh-future results/zhongshan_future.csv \
       --year-val 2022 \
       --years-forecast 2025 2026 2027 \
       --cumulative \
       --subsidence-kind cumulative \
       --grid-res 300 \
       --clip 98 \
       --hotspot delta \
       --hotspot-quantile 0.90 \
       --hotspot-out fig6_hotspot_points.csv \
       --out fig6_spatial_forecasts_main

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--year-val`` and ``--years-forecast``
   Control the observed/predicted validation year and the additional
   forecast columns.

``--cumulative`` and ``--subsidence-kind``
   Control whether forecast panels use cumulative or annualized values.

``--grid-res`` and ``--clip``
   Control rasterization resolution and shared color clipping.

``--hotspot``
   Choose hotspot contour mode: ``none``, ``absolute``, or ``delta``.

``--hotspot-quantile`` and ``--hotspot-out``
   Control hotspot selection and optional hotspot point export.

**Related examples:**

- :doc:`../auto_examples/figure_generation/plot_spatial_forecasts`
- :doc:`../auto_examples/forecasting/index`


``hotspot-analytics``
~~~~~~~~~~~~~~~~~~~~~

Use ``hotspot-analytics`` when you want the **decision-maker hotspot
figure**: anomaly maps, exceedance-risk maps, a timeline view, and a
priority ranking of hotspot clusters.

This command also exports three downstream tables by default:
``hotspot_points.csv``, ``hotspot_years.csv``, and
``hotspot_clusters.csv``.

Usage
^^^^^

Run the default hotspot analytics workflow from run directories:

.. code-block:: bash

   geoprior plot hotspot-analytics \
       --ns-src results/nansha_run \
       --zh-src results/zhongshan_run \
       --years 2025 2026

Use explicit eval/future CSVs with richer hotspot controls:

.. code-block:: bash

   geoprior-plot hotspot-analytics \
       --ns-eval results/nansha_eval.csv \
       --zh-eval results/zhongshan_eval.csv \
       --ns-future results/nansha_future.csv \
       --zh-future results/zhongshan_future.csv \
       --base-year 2022 \
       --years 2025 2026 2027 \
       --focus-year 2027 \
       --subsidence-kind cumulative \
       --hotspot-quantile 0.90 \
       --hotspot-rule combo \
       --hotspot-abs 40 \
       --risk-threshold 50 \
       --risk-min 0.6 \
       --exposure exposure.csv \
       --timeline-mode ever \
       --timeline-overlay-current true \
       --cluster-rank risk \
       --add-persistence \
       --persistence-mode fraction \
       --boundary boundary.geojson \
       --out hotspot_analytics_main \
       --out-points hotspot_points.csv \
       --out-years hotspot_years.csv \
       --out-clusters hotspot_clusters.csv

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--base-year`` / ``--years`` / ``--focus-year``
   Control anomaly reference year, analysis years, and the main focus
   year for maps and ranking.

``--hotspot-rule``
   Choose the hotspot definition used for counts, clusters, and
   persistence: ``percentile``, ``abs``, ``risk``, or ``combo``.

``--hotspot-abs`` / ``--risk-threshold`` / ``--risk-min``
   Control absolute anomaly and exceedance-probability thresholds.

``--exposure`` / ``--exposure-col`` / ``--exposure-default``
   Add an exposure-weighted risk layer.

``--timeline-mode`` and ``--timeline-overlay-current``
   Control how hotspot evolution is summarized over time.

``--add-persistence`` / ``--persistence-mode``
   Add a persistence panel as fraction or count.

``--boundary`` / ``--ns-boundary`` / ``--zh-boundary``
   Overlay or filter to optional boundary files.

``--out-points`` / ``--out-years`` / ``--out-clusters``
   Control the three exported hotspot tables.

**Related examples:**

- :doc:`../auto_examples/figure_generation/plot_hotspot_analytics`
- :doc:`../auto_examples/tables_and_summaries/compute_hotspots`


``uncertainty-extras``
~~~~~~~~~~~~~~~~~~~~~~

Use ``uncertainty-extras`` when you want the **supplementary uncertainty
diagnostics** that go beyond the main uncertainty figure.

This command adds expanded reliability, PIT, coverage, sharpness,
per-horizon calibration-factor views, and writes both CSV and TeX
tables.

Usage
^^^^^

Run the supplementary figure from run directories:

.. code-block:: bash

   geoprior plot uncertainty-extras \
       --ns-src results/nansha_run \
       --zh-src results/zhongshan_run \
       --split auto

Use explicit forecast inputs and table export controls:

.. code-block:: bash

   geoprior-plot uncertainty-extras \
       --ns-forecast results/nansha_eval_calibrated.csv \
       --zh-forecast results/zhongshan_eval_calibrated.csv \
       --ns-phys-json results/nansha_eval_phys.json \
       --zh-phys-json results/zhongshan_eval_phys.json \
       --bootstrap 2000 \
       --out supp-fig-s5-uncertainty-extras \
       --tables-stem supp-table-s5-reliability \
       --no-legends \
       --show-panel-titles true

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--bootstrap``
   Control the number of bootstrap draws used for confidence intervals.

``--tables-stem``
   Set the shared stem for CSV and TeX table export.

``--no-legends``
   Convenient switch for suppressing legends in this figure.

``--ns-phys-json`` / ``--zh-phys-json``
   Provide optional per-horizon calibration factors from GeoPrior JSON.

**Related examples:**

- :doc:`../auto_examples/figure_generation/plot_uncertainty_extras`
- :doc:`../auto_examples/uncertainty/index`


Core physics figures
--------------------

``physics-fields``
~~~~~~~~~~~~~~~~~~

Use ``physics-fields`` when you want the **richer spatial physics-field
figure** with more rendering and presentation controls.

This command renders the six core spatial physics panels
(``log10(K)``, ``log10(Ss)``, ``Hd``, ``log10(tau)``,
``log10(tau_p)``, and ``Δlog10(tau)``), and it supports either gridded
imshow rendering or hexbin rendering, optional boundary overlays,
scalebars, north arrows, PDF export, and optional JSON range export.

Usage
^^^^^

Run the figure from one source directory or one payload:

.. code-block:: bash

   geoprior plot physics-fields \
       --src results/nansha_run

or:

.. code-block:: bash

   geoprior-plot physics-fields \
       --payload results/nansha_phys_payload.npz

Use richer rendering and overlay controls:

.. code-block:: bash

   geoprior-plot physics-fields \
       --src results/nansha_run \
       --coords-npz results/nansha_coords.npz \
       --city Nansha \
       --coord-kind auto \
       --agg mean \
       --grid 240 \
       --render hexbin \
       --hex-gridsize 80 \
       --hex-mincnt 1 \
       --clip-q 2 98 \
       --delta-q 98 \
       --cbar-label-mode title \
       --boundary-auto true \
       --boundary-to-crs EPSG:4326 \
       --boundary-lw 0.4 \
       --boundary-alpha 0.75 \
       --scalebar true \
       --north-arrow true \
       --export-pdf true \
       --out-json fig_physics_fields_ranges.json \
       --out fig_physics_fields_main

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--src`` / ``--payload`` / ``--coords-npz``
   Choose between auto-detected payloads, explicit payload paths, and
   optional coordinate NPZ sidecars.

``--render``
   Choose between ``hexbin`` and imshow-style rendering.

``--hex-gridsize`` / ``--hex-mincnt``
   Matter when ``--render hexbin`` is used.

``--boundary-auto`` / ``--boundary`` / ``--boundary-to-crs`` / ``--boundary-lw`` / ``--boundary-alpha``
   Control optional boundary discovery and overlay styling.

``--scalebar`` / ``--north-arrow`` / ``--scalebar-km``
   Add map-style panel embellishments.

``--export-pdf`` and ``--out-json``
   Control extra figure and range-metadata export.

**Related examples:**

- :doc:`../auto_examples/figure_generation/plot_physics_fields`
- :doc:`../scientific_foundations/physics_formulation`


``physics-maps``
~~~~~~~~~~~~~~~~

Use ``physics-maps`` when you want the **lighter fixed-grid spatial
physics map figure**.

This command renders the same six core physics panels as
``physics-fields``, but with a simpler gridded map interface and fewer
presentation extras.

Usage
^^^^^

Run the figure from one source directory or one payload:

.. code-block:: bash

   geoprior plot physics-maps \
       --src results/nansha_run

or:

.. code-block:: bash

   geoprior-plot physics-maps \
       --payload results/nansha_phys_payload.npz

Use explicit coordinate and styling controls:

.. code-block:: bash

   geoprior-plot physics-maps \
       --src results/nansha_run \
       --coords-npz results/nansha_coords.npz \
       --city Nansha \
       --coord-kind auto \
       --agg median \
       --grid 220 \
       --clip-q 2 98 \
       --delta-q 98 \
       --cmap viridis \
       --cmap-div RdBu_r \
       --hatch-censored true \
       --show-panel-labels true \
       --out-json fig_physics_maps_ranges.json \
       --out fig_physics_maps_main

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--agg`` and ``--grid``
   Control how point payloads are rasterized into maps.

``--clip-q`` and ``--delta-q``
   Control percentile clipping of the main ranges and the symmetric
   tension panel.

``--hatch-censored``
   Overlay censored regions when that mask exists in the payload.

``--out-json``
   Write the mapped value ranges as a JSON side artifact.

**Related examples:**

- :doc:`../auto_examples/figure_generation/plot_physics_maps`
- :doc:`../scientific_foundations/physics_formulation`


``physics-sanity``
~~~~~~~~~~~~~~~~~~

Use ``physics-sanity`` when you want the **two-case physics sanity
figure** that compares learned and prior time scales and summarizes the
consistency residual distribution.

This command is intentionally case-oriented rather than city-hardcoded:
it expects exactly **two** ``--src`` inputs and can infer or accept one
city label per source.

Usage
^^^^^

Run the standard two-case sanity figure:

.. code-block:: bash

   geoprior plot physics-sanity \
       --src results/nansha_run \
       --src results/zhongshan_run

Use a more diagnostic configuration:

.. code-block:: bash

   geoprior-plot physics-sanity \
       --src results/nansha_run \
       --src results/zhongshan_run \
       --city Nansha \
       --city Zhongshan \
       --plot-mode residual \
       --tau-scale log10 \
       --cons-kind scaled \
       --gridsize 140 \
       --hist-bins 50 \
       --clip-q 1 99 \
       --subsample-frac 0.5 \
       --max-points 50000 \
       --extra k-from-tau,closure \
       --paper-format \
       --paper-no-offset \
       --out-json fig4_physics_sanity.json \
       --out fig4_physics_sanity_main

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--src``
   Must be given exactly twice.

``--city``
   Optional per-source city labels, repeated to match ``--src`` order.

``--plot-mode``
   Choose ``joint`` or ``residual``.

``--tau-scale``
   Control ``log10`` or ``linear`` tau display in joint mode.

``--cons-kind``
   Choose between raw ``cons_res_vals`` and scaled
   ``cons_res_scaled``.

``--extra``
   Add extra derived checks such as ``k-from-tau`` and ``closure``.

``--paper-format`` / ``--paper-no-offset``
   Enable paper-style scientific-axis formatting.

``--out-json``
   Export summary sanity statistics as JSON.

**Related examples:**

- :doc:`../auto_examples/figure_generation/plot_physics_sanity`
- :doc:`../auto_examples/model_inspection/index`


``physics-profiles``
~~~~~~~~~~~~~~~~~~~~

Use ``physics-profiles`` when you want **1D physics sensitivity
profiles** from the ablation-record JSONL collection.

This command scans ablation records under the chosen root, keeps the
physics-on rows, and plots metric profiles against ``lambda_prior`` and
``lambda_cons`` for two cities. It also writes a tidy CSV copy next to
the figure.

Usage
^^^^^

Run the default appendix-style profiles from the results root:

.. code-block:: bash

   geoprior plot physics-profiles \
       --root results

Use custom cities and metrics:

.. code-block:: bash

   geoprior-plot physics-profiles \
       --root results \
       --city-a Nansha \
       --city-b Zhongshan \
       --metric-prior epsilon_prior \
       --metric-cons epsilon_cons \
       --show-best true \
       --out appendix_fig_A1_phys_profiles

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--root``
   Root scanned for ``**/ablation_records/ablation_record*.jsonl``.

``--city-a`` / ``--city-b``
   Choose the two city names used in the two profile rows.

``--metric-prior`` / ``--metric-cons``
   Choose which metrics are plotted against ``lambda_prior`` and
   ``lambda_cons``.

``--show-best``
   Highlight the best point in each panel.

``--out-dir``
   Optional backward-compatible output-directory override.

**Related examples:**

- :doc:`../auto_examples/figure_generation/plot_physics_profiles`
- :doc:`../auto_examples/diagnostics/index`

Sensitivity and SM3 diagnostics
-------------------------------

``physics-sensitivity``
~~~~~~~~~~~~~~~~~~~~~~~

Use ``physics-sensitivity`` when you want the **2D sensitivity surface
figure** over ``lambda_prior`` and ``lambda_cons``.

This command scans ablation records, prefers updated records when
possible, supports explicit CSV/JSON/JSONL inputs, and plots one row for
the prior-oriented metric and one row for the consistency-oriented
metric. It also writes the actually used tidy table as
``tableS7_physics_used.csv`` next to the figure. 

Usage
^^^^^

Scan the results root and render the default figure:

.. code-block:: bash

   geoprior plot physics-sensitivity \
       --root results

Use explicit inputs, city/model filters, and richer rendering:

.. code-block:: bash

   geoprior-plot physics-sensitivity \
       --input results/runA/ablation_record.updated.jsonl \
       --input results/runB/ablation_record.updated.jsonl \
       --cities Nansha Zhongshan \
       --models GeoPriorSubsNet \
       --pde-modes both \
       --metric-prior epsilon_prior \
       --metric-cons epsilon_cons \
       --render tricontour \
       --levels 18 \
       --agg median \
       --clip 2,98 \
       --show-points true \
       --trend-arrow true \
       --out supp_fig_S7_physics_sensitivity

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--input`` and ``--root``
   Choose between explicit ablation tables and root-directory scanning.

``--cities`` / ``--models`` / ``--pde-modes``
   Filter the plotted rows before rendering. 

``--metric-prior`` and ``--metric-cons``
   Select the two metrics shown in the two figure rows.

``--render`` / ``--levels`` / ``--agg`` / ``--clip``
   Control how the sensitivity surfaces are rendered and clipped.

``--show-points`` and ``--trend-arrow``
   Add sampled-point overlays and an improvement-direction arrow.

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_physics_sensitivity`


``sm3-identifiability``
~~~~~~~~~~~~~~~~~~~~~~~

Use ``sm3-identifiability`` when you want the main **SM3 synthetic
identifiability figure**.

This command reads the SM3 synthetic runs CSV and builds the four-panel
figure covering timescale recovery, permeability recovery, ridge
degeneracy, and error versus identifiability metric. It also writes a
summary CSV and JSON. 

Usage
^^^^^

Run the default SM3 identifiability figure:

.. code-block:: bash

   geoprior plot sm3-identifiability \
       --csv results/sm3_synth_1d/sm3_synth_runs.csv

Use stricter filtering and richer overlays:

.. code-block:: bash

   geoprior-plot sm3-identifiability \
       --csv results/sm3_synth_1d/sm3_synth_runs.csv \
       --only-identify tau \
       --nx-min 50 \
       --tau-units year \
       --metric ridge_resid \
       --k-from-tau auto \
       --k-cl-source prior \
       --n-boot 3000 \
       --ci 0.95 \
       --show-ci true \
       --show-stats true \
       --show-prior true \
       --show-tau-cl true \
       --out-csv sm3_identifiability_summary.csv \
       --out-json sm3_identifiability_summary.json \
       --out sm3-identifiability

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--only-identify`` and ``--nx-min``
   Filter the SM3 runs before plotting. 

``--tau-units``
   Switch panel (a) between year and second units.

``--metric``
   Choose the identifiability metric used in panel (d).

``--k-from-tau`` and ``--k-cl-source``
   Control whether permeability is derived from tau and which
   ``Ss``/``Hd`` source is used for that closure calculation.

``--n-boot`` / ``--ci`` / ``--show-ci``
   Control bootstrap confidence intervals in the annotation blocks.

``--show-prior`` and ``--show-tau-cl``
   Toggle the prior and closure overlays in panel (a).

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_sm3_identifiability`


``sm3-bounds-ridge-summary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``sm3-bounds-ridge-summary`` when you want the **SM3 failure-mode
summary figure**.

This command summarizes two failure modes in SM3 synthetic runs:
clipping to inferred bounds and ridge-style non-identifiability. It also
exports a category CSV and a JSON summary. 

Usage
^^^^^

Run the default summary from the SM3 runs CSV:

.. code-block:: bash

   geoprior plot sm3-bounds-ridge-summary \
       --csv results/sm3_synth_1d/sm3_synth_runs.csv

Use stricter filtering and paper-style labels:

.. code-block:: bash

   geoprior-plot sm3-bounds-ridge-summary \
       --csv results/sm3_synth_1d/sm3_synth_runs.csv \
       --only-identify both \
       --nx-min 50 \
       --use any \
       --ridge-thr 2.0 \
       --rtol 1e-6 \
       --paper-format \
       --out-csv sm3-clip-vs-ridge-categories.csv \
       --out-json sm3-clip-vs-ridge-summary.json \
       --out sm3-clip-vs-ridge

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--use``
   Choose the clipping definition used in panel (c): ``any`` or
   ``primary``. 

``--ridge-thr`` and ``--rtol``
   Control the ridge threshold and the tolerance used when inferring
   clipped-to-bound flags. 

``--only-identify`` and ``--nx-min``
   Filter the SM3 runs before summary.

``--paper-format``
   Switch to paper-friendly labels in the summary figure.

``--out-csv`` and ``--out-json``
   Export the category table and summary payload.


**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_sm3_bounds_ridge_summary`


``sm3-log-offsets``
~~~~~~~~~~~~~~~~~~~

Use ``sm3-log-offsets`` when you want a **payload-level SM3 offset
diagnostic** rather than a run-summary figure.

This command works directly from a physics payload, builds a raw offsets
table, writes a compact summary table, and renders the histogram and tau
scatter diagnostics. It can also use scalar priors when the payload does
not embed them. 

Usage
^^^^^

Auto-discover the payload under one run directory:

.. code-block:: bash

   geoprior plot sm3-log-offsets \
       --src results/sm3_synth_1d

Use an explicit payload and manual prior values:

.. code-block:: bash

   geoprior-plot sm3-log-offsets \
       --payload results/sm3_synth_1d/physics_payload_run_val.npz \
       --K-prior 1e-7 \
       --Ss-prior 1e-5 \
       --Hd-prior 40 \
       --bins 60 \
       --out-raw-csv sm3-offsets-raw.csv \
       --out-summary-csv sm3-offsets-summary.csv \
       --out-json sm3-offsets.json \
       --out sm3-log-offsets

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--src`` / ``--payload``
   Choose between auto-discovery and one explicit payload file.

``--K-prior`` / ``--Ss-prior`` / ``--Hd-prior``
   Provide scalar priors when the payload does not contain prior series.

``--out-raw-csv`` / ``--out-summary-csv`` / ``--out-json``
   Export the raw offsets table, compact summary, and JSON payload.

``--bins``
   Control the histogram bin count.

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_sm3_log_offsets`


Transfer and validation
-----------------------

``xfer-transferability``
~~~~~~~~~~~~~~~~~~~~~~~~

Use ``xfer-transferability`` when you want the **lighter cross-city
transferability figure** built from ``xfer_results.csv``.

This command focuses on the basic comparison view: transfer metrics
across calibration modes plus the coverage–sharpness panel. It is the
cleaner entry point when you want one concise transfer summary rather
than the broader impact workflow. 

Usage
^^^^^

Auto-detect the latest transfer results under one source directory:

.. code-block:: bash

   geoprior plot xfer-transferability \
       --src results/xfer/nansha__zhongshan \
       --split val

Use explicit CSV input and custom metric choices:

.. code-block:: bash

   geoprior-plot xfer-transferability \
       --xfer-csv results/xfer/nansha__zhongshan/20260126-164613/xfer_results.csv \
       --split test \
       --strategies baseline xfer warm \
       --calib-modes none source target \
       --rescale-mode strict \
       --baseline-rescale as_is \
       --metric-top mae \
       --metric-bottom r2 \
       --reduce best \
       --cov-target 0.80 \
       --out xfer_transferability

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--xfer-csv`` / ``--src``
   Choose between explicit CSV input and latest-result discovery.

``--strategies`` and ``--calib-modes``
   Control which transfer strategies and calibration modes appear in the
   figure. 

``--metric-top`` and ``--metric-bottom``
   Control the two bar-panel metrics.

``--rescale-mode`` and ``--baseline-rescale``
   Matter when matching cross-city rows to their baseline references.

``--reduce`` and ``--cov-target``
   Control duplicate reduction and the reference coverage line.

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_xfer_transferability`

Alias:
``transfer``


``xfer-impact``
~~~~~~~~~~~~~~~

Use ``xfer-impact`` when you want the **broader decision-oriented
transfer figure**.

This command extends the transferability view with retention panels,
risk diagnostics, and optional hotspot stability panels. It can also use
``xfer_results.json`` when available to resolve evaluation CSVs for the
risk and hotspot diagnostics. 

Usage
^^^^^

Run the core impact figure from one transfer-results directory:

.. code-block:: bash

   geoprior plot xfer-impact \
       --src results/xfer/nansha__zhongshan \
       --split val \
       --calib source

Use explicit CSV/JSON inputs and enable hotspot stability panels:

.. code-block:: bash

   geoprior-plot xfer-impact \
       --xfer-csv results/xfer/nansha__zhongshan \
       --xfer-json results/xfer/nansha__zhongshan/xfer_results.json \
       --split test \
       --calib target \
       --strategies baseline xfer warm \
       --rescale-mode strict \
       --baseline-rescale as_is \
       --horizon-metric rmse \
       --cov-target 0.80 \
       --threshold 50 \
       --add-hotspots true \
       --hotspot-k 100 \
       --hotspot-score exceed \
       --hotspot-horizon H3 \
       --hotspot-ref baseline \
       --hotspot-style timeline \
       --hotspot-errorbars true \
       --out figureS_xfer_impact

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--calib``
   Choose the calibration mode emphasized across the impact panels.
   
``--horizon-metric``
   Control the metric used in the horizon-retention panels.

``--threshold``
   Set the exceedance threshold used in the risk panel.

``--xfer-json``
   Provide or override the JSON metadata used to resolve evaluation CSVs
   for risk and hotspot computations. 

``--add-hotspots`` / ``--hotspot-k`` / ``--hotspot-score`` / ``--hotspot-horizon`` / ``--hotspot-ref`` / ``--hotspot-style`` / ``--hotspot-errorbars``
   Enable and tune the hotspot-stability extensions.


**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_xfer_impact`

Alias:
``transfer-impact``


``external-validation``
~~~~~~~~~~~~~~~~~~~~~~~

Use ``external-validation`` when you want the **point-support external
validation figure** for inferred effective fields.

This command builds the three-panel supplementary validation figure from
a site-level table, ``full_inputs.npz``, and one full-city physics
payload. It can also apply a coordinate scaler, add an optional boundary
overlay, and write a compact summary JSON. 

Usage
^^^^^

Run the validation figure with the required artifacts:

.. code-block:: bash

   geoprior plot external-validation \
       --site-csv nat.com/boreholes_zhongshan_with_model.csv \
       --full-inputs-npz results/zhongshan_GeoPriorSubsNet_stage1/external_validation_fullcity/full_inputs.npz \
       --full-payload-npz results/zhongshan_GeoPriorSubsNet_stage1/external_validation_fullcity/physics_payload_fullcity.npz

Use the full validation configuration with paper-style formatting:

.. code-block:: bash

   geoprior-plot external-validation \
       --site-csv nat.com/boreholes_zhongshan_with_model.csv \
       --full-inputs-npz results/zhongshan_GeoPriorSubsNet_stage1/external_validation_fullcity/full_inputs.npz \
       --full-payload-npz results/zhongshan_GeoPriorSubsNet_stage1/external_validation_fullcity/physics_payload_fullcity.npz \
       --coord-scaler results/zhongshan_GeoPriorSubsNet_stage1/artifacts/zhongshan_coord_scaler.joblib \
       --city Zhongshan \
       --horizon-reducer mean \
       --site-reducer median \
       --boundary nat.com/zhongshan_boundary.csv \
       --grid-res 260 \
       --paper-format \
       --paper-no-offset \
       --out-json supp_zhongshan_external_validation.json \
       --out supp_external_validation

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--site-csv`` / ``--full-inputs-npz`` / ``--full-payload-npz``
   These three are the core required inputs. 

``--coord-scaler``
   Inverse-transforms the coordinate stream when the full inputs are in
   scaled space. 

``--horizon-reducer`` and ``--site-reducer``
   Control how horizon-level arrays and site-level matches are reduced.

``--boundary``
   Add an optional boundary overlay in panel (a). 

``--paper-format`` and ``--paper-no-offset``
   Switch to a tighter paper layout and suppress offset text on linear

``--out-json``
   Export a compact validation summary payload. 

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_external_validation`

Supporting and supplementary figures
------------------------------------

``driver-response``
~~~~~~~~~~~~~~~~~~~

Use ``driver-response`` when you want a **driver-versus-response sanity
grid** across one or two cities.

This command builds a compact hexbin figure that compares one response
column against several driver columns, with optional robust trend lines.
It is useful when you want a quick physical sanity check rather than a
formal model-evaluation plot. 

Usage
^^^^^

Run the default driver-response figure from one dataset source:

.. code-block:: bash

   geoprior plot driver-response \
       --src data/final_dataset

Use explicit drivers, response, sampling, and trend controls:

.. code-block:: bash

   geoprior-plot driver-response \
       --src data/final_dataset \
       --drivers rainfall,head_m,building_density,seismic_risk \
       --response subsidence_cum \
       --year 2022 \
       --sample-frac 0.20 \
       --sharey row \
       --gridsize 70 \
       --vmin 1 \
       --trend true \
       --trend-bins 35 \
       --trend-min-n 25 \
       --out figS2_driver_response

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--drivers``
   Comma-separated list of driver columns used for the x-axes.

``--response`` and ``--subs-kind``
   Choose the response column directly, or let the command select
   ``subsidence_cum`` or ``subsidence`` automatically.

``--sharey``
   Choose shared y-axis behavior across panels: ``none``, ``row``, or
   ``all``. 

``--gridsize`` and ``--vmin``
   Control the hexbin density rendering. 

``--trend`` / ``--trend-bins`` / ``--trend-min-n``
   Enable and tune the robust median-trend overlay.
   

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_driver_response`


``core-ablation``
~~~~~~~~~~~~~~~~~

Use ``core-ablation`` when you want the **main core-versus-ablation
comparison figure** built from paired “with physics” and “no physics”
run directories.

This command collects headline metrics from phys JSON and evaluation
diagnostic JSON files, then renders the compact multi-panel comparison
used for the core ablation story. It also writes a CSV and can
optionally export TeX or XLSX tables. 

Usage
^^^^^

Run the default core ablation figure from paired run folders:

.. code-block:: bash

   geoprior plot core-ablation \
       --ns-with results/nansha_with_phys \
       --ns-no results/nansha_no_phys \
       --zh-with results/zhongshan_with_phys \
       --zh-no results/zhongshan_no_phys

Use the alternative metric layout and table exports:

.. code-block:: bash

   geoprior-plot core-ablation \
       --ns-with results/nansha_with_phys \
       --ns-no results/nansha_no_phys \
       --zh-with results/zhongshan_with_phys \
       --zh-no results/zhongshan_no_phys \
       --core-metric r2 \
       --err-metric rmse \
       --out-csv ext-table-fig3-metrics.csv \
       --out-tex ext-table-fig3-metrics.tex \
       --out-xlsx ext-table-fig3-metrics.xlsx \
       --show-values true \
       --show-panel-labels true \
       --out fig3-core-ablation

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--ns-with`` / ``--ns-no`` / ``--zh-with`` / ``--zh-no``
   Supply the paired run directories for the two cities.

``--core-metric``
   Choose whether the main large panel emphasizes ``mae`` or ``r2``.

``--err-metric``
   Choose ``rmse`` or ``mse`` for the error-oriented comparison panel.

``--out-csv`` / ``--out-tex`` / ``--out-xlsx``
   Export the collected comparison table in one or more formats.

``--show-values`` and ``--show-panel-labels``
   Control bar annotations and panel lettering.

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_core_ablation`


``litho-parity``
~~~~~~~~~~~~~~~~

Use ``litho-parity`` when you want a **city-to-city lithology
composition comparison**.

This command compares class proportions between Nansha and Zhongshan,
renders normalized composition bars plus a difference panel, and reports
a chi-square / Cramér’s V style parity summary in the title. It is a
good fit for quick distributional context rather than model evaluation.

Usage
^^^^^

Run the default lithology parity figure:

.. code-block:: bash

   geoprior plot litho-parity \
       --src data/final_dataset \
       -ns -zh

Use a different categorical column, year filter, and sampling setup:

.. code-block:: bash

   geoprior-plot litho-parity \
       --src data/final_dataset \
       --col lithology_class \
       --year 2022 \
       --sample-frac 0.25 \
       --top-n 8 \
       --group-others true \
       --sharey true \
       --out figS1_lithology_parity

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--col``
   Select the categorical column whose parity is being compared.

``--year``
   Restrict the comparison to one year or use ``all``.

``--sample-frac`` / ``--sample-n``
   Subsample the city datasets before computing proportions.

``--top-n`` and ``--group-others``
   Control how many classes are shown explicitly and whether the rest
   are grouped into ``Others``. 

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_litho_parity`


``ablations-sensitivity``
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``ablations-sensitivity`` when you want the **extended S6 ablations
and sensitivity figure**.

This is the richer supplementary ablation command. It can scan ablation
records or accept explicit inputs, prefer updated records, deduplicate
them, normalize the PDE mode into stable buckets, and then render bars,
maps, or even Pareto-style trade-off views. 

Usage
^^^^^

Run the default S6 figure from the results root:

.. code-block:: bash

   geoprior plot ablations-sensitivity \
       --root results

Use multiple heatmap metrics with a richer rendering setup:

.. code-block:: bash

   geoprior-plot ablations-sensitivity \
       --root results \
       --models GeoPriorSubsNet \
       --cities Nansha,Zhongshan \
       --heatmap-metrics mae,coverage80,sharpness80 \
       --no-bars true \
       --map-kind tricontour \
       --levels 14 \
       --contour-lines true \
       --cmap viridis \
       --align-grid true \
       --mark-lambda-cons 0.1 \
       --mark-lambda-prior 0.05 \
       --out supp_fig_S6_ablations

Or switch to a Pareto trade-off view:

.. code-block:: bash

   geoprior-plot ablations-sensitivity \
       --root results \
       --pareto true \
       --pareto-x mae \
       --pareto-y sharpness80 \
       --pareto-color coverage80 \
       --pareto-front true \
       --pareto-density true \
       --pareto-density-gridsize 35 \
       --out supp_fig_S6_pareto

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--input`` and ``--root``
   Choose between explicit ablation tables and root-directory scanning.

``--heatmap-metrics`` and ``--no-bars``
   Build a heatmap-only multi-metric layout instead of the legacy
   bar-plus-map figure. 

``--map-kind``
   Choose ``heatmap``, ``smooth``, ``contour``, or ``tricontour``.

``--align-grid``
   Keep lambda grids aligned across cities and metrics.
   
``--mark-lambda-cons`` and ``--mark-lambda-prior``
   Highlight one selected lambda pair on the maps.

``--pareto`` and related ``--pareto-*`` options
   Switch from map mode to trade-off scatter mode with optional Pareto
   front and density overlay. 

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_physics_sensitivity`


``geo-cumulative``
~~~~~~~~~~~~~~~~~~

Use ``geo-cumulative`` when you want **cumulative subsidence maps on a
satellite basemap** for validation and forecast years.

This command combines calibrated validation CSVs and future forecast
CSVs for Nansha and Zhongshan, converts them into cumulative
subsidence-since-start-year panels, and renders them over satellite
imagery. It can also overlay hotspot points and auto-detect whether the
coordinates are lon/lat or UTM. 

Usage
^^^^^

Run the default cumulative satellite-map figure:

.. code-block:: bash

   geoprior plot geo-cumulative \
       --ns-val results/nansha_eval_calibrated.csv \
       --zh-val results/zhongshan_eval_calibrated.csv \
       --ns-future results/nansha_future.csv \
       --zh-future results/zhongshan_future.csv

Use richer cumulative, rendering, and hotspot options:

.. code-block:: bash

   geoprior-plot geo-cumulative \
       --ns-val results/nansha_eval_calibrated.csv \
       --zh-val results/zhongshan_eval_calibrated.csv \
       --ns-future results/nansha_future.csv \
       --zh-future results/zhongshan_future.csv \
       --start-year 2020 \
       --year-val 2022 \
       --years-forecast 2024 2025 2026 \
       --subsidence-kind cumulative \
       --clip 99 \
       --cmap viridis \
       --render-mode auto \
       --surface-levels 15 \
       --surface-alpha 0.72 \
       --basemap-alpha 1.0 \
       --panel-title-mode column \
       --hotspot-csv fig6_hotspot_points.csv \
       --hotspot-color black \
       --coords-mode auto \
       --utm-epsg 32649 \
       --out spatial_satellite_cumulative

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--start-year`` / ``--year-val`` / ``--years-forecast``
   Control the baseline year, validation-year panels, and forecast-year
   panels. 

``--subsidence-kind``
   Choose whether the input CSV values are already cumulative or must be
   accumulated from yearly increments. 

``--coords-mode`` and ``--utm-epsg``
   Control CRS inference or override it explicitly.
   

``--render-mode`` / ``--surface-levels`` / ``--surface-alpha``
   Control whether the map panels use point rendering or a smoother
   surface-style overlay. 

``--hotspot-csv`` and related hotspot options
   Overlay hotspot points on the forecast panels. 

``--panel-title-mode``
   Choose whether panel titles are hidden, column-only, or shown on
   every panel. 

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_geo_cumulative`

From here
---------

A good next reading path is:

- :doc:`shared_conventions`
- :doc:`../auto_examples/figure_generation/index`
- :doc:`../auto_examples/forecasting/index`
- :doc:`../auto_examples/uncertainty/index`
- :doc:`../auto_examples/diagnostics/index`

In practice, the fastest workflow is often:

1. find the plotting command on this page,
2. inspect its example page in the gallery,
3. then adapt the command line to your own results, payloads, or
   exported tables.

