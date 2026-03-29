Tables and summaries
====================

This gallery focuses on **artifact-building workflows** in GeoPrior.

Unlike the examples in ``figure_generation/``, the pages collected here
do not begin from a paper panel or a final figure. They begin from
forecast CSVs, experiment records, hotspot point clouds, and lightweight
geospatial support inputs, then show how to turn those materials into
clean, reusable analysis products.

The emphasis is therefore on **builders**, not on presentation. These
examples explain how GeoPrior produces the tables and support layers
that later diagnostics, reports, maps, and publication figures depend
on.

What this gallery produces
--------------------------

Typical outputs in this section include:

- run-level metric tables,
- grouped ablation summaries,
- exceedance-scoring tables,
- hotspot summary products,
- extended forecast exports,
- boundary, grid, and exposure layers,
- cluster artifacts tagged with zone IDs.

In other words, this gallery is about building the intermediate
artifacts that make later reporting and interpretation possible.

How these lessons are structured
--------------------------------

Most pages in this section follow the same general pattern:

#. create or load a compact synthetic input,
#. call the real GeoPrior builder,
#. inspect the generated artifact,
#. add a small preview plot only when it helps explain the result.

Even when a page contains a map, chart, or visual check, the main goal
remains the same: to explain the **table or reusable artifact** produced
by the command.

Module guide
------------

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Module
     - Main output
     - Purpose
   * - ``build_ablation_table.py``
     - CSV / TeX summary tables
     - Build tidy ablation archives from sensitivity records, including
       optional grouped summaries and best-per-city views.
   * - ``build_model_metrics.py``
     - Unified metrics tables
     - Build wide run-level and long horizon-level metrics tables from
       experiment outputs.
   * - ``compute_brier_exceedance.py``
     - Exceedance-scoring tables
     - Score exceedance events from calibrated forecast quantiles and
       export tidy Brier-score tables by threshold and year.
   * - ``extend_forecast.py``
     - Extended forecast CSVs
     - Extend future forecast products to later years using simple
       extrapolation rules.
   * - ``compute_hotspots.py``
     - Hotspot summary products
     - Build per-city, per-year hotspot summaries relative to a chosen
       baseline year.
   * - ``summarize_hotspots.py``
     - Grouped hotspot tables
     - Summarize hotspot point clouds into compact city/year/kind
       tables for inspection and reuse.
   * - ``make_boundary.py``
     - Boundary polygons
     - Build simple city boundary layers from forecast point clouds.
   * - ``make_district_grid.py``
     - Grid / zone layers
     - Build district-style grids over the forecast support domain,
       optionally clipped to a boundary and linked to samples.
   * - ``make_exposure.py``
     - Exposure layer
     - Build compact exposure products from sample geometry using
       uniform or density-based weighting schemes.
   * - ``tag_clusters_with_zones.py``
     - Zone-tagged cluster artifacts
     - Attach zone identifiers to cluster outputs so they can be
       interpreted and summarized in a zone-aware way.

Reading path
------------

A useful way to move through this gallery is to follow the logic of a
real workflow:

#. start from experiment logs, evaluation records, or forecast exports,
#. turn them into tidy summary tables,
#. derive secondary forecast products when needed,
#. build spatial support layers for later analytics and mapping,
#. continue into diagnostics, hotspot analysis, or figure generation.

That is why the examples are naturally grouped into three broad themes:
metric builders, forecast-derived products, and spatial support layers.

Gallery organization
--------------------

Metrics and ablations
~~~~~~~~~~~~~~~~~~~~~

These examples start from evaluation records, ablation logs, or
sensitivity runs and convert them into compact tabular outputs that are
easier to compare, archive, and report.

The main pages in this group are:

- ``build_ablation_table.py``
- ``build_model_metrics.py``
- ``compute_brier_exceedance.py``

Forecast-derived products
~~~~~~~~~~~~~~~~~~~~~~~~~

These examples start from forecast or hotspot artifacts and build
secondary products that are easier to inspect, compare, or reuse in
later stages of analysis.

The main pages in this group are:

- ``extend_forecast.py``
- ``compute_hotspots.py``
- ``summarize_hotspots.py``

Spatial support layers
~~~~~~~~~~~~~~~~~~~~~~

These examples build lightweight geospatial artifacts that support later
clustering, analytics, and mapping workflows.

The main pages in this group are:

- ``make_boundary.py``
- ``make_district_grid.py``
- ``make_exposure.py``
- ``tag_clusters_with_zones.py``

Why this separation matters
---------------------------

This gallery deliberately keeps three concerns distinct:

- **data production**,
- **tabulation and support-layer building**,
- **visualization**.

That separation makes the workflow easier to reason about. It also helps
users understand which commands generate reusable data products, which
commands summarize them, and which later pages merely visualize the
results.

Notes
-----

- These examples are intentionally compact and lesson-oriented.
- Many pages include a small preview figure, but that figure is only
  there to make the artifact easier to understand.
- The main output of this gallery is usually a **CSV, JSON, TeX, or
  lightweight geospatial layer**, not a final publication figure.