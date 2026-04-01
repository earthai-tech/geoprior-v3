Tables and summaries
====================

This gallery focuses on **artifact-building workflows** in GeoPrior.

Unlike the examples in ``figure_generation/``, the pages collected here
usually do not begin from a final paper figure. They begin from forecast
CSVs, experiment records, Stage-1 artifacts, hotspot point clouds,
processed city tables, and compact synthetic spatial supports, then show
how to turn those materials into clean, reusable outputs.

The emphasis is therefore on **builders**, not on presentation. These
lessons explain how GeoPrior produces the tables, merged datasets,
selection subsets, and lightweight spatial layers that later diagnostics,
applications, maps, and publication figures depend on.

What this gallery produces
--------------------------

Typical outputs in this section include:

- run-level metric tables,
- grouped ablation summaries,
- exceedance-scoring tables,
- extended or merged forecast artifacts,
- forecast-ready demo panels,
- manifest-driven NPZ bundles,
- spatial samples and non-overlapping batches,
- rectangular ROI extracts and threshold-based zones,
- cluster-labeled support tables,
- nearest-city borehole assignments,
- coordinate-enriched datasets with ``z_surf`` and optional head,
- boundary, grid, exposure, and zone-aware support layers.

In other words, this gallery is about building the intermediate
artifacts that make later reporting, debugging, mapping, and
interpretation possible.

How these lessons are structured
--------------------------------

Most pages in this section follow the same general pattern:

#. create or load a compact synthetic input,
#. call the real GeoPrior builder,
#. inspect the generated artifact,
#. add a small preview plot only when it helps explain the result,
#. finish with the matching command-line usage.

Even when a page contains a map, chart, or spatial preview, the main
teaching goal remains the same: to explain the **table or reusable
artifact** produced by the command.

For the newer spatial build lessons, the synthetic inputs often come
from the reusable spatial-support utilities rather than from ad hoc
random coordinates. That keeps the examples compact while still showing
realistic spatial footprints and stable command behavior.

Module guide
------------

.. list-table::
   :header-rows: 1
   :widths: 28 24 48

   * - Module
     - Main output
     - Purpose
   * - ``build_ablation_table.py``
     - CSV / JSON / TXT / TeX summary tables
     - Build tidy ablation archives from sensitivity records,
       including optional grouped summaries and best-per-city views.
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
   * - ``build_forecast_ready_sample.py``
     - Compact forecast-ready panel sample
     - Sample spatial groups rather than rows so the retained subset
       still preserves enough temporal structure for forecasting demos,
       tests, and tutorials.
   * - ``build_full_inputs_npz.py``
     - Merged ``full_inputs.npz`` archive
     - Merge split Stage-1 input NPZ artifacts through the manifest into
       one reusable bundle.
   * - ``compute_hotspots.py``
     - Hotspot summary products
     - Build per-city, per-year hotspot summaries relative to a chosen
       baseline year.
   * - ``summarize_hotspots.py``
     - Grouped hotspot tables
     - Summarize hotspot point clouds into compact city/year/kind
       tables for inspection and reuse.
   * - ``build_spatial_sampling.py``
     - Stratified spatial sample table
     - Build one representative sample from one or many input tables by
       combining spatial bins with optional extra stratification
       columns.
   * - ``build_batch_spatial_sampling.py``
     - Non-overlapping sampled batches
     - Build repeated spatial samples for benchmarking, demos, or batch
       experiments without reusing the same rows across batches.
   * - ``build_spatial_clusters.py``
     - Cluster-labeled spatial table
     - Attach spatial cluster labels to support points using methods
       such as k-means, DBSCAN, or agglomerative clustering.
   * - ``build_spatial_roi.py``
     - Region-of-interest subset table
     - Extract a rectangular spatial window from one or many tabular
       inputs, with optional snapping to the nearest available support
       coordinates.
   * - ``build_extract_zones.py``
     - Threshold-based zone table
     - Extract high, low, or bounded-response zones from a spatial table
       using percentile or explicit threshold rules.
   * - ``build_assign_boreholes.py``
     - Classified borehole tables
     - Assign boreholes to the nearest city support cloud and export one
       combined table plus optional per-city splits.
   * - ``build_add_zsurf_from_coords.py``
     - ``z_surf``-enriched harmonized datasets
     - Merge rounded coordinate-elevation lookups into city datasets and
       optionally compute hydraulic head.
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

#. start from experiment logs, evaluation records, Stage-1 artifacts,
   or forecast exports,
#. turn them into tidy summary tables or merged reusable bundles,
#. derive compact forecast-ready subsets or extended forecast products,
#. build spatial samples, batches, ROI selections, zones, or clusters
   for downstream analysis,
#. enrich those products with support metadata such as borehole
   assignment or surface elevation,
#. continue into diagnostics, hotspot analysis, applications, or figure
   generation.

That is why the examples are naturally grouped into four broad themes:
metric builders, forecast-derived products, spatial sampling and
selection tools, and spatial support/enrichment layers.

Gallery organization
--------------------

Metrics and experiment summaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These examples start from evaluation records, ablation logs, or
sensitivity runs and convert them into compact tabular outputs that are
much easier to compare, archive, filter, and report.

The main pages in this group are:

- ``build_ablation_table.py``
- ``build_model_metrics.py``
- ``compute_brier_exceedance.py``

Forecast-derived products and merged inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These examples start from forecast products or Stage-1 split artifacts
and build secondary outputs that are easier to inspect, reuse, or carry
into later workflow stages.

The main pages in this group are:

- ``extend_forecast.py``
- ``build_forecast_ready_sample.py``
- ``build_full_inputs_npz.py``
- ``compute_hotspots.py``
- ``summarize_hotspots.py``

Spatial sampling and selection tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These examples focus on choosing *which* part of a large spatial table
should be kept, grouped, or highlighted before later modelling,
inspection, or mapping.

The main pages in this group are:

- ``build_spatial_sampling.py``
- ``build_batch_spatial_sampling.py``
- ``build_spatial_clusters.py``
- ``build_spatial_roi.py``
- ``build_extract_zones.py``

Spatial support and enrichment layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These examples build or enrich lightweight spatial artifacts that help
later analysis stay geographically interpretable.

The main pages in this group are:

- ``build_assign_boreholes.py``
- ``build_add_zsurf_from_coords.py``
- ``make_boundary.py``
- ``make_district_grid.py``
- ``make_exposure.py``
- ``tag_clusters_with_zones.py``

Why this separation matters
---------------------------

This gallery deliberately keeps four concerns distinct:

- **data production and merging**,
- **tabulation and forecast-derived summaries**,
- **spatial selection and reduction**,
- **support-layer enrichment and interpretation**.

That separation makes the workflow easier to reason about. It also
helps users understand which commands generate reusable data products,
which commands reduce large tables into compact subsets, which commands
add geographic meaning, and which later pages merely visualize the
results.

Notes
-----

- These examples are intentionally compact and lesson-oriented.
- Many pages include a small preview figure, but that figure is only
  there to make the artifact easier to understand.
- The main output of this gallery is usually a **CSV, JSON, TXT, TeX,
  NPZ, or lightweight spatial layer**, not a final publication figure.
- Several lessons end with both ``geoprior-build ...`` and
  ``geoprior build ...`` forms so the reader can move easily between the
  family-specific entrypoint and the root dispatcher.
