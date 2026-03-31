.. _cli-build-family:

Build family
============

The **build family** is where you materialize reusable artifacts.

Use this page when your goal is not to run one of the main staged
workflows directly, and not to render a figure, but to **create a data
product** that can be reused later.

In GeoPrior, build commands are used to create things such as:

- merged NPZ payloads,
- compact forecast-ready samples,
- spatially filtered or labeled tables,
- external validation artifacts,
- hotspot and exposure tables,
- ablation and metrics summaries,
- small geospatial side products.

You can invoke these commands either from the root dispatcher:

.. code-block:: bash

   geoprior build <command> [args]

or from the family-specific entry point:

.. code-block:: bash

   geoprior-build <command> [args]

GeoPrior also supports ``make`` as an alias of ``build`` at the root
dispatcher level. 

This page is intentionally structured as a **guide first** and a
**reference scaffold second**:

- the table below helps users find the right build command quickly,
- the grouped sections afterwards explain how the family is organized,
- detailed command subsections can be appended gradually as the build
  modules are reviewed in batches.

How to choose a build command
-----------------------------

A practical way to choose a build command is to first decide **what kind
of artifact** you want:

- if you need a compact or derived dataset, start with the **sampling
  and spatial preparation** commands,
- if you need a model-facing NPZ or validation artifact, start with the
  **payload and validation** commands,
- if you need a summary table or downstream analysis product, start with
  the **tables and summaries** commands,
- if you need a lightweight derived boundary, grid, or zone-linked
  output, start with the **geospatial side products** commands.

Many build commands also reuse the shared tabular input layer, so they
can read one or many files, infer or force the input format, support
Excel ``PATH::SHEET`` syntax, and write common tabular outputs in CSV,
TSV, Parquet, Excel, JSON, Feather, or Pickle formats. That shared
reader/writer behavior is centralized in ``geoprior.cli.utils`` rather
than redefined in each command. 

Build commands at a glance
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 44 24 12

   * - Command
     - Use it when
     - Main outcome
     - Related guide
   * - ``forecast-ready-sample``
     - You want a compact panel sample prepared for forecasting-style
       workflows or demos.
     - Forecast-ready sample table.
     - :doc:`../user_guide/stage1`
   * - ``spatial-sampling``
     - You want a stratified spatial sample from one or more input
       tables.
     - Spatial sample table.
     - :doc:`shared_conventions`
   * - ``batch-spatial-sampling``
     - You want several non-overlapping spatial sampling batches rather
       than one sample.
     - Batched spatial sample outputs.
     - :doc:`shared_conventions`
   * - ``spatial-roi``
     - You want a region-of-interest subset table.
     - ROI-filtered table.
     - :doc:`shared_conventions`
   * - ``spatial-clusters``
     - You want spatial cluster labels added to a table.
     - Cluster-labeled table.
     - :doc:`../user_guide/diagnostics`
   * - ``extract-zones``
     - You want threshold-based zone extraction.
     - Zone extraction table.
     - :doc:`../user_guide/diagnostics`
   * - ``assign-boreholes``
     - You want nearest-city or city-assigned borehole tables.
     - Borehole assignment table.
     - :doc:`../user_guide/inference_and_export`
   * - ``add-zsurf-from-coords``
     - You want to enrich a dataset with ``z_surf`` derived from
       coordinates.
     - ``z_surf``-enriched dataset.
     - :doc:`../scientific_foundations/data_and_units`
   * - ``full-inputs-npz``
     - You want one merged ``full_inputs.npz`` assembled from split
       artifacts.
     - Full merged model-input NPZ.
     - :doc:`../user_guide/stage1`
   * - ``physics-payload-npz``
     - You want a physics payload NPZ for downstream physics-oriented
       workflows.
     - Physics payload NPZ.
     - :doc:`../scientific_foundations/physics_formulation`
   * - ``external-validation-fullcity``
     - You want full-city validation artifacts prepared for external
       validation workflows.
     - Full-city validation artifact set.
     - :doc:`../user_guide/inference_and_export`
   * - ``external-validation-metrics``
     - You want summarized external validation metrics.
     - External validation metrics table.
     - :doc:`../user_guide/inference_and_export`
   * - ``sm3-collect-summaries``
     - You want one combined SM3 summary table from multiple SM3 runs.
     - Combined SM3 summary table.
     - :doc:`run_family`
   * - ``brier-exceedance``
     - You want exceedance-oriented Brier results computed into a table.
     - Exceedance Brier table.
     - :doc:`../user_guide/inference_and_export`
   * - ``hotspots``
     - You want hotspot outputs computed from forecast or risk-style
       data.
     - Hotspot outputs.
     - :doc:`../user_guide/diagnostics`
   * - ``hotspots-summary``
     - You want an already computed hotspot result condensed into a
       summary table.
     - Hotspot summary table.
     - :doc:`../user_guide/diagnostics`
   * - ``model-metrics``
     - You want unified model metrics tables in tabular form.
     - Metrics tables in CSV/JSON style outputs.
     - :doc:`../user_guide/diagnostics`
   * - ``ablation-table``
     - You want a compact ablation table built from ablation records.
     - Ablation summary table.
     - :doc:`../user_guide/diagnostics`
   * - ``update-ablation-records``
     - You want to patch or enrich ablation record JSONL with metrics.
     - Updated ablation records.
     - :doc:`../user_guide/diagnostics`
   * - ``extend-forecast``
     - You want to extend a future forecast CSV by extrapolation.
     - Extended future forecast CSV.
     - :doc:`../user_guide/stage4`
   * - ``boundary``
     - You want to derive a boundary polygon from points.
     - Boundary geospatial output.
     - :doc:`../user_guide/inference_and_export`
   * - ``exposure``
     - You want an exposure proxy table derived from points.
     - Exposure CSV or equivalent table.
     - :doc:`../applications/subsidence_forecasting`
   * - ``district-grid``
     - You want a grid-based district layer.
     - District grid output.
     - :doc:`../user_guide/inference_and_export`
   * - ``clusters-with-zones``
     - You want hotspot clusters assigned to zone IDs.
     - Zone-tagged cluster table.
     - :doc:`../user_guide/diagnostics`


Shared patterns across many build commands
------------------------------------------

Although the build family covers many different artifacts, a large part
of the user experience is intentionally consistent.

Many build commands reuse the shared CLI data-loading layer. That gives
the family a familiar feel across commands:

- one or many input files can be accepted,
- simple glob expansion is supported,
- tabular formats can be inferred from file extensions or forced
  explicitly,
- Excel files can use ``PATH::SHEET`` syntax,
- loaded tables can be concatenated into one DataFrame,
- output formats can be inferred from the destination extension and
  written consistently. 

Build commands also benefit from shared configuration and path helpers
such as optional config installation, repeated ``--set KEY=VALUE``
overrides, reusable output-directory arguments, and consistent creation
of destination folders. 


One build family, two implementation paths
------------------------------------------

The build family is broader than one implementation style.

Some commands are implemented as dedicated wrappers under
``geoprior.cli``. Others are script-backed build commands exposed
through the public registry under ``geoprior.scripts.registry``. From a
user point of view, they still belong to one family, because the whole
purpose of the dispatcher is to make them behave like one coherent
public CLI. 

That unified design also preserves a helpful compatibility path for
older reproducibility workflows. Script-backed commands can still be run
through the legacy interface:

.. code-block:: bash

   python -m scripts <command> [args]

while the modern public interface remains:

.. code-block:: bash

   geoprior build <command> [args]
   geoprior-build <command> [args]

This means the documentation can teach one public build family without
hiding the fact that some commands also support legacy entry points.


How the build family is organized
---------------------------------

To make the build family easier to navigate, this guide groups the
commands into three practical sections.

The first group focuses on **sampling and spatial preparation**. These
commands help turn raw or harmonized tables into compact, filtered,
clustered, or region-specific datasets.

The second group focuses on **payloads and validation artifacts**. These
commands build model-facing or validation-facing artifacts that later
workflows can consume directly.

The third group focuses on **tables, summaries, and derived products**.
These commands produce compact downstream artifacts for comparison,
reporting, extrapolation, and lightweight geospatial analysis.

That structure is meant to help users think in terms of **what they are
trying to build**, not just in terms of command names.


Sampling and spatial preparation
----------------------------------

These commands help you transform raw or harmonized tables into compact,
filtered, clustered, or region-specific datasets.


.. _build-forecast-ready-sample:

``forecast-ready-sample``
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``forecast-ready-sample`` when you want to build a **compact
forecast-ready panel** from one or many tabular inputs.

This command is meant for cases where you want a smaller, structured
dataset that still respects the forecasting window logic. The wrapper
describes it as a command that builds a compact forecast-ready panel
sample, and it passes through options that control group sampling,
window length, forecast horizon, year retention, and whether groups must
contain a consecutive run of the required length. 

This is a good fit when you want to:

- create a lighter-weight dataset for demos or examples,
- test a forecasting workflow without using the full panel,
- preserve valid forecasting windows while reducing data volume,
- prepare a compact table for gallery lessons or downstream scripts.

Usage
^^^^^

Build a simple forecast-ready sample:

.. code-block:: bash

   geoprior build forecast-ready-sample data.csv -o forecast_sample.csv

or:

.. code-block:: bash

   geoprior-build forecast-ready-sample data.parquet -o forecast_sample.parquet

Use multiple inputs with a larger lookback window:

.. code-block:: bash

   geoprior-build forecast-ready-sample data/*.csv \
       -o forecast_sample.csv \
       --time-steps 5 \
       --forecast-horizon 2

Control the sampled group fraction and spatial stratification:

.. code-block:: bash

   geoprior-build forecast-ready-sample data.csv \
       -o forecast_sample.csv \
       --sample-size 0.10 \
       --spatial-bins 10 12 \
       --stratify-by city lithology_class

Keep only the latest years per sampled group:

.. code-block:: bash

   geoprior-build forecast-ready-sample data.csv \
       -o forecast_sample.csv \
       --keep-years 5 \
       --year-mode latest

Retain a fixed number of groups and a subset of columns:

.. code-block:: bash

   geoprior-build forecast-ready-sample data.csv \
       -o forecast_sample.csv \
       --sample-size 500 \
       --max-groups 500 \
       --columns-to-keep longitude latitude year subsidence city

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--out``
   Output table path. Its extension controls the written format.

``--time-steps`` and ``--forecast-horizon``
   Define the lookback window and forecast horizon used to decide which
   sampled groups remain valid. 

``--sample-size``
   Sample size at the group level. A float means a fraction; an integer
   means an absolute group count. 

``--keep-years`` and ``--year-mode``
   Control how many years are retained per sampled group and whether
   they are chosen as the latest, earliest, random, or all available
   years. 

``--require-consecutive``
   Require each sampled group to contain a consecutive run of length
   ``time_steps + forecast_horizon``. 

``--columns-to-keep``
   Trim the final output to a selected subset of columns.


Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/index`

.. _build-spatial-sampling:

``spatial-sampling``
~~~~~~~~~~~~~~~~~~~~

Use ``spatial-sampling`` when you want to build **one stratified spatial
sample table** from one combined input dataset.

The wrapper reads one or many input files into one DataFrame, then calls
``spatial_sampling`` from ``geoprior.utils.spatial_utils``. It supports
sampling by absolute count or fraction, optional stratification columns,
explicit spatial binning, and automatic or explicit spatial coordinate
selection.

This command is a good fit when you want one sampled table that still
tries to respect spatial structure rather than a purely random subset.

Usage
^^^^^

Build one sampled table from a single input:

.. code-block:: bash

   geoprior build spatial-sampling data.csv -o sampled.csv

or from several inputs:

.. code-block:: bash

   geoprior-build spatial-sampling data/*.csv -o sampled.parquet

Sample by fraction:

.. code-block:: bash

   geoprior-build spatial-sampling data.csv \
       -o sampled.csv \
       --sample-size 0.05

Sample by absolute count:

.. code-block:: bash

   geoprior-build spatial-sampling data.csv \
       -o sampled.csv \
       --sample-size 5000

Add stratification and explicit spatial bins:

.. code-block:: bash

   geoprior-build spatial-sampling data.csv \
       -o sampled.csv \
       --stratify-by city year \
       --spatial-bins 12 12 \
       --spatial-cols longitude latitude

Use relative sampling with a minimum ratio:

.. code-block:: bash

   geoprior-build spatial-sampling data.csv \
       -o sampled.csv \
       --method relative \
       --min-relative-ratio 0.02

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--sample-size``
   Accepts either a positive integer or a fraction in ``(0, 1)``.

``--stratify-by``
   Add non-spatial stratification columns on top of spatial sampling.

``--spatial-bins``
   Accept one integer or one value per spatial column. Internally, the
   wrapper normalizes this to either a scalar or a tuple. 

``--spatial-cols``
   Override the coordinate columns used for sampling.

``--method`` and ``--min-relative-ratio``
   Control whether the sampler behaves in absolute or relative mode.


Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/index`

.. _build-batch-spatial-sampling:

``batch-spatial-sampling``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``batch-spatial-sampling`` when you want to build **many
non-overlapping spatial sample batches** rather than one sampled table.

The wrapper reads one combined input table, calls
``batch_spatial_sampling``, and then writes a stacked output with a
batch identifier column. It can also optionally write one file per batch
into a separate directory.

This command is useful when you want repeated sampled subsets for
comparisons, ablations, batch-style downstream processing, or repeated
small-batch experiments.

Usage
^^^^^

Build a stacked batch table:

.. code-block:: bash

   geoprior build batch-spatial-sampling data.csv -o batches.csv

Create more batches with a larger sample size:

.. code-block:: bash

   geoprior-build batch-spatial-sampling data.csv \
       -o batches.csv \
       --sample-size 0.2 \
       --n-batches 20

Control stratification and spatial bins:

.. code-block:: bash

   geoprior-build batch-spatial-sampling data.csv \
       -o batches.parquet \
       --stratify-by city year \
       --spatial-bins 10 12 \
       --spatial-cols longitude latitude

Also write one file per batch:

.. code-block:: bash

   geoprior-build batch-spatial-sampling data.csv \
       -o batches.csv \
       --split-dir sampled_batches \
       --split-prefix batch_ \
       --split-format csv

Change the batch identifier column in the stacked output:

.. code-block:: bash

   geoprior-build batch-spatial-sampling data.csv \
       -o batches.csv \
       --batch-col batch_id

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--n-batches``
   Number of non-overlapping sampled batches to create.

``--batch-col``
   Column name inserted into the stacked output to mark each batch.

``--split-dir``
   Optional directory where one separate file per batch is also written.

``--split-prefix`` and ``--split-format``
   Control the naming and format of the per-batch files.


See also 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/index`

.. _build-spatial-roi:

``spatial-roi``
~~~~~~~~~~~~~~~

Use ``spatial-roi`` when you want to extract a **rectangular region of
interest** from one combined spatial table.

The wrapper merges one or many inputs into one DataFrame, then applies
``extract_spatial_roi``. It requires an ``x`` range and a ``y`` range,
and it can either use the exact requested bounds or snap them to the
nearest available coordinates.

This is a simple and useful command when you want to cut out one study
window, map tile, or local subregion before downstream analysis.

Usage
^^^^^

Extract one ROI from a single input:

.. code-block:: bash

   geoprior build spatial-roi data.csv \
       --x-range 113.4 113.8 \
       --y-range 22.6 22.9 \
       -o roi.csv

Use explicit coordinate column names:

.. code-block:: bash

   geoprior-build spatial-roi data.csv \
       --x-range 1090 1720 \
       --y-range 835 1220 \
       --x-col x_m \
       --y-col y_m \
       -o roi.parquet

Use exact bounds without snapping:

.. code-block:: bash

   geoprior-build spatial-roi data.csv \
       --x-range 113.4 113.8 \
       --y-range 22.6 22.9 \
       --no-snap-to-closest \
       -o roi.csv

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--x-range`` and ``--y-range``
   Required lower and upper bounds for the rectangular ROI.

``--x-col`` and ``--y-col``
   Coordinate column names used for filtering. 

``--no-snap-to-closest``
   Disable snapping to the nearest available coordinates and instead use
   the exact bounds supplied on the command line. 

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/index`


.. _build-spatial-clusters:

``spatial-clusters``
~~~~~~~~~~~~~~~~~~~~

Use ``spatial-clusters`` when you want to add **spatial cluster labels**
to one combined input table.

The wrapper reads one or many tabular files, merges them into one
DataFrame, and applies ``create_spatial_clusters``. It supports several
clustering backends, optional standard scaling, configurable output
column naming, and an optional diagnostic cluster plot.

This is a good fit when you want region labels, cluster-based grouping,
or spatial partitions that can later be summarized, plotted, or joined
to other analysis outputs.

Usage
^^^^^

Create cluster labels with default settings:

.. code-block:: bash

   geoprior build spatial-clusters data.csv -o clustered.csv

Choose explicit spatial columns and output label name:

.. code-block:: bash

   geoprior-build spatial-clusters data.csv \
       --spatial-cols longitude latitude \
       --cluster-col region \
       -o clustered.csv

Fix the number of clusters and backend:

.. code-block:: bash

   geoprior-build spatial-clusters data.csv \
       --n-clusters 8 \
       --algorithm kmeans \
       -o clustered.csv

Use a different clustering backend:

.. code-block:: bash

   geoprior-build spatial-clusters data.csv \
       --algorithm dbscan \
       -o clustered.csv

Display the diagnostic cluster plot:

.. code-block:: bash

   geoprior-build spatial-clusters data.csv \
       --view \
       --figsize 14 10 \
       --marker-size 60 \
       --cmap tab20 \
       -o clustered.csv

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--spatial-cols``
   The two coordinate columns used for clustering. 

``--cluster-col``
   Name of the output column that stores the assigned cluster labels.

``--n-clusters``
   Optional number of clusters. For ``kmeans``, the helper may
   auto-detect this when omitted. 

``--algorithm``
   Choose among ``kmeans``, ``dbscan``, or ``agglo``.

``--no-auto-scale``
   Disable standard scaling before clustering. 

``--view``
   Display a diagnostic cluster plot using the plotting options exposed
   by the wrapper. 

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/index`

.. _build-extract-zones:

``extract-zones``
~~~~~~~~~~~~~~~~~

Use ``extract-zones`` when you want to extract rows that satisfy a
**threshold-based zone criterion**.

The wrapper reads one or many tables, merges them, and applies
``extract_zones_from``. The threshold can be ``auto``, one numeric
value, or two numeric values for a between-range filter, and the command
can optionally display a diagnostic plot of the extracted zone. 

This command is useful when you want to isolate risk zones, anomaly
zones, hotspot candidates, or other threshold-defined subsets from a
larger table.

Usage
^^^^^

Extract rows above or below an automatically chosen threshold:

.. code-block:: bash

   geoprior build extract-zones data.csv \
       --z-col subsidence \
       --threshold auto \
       -o zones.csv

Use a single numeric threshold:

.. code-block:: bash

   geoprior-build extract-zones data.csv \
       --z-col subsidence \
       --threshold 10 \
       --condition above \
       -o zones.csv

Use a between-range filter:

.. code-block:: bash

   geoprior-build extract-zones data.csv \
       --z-col subsidence \
       --threshold 5 15 \
       --condition between \
       -o zones.csv

Use percentile-driven auto thresholding with positive criteria:

.. code-block:: bash

   geoprior-build extract-zones data.csv \
       --z-col subsidence \
       --threshold auto \
       --percentile 90 \
       --positive-criteria \
       -o zones.csv

Display the diagnostic plot with spatial coordinates:

.. code-block:: bash

   geoprior-build extract-zones data.csv \
       --z-col subsidence \
       --threshold auto \
       --x-col longitude \
       --y-col latitude \
       --view \
       --plot-type scatter \
       -o zones.csv

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--z-col``
   Required column used to define the zone criterion.

``--threshold``
   Accept ``auto``, one float, or two floats. The wrapper parses this
   into either an automatic threshold, a single threshold, or a
   between-range tuple. 

``--condition``
   Choose among ``auto``, ``above``, ``below``, and ``between``.

``--percentile`` and ``--positive-criteria``
   Control how the automatic threshold behaves. 

``--x-col`` / ``--y-col`` / ``--view``
   Enable a diagnostic spatial view of the extracted zone.


Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/index`

.. _build-assign-boreholes:

``assign-boreholes``
~~~~~~~~~~~~~~~~~~~~

Use ``assign-boreholes`` when you want to classify each borehole row to
the **nearest city point cloud**.

This command is more structured than the earlier table-only wrappers. It
can resolve city clouds from several sources:

1. explicit processed CSVs via ``--city-csv CITY=PATH``,
2. explicit Stage-1 directories via ``--city-stage1 CITY=DIR``,
3. repeated ``--stage1-dir DIR`` arguments with automatic city-name
   inference,
4. repeated city names via ``--cities`` together with
   ``--results-dir`` and optionally ``--model``.

Its outputs include one combined classified CSV and, unless disabled,
optional per-city split CSVs. 

Usage
^^^^^

Classify boreholes using explicit city processed CSVs:

.. code-block:: bash

   geoprior build assign-boreholes \
       --borehole-csv boreholes.csv \
       --city-csv nansha=results/nansha_proc.csv \
       --city-csv zhongshan=results/zhongshan_proc.csv

Use explicit Stage-1 directories instead:

.. code-block:: bash

   geoprior-build assign-boreholes \
       --borehole-csv boreholes.csv \
       --city-stage1 nansha=results/nansha_GeoPriorSubsNet_stage1 \
       --city-stage1 zhongshan=results/zhongshan_GeoPriorSubsNet_stage1

Resolve cities from a results layout:

.. code-block:: bash

   geoprior-build assign-boreholes \
       --cities nansha zhongshan \
       --results-dir results \
       --model GeoPriorSubsNet

Control output layout explicitly:

.. code-block:: bash

   geoprior-build assign-boreholes \
       --borehole-csv boreholes.csv \
       --city-csv nansha=results/nansha_proc.csv \
       --city-csv zhongshan=results/zhongshan_proc.csv \
       --outdir borehole_assignment \
       --output-stem boreholes

Skip the per-city split files:

.. code-block:: bash

   geoprior-build assign-boreholes \
       --borehole-csv boreholes.csv \
       --city-csv nansha=results/nansha_proc.csv \
       --city-csv zhongshan=results/zhongshan_proc.csv \
       --no-split-files

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--borehole-csv``
   Validation borehole CSV containing point coordinates. When omitted,
   the command can also fall back to configured validation paths.

``--city-csv`` / ``--city-stage1`` / ``--stage1-dir`` / ``--cities``
   Multiple ways to resolve the city clouds used for nearest-city
   classification. 

``--borehole-x-col`` / ``--borehole-y-col`` and
``--city-x-col`` / ``--city-y-col``
   Separate coordinate-column controls for the borehole table and the
   city processed tables. 

``--tie-label`` and ``--tie-tol``
   Control how ties in nearest-city distance are labeled.

``--output-stem`` / ``--classified-out`` / ``--no-split-files``
   Control whether the command writes only one combined CSV or also one
   file per assigned city. 

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/index`

.. _build-add-zsurf-from-coords:

``add-zsurf-from-coords``
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``add-zsurf-from-coords`` when you want to enrich one or more main
datasets with **surface elevation** derived from coordinate-based lookup
tables.

This command is more than a plain join. It merges a main tabular dataset
with a coordinate-to-elevation lookup on rounded longitude/latitude
pairs and can optionally compute hydraulic head when a compatible
depth-below-ground-surface column is available. 

It supports several ways to resolve inputs:

- repeated city names with ``--city``,
- explicit ``CITY=PATH`` mappings for main datasets and elevation
  lookups,
- root directories plus filename patterns for main and elevation CSVs.

This command is a good fit when you want to harmonize datasets before
physics-aware analysis or to derive ``z_surf`` and optionally ``head_m``
in a reproducible way.

Usage
^^^^^

Process one city using explicit input files:

.. code-block:: bash

   geoprior build add-zsurf-from-coords \
       --city nansha \
       --main-csv nansha=data/nansha_final_main_std.harmonized.csv \
       --elev-csv nansha=data/nansha_coords_with_elevation.csv

Process several cities using repeated mappings:

.. code-block:: bash

   geoprior-build add-zsurf-from-coords \
       --city nansha \
       --city zhongshan \
       --main-csv nansha=data/nansha_main.csv \
       --main-csv zhongshan=data/zhongshan_main.csv \
       --elev-csv nansha=data/nansha_elev.csv \
       --elev-csv zhongshan=data/zhongshan_elev.csv

Resolve inputs from directory roots and filename patterns:

.. code-block:: bash

   geoprior-build add-zsurf-from-coords \
       --city nansha \
       --data-root data/main \
       --coords-root data/elevation \
       --main-pattern "{city}_final_main_std.harmonized.csv" \
       --elev-pattern "{city}_coords_with_elevation.csv"

Write outputs to a dedicated folder and emit diagnostics JSON:

.. code-block:: bash

   geoprior-build add-zsurf-from-coords \
       --city nansha \
       --main-csv nansha=data/nansha_main.csv \
       --elev-csv nansha=data/nansha_elev.csv \
       --outdir enriched \
       --summary-json enriched/zsurf_summary.json

Control coordinate rounding and duplicate reduction:

.. code-block:: bash

   geoprior-build add-zsurf-from-coords \
       --city nansha \
       --main-csv nansha=data/nansha_main.csv \
       --elev-csv nansha=data/nansha_elev.csv \
       --round-decimals 6 \
       --reducer median

Disable hydraulic-head computation:

.. code-block:: bash

   geoprior-build add-zsurf-from-coords \
       --city nansha \
       --main-csv nansha=data/nansha_main.csv \
       --elev-csv nansha=data/nansha_elev.csv \
       --no-head

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--main-csv`` and ``--elev-csv``
   Repeated ``CITY=PATH`` mappings for the main dataset and the
   elevation lookup CSV. 

``--data-root`` / ``--coords-root`` plus ``--main-pattern`` /
``--elev-pattern``
   Let the command resolve city-specific files automatically from root
   directories and filename templates. 

``--zsurf-col`` and ``--head-col``
   Control the output column names for surface elevation and hydraulic
   head. 

``--depth-col``
   Provide one or more candidate depth-below-ground-surface column names
   used to compute hydraulic head. 

``--round-decimals`` and ``--reducer``
   Control coordinate rounding before merging and how duplicate
   coordinate-elevation rows are reduced. 

``--summary-json``
   Write merge diagnostics across all processed cities. 


Payloads and validation artifacts
----------------------------------

These commands build model-facing or validation-facing artifacts that
other workflows can consume later. 

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/index`

.. _build-full-inputs-npz:

``full-inputs-npz``
~~~~~~~~~~~~~~~~~~~

Use ``full-inputs-npz`` when you want to build one **merged
``full_inputs.npz``** from the Stage-1 split input artifacts.

This command is the simplest payload builder in the family. It resolves
a Stage-1 ``manifest.json``, reads the split input NPZ artifacts listed
there, concatenates them in split order, and writes one combined NPZ.
By default it looks for the standard ``train``, ``val``, and ``test``
input splits and writes the result under the Stage-1 ``artifacts/``
directory as ``full_inputs.npz``. 

This command is a good fit when you want one reusable full-city input
payload for downstream inference, physics export, or external
validation-style workflows. 

Usage
^^^^^

Build the default merged input payload from a Stage-1 manifest:

.. code-block:: bash

   geoprior build full-inputs-npz \
       --manifest results/nansha_GeoPriorSubsNet_stage1/manifest.json

or from the Stage-1 directory directly:

.. code-block:: bash

   geoprior-build full-inputs-npz \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1

Resolve the manifest from a results layout:

.. code-block:: bash

   geoprior-build full-inputs-npz \
       --results-dir results \
       --city nansha \
       --model GeoPriorSubsNet

Choose a subset of splits explicitly:

.. code-block:: bash

   geoprior-build full-inputs-npz \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --splits train val

Write to an explicit output path:

.. code-block:: bash

   geoprior-build full-inputs-npz \
       --manifest results/nansha_GeoPriorSubsNet_stage1/manifest.json \
       --output exports/nansha_full_inputs.npz

Relax strict key matching across splits:

.. code-block:: bash

   geoprior-build full-inputs-npz \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --allow-missing-keys

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--manifest`` / ``--stage1-dir`` / ``--results-dir`` with
``--city`` and ``--model``
   Multiple ways to resolve the Stage-1 manifest used to discover the
   split NPZ files. 

``--splits``
   Control which split input NPZ files are concatenated and in what
   order. The default is ``train val test``. 

``--output`` and ``--output-name``
   Either choose one explicit output path or let the command write under
   the Stage-1 ``artifacts/`` directory with a default name.

``--allow-missing-keys``
   Relax strict key alignment when the split NPZ files do not all expose
   exactly the same arrays. 


Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/index`

.. _build-physics-payload-npz:

``physics-payload-npz``
~~~~~~~~~~~~~~~~~~~~~~~

Use ``physics-payload-npz`` when you want to export a **physics payload
NPZ** from a trained GeoPrior model together with one resolved input
payload.

This command sits one step downstream from ``full-inputs-npz``. It
resolves a Stage-1 manifest, selects or assembles the inputs to use,
resolves a trained model, builds a TensorFlow dataset, and then calls
the model’s ``export_physics_payload`` method to write a saved NPZ.
When no explicit input NPZ is supplied, it can reuse an existing
``full_inputs.npz`` or assemble one from Stage-1 split artifacts.
The output metadata includes the source inputs, manifest path, model
path, and split label used for export. 

This command is a good fit when you want model-derived physics fields
such as ``K``, ``Ss``, ``Hd``, ``H``, and related payload metrics in a
compact artifact that later workflows can consume. Tests in the package
also confirm that exported payloads include physics arrays such as
``tau``, ``tau_prior``, ``K``, ``Ss``, ``Hd``, and ``metrics``.


Usage
^^^^^

Export a physics payload from a Stage-1 manifest and an explicit model:

.. code-block:: bash

   geoprior build physics-payload-npz \
       --manifest results/nansha_GeoPriorSubsNet_stage1/manifest.json \
       --model-path results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500/model_best.keras

Let the command resolve the model and default inputs automatically:

.. code-block:: bash

   geoprior-build physics-payload-npz \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1

Use one explicit full input NPZ:

.. code-block:: bash

   geoprior-build physics-payload-npz \
       --manifest results/nansha_GeoPriorSubsNet_stage1/manifest.json \
       --inputs-npz results/nansha_GeoPriorSubsNet_stage1/artifacts/full_inputs.npz

Assemble the payload from selected Stage-1 splits instead:

.. code-block:: bash

   geoprior-build physics-payload-npz \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --splits train val test

Write a custom output file and label the source explicitly:

.. code-block:: bash

   geoprior-build physics-payload-npz \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --output exports/nansha_phys_payload_full.npz \
       --source-label full_city_union

Limit export to a smaller number of batches:

.. code-block:: bash

   geoprior-build physics-payload-npz \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --batch-size 128 \
       --max-batches 20

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--inputs-npz``
   Use one explicit input NPZ instead of asking the command to reuse or
   assemble inputs from Stage-1 artifacts. 

``--splits``
   Control which Stage-1 split inputs are merged when no explicit input
   NPZ is supplied. 

``--model-path``
   Point directly to the trained ``.keras`` model used for export.
   Otherwise the command tries to resolve one automatically.

``--output`` / ``--output-name`` / ``--source-label``
   Control the saved payload path and how the source payload is labeled
   in the output metadata and default filename logic. 

``--batch-size`` and ``--max-batches``
   Control how much data is processed per export pass and whether the
   export is limited to only part of the dataset.

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/figure_generation/plot_physics_fields`

.. _build-external-validation-fullcity:

``external-validation-fullcity``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``external-validation-fullcity`` when you want one **end-to-end
external validation artifact workflow** starting from Stage-1 artifacts
and a trained Stage-2 inference bundle.

This command is the most orchestration-heavy builder in this group.
Its implementation can:

- resolve Stage-1 manifest and split input artifacts,
- build ``full_inputs.npz`` when needed,
- resolve the Stage-2 inference bundle from a model path, a Stage-2
  manifest, or a Stage-2 run directory,
- export a full-city physics payload,
- match validation sites to the nearest model pixels,
- compute site-level validation outputs and headline metrics. 

In other words, this is the convenience command you use when you want
the **whole external validation pipeline** rather than only one piece of
it. 

Usage
^^^^^

Run the full workflow with explicit Stage-1 and Stage-2 hints:

.. code-block:: bash

   geoprior build external-validation-fullcity \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --stage2-run-dir results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500 \
       --validation-csv data/nansha_validation.csv

Resolve from manifests instead:

.. code-block:: bash

   geoprior-build external-validation-fullcity \
       --manifest results/nansha_GeoPriorSubsNet_stage1/manifest.json \
       --stage2-manifest results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500/manifest.json \
       --validation-csv data/nansha_validation.csv

Provide one explicit model file:

.. code-block:: bash

   geoprior-build external-validation-fullcity \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --model-path results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500/model_best.keras \
       --validation-csv data/nansha_validation.csv

Reuse a prebuilt full-city input payload:

.. code-block:: bash

   geoprior-build external-validation-fullcity \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --full-inputs-npz results/nansha_GeoPriorSubsNet_stage1/artifacts/full_inputs.npz \
       --validation-csv data/nansha_validation.csv

Choose explicit output locations:

.. code-block:: bash

   geoprior-build external-validation-fullcity \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --stage2-run-dir results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500 \
       --validation-csv data/nansha_validation.csv \
       --outdir exports/external_validation_nansha \
       --out-payload exports/external_validation_nansha/fullcity_phys_payload.npz

Control matching and validation column choices:

.. code-block:: bash

   geoprior-build external-validation-fullcity \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --stage2-run-dir results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500 \
       --validation-csv data/nansha_validation.csv \
       --x-col x \
       --y-col y \
       --productivity-col step3_specific_capacity_Lps_per_m \
       --thickness-col approx_compressible_thickness_m

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--validation-csv``
   External validation table used for site-to-pixel matching and metric
   computation. 

``--full-inputs-npz`` and ``--out-payload``
   Reuse or explicitly place the full-city input and payload artifacts
   involved in the workflow. 

``--stage2-run-dir`` / ``--stage2-manifest`` / ``--model-path``
   Multiple ways to resolve the trained inference bundle used for the
   full-city payload export. 

``--x-col`` / ``--y-col`` / ``--productivity-col`` /
``--thickness-col``
   Select the validation-table columns used for nearest-pixel matching
   and for the site-level metrics. 

``--horizon-reducer`` / ``--site-reducer`` /
``--max-match-distance-m`` / ``--min-unique-pixels``
   Control how horizon-level arrays are collapsed to site-level values
   and how strict the site-to-pixel sanity checks are. 


Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/figure_generation/plot_external_validation`

.. _build-external-validation-metrics:

``external-validation-metrics``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``external-validation-metrics`` when you already have the necessary
artifacts and want to compute **external validation joins and headline
metrics** without rerunning the full-city payload export workflow.

This command focuses on the site-level validation step itself. It loads
a validation CSV, resolves the Stage-1 manifest and the input NPZ used
for matching, loads a saved physics payload, matches each validation
site to the nearest model pixel, and writes both a site-level table and
a metrics JSON. The top-level module docstring describes it as a command
for computing borehole or pumping validation metrics by matching site
coordinates to the nearest pixel in a Stage-1 grid and then joining
model-derived fields from a saved physics payload. 

This is the right command when the payload already exists and you want
to recompute metrics or experiment with different matching and reduction
settings without rebuilding the payload first. 

Usage
^^^^^

Compute metrics from explicit Stage-1 and payload artifacts:

.. code-block:: bash

   geoprior build external-validation-metrics \
       --manifest results/nansha_GeoPriorSubsNet_stage1/manifest.json \
       --physics-payload results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500/nansha_phys_payload_full_city_union.npz \
       --validation-csv data/nansha_validation.csv

Resolve the Stage-1 side from the Stage-1 directory:

.. code-block:: bash

   geoprior-build external-validation-metrics \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --physics-payload results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500/nansha_phys_payload_full_city_union.npz \
       --validation-csv data/nansha_validation.csv

Use a specific split input NPZ for matching:

.. code-block:: bash

   geoprior-build external-validation-metrics \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --split test \
       --inputs-npz results/nansha_GeoPriorSubsNet_stage1/artifacts/test_inputs.npz \
       --physics-payload results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500/nansha_phys_payload_test.npz \
       --validation-csv data/nansha_validation.csv

Let the command resolve the payload from a Stage-2 manifest:

.. code-block:: bash

   geoprior-build external-validation-metrics \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --stage2-manifest results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500/manifest.json \
       --validation-csv data/nansha_validation.csv

Write the metrics artifacts to a chosen directory:

.. code-block:: bash

   geoprior-build external-validation-metrics \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --physics-payload results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500/nansha_phys_payload_full_city_union.npz \
       --validation-csv data/nansha_validation.csv \
       --outdir exports/external_validation_metrics

Use different reducers and stricter sanity checks:

.. code-block:: bash

   geoprior-build external-validation-metrics \
       --stage1-dir results/nansha_GeoPriorSubsNet_stage1 \
       --physics-payload results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500/nansha_phys_payload_full_city_union.npz \
       --validation-csv data/nansha_validation.csv \
       --horizon-reducer mean \
       --site-reducer median \
       --max-distance-m 3000 \
       --min-unique-pixels 5


Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--validation-csv``
  External site table used for the nearest-pixel join. 

``--physics-payload``
  Saved physics payload NPZ from which model-derived fields such as
  ``K`` and ``Hd`` are joined into the site-level validation table.

``--split`` / ``--inputs-npz`` / ``--coord-scaler``
  Control which Stage-1 input coordinates are used for matching and how
  they are inverse-transformed into physical space. 

``--stage2-manifest``
  Optional shortcut for resolving the payload path from a Stage-2 run.

``--horizon-reducer`` / ``--site-reducer`` / ``--max-distance-m`` / ``--min-unique-pixels``
  Control the horizon reduction, the site aggregation, and the match
  sanity thresholds. 


Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/figure_generation/plot_external_validation`

.. _build-sm3-collect-summaries:

``sm3-collect-summaries``
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``sm3-collect-summaries`` when you want to collect **many per-run
SM3 summary CSV files into one combined table**.

This command scans an SM3 suite root, reads the per-regime summary CSV
files found beneath it, injects the inferred regime and run directory
into each collected table, concatenates everything, and writes both a
combined CSV and a combined JSON output. If no explicit suite root is
provided, it can discover the newest suite-like directory under a
results root or resolve one from config. 

This is the natural build-side companion to the run-family
``sm3-suite`` command: the suite creates many regime runs, and
``sm3-collect-summaries`` condenses their summary outputs into one
combined artifact.

Usage
^^^^^

Collect one explicit suite root:

.. code-block:: bash

   geoprior build sm3-collect-summaries \
       --suite-root results/sm3_tau_suite_20260330-120000

or discover the newest suite under a results directory:

.. code-block:: bash

   geoprior-build sm3-collect-summaries \
       --results-dir results

Write the combined outputs into a dedicated folder:

.. code-block:: bash

   geoprior-build sm3-collect-summaries \
       --suite-root results/sm3_tau_suite_20260330-120000 \
       --outdir exports/sm3_summary_bundle

Choose a custom output stem:

.. code-block:: bash

   geoprior-build sm3-collect-summaries \
       --suite-root results/sm3_tau_suite_20260330-120000 \
       --output-stem tau_suite_combined

Write explicit CSV and JSON paths:

.. code-block:: bash

   geoprior-build sm3-collect-summaries \
       --suite-root results/sm3_tau_suite_20260330-120000 \
       --out-csv exports/tau_suite_combined.csv \
       --out-json exports/tau_suite_combined.json

Tighten the collection rules:

.. code-block:: bash

   geoprior-build sm3-collect-summaries \
       --suite-root results/sm3_tau_suite_20260330-120000 \
       --summary-name sm3_synth_summary.csv \
       --regime-pattern "sm3_(?:tau|both)_(.+?)_50$" \
       --strict

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--suite-root``
   Explicit suite directory to scan. If omitted, the command can try to
   discover the newest suite under ``--results-dir`` or from config.

``--summary-name``
   Name of the per-run summary CSV files to collect. 

``--regime-pattern``
   Regular expression used to infer the regime name from each run
   directory. 

``--output-stem`` / ``--out-csv`` / ``--out-json``
   Control whether the command writes to default combined outputs or to
   fully explicit CSV and JSON destinations. 

``--strict``
   Fail on malformed or unreadable summary files instead of skipping
   them.
   

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/figure_generation/plot_sm3_bounds_ridge_summary`

Tables, summaries, and derived products
------------------------------------------
These commands build compact downstream artifacts for analysis,
comparison, or reporting.


.. _build-brier-exceedance:

``brier-exceedance``
~~~~~~~~~~~~~~~~~~~~

Use ``brier-exceedance`` when you want to compute **Brier scores for
subsidence exceedance events** from calibrated forecast CSV files.

This command works from forecast outputs that contain
``subsidence_actual`` together with predictive quantiles such as
``subsidence_q10``, ``subsidence_q50``, and ``subsidence_q90``. It
approximates exceedance probabilities by piecewise-linear interpolation
of the quantile-based CDF, then computes tidy Brier results across one
or more thresholds and years. It supports test-first auto-discovery,
explicit city CSV overrides, and a choice between test or validation
sources. 

This command is a good fit when you want one compact table for event
probability skill rather than a full forecast or uncertainty panel.

Usage
^^^^^

Auto-discover the forecast CSVs under a results root:

.. code-block:: bash

   geoprior build brier-exceedance \
       --root results

Force the command to use validation CSVs:

.. code-block:: bash

   geoprior-build brier-exceedance \
       --root results \
       --source val

Use explicit city CSVs:

.. code-block:: bash

   geoprior-build brier-exceedance \
       --ns-csv results/nansha_eval_calibrated.csv \
       --zh-csv results/zhongshan_eval_calibrated.csv

Evaluate several thresholds over selected years:

.. code-block:: bash

   geoprior-build brier-exceedance \
       --root results \
       --thresholds 30,50,70 \
       --years 2020,2021,2022

Write to an explicit output CSV:

.. code-block:: bash

   geoprior-build brier-exceedance \
       --root results \
       --out external_brier_scores.csv

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--source``
   Choose ``auto``, ``test``, or ``val`` discovery mode for the input
   forecast CSVs. ``auto`` prefers test-set forecasts when available.

``--thresholds``
   Comma-separated exceedance thresholds in mm/yr.

``--years``
   Comma-separated years to evaluate, or ``all``. 

``--ns-csv`` and ``--zh-csv``
   Explicit city-level CSV overrides that bypass directory scanning.

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/compute_brier_exceedance`


.. _build-hotspots:

``hotspots``
~~~~~~~~~~~~

Use ``hotspots`` when you want to compute a **city × year hotspot
characteristics table** from evaluation and future forecast CSVs.

This command compares forecast or annualized subsidence against a
baseline year, computes anomaly magnitude, selects hotspots using a
percentile threshold, and exports a compact summary table. It supports
different subsidence interpretations (``cumulative``, ``rate``, or
``increment``), several future years, and both CSV and LaTeX export.
The implementation summarizes hotspot counts, hotspot subsidence
intensity, anomaly magnitude, baseline mean, and the threshold used for
each city-year pair.

This is the right command when you want a compact **hotspot table**
rather than hotspot point clouds.

Usage
^^^^^

Auto-discover city inputs and summarize the default future years:

.. code-block:: bash

   geoprior build hotspots \
       --ns-src results/nansha \
       --zh-src results/zhongshan

Use explicit eval and future CSVs:

.. code-block:: bash

   geoprior-build hotspots \
       --ns-eval results/nansha_eval.csv \
       --ns-future results/nansha_future.csv \
       --zh-eval results/zhongshan_eval.csv \
       --zh-future results/zhongshan_future.csv

Choose a different baseline year and percentile threshold:

.. code-block:: bash

   geoprior-build hotspots \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --baseline-year 2022 \
       --percentile 95

Use a different forecast quantile and baseline source:

.. code-block:: bash

   geoprior-build hotspots \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --quantile q90 \
       --baseline-source q50

Export both CSV and LaTeX:

.. code-block:: bash

   geoprior-build hotspots \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --format both \
       --out tab_hotspots

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--baseline-year``
   Reference year against which hotspot anomaly is computed.

``--percentile``
   Percentile threshold used to define hotspot anomaly magnitude.

``--subsidence-kind``
   Tell the command whether the CSV values are cumulative, annual rate,
   or annual increment. 

``--baseline-source``
   Choose whether the baseline comes from ``actual`` or ``q50`` in the
   evaluation CSV. 

``--quantile``
   Select the forecast quantile used for hotspot computation.

``--format``
   Export as CSV, TeX, or both. 


Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/compute_hotspots`

.. _build-hotspots-summary:

``hotspots-summary``
~~~~~~~~~~~~~~~~~~~~

Use ``hotspots-summary`` when you already have a **hotspot point CSV**
and want a grouped summary by city, year, and kind.

This command is narrower than ``hotspots``. It does not recompute the
hotspot logic from evaluation and future forecasts; instead, it reads a
hotspot point cloud CSV and summarizes it into a tidy table with counts,
min/mean/max hotspot value, and min/mean/max anomaly metric. When
available, it also preserves grouped baseline and threshold summaries.

This is the right command when the hotspot points already exist and you
only want a reporting table.

Usage
^^^^^

Summarize one hotspot point cloud CSV:

.. code-block:: bash

   geoprior build hotspots-summary \
       --hotspot-csv results/fig6_hotspot_points.csv

Write to a custom output path:

.. code-block:: bash

   geoprior-build hotspots-summary \
       --hotspot-csv results/fig6_hotspot_points.csv \
       --out fig6_hotspot_summary.csv

Suppress console printing:

.. code-block:: bash

   geoprior-build hotspots-summary \
       --hotspot-csv results/fig6_hotspot_points.csv \
       --quiet true

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--hotspot-csv``
   Required hotspot point CSV produced upstream by a hotspot or spatial
   figure workflow. 

``--out``
   Output CSV written under ``scripts/out/`` when relative.

``--quiet``
   Disable console table printing. 


Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/summarize_hotspots`

.. _build-model-metrics:

``model-metrics``
~~~~~~~~~~~~~~~~~

Use ``model-metrics`` when you want a **unified model metrics table**
built from one results root or one run directory.

This command scans ablation record JSONL files, preferring updated
records when duplicates exist, normalizes legacy units when necessary,
flattens per-horizon metrics, and writes a wide metrics table plus
optional long-form horizon-level tables. It can filter by city or model,
deduplicate by timestamp/city/model, and export both CSV and JSON
variants.

This is a good fit when you want a reusable metrics inventory across
many runs rather than a paper-style ablation table.

Usage
^^^^^

Scan the default results root:

.. code-block:: bash

   geoprior build model-metrics

Scan one specific run directory:

.. code-block:: bash

   geoprior-build model-metrics \
       --src results/nansha_GeoPriorSubsNet_stage1/train_20260330-101500

Restrict to selected cities or models:

.. code-block:: bash

   geoprior-build model-metrics \
       --src results \
       --city Nansha,Zhongshan \
       --models GeoPriorSubsNet

Disable the long per-horizon export:

.. code-block:: bash

   geoprior-build model-metrics \
       --src results \
       --include-long false

Write to a custom output stem:

.. code-block:: bash

   geoprior-build model-metrics \
       --src results \
       --out comparison/model_metrics_main

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--src``
   Accept either a results root or one run directory to scan.

``--include-long``
   Control whether per-horizon long tables are also written.

``--dedupe``
   Deduplicate on ``(timestamp, city, model)`` while preferring updated.

``--city`` and ``--models``
   Filter the collected runs before export. 

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/build_model_metrics`


.. _build-ablation-table:

``ablation-table``
~~~~~~~~~~~~~~~~~~

Use ``ablation-table`` when you want a **paper-oriented ablation or
sensitivity table** built from ablation records.

This command is broader than ``model-metrics`` in formatting and table
logic. It can scan a results root, read explicit JSONL/JSON/CSV inputs,
normalize metric units, flatten per-horizon blocks, sort by one metric,
and write compact CSV, JSON, TeX, or text outputs. It also supports
paper mode, best-per-city exports, and grouped supplementary outputs
such as S6/S7-style tables. 

This is the right command when you want something closer to a
publication-ready ablation table rather than a raw metrics inventory.

Usage
^^^^^

Build a default ablation table from the results root:

.. code-block:: bash

   geoprior build ablation-table

Use explicit inputs instead of scanning:

.. code-block:: bash

   geoprior-build ablation-table \
       --input results/runA/ablation_record.updated.jsonl \
       --input results/runB/ablation_record.updated.jsonl

Sort by a chosen metric and export CSV plus TeX:

.. code-block:: bash

   geoprior-build ablation-table \
       --root results \
       --sort-by mae \
       --formats csv,tex \
       --out table_ablations

Use paper mode with one error metric:

.. code-block:: bash

   geoprior-build ablation-table \
       --root results \
       --for-paper \
       --err-metric rmse \
       --keep-r2

Also export best-per-city rows:

.. code-block:: bash

   geoprior-build ablation-table \
       --root results \
       --best-per-city

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--input``
   Repeatable explicit inputs that override the default root scan.
   Supports JSONL, JSON, and CSV. 

``--formats``
   Choose among CSV, JSON, TeX, and TXT style outputs.

``--for-paper`` / ``--err-metric`` / ``--keep-r2``
   Switch to a more compact paper-oriented table layout.

``--best-per-city``
   Also export the best row per city under the chosen sort metric.

``--group-cols`` / ``--s6-metrics`` / ``--s7-cols``
   Support grouped supplementary outputs such as S6 and S7 style
   summaries.

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/build_ablation_table`


.. _build-update-ablation-records:

``update-ablation-records``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``update-ablation-records`` when you want to **patch or enrich
ablation record JSONL files with metrics**.

This command is part of the public build family and is registered with
the description *“Patch ablation record JSONL with metrics.”* At the
moment, I have only the registry exposure for this command in the files
you shared, not the implementation module itself, so this subsection is
kept intentionally short until that file is reviewed directly.

Usage
^^^^^

Inspect the public CLI help first:

.. code-block:: bash

   geoprior build update-ablation-records --help

or:

.. code-block:: bash

   geoprior-build update-ablation-records --help

Note
^^^^

Once you share the ``update_ablation_records.py`` implementation, this
subsection can be expanded in the same style as the others with:

- typical inputs,
- update logic,
- output behavior,
- distinctive options.


Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/index`

.. _build-extend-forecast:

``extend-forecast``
~~~~~~~~~~~~~~~~~~~

Use ``extend-forecast`` when you want to **extend a future forecast CSV
by extrapolation** for one or more cities.

This command resolves eval and future forecast CSVs, chooses a split,
then extends the future trajectory by one or more years using either a
linear-fit or linear-last trend. It also supports uncertainty widening
rules across the added years and lets you control whether the output
kind remains cumulative or is converted to rate form. 

This is a good fit when you want a longer future panel without rerunning
the full Stage-4 forecasting workflow.

Usage
^^^^^

Extend the default future horizon by one year:

.. code-block:: bash

   geoprior build extend-forecast \
       --ns-src results/nansha \
       --zh-src results/zhongshan

Use explicit eval and future CSVs:

.. code-block:: bash

   geoprior-build extend-forecast \
       --ns-eval results/nansha_eval.csv \
       --ns-future results/nansha_future.csv \
       --zh-eval results/zhongshan_eval.csv \
       --zh-future results/zhongshan_future.csv

Add explicit years instead of a count:

.. code-block:: bash

   geoprior-build extend-forecast \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --years 2026 2027

Use a different extrapolation rule and fit window:

.. code-block:: bash

   geoprior-build extend-forecast \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --method linear_last \
       --window 4

Control uncertainty growth:

.. code-block:: bash

   geoprior-build extend-forecast \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --unc-growth linear \
       --unc-scale 1.5

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--method``
   Choose between ``linear_fit`` and ``linear_last`` extrapolation.

``--years`` and ``--add-years``
   Either list the exact years to add or request N more years.

``--subsidence-kind`` and ``--out-kind``
   Control whether the inputs and outputs are treated as cumulative,
   rate, or increment style series.

``--unc-growth`` and ``--unc-scale``
   Control how predictive uncertainty widens into the added years.

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/extend_forecast`

.. _build-boundary:

``boundary``
~~~~~~~~~~~~

Use ``boundary`` when you want to derive a **boundary polygon from
forecast point clouds**.

The public build command ``boundary`` is exposed from the legacy
``make-boundary`` script. It loads coordinate points from the eval and
future CSVs, builds either a convex hull or a concave hull, and exports
GeoJSON, Shapefile, or both. The implementation checks that the written
boundary file actually exists after export.

This is a good fit when you need a quick city outline or a lightweight
polygon boundary for later spatial workflows.

Usage
^^^^^

Create one default boundary export:

.. code-block:: bash

   geoprior build boundary \
       --ns-src results/nansha \
       --zh-src results/zhongshan

Use explicit eval and future CSVs:

.. code-block:: bash

   geoprior-build boundary \
       --ns-eval results/nansha_eval.csv \
       --ns-future results/nansha_future.csv \
       --zh-eval results/zhongshan_eval.csv \
       --zh-future results/zhongshan_future.csv

Choose a concave hull instead of the default convex hull:

.. code-block:: bash

   geoprior-build boundary \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --method concave \
       --alpha 0.2

Write both GeoJSON and Shapefile outputs:

.. code-block:: bash

   geoprior-build boundary \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --format both

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--method``
   Choose ``convex`` or ``concave`` hull construction.

``--alpha``
   Concave-hull ratio used when ``--method concave`` is selected.

``--format``
   Export GeoJSON, Shapefile, or both. 

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/make_boundary`

.. _build-exposure:

``exposure``
~~~~~~~~~~~~

Use ``exposure`` when you want to build a **simple exposure proxy
table** from forecast point clouds.

The public build command ``exposure`` is exposed from the legacy
``make-exposure`` script. It loads spatial points, deduplicates by
``sample_idx``, and writes one exposure table per city set. It supports
a uniform exposure mode as well as a density-based proxy computed from
k-nearest-neighbor distance using scikit-learn. 

This is a good fit when you need a lightweight exposure field for
downstream mapping, ranking, or derived risk proxies.

Usage
^^^^^

Build the default exposure proxy:

.. code-block:: bash

   geoprior build exposure \
       --ns-src results/nansha \
       --zh-src results/zhongshan

Use explicit eval and future CSVs:

.. code-block:: bash

   geoprior-build exposure \
       --ns-eval results/nansha_eval.csv \
       --ns-future results/nansha_future.csv \
       --zh-eval results/zhongshan_eval.csv \
       --zh-future results/zhongshan_future.csv

Switch to a uniform exposure field:

.. code-block:: bash

   geoprior-build exposure \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --mode uniform

Control the density exposure neighborhood size:

.. code-block:: bash

   geoprior-build exposure \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --mode density \
       --k 50

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--mode``
   Choose ``uniform`` or ``density`` exposure behavior.

``--k``
   kNN size used by the density-style exposure proxy.

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/make_exposure`

.. _build-district-grid:

``district-grid``
~~~~~~~~~~~~~~~~~

Use ``district-grid`` when you want to create a **grid-based district
layer** from forecast point clouds.

The public build command ``district-grid`` is exposed from the legacy
``make-district-grid`` script. It resolves eval and future CSVs, builds
a rectangular grid across the spatial extent, optionally clips it to a
boundary polygon, writes GeoJSON or Shapefile outputs, and can also
write a sample-to-zone assignment CSV. 

This is a good fit when you want stable Zone IDs for later hotspot,
exposure, or policy-style spatial aggregation.

Usage
^^^^^

Create a default district grid:

.. code-block:: bash

   geoprior build district-grid \
       --ns-src results/nansha \
       --zh-src results/zhongshan

Change the grid resolution:

.. code-block:: bash

   geoprior-build district-grid \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --nx 16 \
       --ny 14

Use an external boundary and clip the grid:

.. code-block:: bash

   geoprior-build district-grid \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --boundary exports/boundary_nansha.geojson \
       --clip-boundary

Also export sample-to-zone assignments:

.. code-block:: bash

   geoprior-build district-grid \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --assign-samples

Write both GeoJSON and Shapefile outputs:

.. code-block:: bash

   geoprior-build district-grid \
       --ns-src results/nansha \
       --zh-src results/zhongshan \
       --format both

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--nx`` and ``--ny``
   Set the grid resolution in east-west columns and north-south rows.

``--pad``
   Expand the grid extent slightly beyond the raw point span.

``--boundary`` / ``--clip-boundary`` / ``--min-area-frac``
   Control optional clipping of the grid to one boundary polygon and
   dropping very small clipped cells. 

``--assign-samples``
   Also write a CSV mapping ``sample_idx`` to ``zone_id``.

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/make_district_grid`

.. _build-clusters-with-zones:

``clusters-with-zones``
~~~~~~~~~~~~~~~~~~~~~~~

Use ``clusters-with-zones`` when you want to assign **hotspot cluster
centroids to Zone IDs** from a district grid.

The public build command ``clusters-with-zones`` is exposed from the
legacy ``tag-clusters-with-zones`` script. It reads a cluster-centroid
CSV and a district-grid GeoJSON or Shapefile, optionally filters by
city or year, then assigns each centroid to the containing polygon or,
when requested, to the nearest polygon. The final output is a clean CSV
with cluster identifiers, zone IDs, zone labels, centroids, and summary
cluster fields. 

This is the natural follow-up to ``district-grid`` when you want to join
cluster analysis back to stable zone labels.

Usage
^^^^^

Assign all clusters to a district grid:

.. code-block:: bash

   geoprior build clusters-with-zones \
       --clusters results/hotspot_clusters.csv \
       --grid exports/district_grid_nansha.geojson

Restrict the assignment to one city and one year:

.. code-block:: bash

   geoprior-build clusters-with-zones \
       --clusters results/hotspot_clusters.csv \
       --grid exports/district_grid_nansha.geojson \
       --city Nansha \
       --year 2027

Use nearest-zone fallback when a centroid falls outside polygons:

.. code-block:: bash

   geoprior-build clusters-with-zones \
       --clusters results/hotspot_clusters.csv \
       --grid exports/district_grid_nansha.geojson \
       --nearest

Use non-default field names from the grid:

.. code-block:: bash

   geoprior-build clusters-with-zones \
       --clusters results/hotspot_clusters.csv \
       --grid exports/custom_grid.geojson \
       --zone-id-field zid \
       --zone-label-field zlabel

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--clusters``
   Cluster centroid CSV containing at least ``centroid_x`` and
   ``centroid_y``. 

``--grid``
   District grid GeoJSON or Shapefile used for polygon assignment.

``--zone-id-field`` and ``--zone-label-field``
   Select the grid fields used as the output zone identifier and label.

``--nearest``
   Assign the nearest zone when no polygon contains the centroid.

Related examples 
^^^^^^^^^^^^^^^^^
- :doc:`../auto_examples/tables_and_summaries/tag_clusters_with_zones`


From here
---------

A natural next reading path is:

- :doc:`shared_conventions`
- :doc:`../auto_examples/tables_and_summaries/index`
- :doc:`../auto_examples/figure_generation/index`
