Stage-5
=======

Stage-5 is the **cross-city transfer evaluation and
benchmarking stage** of the GeoPrior-v3 workflow.

It is the stage where GeoPrior-v3 moves from single-run
training, tuning, or inference into a structured
**source-to-target transfer analysis**. Rather than focusing
on one city and one saved model in isolation, Stage-5
evaluates how a source-domain model behaves when applied to
another city under different transfer strategies, calibration
policies, and rescaling assumptions.

This makes Stage-5 one of the most scientifically important
stages in the workflow. It is where GeoPrior-v3 asks not
only:

- *Can the model fit one city well?*

but also:

- *How well does the learned structure transfer?*
- *How sensitive is performance to scaling assumptions?*
- *What changes when calibration is reused or refit?*
- *Does warm-start adaptation improve target performance?*

In the current implementation, Stage-5 is explicitly exposed
as a **cross-city transfer evaluation** workflow supporting
**baseline**, **transfer**, and **warm-start** strategies.

What Stage-5 does
-----------------

Stage-5 orchestrates a grid of transfer experiments between
two cities.

At a high level, it:

- loads Stage-1 bundles for two cities;
- builds a set of transfer jobs across directions and
  strategies;
- resolves the source model artifact to reuse;
- aligns the target inputs to the source schema;
- optionally reprojects target dynamic inputs into source
  scaling;
- optionally applies calibration from the source or target
  side;
- evaluates each job on validation or test splits;
- writes per-job forecast CSVs and physics payloads;
- aggregates all jobs into transfer summary files.

This means Stage-5 is best understood as a **benchmarking
matrix stage** rather than as a single model execution step.

Why Stage-5 matters
-------------------

Stage-5 matters because real geohazard models are often used
beyond the exact city or domain on which they were trained.

A model that performs well in one city may fail in another
for reasons such as:

- different scaling conventions,
- different feature distributions,
- different static covariate availability,
- different future-driver ordering,
- or different calibration behavior.

Stage-5 is designed to make those differences explicit.

Instead of hiding transfer behavior inside ad hoc notebooks,
GeoPrior-v3 gives it a dedicated stage with:

- explicit strategies,
- explicit job construction,
- explicit schema audits,
- explicit calibration modes,
- and explicit benchmark outputs.

That makes the transfer study reproducible, comparable, and
scientifically inspectable.

Stage-5 entry point
-------------------

Stage-5 is exposed through the modern GeoPrior wrapper while
preserving the existing transfer implementation underneath.

The wrapper supports a configuration-driven launch style with
options such as:

- ``--config``
- ``--config-root``
- ``--city-a``
- ``--city-b``
- ``--model``
- ``--results-dir``
- repeated ``--set KEY=VALUE`` overrides

The wrapper then forwards the original transfer arguments to
the legacy Stage-5 CLI unchanged.

A good first move is:

.. code-block:: bash

   geoprior-run --help

Then inspect the Stage-5 transfer subcommand in your
installed environment.

.. note::

   The wrapper is intentionally thin. It seeds defaults from
   configuration but preserves the original transfer CLI
   behavior. The installed help output is therefore the best
   source for the exact invocation syntax.

The Stage-5 problem setup
-------------------------

Stage-5 is built around two cities:

- **city A**
- **city B**

It then constructs directional experiments such as:

- ``A_to_A``
- ``B_to_B``
- ``A_to_B``
- ``B_to_A``

These directions are combined with one or more strategies,
splits, calibration modes, and rescale modes to form the full
job matrix.

This is a very important design choice: Stage-5 does not
evaluate only one transfer path. It builds a **systematic
experiment grid**.

Strategies evaluated by Stage-5
-------------------------------

Stage-5 supports three main strategies.

**1. baseline**

This is the in-domain reference case:

- ``A_to_A``
- ``B_to_B``

It answers the question:

*How well does each city perform when evaluated against its
own trained or tuned artifact family?*

Baseline runs are important because they define the natural
comparison point for transfer performance.

**2. xfer**

This is the zero-shot cross-city transfer case:

- ``A_to_B``
- ``B_to_A``

It answers the question:

*How well does a source-domain model generalize directly to a
target city without warm-start adaptation?*

**3. warm**

This is the warm-start transfer case:

- ``A_to_B`` with target adaptation
- ``B_to_A`` with target adaptation

It answers the question:

*Can limited target-domain adaptation improve transfer
performance compared with pure zero-shot reuse?*

Taken together, these three strategies let Stage-5 compare
in-domain performance, zero-shot transfer, and limited target
adaptation in one reproducible framework.

Splits evaluated by Stage-5
---------------------------

Stage-5 supports evaluation on:

- ``val``
- ``test``

These splits are read from the Stage-1 bundles for each city.

This is useful because the same transfer strategy can be
studied under validation-like and held-out evaluation
conditions.

Calibration modes
-----------------

Stage-5 supports several calibration modes:

- ``none``
- ``source``
- ``target``

Each one answers a different question.

**none**

No interval calibration is applied.

This is the cleanest way to inspect raw transferred quantile
behavior.

**source**

A calibrator near the source model is reused.

This asks whether a calibration learned in the source domain
remains useful after transfer.

**target**

A new calibrator is fitted on the target validation set.

This asks whether transfer improves when quantile intervals
are adapted to the target domain.

.. admonition:: Best practice

   Be explicit about which calibration mode you are using
   when interpreting Stage-5 results.

   These modes answer different scientific questions and
   should not be merged casually in reporting.

Rescale modes
-------------

Stage-5 also supports two rescale modes:

- ``as_is``
- ``strict``

These are especially important in cross-city work.

**as_is**

The target city remains in its own scaling convention.

This reflects a more direct target-domain inference path.

**strict**

The target dynamic inputs are reprojected into the source
scaling space before transfer evaluation.

This reflects a stricter source-space transfer setup.

In practice, this helps disentangle whether performance
differences are coming from:

- learned model structure,
- domain shift,
- or mismatched scaling conventions.

.. note::

   Baseline runs do not need strict reprojection. In the
   implementation, baseline is effectively kept in
   ``as_is`` mode.

What Stage-5 consumes
---------------------

Stage-5 consumes **Stage-1 bundles for both cities**.

These bundles provide:

- the manifest for each city,
- train/validation/test NPZ arrays,
- scaler metadata,
- feature lists,
- target semantics,
- and configuration snapshots.

The transfer stage uses these Stage-1 contracts to build both
the source-side and target-side evaluation context.

This is one of the strongest parts of the design: Stage-5
does not guess the schema of either city. It reads both from
their exported preprocessing contracts.

How Stage-5 resolves the source model
-------------------------------------

For each transfer job, Stage-5 resolves a source-side model
artifact.

It supports source model selection modes such as:

- ``auto``
- ``tuned``
- ``trained``

and load policies such as:

- ``auto``
- ``full``
- ``weights``

This allows Stage-5 to reuse:

- tuned winners from Stage-3,
- standard trained models from Stage-2,
- full saved model bundles,
- or weights-plus-rebuild workflows.

This is important because transfer benchmarking should not be
tied to only one artifact format.

Schema alignment and transfer safety
------------------------------------

One of the most important responsibilities of Stage-5 is
**schema validation between source and target**.

The stage explicitly checks:

- static feature compatibility,
- dynamic feature compatibility,
- future feature compatibility,
- dimension agreement,
- and feature-order agreement.

Static inputs are aligned by name so that shared source-side
static semantics can still be used even when the target has
missing or extra static features.

Dynamic and future features are more strict. If their names
or order differ, Stage-5 can:

- fail explicitly, or
- allow controlled reordering in a soft mode when the same
  names exist but the order differs.

This is critical for trustworthy transfer evaluation.

.. admonition:: Best practice

   Prefer harmonized Stage-1 feature schemas across cities.

   Stage-5 can help detect and sometimes soften transfer
   mismatches, but the most trustworthy setup is still to use
   the same feature lists and ordering across the cities being
   compared.

Warm-start behavior
-------------------

The ``warm`` strategy is a special case in Stage-5.

Instead of evaluating the source model directly on the target
city, it first adapts the source model using a subset of
target data, then evaluates the warmed model on the selected
split.

Warm-start settings include items such as:

- warm split,
- warm sample count,
- warm fraction,
- warm epochs,
- warm learning rate,
- warm sampling seed.

This creates a useful middle ground between:

- zero-shot transfer,
- and full retraining on the target domain.

Warm-start is especially important when you want to study
whether a small amount of target adaptation produces a large
gain in performance.

Internal structure of Stage-5
-----------------------------

The user does not need to know every helper function to use
Stage-5, but it is helpful to understand the overall flow.

**1. Load Stage-1 bundles for both cities**

Stage-5 first loads the Stage-1 bundle for city A and city B.

**2. Build directional experiment jobs**

It constructs all jobs implied by the chosen strategies,
splits, calibration modes, and rescale modes.

**3. Prepare target inputs for one job**

For a given job, it loads the target split arrays and
normalizes them into inference-ready form.

**4. Align static schema and validate transfer compatibility**

The target static inputs are aligned to the source schema by
name, and the dynamic/future schema is audited carefully.

**5. Optionally reproject target dynamics**

If strict rescaling is enabled, target dynamics are moved
into source scaling space before evaluation.

**6. Load or reuse the source model**

The source model is resolved from the saved artifact family.

**7. Optionally calibrate**

Depending on calibration mode, Stage-5 may load a source
calibrator, fit a target calibrator, or run without one.

**8. Predict and format outputs**

The transferred or warmed model is used to create evaluation
and future forecast CSVs.

**9. Compute metrics and diagnostics**

Stage-5 computes summary metrics, per-horizon metrics, and
optional physics-facing exports.

**10. Aggregate the experiment matrix**

All completed jobs are written into a shared JSON and CSV
benchmark summary.

Per-job outputs
---------------

Each successful Stage-5 job can write job-level artifacts
such as:

- evaluation CSV,
- future forecast CSV,
- optional physics payload NPZ,
- schema audit details,
- warm-start metadata,
- and per-job metrics returned to the final summary table.

The per-job forecast base name follows the logical pattern:

.. code-block:: text

   <source_city>_to_<target_city>_
   <strategy>_<split>_<calibration>_<rescale_mode>

This keeps the output naming explicit and easy to audit.

Metrics and summaries
---------------------

Stage-5 records several types of metrics.

Typical summary metrics include:

- coverage80,
- sharpness80,
- overall MAE,
- overall MSE,
- overall RMSE,
- overall R².

It also records per-horizon metrics such as:

- per-horizon MAE,
- per-horizon MSE,
- per-horizon RMSE,
- per-horizon R².

This is important because transfer may behave differently at
different forecast horizons, and Stage-5 is designed to make
that visible.

Unit handling
-------------

Stage-5 includes explicit unit handling for subsidence
outputs and metrics.

It supports settings such as:

- exported CSV subsidence unit,
- source physical output unit,
- metrics unit.

This is especially useful in transfer settings, where the
workflow may need to keep CSV presentation units and internal
metric units conceptually distinct.

The stage can also recompute metrics from the exported
evaluation CSV when needed. This is helpful when unit,
scaling, or calibration conventions have changed and you want
the final benchmark summary to reflect the exported products
rather than stale intermediate values.

The main Stage-5 benchmark outputs
----------------------------------

After all jobs finish, Stage-5 writes two main benchmark
artifacts:

**1. xfer_results.json**

This contains the full list of completed job dictionaries,
including metrics, schema audit fields, calibration choices,
rescale mode, model source details, and warm-start metadata.

**2. xfer_results.csv**

This flattens the benchmark results into a tabular form that
is easier to inspect, compare, plot, or export into papers
and reports.

The CSV includes columns such as:

- strategy,
- rescale mode,
- direction,
- source city,
- target city,
- split,
- calibration,
- overall metrics,
- coverage and sharpness,
- warm-start details,
- schema audit details,
- per-horizon metrics.

This makes Stage-5 the natural source for transfer summary
figures and cross-city comparison tables.

Where Stage-5 writes outputs
----------------------------

Stage-5 writes its benchmark outputs under a timestamped
transfer results directory with the general structure:

.. code-block:: text

   <results_dir>/xfer/<city_a>__<city_b>/<timestamp>/

This is a good design because it keeps one complete transfer
experiment grouped together.

Inside that directory, you will typically find:

- job-level eval/future CSVs,
- physics payload NPZ files where available,
- ``xfer_results.json``,
- ``xfer_results.csv``.

What to inspect after Stage-5 completes
---------------------------------------

After a Stage-5 run, inspect at least:

- the experiment output directory,
- ``xfer_results.json``,
- ``xfer_results.csv``,
- a few representative job-level eval CSVs,
- a few representative future CSVs,
- any physics payload files,
- the schema audit fields in the summary,
- and the warm-start metadata when warm runs were included.

You should be able to answer questions such as:

- Which strategies were actually evaluated?
- Which calibration modes were included?
- Which rescale modes were used?
- Did the benchmark include both directions?
- Were any schema reorders or mismatches recorded?
- Did warm-start improve over zero-shot transfer?
- Did strict rescaling help or hurt?
- Were baseline runs clearly separated from transfer runs?

Common Stage-5 mistakes
-----------------------

**Treating baseline and transfer as interchangeable**

Baseline is the in-domain reference. It should not be mixed
casually with cross-city transfer outcomes.

**Ignoring schema audit fields**

Transfer results are much harder to trust if dynamic or
future features were reordered or mismatched.

**Comparing calibrated and uncalibrated runs without care**

Calibration mode changes the interpretation of interval
metrics and should be tracked explicitly.

**Running too many combinations without a clear question**

Stage-5 can generate a large job matrix. It is better to
start from a focused scientific question than to evaluate
every possible combination blindly.

**Forgetting that warm-start is a different regime**

Warm-start is not just “better transfer.” It is a distinct
adaptation setting and should be reported as such.

Best practices
--------------

.. admonition:: Best practice

   Always report Stage-5 results with strategy, calibration
   mode, and rescale mode shown explicitly.

   Otherwise, transfer comparisons become hard to interpret.

.. admonition:: Best practice

   Review the schema audit fields before trusting a transfer
   comparison.

   A model may appear to transfer poorly when the real issue
   is a schema mismatch rather than a modeling limitation.

.. admonition:: Best practice

   Keep baseline, zero-shot transfer, and warm-start results
   separated clearly in figures and tables.

   These are different experimental questions.

.. admonition:: Best practice

   Use ``xfer_results.csv`` as the main summary table for
   downstream analysis.

   It is the cleanest compact artifact for comparing all jobs
   in one place.

A compact Stage-5 map
---------------------

The Stage-5 workflow can be summarized like this:

.. code-block:: text

   Stage-1 bundle (city A) + Stage-1 bundle (city B)
        ↓
   build directions and job matrix
        ↓
   choose strategy: baseline / xfer / warm
        ↓
   choose split: val / test
        ↓
   choose calibration: none / source / target
        ↓
   choose rescale mode: as_is / strict
        ↓
   align target schema to source contract
        ↓
   load or rebuild source model artifact
        ↓
   optionally warm-start on target subset
        ↓
   predict + format eval/future CSVs
        ↓
   compute overall + per-horizon metrics
        ↓
   export physics payload where available
        ↓
   aggregate into xfer_results.json + xfer_results.csv

Read next
---------

The next best pages after Stage-5 are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Diagnostics
      :link: diagnostics
      :link-type: doc
      :class-card: sd-shadow-sm

      Learn how to interpret benchmark metrics, schema audit
      fields, and physics-facing outputs.

   .. grid-item-card:: Inference and export
      :link: inference_and_export
      :link-type: doc
      :class-card: sd-shadow-sm

      See how forecast CSVs, summaries, and related artifacts
      fit into the broader GeoPrior workflow.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc
      :class-card: sd-shadow-sm card--configuration
      :data-preview: configuration.png
      :data-preview-label: GeoPrior configuration system

      Review how transfer defaults such as city pair and
      results directory can be driven from config.

   .. grid-item-card:: CLI guide
      :link: cli
      :link-type: doc
      :class-card: sd-shadow-sm card--cli
      :data-preview: cli.png
      :data-preview-label: GeoPrior command-line interface

      Move from the stage narrative to the full command
      interface.

.. seealso::

   - :doc:`workflow_overview`
   - :doc:`stage3`
   - :doc:`stage4`
   - :doc:`diagnostics`
   - :doc:`inference_and_export`
   - :doc:`configuration`
   - :doc:`cli`
   - :doc:`../scientific_foundations/scaling`
   - :doc:`../scientific_foundations/data_and_units`