Inference and export
====================

GeoPrior-v3 treats **inference** and **export** as
workflow-level concerns, not as small afterthoughts at the
end of training.

In a simple machine-learning workflow, inference may only
mean calling ``predict()`` on a saved model. In GeoPrior-v3,
that is not enough. A usable prediction also depends on:

- the Stage-1 target-domain contract,
- feature ordering and tensor layout,
- scaling and unit conventions,
- quantile handling,
- calibration policy,
- output formatting,
- and the specific scientific context of reuse or transfer.

This page explains how inference and export work across the
GeoPrior-v3 workflow and how to choose the right path for
your use case.

What this page covers
---------------------

This page covers four related things:

1. **model reuse**
   how trained or tuned models are reused after Stage-2 or
   Stage-3;

2. **forecast export**
   how evaluation and future forecast CSVs are generated and
   what they represent;

3. **portable artifacts**
   which files should be kept together if a model needs to be
   reused later or on another machine;

4. **deterministic build exports**
   how the build family generates payload and validation
   artifacts from existing results.

Where inference happens in the workflow
---------------------------------------

Inference and export are distributed across several stages.

**Stage-2**

Stage-2 trains a model, saves training-facing artifacts,
reloads a clean inference model, calibrates validation-based
intervals, and exports evaluation and future forecast
products.

**Stage-3**

Stage-3 does the same for the tuned winner. It saves tuned
model artifacts, best hyperparameters, tuned forecast CSVs,
and tuned diagnostics.

**Stage-4**

Stage-4 is the dedicated **saved-model inference** stage. It
loads a target-domain Stage-1 contract and a saved model
bundle, applies optional calibration policy, and exports
inference-ready products.

**Stage-5**

Stage-5 uses saved artifacts for cross-city transfer and
benchmarking, then exports per-job outputs and aggregated
transfer results.

A good mental model is:

.. code-block:: text

   Stage-2 / Stage-3
     create reusable model artifacts
        ↓
   Stage-4
     reuse one saved model on one target contract
        ↓
   Stage-5
     compare reuse across cities, strategies, and settings

Two broad inference styles
--------------------------

GeoPrior-v3 supports two broad styles of inference.

**1. In-process inference**

This is the simplest case. A model object already exists in
the current Python process, and you run inference directly in
memory.

This is useful for:

- quick checks during development,
- validation during training,
- debugging tensor outputs,
- inspecting raw model behavior before formatting.

**2. Portable saved-model inference**

This is the main workflow-facing case. A model trained or
tuned earlier is saved to disk, then later reloaded and used
in a new process or a new stage.

This is useful for:

- clean inference after training,
- reproducible saved-model reuse,
- same-domain evaluation,
- cross-domain transfer,
- future forecast generation,
- downstream reporting and benchmarking.

The second style is the more important one for GeoPrior-v3.

What a reusable GeoPrior artifact really is
-------------------------------------------

A reusable GeoPrior model is not just a single model file.

In practice, reliable reuse usually depends on a **bundle**
that may include:

- a full Keras model file,
- model weights,
- a model initialization manifest,
- scaling metadata,
- feature metadata,
- calibration factors,
- and the Stage-1 target contract that defines what the
  inputs and outputs mean.

This is why GeoPrior-v3 uses explicit manifests and summary
files. Saved-model reuse is not only about deserialization.
It is about preserving meaning.

Portable model artifacts
------------------------

Stage-2 and Stage-3 commonly leave behind artifacts such as:

- best full-model bundles,
- best weights files,
- final model exports,
- architecture JSON,
- ``model_init_manifest.json``,
- ``training_summary.json`` or ``tuning_summary.json``,
- ``scaling_kwargs.json``,
- and forecast / diagnostics files.

These files should be treated as a coherent family rather
than as unrelated leftovers.

A useful bundle mental model is:

.. code-block:: text

   run_dir/
     model_init_manifest.json
     best_model.keras
     best_model.weights.h5
     architecture.json
     training_summary.json or tuning_summary.json
     scaling_kwargs.json
     calibrated forecast CSVs
     diagnostics JSON
     physics payload NPZ

Not every run will contain every file, but the more of this
context you keep together, the easier reuse becomes.

Why the Stage-1 contract still matters
--------------------------------------

Even when a saved model already exists, GeoPrior-v3 still
needs the Stage-1 contract.

That contract defines:

- feature lists and ordering,
- input dimensions,
- horizon and mode,
- target semantics,
- coordinate treatment,
- and scaling-aware interpretation of outputs.

This is especially important in Stage-4 and Stage-5.

A model may come from one run, but the **target-domain
contract** comes from the Stage-1 manifest attached to the
data being evaluated. This is one of the most important
ideas in the whole inference story.

.. admonition:: Best practice

   Treat the Stage-1 manifest as the authoritative target
   contract for inference.

   A saved model file without the right target contract is
   often not enough for trustworthy export.

Saved-model inference in Stage-4
--------------------------------

Stage-4 is the main saved-model inference stage.

Its role is to:

- load a target-domain Stage-1 manifest,
- resolve a saved model bundle,
- rebuild or load an inference-capable model,
- optionally load or fit a calibrator,
- run prediction on a chosen split or custom NPZ input,
- format evaluation and future forecast CSVs,
- write an ``inference_summary.json``,
- and optionally write ``transfer_eval.json``.

This makes Stage-4 the cleanest path when you want to reuse a
trained or tuned model without rerunning Stage-2 or Stage-3.

Typical Stage-4 use cases
-------------------------

Stage-4 supports several practical reuse patterns.

**Same-domain inference**

A model is reused on the same city and the same general
contract it was trained on.

**Zero-shot transfer**

A source-domain model is evaluated directly on a different
target city without target adaptation.

**Source-calibrator reuse**

A source-domain interval calibrator is reused together with
the source model.

**Target-validation recalibration**

A fresh calibrator is fitted using target validation outputs
before export.

These are not interchangeable. They answer different
scientific questions.

Choosing the dataset for inference
----------------------------------

Stage-4 can run on several kinds of input datasets.

Common cases include:

- ``train``
- ``val``
- ``test``
- ``custom``

For standard splits, the stage uses the NPZ artifacts
recorded in the Stage-1 manifest.

For custom inference, a dedicated input NPZ path is supplied
explicitly, and optional target NPZ can also be provided if
metrics should be computed.

This means:

- standard split inference is **manifest-driven**;
- custom inference is **artifact-driven**.

Forecast export products
------------------------

GeoPrior export usually produces two main tabular products.

**Evaluation CSV**

This is used when targets are available. It lets the user
compare predictions with actual values and compute metrics.

**Future forecast CSV**

This is used when the workflow is forecasting beyond the
observed evaluation window. It contains forecast outputs
without ground-truth target columns.

These exports are central to the GeoPrior workflow because
they are the most portable artifacts for later scientific
analysis, plotting, reporting, or transfer comparison.

Typical information in exported CSVs
------------------------------------

Although exact columns depend on the formatter and active
configuration, forecast CSVs often include:

- sample or sequence identifiers,
- forecast step,
- time labels,
- coordinates,
- actual target values when available,
- central predictions,
- quantile predictions when enabled,
- and sometimes additional target-family outputs such as
  groundwater predictions.

A useful interpretation rule is:

- evaluation CSVs are for **comparison and metrics**;
- future CSVs are for **forward-looking scientific use**.

Quantile outputs and uncertainty export
---------------------------------------

GeoPrior-v3 supports probabilistic outputs through quantile
forecasts.

In that setting, a prediction tensor may represent multiple
quantiles rather than only a single central value. Export
logic therefore needs to:

- preserve quantile identity,
- ensure the quantile axis is interpreted correctly,
- optionally canonicalize the layout before calibration,
- and then format these outputs into CSV columns.

This matters because inference quality is not only about a
median or point forecast. It is also about the uncertainty
band around it.

Calibration in the export path
------------------------------

Calibration is an explicit part of the inference/export
story.

Later workflow stages may:

- fit interval calibration after training,
- apply calibration to tuned quantile outputs,
- reuse a source calibrator,
- or fit a target-side calibrator before export.

This means that exported uncertainty intervals are not always
raw model outputs. They may be **post-calibration forecast
products**.

.. admonition:: Best practice

   Always record whether an exported forecast is calibrated or
   uncalibrated.

   This strongly affects how interval metrics such as coverage
   and sharpness should be interpreted.

How to think about calibrated export
------------------------------------

A good practical rule is:

- the model provides the initial quantile structure;
- the calibrator rescales interval width around the center;
- the exported calibrated CSV becomes the artifact used for
  downstream uncertainty interpretation.

So when both raw and calibrated outputs exist, do not read
only one of them casually. Compare them intentionally.

Inference summaries
-------------------

GeoPrior-v3 writes JSON-side summaries so the CSV products
can be interpreted in context.

The most important one in Stage-4 is:

- ``inference_summary.json``

This summary typically records items such as:

- dataset name,
- city,
- model identity,
- horizon,
- mode,
- whether calibration was used,
- output CSV paths,
- and, when possible, physical-unit summary metrics.

This file is extremely useful because it explains **what the
inference run actually did**.

Optional transfer-eval summaries
--------------------------------

When Stage-4 is run with the optional evaluation flags, it
can also produce:

- ``transfer_eval.json``

This file extends the standard inference story with optional
scaled-space losses, approximate physics diagnostics, and
debug information about the rebuild path.

This is especially useful when the user wants more than just
CSV products and needs a compact machine-readable record of
how the saved-model reuse behaved scientifically.

Cross-city transfer export in Stage-5
-------------------------------------

Stage-5 expands the export story from one saved-model reuse
task to an entire benchmark matrix.

Instead of one model and one target contract, Stage-5 works
across:

- two cities,
- multiple directions,
- multiple strategies,
- optional calibration modes,
- and optional rescale modes.

Each successful transfer job can write per-job forecast CSVs
and related artifacts, while the stage also aggregates
results into benchmark tables.

The two main benchmark outputs are:

- ``xfer_results.json``
- ``xfer_results.csv``

These files are among the most important final scientific
exports in the whole workflow because they support
cross-city comparison and reporting.

How to interpret Stage-5 exports
--------------------------------

A useful interpretation rule is:

- per-job CSVs show what happened for one source-target
  transfer condition;
- ``xfer_results.csv`` shows the whole comparison table;
- ``xfer_results.json`` preserves the richer structured
  metadata for downstream inspection or automation.

This makes Stage-5 especially valuable for:

- benchmark figures,
- ablation summaries,
- transfer comparison tables,
- scientific reporting.

Deterministic build exports
---------------------------

Inference and export in GeoPrior-v3 are not limited to the
stage commands.

The **build** family adds deterministic post-processing and
artifact materialization commands that operate on existing
workflow outputs rather than retraining a model. The current
registered build commands include:

- ``full-inputs-npz``
- ``physics-payload-npz``
- ``external-validation-fullcity``
- ``external-validation-metrics``

These are useful when you want to materialize additional
artifacts from existing results in a controlled way.

What the build commands are for
-------------------------------

A practical way to understand the current build family is:

**full-inputs-npz**

Build a merged ``full_inputs.npz`` from Stage-1 split
artifacts.

**physics-payload-npz**

Build a physics payload NPZ from a model and compatible
inputs.

**external-validation-fullcity**

Build full-city external validation artifacts and associated
metrics.

**external-validation-metrics**

Build external validation metrics from Stage-1 and payload
artifacts.

These commands are useful because some downstream analysis
should be deterministic and reproducible without rerunning
the whole training or transfer pipeline.

A few practical CLI patterns
----------------------------

Inspect the help pages first:

.. code-block:: bash

   geoprior-run infer --help
   geoprior-run transfer --help
   geoprior-build full-inputs-npz --help
   geoprior-build physics-payload-npz --help
   geoprior-build external-validation-fullcity --help

Typical patterns include:

**Saved-model inference**

.. code-block:: bash

   geoprior-run infer --help

**Cross-city transfer benchmarking**

.. code-block:: bash

   geoprior-run transfer --help

**Build merged Stage-1 full inputs**

.. code-block:: bash

   geoprior-build full-inputs-npz --stage1-dir results/foo_stage1

**Build physics payload from existing artifacts**

.. code-block:: bash

   geoprior-build physics-payload-npz --help

These examples keep the page aligned with the actual CLI
surface without hard-coding a brittle invocation pattern in
the docs.

Programmatic inference
----------------------

Although the workflow is heavily CLI-oriented, some users
also work directly from Python.

Common programmatic needs include:

- loading a saved model bundle,
- running forward inference on prepared tensors,
- requesting auxiliary outputs for debugging,
- applying interval calibration,
- formatting predictions into DataFrames,
- loading and inspecting physics payloads.

For those users, it is important to remember that the Python
side should still respect the same contracts as the CLI side:

- feature order,
- target semantics,
- scaling kwargs,
- calibration policy,
- and manifest-driven interpretation.

This is why the public model-facing utilities and payload
helpers remain relevant even when the workflow is not driven
entirely from the command line.

What to inspect after export
----------------------------

After inference or export completes, inspect at least:

- the output directory,
- the evaluation CSV,
- the future forecast CSV,
- ``inference_summary.json`` when present,
- ``transfer_eval.json`` when present,
- the calibration mode,
- and any benchmark or payload files created by the run.

For Stage-5, also inspect:

- ``xfer_results.csv``
- ``xfer_results.json``
- representative per-job evaluation CSVs
- representative per-job future CSVs

You should be able to answer questions such as:

- Which target contract was used?
- Which saved model bundle was used?
- Was the export calibrated or uncalibrated?
- Were outputs written in the expected place?
- Was this same-domain inference, transfer, or warm-start?
- Were benchmark summaries aggregated successfully?

Common mistakes
---------------

**Treating a model file as self-sufficient**

A saved model often still depends on manifests, scaling
metadata, and target-contract information.

**Ignoring the Stage-1 target contract**

This is one of the most common ways to misinterpret exported
results.

**Mixing calibrated and uncalibrated outputs**

These outputs answer different questions and should not be
compared casually.

**Reading only CSVs and not the JSON summaries**

The summaries explain what the export actually did.

**Confusing same-domain inference with transfer**

A source model reused on a different target city should be
interpreted through the transfer logic, not as ordinary
in-domain evaluation.

Best practices
--------------

.. admonition:: Best practice

   Keep model artifacts, manifests, scaling metadata, and
   forecast outputs together in the same run directory.

   Reuse is much easier when the whole inference context is
   preserved.

.. admonition:: Best practice

   Treat the Stage-1 manifest as part of the export bundle.

   It defines the meaning of the target inputs and outputs.

.. admonition:: Best practice

   Record calibration policy explicitly.

   This is necessary for honest uncertainty interpretation.

.. admonition:: Best practice

   Use build commands for deterministic artifact generation
   rather than rerunning earlier stages unnecessarily.

.. admonition:: Best practice

   Inspect summary JSON files before trusting downstream CSVs.

   The tables are easier to read, but the summaries explain
   the actual workflow path.

A compact inference/export map
------------------------------

The GeoPrior-v3 inference and export story can be summarized
like this:

.. code-block:: text

   Stage-2 / Stage-3
     train or tune
     ↓
     save model bundle + manifest + scaling metadata
     ↓
   Stage-4
     load target Stage-1 contract
     + load saved model bundle
     ↓
     optionally load or fit calibrator
     ↓
     export eval/future CSVs
     + inference_summary.json
     + optional transfer_eval.json
     ↓
   Stage-5
     run cross-city benchmark jobs
     ↓
     export per-job CSVs
     + xfer_results.json
     + xfer_results.csv
     ↓
   build family
     create deterministic payload / validation artifacts
     from existing workflow outputs

Read next
---------

The best next pages after this one are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Diagnostics
      :link: diagnostics
      :link-type: doc
      :class-card: sd-shadow-sm

      See how exported summaries, benchmark files, and
      payloads should be interpreted.

   .. grid-item-card:: CLI guide
      :link: cli
      :link-type: doc
      :class-card: sd-shadow-sm card--cli

      Review the command families used for inference,
      transfer, and deterministic build export.

   .. grid-item-card:: Stage-4
      :link: stage4
      :link-type: doc
      :class-card: sd-shadow-sm

      Read the saved-model inference stage in detail.

   .. grid-item-card:: Stage-5
      :link: stage5
      :link-type: doc
      :class-card: sd-shadow-sm

      Read the transfer and benchmark export stage in detail.

.. seealso::

   - :doc:`stage2`
   - :doc:`stage3`
   - :doc:`stage4`
   - :doc:`stage5`
   - :doc:`diagnostics`
   - :doc:`cli`
   - :doc:`../scientific_foundations/data_and_units`
   - :doc:`../scientific_foundations/scaling`