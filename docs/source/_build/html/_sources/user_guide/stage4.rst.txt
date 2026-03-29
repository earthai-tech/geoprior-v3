Stage-4
=======

Stage-4 is the **clean inference, transfer, and
forecast-export stage** of the GeoPrior-v3 workflow.

It is the stage where a previously trained or tuned model is
reused on a selected dataset split or on custom NPZ inputs,
without rerunning Stage-2 training or Stage-3 tuning. The
current implementation describes Stage-4 as
**deterministic/probabilistic inference for GeoPriorSubsNet**
and explicitly positions it as the **inference counterpart of
the Stage-2/Stage-3 workflows**. It is written to preserve
I/O, scaling-aware behavior, and Keras 2/3-compatible
model loading while keeping inference reproducible. 

In practical terms, Stage-4:

- loads a Stage-1 manifest as the **target-domain contract**,
- resolves a saved model bundle from Stage-2 or Stage-3,
- optionally reloads interval calibration,
- runs inference on test, validation, training, or custom
  NPZ inputs,
- formats calibrated or uncalibrated forecast CSVs,
- writes an ``inference_summary.json``,
- and can optionally export a ``transfer_eval.json`` with
  scaled-space losses and approximate physics diagnostics. 
  

What Stage-4 does
-----------------

Stage-4 exists so that model reuse is treated as a
first-class workflow step rather than as a loose collection
of ad hoc notebook calls.

Its high-level responsibilities are:

- consuming the Stage-1 artifact contract for the **target**
  dataset;
- loading or rebuilding an inference-capable model from a
  saved bundle;
- resolving optional best-hyperparameter context near the
  model;
- optionally reusing or refitting an interval calibrator;
- predicting on a chosen split or custom NPZ bundle;
- canonicalizing quantile tensor layout before formatting;
- exporting evaluation and future forecast CSVs;
- computing physical-unit summary metrics where possible;
- optionally computing scaled-space evaluation losses and
  physics diagnostics.

This makes Stage-4 the main bridge between **saved model
artifacts** and **portable, inspectable forecast products**.

Why Stage-4 matters
-------------------

Stage-4 matters because trained models are only useful if
they can be reused cleanly and consistently.

In GeoPrior-v3, model reuse is not trivial, because the
correct interpretation of a saved model depends on:

- the Stage-1 feature contract,
- target semantics,
- coordinate handling,
- scaling metadata,
- quantile layout,
- and, in some cases, calibration policy.

A raw ``.keras`` file alone is often not enough. Stage-4 is
therefore designed to reconstruct the correct inference
context rather than assuming that every artifact can be used
in isolation.

This is especially important for **transfer-style** use
cases, where a model trained in one source run may be applied
to a different target Stage-1 contract. The legacy script
explicitly includes examples for same-domain inference,
zero-shot cross-city transfer, source-calibrator reuse, and
target-validation recalibration. 

Stage-4 entry point
-------------------

Stage-4 is exposed through the newer GeoPrior wrapper while
preserving the existing inference implementation underneath.

The wrapper is built to make Stage-4 consistent with the
modern ``geoprior.cli`` dispatcher pattern. It supports:

- using the installed ``nat.com/config.py`` as-is,
- installing a user-supplied config file before running,
- applying one-off ``--set KEY=VALUE`` overrides,
- pointing at a specific Stage-1 manifest,
- and forwarding all remaining inference arguments to the
  original Stage-4 CLI unchanged. 

At the wrapper level, Stage-4 supports options such as:

- ``--config``
- ``--config-root``
- ``--city``
- ``--model``
- ``--data-dir``
- ``--stage1-manifest``
- repeated ``--set KEY=VALUE`` overrides. 

A good first move is always:

.. code-block:: bash

   geoprior-run --help

Then inspect the Stage-4-specific subcommand in your
installed environment.

.. note::

   The wrapper is intentionally thin. Unknown arguments are
   forwarded to the legacy inference CLI, so the installed
   help output remains the best source for the exact command
   form. 

What Stage-4 consumes
---------------------

Stage-4 needs two main inputs:

1. a **Stage-1 manifest** that defines the target-domain
   contract;
2. a **saved model artifact** that can be reused for
   inference.

The legacy CLI accepts the target contract through either:

- ``--manifest``
- ``--stage1-dir``

and, if neither is supplied, it can auto-discover a Stage-1
manifest using the same general policy as the earlier
stages. 

The model artifact is required through:

- ``--model-path``

and may point to:

- a full ``.keras`` model,
- a ``*_best_savedmodel/`` TensorFlow export directory,
- or a run directory containing the relevant bundle files. 

Which dataset Stage-4 runs on
-----------------------------

Stage-4 supports multiple inference datasets through the
legacy CLI:

- ``test``
- ``val``
- ``train``
- ``custom``. 

For standard splits, Stage-4 uses the NPZ artifacts recorded
in the Stage-1 manifest. For ``custom``, it expects an
explicit ``--inputs-npz`` path and can optionally use
``--targets-npz`` if evaluation metrics should also be
computed. 

This is an important distinction:

- standard split inference is **manifest-driven**;
- custom inference is **artifact-path-driven**.

How the Stage-1 contract is used
--------------------------------

Stage-4 treats the Stage-1 manifest as the authoritative
description of the **target** domain.

Once the manifest is loaded, Stage-4 derives:

- city and model identity,
- mode,
- forecast horizon,
- forecast start year,
- train end year,
- quantiles,
- output dimensions,
- and encoder / scaler metadata such as coordinate scalers
  and target scaling summaries. 

This is why Stage-4 can support transfer use cases. The model
artifact may come from one source run, but the prediction
formatting and target interpretation are driven by the
**target Stage-1 contract**.

Model bundle resolution
-----------------------

Stage-4 does not assume a single saved-model format.

Instead, it resolves a model bundle using Stage-2 naming
conventions and supports three common cases:

- user points directly to a TensorFlow SavedModel export;
- user points directly to a ``.keras`` file;
- user points to a run directory that contains the expected
  artifacts. 

The bundle resolution step looks for items such as:

- the run directory,
- a full-model ``.keras`` file,
- best weights,
- a TensorFlow export directory,
- and ``model_init_manifest.json``. 

This helps Stage-4 rebuild the correct inference-capable
model even when the full model file alone is not enough.

How Stage-4 rebuilds the model
------------------------------

Stage-4 prefers to reconstruct the model from the most
faithful metadata available.

The current implementation explicitly prefers
``model_init_manifest.json`` because it is treated as the
most faithful JSON-safe snapshot of model initialization
parameters. It then merges in Stage-1 contract details and,
when helpful, nearby best-hyperparameter files associated
with a tuned model. 

The builder reconstruction logic carries forward important
runtime semantics such as:

- static, dynamic, and future input dimensions,
- output dimensions,
- forecast horizon,
- quantiles,
- PDE mode,
- bounds mode,
- residual method,
- scaling kwargs,
- groundwater semantics,
- subsidence kind,
- SI affine factors,
- and GeoPrior physical parameter settings. 

This is what makes Stage-4 more robust than a plain
``keras.models.load_model(...)`` call.

Prediction modes
----------------

Stage-4 supports several practical reuse patterns.

**1. Same-domain inference**

A model trained or tuned on one Stage-1 run is reused on the
same target-domain artifact contract.

**2. Zero-shot transfer**

A model from source domain A is evaluated directly against
target domain B without fitting a new calibrator.

**3. Source-calibrator transfer**

A source-domain interval calibrator is reused together with
the source model.

**4. Target-validation recalibration**

A new interval calibrator is fitted on the target validation
set and then applied during inference. 

These patterns are explicitly shown in the legacy examples,
which makes Stage-4 one of the main places where
cross-domain behavior becomes operational in the workflow. 

How calibration is resolved
---------------------------

Stage-4 resolves interval calibration in a clear order.

The calibrator helper currently supports:

1. loading a calibrator from the **source model directory**
   when ``--use-source-calibrator`` is enabled;
2. loading a calibrator from an explicit ``--calibrator``
   path;
3. fitting a new calibrator on the validation set when
   ``--fit-calibrator`` is enabled and validation data are
   available. 

This keeps the calibration policy explicit instead of hidden
inside the prediction path.

.. admonition:: Best practice

   Be explicit about calibration policy during transfer runs.

   Zero-shot transfer, source-calibrator reuse, and
   target-validation recalibration answer different scientific
   questions. They should not be treated as interchangeable.

Quantile handling and output formatting
---------------------------------------

After prediction, Stage-4 uses the eval-capable GeoPrior
model to extract the output tensors and then canonicalizes
quantile layouts before applying interval calibration. The
implementation explicitly normalizes the quantile layout into
a BHQO-style convention before calibration is applied. 

Once outputs are ready, Stage-4 prepares a formatting dict
and calls the same forecast formatting utility used elsewhere
in the pipeline. The formatter writes:

- an evaluation CSV, when actuals are available;
- a future forecast CSV, when future outputs are defined. 

The base output naming follows the pattern:

- ``<city>_<model>_inference_<dataset>_H<horizon>_eval.csv``
- ``<city>_<model>_inference_<dataset>_H<horizon>_future.csv``

with a ``_calibrated`` suffix added when a calibrator is
active. 

Where Stage-4 writes outputs
----------------------------

Stage-4 writes its results under the **target Stage-1 run
directory** to avoid confusion between source and target
artifacts.

The current implementation places outputs under a timestamped
subdirectory of:

.. code-block:: text

   <target_stage1_run_dir>/inference/<timestamp>/

This is a good design choice for transfer work, because it
keeps the inference artifacts grouped with the target-domain
contract rather than with the source training run. 

Primary Stage-4 outputs
-----------------------

The legacy script header highlights three key outputs when
available:

- calibrated forecast CSVs for evaluation and future use,
- ``inference_summary.json``,
- ``transfer_eval.json``. 

In practice, these are the main artifacts users should expect
after a successful Stage-4 run.

**Evaluation and future CSVs**

These are the main tabular forecast products written by
``format_and_forecast(...)``. They can be calibrated or
uncalibrated depending on the inference mode. 

**inference_summary.json**

This summary records:

- dataset name,
- city,
- model,
- horizon,
- mode,
- whether the run was calibrated,
- quantiles when relevant,
- output CSV paths,
- and, when targets and scaler info are available, physical-
  unit metrics such as coverage, sharpness, point metrics,
  and per-horizon metrics. 

**transfer_eval.json**

When optional evaluation diagnostics are requested, this file
stores:

- dataset name,
- CSV paths,
- scaled-space Keras evaluation results,
- approximate physics diagnostics,
- the inference summary,
- and small debug details about rebuild behavior.

Optional evaluation behavior
----------------------------

Stage-4 can do more than prediction-only inference.

The legacy CLI supports:

- ``--eval-losses`` to compute scaled-space MAE/MSE-style
  losses using ``model.evaluate()``;
- ``--eval-physics`` to compute approximate
  ``epsilon_prior`` / ``epsilon_cons``-style diagnostics via
  ``model.evaluate_physics()``. 

If these diagnostics are requested and the model is not yet
compiled in the needed way, Stage-4 attempts to recompile it
for evaluation using the Stage-1 manifest and nearby best
hyperparameters. 

This keeps inference reusable while still allowing optional
scientific audits.

Including groundwater outputs
-----------------------------

By default, Stage-4 focuses on subsidence-facing outputs in
the formatted CSV path. The legacy CLI also supports an
``--include-gwl`` switch so groundwater predictions can be
carried into the formatted forecast products when available. 

This is useful when the inference consumer needs both target
families rather than only the main subsidence export.

What to inspect after Stage-4 completes
---------------------------------------

Do not stop at “the prediction ran.”

After a Stage-4 run, inspect at least:

- the target inference directory,
- the evaluation CSV,
- the future forecast CSV,
- ``inference_summary.json``,
- ``transfer_eval.json`` when evaluation flags were enabled,
- whether the run was calibrated or not,
- and whether the target Stage-1 contract matches what you
  intended to evaluate.

You should be able to answer questions such as:

- Which Stage-1 manifest defined the target domain?
- Which saved model bundle was actually used?
- Was the run calibrated, and how?
- Were CSVs written for evaluation, future, or both?
- Were physical-unit metrics available?
- Were scaled-space losses or physics diagnostics requested?

Common Stage-4 mistakes
-----------------------

**Mixing source and target semantics**

In transfer settings, the source model run and the target
Stage-1 manifest serve different purposes. The model comes
from one place, but the formatting and target semantics come
from the target contract.

**Treating calibration as an afterthought**

Calibration policy should be explicit. Source-calibrator
reuse and target recalibration are not the same thing.

**Assuming every model artifact is self-sufficient**

A model file may still need its init manifest, best weights,
or nearby hyperparameter metadata for robust reuse.

**Ignoring quantile layout issues**

Stage-4 explicitly canonicalizes quantile outputs before
calibration. Users should not assume arbitrary saved-model
outputs are already in the right layout for export.

**Reading only the CSVs and not the summaries**

The JSON summaries contain the metadata that explains what
the Stage-4 run actually did.

Best practices
--------------

.. admonition:: Best practice

   Treat the target Stage-1 manifest as the authoritative
   inference contract.

   This is especially important for transfer and custom reuse
   scenarios.

.. admonition:: Best practice

   Keep calibration policy explicit in run notes and saved
   metadata.

   Whether a run is zero-shot, source-calibrated, or
   target-calibrated strongly affects interpretation.

.. admonition:: Best practice

   Inspect ``inference_summary.json`` after every serious
   inference run.

   It is the quickest compact record of what was inferred,
   what was exported, and which metrics were available.

.. admonition:: Best practice

   Use ``transfer_eval.json`` for debug and scientific audit,
   not only for deployment.

   It preserves scaled-space and physics-facing information
   that may not be visible in the forecast CSVs.

A compact Stage-4 map
---------------------

The Stage-4 workflow can be summarized like this:

.. code-block:: text

   target Stage-1 manifest
        +
   saved model bundle (.keras / TF export / run dir)
        ↓
   resolve target contract + model bundle
        ↓
   rebuild or load inference-capable model
        ↓
   choose dataset: test / val / train / custom
        ↓
   optionally load or fit interval calibrator
        ↓
   predict + canonicalize quantile layout
        ↓
   apply calibration if enabled
        ↓
   format evaluation / future forecast CSVs
        ↓
   write inference_summary.json
        ↓
   optionally compute eval losses + physics diagnostics
        ↓
   write transfer_eval.json

Read next
---------

The next best pages after Stage-4 are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Stage-5
      :link: stage5
      :link-type: doc
      :class-card: sd-shadow-sm

      Continue from inference into the final workflow stage.

   .. grid-item-card:: Inference and export
      :link: inference_and_export
      :link-type: doc
      :class-card: sd-shadow-sm

      Read the broader reusable inference and export guidance
      that complements Stage-4.

   .. grid-item-card:: Diagnostics
      :link: diagnostics
      :link-type: doc
      :class-card: sd-shadow-sm

      See how to interpret exported metrics, audits, and
      physics-facing diagnostic outputs.

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
   - :doc:`stage2`
   - :doc:`stage3`
   - :doc:`stage5`
   - :doc:`configuration`
   - :doc:`cli`
   - :doc:`diagnostics`
   - :doc:`inference_and_export`
   - :doc:`../scientific_foundations/scaling`
   - :doc:`../scientific_foundations/data_and_units`