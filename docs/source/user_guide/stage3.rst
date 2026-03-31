Stage-3
=======

Stage-3 is the **hyperparameter tuning stage** of the
GeoPrior-v3 workflow.

Where Stage-2 focuses on one fully specified training run,
Stage-3 explores a controlled hyperparameter search space on
top of the **Stage-1 artifact contract**. Its purpose is to
identify a stronger model configuration without rebuilding
the preprocessing pipeline from scratch.

In practical terms, Stage-3:

- loads the Stage-1 manifest and exported train/validation
  arrays,
- reconstructs the fixed dimensions and non-HP runtime
  contract,
- defines and runs a tuner search space,
- saves the best tuned model and best hyperparameters,
- evaluates the tuned winner,
- calibrates forecast intervals,
- exports tuned forecast products and diagnostics,
- and optionally writes a Stage-3 audit package.

.. note::

   The current GeoPrior wrapper and CLI clearly treat this as
   **Stage-3**. Some legacy script text still uses an older
   “Stage-2 tuning” label inherited from the earlier pipeline.
   In the documentation, it is best to describe the workflow
   according to the current packaged structure:
   **Stage-3 = tuning**.

What Stage-3 does
-----------------

Stage-3 is designed to answer a practical question:

**Given the data contract already fixed by Stage-1, which
model and physics settings work best inside a constrained
search space?**

The current implementation follows this high-level flow:

1. load the Stage-1 manifest and train/validation NPZ arrays;
2. infer the fixed, non-hyperparameter settings from Stage-1
   and the runtime config;
3. define a compact but meaningful search space;
4. create and run ``SubsNetTuner``;
5. save the best model and best hyperparameters to disk;
6. evaluate the tuned winner and export diagnostics,
   forecasts, and audit artifacts.

This makes Stage-3 the stage where GeoPrior-v3 moves from a
single configured run to a **systematic tuning workflow**.

Why Stage-3 matters
-------------------

Stage-3 matters because not every important decision should
be hard-coded before training begins.

In GeoPrior-v3, performance and behavior can depend on a
combination of:

- architectural choices,
- attention-related settings,
- hidden dimensions,
- dropout,
- physics-related switches,
- physical prior strengths,
- and selected loss-weight settings.

Stage-3 lets you explore those choices while still respecting
the workflow boundary already established by Stage-1.

Just as importantly, Stage-3 does **not** re-run
preprocessing. That means the comparison between trials is
based on a stable tensor contract, which makes the tuning
process easier to trust and easier to reproduce.

Stage-3 entry point
-------------------

Stage-3 is exposed through the GeoPrior CLI wrapper rather
than being treated only as a legacy standalone script.

At the wrapper level, Stage-3 supports a config-driven launch
style with options such as:

- ``--config``
- ``--config-root``
- ``--city``
- ``--model``
- ``--data-dir``
- ``--stage1-manifest``
- repeated ``--set KEY=VALUE`` overrides

This means the tuning stage can be launched safely from the
same CLI surface used by the other workflow stages.

.. code-block:: bash

   geoprior-run --help

Then inspect the Stage-3 tuning subcommand in your installed
environment.

.. note::

   Prefer the live CLI help output for exact invocation
   syntax. The documentation explains the workflow contract,
   while the installed CLI remains the authoritative source
   for the precise command form.

What Stage-3 consumes
---------------------

Stage-3 consumes the durable outputs produced earlier by
Stage-1.

In particular, the tuning workflow reads:

- the Stage-1 ``manifest.json``,
- training input NPZ,
- training target NPZ,
- validation input NPZ,
- validation target NPZ,
- and optional scaler / encoder metadata when available.

It then sanitizes the arrays, restores any optional zero-width
placeholders that are required by the model input contract,
and infers the dimensions needed for tuning.

This is an important design choice: Stage-3 does not tune
against ad hoc in-memory preprocessing. It tunes against the
same exported artifact boundary used elsewhere in the
workflow.

How configuration is resolved
-----------------------------

Stage-3 combines:

1. the Stage-1 configuration snapshot stored in the manifest,
2. the current live configuration from the GeoPrior config
   payload.

The tuning workflow uses the Stage-1 configuration as the
source of truth for sequence-building choices such as:

- time steps,
- forecast horizon,
- mode,
- exported feature layout,
- and other preprocessing-derived constraints.

The live config remains important for broader training and
tuning settings such as:

- epoch count,
- batch size,
- quantiles,
- physics defaults,
- search-space overrides,
- and tuner-specific behavior.

This hybrid design keeps the tuning stage aligned with what
Stage-1 actually exported while still allowing the current
run configuration to control the search procedure.

Internal structure of Stage-3
-----------------------------

The user does not need to know every implementation detail to
use Stage-3, but it helps to understand the structure.

**1. Resolve Stage-1 provenance**

Stage-3 locates a Stage-1 manifest, checks city alignment,
and loads the exported train and validation bundles.

**2. Rebuild the fixed runtime contract**

It infers the static, dynamic, future, and output dimensions
from the exported arrays and Stage-1 metadata, then rebuilds
the non-hyperparameter portion of the model contract.

**3. Finalize scaling and physics context**

The stage carries forward the Stage-1 scaling contract and
physics-sensitive metadata so that tuned trials remain
consistent with the preprocessing outputs.

**4. Define the search space**

Stage-3 creates a search space that may include model
dimensions, dropout, selected physics controls, and other
tunable quantities, either from the configured
``TUNER_SEARCH_SPACE`` or from a built-in fallback.

**5. Create and run the tuner**

It creates a ``SubsNetTuner`` instance, configures callbacks,
and launches the search over the training and validation
arrays.

**6. Persist the tuning winner**

Once the search completes, the stage serializes the best
hyperparameters and saves the best model artifact bundle.

**7. Evaluate the tuned winner**

The winning model is reloaded or rebuilt robustly if needed,
then evaluated on a forecast-oriented split.

**8. Calibrate and export**

The tuned model’s subsidence quantiles are calibrated,
forecast CSVs are written, diagnostics are exported, and a
Stage-3 audit can be recorded.

The fixed-parameter contract
----------------------------

Stage-3 does not tune everything.

A large part of the model contract remains fixed and is
derived from Stage-1 and the current configuration. This
includes, for example:

- static, dynamic, and future input dimensions,
- output dimensions,
- forecast horizon,
- sequence mode,
- quantiles,
- physics mode defaults,
- scaling kwargs,
- and selected architecture-level invariants.

This is a key point for the docs: Stage-3 is a
**constrained tuning stage**, not an unconstrained search
over arbitrary model definitions.

What the tuner searches over
----------------------------

The tuning space is meant to be meaningful without becoming
too large or too disconnected from the scientific workflow.

Depending on configuration, Stage-3 may tune parameters such
as:

- embedding dimension,
- hidden units,
- LSTM units,
- attention units,
- number of heads,
- dropout rate,
- VSN width,
- selected physics switches,
- selected physical initialization values,
- and some loss- or physics-related multipliers.

If ``TUNER_SEARCH_SPACE`` is explicitly defined in the
configuration, Stage-3 uses that as the search-space source.
Otherwise, it falls back to a built-in compact search space.

.. admonition:: Best practice

   Keep the Stage-3 search space compact and scientifically
   motivated.

   A huge search space can make tuning expensive and harder to
   interpret, especially when the model also includes
   physics-guided terms and constrained semantics inherited
   from Stage-1.

Callbacks and search behavior
-----------------------------

Stage-3 uses a callback-driven tuning loop rather than a bare
search call.

The current stage setup includes:

- early stopping,
- termination on NaNs,
- and a NaN guard that watches key training and physics
  losses.

This is important because tuning is one of the easiest places
for unstable trials to appear. The callback structure helps
the stage recover gracefully and still preserve best-so-far
artifacts when a trial or search path becomes unstable.

What Stage-3 saves before evaluation
------------------------------------

After the tuning search, Stage-3 persists the winning model
artifacts and its metadata.

Typical persisted outputs include:

- the best tuned model bundle,
- best weights,
- a lightweight tuned model manifest,
- the serialized best hyperparameters JSON,
- a compact ``tuning_summary.json``,
- and tuner log directories.

The tuned model manifest records the key dimensions and the
JSON-safe configuration needed to reconstruct the tuned model
later, including the fixed parameters and best hyperparameter
values.

These outputs are especially important because later stages
or scripts may want to reuse the tuned model without
rerunning the search.

A subtle but important point
----------------------------

Stage-3 does **not** stop at “best model found.”

After persistence, it continues by evaluating the tuned model
and exporting downstream artifacts. That means Stage-3 should
be thought of as:

- a tuning stage,
- an evaluation stage,
- and a tuned artifact export stage.

That is one reason the documentation should not describe it
as “KerasTuner only.” The current implementation goes beyond
the search itself.

How tuned evaluation works
--------------------------

Once the best tuned model is selected, Stage-3 determines
which model instance will actually be evaluated.

If the tuner directly returned a usable model object, that
model can be used.

If not, Stage-3 reloads the best model from disk or rebuilds
it from the saved manifest and the known fixed parameters.
This makes the workflow more robust and better aligned with
later inference or transfer use cases.

After that, Stage-3 forecasts on the selected split, handles
quantile tensor canonicalization when needed, applies
interval calibration, and formats the tuned forecast
products.

Calibration and tuned forecasts
-------------------------------

Stage-3 explicitly supports tuned forecast calibration.

Once the tuned model produces quantile outputs, the stage can
apply interval calibration to the subsidence forecast tensor
before formatting evaluation and future forecast tables.

Typical tuned forecast outputs include:

- a calibrated evaluation forecast CSV,
- a future forecast CSV,
- and an evaluation diagnostics JSON file associated with the
  tuned forecasts.

This is an important workflow distinction:

- the tuner chooses the **winning model configuration**,
- then Stage-3 produces the **usable tuned forecast outputs**
  that later analysis can consume.

Physics diagnostics and payloads
--------------------------------

Stage-3 also evaluates the tuned winner from the physics
side.

After recompiling the tuned model for evaluation where
needed, the stage can:

- run Keras evaluation on the tuned model,
- extract physics diagnostics such as epsilon-style terms,
- export a tuned physics payload,
- and attach these into the Stage-3 audit and ablation
  record.

This keeps the tuned workflow consistent with the broader
GeoPrior philosophy: a strong tuned model is not only one
that predicts well, but also one whose physics-facing
diagnostics remain interpretable.

Typical Stage-3 artifacts
-------------------------

A successful Stage-3 run will often leave behind artifacts
such as:

- ``kt_logs/`` tuner logs,
- ``<city>_GeoPrior_best.keras`` or a corresponding saved
  model directory,
- ``<city>_GeoPrior_best.weights.h5``,
- ``<city>_GeoPrior_best_hps.json``,
- ``<city>_GeoPrior_best_manifest.json``,
- ``tuning_summary.json``,
- tuned calibrated evaluation forecast CSV,
- tuned future forecast CSV,
- ``eval_diagnostics_tuned.json``,
- a tuned metrics JSON payload,
- a tuned physics payload NPZ,
- and optional Stage-3 audit outputs.

These are the main files you should expect when tuning has
completed successfully and the downstream export path has
also run.

What to inspect after Stage-3 completes
---------------------------------------

Do not treat Stage-3 as complete just because the tuner
finished.

Inspect at least the following:

- the Stage-3 run directory,
- the saved best hyperparameters JSON,
- the tuned model manifest,
- ``tuning_summary.json``,
- the tuner logs,
- the tuned evaluation forecast CSV,
- the tuned future forecast CSV,
- the tuned evaluation diagnostics JSON,
- the tuned physics payload,
- and any audit output written by the stage.

You should be able to answer questions such as:

- Which Stage-1 manifest did this tuning run use?
- Which dimensions were fixed from the exported arrays?
- Which hyperparameters were actually selected?
- Did the tuned model save as a full model, weights, or TF
  export bundle?
- Were tuned forecast CSVs written successfully?
- Were the physics diagnostics and audit artifacts saved?

Common Stage-3 mistakes
-----------------------

**Treating Stage-3 as independent from Stage-1**

Stage-3 depends directly on Stage-1 exports. If the Stage-1
manifest is stale or mismatched, the tuning stage can still
become misleading.

**Tuning too much at once**

A very wide search space can make interpretation harder and
can bury scientifically useful comparisons under too much
variation.

**Ignoring the tuned export products**

The search result alone is not enough. The tuned forecasts,
diagnostics, and saved manifests are part of the real output
of the stage.

**Forgetting that tuned models must still be interpretable**

A lower validation loss is not the only criterion that
matters in a physics-guided workflow.

**Confusing tuned-model artifacts with Stage-2 artifacts**

The tuned winner has its own model bundle and associated
summary files. Keep those separated clearly from standard
Stage-2 training outputs.

Best practices
--------------

.. admonition:: Best practice

   Pass an explicit Stage-1 manifest when you care about
   reproducibility.

   This makes it clear exactly which preprocessing run the
   tuning stage is built on.

.. admonition:: Best practice

   Keep the tuned model bundle, best hyperparameters, and
   tuning summary together in the same run directory.

   These files form the minimum reproducible record of the
   tuning outcome.

.. admonition:: Best practice

   Inspect the tuned forecast CSVs and diagnostics, not only
   the tuner logs.

   The tuning stage is designed to produce downstream
   scientific outputs, not only search metadata.

.. admonition:: Best practice

   Treat tuned physics diagnostics as part of model selection.

   In GeoPrior-v3, a tuned winner should still be reviewed for
   physical plausibility, not only for raw supervised metrics.

A compact Stage-3 map
---------------------

The Stage-3 workflow can be summarized like this:

.. code-block:: text

   Stage-1 manifest + train/val NPZ
        ↓
   infer fixed dims + runtime contract
        ↓
   define compact hyperparameter search space
        ↓
   create SubsNetTuner
        ↓
   run tuning with callbacks
        ↓
   save best model + best_hps + tuning_summary
        ↓
   reload or rebuild tuned winner if needed
        ↓
   calibrate tuned quantile outputs
        ↓
   export tuned eval/future forecast CSVs
        ↓
   save diagnostics, physics payload, and audit outputs

Read next
---------

The next best pages after Stage-3 are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Stage-4
      :link: stage4
      :link-type: doc
      :class-card: sd-shadow-sm

      Move from tuning into clean inference and tuned-model
      reuse.

   .. grid-item-card:: Diagnostics
      :link: diagnostics
      :link-type: doc
      :class-card: sd-shadow-sm

      Understand how to interpret the tuned diagnostics,
      audits, and exported physics payloads.

   .. grid-item-card:: Inference and export
      :link: inference_and_export
      :link-type: doc
      :class-card: sd-shadow-sm

      See how tuned model bundles and tuned forecasts are used
      downstream.

   .. grid-item-card:: CLI guide
      :link: cli
      :link-type: doc
      :class-card: sd-shadow-sm card--cli

      Go from the stage narrative to the full command
      interface.

.. seealso::

   - :doc:`workflow_overview`
   - :doc:`stage1`
   - :doc:`stage2`
   - :doc:`stage4`
   - :doc:`configuration`
   - :doc:`cli`
   - :doc:`diagnostics`
   - :doc:`inference_and_export`
   - :doc:`../scientific_foundations/scaling`
   - :doc:`../scientific_foundations/identifiability`