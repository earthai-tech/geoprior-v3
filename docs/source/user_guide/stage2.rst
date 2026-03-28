Stage-2
=======

Stage-2 is the **training, calibration, and forecast-export
stage** of the GeoPrior-v3 workflow.

Where Stage-1 establishes the preprocessing and tensor
handoff contract, Stage-2 consumes those exported artifacts,
rebuilds the training-ready workflow state, trains the
selected model variant, evaluates the resulting model,
calibrates interval forecasts, and exports the first
model-facing scientific outputs.

In practice, Stage-2 is where the workflow moves from
**prepared data artifacts** to **model execution and
forecasting artifacts**.

What Stage-2 does
-----------------

Stage-2 is responsible for the following high-level tasks:

- loading the Stage-1 manifest and NPZ bundles;
- validating the Stage-1 → Stage-2 handshake;
- resolving the hybrid configuration used for training;
- rebuilding train and validation datasets;
- constructing and compiling the model;
- training with callbacks, checkpoints, and logs;
- saving model bundles and training summaries;
- reloading a clean inference model;
- calibrating predictive intervals on the validation set;
- generating forecast CSV outputs and diagnostics;
- exporting physics payloads and physical parameter summaries.

This makes Stage-2 the main bridge between **data
preparation** and **scientific forecast outputs**.

Why Stage-2 matters
-------------------

Stage-2 is the first stage where three things come together
at once:

- the **Stage-1 artifact contract**,
- the **runtime configuration**,
- and the **physics-guided model behavior**.

That combination is versatile, but it also means Stage-2 is a
sensitive point in the workflow. A training run can complete
successfully even when the scientific assumptions are wrong,
for example because of:

- unit mismatches,
- wrong coordinate conventions,
- inconsistent feature ordering,
- incorrect groundwater semantics,
- stale artifacts from another run,
- or badly chosen physics settings.

For that reason, Stage-2 is not just “model.fit()”. It is a
structured training stage with explicit audits, manifests,
and export logic.

Stage-2 entry point
-------------------

Stage-2 is exposed through the modern CLI wrapper rather than
only through the older legacy script body.

The wrapper exists so that Stage-2 can be dispatched safely
from the GeoPrior CLI while still reusing the established
training pipeline underneath.

At the wrapper level, Stage-2 supports a configuration-driven
launch style with options such as:

- ``--config``
- ``--config-root``
- ``--city``
- ``--model``
- ``--data-dir``
- ``--stage1-manifest``
- repeated ``--set KEY=VALUE`` overrides

A common pattern is to start with the live CLI help:

.. code-block:: bash

   geoprior-run --help

and then inspect the Stage-2-specific help in your installed
environment.

.. note::

   The exact command form should always be taken from the
   current local CLI help output. The documentation explains
   the workflow contract, but the installed command surface is
   the authoritative reference for invocation syntax.

What Stage-2 consumes from Stage-1
----------------------------------

Stage-2 does not rebuild preprocessing from scratch. It is
designed to **consume Stage-1 artifacts directly**.

The required inputs come from the Stage-1 manifest and its
exported NPZ bundles. In particular, Stage-2 expects the
Stage-1 handoff to provide at least the core training and
validation bundles.

Conceptually, Stage-2 needs:

- the Stage-1 ``manifest.json``,
- training input arrays,
- training target arrays,
- validation input arrays,
- validation target arrays.

Depending on the Stage-1 run, additional artifacts may also
be available and used later in the workflow, such as:

- test NPZ bundles,
- future-sequence bundles,
- scaler and encoder metadata,
- split summaries,
- optional holdout artifacts.

If the user provides ``--stage1-manifest``, Stage-2 uses that
specific Stage-1 manifest explicitly. Otherwise, it resolves
the most appropriate Stage-1 manifest from the results area.

Stage-1 → Stage-2 handshake
---------------------------

One of the most important Stage-2 responsibilities is to
validate the handoff coming from Stage-1.

This includes checks such as:

- the manifest really corresponds to Stage-1,
- the city in the manifest matches the active configuration,
- the required NPZ bundle paths exist,
- the tensor last-dimension sizes match the recorded feature
  lists,
- the sequence horizon and time-step settings are consistent,
- the scaling metadata is present and usable,
- and the coordinate treatment is compatible with the model
  setup.

This handshake is not a cosmetic extra. It is one of the main
defenses against silent workflow drift.

.. admonition:: Best practice

   Do not treat Stage-1 and Stage-2 as loosely connected
   scripts.

   Stage-2 is designed to trust Stage-1 as the authoritative
   preprocessing boundary, but only after the handshake audit
   confirms that the exported tensors, feature names, and
   scaling information remain internally consistent.

How configuration is resolved
-----------------------------

Stage-2 combines two configuration sources:

1. the live or installed GeoPrior configuration,
2. the Stage-1 manifest configuration snapshot.

These are not merged blindly.

The training workflow follows a **hybrid configuration**
pattern in which Stage-1 provenance remains authoritative for
the parts of the workflow that must reflect what was actually
used during preprocessing, while the live configuration can
still control training- and physics-facing options.

In practical terms, this usually means:

- Stage-1 remains the source of truth for what was exported;
- training and physics choices can still come from the live
  config;
- overlapping keys are resolved intentionally rather than by
  accidental overwrite.

This is important because Stage-2 should not silently train a
model under assumptions that no longer match the tensors it
received.

Internal structure of Stage-2
-----------------------------

The user does not need to know every helper function inside
the training script, but it is useful to understand the
overall flow.

**1. Resolve manifest and hybrid config**

Stage-2 locates or receives a Stage-1 manifest, verifies that
it is valid, and resolves the effective configuration that
will control the training run.

**2. Load NPZ bundles and metadata**

The stage reads the exported train and validation arrays,
loads scaling and encoder metadata, and reconstructs the
feature-space contract needed by the model.

**3. Finalize scaling and semantics**

Groundwater semantics, coordinate treatment, SI conversion
rules, and feature naming are consolidated into the final
``scaling_kwargs`` and related runtime metadata.

**4. Run the Stage-2 handshake audit**

If auditing is enabled, the stage records a Stage-2 handshake
report before the model is built.

**5. Build and compile the model**

The configured model class is instantiated with the resolved
input dimensions, forecast horizon, quantiles, PDE mode, and
identifiability regime.

**6. Train with callbacks and checkpoints**

Training uses datasets built from the Stage-1 tensors and
runs with checkpointing, early stopping, CSV logging, NaN
termination, and optional scheduling of physics-related
weights.

**7. Save model artifacts**

Stage-2 writes multiple model-facing artifacts, including
weights, best-model bundles, architecture JSON,
training summaries, manifests, and plots.

**8. Reload inference model and calibrate intervals**

A clean inference model is loaded, validation-based interval
calibration is fitted, and interval factors are saved.

**9. Forecast, evaluate, and export**

The trained model is used to produce forecast tables,
calibrated forecast tables, metrics JSON files, and physics
payload exports.

Model construction in Stage-2
-----------------------------

Stage-2 builds the configured model using the dimensions
derived from the Stage-1 tensors and the resolved runtime
configuration.

The stage is primarily designed around the GeoPrior model
family and can support different model flavors depending on
the active configuration. The model is instantiated with
items such as:

- static, dynamic, and future feature dimensions,
- output dimensions,
- forecast horizon,
- quantiles,
- PDE mode,
- identifiability regime,
- finalized scaling kwargs,
- and model-specific physical parameters.

This is why Stage-2 should never be run against stale or
ambiguous Stage-1 artifacts. The model structure is tied
directly to what Stage-1 exported.

Compile-time behavior
---------------------

Once the model is built, Stage-2 compiles it using the
resolved supervised losses, metrics, and physics-related loss
weights.

This stage may include:

- supervised losses for subsidence and groundwater targets,
- quantile-aware objectives when probabilistic outputs are
  enabled,
- physics-related residual penalties,
- optional prior, smoothness, or bounds terms,
- and model-flavor-specific compile adjustments.

Stage-2 therefore sits at the point where the workflow
switches from **prepared tensors** to **scientific training
behavior**.

Training behavior and callbacks
-------------------------------

Stage-2 does not just train and discard the result. It uses a
tracked training workflow with persistent outputs.

Typical callback behavior includes:

- best full-model checkpoint saving,
- best weights saving,
- early stopping,
- CSV training log export,
- termination on NaNs,
- and optional offset scheduling for physics-related terms.

This makes the stage more reproducible and easier to inspect
after a run completes.

Artifacts produced during training
----------------------------------

Stage-2 writes several important training-facing artifacts.

**Model init manifest**

A lightweight ``model_init_manifest.json`` is created so the
model can be reconstructed cleanly for later inference.

**Best model bundles**

Stage-2 saves checkpointed best-model artifacts, usually in
both full-model and weights-oriented forms.

**Final model bundle**

A final Keras model export is also written when possible.

**Architecture JSON**

The model architecture is saved separately for inspection or
reconstruction workflows.

**Training summary**

A JSON summary captures the best epoch, tracked metrics,
compile settings, environment details, and key hyperparameter
information.

**Run manifest**

Stage-2 writes a small run manifest that downstream stages or
scripts can read without parsing the full training summary.

**Scaling snapshot**

The finalized ``scaling_kwargs`` are saved explicitly so the
training-time interpretation can be audited later.

These artifacts make Stage-2 durable. The stage is not merely
a transient training step.

Forecasting and calibration
---------------------------

After training, Stage-2 performs a second critical role:
**forecast preparation and calibration**.

The workflow first reloads or rebuilds a clean inference
model, then fits an interval calibrator on the validation
data before formatting forecast DataFrames.

This ordering is important.

The intended logic is:

1. enforce physics during training,
2. fit interval calibration on validation outputs,
3. apply calibration to forecast quantiles,
4. export calibrated and uncalibrated forecast products.

By doing this, Stage-2 keeps the distinction clear between:

- **physics-guided training behavior**, and
- **post-training interval calibration behavior**.

Forecast target split
---------------------

When available, Stage-2 forecasts on the exported test split.
If a test NPZ is not available, it can fall back to the
validation split for forecast-oriented export.

This gives the stage a practical default behavior while still
remaining compatible with runs that do not yet expose a
dedicated test handoff.

Forecast and evaluation exports
-------------------------------

Stage-2 writes a rich set of evaluation-facing outputs.

Typical exports include:

- evaluation forecast CSV,
- future forecast CSV,
- calibrated evaluation forecast CSV,
- calibrated future forecast CSV,
- calibration statistics JSON,
- evaluation diagnostics JSON,
- calibrated evaluation diagnostics JSON,
- interval calibration factors,
- and physics payload exports.

These outputs are what make Stage-2 more than a pure training
step. It is the stage where the workflow begins to produce
scientifically interpretable forecast products.

Physics diagnostics and payloads
--------------------------------

Stage-2 also evaluates and exports physics-facing
diagnostics.

Depending on the run and model behavior, this may include:

- evaluation-time physics diagnostics such as epsilon-style
  residual summaries,
- physics payload export in NPZ form,
- extracted physical parameter tables,
- and optional plots of physical parameter values.

This is especially useful for later scientific analysis,
because it allows the user to inspect not only forecast
quality but also the internal physical consistency behavior
of the model.

Typical files you should expect after Stage-2
---------------------------------------------

A successful Stage-2 run will often leave behind files such
as:

- ``model_init_manifest.json``
- training log CSV
- best-model checkpoint bundle
- best weights file
- final Keras model
- architecture JSON
- training summary JSON
- Stage-2 run manifest
- ``scaling_kwargs.json``
- training history plots
- physical parameter CSV
- interval factor ``.npy`` file
- evaluation and future forecast CSVs
- calibrated forecast CSVs
- evaluation diagnostics JSON
- physics payload NPZ

The exact file names depend on the run naming convention, but
the presence of this artifact family is a good sign that
Stage-2 completed the full training-and-export workflow.

What to inspect after Stage-2 completes
---------------------------------------

Do not move forward immediately after training finishes.
Inspect the outputs first.

At minimum, review:

- the Stage-2 run directory,
- the training summary JSON,
- the run manifest,
- the CSV training log,
- the best-model and final-model files,
- the saved scaling kwargs,
- the forecast CSVs,
- the calibrated forecast products,
- the evaluation diagnostics JSON,
- and the physics payload export.

You should be able to answer questions such as:

- Did Stage-2 use the intended Stage-1 manifest?
- Do the recorded feature dimensions match expectations?
- Did the best epoch occur reasonably early or late?
- Were calibrated forecast outputs written?
- Did the evaluation diagnostics look plausible?
- Was a physics payload exported successfully?

Common Stage-2 mistakes
-----------------------

The most common Stage-2 problems are not purely algorithmic.

**Using the wrong Stage-1 manifest**

This can silently train the model on artifacts from another
city, run, or feature contract.

**Ignoring city mismatch**

Stage-2 is designed to reject obvious city mismatch between
the active config and the Stage-1 manifest. Do not work
around that unless you are intentionally debugging.

**Treating Stage-2 as only training**

Stage-2 is also a calibration and export stage. A run that
produces only model weights but no clean forecast artifacts
is usually incomplete.

**Skipping artifact inspection**

A completed ``fit()`` call does not guarantee trustworthy
forecasting outputs.

**Forgetting that scaling semantics matter**

Stage-2 carries forward the SI conversion and coordinate
rules from Stage-1. If those assumptions are wrong, later
physics interpretation can also be wrong.

Best practices
--------------

.. admonition:: Best practice

   Treat the Stage-1 manifest as the authoritative handoff
   contract.

   If you need to force a specific preprocessing run, pass the
   exact manifest explicitly rather than relying on accidental
   auto-resolution.

.. admonition:: Best practice

   Inspect ``training_summary.json`` and the Stage-2 run
   manifest after every serious run.

   These are the fastest way to confirm what was actually
   trained, which settings were used, and which artifacts
   were saved.

.. admonition:: Best practice

   Review the calibrated forecast exports, not only the raw
   model outputs.

   Stage-2 is explicitly designed to calibrate interval
   forecasts before downstream interpretation.

.. admonition:: Best practice

   Keep Stage-2 outputs grouped by run directory.

   Mixing model bundles, forecast CSVs, or diagnostics from
   different Stage-2 runs is a common source of silent
   confusion.

A compact Stage-2 map
---------------------

The Stage-2 workflow can be summarized like this:

.. code-block:: text

   Stage-1 manifest + NPZ bundles
        ↓
   resolve hybrid config
        ↓
   validate Stage-1 → Stage-2 handshake
        ↓
   build datasets + finalize scaling semantics
        ↓
   build and compile model
        ↓
   train with checkpoints and logs
        ↓
   save model bundles + summaries + manifests
        ↓
   reload clean inference model
        ↓
   fit interval calibration on validation set
        ↓
   forecast + evaluate + export calibrated outputs
        ↓
   export physics payload and diagnostics

Read next
---------

The next pages after Stage-2 are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Stage-3
      :link: stage3
      :link-type: doc
      :class-card: sd-shadow-sm

      Continue from training into the next downstream workflow
      stage.

   .. grid-item-card:: Diagnostics
      :link: diagnostics
      :link-type: doc
      :class-card: sd-shadow-sm

      Learn how to interpret the diagnostics and audit outputs
      produced during later workflow steps.

   .. grid-item-card:: Inference and export
      :link: inference_and_export
      :link-type: doc
      :class-card: sd-shadow-sm

      Review the broader inference and export logic that
      builds on the artifacts produced here.

   .. grid-item-card:: CLI guide
      :link: cli
      :link-type: doc
      :class-card: sd-shadow-sm card--cli
      :data-preview: cli.png
      :data-preview-label: GeoPrior command-line interface

      Move from the stage narrative to the full command
      reference.

.. seealso::

   - :doc:`workflow_overview`
   - :doc:`stage1`
   - :doc:`stage3`
   - :doc:`configuration`
   - :doc:`cli`
   - :doc:`diagnostics`
   - :doc:`inference_and_export`
   - :doc:`../scientific_foundations/data_and_units`
   - :doc:`../scientific_foundations/scaling`
   - :doc:`../scientific_foundations/identifiability`