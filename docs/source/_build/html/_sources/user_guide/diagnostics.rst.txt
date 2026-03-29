Diagnostics
===========

Diagnostics are a first-class part of GeoPrior-v3.

In a physics-guided workflow, a run is not trustworthy merely
because it finished. A useful diagnostic workflow must help
you answer questions such as:

- Did the stage use the intended inputs and configuration?
- Did the tensors, feature lists, and manifests agree?
- Did training remain numerically stable?
- Did forecast quality improve after calibration?
- Did the physics-facing diagnostics remain plausible?
- Are transfer or inference results being interpreted under
  the correct target-domain contract?

This page explains how to read diagnostics across the staged
GeoPrior-v3 workflow and how to connect them back to the
artifacts each stage produces.

Why diagnostics matter
----------------------

GeoPrior-v3 is explicitly organized as a staged scientific
workflow in which diagnostics, artifact inspection, and
audits are part of the normal execution model rather than an
optional extra. The workflow guide already emphasizes
progressive validation and warns that a numerically
successful run is not automatically a scientifically
trustworthy one.

That is why diagnostics appear throughout the workflow:

- Stage-1 validates the preprocessing and sequence-export
  contract.
- Stage-2 validates training-time handoff, optimization, and
  forecast export.
- Stage-3 validates tuning outcomes and tuned forecast
  behavior.
- Stage-4 validates saved-model reuse, calibration policy,
  and inference outputs.
- Later workflow steps continue this pattern through export,
  reporting, and downstream interpretation.

A useful way to think about the diagnostics system is:
**contract diagnostics + optimization diagnostics +
forecast diagnostics + physics diagnostics**.

Diagnostic layers in GeoPrior-v3
--------------------------------

GeoPrior-v3 diagnostics naturally fall into four layers.

**1. Contract diagnostics**

These diagnose whether a stage received the right inputs and
whether the workflow contract remains internally consistent.

Typical examples include:

- manifest presence and correctness,
- feature-list agreement,
- tensor shape agreement,
- split and holdout summaries,
- city or run mismatch detection,
- scaling and coordinate metadata consistency.

**2. Optimization diagnostics**

These diagnose training or tuning behavior.

Typical examples include:

- training history curves,
- early stopping behavior,
- NaN termination,
- best-epoch selection,
- tuning-summary comparison,
- trial instability during hyperparameter search.

**3. Forecast diagnostics**

These diagnose the quality and usability of outputs.

Typical examples include:

- point metrics such as MAE or RMSE,
- interval metrics such as coverage and sharpness,
- calibrated vs uncalibrated forecast comparison,
- per-horizon metrics,
- evaluation and future CSV inspection.

**4. Physics diagnostics**

These diagnose whether the model remains physically
interpretable.

Typical examples include:

- epsilon-style residual summaries,
- physics payload exports,
- physical-parameter summaries,
- identifiability-oriented summaries,
- scaled vs raw residual interpretation.

A compact mental model
----------------------

A practical mental model is:

.. code-block:: text

   contract checks
        ↓
   optimization stability
        ↓
   forecast quality
        ↓
   physics consistency
        ↓
   transfer / inference interpretation

If any early layer is wrong, later layers become harder to
trust.

Stage-1 diagnostics
-------------------

Stage-1 diagnostics are mostly about the **data contract**.

Stage-1 writes the manifest, train/validation/test NPZ
bundles, and related CSV snapshots. The manifest records
artifact paths, tensor shapes, sequence dimensions, feature
lists, column roles, configuration snapshot, split details,
and software metadata. It is also the place where you can
inspect the canonical naming block for subsidence,
groundwater, thickness, surface-elevation, and coordinate
roles.

After Stage-1, the first things to inspect are:

- the run directory,
- saved raw / cleaned / scaled CSV snapshots,
- exported NPZ bundles,
- the manifest,
- recorded feature lists,
- recorded shapes,
- holdout summaries,
- and scaling or censoring metadata.

A Stage-1 run is not trustworthy if the manifest and the NPZ
artifacts disagree, if feature ordering drifted, or if the
city / model identity in the artifacts does not match the
intended run.

.. admonition:: Best practice

   Inspect the Stage-1 manifest before moving to Stage-2.

   It is the main handshake artifact for the rest of the
   workflow.

Stage-2 diagnostics
-------------------

Stage-2 is where diagnostics become much richer.

The training stage already performs a Stage-1 → Stage-2
handoff audit, checking items such as manifest identity, city
match, required NPZ existence, tensor last-dimension
agreement with recorded feature lists, horizon / time-step
consistency, scaling metadata availability, and coordinate
compatibility.

During and after training, typical Stage-2 artifacts include:

- training log CSV,
- training summary JSON,
- Stage-2 run manifest,
- model checkpoint artifacts,
- ``scaling_kwargs.json``,
- evaluation forecast CSVs,
- calibrated forecast CSVs,
- evaluation diagnostics JSON,
- physics payload NPZ,
- extracted physical parameter tables.

This is why Stage-2 should not be interpreted only as
``model.fit()``. It is also a diagnostics and export stage.

The first things to inspect after Stage-2 are:

- the training summary,
- the run manifest,
- the CSV history log,
- the best-model bundle,
- the saved scaling snapshot,
- evaluation diagnostics,
- calibrated forecast outputs,
- and the physics payload.

Stage-3 diagnostics
-------------------

Stage-3 diagnostics focus on **selection quality** and
**tuned-model trustworthiness**.

The tuning stage records:

- tuner logs,
- best hyperparameters JSON,
- tuned model manifest,
- ``tuning_summary.json``,
- tuned evaluation forecast CSVs,
- tuned future forecast CSVs,
- ``eval_diagnostics_tuned.json``,
- tuned physics payloads,
- and optional Stage-3 audit outputs.

These diagnostics matter because Stage-3 is not only search.
It is also a tuned evaluation and export stage.

A tuned winner should therefore be reviewed along at least
three axes:

- did the search remain stable?
- did the chosen hyperparameters actually improve the run?
- did the physics-facing diagnostics remain interpretable?

A lower validation loss alone is not enough in a
physics-guided setting.

Stage-4 diagnostics
-------------------

Stage-4 diagnostics are about **saved-model reuse** and
**target-contract-aware inference**.

The inference stage writes its outputs under a target-domain
inference directory and can produce:

- evaluation CSVs,
- future forecast CSVs,
- ``inference_summary.json``,
- and, when optional evaluation flags are enabled,
  ``transfer_eval.json``.

The summary JSON records items such as:

- dataset name,
- city,
- model,
- horizon,
- mode,
- whether the run was calibrated,
- output CSV paths,
- and, when targets and scaler information are available,
  physical-unit point and interval metrics.

The optional ``transfer_eval.json`` extends this with
scaled-space loss information, approximate physics
diagnostics, and debug details about the rebuild path.

This stage is especially important in same-domain reuse,
zero-shot transfer, source-calibrator reuse, and
target-validation recalibration settings, where you must
distinguish carefully between source-model artifacts and the
target-domain contract.

What to read first in a diagnostic review
-----------------------------------------

A good diagnostic review sequence is:

**After Stage-1**

1. manifest
2. feature lists
3. shapes
4. split summary

**After Stage-2**

1. training summary JSON
2. history log
3. evaluation diagnostics JSON
4. calibrated forecast CSVs
5. physics payload

**After Stage-3**

1. tuning summary
2. best hyperparameters JSON
3. tuned diagnostics JSON
4. tuned forecast CSVs
5. tuned physics payload

**After Stage-4**

1. inference summary JSON
2. transfer-eval JSON, if present
3. evaluation CSV
4. future CSV
5. calibration policy

This order makes it easier to separate contract problems from
optimization problems and optimization problems from physics
problems.

Interpreting forecast metrics
-----------------------------

GeoPrior-v3 diagnostics span both point forecasts and
interval forecasts.

Common point-facing metrics include:

- MAE,
- MSE,
- RMSE,
- and, in some summaries, :math:`R^2`.

Common interval-facing metrics include:

- coverage,
- sharpness,
- and per-horizon variants of the same ideas.

A good mental model is:

- **point metrics** tell you how close the central forecast
  is;
- **interval metrics** tell you whether the uncertainty band
  is both useful and honest.

When calibrated and uncalibrated forecast outputs both
exist, compare them explicitly rather than reading only one
set of CSVs.

Interpreting calibration
------------------------

Calibration is an explicit part of later GeoPrior stages.

Stage-2 fits interval calibration after training.
Stage-3 can calibrate tuned quantile outputs before export.
Stage-4 can reuse a source calibrator, load a calibrator
explicitly, or fit a new one on target validation data.

That means calibration should be treated as a visible part of
interpretation, not as a hidden post-processing detail.

Ask these questions:

- Was the output calibrated or not?
- Which calibrator policy was used?
- Was the calibrator source-side or target-side?
- Are you comparing calibrated and uncalibrated outputs
  fairly?

If those questions are ignored, interval comparisons become
harder to trust.

Interpreting physics diagnostics
--------------------------------

The physics side has its own diagnostic language.

The math page already distinguishes two important families of
epsilon diagnostics:

- **raw epsilons**, which still carry the original residual
  scale;
- **scaled epsilons**, which are unitless and correspond more
  directly to the optimization view.

Typical keys include:

- ``epsilon_cons_raw`` and ``epsilon_cons``
- ``epsilon_gw_raw`` and ``epsilon_gw``
- ``epsilon_prior``

A useful interpretation rule is:

- if raw epsilons are very large but scaled epsilons are
  moderate, scaling may be doing its job;
- if both are unstable or exploding, you likely need to
  inspect units, coordinate spans, or scaling assumptions.

The same math page also emphasizes that GeoPrior uses
SI-consistent physics internally, with head-like quantities
treated in meters, time converted to seconds for physics, and
positive SI fields such as :math:`K`, :math:`S_s`,
:math:`H`, and :math:`\\tau`. This is why diagnostics should
always be interpreted together with the data-and-units and
scaling pages.

A practical physics-diagnostics checklist
-----------------------------------------

When reviewing a physics-aware run, ask:

- Are epsilon curves finite?
- Are raw and scaled epsilons telling a consistent story?
- Does ``epsilon_prior`` suggest a severe timescale mismatch?
- Were physics payloads exported successfully?
- Do extracted physical-parameter tables look plausible?
- Are unit assumptions and coordinate conventions still the
  same as those recorded by Stage-1 and Stage-2?

If the answer to any of these is “not sure,” pause before
making a scientific interpretation.

Programmatic diagnostics helpers
--------------------------------

The public model-facing namespace also exposes several helper
utilities that are useful when diagnostics are inspected from
Python rather than only from files.

Examples include:

- ``debug_tensor_interval``
- ``debug_val_interval``
- ``debug_quantile_crossing_np``
- ``plot_history_in``
- ``fit_interval_calibrator_on_val``
- ``apply_calibrator_to_subs``
- ``extract_physical_parameters``
- ``load_physics_payload``
- ``plot_physics_values_in``
- ``identifiability_diagnostics_from_payload``
- ``summarise_effective_params``
- ``ident_audit_dict``

These tools are worth documenting here because they give the
user a path from stage-level artifact inspection to
programmatic post hoc diagnosis.

Common diagnostic patterns
--------------------------

**Pattern 1: Good training loss, poor forecast CSVs**

This often means the optimization loop was stable, but the
forecast interpretation step still needs inspection. Check:

- calibration policy,
- quantile layout handling,
- output formatting,
- and the specific split being evaluated.

**Pattern 2: Good point metrics, poor interval behavior**

This often means the central prediction is fine, but the
uncertainty bands are too narrow, too wide, or poorly
calibrated. Compare calibrated and uncalibrated outputs.

**Pattern 3: Good supervised metrics, unstable physics
diagnostics**

This often points to unit, scaling, coordinate, or physics
weight problems rather than a pure forecast-quality issue.

**Pattern 4: Good tuned result, poor transfer or inference
reuse**

This often means the saved model is fine, but the target
contract, calibration policy, or bundle reconstruction path
needs inspection.

**Pattern 5: Strange behavior very early in the workflow**

This often goes back to Stage-1. Recheck the manifest,
feature lists, split summaries, and exported tensors before
debugging later stages.

Common mistakes
---------------

**Reading only one summary file**

A training summary without the forecast diagnostics, or an
inference summary without the target contract, gives only a
partial picture.

**Ignoring manifests**

The manifest is not bookkeeping clutter. It is how the
workflow records meaning.

**Treating calibration as invisible**

Later-stage diagnostics must be interpreted in light of the
calibration policy.

**Reading physics metrics without checking units**

Residual diagnostics can look alarming or reassuring for the
wrong reason if unit assumptions are wrong.

**Comparing runs with different contracts**

Two runs are not automatically comparable if their Stage-1
feature order, target semantics, or scaling conventions
differ.

Best practices
--------------

.. admonition:: Best practice

   Review diagnostics in layers.

   Start with contract checks, then optimization, then
   forecast quality, then physics interpretation.

.. admonition:: Best practice

   Keep summary JSONs, manifests, and forecast CSVs together.

   A single artifact rarely explains the full story.

.. admonition:: Best practice

   Compare calibrated and uncalibrated outputs explicitly.

   This is especially important when interval behavior is part
   of the scientific claim.

.. admonition:: Best practice

   Treat physics diagnostics as scientific evidence, not only
   as debugging metadata.

   They are part of the model’s interpretability contract.

.. admonition:: Best practice

   When in doubt, go back one stage earlier.

   Many late-stage diagnostic surprises originate in earlier
   artifact or semantics drift.

A compact diagnostics map
-------------------------

The GeoPrior-v3 diagnostics story can be summarized like
this:

.. code-block:: text

   Stage-1
     manifest + shapes + feature lists + split summaries
        ↓
   Stage-2
     history logs + training summary + eval diagnostics
     + calibrated forecasts + physics payload
        ↓
   Stage-3
     tuning summary + best_hps + tuned diagnostics
     + tuned forecasts + tuned physics payload
        ↓
   Stage-4
     inference_summary + transfer_eval + eval/future CSVs
     + calibration-policy interpretation
        ↓
   scientific review
     point metrics + interval metrics + physics diagnostics
     + transfer / reuse interpretation

Read next
---------

The best next pages after this one are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Inference and export
      :link: inference_and_export
      :link-type: doc
      :class-card: sd-shadow-sm

      See how diagnostic artifacts relate to the reusable
      inference and export workflow.

   .. grid-item-card:: Data and units
      :link: ../scientific_foundations/data_and_units
      :link-type: doc
      :class-card: sd-shadow-sm

      Revisit the unit and coordinate assumptions that affect
      physics-facing diagnostics.

   .. grid-item-card:: Scaling
      :link: ../scientific_foundations/scaling
      :link-type: doc
      :class-card: sd-shadow-sm

      Review the scaling rules that shape both optimization
      and physics residual interpretation.

   .. grid-item-card:: Maths
      :link: ../scientific_foundations/maths
      :link-type: doc
      :class-card: sd-shadow-sm card--physics
      :data-preview: physics.png
      :data-preview-label: GeoPrior physics and diagnostics

      Connect the workflow diagnostics back to the formal math
      behind residuals, losses, and epsilon summaries.

.. seealso::

   - :doc:`workflow_overview`
   - :doc:`stage1`
   - :doc:`stage2`
   - :doc:`stage3`
   - :doc:`stage4`
   - :doc:`inference_and_export`
   - :doc:`../scientific_foundations/data_and_units`
   - :doc:`../scientific_foundations/scaling`
   - :doc:`../scientific_foundations/maths`