.. _applications_reproducibility_scripts:

=========================
Reproducibility scripts
=========================

GeoPrior-v3 is designed not only for model development, but
also for **reproducible scientific workflows**.

In many research projects, the real difficulty does not end
when the model trains successfully. A result becomes useful
only when you can reliably:

- rerun the preprocessing contract,
- recover the exact training configuration,
- regenerate the main forecast outputs,
- rebuild physics payloads and validation artifacts,
- and recreate the figures or tables used in a manuscript.

This page explains how GeoPrior supports that workflow and
how the ``scripts/`` directory fits into the broader
reproducibility story.

What “reproducibility” means in GeoPrior
----------------------------------------

In GeoPrior-v3, reproducibility does **not** mean only
“rerun one training command.”

It usually means being able to reconstruct a whole result
chain:

1. the authored configuration,
2. the effective runtime configuration,
3. the Stage-1 manifest and exported NPZ bundles,
4. the Stage-2 or Stage-3 model artifacts,
5. the inference or transfer outputs,
6. and the figure/table scripts that consume those outputs.

A useful summary is:

.. code-block:: text

   config
      + manifests
      + artifacts
      + saved models
      + summaries
      + scripts
      = reproducible result

This is the guiding idea behind the application-facing
reproducibility layer.

Why scripts matter
------------------

GeoPrior-v3 separates two concerns:

- the **workflow stages**, which create durable model and
  data artifacts;
- the **reproducibility scripts**, which consume those
  artifacts to produce paper-facing outputs such as figures,
  parity plots, payload summaries, or validation tables.

That separation is good scientific practice because it avoids
mixing:

- heavy training logic,
- with lightweight figure regeneration.

It also makes it easier to rerun figures without retraining a
model from scratch.

The role of the ``scripts/`` directory
--------------------------------------

The project structure already separates out a dedicated
``scripts/`` directory for:

- reproducibility scripts,
- figure-generation utilities,
- and paper-facing export helpers.

Typical examples mentioned in the project structure include:

- ``plot_physics_fields.py``
- ``plot_litho_parity.py``

alongside other reproducibility or figure scripts.

This is exactly the right design for a scientific software
project: training and inference stay in the core workflow,
while figure-oriented logic lives in a clearly separate
location.

What the scripts should consume
-------------------------------

A GeoPrior reproducibility script should generally consume
**existing durable artifacts**, not regenerate the full model
pipeline unless that is the explicit goal.

Typical inputs for a reproducibility script include:

- Stage-1 manifest and NPZ bundles,
- ``training_summary.json`` or ``tuning_summary.json``,
- saved model bundles,
- ``scaling_kwargs.json``,
- evaluation and future forecast CSVs,
- physics payload NPZ files,
- ``inference_summary.json``,
- ``transfer_eval.json``,
- ``xfer_results.csv`` / ``xfer_results.json``.

This is a key principle:

- scripts are best when they are **artifact-driven**,
- not when they hide fresh training inside a plotting call.

Relationship to the staged workflow
-----------------------------------

The reproducibility layer sits on top of the main GeoPrior
workflow.

A typical scientific lifecycle looks like:

.. code-block:: text

   Stage-1
     create preprocessing contract
        ↓
   Stage-2 / Stage-3
     create trained or tuned model artifacts
        ↓
   Stage-4 / Stage-5
     create forecast and benchmark outputs
        ↓
   scripts/
     regenerate figures, tables, and paper-facing summaries

This is one of the cleanest ways to make a research workflow
auditable.

What should be reproducible
---------------------------

For a serious GeoPrior experiment, the following should be
reproducible without ambiguity:

- the data contract,
- the training or tuning setup,
- the inference/export outputs,
- the transfer comparison outputs,
- and the final figures or tables.

A good practical test is:

**Could another person regenerate the same paper figure from
the saved run directory and the corresponding script?**

If the answer is yes, the workflow is in good shape.

The minimum reproducibility bundle
----------------------------------

A good minimum reproducibility bundle for one experiment
usually includes:

- ``nat.com/config.py``
- ``nat.com/config.json``
- Stage-1 ``manifest.json``
- exported NPZ bundles
- saved model artifact(s)
- ``model_init_manifest.json``
- ``training_summary.json`` or ``tuning_summary.json``
- ``scaling_kwargs.json``
- evaluation / future forecast CSVs
- diagnostics JSON files
- physics payload NPZ files
- the script used to produce the final figure or table

This is usually a much stronger reproducibility package than
a notebook alone.

Configuration as a reproducibility artifact
-------------------------------------------

The reproducibility story starts with configuration.

GeoPrior already distinguishes between:

- ``config.py`` as the human-authored project config,
- ``config.json`` as the persisted effective runtime config.

That is extremely valuable for scientific reproducibility,
because it means a script or figure can be tied back to the
actual effective runtime state rather than to a remembered or
informal setup.

.. admonition:: Best practice

   Preserve both the authored config and the effective runtime
   snapshot for every serious experiment.

   A figure script is much easier to trust if the config used
   to generate its source artifacts is still available.

Manifests as provenance anchors
-------------------------------

Stage-level manifests are one of the strongest parts of the
GeoPrior reproducibility design.

For example:

- Stage-1 manifest anchors the preprocessing contract;
- model-init and run manifests anchor the trained or tuned
  model configuration;
- inference and transfer summaries anchor downstream reuse.

These files are what make later scripts scientifically
defensible. They tell the user not only where data came from,
but what the workflow believed those artifacts meant.

A useful rule is:

- if a figure depends on a result,
- the corresponding manifest or summary file should be saved
  with that result.

What the scripts should avoid
-----------------------------

A reproducibility script should **not** casually hide major
workflow changes.

Avoid scripts that do things like:

- silently retrain the model,
- silently rewrite manifests,
- silently change scaling conventions,
- silently mix artifacts from two runs,
- silently regenerate Stage-1 under a different config.

That makes figure regeneration much harder to trust.

A good reproducibility script should ideally do one of three
things:

1. read existing artifacts and plot them;
2. read existing artifacts and build a table/summary;
3. run a deterministic post-processing step whose inputs and
   outputs are both explicit.

Typical reproducibility use cases
---------------------------------

GeoPrior reproducibility scripts are especially useful for:

**Physics-field figures**

For example, plotting learned or summarized values such as:

- :math:`K`
- :math:`S_s`
- :math:`\tau`
- :math:`H_d`

from a saved physics payload.

**Parity and validation figures**

For example, comparing:

- predicted vs observed subsidence,
- predicted vs observed groundwater,
- or lithology-conditioned parity behavior.

**Transfer benchmark figures**

For example, plotting:

- baseline vs transfer vs warm-start performance,
- calibration-mode comparison,
- per-horizon transfer degradation,
- cross-city uncertainty behavior.

**Table regeneration**

For example, rebuilding:

- summary metric tables,
- benchmark tables,
- per-horizon comparison tables,
- calibration comparison tables.

The role of deterministic build commands
----------------------------------------

Not every reproducibility task belongs in ``scripts/`` alone.

GeoPrior’s **build** family exists for deterministic artifact
generation from existing workflow outputs. This is a very
useful complement to paper-facing scripts.

Typical build commands include tasks such as:

- merged full-input NPZ generation,
- physics payload NPZ generation,
- external validation artifact construction,
- external validation metric computation.

This suggests a good separation of labor:

- use **build commands** for deterministic artifact
  materialization;
- use **scripts** for paper-facing visualization and summary
  logic.

That separation is both cleaner and easier to maintain.

Figure-generation philosophy
----------------------------

A good GeoPrior figure script should be:

- **artifact-driven**
- **lightweight**
- **deterministic**
- **well-labeled**
- **easy to rerun**

It should ideally take explicit inputs such as:

- a run directory,
- a manifest path,
- a payload NPZ path,
- a forecast CSV path,
- or an output figure path.

This is much better than hard-coding one user’s local paths
inside the script body.

A practical script pattern
--------------------------

A clean reproducibility script often follows this pattern:

.. code-block:: text

   parse explicit input paths
        ↓
   load manifest / summary / payload / CSV
        ↓
   validate that required fields exist
        ↓
   compute any deterministic derived summaries
        ↓
   write figure(s) or table(s)
        ↓
   optionally write a small sidecar metadata file

This pattern works well for both plots and tables.

Examples of good script inputs
------------------------------

A good figure or summary script might accept arguments such
as:

- ``--stage1-manifest``
- ``--stage2-run-dir``
- ``--payload-npz``
- ``--forecast-csv``
- ``--xfer-results-csv``
- ``--outdir``
- ``--format``

The goal is not that every script must support all of these,
but that each script makes its dependency chain explicit.

Physics-payload reproducibility
-------------------------------

One especially strong GeoPrior pattern is the use of
**physics payloads**.

These payloads make it possible to regenerate:

- parameter maps,
- effective-parameter summaries,
- identifiability diagnostics,
- physics-consistency plots,

without retraining the model.

This is one of the best parts of the package for scientific
reproducibility, because it lets the user separate:

- training-time cost,
- from post hoc scientific analysis.

Transfer-result reproducibility
-------------------------------

Stage-5 is another strong reproducibility case.

The transfer workflow already aggregates outcomes into:

- ``xfer_results.json``
- ``xfer_results.csv``

Those two files are natural inputs to reproducibility
scripts for:

- benchmark tables,
- transfer comparison figures,
- per-strategy summaries,
- calibration-mode comparisons.

That means a paper-quality transfer figure does not need to
rerun Stage-5 every time. It should be reproducible from the
saved benchmark outputs.

Uncertainty reproducibility
---------------------------

Calibration and uncertainty products are also excellent
targets for reproducibility scripts.

Useful artifacts include:

- calibrated forecast CSVs,
- calibration statistics JSON,
- calibrated diagnostics JSON,
- interval factor files.

These can be used to reproduce:

- uncertainty band plots,
- coverage vs sharpness summaries,
- horizon-wise interval comparison figures,
- same-domain vs transfer uncertainty comparisons.

This is another reason the application scripts should remain
artifact-driven.

What a paper-ready run directory should preserve
------------------------------------------------

A good paper-ready run directory should preserve enough
information that later scripts can be rerun without guessing.

For example, it should be possible to recover:

- what city or city-pair the result came from,
- what stage artifacts were used,
- whether the model was trained or tuned,
- whether outputs were calibrated,
- which forecast horizon was used,
- where the source payload or CSV lives.

If a script cannot answer those questions from the saved
artifacts, the reproducibility chain is weaker than it should
be.

Common mistakes
---------------

**Putting figure logic inside training scripts**

This makes later reruns heavy and harder to audit.

**Relying on local absolute paths**

A script should be portable enough to run from explicit
artifact paths.

**Mixing artifacts from different runs**

This is one of the fastest ways to create misleading figures.

**Recomputing major workflow steps silently**

A plotting script should not secretly rerun training unless
it clearly says so.

**Not preserving summary JSON files**

CSVs are useful, but the JSON summaries explain what the
artifacts actually represent.

Best practices
--------------

.. admonition:: Best practice

   Keep figure scripts artifact-driven.

   They should consume saved outputs, not recreate the whole
   experiment implicitly.

.. admonition:: Best practice

   Save manifests and summaries next to the artifacts they
   describe.

   This makes later figure regeneration far easier to trust.

.. admonition:: Best practice

   Preserve transfer and calibration outputs in their final
   exported form.

   These are often the true inputs to paper-ready figures.

.. admonition:: Best practice

   Keep scripts lightweight and explicit about inputs.

   A clear command-line interface for one script is often more
   reproducible than a notebook with hidden state.

.. admonition:: Best practice

   Separate deterministic artifact builders from figure
   scripts.

   Use build commands for materialization and scripts for
   visualization or summarization.

A compact reproducibility map
-----------------------------

The GeoPrior reproducibility workflow can be summarized as:

.. code-block:: text

   config + runtime snapshot
        ↓
   stage manifests + saved artifacts
        ↓
   trained / tuned / inferred outputs
        ↓
   deterministic build exports (optional)
        ↓
   scripts/ consumes payloads, CSVs, summaries
        ↓
   figures, tables, and paper-ready artifacts

Read next
---------

The strongest next pages after this one are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Figure generation
      :link: figure_generation
      :link-type: doc
      :class-card: sd-shadow-sm

      Move from reproducibility logic to the figure-facing
      workflow directly.

   .. grid-item-card:: Diagnostics
      :link: ../user_guide/diagnostics
      :link-type: doc
      :class-card: sd-shadow-sm

      Revisit the summaries and payloads that scripts most
      often consume.

   .. grid-item-card:: Inference and export
      :link: ../user_guide/inference_and_export
      :link-type: doc
      :class-card: sd-shadow-sm

      Review the exported artifacts that form the inputs to
      reproducibility scripts.

   .. grid-item-card:: Stage-5
      :link: ../user_guide/stage5
      :link-type: doc
      :class-card: sd-shadow-sm

      See how cross-city benchmark outputs become reproducible
      downstream figures and tables.

.. seealso::

   - :doc:`subsidence_forecasting`
   - :doc:`calibration_and_uncertainty`
   - :doc:`figure_generation`
   - :doc:`../user_guide/diagnostics`
   - :doc:`../user_guide/inference_and_export`
   - :doc:`../user_guide/stage4`
   - :doc:`../user_guide/stage5`