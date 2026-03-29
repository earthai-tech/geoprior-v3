FAQ
===

This page collects the questions that users are most likely
to ask when bringing up, running, debugging, and reusing
GeoPrior-v3.

The goal of the FAQ is not to replace the main guides. It is
to answer practical questions quickly and point you to the
right page when you need more detail.

General questions
-----------------

What is GeoPrior-v3 meant to do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoPrior-v3 is a workflow-oriented scientific framework for
**physics-guided geohazard modeling, forecasting, and risk
analytics**.

Its current flagship application is **land subsidence**
through GeoPriorSubsNet v3.x, but the broader project is
structured for wider geohazard and transfer-oriented
workflows.

It is designed as:

- a Python package,
- a staged CLI workflow,
- and a reproducible scientific software platform.

Where should I start if I am new?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start in this order:

1. :doc:`../getting_started/overview`
2. :doc:`../getting_started/installation`
3. :doc:`../getting_started/quickstart`
4. :doc:`../getting_started/first_project_run`
5. :doc:`workflow_overview`

That sequence gives the cleanest path from installation to a
real stage-based workflow run.

Is GeoPrior-v3 only a Python library?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No.

GeoPrior-v3 is both:

- a Python package for models and utilities,
- and a CLI-driven application framework.

If you want the workflow surface, start from:

- ``geoprior``
- ``geoprior-run``
- ``geoprior-build``
- ``geoprior-plot``
- ``geoprior-init``

See :doc:`cli` for the full structure.

Installation and environment
----------------------------

Which Python version do I need?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoPrior-v3 currently targets **Python 3.10 and above**.

If installation behaves strangely, confirm the interpreter
first:

.. code-block:: bash

   python --version

Why does installation feel heavier than a small utility package?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because GeoPrior-v3 is not a lightweight pure-Python tool.

It depends on a scientific stack that includes packages for
numerical computing, machine learning, Keras / TensorFlow
workflows, configuration, and artifact handling.

That is why isolated environments are strongly recommended.

Do I really need a virtual environment?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes.

Using a dedicated environment is one of the easiest ways to
avoid:

- version conflicts,
- wrong CLI resolution,
- accidental mixing of packages from other projects,
- and confusing import behavior.

Why does the package import, but a stage still fails later?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because a successful import is only the first check.

Later stages may exercise parts of the environment that the
initial import did not fully test, such as:

- TensorFlow / Keras execution,
- file and path handling,
- stage artifact loading,
- calibration utilities,
- or saved-model reconstruction.

A working import is necessary, but it is not the same thing
as a fully working workflow environment.

CLI questions
-------------

What is the difference between ``geoprior`` and ``geoprior-run``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``geoprior`` is the root dispatcher.

It expects a command family first, such as:

.. code-block:: bash

   geoprior run ...
   geoprior build ...
   geoprior plot ...

``geoprior-run`` is the family-scoped shortcut for the run
family.

So these ideas are related:

.. code-block:: bash

   geoprior run stage1-preprocess
   geoprior-run stage1-preprocess

Use the second form once you already know you are in the run
family.

What is ``geoprior-init`` for?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It bootstraps the active configuration.

That is usually the easiest way to create the working
``nat.com/config.py`` and refresh the corresponding runtime
snapshot.

If you are not sure where to begin, start here:

.. code-block:: bash

   geoprior-init --help

Why does ``geoprior-plot`` exist if there are no plot commands yet?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plot family already exists as part of the public CLI
shape, but the current command registry may not yet expose
concrete plot commands.

So the entry point is real, but it should currently be
treated as a reserved family surface rather than a populated
command set.

Should I memorize all command names?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No.

Use help first:

.. code-block:: bash

   geoprior --help
   geoprior-run --help
   geoprior-build --help
   geoprior-init --help

That is safer than relying on old notes or shell history,
especially while the CLI is still evolving.

Configuration questions
-----------------------

What is the difference between ``config.py`` and ``config.json``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A useful rule is:

- ``config.py`` is the **human-edited source**
- ``config.json`` is the **persisted effective runtime
  snapshot**

So you usually edit ``config.py`` and inspect ``config.json``
when you want to see what a CLI command actually used after
overrides.

Should I edit ``config.json`` directly?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usually no.

Treat ``config.json`` as a recorded runtime artifact, not as
your main authoring surface.

Your main project-level editing should usually happen in
``config.py``.

What does ``--config`` do?
~~~~~~~~~~~~~~~~~~~~~~~~~~

It installs a user-supplied ``config.py`` into the active
config root before the command runs.

That is useful when you want to run with a different authored
config file without manually copying files yourself.

What does ``--set KEY=VALUE`` do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It applies one-off runtime overrides.

This is useful for temporary experiments, such as:

.. code-block:: bash

   --set TIME_STEPS=6
   --set FORECAST_HORIZON_YEARS=3
   --set USE_VSN=True
   --set QUANTILES=[0.1,0.5,0.9]

These overrides are typically merged into the effective
runtime config and then persisted.

When should I use ``config.py`` vs ``--set``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``config.py`` for **stable project defaults**.

Use ``--set`` for **temporary experiment changes**.

A good rule is:

- if the value is part of the long-lived project setup, put
  it in ``config.py``;
- if the value is only for this one run, use ``--set``.

Why did my override not work?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common reasons:

- you forgot the ``KEY=VALUE`` format,
- the key name is wrong,
- you ran the command from the wrong config root,
- the stage uses a refresh step that changes derived fields,
- you are inspecting old artifacts from a previous run.

When in doubt, inspect the effective config snapshot and the
stage manifest.

Stage and artifact questions
----------------------------

Why is Stage-1 so important?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because Stage-1 establishes the **artifact contract** for the
rest of the workflow.

It writes the manifest, split artifacts, NPZ bundles, and
related metadata that later stages depend on.

If Stage-1 is wrong, later stages can still look “successful”
while actually being misconfigured.

Can I skip Stage-1 and jump directly to training?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

That is usually a bad idea.

Stage-2 and Stage-3 are designed to consume the Stage-1
exports and manifest. Skipping that boundary makes the
workflow less reproducible and much harder to debug.

What is the Stage-1 manifest?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is the main handshake artifact that tells later stages:

- what was exported,
- where it was written,
- what the shapes are,
- what the feature lists are,
- what the target semantics are,
- and what the config snapshot looked like.

Treat it as authoritative.

What if my Stage-1 manifest is stale?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then later stages may still run, but the run can become
misleading.

Common symptoms:

- strange feature mismatches,
- wrong city or path assumptions,
- dimension disagreements,
- outputs that look plausible but correspond to the wrong
  configuration.

If you suspect this, regenerate or explicitly point to the
correct manifest.

What does Stage-2 add beyond training?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A lot.

Stage-2 is not only a fit loop. It also handles:

- training-time handoff validation,
- summary and manifest writing,
- clean model export,
- interval calibration,
- forecast CSV generation,
- diagnostics export,
- and physics payload export.

Why does Stage-3 still depend on Stage-1?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because tuning is built on the same exported tensor contract.

Stage-3 does **not** redo preprocessing. It tunes against the
Stage-1 exports so that trials remain comparable under a
stable data contract.

What is the difference between Stage-2 and Stage-3?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A useful distinction is:

- **Stage-2** = one configured training run
- **Stage-3** = systematic hyperparameter search on top of
  the Stage-1 contract

Stage-3 then continues beyond the search itself by evaluating
the tuned winner and exporting tuned outputs.

Inference and model reuse
-------------------------

Why is Stage-4 a separate stage?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because reusable inference deserves its own workflow logic.

Stage-4 is not just “call predict on a saved model.” It is a
target-contract-aware inference stage that can:

- load a target Stage-1 contract,
- resolve a saved bundle,
- rebuild a robust inference-capable model,
- apply optional calibration logic,
- and export evaluation and future forecast products.

Why is a saved model file not enough on its own?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because saved-model reuse in GeoPrior-v3 often still depends
on:

- the Stage-1 target contract,
- feature ordering,
- scaling metadata,
- quantile handling,
- and sometimes nearby model-init metadata or best weights.

A raw model file without that context can be hard to reuse
correctly.

What is ``inference_summary.json``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is the compact record of what a Stage-4 inference run
actually did.

It helps answer questions such as:

- which dataset was used,
- whether calibration was active,
- which outputs were written,
- and what metrics were available.

What is ``transfer_eval.json``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is an optional, richer Stage-4 evaluation summary.

It is especially useful for debug and scientific audit when
you want more than just the forecast CSVs, for example:

- scaled-space evaluation loss information,
- approximate physics diagnostics,
- and rebuild-path details.

Calibration and uncertainty questions
-------------------------------------

What does “calibrated” mean here?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In GeoPrior-v3, calibration usually refers to **interval
calibration** applied to probabilistic subsidence outputs.

A helpful way to think about it is:

- the model provides the original quantile structure,
- the calibrator adjusts interval width around the center,
- the calibrated export becomes the artifact used for later
  uncertainty interpretation.

Should I always export calibrated forecasts?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not always, but you should always know **whether** a forecast
was calibrated.

Calibrated and uncalibrated outputs answer different
questions, and both can be useful in analysis.

Why do point metrics look good while interval behavior looks poor?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because those are different diagnostic layers.

A model can produce:

- a good central prediction,
- but poor uncertainty bands.

If that happens, compare raw and calibrated interval outputs
explicitly rather than looking only at central forecasts.

What is quantile canonicalization?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is the step that ensures quantile outputs are interpreted
in the expected layout before calibration and CSV formatting
happen.

This matters because inference and export should not assume
that arbitrary saved-model outputs already have the correct
axis semantics.

Transfer questions
------------------

What is Stage-5 for?
~~~~~~~~~~~~~~~~~~~~

Stage-5 is the **cross-city transfer benchmarking stage**.

It compares source-to-target behavior across:

- direction,
- strategy,
- split,
- calibration mode,
- and rescale mode.

It is one of the most scientifically important stages in the
workflow.

What is the difference between baseline, xfer, and warm?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A useful summary is:

- **baseline** = in-domain reference case
- **xfer** = zero-shot cross-city transfer
- **warm** = warm-start target adaptation

These are different regimes and should not be collapsed into
one category during interpretation.

What is the difference between source and target calibration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

They answer different questions.

- **source calibration** asks whether source-side calibration
  remains useful after transfer
- **target calibration** asks what happens when uncertainty
  intervals are adapted to the target domain

These should not be interpreted as equivalent.

What is the difference between ``as_is`` and ``strict`` rescale modes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Very roughly:

- **as_is** keeps the target domain in its own scaling
  convention
- **strict** tries to enforce a stricter source-space
  transfer interpretation

This matters because transfer differences can come from model
limitations, domain shift, or scaling mismatch.

Why does Stage-5 care so much about schema alignment?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because cross-city transfer is not trustworthy if the source
and target contracts do not align.

Static feature alignment can sometimes be softened by name,
but dynamic and future feature compatibility usually need to
be much stricter.

If feature names or order drift, transfer results become
harder to trust.

Artifacts and reproducibility
-----------------------------

Which files should I keep together after a serious run?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try to keep these together per run:

- the stage manifest,
- the saved model bundle,
- best weights if present,
- model-init manifest,
- scaling snapshot,
- training or tuning summary,
- forecast CSVs,
- diagnostics JSON,
- and physics payloads.

This makes later reuse much easier.

Can I mix artifacts from different runs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You usually should not.

Mixing manifests, models, scalers, forecast CSVs, or
diagnostics from different runs is one of the fastest ways to
create silent workflow drift.

Why do the JSON summaries matter if I already have the CSVs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the CSVs are easier to read, but the JSON summaries
tell you:

- what the run actually used,
- whether calibration was active,
- what bundle was loaded,
- which metrics were computed,
- and what the stage believed the contract to be.

A CSV alone rarely tells the whole story.

Do I need to keep run directories separate?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes.

Keeping one coherent run directory per configuration or
experiment family is one of the easiest ways to keep the
workflow trustworthy.

Model and debugging questions
-----------------------------

What is the preferred import path for GeoPriorSubsNet?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prefer the public import surface, for example:

.. code-block:: python

   from geoprior.models import GeoPriorSubsNet

or:

.. code-block:: python

   from geoprior.models.subsidence import GeoPriorSubsNet

Avoid relying on unnecessarily deep internal paths unless you
are explicitly working on the internals.

Why does the model complain about missing ``coords``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because coordinates are part of the expected input contract
for the current model family.

If you are building custom inputs or debugging from Python,
make sure your input dict contains the required keys and that
their shapes match the model contract.

Why do I get an error about hard bounds requiring bounds?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because some physics or constraint modes require explicit
bounds metadata.

If you enable hard-bounds behavior, make sure the required
bounds are actually present in the scaling or configuration
context.

Can I reconstruct a model from config alone?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In many cases, yes.

The model family is designed so that configuration and
serialized initialization metadata can support robust
reconstruction, which is one reason the saved manifests
matter so much.

Docs and documentation-workflow questions
-----------------------------------------

How do I build the docs?
~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root, after installing the docs extra:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

Then open:

.. code-block:: text

   docs/build/html/index.html

Which theme does the docs site use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The docs are now configured around the **PyData Sphinx
Theme**, together with Sphinx design components, MyST,
NumPy-style doc support, and BibTeX integration.

Why do some old links still look wrong?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the docs were previously hosted under the older
FusionLab / inherited structure and are now being rebuilt as
their own GeoPrior-v3 documentation site.

Part of the migration is cleaning old internal links,
outdated paths, and inherited references from the previous
doc layout.

How should I use this FAQ?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use it as a fast navigation aid.

A practical pattern is:

- if the issue is about stage meaning, go to the stage page;
- if the issue is about commands, go to :doc:`cli`;
- if the issue is about config, go to :doc:`configuration`;
- if the issue is about interpretation, go to
  :doc:`diagnostics`;
- if the issue is about saved-model reuse or outputs, go to
  :doc:`inference_and_export`.

A compact FAQ map
-----------------

A good mental summary is:

.. code-block:: text

   install correctly
        ↓
   initialize config
        ↓
   run Stage-1 and trust the manifest
        ↓
   train or tune with explicit artifact handoff
        ↓
   inspect summaries and diagnostics
        ↓
   reuse saved models through Stage-4
        ↓
   compare transfer regimes through Stage-5
        ↓
   keep artifacts, summaries, and manifests together

Read next
---------

The most useful follow-up pages after the FAQ are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: CLI guide
      :link: cli
      :link-type: doc
      :class-card: sd-shadow-sm card--cli
      :data-preview: cli.png
      :data-preview-label: GeoPrior command-line interface

      Go from quick questions to the full command structure.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc
      :class-card: sd-shadow-sm card--configuration
      :data-preview: configuration.png
      :data-preview-label: GeoPrior configuration system

      Review the config layer, override model, and runtime
      snapshot logic.

   .. grid-item-card:: Diagnostics
      :link: diagnostics
      :link-type: doc
      :class-card: sd-shadow-sm

      Understand how to inspect workflow health and scientific
      trustworthiness.

   .. grid-item-card:: Inference and export
      :link: inference_and_export
      :link-type: doc
      :class-card: sd-shadow-sm

      Read the saved-model reuse, forecast export, and
      transfer-output story in one place.

.. seealso::

   - :doc:`cli`
   - :doc:`configuration`
   - :doc:`workflow_overview`
   - :doc:`stage1`
   - :doc:`stage2`
   - :doc:`stage3`
   - :doc:`stage4`
   - :doc:`stage5`
   - :doc:`diagnostics`
   - :doc:`inference_and_export`