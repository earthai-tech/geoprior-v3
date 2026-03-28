Workflow 
=================

GeoPrior-v3 is designed as a **staged scientific workflow**
rather than as a single monolithic command or a model class
used in isolation.

This design is intentional. In physics-guided geohazard
modeling, many problems do not come from model architecture
alone. They come from the interaction between:

- data preparation,
- scaling and unit conventions,
- feature assembly,
- physics-aware configuration,
- training or inference behavior,
- diagnostics,
- export logic,
- and reproducibility requirements.

For that reason, GeoPrior-v3 organizes work into explicit
stages and treats configuration, artifacts, and audits as
first-class workflow objects.

Why the workflow is staged
--------------------------

A staged workflow is useful because it makes scientific and
technical problems easier to isolate.

Instead of pushing everything through one long opaque run,
GeoPrior-v3 encourages a stepwise progression in which each
stage has a clear role, a clear set of inputs, and a clear
set of outputs. This helps with:

- debugging path or data issues early,
- validating units and scaling before physics-heavy runs,
- separating preprocessing from modeling,
- keeping intermediate artifacts inspectable,
- making experiments easier to reproduce,
- reducing the risk of silent workflow drift.

This is especially important for a framework like GeoPrior-v3,
where a numerically successful run is not automatically a
scientifically trustworthy run.

Core workflow philosophy
------------------------

The staged workflow is built around a few guiding ideas.

**1. Configuration drives the run**

GeoPrior-v3 favors explicit configuration over hidden
assumptions. A run should be defined primarily by its
configuration and inputs, not by scattered code edits.

**2. Stages communicate through artifacts**

Instead of passing everything in memory across an opaque
pipeline, stages typically hand off structured artifacts such
as tensors, metadata, manifests, scaling contracts, logs,
forecasts, or figures.

**3. Audits matter**

Shape checks, scaling checks, unit checks, and handshake
audits are not side details. They are part of the scientific
workflow and help catch common silent failures early.

**4. Reproducibility is part of the design**

The workflow is meant to support not only model execution,
but also diagnostics, export, plotting, and figure-oriented
research pipelines.

How the five-stage view is organized
------------------------------------

In the current documentation, GeoPrior-v3 is organized into
a **five-stage workflow view**:

1. **Stage-1**
   prepares the initial workflow state, including early data
   processing, feature assembly, and stage-ready inputs.

2. **Stage-2**
   moves into the modeling-facing pipeline, including
   training-ready preparation, scaling or handshake logic,
   and model bring-up.

3. **Stage-3**
   focuses on downstream evaluation-oriented workflow steps,
   which may include diagnostics, calibration, or related
   post-training analysis depending on the run design.

4. **Stage-4**
   covers inference- or build-oriented workflow actions and
   the generation of deployable or analysis-ready outputs.

5. **Stage-5**
   completes the workflow with final export, plotting,
   reporting, or reproducibility-facing outputs.

.. note::

   Earlier internal or inherited documentation may still
   describe a narrower **Stage-1 → Stage-2** pipeline view.
   The current GeoPrior-v3 documentation expands that into a
   clearer five-stage structure so the full application
   workflow can be documented consistently as the project
   evolves.

What stays constant across stages
---------------------------------

Although the exact role of each stage may differ, the same
workflow principles apply throughout.

Across the stages, GeoPrior-v3 generally expects:

- explicit configuration,
- traceable artifacts,
- inspectable logs and outputs,
- stable naming conventions,
- stage-local validation,
- and clear handoff into the next step.

This means that each stage should be understandable not only
as code execution, but also as part of a larger scientific
contract.

Typical stage handoff artifacts
-------------------------------

A GeoPrior-v3 stage may read from or write to several kinds
of workflow artifacts.

Common examples include:

- processed arrays or tensors,
- exported NPZ bundles,
- scalers or encoders,
- metadata manifests,
- scaling or unit contracts,
- diagnostics JSON files,
- trained model bundles,
- forecast CSV outputs,
- figures and plot assets.

The exact files depend on the stage and application mode, but
the principle is the same: **each stage should leave behind a
traceable artifact boundary**.

Why artifact boundaries matter
------------------------------

Artifact boundaries are important because they make the
workflow inspectable.

For example, instead of assuming that the next stage received
the right data, you can inspect:

- whether a tensor export exists,
- whether scaling metadata was written,
- whether the output timestamp matches the current run,
- whether diagnostics indicate a mismatch,
- whether the next stage is using the intended manifest.

That is much safer than treating the workflow as a black box.

Relationship between configuration and stages
---------------------------------------------

The workflow stages are controlled by configuration.

A properly initialized configuration should define the
practical context of the run, including items such as:

- local paths,
- dataset or case-study identifiers,
- feature and artifact settings,
- runtime toggles,
- output locations,
- stage-specific options.

In practice, this means that users should not think of the
stages as isolated scripts. They should think of them as
**configuration-driven workflow steps** within one coherent
project run.

How a typical run progresses
----------------------------

A typical GeoPrior-v3 run follows this pattern:

1. initialize or review configuration;
2. run the earliest stage that prepares the workflow inputs;
3. inspect generated artifacts and basic diagnostics;
4. continue into modeling-facing stages;
5. inspect results, diagnostics, and exported summaries;
6. move to inference, build, plotting, or reproducibility
   steps as needed.

This pattern is more robust than trying to jump directly to a
late stage before confirming that earlier workflow contracts
were satisfied.

Recommended user mindset
------------------------

When using the workflow, it is best to think in terms of
**progressive validation**.

At each stage, ask:

- Did the stage read the intended config?
- Were the expected inputs found?
- Were outputs written where expected?
- Are shapes, units, and basic summaries plausible?
- Is the next stage reading the correct artifacts?

This mindset helps avoid one of the most common mistakes in
scientific workflow usage: assuming that command completion
automatically means scientific correctness.

How this connects to the CLI
----------------------------

GeoPrior-v3 exposes a command-line workflow surface through
dedicated entry points such as:

- ``geoprior``
- ``geoprior-run``
- ``geoprior-build``
- ``geoprior-plot``
- ``geoprior-init``

These entry points are part of the intended user experience.
The workflow is therefore not documented only as internal
Python code, but also as a real command-driven application
surface.

The stage pages in this section should be read together with:

- :doc:`cli`
- :doc:`configuration`

Those pages explain how the workflow is launched and how the
configuration layer controls it.

How this connects to the scientific foundations
-----------------------------------------------

The workflow is not independent from the scientific design.

In GeoPrior-v3, choices about:

- scaling,
- coordinates,
- units,
- physical residuals,
- and identifiability assumptions

can strongly affect what happens during later stages of the
workflow.

That is why users should not treat the workflow guide as
separate from the scientific foundations. A well-structured
run still depends on well-posed scientific assumptions.

In particular, the following pages become important once the
workflow reaches physics-guided execution:

- :doc:`../scientific_foundations/data_and_units`
- :doc:`../scientific_foundations/scaling`
- :doc:`../scientific_foundations/physics_formulation`
- :doc:`../scientific_foundations/identifiability`

Best practices for working stage by stage
-----------------------------------------

The most reliable way to use GeoPrior-v3 is to move through
the workflow incrementally.

Good practice includes:

- starting from a reviewed configuration;
- running one stage at a time when bringing up a project;
- inspecting artifacts before moving onward;
- keeping runs organized by output directory or manifest;
- avoiding ad hoc code edits when configuration is enough;
- using diagnostics and audits as part of the workflow, not
  as optional extras.

Bad practice includes:

- skipping directly to a late stage without checking earlier
  outputs;
- mixing artifacts from multiple incompatible runs;
- assuming old configs remain valid after workflow changes;
- interpreting forecasts before checking scaling and units.

A compact workflow map
----------------------

The GeoPrior-v3 workflow can be summarized like this:

.. code-block:: text

   initialize config
        ↓
   Stage-1: prepare inputs and early artifacts
        ↓
   Stage-2: bring up modeling-facing workflow
        ↓
   Stage-3: evaluate, diagnose, calibrate, refine
        ↓
   Stage-4: infer, build, or assemble final outputs
        ↓
   Stage-5: export, plot, and support reproducibility

This is a conceptual overview. The exact mechanics of each
stage are described in their dedicated pages.

Read the stages next
--------------------

The next best step is to move from the overview into the
individual stages.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Stage-1
      :link: stage1
      :link-type: doc
      :class-card: sd-shadow-sm

      Learn how the workflow begins, how initial inputs are
      prepared, and what the first artifact boundary looks
      like.

   .. grid-item-card:: Stage-2
      :link: stage2
      :link-type: doc
      :class-card: sd-shadow-sm

      See how the workflow transitions into model-facing
      execution and stage-to-stage validation.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc
      :class-card: sd-shadow-sm card--configuration
      :data-preview: configuration.png
      :data-preview-label: GeoPrior configuration layer

      Understand the configuration system that controls the
      staged workflow.

   .. grid-item-card:: CLI guide
      :link: cli
      :link-type: doc
      :class-card: sd-shadow-sm card--cli
      :data-preview: cli.png
      :data-preview-label: GeoPrior command-line interface

      Move from the workflow concept into the actual command
      surface.

.. seealso::

   - :doc:`../getting_started/first_project_run`
   - :doc:`configuration`
   - :doc:`cli`
   - :doc:`stage1`
   - :doc:`stage2`
   - :doc:`../scientific_foundations/data_and_units`
   - :doc:`../scientific_foundations/scaling`