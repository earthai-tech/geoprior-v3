Contributing
============

Thank you for contributing to GeoPrior-v3.

GeoPrior-v3 is both:

- a scientific Python package,
- a staged workflow framework,
- and a reproducible research software project.

That means good contributions are not only about “making the
code run.” A strong contribution should also respect:

- the scientific meaning of the model,
- the workflow boundaries between stages,
- the public API and CLI surface,
- and the reproducibility story of the project.

This page explains how to contribute changes cleanly and in a
way that keeps the codebase coherent.

Contribution philosophy
-----------------------

GeoPrior-v3 is organized around a few strong principles.

**1. Keep responsibilities separated**

- scientific model logic belongs in the model stack,
- workflow orchestration belongs in the CLI layer,
- figure and manuscript logic belongs in ``scripts/``.

**2. Prefer explicit contracts**

If a change affects:

- scaling,
- configuration,
- manifests,
- saved payloads,
- or exported CSV/JSON outputs,

then it is usually a **workflow contract change**, not a
small local tweak.

**3. Documentation is part of the contribution**

A change is usually not complete until the relevant docs are
updated as well.

**4. Preserve reproducibility**

A contribution should make the workflow easier to rerun,
audit, and interpret — not harder.

A useful summary is:

.. code-block:: text

   good contribution
      = correct code
      + clear boundaries
      + updated docs
      + preserved reproducibility

What kinds of contributions are welcome
---------------------------------------

GeoPrior-v3 benefits from several kinds of contributions,
including:

- bug fixes,
- documentation improvements,
- workflow and CLI improvements,
- model and physics improvements,
- diagnostics and export improvements,
- tuning and transfer workflow improvements,
- reproducibility and figure-generation improvements.

Examples of especially valuable contributions include:

- fixing wrong assumptions in stage handoff logic,
- improving scaling or unit safety,
- clarifying docs for scientific formulas,
- improving public API clarity,
- making figure scripts more artifact-driven,
- tightening model serialization and payload export.

Before you start
----------------

Before making a non-trivial change, it helps to identify what
kind of change it is.

Ask these questions first:

- Does this belong in the model layer, the CLI layer, the
  utility layer, or the scripts layer?
- Does it change public behavior?
- Does it affect saved artifacts?
- Does it affect the scientific interpretation of the model?
- Does it require documentation updates?

A small bug fix may touch only one file.

A workflow or physics change often requires updates in
several places, such as:

- code,
- docs,
- tests,
- release notes.

Where to make changes
---------------------

Use the package structure intentionally.

Model and scientific changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the change affects:

- the flagship model,
- physics closures,
- scaling,
- identifiability,
- residual assembly,
- stability behavior,

then it likely belongs under:

.. code-block:: text

   geoprior/models/subsidence/

Do **not** put scientific model logic into the CLI wrappers.

Workflow and CLI changes
~~~~~~~~~~~~~~~~~~~~~~~~

If the change affects:

- stage entry points,
- command dispatch,
- config bootstrap,
- CLI arguments,
- family routing,

then it likely belongs under:

.. code-block:: text

   geoprior/cli/

Try to keep this layer thin. It should orchestrate and
forward, not absorb the scientific core.

Workflow-facing utilities
~~~~~~~~~~~~~~~~~~~~~~~~~

If the change is reusable across stages or scripts and does
not belong tightly inside one model module, it likely belongs
under:

.. code-block:: text

   geoprior/utils/

Examples include:

- artifact discovery,
- forecast formatting,
- holdout helpers,
- config loading,
- audit helpers.

Resources and templates
~~~~~~~~~~~~~~~~~~~~~~~

If the change affects packaged templates or static runtime
resources, it likely belongs under:

.. code-block:: text

   geoprior/resources/

Keep resource files simple and renderable.

Paper and figure scripts
~~~~~~~~~~~~~~~~~~~~~~~~

If the change is mainly for:

- plotting,
- manuscript figures,
- reproducibility scripts,
- paper-facing summaries,

then it likely belongs under:

.. code-block:: text

   scripts/

Do not move paper-specific logic into the reusable package
unless it has become genuinely general-purpose.

General workflow for a contribution
-----------------------------------

A good contribution flow looks like this:

1. identify the correct layer for the change;
2. make the smallest clean change that solves the real
   problem;
3. update docs if behavior or usage changed;
4. add or update tests if behavior changed;
5. update release notes if the change is user-visible;
6. make sure the change still fits the package structure.

A compact version is:

.. code-block:: text

   understand the change
        ↓
   edit the right layer
        ↓
   update docs and notes
        ↓
   verify behavior
        ↓
   submit cleanly

Coding style and formatting
---------------------------

GeoPrior-v3 benefits from a consistent, disciplined coding
style.

Current project preferences include:

- Black-formatted code,
- relatively short lines,
- focused modules with clear responsibilities,
- explicit naming over cleverness,
- high-quality docstrings.

For this project, shorter lines are especially helpful in:

- docstrings,
- equations,
- dense scientific parameter lists,
- code that is frequently reviewed inside docs.

A good rule is:

- write for clarity and maintainability first.

Docstrings
~~~~~~~~~~

Prefer well-structured docstrings for public classes,
functions, and methods.

Especially for scientific code, docstrings should make clear:

- what the object does,
- what the inputs mean physically,
- what the outputs represent,
- and whether units, shapes, or conventions matter.

A strong docstring is often the first line of defense against
misuse in a scientific workflow.

Naming
~~~~~~

Prefer names that make the role of the object clear.

Examples of good naming habits:

- use ``stage`` language for workflow wrappers,
- use physics-aware names for physical quantities,
- keep helper names close to the scientific or workflow idea
  they implement,
- avoid ambiguous abbreviations unless they are standard in
  the domain.

A good rule is:

- if a future contributor cannot guess the module’s purpose
  from its name, the name may be too vague.

Documentation expectations
--------------------------

Documentation is not optional for significant changes.

When you change code, ask which docs need to move with it.

If you change the CLI
~~~~~~~~~~~~~~~~~~~~~

Update pages such as:

- ``user_guide/cli.rst``
- ``user_guide/configuration.rst``
- relevant stage pages
- ``api/cli.rst``

If you change the model or scientific logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update pages such as:

- ``scientific_foundations/geoprior_subsnet.rst``
- ``physics_formulation.rst``
- ``residual_assembly.rst``
- ``losses_and_training.rst``
- ``data_and_units.rst``
- ``scaling.rst``
- ``identifiability.rst``
- ``api/subsidence.rst``

If you change tuning or transfer workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update pages such as:

- ``applications/tuner_workflow.rst``
- ``user_guide/stage3.rst``
- ``user_guide/stage5.rst``
- ``api/tuner.rst``

If you change reproducibility or plotting behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update pages such as:

- ``applications/reproducibility_scripts.rst``
- ``applications/figure_generation.rst``
- relevant CLI or API pages

A good rule is:

- if users would notice the change, the docs should notice
  it too.

How to treat scientific changes
-------------------------------

Scientific changes deserve extra care.

Examples include changes to:

- physical formulas,
- scaling conventions,
- timescale priors,
- bounds semantics,
- drawdown logic,
- uncertainty interpretation,
- identifiability regimes.

These are not just implementation changes. They can alter the
meaning of trained models and exported artifacts.

So when making scientific changes:

- explain the motivation clearly,
- update the relevant scientific docs,
- check whether saved artifacts or configs remain compatible,
- record user-visible impact in the release notes.

A good rule is:

- if the change affects interpretation, document it like a
  scientific change, not only like a refactor.

How to treat workflow contract changes
--------------------------------------

Some changes affect the workflow contract even if they do not
change the model equations directly.

Examples include changes to:

- Stage-1 manifest structure,
- config fields,
- payload formats,
- evaluation JSON schemas,
- output CSV conventions,
- CLI argument meaning.

These changes require extra care because they can break:

- downstream scripts,
- figure pipelines,
- saved-model reuse,
- or cross-stage compatibility.

When making this kind of change:

- keep backward compatibility where reasonable;
- if not possible, document the break clearly;
- update any affected docs and examples;
- note it in release notes.

Tests and validation
--------------------

When behavior changes, the contribution should be verified.

The exact verification depends on the type of change.

For model changes
~~~~~~~~~~~~~~~~~

Useful validation often includes:

- shape and forward-pass checks,
- serialization / deserialization checks,
- stability checks,
- regime- or closure-related behavior checks.

For workflow changes
~~~~~~~~~~~~~~~~~~~~

Useful validation often includes:

- config resolution,
- stage handoff behavior,
- artifact generation,
- CLI argument behavior,
- manifest consistency.

For docs changes
~~~~~~~~~~~~~~~~

Useful validation includes:

- confirming Sphinx references resolve,
- checking code paths in examples,
- verifying that ``literalinclude`` paths are correct,
- ensuring page navigation still makes sense.

A good rule is:

- verify the layer you changed and any layer it directly
  affects.

Backward compatibility
----------------------

Not every contribution must preserve every old path, but
contributors should think about compatibility explicitly.

Try to preserve compatibility when the change affects:

- public imports,
- CLI command names,
- stage entry patterns,
- output artifact naming,
- saved model reload behavior.

If a break is unavoidable:

- keep it as small as possible,
- document it clearly,
- update release notes,
- update examples and docs.

This is especially important in GeoPrior-v3 because the
package serves both:

- interactive users,
- and longer-lived reproducibility workflows.

How to contribute docs
----------------------

Documentation-only contributions are valuable and encouraged.

Good documentation improvements include:

- clarifying scientific assumptions,
- fixing stale links,
- improving page structure,
- updating imports and code examples,
- exposing comments or source listings more clearly,
- improving migration and developer guidance.

A good docs contribution should aim to make the next
contributor faster and less confused.

How to contribute CLI improvements
----------------------------------

CLI contributions are especially useful when they improve:

- discoverability,
- consistency,
- argument clarity,
- stage-to-stage continuity,
- config override behavior.

But keep the CLI layer disciplined.

A good CLI contribution should:

- preserve the thin-wrapper philosophy,
- avoid dragging in scientific logic,
- update user-guide and API docs together,
- keep aliases and help text coherent.

How to contribute model improvements
------------------------------------

Model contributions are especially valuable when they improve:

- scientific clarity,
- numerical stability,
- interpretability,
- identifiability,
- uncertainty handling,
- or serialization robustness.

But model changes should stay modular.

A good model contribution should avoid:

- turning ``models.py`` into a monolith,
- hiding scientific assumptions in obscure helpers,
- adding configuration behavior without documenting it,
- creating silent differences between training and
  inference-time interpretation.

Pull request checklist
----------------------

Before submitting, check the following.

Code and structure
~~~~~~~~~~~~~~~~~~

- the change lives in the correct package layer
- module boundaries still make sense
- naming is clear
- formatting is clean

Behavior
~~~~~~~~

- the intended behavior is verified
- no unrelated behavior changed accidentally
- public API changes are deliberate

Documentation
~~~~~~~~~~~~~

- relevant docs were updated
- examples still make sense
- API pages remain aligned
- developer notes were updated if needed

Release notes
~~~~~~~~~~~~~

- user-visible changes are recorded
- breaking or behavior-changing updates are explained

A compact checklist is:

.. code-block:: text

   right layer?
   right behavior?
   docs updated?
   notes updated?
   easy to review?

Common mistakes
---------------

**Putting scientific logic in the CLI layer**

This makes the workflow wrappers harder to maintain.

**Using top-level utils as a dumping ground**

A helper should live where its real responsibility belongs.

**Changing a public workflow contract without updating docs**

This creates silent confusion for users.

**Making a scientific change sound like a refactor**

Interpretation-changing changes should be documented
explicitly.

**Updating code but not release notes**

This weakens project history and migration clarity.

**Writing docs as an afterthought**

For a scientific workflow package, docs are part of the
product.

Review mindset
--------------

When reviewing your own contribution, ask:

- Would a new contributor know where this code belongs?
- Does this preserve the structure explained in
  :doc:`package_structure`?
- If a user reads the docs, will they understand the new
  behavior?
- If someone reruns an old workflow, what changes?
- Is the scientific meaning clearer or more ambiguous after
  this change?

These questions are often more useful than only asking
whether the code “works.”

Best practices
--------------

.. admonition:: Best practice

   Make the smallest clean change that solves the real
   problem.

   Small, clear contributions are easier to review and easier
   to trust.

.. admonition:: Best practice

   Keep boundaries intact.

   GeoPrior-v3 stays maintainable because the model, CLI,
   utilities, resources, and scripts layers remain distinct.

.. admonition:: Best practice

   Update docs and release notes together with user-visible
   changes.

   This prevents silent drift between code and documentation.

.. admonition:: Best practice

   Treat scientific assumptions as public behavior.

   If a change affects interpretation, document it clearly.

.. admonition:: Best practice

   Prefer artifact-driven and reproducible workflows.

   Avoid contributions that hide important work behind
   side effects or implicit state.

A compact contribution map
--------------------------

Contributing to GeoPrior-v3 can be summarized as:

.. code-block:: text

   identify the right layer
        ↓
   implement a clean, focused change
        ↓
   verify behavior
        ↓
   update docs
        ↓
   update release notes if needed
        ↓
   keep the workflow reproducible and interpretable

Relationship to the rest of the developer docs
----------------------------------------------

This page explains **how to contribute well**.

The surrounding developer pages explain:

- :doc:`package_structure`
  where different kinds of code should live;

- :doc:`migration_from_fusionlab`
  how to handle older documentation context and stale
  references;

- :doc:`../release_notes`
  how to record changes cleanly in project history.

See also
--------

.. seealso::

   - :doc:`package_structure`
   - :doc:`migration_from_fusionlab`
   - :doc:`../release_notes`
   - :doc:`../api/cli`
   - :doc:`../api/subsidence`
   - :doc:`../api/tuner`
   - :doc:`../api/utils`
   - :doc:`../user_guide/cli`
   - :doc:`../user_guide/configuration`