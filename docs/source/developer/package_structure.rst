Package structure
=================

GeoPrior-v3 is organized as a **workflow-oriented scientific
package**.

Its structure reflects three main goals:

- keep the **scientific model core** separate from the
  workflow wrappers;
- keep the **CLI and reproducibility layers** separate from
  the model implementation;
- keep the **documentation, resources, and paper-facing
  scripts** easy to evolve without destabilizing the model
  stack.

This page explains how the repository is organized, how the
major packages relate to each other, and where developers
should add new functionality.

Why the structure matters
-------------------------

GeoPrior-v3 is not only a library of neural-network classes.

It is also:

- a staged workflow system,
- a reproducible scientific software project,
- and a paper-facing application framework.

That means the repository structure must support more than one
kind of work:

- model development,
- CLI integration,
- scientific diagnostics,
- transfer benchmarking,
- artifact generation,
- and figure production.

A useful summary is:

.. code-block:: text

   model core
      + workflow wrappers
      + config/resources
      + scripts/build/plot layer
      + documentation
      = GeoPrior-v3

Top-level repository layout
---------------------------

A simplified repository view looks like this:

.. code-block:: text

   geoprior-v3/
   ├── geoprior/
   │   ├── __init__.py
   │   ├── cli/
   │   ├── models/
   │   ├── resources/
   │   └── utils/
   ├── nat.com/
   ├── scripts/
   ├── docs/
   ├── pyproject.toml
   └── ...

The role of each major area is:

``geoprior/``
    Main Python package.

``nat.com/``
    Active configuration root used by the workflow.

``scripts/``
    Paper-facing commands, plot scripts, and reproducibility
    helpers.

``docs/``
    Sphinx documentation source.

``pyproject.toml``
    Packaging, dependencies, and CLI entry-point definition.

A good developer rule is:

- code that belongs to the reusable package goes in
  ``geoprior/``,
- code that belongs to paper workflows or figure production
  usually goes in ``scripts/``.

The main Python package
-----------------------

The ``geoprior/`` package is the core of the project.

A simplified internal view is:

.. code-block:: text

   geoprior/
   ├── __init__.py
   ├── cli/
   ├── models/
   ├── resources/
   └── utils/

These subpackages have distinct responsibilities and should
stay conceptually separated.

``geoprior.__init__``
---------------------

The top-level package should remain lightweight and define the
high-level project identity.

In practice, it is the place for:

- package metadata,
- selected public re-exports,
- lightweight import-time dependency checks,
- and a concise package-level description.

A good maintenance rule is:

- avoid turning ``geoprior/__init__.py`` into a dumping
  ground for many deep imports.

CLI layer
---------

The ``geoprior/cli/`` package contains the **workflow-facing
command wrappers**.

A simplified view is:

.. code-block:: text

   geoprior/cli/
   ├── __init__.py
   ├── __main__.py
   ├── _dispatch.py
   ├── init_config.py
   ├── stage1.py
   ├── stage2.py
   ├── stage3.py
   ├── stage4.py
   └── stage5.py

This layer is responsible for:

- root and family-based command dispatch,
- config installation and overrides,
- stage wrapper entry points,
- stable public CLI behavior.

What belongs here
~~~~~~~~~~~~~~~~~

Good candidates for ``geoprior/cli/`` include:

- thin stage wrappers,
- shared argument-handling helpers,
- command registry and dispatch helpers,
- config bootstrap commands.

What does **not** belong here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Avoid putting these directly in the CLI layer:

- heavy scientific logic,
- deep model-building internals,
- figure rendering code,
- large preprocessing logic that should live in reusable
  helpers.

A good rule is:

- the CLI layer should **call** the workflow,
- not **contain** the scientific core.

Model layer
-----------

The ``geoprior/models/`` package contains the **scientific
model implementation**.

For the current GeoPrior-v3 application, the most important
part is the subsidence model family and its tuner.

A simplified view is:

.. code-block:: text

   geoprior/models/
   ├── __init__.py
   ├── forecast_tuner/
   └── subsidence/

This separation is important:

- ``subsidence/`` contains the flagship scientific model
  stack;
- ``forecast_tuner/`` contains the tuning infrastructure for
  that model family.

Subsidence model package
~~~~~~~~~~~~~~~~~~~~~~~~

The subsidence package contains the core scientific logic:

.. code-block:: text

   geoprior/models/subsidence/
   ├── __init__.py
   ├── models.py
   ├── maths.py
   ├── scaling.py
   ├── losses.py
   ├── identifiability.py
   ├── payloads.py
   ├── plot.py
   ├── stability.py
   ├── step_core.py
   ├── derivatives.py
   └── utils.py

A practical separation of concerns is:

``models.py``
    public model classes and their Keras-facing behavior

``maths.py``
    closure formulas, field composition, and math helpers

``scaling.py``
    scaling contract and serialization

``losses.py``
    packed results and loss-oriented helpers

``identifiability.py``
    regimes, locks, compile defaults, and audits

``payloads.py``
    saved physics payload creation and loading

``stability.py``
    gradient filtering, gates, and safety helpers

``step_core.py``
    shared physics-core execution path

``utils.py``
    SI conversion and subsidence-specific support helpers

A good developer rule is:

- keep reusable scientific logic in these focused modules
  rather than pushing everything into ``models.py``.

Forecast tuner package
~~~~~~~~~~~~~~~~~~~~~~

The tuner stack lives separately:

.. code-block:: text

   geoprior/models/forecast_tuner/
   ├── __init__.py
   ├── tuners.py
   ├── _base_tuner.py
   └── _geoprior_tuner.py

This is a good example of architecture discipline:

- generic tuning logic belongs in the base layer,
- GeoPrior-specific tuner behavior belongs in the specialized
  tuner module,
- public imports are re-exported from the package surface.

This keeps the tuning API clean while still allowing internal
evolution.

Utilities layer
---------------

The ``geoprior/utils/`` package contains **workflow-facing
reusable helpers** that do not belong directly inside the CLI
or model core.

Typical responsibilities include:

- config loading,
- artifact discovery,
- forecast formatting,
- diagnostics utilities,
- holdout logic,
- geospatial helpers,
- sequence generation,
- runtime helpers.

A useful mental model is:

- ``geoprior/models/...`` contains model-specific scientific
  logic;
- ``geoprior/utils/...`` contains broader workflow helpers.

This distinction is important because not every reusable
helper belongs in the model namespace.

What belongs in ``geoprior/utils/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Good candidates include:

- helpers used across several stages,
- utilities needed by scripts and build commands,
- forecast/export formatting logic,
- audit and validation helpers,
- data-contract logic shared by more than one stage.

What does **not** belong there
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Avoid placing these in top-level utils:

- core PDE or closure math,
- model-layer internals tied tightly to one neural class,
- one-off paper plotting code.

Those belong in the model package or scripts layer instead.

Resources layer
---------------

The ``geoprior/resources/`` area contains **packaged runtime
resources**, especially templates.

The most important current example is the NATCOM config
template used during bootstrap.

A simplified view is:

.. code-block:: text

   geoprior/resources/
   └── natcom_config_template.py

This is not the place for active workflow code. It is the
place for:

- template files,
- packaged default assets,
- schema-like resource text,
- static project resources needed at runtime.

A good maintenance rule is:

- keep resource files simple, stable, and easy to render.

Active config root
------------------

The ``nat.com/`` directory is the **active configuration
root** used by the workflow.

A simplified view is:

.. code-block:: text

   nat.com/
   ├── config.py
   └── config.json

Its purpose is different from ``geoprior/resources/``:

- ``resources/`` contains packaged defaults and templates;
- ``nat.com/`` contains the active, user-facing config state
  for the current project or run environment.

This is a very important distinction for developers.

A good rule is:

- do not hard-code project-specific values into packaged
  resources when they belong in the active config root.

Scripts layer
-------------

The ``scripts/`` package is the **paper- and artifact-facing
command layer**.

A simplified view is:

.. code-block:: text

   scripts/
   ├── __init__.py
   ├── __main__.py
   ├── registry.py
   ├── plot_physics_fields.py
   ├── plot_litho_parity.py
   └── ...

This layer is intended for:

- figure generation,
- reproducibility scripts,
- post-processing commands,
- paper-oriented summary and plotting workflows.

Why this stays separate
~~~~~~~~~~~~~~~~~~~~~~~

Keeping scripts separate from ``geoprior/`` helps maintain a
clean architecture:

- the package stays reusable,
- the paper workflow remains explicit,
- figure code does not clutter the scientific model core.

A good developer rule is:

- if code is mainly for manuscript figures, reports, or
  reproducibility scripts, it probably belongs in
  ``scripts/``.

How the scripts layer relates to the CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The modern CLI can dispatch to commands that conceptually
belong to the scripts layer through ``build`` and ``plot``
families.

That means the architecture is:

- one unified public command surface,
- but still a clean code separation underneath.

This is a strong design and should be preserved.

Documentation layer
-------------------

The ``docs/`` tree contains the Sphinx source for the
project documentation.

A simplified view is:

.. code-block:: text

   docs/
   └── source/
       ├── getting_started/
       ├── user_guide/
       ├── scientific_foundations/
       ├── applications/
       ├── api/
       ├── developer/
       └── ...

This structure mirrors the intended audience split:

- users starting the project,
- users running workflows,
- readers interested in the scientific model,
- users focused on applications,
- developers working with the codebase.

That is exactly the right direction for a project like
GeoPrior-v3.

How layers should depend on each other
--------------------------------------

A good dependency direction is:

.. code-block:: text

   resources/config
        ↓
   utils
        ↓
   models
        ↓
   cli
        ↓
   scripts/docs

More practically:

- models should not depend on the CLI layer;
- utilities should not depend on manuscript scripts;
- resources should stay lightweight and not import heavy
  runtime logic;
- scripts may depend on package utilities and saved
  artifacts;
- docs describe all of the above but are not part of runtime.

This is not a rigid law, but it is a good architectural
guide.

Where to add new code
---------------------

A new CLI command
~~~~~~~~~~~~~~~~~

Add it when the feature is a workflow action exposed to users.

Likely places:

- ``geoprior/cli/`` for stage/wrapper commands,
- ``scripts/`` plus registry integration for plot/build/paper
  commands.

A new scientific model helper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add it when the feature is tightly tied to the subsidence
model family.

Likely places:

- ``geoprior/models/subsidence/maths.py``
- ``geoprior/models/subsidence/utils.py``
- ``geoprior/models/subsidence/stability.py``
- or another focused subsidence module

A new general workflow helper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add it when the feature is reusable across stages or scripts.

Likely place:

- ``geoprior/utils/``

A new template or packaged asset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add it when the feature is a runtime resource, not a Python
workflow function.

Likely place:

- ``geoprior/resources/``

A new paper or figure script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add it when the feature is mainly for reproducibility,
figures, or report generation.

Likely place:

- ``scripts/``

Package boundaries to preserve
------------------------------

For long-term maintainability, these boundaries are worth
preserving.

**CLI vs scientific core**

The CLI should wrap and orchestrate the workflow, not own the
scientific implementation.

**Models vs top-level utils**

Model-specific physics logic should stay near the model, not
become a catch-all utility function.

**Resources vs active config**

Packaged templates and active project configs should stay
distinct.

**Scripts vs reusable package**

Paper and figure workflows should stay separate from the main
library surface unless they become generally reusable.

How to read the repository as a new developer
---------------------------------------------

A good reading order is:

1. ``geoprior/__init__.py``
2. ``geoprior/cli/__main__.py``
3. ``geoprior/models/subsidence/models.py``
4. ``geoprior/models/subsidence/step_core.py``
5. ``geoprior/models/subsidence/scaling.py``
6. ``geoprior/utils/nat_utils.py``
7. ``scripts/registry.py``
8. ``nat.com/config.py``

That order gives a good progression from:

- package identity,
- to public execution surface,
- to scientific core,
- to workflow helpers,
- to paper-facing commands,
- to active configuration.

Common mistakes
---------------

**Putting heavy scientific logic in the CLI wrapper**

This makes the public command surface hard to maintain.

**Using top-level utils as a dumping ground**

A helper should live where its real responsibility belongs.

**Mixing paper scripts into reusable library code**

This weakens package clarity.

**Treating the active config root as a packaged resource**

Templates and active project state are not the same thing.

**Adding cross-layer imports casually**

That can slowly destroy the architecture.

Best practices
--------------

.. admonition:: Best practice

   Prefer small, responsibility-focused modules.

   GeoPrior already benefits from this in the subsidence
   package and tuner stack.

.. admonition:: Best practice

   Keep public re-export surfaces clean.

   Let package ``__init__.py`` files expose the intended API,
   but keep deep implementation details in focused modules.

.. admonition:: Best practice

   Preserve the stage/workflow separation from the scientific
   model core.

   This is one of the strongest parts of the current
   repository design.

.. admonition:: Best practice

   Treat ``scripts/`` as an application and paper layer, not
   as a substitute for the main package.

.. admonition:: Best practice

   Add new code where its natural long-term home is, not only
   where it is easiest to patch today.

A compact structure map
-----------------------

The GeoPrior-v3 repository can be summarized as:

.. code-block:: text

   geoprior/
     main reusable package
       ├── cli/         public workflow wrappers
       ├── models/      scientific model core
       ├── utils/       reusable workflow helpers
       └── resources/   packaged templates/assets

   nat.com/
     active configuration root

   scripts/
     reproducibility, build, and figure layer

   docs/
     Sphinx documentation source

Relationship to the rest of the developer docs
----------------------------------------------

This page explains **where things live**.

The next developer pages should explain:

- how the docs migrated away from the older FusionLab context,
- how to contribute changes cleanly,
- and how release notes should be maintained.

See also
--------

.. seealso::

   - :doc:`migration_from_fusionlab`
   - :doc:`contributing`
   - :doc:`../release_notes`
   - :doc:`../user_guide/cli`
   - :doc:`../user_guide/configuration`
   - :doc:`../api/cli`
   - :doc:`../api/subsidence`
   - :doc:`../api/utils`
   - :doc:`../api/resources`