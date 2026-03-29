CLI guide
=========

GeoPrior-v3 provides a **family-based command-line
interface** for staged workflows, deterministic build steps,
and configuration bootstrapping.

The CLI is one of the main public interfaces of the project.
It is not an afterthought layered on top of the Python API.
The package metadata registers dedicated console scripts for:

- ``geoprior``
- ``geoprior-run``
- ``geoprior-build``
- ``geoprior-plot``
- ``geoprior-init``

These entry points are part of the intended workflow surface
for GeoPrior-v3. The documentation homepage and overview
already present the project as a workflow-oriented framework
with staged CLI execution as a first-class capability. 

What the CLI is designed to do
------------------------------

The CLI is designed around three ideas:

- **family-based dispatch**, so run, build, and plot tasks
  remain clearly separated;
- **configuration-driven execution**, so commands can install
  a config, apply one-off overrides, and persist the
  effective runtime settings;
- **stable aliases**, so users can work with intuitive names
  such as ``train``, ``tune``, ``infer``, or ``transfer``
  without losing the canonical command names underneath. 

The CLI layer is intentionally thin. Shared parser behavior
is centralized in reusable helpers, while each command keeps
its own domain-specific artifact logic and validation. 

CLI structure at a glance
-------------------------

GeoPrior-v3 supports two closely related ways to access the
CLI.

**1. Root entry point**

The root entry point requires a **family token** first:

.. code-block:: text

   geoprior run <command> [args]
   geoprior build <command> [args]
   geoprior plot <command> [args]

The dispatcher treats ``make`` as an alias of ``build``. If a
family token is missing, the root CLI raises an error and
prints help. 

**2. Family-scoped entry points**

You can also call the family directly:

.. code-block:: text

   geoprior-run <command> [args]
   geoprior-build <command> [args]
   geoprior-plot <command> [args]

These are convenience entry points bound to a single family.
They are usually the most convenient way to work once you
already know which family you need. 

There is also a dedicated initialization entry point:

.. code-block:: text

   geoprior-init

which maps directly to the config bootstrap command. 

The main command families
-------------------------

GeoPrior-v3 currently organizes commands into three
families.

Run family
~~~~~~~~~~

The ``run`` family is the main staged workflow surface.

Its registered commands include:

- ``init-config``
- ``stage1-preprocess``
- ``stage2-train``
- ``stage3-tune``
- ``stage4-infer``
- ``stage5-transfer``
- ``sensitivity``
- ``sm3-identifiability``
- ``sm3-offset-diagnostics``
- ``sm3-suite`` 

Build family
~~~~~~~~~~~~

The ``build`` family is for deterministic artifact-generation
steps that operate on existing workflow outputs.

Its registered commands include:

- ``full-inputs-npz``
- ``physics-payload-npz``
- ``external-validation-fullcity``
- ``sm3-collect-summaries``
- ``assign-boreholes``
- ``add-zsurf-from-coords``
- ``external-validation-metrics`` 

Plot family
~~~~~~~~~~~

The ``plot`` family exists in the dispatcher and has its own
entry point, but the current CLI registry does **not yet**
list any plot commands. The help printer explicitly reports
that no plot commands are registered yet. So the family is
already part of the public CLI design, but it is not yet
populated in the current command registry. 

Canonical commands and aliases
------------------------------

GeoPrior-v3 supports a large alias layer so that users can
work with intuitive names while the dispatcher still resolves
everything to canonical commands.

Examples include:

- ``preprocess`` or ``stage1`` → ``stage1-preprocess``
- ``train`` or ``fit`` → ``stage2-train``
- ``tune`` or ``search`` → ``stage3-tune``
- ``infer`` or ``predict`` → ``stage4-infer``
- ``transfer`` or ``xfer`` → ``stage5-transfer``
- ``init`` or ``bootstrap`` → ``init-config``

The build family also has practical aliases such as:

- ``full-inputs`` → ``full-inputs-npz``
- ``physics-payload`` → ``physics-payload-npz``
- ``external-validation`` → ``external-validation-fullcity``

This means the CLI is friendly to both exact command names
and shorter workflow-oriented names. 

A good mental model
-------------------

A useful way to think about the CLI is:

- ``geoprior`` is the **root dispatcher**
- ``geoprior-run`` is the **workflow runner**
- ``geoprior-build`` is the **artifact builder**
- ``geoprior-plot`` is the **plot family entry point**
- ``geoprior-init`` is the **config bootstrapper** 

If you already know what family you want, the family-scoped
entry points are usually cleaner. If you want the whole CLI
from one root entry point, use ``geoprior`` with a family
token.

Basic usage patterns
--------------------

Inspect the main help pages first:

.. code-block:: bash

   geoprior --help
   geoprior-run --help
   geoprior-build --help
   geoprior-plot --help
   geoprior-init --help

Typical root-entry usage looks like:

.. code-block:: bash

   geoprior run stage1-preprocess
   geoprior run stage2-train
   geoprior build full-inputs-npz

The same actions through family entry points look like:

.. code-block:: bash

   geoprior-run stage1-preprocess
   geoprior-run train
   geoprior-run tune
   geoprior-build full-inputs-npz

The help text itself prints examples such as:

- ``geoprior run stage1-preprocess``
- ``geoprior run sensitivity --epochs 10 --gold``
- ``geoprior build full-inputs-npz --stage1-dir results/foo_stage1``
- ``geoprior make full-inputs`` 

Initialization and config bootstrap
-----------------------------------

GeoPrior-v3 provides a dedicated initialization command for
creating ``nat.com/config.py`` interactively. The template it
writes is an auto-generated NATCOM config for GeoPrior-v3,
and the command also refreshes ``config.json``. The built-in
template covers dataset identity, file naming, time layout,
required columns, physical interpretation of groundwater,
feature groups, training defaults, and scaling settings. 

Typical usage:

.. code-block:: bash

   geoprior-init
   geoprior-init --yes
   geoprior-init --city zhongshan --time-steps 6

The initializer itself prints practical next commands such as:

- ``geoprior-run preprocess``
- ``geoprior-run train``
- ``geoprior-run tune``
- ``geoprior-run infer --help``
- ``geoprior-run transfer --help`` 

You can also access the same bootstrap logic through the run
family:

.. code-block:: bash

   geoprior-run init-config

or through the module form shown by the initializer comments:

.. code-block:: bash

   python -m geoprior.cli init-config

The dispatcher logic is written so that module-style use also
remains supported. 

Shared configuration behavior
-----------------------------

Many GeoPrior commands share a common configuration pattern.
The shared CLI config helpers support:

- ``--config`` to install a user-supplied ``config.py`` into
  the active config root before running;
- ``--config-root`` to choose the config root directory;
- repeated ``--set KEY=VALUE`` overrides to patch the active
  runtime config for one command;
- mapping selected parsed fields such as ``--city`` or
  ``--model`` back onto config keys;
- persisting the effective runtime config into
  ``config.json``. 

This means a command can be launched in a reproducible way
without manual code edits.

Typical pattern:

.. code-block:: bash

   geoprior-run stage1-preprocess \
     --config my_config.py \
     --set TIME_STEPS=6 \
     --set FORECAST_HORIZON_YEARS=3

The helper layer parses booleans, ``None``, numbers, and even
literal containers conservatively from ``--set`` values, then
writes the effective merged config to ``config.json``. 

Common shared arguments
-----------------------

Not every command exposes every option, but the shared helper
layer defines reusable arguments that many commands draw on.

Common examples include:

- ``--config``
- ``--config-root``
- ``--set KEY=VALUE``
- ``--city``
- ``--model``
- ``--results-dir`` or ``--results-root``
- ``--manifest``
- ``--stage1-dir``
- ``--stage2-manifest``
- ``--split``
- ``--outdir``
- ``--format``
- ``--output-stem`` 

In practice, the most important ones to remember are the
config-related flags and the manifest or stage-directory
arguments used for artifact-driven commands.

Run-family workflow commands
----------------------------

The run family is the main pipeline surface for GeoPrior-v3.
Its canonical stage commands map directly onto the staged
workflow documented elsewhere in this guide:

- Stage-1 preprocessing → ``stage1-preprocess``
- Stage-2 training → ``stage2-train``
- Stage-3 tuning → ``stage3-tune``
- Stage-4 inference → ``stage4-infer``
- Stage-5 transfer evaluation → ``stage5-transfer`` 

Typical examples:

.. code-block:: bash

   geoprior-run stage1-preprocess
   geoprior-run stage2-train
   geoprior-run stage3-tune
   geoprior-run stage4-infer --help
   geoprior-run stage5-transfer --help

The same commands can be called through aliases:

.. code-block:: bash

   geoprior-run preprocess
   geoprior-run train
   geoprior-run tune
   geoprior-run infer
   geoprior-run transfer

These aliases are resolved internally to the canonical stage
commands. 

Supplementary run commands
--------------------------

The run family also includes supplementary diagnostic and
sensitivity workflows:

- ``sensitivity``
- ``sm3-identifiability``
- ``sm3-offset-diagnostics``
- ``sm3-suite`` 

These sit outside the main Stage-1 → Stage-5 sequence but
still belong to the run family because they execute workflow
logic rather than deterministic build materialization.

Build-family commands
---------------------

The build family is for deterministic artifact assembly and
post-processing from existing workflow outputs. In other
words, these commands generally consume artifacts that
already exist, rather than running the whole model-training
pipeline again. The dispatcher groups them under a dedicated
build family, and the helper layer includes manifest and
stage-directory arguments specifically for this kind of
artifact-driven work. 

Typical examples:

.. code-block:: bash

   geoprior-build full-inputs-npz --stage1-dir results/foo_stage1
   geoprior-build physics-payload-npz --help
   geoprior-build external-validation-fullcity --help

There is also a root alias for the family:

.. code-block:: bash

   geoprior make full-inputs

because ``make`` is treated as a family alias for
``build``. 

Current plot-family status
--------------------------

The CLI surface already reserves a plot family through both
the project scripts and the dispatcher. However, the current
registry does not yet expose concrete plot commands. The help
printer explicitly reports that no plot commands are
registered yet. So it is best to describe ``geoprior-plot``
as an **available family entry point with future expansion
room**, not as a currently populated plotting surface. 

Command family rules and errors
-------------------------------

The dispatcher enforces command-family boundaries.

For example:

- the root entry point requires a family token first;
- a command that belongs to ``build`` cannot be dispatched as
  part of ``run``;
- family-scoped entry points reject repeated family tokens,
  such as ``geoprior-run run ...``;
- unknown commands trigger help output and exit with an
  error. 

This is useful because it prevents ambiguous or misleading
command dispatch.

A few important examples
------------------------

**Bootstrap a config**

.. code-block:: bash

   geoprior-init --yes --city zhongshan

**Run Stage-1 with overrides**

.. code-block:: bash

   geoprior-run preprocess \
     --config my_config.py \
     --set TIME_STEPS=6 \
     --set FORECAST_HORIZON_YEARS=3

**Train the model**

.. code-block:: bash

   geoprior-run train

**Tune hyperparameters**

.. code-block:: bash

   geoprior-run tune

**Run inference help**

.. code-block:: bash

   geoprior-run infer --help

**Run transfer help**

.. code-block:: bash

   geoprior-run transfer --help

**Build merged inputs from Stage-1 outputs**

.. code-block:: bash

   geoprior-build full-inputs-npz \
     --stage1-dir results/foo_stage1

These examples follow the actual registered entry points,
family aliases, and helper-layer argument design in the
current code. 

CLI and the broader project design
----------------------------------

The CLI matches the broader identity of GeoPrior-v3 as a
workflow-oriented scientific framework rather than only an
importable model package. The top-level package docstring
describes GeoPrior-v3 as **Physics-guided AI for geohazards
& risk analytics**, with land subsidence as the current main
focus and broader geohazard expansion next. The CLI is part
of that application-facing posture. 

One practical note
------------------

There is a small packaging inconsistency worth fixing later:
the top-level package lists ``click`` among its expected
dependencies during import-time checks, but the declared
runtime dependencies in ``pyproject.toml`` do not currently
include ``click``. The current CLI implementation shown here
is built around ``argparse`` and custom dispatch helpers, so
this does not block the docs page, but it is worth cleaning
up in the package metadata for consistency. 

Best practices
--------------

.. admonition:: Best practice

   Start with ``--help`` before guessing subcommand syntax.

   The dispatcher and wrapper layer are explicit, and the
   installed help text is the most reliable source for the
   exact current command form.

.. admonition:: Best practice

   Prefer ``geoprior-run`` and ``geoprior-build`` once you
   know the family you need.

   They are simpler to read and avoid the extra root-family
   token.

.. admonition:: Best practice

   Use ``--config`` and ``--set KEY=VALUE`` instead of making
   ad hoc code edits for one-off runs.

   The shared helper layer is specifically designed to
   install, merge, and persist runtime config cleanly.

.. admonition:: Best practice

   Keep stage commands and build commands conceptually
   separate.

   Run commands execute workflow logic. Build commands
   materialize deterministic artifacts from existing outputs.

A compact CLI map
-----------------

The GeoPrior-v3 CLI can be summarized like this:

.. code-block:: text

   geoprior
     ├── run
     │    ├── init-config
     │    ├── stage1-preprocess
     │    ├── stage2-train
     │    ├── stage3-tune
     │    ├── stage4-infer
     │    ├── stage5-transfer
     │    └── supplementary diagnostics
     │
     ├── build
     │    └── deterministic artifact builders
     │
     └── plot
          └── family reserved; no commands registered yet

   family shortcuts:
     geoprior-run
     geoprior-build
     geoprior-plot
     geoprior-init

Read next
---------

The best next pages after this one are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc
      :class-card: sd-shadow-sm card--configuration
      :data-preview: configuration.png
      :data-preview-label: GeoPrior configuration system

      Learn how the config layer interacts with CLI
      installation, overrides, and runtime persistence.

   .. grid-item-card:: Workflow overview
      :link: workflow_overview
      :link-type: doc
      :class-card: sd-shadow-sm card--workflow
      :data-preview: workflow.png
      :data-preview-label: GeoPrior staged workflow

      See how the CLI maps onto the five-stage workflow.

   .. grid-item-card:: Stage-1
      :link: stage1
      :link-type: doc
      :class-card: sd-shadow-sm

      Move from the CLI overview into the first real workflow
      stage.

   .. grid-item-card:: Inference and export
      :link: inference_and_export
      :link-type: doc
      :class-card: sd-shadow-sm

      See how saved artifacts and CLI-driven outputs are
      reused downstream.

.. seealso::

   - :doc:`workflow_overview`
   - :doc:`configuration`
   - :doc:`stage1`
   - :doc:`stage2`
   - :doc:`stage3`
   - :doc:`stage4`
   - :doc:`stage5`
   - :doc:`inference_and_export`