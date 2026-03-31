Configuration
=============

GeoPrior-v3 is a **configuration-driven workflow framework**.

That means the intended way to control a run is not by
editing random values inside stage scripts. Instead, the
workflow is designed so that:

- a configuration file defines the main project settings,
- commands can install or override that configuration,
- the effective runtime configuration is persisted,
- and later stages can inspect what was actually used.

This is one of the key design choices in GeoPrior-v3. It
helps keep runs reproducible, auditable, and easier to
debug across the Stage-1 → Stage-5 workflow.

Why configuration matters
-------------------------

In GeoPrior-v3, many workflow outcomes depend on settings
that are easy to get wrong if they remain implicit.

Examples include:

- city and dataset identity,
- input file locations,
- time-window layout,
- forecast horizon,
- column semantics,
- groundwater conventions,
- scaling behavior,
- model defaults,
- results directories,
- and physics-related switches.

If these values live only in scattered code edits, it becomes
hard to answer basic questions later, such as:

- Which city did this run really use?
- Which horizon and time-step layout were active?
- Which config values were overridden from the CLI?
- Did this stage use the same assumptions as the previous
  stage?
- Can this run be reproduced on another machine?

GeoPrior-v3 uses explicit configuration so those questions
have clear answers.

The two main config artifacts
-----------------------------

GeoPrior-v3 uses two closely related configuration artifacts:

**1.** ``nat.com/config.py``  
This is the main **human-edited configuration file**.

**2.** ``nat.com/config.json``  
This is the **persisted effective runtime snapshot** used to
record the currently active config state after initialization
or CLI overrides.

A useful mental model is:

- ``config.py`` is the source you edit,
- ``config.json`` is the resolved state the workflow records.

This separation is important because a run often combines:

- the installed base config,
- mapped CLI fields such as ``--city`` or ``--model``,
- repeated ``--set KEY=VALUE`` overrides,
- and any refresh logic used to derive dependent fields.

Where the config lives
----------------------

By default, GeoPrior-v3 uses a config root directory named:

.. code-block:: text

   nat.com

Within that root, the expected config artifacts are:

.. code-block:: text

   nat.com/config.py
   nat.com/config.json

The shared CLI helper layer also supports:

.. code-block:: text

   --config-root

so advanced users can point commands at an alternate config
root when needed.

This is useful for cases such as:

- comparing multiple experimental setups,
- isolating one project from another,
- testing a new config without touching the default one,
- or running CI / automation with a dedicated config root.

How configuration enters the workflow
-------------------------------------

There are three main ways configuration enters a GeoPrior
command.

**1. Interactive or scripted initialization**

The command:

.. code-block:: bash

   geoprior-init

creates or refreshes the active ``config.py`` and also
refreshes the JSON-side runtime snapshot.

**2. Install a user-supplied config**

Many commands support:

.. code-block:: text

   --config path/to/config.py

When supplied, the user config is copied into the active
config root before the command runs.

**3. Apply one-off runtime overrides**

Many commands support:

.. code-block:: text

   --set KEY=VALUE

repeated as needed. These values are merged into the active
configuration for the current run and then persisted into the
runtime JSON snapshot.

This layered model is one of the strengths of the CLI design.

Configuration precedence
------------------------

In practice, the effective configuration is built in a simple
and predictable order.

A good way to think about precedence is:

1. start from the active ``config.py`` in the chosen
   config root;
2. if ``--config`` is provided, install that config into the
   active config root first;
3. map selected explicit CLI fields, such as ``--city`` or
   ``--model``, onto their corresponding config keys;
4. apply repeated ``--set KEY=VALUE`` overrides;
5. refresh any derived config fields if the command uses a
   refresh helper;
6. persist the effective result to ``config.json``.

This means that CLI overrides do not merely affect a local
in-memory object. They are also recorded into the runtime
config snapshot.

.. admonition:: Best practice

   Treat the persisted runtime config as the record of what a
   command actually used.

   This is especially important when the run was launched with
   one-off overrides.

How ``--set`` values are parsed
-------------------------------

GeoPrior-v3 parses ``--set KEY=VALUE`` conservatively so that
common value types work without forcing everything to remain
a raw string.

The parser tries values in this order:

1. case-insensitive ``true``, ``false``, and ``none``
2. integers
3. floats
4. ``ast.literal_eval(...)`` for values such as lists,
   tuples, dicts, and quoted strings
5. fallback to the stripped input string

This gives practical behavior such as:

.. code-block:: bash

   --set TIME_STEPS=6
   --set USE_VSN=True
   --set QUANTILES=[0.1,0.5,0.9]
   --set SOME_OPTION=None

without forcing the user to implement a custom parser for
every stage command.

.. warning::

   Every ``--set`` item must use the form ``KEY=VALUE``.
   Empty keys are rejected, and malformed items exit early.

What ``geoprior-init`` creates
------------------------------

The initialization command generates a NATCOM-style
``config.py`` template for GeoPrior-v3.

The generated file is not tiny. It already organizes the
configuration into meaningful blocks such as:

- dataset identity and file naming,
- time layout,
- required columns,
- physics conventions and target semantics,
- feature groups,
- training defaults,
- model defaults,
- scaling and bookkeeping.

This is a design choice because it makes the config
feel like a project control file rather than a loose set of
unrelated constants.

A typical initialization flow is:

.. code-block:: bash

   geoprior-init
   geoprior-init --yes
   geoprior-init --city zhongshan --time-steps 6

The initializer also prints suggested next commands such as
preprocess, train, tune, infer, and transfer.

The configuration template structure
------------------------------------

The generated config template is organized into practical
sections that mirror how the workflow is used.

Dataset identity and file naming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section usually defines core identifiers such as:

- ``CITY_NAME``
- ``MODEL_NAME``
- ``DATA_DIR``
- dataset variant and file naming choices

These values control the basic identity of the run.

Time layout
~~~~~~~~~~~

This section defines the temporal structure of the workflow,
for example:

- ``TRAIN_END_YEAR``
- ``FORECAST_START_YEAR``
- ``FORECAST_HORIZON_YEARS``
- ``TIME_STEPS``
- ``MODE``

These values are especially important because Stage-1, Stage-2,
Stage-3, and Stage-4 all depend on a consistent understanding
of horizon and sequence structure.

Required columns
~~~~~~~~~~~~~~~~

This section defines the core dataset columns used by the
workflow, for example:

- time column,
- longitude and latitude columns,
- subsidence column,
- groundwater column,
- thickness column,
- surface-elevation or head-related columns.

This is where raw tabular data begins to become a scientific
contract for the pipeline.

Physics conventions and target semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section is especially important for GeoPrior-v3.

It includes choices such as:

- groundwater kind,
- groundwater sign convention,
- whether to use a head proxy,
- PDE mode defaults,
- coordinate normalization or retention,
- scaling switches for groundwater, thickness, and surface
  elevation.

These settings strongly influence how the workflow interprets
the data physically, not just statistically.

Feature groups
~~~~~~~~~~~~~~

The config also organizes model-facing features into groups,
such as:

- static features,
- dynamic features,
- future-known features,
- optional categorical or numeric groups,
- and bookkeeping lists used later in manifests and audits.

This section matters because later stages depend on stable
feature semantics and ordering.

Training defaults
~~~~~~~~~~~~~~~~~

This section usually stores common training-time defaults
such as:

- batch size,
- epoch count,
- learning rate,
- optimizer-related defaults.

These settings become most visible in Stage-2 and Stage-3.

Model defaults
~~~~~~~~~~~~~~

This section typically contains model architecture defaults,
for example:

- hidden units,
- LSTM units,
- attention units,
- number of heads,
- dropout rate,
- batch normalization usage,
- VSN usage.

These values give the workflow a usable default model without
forcing every run to spell out the entire architecture.

Scaling and bookkeeping
~~~~~~~~~~~~~~~~~~~~~~~

The template also contains scaling and audit-oriented values,
for example:

- subsidence scale,
- groundwater scale,
- verbose or audit-stage settings.

This helps keep the workflow explicit about units, scaling,
and stage tracing.

How CLI fields map back into config
-----------------------------------

Some CLI arguments are not only command-local inputs. They
are also mapped back into config keys.

Examples include mappings such as:

- ``--city`` → ``CITY_NAME``
- ``--model`` → ``MODEL_NAME``
- ``--results-dir`` → a results-root config key

This design is useful because it lets a single command
override a small number of important fields without forcing
the user to modify the underlying config file by hand.

Typical pattern:

.. code-block:: bash

   geoprior-run stage1-preprocess \
     --city zhongshan \
     --model GeoPriorSubsNet

or:

.. code-block:: bash

   geoprior-run stage1-preprocess \
     --config my_config.py \
     --set TIME_STEPS=6 \
     --set FORECAST_HORIZON_YEARS=3

The effective values are then written into the persisted
runtime config snapshot.

Why ``config.json`` exists
--------------------------

It is natural to ask why GeoPrior-v3 uses both ``config.py``
and ``config.json``.

The short answer is that they serve different roles.

``config.py`` is convenient for:

- human editing,
- comments,
- structured grouping of settings,
- readable project control.

``config.json`` is convenient for:

- persisting the effective merged runtime config,
- recording CLI-applied overrides,
- lightweight downstream inspection,
- and providing a machine-friendly snapshot.

The persisted JSON payload also records key identity fields
such as city and model alongside the config dictionary
itself.

This makes it easier for later stages or tools to answer:
*what was the actual effective runtime config for this run?*

Configuration across stages
---------------------------

Configuration is not a Stage-1-only concern.

Across the workflow:

- Stage-1 uses config to define preprocessing, features,
  splits, and export structure.
- Stage-2 uses config together with the Stage-1 manifest to
  define training, scaling, model construction, and export.
- Stage-3 uses config to define the tuning search procedure
  on top of the Stage-1 contract.
- Stage-4 uses config to control model reuse, calibration,
  and inference behavior.
- Stage-5 uses config to define city-pair transfer defaults,
  results directories, and experiment behavior.

This is why the config page belongs in the **user guide**
rather than in a small appendix.

A practical lifecycle
---------------------

A good mental model for GeoPrior configuration is:

.. code-block:: text

   initialize config
        ↓
   edit config.py for project-level defaults
        ↓
   run one command with optional --config and --set overrides
        ↓
   persist effective runtime config to config.json
        ↓
   inspect artifacts and manifests
        ↓
   reuse or refine config for later stages

This lifecycle is much safer than editing stage scripts
directly every time something changes.

Recommended workflow
--------------------

A reliable configuration workflow looks like this:

**1. Bootstrap once**

.. code-block:: bash

   geoprior-init --yes

**2. Edit ``nat.com/config.py``**

Adjust city, paths, time layout, columns, feature lists,
training defaults, and physics conventions.

**3. Run the first stage**

.. code-block:: bash

   geoprior-run preprocess

**4. Use one-off overrides sparingly**

When you want a temporary change, use:

.. code-block:: bash

   --set KEY=VALUE

instead of editing the config file for every short-lived
experiment.

**5. Inspect the persisted runtime snapshot**

Review ``nat.com/config.json`` when you need to confirm what
a particular command actually used.

Common configuration mistakes
-----------------------------

**Mixing permanent defaults and temporary overrides**

Not every change belongs in ``config.py``. Temporary
experiments are usually better expressed through ``--set``.

**Forgetting that stages depend on shared semantics**

A harmless-looking change to horizon, feature lists, or
groundwater semantics can affect multiple later stages.

**Editing code instead of config**

If a value is really part of the run contract, it should
usually live in config.

**Treating ``config.json`` as the main editing surface**

The JSON file is best treated as a persisted runtime
snapshot, not as the main human-authored config.

**Using stale config with new artifacts**

If a config changed materially, be careful about reusing old
Stage-1 or Stage-2 artifacts whose semantics may no longer
match.

Configuration and reproducibility
---------------------------------

Configuration is one of the main reasons GeoPrior-v3 can
support reproducible staged workflows.

A run can be understood much more clearly when you can trace:

- the authored ``config.py``,
- the effective ``config.json``,
- the stage manifest,
- the generated artifacts,
- and the CLI command that launched the stage.

Together, these provide a much stronger provenance trail than
a notebook or script with hidden local edits.

Best practices
--------------

.. admonition:: Best practice

   Keep ``config.py`` as the project-level source of truth.

   Use it for stable defaults, not for one-off experiment
   clutter.

.. admonition:: Best practice

   Use ``--set`` for temporary variations.

   This keeps the project config readable while still making
   experiments easy.

.. admonition:: Best practice

   Inspect ``config.json`` when debugging a run.

   It is often the fastest way to see what effective values a
   CLI command actually used.

.. admonition:: Best practice

   Change configuration before the stage runs, not after.

   Later stages assume earlier artifact contracts are already
   fixed.

.. admonition:: Best practice

   Keep configuration, manifests, and outputs conceptually
   aligned.

   If you change the config materially, consider whether the
   old artifacts should still be trusted.

A compact configuration map
---------------------------

The GeoPrior-v3 configuration system can be summarized like
this:

.. code-block:: text

   geoprior-init
        ↓
   nat.com/config.py
        ↓
   optional --config install
        ↓
   optional mapped CLI fields (--city, --model, ...)
        ↓
   optional repeated --set KEY=VALUE
        ↓
   refresh derived config fields
        ↓
   nat.com/config.json
        ↓
   stage command reads effective config
        ↓
   stage manifest + workflow artifacts record the result

A few useful examples
---------------------

**Initialize the default config**

.. code-block:: bash

   geoprior-init --yes

**Run Stage-1 with one-off horizon overrides**

.. code-block:: bash

   geoprior-run preprocess \
     --set TIME_STEPS=6 \
     --set FORECAST_HORIZON_YEARS=3

**Use an alternate authored config file**

.. code-block:: bash

   geoprior-run train \
     --config my_project_config.py

**Override identity fields directly from the CLI**

.. code-block:: bash

   geoprior-run stage1-preprocess \
     --city zhongshan \
     --model GeoPriorSubsNet

**Work from an alternate config root**

.. code-block:: bash

   geoprior-run tune \
     --config-root natcom_experiment_a

Read next
---------

The best next pages after this one are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: CLI guide
      :link: cli
      :link-type: doc
      :class-card: sd-shadow-sm card--cli

      See how configuration is installed, overridden, and
      persisted from the CLI.

   .. grid-item-card:: Stage-1
      :link: stage1
      :link-type: doc
      :class-card: sd-shadow-sm

      Learn how configuration first becomes concrete workflow
      artifacts.

   .. grid-item-card:: Workflow overview
      :link: workflow_overview
      :link-type: doc
      :class-card: sd-shadow-sm card--workflow

      Review how config participates in the staged pipeline.

   .. grid-item-card:: Diagnostics
      :link: diagnostics
      :link-type: doc
      :class-card: sd-shadow-sm

      Understand how config, manifests, and diagnostics work
      together when debugging a run.

.. seealso::

   - :doc:`cli`
   - :doc:`workflow_overview`
   - :doc:`stage1`
   - :doc:`stage2`
   - :doc:`stage3`
   - :doc:`stage4`
   - :doc:`stage5`
   - :doc:`diagnostics`