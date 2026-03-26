Packaged resources reference
============================

The ``geoprior/resources`` area contains **packaged runtime
resources** used by the GeoPrior-v3 workflow.

Unlike the CLI, model, or tuner layers, this part of the
project is not primarily about executable public APIs. It is
about **static or template assets** that the workflow relies
on at runtime, especially during configuration bootstrap and
project setup.

This page documents those resources and exposes their source
content directly so that comments and inline guidance remain
visible in the built documentation.

Overview
--------

At the moment, the most important packaged resource is:

- ``geoprior/resources/natcom_config_template.py``

This file is the NATCOM-style configuration template used by
the GeoPrior configuration bootstrap flow.

A useful mental model is:

.. code-block:: text

   resources/
      └── natcom_config_template.py
           ↓
      used by init-config / geoprior-init
           ↓
      rendered into the user's active config.py

Why this page exists
--------------------

This page belongs in the API section because the resource
template is part of the **runtime contract** of the package.

Although it is not a model or CLI entry point, it still
defines a large part of the public workflow surface by
specifying:

- the experiment and city identifiers,
- dataset file patterns,
- time-window structure,
- column conventions,
- groundwater semantics,
- feature groups,
- holdout strategy,
- model defaults,
- physics defaults,
- scaling rules,
- and training defaults.

In practice, that makes the resource template one of the most
important reference files in the repository.

Resource role in the workflow
-----------------------------

The NATCOM template is the source text used by the config
bootstrap command to create the working configuration file.

The template itself states that it is rendered by:

.. code-block:: text

   python -m scripts init-config

and that it should live at:

.. code-block:: text

   geoprior/resources/natcom_config_template.py

The header comments also make the intended maintenance policy
clear:

- keep placeholders only for values supplied by the bootstrap
  script;
- keep the file limited to assignments and comments;
- avoid imports or helper code.

That is exactly the kind of resource that deserves to be
shown directly in the docs.

What the template defines
-------------------------

The current NATCOM template is structured into clear sections,
which makes it a strong reference file for users and
developers alike.

The main sections are:

1. **Core experiment setup**
2. **Feature registry**
3. **Censoring / effective-thickness configuration**
4. **Holdout strategy**
5. **Model architecture defaults**
6. **Physics + training schedule**
7. **GeoPrior scalar parameters**
8. **Training loop defaults**
9. **Hardware / runtime**
10. **Scaling / bookkeeping**

This is useful because the template is not just a flat list
of constants. It is already organized like a domain-specific
control file for the GeoPrior workflow.

Why the comments matter
-----------------------

For this resource file, the comments are part of the real
documentation value.

The section headers and inline comments explain:

- what each configuration block is for,
- which values are expected to be bootstrapped,
- which values are scientific defaults,
- and how the config is intended to remain lightweight and
  assignment-only.

That is why this page uses a full source listing rather than
only summarizing the file abstractly.

Relationship to the CLI and configuration system
------------------------------------------------

This resource is tightly connected to:

- :doc:`cli`
- :doc:`../user_guide/configuration`

In practical terms:

- ``geoprior-init`` or the corresponding init-config command
  uses this template as the starting point for config
  generation;
- the resulting ``config.py`` becomes the authored config;
- the workflow then persists the effective runtime state into
  ``config.json``.

So the template is the **packaged default source**, while the
generated config files are the **active project-specific
artifacts**.

What kind of resource this is
-----------------------------

The NATCOM template is best understood as a **renderable
configuration resource**.

It is not meant to be:

- imported for scientific computation,
- executed as a stage script,
- or modified dynamically at runtime.

Instead, it is meant to be:

- shipped with the package,
- rendered with placeholders substituted,
- and copied into the active configuration root.

That makes it different from the other API pages in this
section, and that difference should be reflected in the docs.

Recommended way to document resource files
------------------------------------------

For resource-heavy pages like this one, the best Sphinx
pattern is usually:

- short explanatory prose,
- followed by one or more ``literalinclude`` blocks.

This keeps the page useful for both:

- users who want to understand what the resource is for,
- developers who want to inspect the exact template text and
  comments.

NATCOM config template
----------------------

The full source of the current NATCOM template is shown
below.

.. literalinclude:: ../../geoprior/resources/natcom_config_template.py
   :language: python
   :caption: geoprior/resources/natcom_config_template.py

How to read the template
------------------------

A good reading order for this file is:

**1. Core experiment setup**

Read these fields first to understand:

- city identity,
- model identity,
- data root,
- file naming,
- time-window settings,
- main column names.

**2. Physics and scaling defaults**

Read the physics and scaling sections next to understand:

- PDE mode,
- lambda defaults,
- physical bounds,
- time units,
- coordinate mode,
- SI conversion fields.

**3. Training defaults**

Finally, read:

- epochs,
- batch size,
- learning rate,
- hardware/runtime options.

This gives the clearest progression from high-level workflow
identity to numerical training settings.

A few especially important fields
---------------------------------

Although the whole template matters, several groups of fields
are especially central to GeoPrior-v3.

**Groundwater semantics**

Examples:

- ``GWL_KIND``
- ``GWL_SIGN``
- ``USE_HEAD_PROXY``
- ``HEAD_COL``
- ``Z_SURF_COL``

These define how the groundwater variable is interpreted
physically.

**Temporal structure**

Examples:

- ``TRAIN_END_YEAR``
- ``FORECAST_START_YEAR``
- ``FORECAST_HORIZON_YEARS``
- ``TIME_STEPS``
- ``MODE``

These define the forecast contract.

**Physics defaults**

Examples:

- ``PDE_MODE_CONFIG``
- ``SCALE_PDE_RESIDUALS``
- ``LAMBDA_CONS``
- ``LAMBDA_GW``
- ``LAMBDA_PRIOR``
- ``LAMBDA_SMOOTH``
- ``PHYSICS_BOUNDS``

These define the default physics posture of the project.

**Scaling and SI handling**

Examples:

- ``TIME_UNITS``
- ``SUBS_SCALE_SI``
- ``HEAD_SCALE_SI``
- ``THICKNESS_UNIT_TO_SI``
- ``COORD_MODE``

These define how raw dataset values become the quantities
used by the physics core.

Maintenance notes
-----------------

A resource file like this is especially easy to damage if its
role is misunderstood.

Good maintenance rules are:

- keep it renderable with simple placeholder substitution;
- keep the comments because they are part of the workflow
  documentation;
- avoid adding runtime logic, imports, or helper functions;
- keep section structure stable so the generated config stays
  readable.

The current header comments already reflect these principles,
which is one reason exposing the full file in the docs is so
useful.

Future resource pages
---------------------

If the resources area later grows to include additional
templates or packaged assets, this page can expand naturally.

Typical future additions might include:

- alternate config templates,
- packaged schema files,
- static JSON defaults,
- or other workflow bootstrap resources.

If that happens, this page can evolve into a small resource
catalog, with one subsection and one ``literalinclude`` block
per resource.

API notes
---------

A few practical notes are worth keeping in mind:

- this page is intentionally **resource-oriented**, not
  import-oriented;
- ``literalinclude`` is the main documentation mechanism here
  because it preserves inline comments;
- the NATCOM template is one of the most important packaged
  files in the project because it defines the generated
  configuration surface;
- changes to this template should be treated as changes to
  the public workflow contract.

See also
--------

.. seealso::

   - :doc:`cli`
   - :doc:`../user_guide/configuration`
   - :doc:`../user_guide/cli`
   - :doc:`subsidence`
   - :doc:`utils`