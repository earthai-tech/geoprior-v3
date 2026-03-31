.. _cli-introduction:

Introduction
============

GeoPrior provides a **family-based command-line interface** designed to
cover the full life cycle of the package:

- running staged workflows,
- building reusable artifacts,
- rendering figures and maps,
- supporting reproducible analysis pipelines.

Rather than exposing one long flat list of unrelated commands, GeoPrior
groups commands into a small number of clear public families. This makes
the CLI easier to learn, easier to search, and easier to document.

The mental model
----------------

The GeoPrior CLI is built around one simple idea:

- use **run** when you want to **execute** a workflow,
- use **build** when you want to **materialize** an artifact,
- use **plot** when you want to **render** a figure or map.

This is why the command line is structured into three families rather
than one large undifferentiated command space.

At the top level, the CLI looks like this:

.. code-block:: bash

   geoprior run <command> [args]
   geoprior build <command> [args]
   geoprior plot <command> [args]

GeoPrior also supports:

.. code-block:: bash

   geoprior make <command> [args]

where ``make`` is simply an alias of the ``build`` family.

Root dispatcher and family entry points
---------------------------------------

GeoPrior offers two equivalent ways to invoke commands.

The first is the **root dispatcher**, where you specify the family
explicitly:

.. code-block:: bash

   geoprior run stage1-preprocess
   geoprior build forecast-ready-sample
   geoprior plot physics-fields --help

The second is the set of **family-specific console scripts**, which
remove the need to repeat the family token:

.. code-block:: bash

   geoprior-run stage4-infer --help
   geoprior-build full-inputs-npz
   geoprior-plot uncertainty
   geoprior-init --yes

In practice, both forms are useful:

- the root dispatcher is excellent for discoverability and teaching,
- the family-specific entry points are convenient in day-to-day work.

A simple way to think about it is:

- use ``geoprior ...`` when learning or browsing the CLI,
- use ``geoprior-run``, ``geoprior-build``, or ``geoprior-plot`` when
  you already know which family you need.

What each family means
----------------------

Run family
~~~~~~~~~~

The **run** family is for commands that **execute workflows**.

This includes:

- the staged GeoPrior pipeline,
- training and inference-oriented entry points,
- supplementary runtime diagnostics and research drivers.

Typical examples include preprocessing, training, tuning, inference,
transfer evaluation, sensitivity sweeps, and SM3-oriented run drivers.

.. code-block:: bash

   geoprior run stage1-preprocess
   geoprior run stage2-train
   geoprior run stage3-tune
   geoprior run stage4-infer

Build family
~~~~~~~~~~~~

The **build** family is for commands that **produce reusable outputs**
rather than directly executing the main training pipeline.

These outputs may include:

- merged NPZ payloads,
- summary tables,
- compact sampled datasets,
- spatial extraction tables,
- validation artifacts,
- reusable geospatial side products.

Examples:

.. code-block:: bash

   geoprior build full-inputs-npz
   geoprior build physics-payload-npz
   geoprior build spatial-sampling
   geoprior build forecast-ready-sample

Plot family
~~~~~~~~~~~

The **plot** family is for commands that **render visual outputs**.

These commands generate:

- forecasting figures,
- uncertainty panels,
- physics maps,
- transfer plots,
- hotspot analytics,
- supplementary diagnostic figures.

Examples:

.. code-block:: bash

   geoprior plot uncertainty
   geoprior plot spatial-forecasts
   geoprior plot physics-fields
   geoprior plot hotspot-analytics

Why this structure matters
--------------------------

This family-based layout is not just cosmetic.

It helps separate three very different user intentions:

1. **I want to run a model workflow**
2. **I want to generate a reusable artifact**
3. **I want to produce a figure**

That distinction matters in a package like GeoPrior because the command
surface is broad. Some commands orchestrate staged pipelines, some build
intermediate or final products, and others exist primarily to generate
publication-ready or gallery-ready outputs.

With a family-based CLI, users do not need to memorize every command
immediately. They only need to decide which *kind* of task they are
trying to perform.

Help and discovery
------------------

The CLI is designed to be discoverable from the terminal.

To see the top-level help:

.. code-block:: bash

   geoprior --help

To inspect a family:

.. code-block:: bash

   geoprior run --help
   geoprior build --help
   geoprior plot --help

To inspect one command:

.. code-block:: bash

   geoprior plot physics-fields --help
   geoprior build exposure --help
   geoprior run stage1-preprocess --help

The family-specific entry points support the same discovery pattern:

.. code-block:: bash

   geoprior-run --help
   geoprior-build --help
   geoprior-plot --help

Aliases and command normalization
---------------------------------

GeoPrior supports aliases for both families and commands.

The most visible family alias is:

- ``make`` -> ``build``

So these are equivalent:

.. code-block:: bash

   geoprior build exposure
   geoprior make exposure

Individual commands may also expose shorter or legacy aliases. This
keeps the public CLI ergonomic while still preserving stable canonical
names for documentation and help output.

As a documentation principle, this section and the family pages use the
**canonical public names** for commands. Aliases are useful in the
terminal, but the docs should teach the primary names first.

Shared conventions across commands
----------------------------------

Although GeoPrior commands do different jobs, many of them follow common
patterns.

Across the CLI, you will repeatedly see shared ideas such as:

- optional configuration installation via ``--config``,
- runtime overrides via repeated ``--set KEY=VALUE``,
- common output-location arguments such as output directories or results
  roots,
- consistent loading of tabular inputs from one or many files,
- format inference from file extensions,
- support for CSV, TSV, Excel, Parquet, JSON, Feather, and Pickle,
- Excel sheet selection using a ``PATH::SHEET`` syntax.

These shared behaviors are implemented in reusable CLI helpers so that
individual command modules stay focused on their own business logic
instead of re-implementing the same parsing and I/O code over and over.

For the full details, see :doc:`shared_conventions`.

How this section fits with the rest of the docs
-----------------------------------------------

The CLI pages answer a very practical question:

**Which command should I run for this task?**

That is different from the goals of the other major documentation
sections:

- :doc:`../user_guide/index` explains the broader workflow and how the
  stages fit together conceptually.
- :doc:`../examples/index` provides worked lessons with concrete outputs,
  figures, and interpretation.
- :doc:`../api/index` documents the Python modules and importable
  interfaces.

A useful rule of thumb is:

- go to the **CLI** section when you already know you want to run a
  command,
- go to the **User Guide** when you want the conceptual workflow,
- go to **Examples** when you want a worked lesson.

Suggested reading order
-----------------------

If you are new to the command line interface, a good reading order is:

#. this introduction,
#. :doc:`shared_conventions`,
#. one of :doc:`run_family`, :doc:`build_family`, or
   :doc:`plot_family` depending on your task.

From here
---------

Continue with one of the following pages:

- :doc:`shared_conventions`
- :doc:`run_family`
- :doc:`build_family`
- :doc:`plot_family`