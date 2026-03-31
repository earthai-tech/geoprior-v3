.. _cli-shared-conventions:

Shared conventions
==================

Many GeoPrior commands do very different jobs, but they still follow a
common set of command-line conventions.

This page gathers those shared patterns in one place so they do not need
to be re-explained in every command family page.

In practice, most shared conventions fall into four categories:

- configuration installation and runtime overrides,
- common path and output semantics,
- reusable tabular input handling,
- consistent writing of derived outputs.

Why this page exists
--------------------

GeoPrior deliberately keeps repeated CLI behavior in shared helper
modules rather than re-implementing the same parsing logic in every
command. This makes the command surface more coherent and reduces drift
between commands over time. 

As a user, the benefit is simple:

- once you understand these conventions, many commands will feel
  familiar immediately,
- commands can stay focused on their scientific or operational purpose,
  instead of repeating generic I/O and config behavior.

Configuration handling
----------------------

Optional config installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many commands support an optional ``--config`` argument that installs a
user-provided ``config.py`` into the active config root before the
command runs. The shared config helper also supports ``--config-root``,
which defaults to ``nat.com``. 

Typical pattern:

.. code-block:: bash

   geoprior run stage1-preprocess \
       --config my_experiment_config.py

or with an explicit config root:

.. code-block:: bash

   geoprior build full-inputs-npz \
       --config my_config.py \
       --config-root nat.com

This pattern is useful when you want a command to run against a specific
configuration file without manually copying it into the active config
area first.

Runtime overrides with ``--set``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many commands also support repeated runtime overrides via:

.. code-block:: bash

   --set KEY=VALUE

These overrides are parsed centrally and merged into the active runtime
configuration. The shared parser accepts a few convenient value forms:

- case-insensitive booleans such as ``true`` and ``false``,
- ``none``,
- integers and floats,
- Python-literal-style containers when they can be parsed safely,
- otherwise the raw stripped string. 

Examples:

.. code-block:: bash

   geoprior run stage2-train \
       --set TIME_STEPS=6 \
       --set FORECAST_HORIZON=3 \
       --set USE_BATCH_NORM=true

   geoprior run stage3-tune \
       --set QUANTILES="[0.1, 0.5, 0.9]"

Each ``--set`` item must have the form ``KEY=VALUE``. Empty keys are
rejected. 

Field-to-config mapping
~~~~~~~~~~~~~~~~~~~~~~~

Some commands expose explicit arguments such as ``--city``,
``--model``, or ``--results-dir``. Shared config helpers can map those
parsed fields into config keys before persisting the effective runtime
configuration. This is one reason different commands can feel
consistent even when their domain-specific options differ. 

Common reusable arguments
-------------------------

Several arguments appear repeatedly across commands because they are
provided by shared helper functions rather than defined ad hoc in each
module. Examples include:

- ``--city``
- ``--model``
- ``--results-dir`` / ``--results-root``
- ``--manifest``
- ``--stage1-dir``
- ``--stage2-manifest``
- ``--outdir``
- ``--format``
- ``--output-stem``
- ``--split``
- ``--validation-csv`` 

This does **not** mean every command accepts every one of these
arguments. It means many commands reuse the same definitions, wording,
and destination semantics when they need them.

Tabular input conventions
-------------------------

One or many input files
~~~~~~~~~~~~~~~~~~~~~~~

GeoPrior provides shared helpers for commands that read tabular inputs.
These helpers support:

- one or many input files,
- simple glob expansion,
- duplicate-path removal,
- format inference from file extension,
- consistent concatenation into one DataFrame when multiple tables are
  loaded. 

This is especially useful for build commands that need to combine
multiple datasets into one unified table.

Supported input formats
~~~~~~~~~~~~~~~~~~~~~~~

The shared reader layer supports the following canonical input formats:

- CSV
- TSV
- Excel
- Parquet
- JSON
- Feather
- Pickle 

Format names can be inferred from file extensions, or forced explicitly
with ``--input-format``. Aliases such as ``xls``/``xlsx`` for Excel,
``pq`` for Parquet, and ``pkl`` for Pickle are normalized centrally.


Excel sheet syntax
~~~~~~~~~~~~~~~~~~

Excel inputs can be specified using a compact inline sheet selector:

.. code-block:: text

   data.xlsx::Sheet1
   data.xlsx::0

The part after ``::`` selects either a sheet name or a zero-based sheet
index. A command can also expose a default ``--sheet-name`` used when no
inline selector is provided. 

Common reader arguments
~~~~~~~~~~~~~~~~~~~~~~~

Commands that use the shared table-reader layer often expose options
such as:

- ``--input-format``
- ``--sheet-name``
- ``--excel-engine``
- ``--sep``
- ``--encoding``
- ``--header``
- ``--usecols``
- ``--nrows``
- ``--source-col``
- ``--read-kwargs`` 

A few of these deserve special attention.

Header handling
^^^^^^^^^^^^^^^

The shared parser for ``--header`` accepts three main forms:

- ``infer``
- ``none``
- an integer row index 

This mirrors the common pandas patterns while keeping the command-line
interface predictable.

Extra pandas reader kwargs
^^^^^^^^^^^^^^^^^^^^^^^^^^

``--read-kwargs`` accepts a JSON object string that is passed to the
underlying pandas reader. It must decode to a JSON object, not a list or
other JSON type. 

Example:

.. code-block:: bash

   geoprior build spatial-sampling data.parquet \
       --read-kwargs '{"columns": ["x", "y", "year"]}'

Attach source paths
^^^^^^^^^^^^^^^^^^^

When ``--source-col`` is used, the shared loader adds a column
containing the source file path for each loaded row. This is convenient
when many input files are merged into one combined table. 

How multi-file loading behaves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple tables are loaded through the shared helper layer:

- each resolved input source is read into a DataFrame,
- optional source-path tracking can be added,
- the tables are concatenated row-wise,
- the result is returned as one combined DataFrame,
- index resetting can be applied consistently. 

This behavior is one reason GeoPrior build commands can stay compact:
the repeated mechanics of “read many files, normalize them, and combine
them” already live in one place.

Output and path conventions
---------------------------

Output directories
~~~~~~~~~~~~~~~~~~

Many commands expose explicit output locations such as ``--outdir`` or
``--results-dir``. Shared helpers create output directories when needed
rather than forcing each command to duplicate that boilerplate.


In practice:

- use ``--results-dir`` when a command expects a results root or staged
  output area,
- use ``--outdir`` when a command writes one or more derived outputs to
  a dedicated folder.

Script-backed artifact roots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoPrior also provides shared artifact-root conventions for figure and
tabular exports in script-backed plotting and reproducibility commands.

The key environment variables are:

- ``GEOPRIOR_ARTIFACT_ROOT``
- ``GEOPRIOR_FIG_DIR``
- ``GEOPRIOR_OUT_DIR`` 

When these are not explicitly set, the default artifact root is
discovered from the project root and placed under:

.. code-block:: text

   scripts/
     figs/
     out/

This gives script-backed commands a stable default place to write
figures and tabular outputs. 

Bare filenames versus explicit paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For script-backed exports, GeoPrior distinguishes between:

- an **absolute path**, which is used as given,
- a **relative path with an explicit parent**, which keeps that parent,
- a **bare filename or stem**, which is placed under the shared figure
  or output directory. 

For example, a bare figure name such as:

.. code-block:: text

   physics_map

is resolved under the figure export area, while a bare tabular name is
resolved under the tabular export area. Explicit paths such as
``results/custom/my_plot.png`` are respected as explicit user choices.


Figure stems and multi-format export
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared plotting utilities treat figure outputs as **stems**. If a suffix
is present, it may be stripped so one logical output can be written to
multiple formats. Shared helpers support figure export to formats such
as PNG, SVG, PDF, and EPS. 

This is why many plot commands can naturally write a family of outputs
from one stem rather than requiring separate filenames for each format.

Tabular output writing
~~~~~~~~~~~~~~~~~~~~~~

Shared DataFrame-writing helpers infer the output format from the file
extension and support writing to:

- CSV
- TSV
- Parquet
- Excel
- JSON
- Feather
- Pickle 

This keeps output behavior aligned with the shared input behavior.

Practical examples
------------------

Install a config and override a few values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   geoprior run stage2-train \
       --config experiment_config.py \
       --set TIME_STEPS=6 \
       --set FORECAST_HORIZON=3 \
       --set USE_BATCH_NORM=true

Read an Excel sheet explicitly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   geoprior build spatial-roi data.xlsx::Sheet1

Merge multiple files while keeping the source path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   geoprior build spatial-sampling inputs/*.csv \
       --source-col source_file

Force a format and limit columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   geoprior build forecast-ready-sample data.parquet \
       --input-format parquet \
       --usecols longitude latitude year subsidence \
       --nrows 50000

Let a plot command write under the shared figure root
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   geoprior plot physics-fields --out physics_fields_main

When a command follows the shared figure-path convention, a bare output
name like this is routed to the configured figure export area rather
than to the current working directory. 

Guiding principle
-----------------

The main idea behind these conventions is simple:

**shared mechanics should be learned once and reused everywhere.**

That is why GeoPrior centralizes:

- configuration installation,
- runtime overrides,
- common parser arguments,
- tabular input loading,
- artifact path resolution,
- repeated output-writing logic. 

From here
---------

Now that the shared behaviors are defined, move to the family page that
matches your task:

- :doc:`run_family`
- :doc:`build_family`
- :doc:`plot_family`