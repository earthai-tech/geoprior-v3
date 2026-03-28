First run
=================

This page walks through a **first real GeoPrior-v3 run**.

The goal is not to produce a final scientific result on the
first attempt. Instead, this page helps you perform a clean
bring-up of the workflow so that you can confirm:

- the package is installed correctly,
- the CLI entry points are available,
- a configuration can be initialized and edited,
- the staged workflow can begin successfully,
- the expected artifacts are produced and inspected.

GeoPrior-v3 is designed as a **physics-guided geohazard
workflow framework**, not only as a model API. Its current
flagship application is **land subsidence**, and the package
registers dedicated console commands for initialization,
running, building, and plotting. 

What a first project run means
------------------------------

A first project run is a **workflow bring-up**.

That means you are not yet trying to optimize performance,
tune physics weights, or produce publication-ready results.
Instead, you are checking that the following pieces work
together:

- configuration initialization,
- stage execution,
- file and path resolution,
- environment dependencies,
- model and workflow plumbing,
- artifact creation and inspection.

This first pass is successful if the workflow starts cleanly,
uses the intended configuration, and produces outputs that
can be reviewed and extended in later stages.

Before you begin
----------------

Before starting, make sure you already completed the steps in
:doc:`installation` and :doc:`quickstart`.

In particular, the following commands should already work in
your active environment:

.. code-block:: bash

   python -c "import geoprior as gp; print(gp.__version__)"
   geoprior --help
   geoprior-run --help
   geoprior-build --help
   geoprior-plot --help
   geoprior-init --help

GeoPrior-v3 currently targets Python 3.10 and above and
ships with a staged CLI-oriented workflow surface rather than
a package-only interface. 

Step 1. Create or activate the working environment
--------------------------------------------------

Use a dedicated virtual environment or Conda environment for
the project. Then install GeoPrior-v3 in that environment.

For a local editable install from the repository root:

.. code-block:: bash

   pip install -e .

If you are also building the documentation while working on
the project:

.. code-block:: bash

   pip install -e ".[docs]"

If you are contributing code, tests, and documentation
together:

.. code-block:: bash

   pip install -e ".[dev,docs]"

GeoPrior-v3 currently declares a scientific runtime stack
that includes NumPy, pandas, SciPy, matplotlib,
scikit-learn, TensorFlow, Keras, statsmodels, PyYAML,
platformdirs, lz4, and psutil. 

Step 2. Verify the package and workflow surface
-----------------------------------------------

Confirm that the base package import works:

.. code-block:: bash

   python -c "import geoprior as gp; print(gp.__version__)"

GeoPrior-v3 keeps import-time output relatively quiet and
tries to degrade gracefully if some optional pieces are not
available, but the environment should still be treated as a
real scientific runtime, not as a lightweight pure-Python
package. The package also exposes helper behavior such as
warning suppression and optional bridging to ``kdiagram``.

Next, inspect the CLI help pages:

.. code-block:: bash

   geoprior --help
   geoprior-init --help
   geoprior-run --help
   geoprior-build --help
   geoprior-plot --help

This confirms that the entry points registered by the
package metadata are available in the active environment.

Step 3. Initialize a project configuration
------------------------------------------

A first real run should start from a configuration template,
not from ad hoc edits scattered across the codebase.

GeoPrior-v3 includes a dedicated initialization entry point:

.. code-block:: bash

   geoprior-init --help

Use the initialization command to generate, copy, or inspect
the starting configuration for your run. The package layout
includes template and configuration resources so that the
workflow can begin from an explicit config rather than from
hard-coded assumptions.

.. note::

   Treat the initialized configuration as the **single source
   of truth** for the first run. Do not mix manual code edits
   with configuration-driven execution unless you are
   intentionally debugging the framework itself.

In many setups, the first run will begin from a project
configuration derived from the packaged template and then
adapted to your local paths, data layout, and stage outputs.

Step 4. Review the configuration before running anything
--------------------------------------------------------

Before launching the first stage, open the generated config
and check the values that matter most locally.

Typical items to verify are:

- project or working directory paths,
- input data locations,
- output or artifact directories,
- city, case-study, or experiment identifiers,
- stage-specific toggles,
- plotting and export locations,
- hardware or runtime assumptions if applicable.

A first project run often fails for simple reasons such as:

- paths that still point to a previous machine,
- missing input files,
- output directories that do not yet exist,
- stale references copied from an older workflow.

That is why configuration review should happen **before**
running the stages.

Step 5. Start with one stage, not the whole pipeline
----------------------------------------------------

For the first project run, do not try to run everything at
once.

Start with the **earliest relevant stage** and confirm that
it completes cleanly. GeoPrior-v3 is documented around a
staged workflow precisely because that makes failures easier
to isolate and outputs easier to inspect.

A safe pattern is:

.. code-block:: bash

   geoprior-run --help

Then inspect the stage-oriented help and launch the earliest
stage supported by your current configuration and data.

.. important::

   Use the installed CLI help output as the authority for the
   exact invocation syntax. The command surface may evolve as
   the staged workflow is refined, so it is better to confirm
   the current local syntax than to rely on memory or on an
   outdated shell history.

At this point, the goal is simple:

- the stage starts,
- the config is read correctly,
- required inputs are found,
- outputs are written where expected,
- no silent path confusion occurs.

Step 6. Inspect the generated artifacts
---------------------------------------

After the first stage completes, inspect what it produced.

Depending on the stage, that may include:

- processed intermediate data,
- cached tensors or prepared arrays,
- logs and diagnostics,
- figures,
- serialized metadata,
- build-ready artifacts for the next stage.

A successful first run is not only about whether the command
finished. It is also about whether the generated outputs make
sense structurally.

Check the following:

- Were the files written to the expected directory?
- Do the filenames or folder names match the run you
  intended?
- Are artifact timestamps recent and consistent?
- Do logs suggest the correct configuration was used?
- Are obvious diagnostics or summary values reasonable?

If the first stage produces plots or summaries, review them
visually rather than assuming correctness from command
success alone.

Step 7. Continue gradually to the next stage
--------------------------------------------

Once the first stage works, continue stage by stage rather
than immediately attempting a full end-to-end workflow.

A sensible progression is:

1. initialize and validate configuration,
2. run the earliest stage,
3. inspect outputs,
4. run the next stage,
5. inspect again,
6. only then move toward builds, plots, and export.

This is especially important in GeoPrior-v3 because the
framework is built around **physics-guided** modeling. A run
that finishes numerically is not automatically a run that is
scientifically well-scaled or physically well-posed.

Optional Python check after the first run
-----------------------------------------

After the workflow is initialized and the first stage has
started producing expected artifacts, a small Python-side
check is often useful.

A simple import-level confirmation is:

.. code-block:: python

   import geoprior as gp

   print("GeoPrior version:", gp.__version__)

If you want a model-facing import, prefer the public package
surface rather than a deep internal path:

.. code-block:: python

   from geoprior.models import GeoPriorSubsNet

or:

.. code-block:: python

   from geoprior.models.subsidence import GeoPriorSubsNet

This keeps example code aligned with the intended public
import style.

What not to expect from the first run
-------------------------------------

A first run is **not** expected to guarantee:

- optimal hyperparameters,
- correct physics weighting,
- stable uncertainty calibration,
- final-quality plots,
- publication-ready diagnostics,
- validated scientific interpretation.

Those come later.

The first run is successful if it gives you a trustworthy
base from which the real analysis can proceed.

Common first-run mistakes
-------------------------

The most common problems at this stage are practical rather
than theoretical.

**Using the wrong environment**

The package may be installed in one environment while the CLI
is being executed from another.

**Skipping configuration review**

A template-generated config is only a starting point. Local
paths and artifact locations usually still need adjustment.

**Running too many stages at once**

When something fails, it is much easier to diagnose one stage
than an entire chained workflow.

**Treating command success as scientific success**

A completed run may still contain wrong units, incorrect
paths, mismatched scaling assumptions, or misleading outputs.

**Editing code when configuration is the real issue**

For a first run, most failures are caused by setup or path
problems, not by model bugs.

A practical first-run checklist
-------------------------------

Use this checklist before moving on:

.. checklist::

   * GeoPrior-v3 imports successfully.
   * CLI entry points resolve with ``--help``.
   * A project configuration has been initialized.
   * Local paths in the config have been reviewed.
   * The earliest stage runs successfully.
   * Output artifacts are written to the intended location.
   * The generated files and summaries look structurally
     reasonable.
   * You know which stage or guide page you will read next.

If all of these are true, the project bring-up is in good
shape.

Where to go next
----------------

Once the first project run is working, the next best pages
are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Workflow overview
      :link: ../user_guide/workflow_overview
      :link-type: doc
      :class-card: sd-shadow-sm card--workflow
      :data-preview: workflow.png
      :data-preview-label: GeoPrior staged workflow

      Understand how the full staged pipeline is organized.

   .. grid-item-card:: Configuration
      :link: ../user_guide/configuration
      :link-type: doc
      :class-card: sd-shadow-sm card--configuration
      :data-preview: configuration.png
      :data-preview-label: GeoPrior configuration system

      Learn how the configuration layer controls the run.

   .. grid-item-card:: CLI guide
      :link: ../user_guide/cli
      :link-type: doc
      :class-card: sd-shadow-sm card--cli
      :data-preview: cli.png
      :data-preview-label: GeoPrior command-line interface

      Move from the first run to the full command reference.

   .. grid-item-card:: Data and units
      :link: ../scientific_foundations/data_and_units
      :link-type: doc
      :class-card: sd-shadow-sm

      Review units, coordinates, and scaling before trusting
      physics-guided results.

.. seealso::

   - :doc:`installation`
   - :doc:`quickstart`
   - :doc:`../user_guide/workflow_overview`
   - :doc:`../user_guide/configuration`
   - :doc:`../user_guide/cli`
   - :doc:`../scientific_foundations/data_and_units`
   - :doc:`../scientific_foundations/scaling`