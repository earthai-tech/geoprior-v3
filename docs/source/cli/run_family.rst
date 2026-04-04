.. _cli-run-family:

Run family
==========

The **run family** is where you execute GeoPrior workflows.

Use this page when your goal is to **run a process**, not just to build
an artifact or render a figure. In practice, this includes:

- the staged GeoPrior workflow,
- supplementary physics-oriented run drivers,
- SM3 synthetic diagnostics and preset suites.

You can invoke these commands either from the root dispatcher:

.. code-block:: bash

   geoprior run <command> [args]

or from the family-specific entry point:

.. code-block:: bash

   geoprior-run <command> [args]

This page is designed as a **guide first** and a **reference second**:

- the table below helps you find the right run command quickly,
- the sections afterwards explain when each command should be used,
- detailed parameter coverage can be expanded gradually command by
  command.

For shared CLI behaviors such as ``--config``, ``--config-root``,
``--set KEY=VALUE``, and repeated path conventions, see
:doc:`shared_conventions`. GeoPrior centralizes these shared patterns in
common CLI helpers rather than redefining them independently in each run
module. 

How to choose a run command
---------------------------

A practical rule is:

- choose **Stage 1** when you need preprocessing and sequence export,
- choose **Stage 2** when you need training,
- choose **Stage 3** when you need hyperparameter tuning,
- choose **Stage 4** when you need inference or evaluation,
- choose **Stage 5** when you need cross-city transfer evaluation,
- choose **sensitivity** when you want a physics sensitivity driver,
- choose **identifiability** when you want SM3 synthetic
  identifiability,
- choose **sm3-offset-diagnostics** when you want log-offset
  diagnostics,
- choose **sm3-suite** when you want a preset-driven multi-regime SM3
  batch run.

Run commands at a glance
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 46 34

   * - Command
     - Use it when
     - Main outcome
   * - :ref:`run-init-config`
     - You need to bootstrap or install the active configuration before
       using the workflow.
     - Active config root prepared for later runs.
   * - :ref:`run-stage1`
     - You need preprocessing, cleaning, scaling, sequence construction,
       and Stage-1 exports.
     - Stage-1 artifacts, manifests, NPZ inputs, scalers, and prepared
       data products.
   * - :ref:`run-stage2`
     - You already have Stage-1 outputs and want to train the model.
     - Stage-2 training run.
   * - :ref:`run-stage3`
     - You want hyperparameter search or tuning on top of Stage-1
       artifacts.
     - Stage-3 tuning run.
   * - :ref:`run-stage4`
     - You want inference, evaluation, calibration, or forecast export.
     - Stage-4 inference and evaluation outputs.
   * - :ref:`run-stage5`
     - You want cross-city transfer evaluation or warm-start style
       transfer workflows.
     - Stage-5 transfer evaluation run.
   * - :ref:`run-sensitivity`
     - You want a physics sensitivity grid driver rather than a staged
       model workflow.
     - Sensitivity run outputs for later inspection and plotting.
   * - :ref:`run-identifiability`
     - You want to run SM3 synthetic identifiability experiments.
     - SM3 identifiability run outputs.
   * - :ref:`run-sm3-offsets`
     - You want SM3 log-offset diagnostics.
     - SM3 offset-diagnostics run outputs.
   * - :ref:`run-sm3-suite`
     - You want a preset-driven SM3 batch across multiple regimes.
     - Multi-regime suite outputs and collected summaries.


Shared invocation pattern
-------------------------
In the run family, the canonical SM3 command is
``identifiability``. The plot-side figure command uses a different
public name. This page follows the **run-family canonical names** so
users can copy commands exactly as exposed by the run dispatcher.

Most run commands follow the same outer usage pattern:

.. code-block:: bash

   geoprior run <command> --config my_config.py --set KEY=VALUE

or:

.. code-block:: bash

   geoprior-run <command> --help

Many run commands support optional config installation and one-off
runtime overrides, while some also accept Stage-1 manifest paths or
forward additional legacy CLI arguments downstream. The details vary by
command, but the overall user experience is intentionally similar across
the run family. Stage wrappers for Stages 2–5 explicitly support config
installation and runtime overrides, while the shared CLI layer provides
the reusable argument and config mechanisms used more broadly across the
package. 


.. _run-init-config:

``init-config``
--------------------

Use this command when you want to create or bootstrap the active
``nat.com/config.py`` before running the rest of the pipeline.

Typical usage:

.. code-block:: bash

   geoprior run init-config
   geoprior-init --yes

This command belongs at the beginning of the workflow, especially when
you are setting up a new environment or new experiment root. GeoPrior
registers it as part of the run family, alongside the staged workflow
commands. 

See also:
:doc:`../user_guide/configuration`

.. _run-stage1:


``stage1-preprocess``
---------------------

Use ``stage1-preprocess`` when you want to turn a raw or harmonized
city dataset into the **structured Stage-1 artifact set** used by the
rest of the GeoPrior pipeline.

This command is the real entry point to the preprocessing workflow. The
underlying Stage-1 pipeline is described as a six-step process:

1. load the dataset,
2. clean and select features,
3. encode and scale,
4. define feature sets,
5. split by year and build PINN-style sequences,
6. build train/validation datasets and export arrays plus metadata. 

In other words, this is the command you run when you want GeoPrior to
take a city-level table and convert it into the **manifested inputs**
that later stages can consume directly.

Why this command matters
^^^^^^^^^^^^^^^^^^^^^^^^

Stage 1 is the foundation of the run family.

Later stages are designed so they do **not** need to recompute the
preprocessing logic again. Instead, Stage 1 writes a structured set of
artifacts that downstream stages can load directly. The Stage-1 module
states this very explicitly: Stage 2 only needs to read
``manifest.json``, load the exported NPZ files, and optionally reload
the saved scalers and encoders if needed. 

A good practical rule is:

- run ``stage1-preprocess`` when your data, city choice, or core window
  settings have changed,
- rerun later stages without rebuilding Stage 1 only when those Stage-1
  artifacts are still the ones you want.

What Stage 1 produces
^^^^^^^^^^^^^^^^^^^^^

The Stage-1 pipeline writes a reusable artifact set rather than a single
file. Its declared outputs include:

- CSV exports for raw, cleaned, and scaled tables,
- Joblib artifacts such as the one-hot encoder, main scaler, and
  coordinate scaler,
- NPZ arrays such as training and validation inputs and targets,
- a ``manifest.json`` file that records paths, shapes, dimensions,
  columns, and config information for later stages. 

That manifest is the most important product conceptually, because it is
the hand-off object between preprocessing and the rest of the pipeline.

What this command is a good fit for
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``stage1-preprocess`` when you need to:

- prepare a city-specific dataset for training or tuning,
- regenerate sequence windows after changing ``TIME_STEPS`` or forecast
  horizon settings,
- refresh scaling and feature artifacts after changing feature groups,
- rebuild the manifest expected by Stage 2, Stage 3, or Stage 4,
- create a clean, reproducible preprocessing checkpoint before running
  experiments.

It is especially useful when you want your experiments to begin from a
stable artifact boundary rather than from ad hoc notebook preprocessing.

Common invocation patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^

Run with the active config as-is:

.. code-block:: bash

   geoprior run stage1-preprocess

or:

.. code-block:: bash

   geoprior-run stage1-preprocess

Override the city for one run:

.. code-block:: bash

   geoprior-run stage1-preprocess --city nansha

Install a specific config file first:

.. code-block:: bash

   geoprior-run stage1-preprocess --config my_config.py

Apply one-off config overrides:

.. code-block:: bash

   geoprior-run stage1-preprocess \
       --set TIME_STEPS=6 \
       --set FORECAST_HORIZON_YEARS=3

Override the main run identity fields directly:

.. code-block:: bash

   geoprior-run stage1-preprocess \
       --city zhongshan \
       --model GeoPriorSubsNet \
       --data-dir ./data

These patterns are directly supported by the Stage-1 CLI wrapper, which
defines ``--config``, ``--config-root``, ``--city``, ``--model``,
``--data-dir``, and repeated ``--set KEY=VALUE`` overrides. The wrapper
maps those explicit arguments into runtime config updates for
``CITY_NAME``, ``MODEL_NAME``, and ``DATA_DIR`` before launching the
Stage-1 pipeline. 

Key command-line options
^^^^^^^^^^^^^^^^^^^^^^^^
This first run-stage subsection introduces the **shared run options**
used repeatedly across the staged workflow:

``--config``
   Install a user-provided ``config.py`` into the active config root
   before the run starts. This is useful when you want the whole Stage-1
   execution to follow a specific experiment configuration. 

``--config-root``
   Select the active config root directory. The Stage-1 parser defaults
   to ``nat.com``. 

``--city``
   Override ``CITY_NAME`` for the current run without manually editing
   the config file. 

``--model``
   Override ``MODEL_NAME`` for the current run. This is helpful when you
   want the produced Stage-1 directory and manifest to align with a
   specific model identity. 

``--data-dir``
   Override ``DATA_DIR`` for the current run. Use this when your input
   tables live somewhere other than the default config location. 

``--set KEY=VALUE``
   Apply one or more one-off config overrides without editing the base
   config file. The Stage-1 help text explicitly shows examples such as
   ``--set TIME_STEPS=6``. 

How Stage 1 behaves
^^^^^^^^^^^^^^^^^^^

The Stage-1 implementation is deliberately robust about input discovery
and preprocessing. Beyond the high-level six-step workflow, it can:

- search configured primary and fallback input paths,
- optionally unpack a city table from a merged all-cities parquet when
  needed,
- fall back to built-in dataset fetchers when the expected CSV is not
  found,
- normalize groundwater-level aliases early,
- resolve optional feature columns from declared config groups,
- build censor-aware transformed columns,
- create explicit SI-unit physics columns for the physics-aware parts of
  the workflow,
- filter groups so only valid train and forecast candidates continue
  downstream. 

For users, the main takeaway is that ``stage1-preprocess`` is much more
than a format conversion step. It is where GeoPrior decides how the
dataset becomes a consistent forecasting and physics-aware learning
payload.

Future-aware export behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage 1 can also prepare artifacts for later forecasting-oriented
workflows. In particular, the Stage-1 code exposes a
``BUILD_FUTURE_NPZ`` setting that controls whether future-oriented NPZ
artifacts are pre-built during Stage 1 for later use. 

That means Stage 1 is not only about training-set preparation; it can
also prepare forward-looking artifacts when the config requests them.

How later stages depend on it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cleanest way to think about Stage 1 is:

**it creates the experiment-ready data contract for the rest of the
pipeline.**

- Stage 2 uses it for training,
- Stage 3 uses it for tuning,
- Stage 4 can use it for inference and evaluation,
- transfer and analysis workflows often rely on its exported structure
  as well.

This is why the manifest matters so much. It is the structured record of
what was built, where it was written, and how later code should reload
it. 

Practical advice
^^^^^^^^^^^^^^^^

When documenting or teaching this command, it helps to frame it in this
way:

- **Input**: raw or harmonized city-level data plus config,
- **Transformation**: cleaning, feature resolution, scaling, sequence
  construction, and physics-aware preparation,
- **Output**: a reproducible Stage-1 artifact directory with manifest,
  arrays, and preprocessing objects.

That framing makes it much easier for users to understand why this
command comes first and why later stages often ask for Stage-1
artifacts, manifests, or derived directories rather than raw tables.

See also
^^^^^^^^^^^^

- :doc:`../user_guide/stage1`
- :doc:`../auto_examples/diagnostics/plot_stage1_data_checks`
- :doc:`../auto_examples/diagnostics/plot_holdout_group_masks`

.. _run-stage2:

``stage2-train``
--------------------

Use ``stage2-train`` when your Stage-1 preprocessing artifacts already
exist and you want to launch a **training run** from that prepared
artifact set.

The Stage-2 wrapper is specifically designed to make training safe and
consistent from ``geoprior.cli``. Its documented flows include:

- using the existing ``nat.com/config.py`` as-is,
- installing a user-supplied config before training,
- applying one-off ``--set KEY=VALUE`` overrides,
- pointing the run at an explicit Stage-1 manifest via
  ``--stage1-manifest``. 

This means Stage 2 is the natural next step after
``stage1-preprocess``. Instead of rebuilding preprocessing logic, it
starts from the Stage-1 outputs and focuses on **model fitting**.

Why you would use Stage 2
^^^^^^^^^^^^^^^^^^^^^^^^^

Choose ``stage2-train`` when you want to:

- train one concrete model configuration,
- rerun training after changing a few hyperparameters,
- launch training from a known Stage-1 manifest,
- test a new model identity or city setting without manually editing the
  base config each time.

A useful mental model is:

- Stage 1 prepares the experiment-ready inputs,
- Stage 2 consumes those inputs and performs the actual training run. 

Usage
^^^^^^^^

Use the active config exactly as it is:

.. code-block:: bash

   geoprior run stage2-train

or:

.. code-block:: bash

   geoprior-run stage2-train

Train from an explicit Stage-1 manifest:

.. code-block:: bash

   geoprior-run stage2-train \
       --stage1-manifest path/to/manifest.json

Install a specific config before training:

.. code-block:: bash

   geoprior-run stage2-train \
       --config my_config.py

Apply one-off overrides without editing the config file:

.. code-block:: bash

   geoprior-run stage2-train \
       --set EPOCHS=150 \
       --set BATCH_SIZE=64 \
       --set LEARNING_RATE=0.0005

Override the main run identity directly from the CLI:

.. code-block:: bash

   geoprior-run stage2-train \
       --city nansha \
       --model GeoPriorSubsNet \
       --data-dir ./data

Combine a fixed Stage-1 manifest with temporary overrides:

.. code-block:: bash

   geoprior-run stage2-train \
       --stage1-manifest results/nansha_run/artifacts/manifest.json \
       --set EPOCHS=200 \
       --set USE_BATCH_NORM=true

Key command-line options
^^^^^^^^^^^^^^^^^^^^^^^^

Stage 2 supports the same shared run options introduced in
``stage1-preprocess``. The main additional training-specific option is:

``--stage1-manifest``
   Point Stage 2 at one exact Stage-1 ``manifest.json``. The wrapper
   documents that this is forwarded through the
   ``STAGE1_MANIFEST`` environment variable. 

``--set KEY=VALUE``
   Often used here for training-specific overrides such as
   ``--set EPOCHS=150`` or ``--set BATCH_SIZE=64``. 

How to think about this command
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This command is best understood as the **single-run training entry
point**.

If you already know the model setup you want, Stage 2 is usually the
right tool. You use Stage 3 only when you want to *search* across many
candidate settings rather than commit to one fixed training
configuration. That distinction is reflected directly in the wrappers:
Stage 2 is described as the safe training entry point, while Stage 3 is
described as the safe tuning entry point. 

See also
^^^^^^^^

- :doc:`../user_guide/stage2`
- :doc:`../auto_examples/diagnostics/plot_stage2_training_curves`

.. _run-stage3:

``stage3-tune``
--------------------

Use ``stage3-tune`` when you want to **search for better
hyperparameters** rather than run one fixed training configuration.

The Stage-3 wrapper is documented as a safe tuning entry point from
``geoprior.cli``. Its supported flows include:

- using the existing ``nat.com/config.py`` as-is,
- installing a user-supplied config file before running,
- applying one-off ``--set KEY=VALUE`` overrides,
- pointing the tuning run at a specific Stage-1 manifest via
  ``--stage1-manifest``. 

In practice, Stage 3 is the command you reach for when Stage 1 is
already done and you want to **explore the search space** instead of
training just one configuration.

Why you would use Stage 3
^^^^^^^^^^^^^^^^^^^^^^^^^

Choose ``stage3-tune`` when you want to:

- search for stronger hyperparameter settings,
- tune from a stable Stage-1 artifact set,
- compare candidate training configurations more systematically,
- experiment with search budget controls such as trial counts via
  temporary overrides.

A simple rule is:

- use **Stage 2** when the configuration is already chosen,
- use **Stage 3** when you still want the CLI to help select that
  configuration. 

Usage
^^^^^^^^^

Run tuning with the active config:

.. code-block:: bash

   geoprior run stage3-tune

or:

.. code-block:: bash

   geoprior-run stage3-tune

Tune from one explicit Stage-1 manifest:

.. code-block:: bash

   geoprior-run stage3-tune \
       --stage1-manifest path/to/manifest.json

Install a specific config before launching the search:

.. code-block:: bash

   geoprior-run stage3-tune \
       --config my_config.py

Apply a simple trial-budget override:

.. code-block:: bash

   geoprior-run stage3-tune \
       --set MAX_TRIALS=20

Tune with several temporary search overrides:

.. code-block:: bash

   geoprior-run stage3-tune \
       --set MAX_TRIALS=30 \
       --set EPOCHS=80 \
       --set BATCH_SIZE=64

Override the run identity together with the Stage-1 manifest:

.. code-block:: bash

   geoprior-run stage3-tune \
       --city zhongshan \
       --model GeoPriorSubsNet \
       --stage1-manifest results/zhongshan_stage1/artifacts/manifest.json \
       --set MAX_TRIALS=25


Key command-line options
^^^^^^^^^^^^^^^^^^^^^^^^

Stage 3 follows the same shared run conventions as Stages 1 and 2.
In practice, the most important tuning-oriented option to highlight is:

``--set KEY=VALUE``
   Apply temporary tuning-related overrides such as
   ``--set MAX_TRIALS=20`` without editing the base config. 

How to think about this command
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This command is best understood as the **search-oriented companion** to
Stage 2.

Both commands rely on the same general outer pattern:

- optional config installation,
- optional runtime overrides,
- optional explicit Stage-1 manifest selection. 

The difference is user intent:

- Stage 2 says, “train this configuration,”
- Stage 3 says, “search for a stronger configuration.”

That distinction is important in a guide, because it helps users decide
quickly which command family member they actually need.

See also
^^^^^^^^

- :doc:`../user_guide/stage3`
- :doc:`../auto_examples/diagnostics/plot_stage3_tuning_summary`


.. _run-stage4:

``stage4-infer``
--------------------

Use ``stage4-infer`` when you want to run **inference, evaluation,
calibration, or forecast export** from an already prepared and trained
pipeline state.

The Stage-4 wrapper keeps the newer GeoPrior CLI style, but it also
forwards inference-specific arguments to the legacy inference backend.
In addition to the shared run conventions introduced earlier, its help
surface explicitly exposes forwarded inference controls such as:

- ``--stage1-dir``
- ``--manifest``
- ``--model-path``
- ``--dataset {test,val,train,custom}``
- ``--inputs-npz``
- ``--targets-npz``
- ``--eval-losses``
- ``--eval-physics``
- ``--calibrator``
- ``--use-source-calibrator``
- ``--fit-calibrator``
- ``--cov-target``
- ``--batch-size``
- ``--no-figs``
- ``--include-gwl``. 

This makes Stage 4 the most flexible of the main staged commands: it can
run simple inference from the existing config, but it can also drive
more explicit evaluation and calibration workflows when needed. 

Usage
^^^^^^

Run inference with the active config and default artifact resolution:

.. code-block:: bash

   geoprior run stage4-infer

or:

.. code-block:: bash

   geoprior-run stage4-infer

Point Stage 4 at one explicit Stage-1 manifest:

.. code-block:: bash

   geoprior-run stage4-infer \
       --stage1-manifest path/to/manifest.json

Use the forwarded legacy manifest path explicitly:

.. code-block:: bash

   geoprior-run stage4-infer \
       --manifest path/to/stage1/manifest.json

Run a more explicit evaluation pass:

.. code-block:: bash

   geoprior-run stage4-infer \
       --manifest path/to/stage1/manifest.json \
       --eval-losses \
       --eval-physics

Select a particular dataset split or custom NPZ inputs:

.. code-block:: bash

   geoprior-run stage4-infer \
       --dataset custom \
       --inputs-npz results/custom_inputs.npz \
       --targets-npz results/custom_targets.npz

Use a trained model path directly:

.. code-block:: bash

   geoprior-run stage4-infer \
       --model-path results/models/best_model.keras \
       --dataset test

Run calibration-oriented inference:

.. code-block:: bash

   geoprior-run stage4-infer \
       --manifest path/to/stage1/manifest.json \
       --fit-calibrator \
       --cov-target 0.80

Or reuse an existing calibrator:

.. code-block:: bash

   geoprior-run stage4-infer \
       --manifest path/to/stage1/manifest.json \
       --calibrator results/calibration/calibrator.joblib \
       --use-source-calibrator

Stage-specific options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage 4 follows the shared staged-run conventions introduced earlier.
The additional options worth highlighting here are the ones that make
inference more explicit or more controllable:

``--stage1-manifest``
   Use one exact Stage-1 manifest for the run. As in the earlier stages,
   this is the wrapper-level way to anchor execution to a known
   preprocessing output. 

``--manifest``
   Forward a legacy inference manifest path explicitly. This is useful
   when you want to drive the older inference backend more directly.
   

``--model-path``
   Point inference to a specific saved model artifact. 

``--dataset``
   Select which data split or input mode to evaluate, including a
   ``custom`` mode. 

``--eval-losses`` and ``--eval-physics``
   Enable more explicit evaluation passes beyond a simple prediction
   run. 

``--calibrator`` / ``--fit-calibrator`` / ``--use-source-calibrator``
   Control whether calibration is reused or fit during the Stage-4 run.
   

See also
^^^^^^^^

- :doc:`../user_guide/stage4`
- :doc:`../auto_examples/forecasting/index`
- :doc:`../user_guide/inference_and_export`

.. _run-stage5:

``stage5-transfer``
--------------------

Use ``stage5-transfer`` when you want to run **cross-city transfer
evaluation**.

The Stage-5 wrapper aligns transfer evaluation with the GeoPrior CLI
while preserving the existing transfer backend. In addition to the
shared staged-run conventions, it can seed transfer defaults such as the
city pair, model name, and results directory before forwarding richer
transfer arguments downstream. Its help text explicitly lists forwarded
transfer controls such as:

- ``--city-a``
- ``--city-b``
- ``--results-dir``
- ``--splits {val,test}``
- ``--strategies {baseline,xfer,warm}``
- ``--calib-modes {none,source,target}``
- ``--rescale-modes {as_is,strict}``
- ``--model-name``
- ``--source-model {auto,tuned,trained}``
- ``--source-load {auto,full,weights}``
- ``--hps-mode {auto,tuned,trained}``
- ``--prefer-artifact {keras,weights}``
- ``--warm-split {train,val}``. 

That makes Stage 5 more than a simple “transfer on/off” switch. It is a
structured command for comparing transfer strategies, calibration modes,
and artifact-loading choices across source and target cities. 

From basic to more advanced usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run transfer evaluation with the active config:

.. code-block:: bash

   geoprior run stage5-transfer

or:

.. code-block:: bash

   geoprior-run stage5-transfer

Seed the transfer city pair from the wrapper:

.. code-block:: bash

   geoprior-run stage5-transfer \
       --city-a nansha \
       --city-b zhongshan

Set an explicit results directory:

.. code-block:: bash

   geoprior-run stage5-transfer \
       --city-a nansha \
       --city-b zhongshan \
       --results-dir results/transfer_run

Compare several transfer strategies on more than one split:

.. code-block:: bash

   geoprior-run stage5-transfer \
       --city-a nansha \
       --city-b zhongshan \
       --splits val test \
       --strategies baseline xfer warm

Control calibration and rescaling behavior:

.. code-block:: bash

   geoprior-run stage5-transfer \
       --city-a nansha \
       --city-b zhongshan \
       --calib-modes none source target \
       --rescale-modes as_is strict

Steer which source artifacts and hyperparameter mode are preferred:

.. code-block:: bash

   geoprior-run stage5-transfer \
       --city-a nansha \
       --city-b zhongshan \
       --source-model tuned \
       --source-load full \
       --hps-mode tuned \
       --prefer-artifact keras

Run a more composed warm-transfer experiment:

.. code-block:: bash

   geoprior-run stage5-transfer \
       --city-a nansha \
       --city-b zhongshan \
       --strategies warm \
       --warm-split train \
       --calib-modes target \
       --results-dir results/warm_transfer_eval

Stage-specific options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage 5 follows the shared staged-run conventions introduced earlier.
The most distinctive transfer-oriented options are:

``--city-a`` and ``--city-b``
   Define the source and target cities for transfer evaluation. The
   wrapper can seed these defaults before forwarding to the legacy
   backend. 

``--results-dir``
   Set the transfer results directory explicitly. 

``--strategies``
   Compare transfer modes such as ``baseline``, ``xfer``, and
   ``warm``. 

``--splits``
   Choose which evaluation splits to run, such as ``val`` or ``test``.

``--calib-modes`` and ``--rescale-modes``
   Control how calibration and scaling behavior are treated during
   transfer evaluation. 

``--source-model`` / ``--source-load`` / ``--hps-mode`` / ``--prefer-artifact``
   Select how the source model and its preferred artifacts are resolved
   for the transfer workflow. 

See also
^^^^^^^^

- :doc:`../user_guide/stage5`
- :doc:`../auto_examples/figure_generation/plot_xfer_transferability`

.. _run-sensitivity:

``sensitivity``
--------------------

Use ``sensitivity`` when you want to run a **physics sensitivity grid**
rather than one fixed staged training or inference workflow.

GeoPrior exposes ``sensitivity`` as a public run-family command through
the main dispatcher. The driver is designed to sweep combinations of
physics-related weights such as ``lambda_cons`` and ``lambda_prior``
across one or more PDE modes, using the Stage-2 sensitivity training
path underneath. It also includes resume logic so previously completed
grid points can be skipped on restart. 

This command is a good fit when you want to answer questions like:

- how sensitive is training or evaluation to physics weighting,
- which physics-loss balance produces more stable behavior,
- whether a smaller “fast” grid is enough before committing to a larger
  sweep.

Usage
^^^^^

Run the default sensitivity grid with the current environment:

.. code-block:: bash

   geoprior run sensitivity

or:

.. code-block:: bash

   geoprior-run sensitivity

Inspect the full CLI surface first:

.. code-block:: bash

   geoprior-run sensitivity --help

Run a shorter sweep with fewer epochs:

.. code-block:: bash

   geoprior-run sensitivity \
       --epochs 10

Sweep a custom grid of physics weights:

.. code-block:: bash

   geoprior-run sensitivity \
       --pde-modes both \
       --lcons 0.0 0.05 0.2 1.0 \
       --lprior 0.0 0.05 0.2 1.0

Run a lighter and faster experiment:

.. code-block:: bash

   geoprior-run sensitivity \
       --epochs 10 \
       --fast \
       --eval-max-batches 50

Control execution mode and parallelism:

.. code-block:: bash

   geoprior-run sensitivity \
       --gold \
       --n-jobs -1 \
       --threads 20

Steer device selection explicitly:

.. code-block:: bash

   geoprior-run sensitivity \
       --device gpu \
       --gpu-ids 0 1 \
       --gpu-allow-growth

Resume-aware or dry-run planning:

.. code-block:: bash

   geoprior-run sensitivity \
       --scan-root results/zhongshan \
       --dry-run

or force a fresh rerun:

.. code-block:: bash

   geoprior-run sensitivity \
       --epochs 20 \
       --no-resume

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This command follows the shared run conventions introduced earlier, but
the options that matter most here are the sensitivity-grid controls:

``--epochs``
   Set the number of epochs per grid run. The driver describes these as
   short sensitivity runs. 

``--pde-modes``
   Choose which PDE modes to sweep. 

``--lcons`` and ``--lprior``
   Define the grid for ``lambda_cons`` and ``lambda_prior``. These are
   the core sweep dimensions of the command. 

``--fast`` and ``--eval-max-batches``
   Reduce the workload for exploratory sweeps. 

``--gold`` / ``--inprocess`` / ``--n-jobs`` / ``--threads``
   Control how the sweep is executed and how much parallelism is used.
   

``--device`` / ``--gpu-ids`` / ``--gpu-allow-growth``
   Control CPU or GPU execution behavior. 

``--no-resume`` / ``--scan-root`` / ``--dry-run``
   Control restart behavior, completed-run scanning, and planning
   without execution. 

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_physics_sensitivity`

.. _run-identifiability:

``identifiability``
--------------------

Use ``identifiability`` when you want to run **SM3 synthetic
identifiability experiments** from the run family.

This is the canonical public run-family command name registered by the
dispatcher. The wrapper integrates the standalone SM3 identifiability
script into ``geoprior.cli`` so it can be launched from
``geoprior-run`` while still forwarding the richer legacy SM3 argument
surface. In addition to the shared run conventions, it can seed a
default ``--outdir`` and ``--ident-regime`` from config when those are
not supplied explicitly. 

This command is the run-side companion to the plot-side SM3
identifiability figure: you use this command to **generate the
experiment outputs**, and the figure command later visualizes them.

Usage
^^^^^

Run the default SM3 identifiability workflow:

.. code-block:: bash

   geoprior run identifiability

or:

.. code-block:: bash

   geoprior-run identifiability

Inspect the full wrapper plus forwarded legacy help:

.. code-block:: bash

   geoprior-run identifiability --help

Choose an explicit output directory:

.. code-block:: bash

   geoprior-run identifiability \
       --outdir results/sm3_ident_run

Choose an identifiability regime explicitly:

.. code-block:: bash

   geoprior-run identifiability \
       --ident-regime anchored

Control the synthetic experiment size:

.. code-block:: bash

   geoprior-run identifiability \
       --n-realizations 50 \
       --n-years 25 \
       --time-steps 5 \
       --forecast-horizon 3

Control optimization settings:

.. code-block:: bash

   geoprior-run identifiability \
       --epochs 40 \
       --noise-std 0.02 \
       --load-type step

Choose what to identify and under which scenario:

.. code-block:: bash

   geoprior-run identifiability \
       --identify both \
       --scenario base \
       --ident-regime closure_locked

Combine wrapper defaults with one-off config overrides:

.. code-block:: bash

   geoprior-run identifiability \
       --outdir results/sm3_both_run \
       --set IDENTIFIABILITY_REGIME='data_relaxed'

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This command follows the shared run conventions introduced earlier, but
the most distinctive options here are the SM3-oriented controls:

``--outdir``
   Set the default SM3 output directory when it is not already supplied
   downstream. The wrapper can also seed this from config. 

``--ident-regime``
   Choose the identifiability regime explicitly. The wrapper can seed a
   default regime from config, and the forwarded help lists regimes such
   as ``none``, ``base``, ``anchored``, ``closure_locked``, and
   ``data_relaxed``. 

``--n-realizations`` / ``--n-years`` / ``--time-steps`` / ``--forecast-horizon``
   Control the synthetic experiment size and time structure. These are
   forwarded legacy arguments exposed by the wrapper help. 

``--epochs`` / ``--noise-std`` / ``--load-type``
   Control the optimization and noise structure of the identifiability
   experiment. 

``--identify`` and ``--scenario``
   Select what quantity is being identified and under which synthetic
   scenario. The forwarded help lists values such as ``tau``, ``k``,
   and ``both`` for ``--identify``. 

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_sm3_identifiability`

.. _run-sm3-offsets:

``sm3-offset-diagnostics``
------------------------------

Use ``sm3-offset-diagnostics`` when you want to run **SM3 log-offset
diagnostics** from the run family.

GeoPrior registers this as a dedicated public run-family command for the
SM3 diagnostic workflow. In the command registry, it is exposed under
the canonical name ``sm3-offset-diagnostics`` and also accepts aliases
such as ``offset-diagnostics``, ``offsets``, and ``sm3-offsets``.

This command is the run-side companion to the log-offset figure page:
you use it to generate the diagnostic outputs, then the plot-side
workflow visualizes those results.

Usage
^^^^^

Run the default offset-diagnostics workflow:

.. code-block:: bash

   geoprior run sm3-offset-diagnostics

or:

.. code-block:: bash

   geoprior-run sm3-offset-diagnostics

Inspect the command surface first:

.. code-block:: bash

   geoprior-run sm3-offset-diagnostics --help

The command also accepts its shorter aliases:

.. code-block:: bash

   geoprior-run offsets --help
   geoprior-run sm3-offsets --help

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This command follows the shared run conventions introduced earlier. For
the detailed diagnostic-specific arguments, the best reference is the
command help itself:

.. code-block:: bash

   geoprior-run sm3-offset-diagnostics --help

That keeps this page focused on the command's role in the workflow while
avoiding duplication once the full offset wrapper details are documented
elsewhere.

**Related figure:**

- :doc:`../auto_examples/figure_generation/plot_sm3_log_offsets`

.. _run-sm3-suite:

``sm3-suite``
--------------------

Use ``sm3-suite`` when you want to launch a **preset-driven multi-regime
SM3 suite** rather than a single SM3 run.

The suite runner is designed as a portable Python CLI that replaces the
earlier shell-only SM3 launchers. It supports named presets, regime
selection, device selection, explicit suite roots, resume-latest
behavior, dry-run mode, and optional combined-summary collection at the
end. 

The preset layer currently defines named presets such as ``tau50`` and
``both50``. The available SM3 regimes include ``none``, ``base``,
``anchored``, ``closure_locked``, and ``data_relaxed``. 

This command is a good fit when you want to:

- run the same SM3 workflow across several regimes,
- compare preset bundles without hand-building each command,
- resume the latest suite directory instead of starting over,
- collect one combined summary table after all regimes finish.

Usage
^^^^^

Run the default preset suite:

.. code-block:: bash

   geoprior run sm3-suite --preset tau50

or:

.. code-block:: bash

   geoprior-run sm3-suite --preset tau50

Run a different preset across selected regimes:

.. code-block:: bash

   geoprior-run sm3-suite \
       --preset both50 \
       --regime anchored \
       --regime closure_locked

List the available regimes and exit:

.. code-block:: bash

   geoprior-run sm3-suite --list-regimes

Choose the execution device explicitly:

.. code-block:: bash

   geoprior-run sm3-suite \
       --preset tau50 \
       --device gpu

Reuse the latest matching suite directory:

.. code-block:: bash

   geoprior-run sm3-suite \
       --preset tau50 \
       --resume-latest

Use an explicit suite root instead:

.. code-block:: bash

   geoprior-run sm3-suite \
       --preset tau50 \
       --suite-root results/sm3_tau_suite_custom

Run a planning pass without execution:

.. code-block:: bash

   geoprior-run sm3-suite \
       --preset both50 \
       --dry-run

Adjust a few suite-level training settings:

.. code-block:: bash

   geoprior-run sm3-suite \
       --preset both50 \
       --epochs 30 \
       --batch 64 \
       --patience 5 \
       --n-realizations 25

Skip combined summary collection:

.. code-block:: bash

   geoprior-run sm3-suite \
       --preset tau50 \
       --skip-collect

Distinctive options to notice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This command follows the shared run conventions introduced earlier, but
the most distinctive suite-oriented options are:

``--preset``
   Choose a named preset bundle such as ``tau50`` or ``both50``.
   

``--regime`` / ``--regimes`` / ``--regime-ids``
   Select which SM3 regimes to include in the suite. 

``--list-regimes``
   Print the available regimes and exit. 

``--device``
   Choose ``auto``, ``cpu``, or ``gpu`` execution. 

``--suite-root`` and ``--resume-latest``
   Control whether the suite writes to a new location, an explicit
   location, or the newest matching previous suite directory.
   

``--dry-run``
   Show the resolved suite commands without executing them.
   

``--skip-collect``
   Skip the final combined-summary collection step. 

``--epochs`` / ``--batch`` / ``--patience`` / ``--n-realizations``
   Adjust the size and training behavior of each regime run in the
   suite. 

**Related figures:**

- :doc:`../auto_examples/figure_generation/plot_sm3_bounds_ridge_summary`
- :doc:`../auto_examples/figure_generation/plot_sm3_log_offsets`


From here
---------

A good next reading path is:

- :doc:`../user_guide/stage1`
- :doc:`../user_guide/stage2`
- :doc:`../user_guide/stage3`
- :doc:`../user_guide/stage4`
- :doc:`../user_guide/stage5`
