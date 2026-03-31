Stage-1
=======

Stage-1 is the **preprocessing and sequence-export stage**
of the GeoPrior-v3 workflow.

Its role is to transform raw city-level data into a
well-defined, reproducible artifact set that later stages
can consume without repeating the data preparation logic.
For GeoPrior-v3, this is a critical boundary: Stage-2 and
later stages are expected to read the manifest and exported
NPZ bundles produced here, rather than rebuilding the same
inputs independently.

In the current implementation, Stage-1 is explicitly
described as:

- loading the dataset,
- cleaning and selecting features,
- encoding and scaling,
- defining feature sets,
- splitting by year and building PINN-style sequences,
- building train/validation datasets and exporting all
  arrays and metadata. 

The script header also makes clear that the main outputs are
CSV snapshots, joblib-serialized encoders and scalers, NPZ
input/target bundles, and a ``manifest.json`` that describes
paths, shapes, dimensions, columns, and configuration for the
handoff into later stages. 

Why Stage-1 matters
-------------------

Stage-1 is not a cosmetic preprocessing step. It establishes
the **data contract** for the rest of the workflow.

That contract includes:

- which columns are treated as raw identifiers,
- which columns become model-facing targets,
- which features are static, dynamic, or future-known,
- how coordinates are prepared,
- how unit-aware scaling metadata is recorded,
- how train/validation/test splits are formed,
- and which exact arrays later stages should load.

This matters because GeoPrior-v3 is a physics-guided system.
A mismatch in feature order, units, coordinate treatment, or
target semantics can make a later run look numerically valid
while still being scientifically inconsistent.

Stage-2 relies directly on the Stage-1 manifest and NPZ
artifacts, and it checks that exported dynamic and future
feature dimensions remain consistent with the recorded
feature lists. 

What Stage-1 does
-----------------

The Stage-1 script is organized around a staged internal
pipeline.

The current script header describes six main steps:

1. load the dataset,
2. clean and select features,
3. encode and scale,
4. define feature sets,
5. split by year and build PINN sequences,
6. build train/validation data and export arrays and
   metadata. 

In practice, the implementation also includes additional
export-oriented work after the main preprocessing path, such
as:

- building test split NPZ bundles,
- optionally building out-of-sample time bundles,
- optionally building future-sequence NPZ bundles for later
  stages,
- and finally writing the manifest. 

Stage-1 responsibilities
------------------------

Stage-1 is responsible for preparing the workflow state that
all later stages assume.

Its responsibilities include:

- resolving the effective configuration for the run,
- locating or reconstructing the input dataset,
- normalizing groundwater column aliases when needed,
- reconciling subsidence naming and rate/cumulative forms,
- resolving optional numeric and categorical features,
- applying censoring logic and effective-field handling,
- encoding and scaling where appropriate,
- building sequence-ready tensors,
- exporting durable artifacts,
- and recording the full handshake in ``manifest.json``. 

Configuration and CLI entry
---------------------------

Stage-1 is exposed through the newer CLI wrapper pattern and
is meant to be launched from the GeoPrior command surface,
not only by running a legacy standalone script.

The Stage-1 CLI parser currently supports the following
wrapper-level options:

- ``--config``
- ``--config-root``
- ``--city``
- ``--model``
- ``--data-dir``
- repeated ``--set KEY=VALUE`` overrides. 

This design allows a user to:

- install a user-supplied configuration into the working
  config location,
- override selected configuration values for a single run,
- and keep the actual preprocessing logic isolated behind a
  stable stage entry point.

.. note::

   For the exact invocation syntax in your installed
   environment, prefer the live CLI help output. The wrapper
   interface is explicit and stable in intent, but the best
   source for the command form is still:

   .. code-block:: bash

      geoprior-run --help

   and the relevant Stage-1 subcommand help exposed there.

A typical Stage-1 launch pattern
--------------------------------

A common first pattern is:

.. code-block:: bash

   geoprior-init --help
   geoprior-run --help

Then launch Stage-1 using your project configuration and any
one-off overrides needed for the current run.

Examples of wrapper-level override style include values such
as:

.. code-block:: bash

   --city zhongshan
   --model GeoPriorSubsNet
   --data-dir /path/to/data
   --set TIME_STEPS=6
   --set FORECAST_HORIZON_YEARS=3

The important idea is that Stage-1 should be driven by
configuration and explicit overrides, not by ad hoc edits
inside the script body. 

Inputs expected by Stage-1
--------------------------

Stage-1 expects a valid GeoPrior configuration and a dataset
layout consistent with the configured city, file naming, and
column conventions.

At runtime, the stage resolves core identifiers such as:

- city name,
- model name,
- data directory,
- large and fallback filenames,
- time-window settings,
- coordinate and target column names,
- optional numeric and categorical feature specifications,
- future-driver feature lists,
- censoring settings,
- scaling and coordinate controls,
- holdout and split settings,
- and the base output directory. 

The stage is also written to search multiple configured paths
for the input CSVs and, if needed, can fall back to a merged
source or dataset fetcher path before saving a raw trace copy
for the run. 

Internal step structure
-----------------------

The user does not need to understand every internal helper to
use Stage-1, but it is useful to know the overall structure.

**Dataset loading**

Stage-1 first locates and reads the input dataset, then saves
a raw CSV snapshot into the run directory for traceability. It
also normalizes groundwater aliasing early so later logic can
use a consistent groundwater column reference. 

**Initial preprocessing**

The stage reconciles key column semantics, including the
subsidence column contract, optional feature resolution, and
censoring logic. This is where raw data begins to acquire the
clear model-facing roles used later in the workflow. 

**Encoding and scaling**

Stage-1 creates or records the encoders and scalers required
by later stages. These are stored as artifacts so the same
transformations can be reused rather than reconstructed by
guesswork. 

**Sequence building**

The stage prepares sequence tensors for the GeoPrior
workflow, including the model-facing inputs and targets
needed for training and evaluation. It supports the current
time-window and forecast-horizon logic, and its output
shapes are recorded in the manifest for handshake checks. 

**Artifact export**

The stage exports train, validation, and test NPZ bundles and
writes a manifest that becomes the canonical reference for
later stages. Optional out-of-sample time and future-sequence
bundles may also be attached. 

Primary artifacts written by Stage-1
------------------------------------

Stage-1 writes several kinds of artifacts.

**CSV snapshots**

The script header and run logic indicate that Stage-1 writes
CSV outputs such as raw, cleaned, and scaled tables for
traceability and inspection. 

**Encoders and scalers**

The manifest records encoder and scaler artifacts, including
one-hot encoders, coordinate scalers, and the main scaler
when applicable. It also records which ML numeric columns
were scaled. 

**NPZ bundles**

Stage-1 exports compressed NumPy bundles for the core
workflow handoff, including:

- ``train_inputs.npz``
- ``train_targets.npz``
- ``val_inputs.npz``
- ``val_targets.npz``
- ``test_inputs.npz``
- ``test_targets.npz``. 

**Manifest**

Finally, Stage-1 writes ``manifest.json`` in the run
directory. This manifest is the main handshake artifact for
later stages. 

The manifest contract
---------------------

The Stage-1 manifest is intentionally rich. It is not merely
a filename index.

The manifest records, among other things:

- artifact paths,
- tensor shapes,
- sequence dimensions,
- feature lists,
- column roles,
- configuration snapshot,
- split and holdout details,
- and software version metadata. 

One especially important part of the manifest is the
canonical naming block that records column roles such as:

- raw and model-space subsidence columns,
- groundwater driver and target columns,
- thickness columns,
- surface-elevation columns when present,
- time and coordinate columns actually used in sequences. 

This is what makes Stage-1 the authoritative source for the
later workflow. The manifest tells later stages not just
*where* the files are, but *what those files mean*.

Train, validation, and test handoff
-----------------------------------

Stage-1 prepares more than a single train/validation split.

The current implementation records train, validation, and
test NPZ bundles, and also stores corresponding shape
metadata and split summaries in the manifest. The holdout
block records sequence counts, group counts, row counts, and
group-artifact CSV paths, making the split strategy visible
rather than hidden. 

This is valuable for later troubleshooting. If a downstream
result looks odd, the first question is often whether the
Stage-1 split, grouping, or window validity logic produced
what you expected.

Optional OOS-time and future artifacts
--------------------------------------

Stage-1 can also generate additional artifacts beyond the
main train/validation/test handoff.

The current script supports:

- optional out-of-sample time NPZ exports, and
- optional ``future_*`` NPZ construction when
  ``BUILD_FUTURE_NPZ`` is enabled. 

These are useful when later stages need future-oriented
sequence material or transfer-oriented evaluation artifacts
without forcing Stage-1 to be rerun from scratch.

How Stage-2 uses Stage-1 outputs
--------------------------------

Later stages are designed to consume Stage-1 directly.

The Stage-2 workflow loads the Stage-1 manifest, reads the
exported NPZ bundles, infers or checks dimensions, and
verifies consistency between tensor last dimensions and the
recorded feature lists. In other words, Stage-2 treats
Stage-1 as the authoritative preprocessing boundary. 

This is why Stage-1 should be treated carefully. If Stage-1
artifacts drift away from the configuration or the recorded
feature order, the entire downstream workflow becomes harder
to trust.

What to inspect after Stage-1 completes
---------------------------------------

A successful Stage-1 run is not only one that exits cleanly.
It is one whose outputs also make sense.

After Stage-1, inspect at least the following:

- the run directory and artifacts directory,
- the saved raw and processed CSVs,
- the NPZ file presence and sizes,
- the manifest,
- the recorded feature lists,
- the recorded shapes,
- the holdout summary,
- and any censoring or scaling metadata. 

You should be able to answer basic questions such as:

- Did the correct city and model flow into the run?
- Were the expected feature lists recorded?
- Do the tensor shapes match the intended horizon and window?
- Were train/validation/test outputs all written?
- Does the manifest reflect the configuration you meant to
  run?

Common Stage-1 mistakes
-----------------------

The most common Stage-1 problems are workflow and data
contract issues.

**Using the wrong config**

Stage-1 is configuration-driven. A wrong city, data path, or
column setting can produce apparently valid artifacts that do
not correspond to the intended case.

**Assuming column semantics**

Groundwater, head, depth, and subsidence naming are handled
explicitly in Stage-1. Do not assume that a raw dataset name
already matches the model-facing semantics.

**Ignoring feature-order drift**

Later stages validate tensor dimensions against Stage-1
feature lists. Even if shapes look plausible, the ordering of
dynamic or future features still matters. 

**Treating the manifest as optional**

The manifest is the handshake. If it is missing, stale, or
out of sync with the NPZ exports, later stages lose their
most reliable source of truth.

Best practices
--------------

.. admonition:: Best practice

   Treat Stage-1 as the authoritative preprocessing boundary.

   Once a Stage-1 run is complete, later stages should load
   its exported artifacts rather than reconstructing the
   preprocessing path from scratch.

.. admonition:: Best practice

   Inspect the manifest before moving to Stage-2.

   The manifest is where you can confirm feature order,
   target semantics, split summaries, shape expectations, and
   artifact paths in one place.

.. admonition:: Best practice

   Keep one run directory per coherent configuration.

   Mixing artifacts from multiple Stage-1 runs is one of the
   fastest ways to create silent workflow drift.

A compact Stage-1 map
---------------------

The Stage-1 workflow can be summarized like this:

.. code-block:: text

   config + data paths
        ↓
   load dataset
        ↓
   clean / select / reconcile columns
        ↓
   encode / scale / record metadata
        ↓
   build train / val / test sequences
        ↓
   export NPZ bundles + encoders/scalers
        ↓
   write manifest.json
        ↓
   hand off to Stage-2

Read next
---------

The next pages after Stage-1 are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Stage-2
      :link: stage2
      :link-type: doc
      :class-card: sd-shadow-sm

      See how GeoPrior consumes the Stage-1 manifest and
      exported tensors for training-oriented execution.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc
      :class-card: sd-shadow-sm card--configuration

      Understand how configuration drives Stage-1 inputs,
      scaling, paths, and split behavior.

   .. grid-item-card:: CLI guide
      :link: cli
      :link-type: doc
      :class-card: sd-shadow-sm card--cli


      Move from the stage narrative to the full command
      surface.

   .. grid-item-card:: Data and units
      :link: ../scientific_foundations/data_and_units
      :link-type: doc
      :class-card: sd-shadow-sm

      Review unit conventions and scaling assumptions that
      begin in Stage-1 and matter later for physics-guided
      execution.

.. seealso::

   - :doc:`workflow_overview`
   - :doc:`stage2`
   - :doc:`configuration`
   - :doc:`cli`
   - :doc:`../scientific_foundations/data_and_units`
   - :doc:`../scientific_foundations/scaling`