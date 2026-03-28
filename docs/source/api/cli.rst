CLI and command registry
========================

GeoPrior-v3 exposes a staged command-line interface for
pipeline execution, artifact materialization, and figure
rendering. The refactored layout uses
``geoprior._scripts`` as the authoritative home for
reproducibility commands, while the top-level
``scripts`` package remains a backward-compatible launcher.

The practical rule is simple:

- ``geoprior`` and its family entry points are the modern
  public interface.
- ``geoprior._scripts`` owns the real script registry and
  artifact-path policy.
- ``scripts`` mirrors that registry for legacy invocations
  such as ``python -m scripts ...``.

.. note::

   The root dispatcher no longer depends on the legacy
   ``scripts`` package as its source of truth. Instead,
   both the modern CLI and the compatibility launcher read
   from ``geoprior._scripts.registry``.

Architecture overview
---------------------

The command stack is split into three layers.

1. **Public entry points**

   - ``geoprior``
   - ``geoprior-run``
   - ``geoprior-build``
   - ``geoprior-plot``
   - ``geoprior-init``

2. **Internal implementation**

   - ``geoprior.__main__``
   - ``geoprior.cli``
   - ``geoprior.cli._dispatch``
   - ``geoprior._scripts.registry``
   - ``geoprior._scripts.config``

3. **Compatibility layer**

   - ``scripts.__main__``
   - ``scripts.registry``
   - ``python -m scripts <command>``

A compact conceptual map is shown below.

.. code-block:: text

   geoprior/
   ├── __main__.py
   ├── cli/
   │   ├── __init__.py
   │   ├── __main__.py
   │   ├── _dispatch.py
   │   ├── _presets.py
   │   ├── config.py
   │   ├── init_config.py
   │   ├── run_sensitivity.py
   │   ├── run_sm3_suite.py
   │   ├── stage1.py
   │   ├── stage2.py
   │   ├── stage3.py
   │   ├── stage4.py
   │   ├── stage5.py
   │   └── build_*.py / sm3_*.py / ...
   ├── _scripts/
   │   ├── registry.py
   │   ├── config.py
   │   └── plot_*.py / build_*.py / ...
   └── ...

   scripts/
   ├── __init__.py
   ├── __main__.py
   └── registry.py


Entry points and usage model
----------------------------

The modern CLI supports two complementary styles.

Root entry point
~~~~~~~~~~~~~~~~

The root command uses explicit families:

.. code-block:: bash

   geoprior run <command> [args]
   geoprior build <command> [args]
   geoprior plot <command> [args]

Examples:

.. code-block:: bash

   geoprior run stage1-preprocess
   geoprior build exposure --help
   geoprior plot physics-fields --help

Family-specific entry points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dedicated entry points remove the family prefix:

.. code-block:: bash

   geoprior-run <command> [args]
   geoprior-build <command> [args]
   geoprior-plot <command> [args]
   geoprior-init [args]

Examples:

.. code-block:: bash

   geoprior-run stage4-infer --help
   geoprior-build model-metrics
   geoprior-plot transfer-impact
   geoprior-init --yes

Backward compatibility launcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The legacy launcher remains available:

.. code-block:: bash

   python -m scripts <command> [args]

Examples:

.. code-block:: bash

   python -m scripts plot-physics-fields --help
   python -m scripts make-exposure --help

This compatibility surface is intentionally thin. It
re-exports the registry and dispatches to modules stored in
``geoprior._scripts``.

Command families
----------------

GeoPrior-v3 groups commands into three public families.

Run family
~~~~~~~~~~

The ``run`` family executes staged workflows and related
experiment drivers.

Typical commands include:

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

The ``build`` family materializes derived artifacts, helper
products, and summary tables.

Typical commands include:

- ``full-inputs-npz``
- ``physics-payload-npz``
- ``external-validation-fullcity``
- ``sm3-collect-summaries``
- ``assign-boreholes``
- ``add-zsurf-from-coords``
- ``external-validation-metrics``
- ``brier-exceedance``
- ``hotspots``
- ``hotspots-summary``
- ``extend-forecast``
- ``update-ablation-records``
- ``model-metrics``
- ``ablation-table``
- ``boundary``
- ``exposure``
- ``district-grid``
- ``clusters-with-zones``

Plot family
~~~~~~~~~~~

The ``plot`` family renders publication-facing figures,
appendix panels, maps, and transfer-analysis visuals.

Typical commands include:

- ``driver-response``
- ``core-ablation``
- ``litho-parity``
- ``uncertainty``
- ``spatial-forecasts``
- ``physics-sanity``
- ``physics-maps``
- ``physics-fields``
- ``physics-profiles``
- ``uncertainty-extras``
- ``ablations-sensitivity``
- ``physics-sensitivity``
- ``sm3-identifiability``
- ``sm3-bounds-ridge-summary``
- ``sm3-log-offsets``
- ``xfer-transferability``
- ``xfer-impact``
- ``transfer``
- ``transfer-impact``
- ``geo-cumulative``
- ``hotspot-analytics``
- ``external-validation``

.. note::

   The exact command list is defined in the registry. The
   examples above reflect the current public surface exposed
   by the dispatcher and the script registry.

Registry ownership
------------------

The central design change in the refactor is the ownership
model for reproducibility commands.

Before the refactor, the modern CLI treated ``scripts`` as a
registry bridge. After the refactor, the authoritative
registry lives in ``geoprior._scripts.registry``.

That yields a cleaner dependency direction:

- ``geoprior.cli`` → ``geoprior._scripts``
- ``scripts`` → ``geoprior._scripts``

and avoids making the main package depend on a legacy shim.

Registry structure
~~~~~~~~~~~~~~~~~~

The registry stores command metadata such as:

- module name
- callable name
- human-readable description
- family
- aliases
- public command name

This information is consumed by the dispatcher in
``geoprior.cli.__main__`` and mirrored by the compatibility
launcher.

Dispatch model
--------------

The modern CLI resolves commands through the registry and
then imports the target module lazily.

This keeps the root entry point compact and makes each script
module independently testable.

A simplified execution path is:

.. code-block:: text

   user command
      ↓
   geoprior / geoprior-run / geoprior-build / geoprior-plot
      ↓
   geoprior.cli.__main__
      ↓
   registry lookup
      ↓
   import target module
      ↓
   call exported main function

Artifact-root policy
--------------------

The CLI also standardizes artifact-path resolution through
shared environment-variable and config helpers.

In practice this means that figures, tables, and helper
artifacts can be redirected consistently without forcing
script-specific path rewrites.

Typical environment variables include:

- ``GEOPRIOR_ARTIFACT_ROOT``
- ``GEOPRIOR_FIG_DIR``
- ``GEOPRIOR_OUT_DIR``

The conceptual layout is:

.. code-block:: text

   <artifact root>/
   ├── results/
   ├── fig/
   └── out/

This keeps compatibility with the historical folder layout
while allowing explicit relocation of generated outputs.

Examples:

.. code-block:: bash

   export GEOPRIOR_ARTIFACT_ROOT=/tmp/geoprior_artifacts
   export GEOPRIOR_FIG_DIR=/tmp/geoprior_figures
   export GEOPRIOR_OUT_DIR=/tmp/geoprior_tables

On Windows PowerShell:

.. code-block:: powershell

   $env:GEOPRIOR_ARTIFACT_ROOT = "D:\\geoprior\\artifacts"
   $env:GEOPRIOR_FIG_DIR = "D:\\geoprior\\figures"
   $env:GEOPRIOR_OUT_DIR = "D:\\geoprior\\tables"

Why ``geoprior.__main__`` matters
---------------------------------

Adding ``geoprior.__main__`` allows direct execution with:

.. code-block:: bash

   python -m geoprior --help

This complements installed console scripts and gives one more
stable execution path for development, testing, and package
verification.


API reference strategy
----------------------

Top-level entry points
----------------------

.. autosummary::
   :toctree: ../generated/api/cli
   :nosignatures:

   ~geoprior.__main__
   ~geoprior.cli
   ~geoprior.cli.__main__
   ~scripts.__main__
   ~scripts.registry

Dispatcher and shared CLI helpers
---------------------------------

.. autosummary::
   :toctree: ../generated/api/cli
   :nosignatures:

   ~geoprior.cli._dispatch
   ~geoprior.cli._presets
   ~geoprior.cli.config

Run wrappers and workflow drivers
---------------------------------

.. autosummary::
   :toctree: ../generated/api/cli
   :nosignatures:

   ~geoprior.cli.init_config
   ~geoprior.cli.stage1
   ~geoprior.cli.stage2
   ~geoprior.cli.stage3
   ~geoprior.cli.stage4
   ~geoprior.cli.stage5
   ~geoprior.cli.run_sensitivity
   ~geoprior.cli.run_sm3_suite
   ~geoprior.cli.sensitivity_lib
   ~geoprior.cli.sm3_collect_summaries
   ~geoprior.cli.sm3_log_offsets_diagnostics
   ~geoprior.cli.sm3_synthetic_identifiability

Build wrappers under ``geoprior.cli``
-------------------------------------

.. autosummary::
   :toctree: ../generated/api/cli
   :nosignatures:

   ~geoprior.cli.build_add_zsurf_from_coords
   ~geoprior.cli.build_assign_boreholes
   ~geoprior.cli.build_external_validation_fullcity
   ~geoprior.cli.build_external_validation_metrics
   ~geoprior.cli.build_full_inputs_npz
   ~geoprior.cli.build_physics_payload_npz
   ~geoprior.cli.build_sm3_collect_summaries

Private workflow backends excluded from autosummary
---------------------------------------------------

The modules below are intentionally not imported by the API
reference because they depend on workflow state such as
manifests or runtime environment setup.

- ``geoprior.cli._sensitivity``
- ``geoprior.cli._stage2``
- ``geoprior.cli._stage3``
- ``geoprior.cli._stage4``
- ``geoprior.cli._stage5``

Reproducibility registry and shared helpers
-------------------------------------------

.. autosummary::
   :toctree: ../generated/api/cli
   :nosignatures:

   ~geoprior._scripts.config
   ~geoprior._scripts.registry
   ~geoprior._scripts.utils
   ~geoprior._scripts.extend_utils

Build and compute scripts under ``geoprior._scripts``
-----------------------------------------------------

.. autosummary::
   :toctree: ../generated/api/cli
   :nosignatures:

   ~geoprior._scripts.build_ablation_table
   ~geoprior._scripts.build_model_metrics
   ~geoprior._scripts.compute_brier_exceedance
   ~geoprior._scripts.compute_hotspots
   ~geoprior._scripts.extend_forecast
   ~geoprior._scripts.make_boundary
   ~geoprior._scripts.make_district_grid
   ~geoprior._scripts.make_exposure
   ~geoprior._scripts.rebuild_confusion_tables
   ~geoprior._scripts.summarize_hotspots
   ~geoprior._scripts.tag_clusters_with_zones
   ~geoprior._scripts.update_ablation_record_posthoc
   ~geoprior._scripts.update_ablation_records

Plot scripts under ``geoprior._scripts``
----------------------------------------

.. autosummary::
   :toctree: ../generated/api/cli
   :nosignatures:

   ~geoprior._scripts.plot_ablations_sensitivity
   ~geoprior._scripts.plot_core_ablation
   ~geoprior._scripts.plot_driver_response
   ~geoprior._scripts.plot_external_validation
   ~geoprior._scripts.plot_geo_cumulative
   ~geoprior._scripts.plot_hotspot_analytics
   ~geoprior._scripts.plot_litho_parity
   ~geoprior._scripts.plot_physics_fields
   ~geoprior._scripts.plot_physics_maps
   ~geoprior._scripts.plot_physics_profiles
   ~geoprior._scripts.plot_physics_sanity
   ~geoprior._scripts.plot_physics_sensitivity
   ~geoprior._scripts.plot_sm3_bounds_ridge_summary
   ~geoprior._scripts.plot_sm3_identifiability
   ~geoprior._scripts.plot_sm3_log_offsets
   ~geoprior._scripts.plot_spatial_forecasts
   ~geoprior._scripts.plot_transfer
   ~geoprior._scripts.plot_uncertainty
   ~geoprior._scripts.plot_uncertainty_extras
   ~geoprior._scripts.plot_xfer_impact
   ~geoprior._scripts.plot_xfer_transferability

Registry ownership
------------------

The authoritative command registry lives under
``geoprior._scripts.registry``. The modern CLI and the
compatibility launcher both read from that registry rather
than maintaining separate command maps.

The dependency direction is therefore:

- ``geoprior.cli`` → ``geoprior._scripts``
- ``scripts`` → ``geoprior._scripts``

This keeps the compatibility layer thin and avoids making the
main package depend on legacy launch code.

Dispatch model
--------------

The modern CLI resolves a command through the registry,
imports the target module lazily, and calls the exported main
function.

.. code-block:: text

   user command
      ↓
   geoprior / geoprior-run / geoprior-build / geoprior-plot
      ↓
   geoprior.cli.__main__
      ↓
   registry lookup
      ↓
   import target module
      ↓
   call exported main function

Artifact-root policy
--------------------

The CLI standardizes artifact-path resolution through shared
configuration and environment-variable helpers. In practice,
this allows figures, tables, and helper artifacts to be
redirected without rewriting each script individually.

Typical environment variables include:

- ``GEOPRIOR_ARTIFACT_ROOT``
- ``GEOPRIOR_FIG_DIR``
- ``GEOPRIOR_OUT_DIR``

Conceptually:

.. code-block:: text

   <artifact root>/
   ├── results/
   ├── fig/
   └── out/

Examples:

.. code-block:: bash

   export GEOPRIOR_ARTIFACT_ROOT=/tmp/geoprior_artifacts
   export GEOPRIOR_FIG_DIR=/tmp/geoprior_figures
   export GEOPRIOR_OUT_DIR=/tmp/geoprior_tables

On Windows PowerShell:

.. code-block:: powershell

   $env:GEOPRIOR_ARTIFACT_ROOT = "D:\\geoprior\\artifacts"
   $env:GEOPRIOR_FIG_DIR = "D:\\geoprior\\figures"
   $env:GEOPRIOR_OUT_DIR = "D:\\geoprior\\tables"

Source listings
---------------

Source listings
---------------

The most important source files for the CLI stack are listed
below. These listings are useful when the API reference is
read together with the developer notes.

Package entry point
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/__main__.py
   :language: python
   :caption: ``geoprior/__main__.py``

Modern CLI package
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/cli/__init__.py
   :language: python
   :caption: ``geoprior/cli/__init__.py``

.. literalinclude:: ../../geoprior/cli/__main__.py
   :language: python
   :caption: ``geoprior/cli/__main__.py``

.. literalinclude:: ../../geoprior/cli/_dispatch.py
   :language: python
   :caption: ``geoprior/cli/_dispatch.py``

Script registry and shared config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/_scripts/registry.py
   :language: python
   :caption: ``geoprior/_scripts/registry.py``

.. literalinclude:: ../../geoprior/_scripts/config.py
   :language: python
   :caption: ``geoprior/_scripts/config.py``

Compatibility package
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../scripts/__init__.py
   :language: python
   :caption: ``scripts/__init__.py``

.. literalinclude:: ../../scripts/__main__.py
   :language: python
   :caption: ``scripts/__main__.py``

.. literalinclude:: ../../scripts/registry.py
   :language: python
   :caption: ``scripts/registry.py``

Design notes
------------

A few design choices are intentional.

Single source of truth
~~~~~~~~~~~~~~~~~~~~~~

The real registry now lives under ``geoprior._scripts``.
This avoids circular dependency pressure and makes the main
package self-contained.

Thin compatibility layer
~~~~~~~~~~~~~~~~~~~~~~~~

The top-level ``scripts`` package should stay small. Its role
is to preserve historical commands, not to own the actual
implementation.

Separated artifact policy
~~~~~~~~~~~~~~~~~~~~~~~~~

Artifact-path logic belongs in configuration rather than in
individual plotting or build scripts. That keeps export
behavior consistent and easier to override.

Shared dispatcher helpers
~~~~~~~~~~~~~~~~~~~~~~~~~

Both the modern CLI and the legacy launcher reuse the same
helpers from ``geoprior.cli._dispatch``. This keeps help
formatting, callable loading, alias handling, and module
execution consistent across entry points.

See also
--------

- :doc:`../user_guide/cli`
- :doc:`../developer/package_structure`
- :doc:`../applications/reproducibility_scripts`
