Subsidence API reference
========================

The ``geoprior.models.subsidence`` package is the scientific
core of GeoPrior-v3.

It contains:

- the flagship subsidence models,
- the physics and residual math helpers,
- scaling and unit-handling infrastructure,
- identifiability utilities,
- payload and diagnostics helpers,
- plotting and debugging support for scientific inspection,
- batch and derivative helpers used by training and
  evaluation, and
- the training-step support code used by the staged
  GeoPrior workflow.

This page is the main API entry point for the subsidence
stack. It is intentionally written as a map of the package,
not merely as a compact symbol dump, so the explanatory text
stays visible alongside the generated API sections.

Overview
--------

At a high level, the subsidence package is organized around
these layers:

+-----------------------------+------------------------------------------+
| Layer                       | Purpose                                  |
+=============================+==========================================+
| ``models``                  | Public model classes                     |
+-----------------------------+------------------------------------------+
| ``maths``                   | Physics field composition and closures   |
+-----------------------------+------------------------------------------+
| ``scaling``                 | Scaling contract and serialization       |
+-----------------------------+------------------------------------------+
| ``losses``                  | Packed step results and loss helpers     |
+-----------------------------+------------------------------------------+
| ``identifiability``         | Regimes, locks, and audit helpers        |
+-----------------------------+------------------------------------------+
| ``payloads``                | Physics payload gather/save/load helpers |
+-----------------------------+------------------------------------------+
| ``utils``                   | SI conversion and physics utilities      |
+-----------------------------+------------------------------------------+
| ``stability``               | Gradient filtering and stability helpers |
+-----------------------------+------------------------------------------+
| ``step_core``               | Shared physics-core execution path       |
+-----------------------------+------------------------------------------+
| ``derivatives``             | Differentiation and residual helpers     |
+-----------------------------+------------------------------------------+
| ``batch_io``                | Batch extraction and transport helpers   |
+-----------------------------+------------------------------------------+
| ``debugs``                  | Debug-oriented scientific checks         |
+-----------------------------+------------------------------------------+
| ``plot``                    | Subsidence-specific plotting helpers     |
+-----------------------------+------------------------------------------+
| ``log_offsets_diagnostics`` | Offset-field diagnostics                 |
+-----------------------------+------------------------------------------+
| ``doc``                     | Package-level scientific text helpers    |
+-----------------------------+------------------------------------------+

A useful mental model is:

.. code-block:: text

   subsidence package
      ├── public models
      ├── scientific math helpers
      ├── scaling + conventions
      ├── residual / loss support
      ├── identifiability controls
      ├── payload / export helpers
      ├── diagnostics / plots / debug helpers
      └── shared physics-core execution

Public package surface
----------------------

The package namespace gathers the main public surface for the
subsidence stack. The generated API block below is kept, but
this page also breaks the package apart module by module so
that the purpose of each layer remains clear.

.. automodule:: geoprior.models.subsidence
   :members:
   :undoc-members:
   :show-inheritance:

Module index
------------

The index below uses fully qualified import paths
throughout. It deliberately documents the submodules
explicitly and avoids package-relative lookup.

The package itself is documented in the ``automodule`` block
above, so it is **not** repeated in the autosummary tables
below. Keeping the package out of the autosummary index makes
stub generation more predictable and avoids self-referential
package entries.

Core scientific modules
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/subsidence/core
   :nosignatures:

   geoprior.models.subsidence.models
   geoprior.models.subsidence.maths
   geoprior.models.subsidence.scaling
   geoprior.models.subsidence.losses
   geoprior.models.subsidence.identifiability
   geoprior.models.subsidence.payloads
   geoprior.models.subsidence.utils
   geoprior.models.subsidence.stability
   geoprior.models.subsidence.step_core

Supporting scientific and diagnostics modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/subsidence/support
   :nosignatures:

   geoprior.models.subsidence.batch_io
   geoprior.models.subsidence.debugs
   geoprior.models.subsidence.derivatives
   geoprior.models.subsidence.doc
   geoprior.models.subsidence.log_offsets_diagnostics
   geoprior.models.subsidence.plot

Core model classes
------------------

The most important public exports in this package are the two
model classes:

- :class:`~geoprior.models.subsidence.models.GeoPriorSubsNet`
- :class:`~geoprior.models.subsidence.models.PoroElasticSubsNet`

These are the main entry points for physics-guided
subsidence forecasting in GeoPrior-v3.

.. automodule:: geoprior.models.subsidence.models
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Key model classes
~~~~~~~~~~~~~~~~~

.. autoclass:: geoprior.models.subsidence.models.GeoPriorSubsNet
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: geoprior.models.subsidence.models.PoroElasticSubsNet
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Scientific math helpers
-----------------------

The ``maths`` module contains the low-level mathematical
helpers used to:

- compose effective physical fields,
- derive closure timescales,
- compute equilibrium compaction,
- map forcing terms into groundwater-source form, and
- compute soft-bounds residuals.

This layer is the most compact expression of the physical
assumptions used by the model family, so it is often the best
place to start when you want to understand how the learned
fields are turned into physically meaningful quantities.

.. automodule:: geoprior.models.subsidence.maths
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Important helper functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.maths.compose_physics_fields
   :no-index:

.. autofunction:: geoprior.models.subsidence.maths.tau_phys_from_fields
   :no-index:

.. autofunction:: geoprior.models.subsidence.maths.equilibrium_compaction_si
   :no-index:

.. autofunction:: geoprior.models.subsidence.maths.compute_mv_prior
   :no-index:

.. autofunction:: geoprior.models.subsidence.maths.q_to_gw_source_term_si
   :no-index:

.. autofunction:: geoprior.models.subsidence.maths.compute_bounds_residual
   :no-index:

Scaling and serialization
-------------------------

The scaling layer defines the contract that tells GeoPrior
what the data mean physically. It is one of the most
important pieces of the whole subsidence stack, because it is
not merely a preprocessing detail: it records the
interpretation needed to connect model-space quantities to SI
units, coordinates, bounds, and downstream diagnostics.

.. automodule:: geoprior.models.subsidence.scaling
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Key scaling object
~~~~~~~~~~~~~~~~~~

.. autoclass:: geoprior.models.subsidence.scaling.GeoPriorScalingConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Loss and step-packing helpers
-----------------------------

The ``losses`` module contains helpers used by the custom
training and evaluation steps to pack structured outputs,
including physics losses and diagnostics. In practice, this
layer is where the public model outputs are reorganized into
the richer objects consumed by the staged workflow.

.. automodule:: geoprior.models.subsidence.losses
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Important helpers
~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.losses.pack_step_results
   :no-index:

Identifiability controls
------------------------

The identifiability layer exposes regime setup, compile-time
weight resolution, head locks, and audit helpers. This is an
important part of the scientific design of GeoPrior-v3,
because it controls how strongly different fields are tied to
their priors and which degrees of freedom remain open during
training.

.. automodule:: geoprior.models.subsidence.identifiability
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Important helpers
~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.identifiability.init_identifiability
   :no-index:

.. autofunction:: geoprior.models.subsidence.identifiability.apply_ident_locks
   :no-index:

.. autofunction:: geoprior.models.subsidence.identifiability.resolve_compile_weights
   :no-index:

Payloads and exported physics artifacts
---------------------------------------

The payload layer provides helpers for gathering, saving,
loading, and subsampling physics payloads used later in
diagnostics, inference, and figure generation. These helpers
matter for reproducibility because they make it possible to
inspect the internal physical state of a run after training.

.. automodule:: geoprior.models.subsidence.payloads
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Important helpers
~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.payloads.gather_physics_payload
   :no-index:

.. autofunction:: geoprior.models.subsidence.payloads.save_physics_payload
   :no-index:

.. autofunction:: geoprior.models.subsidence.payloads.load_physics_payload
   :no-index:

Stability helpers
-----------------

The stability layer contains helper utilities used to keep
training and evaluation robust in the presence of stiff
physics branches, unstable residual terms, or bad gradients.

.. automodule:: geoprior.models.subsidence.stability
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Important helpers
~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.stability.filter_nan_gradients
   :no-index:

Utility helpers
---------------

The ``utils`` module contains conversion and convenience
helpers used across the subsidence stack, including SI
mapping, groundwater/head conversion, initialization helpers,
and policy gating.

.. automodule:: geoprior.models.subsidence.utils
   :undoc-members:
   :show-inheritance:
   :no-index:

Selected public helpers
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.utils.to_si_head
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.to_si_thickness
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.from_si_subsidence
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.gwl_to_head_m
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.get_h_ref_si
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.get_s_init_si
   :no-index:

Shared physics core
-------------------

The ``step_core`` module contains the shared physics-core
execution path that assembles the structured physics bundle
used in training and evaluation. It is especially useful when
you want to trace how model outputs, scaling metadata,
physics residuals, and diagnostics are joined together in one
place.

.. automodule:: geoprior.models.subsidence.step_core
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Important helper
~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.step_core.physics_core
   :no-index:

Derivative and residual helpers
-------------------------------

The ``derivatives`` module groups the derivative-oriented
helpers used by the physics stack. These routines are useful
when you want to understand how gradients, rates, or spatial
and temporal derivative terms are assembled before they are
fed into the residual machinery.

.. automodule:: geoprior.models.subsidence.derivatives
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Batch and transport helpers
---------------------------

The ``batch_io`` module provides helpers for moving
structured batches into the forms expected by the subsidence
models and their staged workflow. It is especially useful
when tracing how batch dictionaries or arrays are normalized
before they enter the shared physics core.

.. automodule:: geoprior.models.subsidence.batch_io
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Debugging and scientific inspection
-----------------------------------

The ``debugs`` module contains helpers intended for
inspection-oriented workflows. These routines are useful when
you need to verify scientific assumptions, track internal
state, or make sense of unexpected training or evaluation
behavior.

.. automodule:: geoprior.models.subsidence.debugs
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Log-offset diagnostics
----------------------

The ``log_offsets_diagnostics`` module focuses on diagnostics
for inferred log-offset fields and related quantities. It is
particularly relevant when you are studying prior anchoring,
offset magnitudes, or identifiability behavior.

.. automodule:: geoprior.models.subsidence.log_offsets_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Plotting helpers
----------------

The ``plot`` module contains subsidence-specific plotting
utilities that help inspect scientific outputs, internal
fields, and derived diagnostics from this package.

.. automodule:: geoprior.models.subsidence.plot
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Package-level scientific text helpers
-------------------------------------

The ``doc`` module provides text-oriented helpers or package
documentation utilities that support the scientific
explanation layer around the subsidence stack.

.. automodule:: geoprior.models.subsidence.doc
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Suggested reading order
-----------------------

If you are new to the codebase, a good order is:

1. :class:`~geoprior.models.subsidence.models.GeoPriorSubsNet`
2. :class:`~geoprior.models.subsidence.scaling.GeoPriorScalingConfig`
3. :func:`~geoprior.models.subsidence.maths.compose_physics_fields`
4. :func:`~geoprior.models.subsidence.step_core.physics_core`
5. :func:`~geoprior.models.subsidence.losses.pack_step_results`

This gives the clearest path from the public model surface to
the shared physics core. After that, ``identifiability``,
``payloads``, and ``log_offsets_diagnostics`` are usually the
most useful follow-up modules for understanding the
scientific behavior of a trained run.

Source listings
---------------

These source listings are useful when you want to see the
implementation structure directly.

Main model module
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../geoprior/models/subsidence/models.py
   :language: python
   :caption: geoprior/models/subsidence/models.py

Scientific math helpers
~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../geoprior/models/subsidence/maths.py
   :language: python
   :caption: geoprior/models/subsidence/maths.py
   :lines: 1-260

Utility helpers
~~~~~~~~~~~~~~~

.. literalinclude:: ../../../geoprior/models/subsidence/utils.py
   :language: python
   :caption: geoprior/models/subsidence/utils.py
   :lines: 1-260

Step-core helpers
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../geoprior/models/subsidence/step_core.py
   :language: python
   :caption: geoprior/models/subsidence/step_core.py
   :lines: 1-260

API notes
---------

A few practical notes are worth keeping in mind when reading
this API:

- ``GeoPriorSubsNet`` is the main coupled model.
- ``PoroElasticSubsNet`` is a stronger
  consolidation-first preset built on the same broad model
  family.
- The scaling layer is part of the scientific contract, not
  just preprocessing support.
- The physics-core path is intentionally shared between
  training and evaluation to keep diagnostics consistent.
- The payload helpers are important for downstream scripts,
  diagnostics, and reproducibility workflows.
- The diagnostics-oriented modules are worth reading even if
  you only use the public model classes, because they expose
  much of the scientific audit surface used by the staged
  GeoPrior workflow.

See also
--------

.. seealso::

   - :doc:`../scientific_foundations/models_overview`
   - :doc:`../scientific_foundations/geoprior_subsnet`
   - :doc:`../scientific_foundations/physics_formulation`
   - :doc:`../scientific_foundations/residual_assembly`
   - :doc:`../scientific_foundations/losses_and_training`
   - :doc:`../scientific_foundations/scaling`
   - :doc:`../scientific_foundations/identifiability`
   - :doc:`tuner`
   - :doc:`utils`
