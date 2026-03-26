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
- and the training-step support code used by the staged
  GeoPrior workflow.

This page is the main API entry point for the subsidence
stack.

Overview
--------

At a high level, the subsidence package is organized around
these layers:

+----------------------+-------------------------------------------+
| Layer                | Purpose                                   |
+======================+===========================================+
| ``models``           | Public model classes                      |
+----------------------+-------------------------------------------+
| ``maths``            | Physics field composition and closures    |
+----------------------+-------------------------------------------+
| ``scaling``          | Scaling contract and serialization        |
+----------------------+-------------------------------------------+
| ``losses``           | Packed step results and loss helpers      |
+----------------------+-------------------------------------------+
| ``identifiability``  | Regimes, locks, and audit helpers         |
+----------------------+-------------------------------------------+
| ``payloads``         | Physics payload gather/save/load helpers  |
+----------------------+-------------------------------------------+
| ``utils``            | SI conversion and physics utility helpers |
+----------------------+-------------------------------------------+
| ``stability``        | Gradient filtering and stability helpers  |
+----------------------+-------------------------------------------+
| ``step_core``        | Shared physics-core execution path        |
+----------------------+-------------------------------------------+

A useful mental model is:

.. code-block:: text

   subsidence package
      ├── public models
      ├── scientific math helpers
      ├── scaling + conventions
      ├── residual/loss support
      ├── identifiability controls
      └── payload/export helpers

Public package surface
----------------------

.. automodule:: geoprior.models.subsidence
   :members:
   :undoc-members:
   :show-inheritance:

Module index
------------

.. autosummary::
   :toctree: generated/
   :recursive:

   geoprior.models.subsidence
   geoprior.models.subsidence.models
   geoprior.models.subsidence.maths
   geoprior.models.subsidence.scaling
   geoprior.models.subsidence.losses
   geoprior.models.subsidence.identifiability
   geoprior.models.subsidence.payloads
   geoprior.models.subsidence.utils
   geoprior.models.subsidence.stability
   geoprior.models.subsidence.step_core

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

Key model classes
~~~~~~~~~~~~~~~~~

.. autoclass:: geoprior.models.subsidence.models.GeoPriorSubsNet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: geoprior.models.subsidence.models.PoroElasticSubsNet
   :members:
   :undoc-members:
   :show-inheritance:

Scientific math helpers
-----------------------

The ``maths`` module contains the low-level mathematical
helpers used to:

- compose effective physical fields,
- derive closure timescales,
- compute equilibrium compaction,
- map forcing terms into groundwater-source form,
- and compute soft bounds residuals.

.. automodule:: geoprior.models.subsidence.maths
   :members:
   :undoc-members:
   :show-inheritance:

Important helper functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.maths.compose_physics_fields

.. autofunction:: geoprior.models.subsidence.maths.tau_phys_from_fields

.. autofunction:: geoprior.models.subsidence.maths.equilibrium_compaction_si

.. autofunction:: geoprior.models.subsidence.maths.compute_mv_prior

.. autofunction:: geoprior.models.subsidence.maths.q_to_gw_source_term_si

.. autofunction:: geoprior.models.subsidence.maths.compute_bounds_residual

Scaling and serialization
-------------------------

The scaling layer defines the contract that tells GeoPrior
what the data mean physically. It is one of the most
important pieces of the whole subsidence stack.

.. automodule:: geoprior.models.subsidence.scaling
   :members:
   :undoc-members:
   :show-inheritance:

Key scaling object
~~~~~~~~~~~~~~~~~~

.. autoclass:: geoprior.models.subsidence.scaling.GeoPriorScalingConfig
   :members:
   :undoc-members:
   :show-inheritance:

Loss and step-packing helpers
-----------------------------

The ``losses`` module contains helpers used by the custom
training and evaluation steps to pack structured outputs,
including physics losses and diagnostics.

.. automodule:: geoprior.models.subsidence.losses
   :members:
   :undoc-members:
   :show-inheritance:

Important helpers
~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.losses.pack_step_results

Identifiability controls
------------------------

The identifiability layer exposes regime setup, compile-time
weight resolution, head locks, and audit helpers. This is an
important part of the scientific design of GeoPrior-v3.

.. automodule:: geoprior.models.subsidence.identifiability
   :members:
   :undoc-members:
   :show-inheritance:

Important helpers
~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.identifiability.init_identifiability

.. autofunction:: geoprior.models.subsidence.identifiability.apply_ident_locks

.. autofunction:: geoprior.models.subsidence.identifiability.resolve_compile_weights

Payloads and exported physics artifacts
---------------------------------------

The payload layer provides helpers for gathering, saving,
loading, and subsampling physics payloads used later in
diagnostics, inference, and figure generation.

.. automodule:: geoprior.models.subsidence.payloads
   :members:
   :undoc-members:
   :show-inheritance:

Important helpers
~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.payloads.gather_physics_payload

.. autofunction:: geoprior.models.subsidence.payloads.save_physics_payload

.. autofunction:: geoprior.models.subsidence.payloads.load_physics_payload

Stability helpers
-----------------

The stability layer contains helper utilities used to keep
training and evaluation robust in the presence of stiff
physics branches or bad gradients.

.. automodule:: geoprior.models.subsidence.stability
   :members:
   :undoc-members:
   :show-inheritance:

Important helpers
~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.stability.filter_nan_gradients

Utility helpers
---------------

The ``utils`` module contains conversion and convenience
helpers used across the subsidence stack, including SI
mapping, groundwater/head conversion, initialization helpers,
and policy gating.

.. automodule:: geoprior.models.subsidence.utils
   :members:
   :undoc-members:
   :show-inheritance:

Selected public helpers
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.utils.to_si_head

.. autofunction:: geoprior.models.subsidence.utils.to_si_thickness

.. autofunction:: geoprior.models.subsidence.utils.from_si_subsidence

.. autofunction:: geoprior.models.subsidence.utils.gwl_to_head_m

.. autofunction:: geoprior.models.subsidence.utils.get_h_ref_si

.. autofunction:: geoprior.models.subsidence.utils.get_s_init_si

Shared physics core
-------------------

The ``step_core`` module contains the shared physics-core
execution path that assembles the structured physics bundle
used in training and evaluation.

.. automodule:: geoprior.models.subsidence.step_core
   :members:
   :undoc-members:
   :show-inheritance:

Important helper
~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.models.subsidence.step_core.physics_core

Suggested reading order
-----------------------

If you are new to the codebase, a good order is:

1. :class:`~geoprior.models.subsidence.models.GeoPriorSubsNet`
2. :class:`~geoprior.models.subsidence.scaling.GeoPriorScalingConfig`
3. :func:`~geoprior.models.subsidence.maths.compose_physics_fields`
4. :func:`~geoprior.models.subsidence.step_core.physics_core`
5. :func:`~geoprior.models.subsidence.losses.pack_step_results`

This gives the clearest path from the public model surface to
the shared physics core.

Source listings
---------------

These source listings are useful when you want to see the
implementation structure directly.

Main model module
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/models/subsidence/models.py
   :language: python
   :caption: geoprior/models/subsidence/models.py

Scientific math helpers
~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/models/subsidence/maths.py
   :language: python
   :caption: geoprior/models/subsidence/maths.py
   :lines: 1-260

Utility helpers
~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/models/subsidence/utils.py
   :language: python
   :caption: geoprior/models/subsidence/utils.py
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
   - :doc:`resources`