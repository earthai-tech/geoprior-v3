Tuner API reference
===================

The GeoPrior-v3 tuning stack provides a structured
hyperparameter-search interface for the flagship
subsidence-forecasting models.

This API layer is centered on:

- a **public tuner entry point**
  :class:`~geoprior.models.forecast_tuner.SubsNetTuner`,
- a **shared base tuner**
  :class:`~geoprior.models.forecast_tuner._base_tuner.PINNTunerBase`,
- and the GeoPrior-specific specialization implemented in the
  forecast-tuner package.

This page documents the tuning API at both levels:

- the **public tuning surface** used by application workflows;
- the **supporting base classes and source modules** that
  explain how the tuning system is structured internally.

Overview
--------

The forecast-tuner package is organized around the following
pieces:

.. code-block:: text

   geoprior.models.forecast_tuner
      ├── __init__.py
      ├── tuners.py
      ├── _base_tuner.py
      └── _geoprior_tuner.py

A useful mental model is:

- ``SubsNetTuner`` is the main public tuner you instantiate;
- ``PINNTunerBase`` provides the reusable keras-tuner
  integration and search loop;
- ``_geoprior_tuner.py`` implements the GeoPrior-specific
  tuning logic for ``GeoPriorSubsNet``.

Public package surface
----------------------

.. automodule:: geoprior.models.forecast_tuner
   :members:
   :undoc-members:
   :show-inheritance:

Module index
------------

.. autosummary::
   :toctree: generated/


   geoprior.models.forecast_tuner
   geoprior.models.forecast_tuner.tuners
   geoprior.models.forecast_tuner._base_tuner
   geoprior.models.forecast_tuner._geoprior_tuner

Public tuner entry point
------------------------

The main public entry point is
:class:`~geoprior.models.forecast_tuner.SubsNetTuner`.

This is the tuner documented in the application pages as the
recommended hyperparameter-search interface for
:class:`~geoprior.models.GeoPriorSubsNet`.

.. automodule:: geoprior.models.forecast_tuner.tuners
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geoprior.models.forecast_tuner._geoprior_tuner
   :members:
   :undoc-members:
   :show-inheritance:

Key public class
~~~~~~~~~~~~~~~~

.. autoclass:: geoprior.models.forecast_tuner._geoprior_tuner.SubsNetTuner
   :members:
   :undoc-members:
   :show-inheritance:

Important entry methods
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: geoprior.models.forecast_tuner._geoprior_tuner.SubsNetTuner.create

.. automethod:: geoprior.models.forecast_tuner._geoprior_tuner.SubsNetTuner.run

.. automethod:: geoprior.models.forecast_tuner._geoprior_tuner.SubsNetTuner.build

What ``SubsNetTuner`` is responsible for
----------------------------------------

The GeoPrior-specific tuner is responsible for:

- binding the search workflow to
  :class:`~geoprior.models.GeoPriorSubsNet`;
- inferring fixed input and output dimensions from arrays;
- canonicalizing GeoPrior input keys and target names;
- separating **model-init hyperparameters** from
  **compile-only hyperparameters**;
- building datasets and delegating the actual search to the
  base tuner logic.

This makes it more than a thin keras-tuner wrapper. It is the
main application-facing tuning layer for GeoPrior
subsidence models.

Shared base tuner
-----------------

The tuning package also provides a reusable base class:

:class:`~geoprior.models.forecast_tuner._base_tuner.PINNTunerBase`

This class handles the generic search orchestration,
including:

- keras-tuner backend selection,
- objective normalization,
- project/directory layout,
- trial execution,
- callback handling,
- best-model recovery,
- tuning summary persistence.

.. automodule:: geoprior.models.forecast_tuner._base_tuner
   :members:
   :undoc-members:
   :show-inheritance:

Key base class
~~~~~~~~~~~~~~

.. autoclass:: geoprior.models.forecast_tuner._base_tuner.PINNTunerBase
   :members:
   :undoc-members:
   :show-inheritance:

Important base methods
~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: geoprior.models.forecast_tuner._base_tuner.PINNTunerBase.search

.. automethod:: geoprior.models.forecast_tuner._base_tuner.PINNTunerBase.build

.. automethod:: geoprior.models.forecast_tuner._base_tuner.PINNTunerBase.summary

How the tuning stack is layered
-------------------------------

A useful design map is:

.. code-block:: text

   PINNTunerBase
      ├── generic keras-tuner search loop
      ├── backend selection
      ├── project persistence
      └── best-trial recovery

   SubsNetTuner
      ├── binds GeoPriorSubsNet
      ├── infers data dimensions
      ├── checks GeoPrior input keys
      ├── canonicalizes targets
      ├── separates init vs compile HPs
      └── builds the tuned GeoPrior model

This layering is one of the strengths of the current tuner
design.

GeoPrior-specific tuning logic
------------------------------

The ``SubsNetTuner`` specialization encodes several
GeoPrior-specific assumptions.

These include:

- required inputs such as
  ``coords``, ``dynamic_features``, and ``H_field``;
- canonical target names
  ``subs_pred`` and ``gwl_pred``;
- GeoPrior architecture defaults;
- physics-aware compile defaults;
- and search-space routing for compile-only keys such as
  learning rate or physics lambdas.

This is what makes the tuner scientifically aligned with the
model, rather than being a generic high-level tuning wrapper.

Compile-only hyperparameters
----------------------------

One especially important API concept in the tuner stack is
the distinction between:

- **model-init hyperparameters**,
- and **compile-only hyperparameters**.

The GeoPrior-specific tuner explicitly treats keys such as:

- ``learning_rate``
- ``lambda_gw``
- ``lambda_cons``
- ``lambda_prior``
- ``lambda_smooth``
- ``lambda_mv``
- ``lambda_bounds``
- ``lambda_q``
- ``lambda_offset``
- ``scale_mv_with_offset``
- ``scale_q_with_offset``
- ``mv_lr_mult``
- ``kappa_lr_mult``

as compile-only search keys.

That separation is important because it keeps the tuned model
definition conceptually distinct from the tuned optimization
and physics-loss balance.

Supported tuner backends
------------------------

The base tuner currently supports several keras-tuner
backends, including:

- random search,
- Bayesian optimization,
- Hyperband.

These are selected through the ``tuner_type`` argument and
normalized inside the base class.

Typical usage therefore looks like:

.. code-block:: python

   tuner = SubsNetTuner.create(
       inputs_data=inputs_np,
       targets_data=targets_np,
       search_space=space,
       tuner_type="randomsearch",
       max_trials=20,
   )

Practical read order
--------------------

If you are new to the tuning code, a good order is:

1. :class:`~geoprior.models.forecast_tuner.SubsNetTuner`
2. :meth:`~geoprior.models.forecast_tuner._geoprior_tuner.SubsNetTuner.create`
3. :meth:`~geoprior.models.forecast_tuner._base_tuner.PINNTunerBase.search`
4. :meth:`~geoprior.models.forecast_tuner._geoprior_tuner.SubsNetTuner.build`

This gives the clearest path from public application API to
the underlying search engine.

Source listings with comments
-----------------------------

These source listings are included intentionally so the
inline implementation comments remain visible.

Public tuner shim
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/models/forecast_tuner/tuners.py
   :language: python
   :caption: geoprior/models/forecast_tuner/tuners.py

Shared base tuner
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/models/forecast_tuner/_base_tuner.py
   :language: python
   :caption: geoprior/models/forecast_tuner/_base_tuner.py

GeoPrior-specific tuner
~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/models/forecast_tuner/_geoprior_tuner.py
   :language: python
   :caption: geoprior/models/forecast_tuner/_geoprior_tuner.py

API notes
---------

A few design details are worth keeping in mind when reading
this API:

- ``SubsNetTuner.create(...)`` is the recommended user entry
  point because it infers dimensions from data arrays and
  applies GeoPrior-aware canonicalization.
- ``PINNTunerBase`` is abstract in spirit: subclasses are
  expected to implement the actual ``build(hp)`` method.
- The tuning stack is intentionally aware of the GeoPrior
  physics contract, not only the neural architecture.
- Target canonicalization and required-key checking are part
  of the tuning API because tuning must still respect the
  same scientific contract as ordinary training.

See also
--------

.. seealso::

   - :doc:`../applications/tuner_workflow`
   - :doc:`../user_guide/stage3`
   - :doc:`subsidence`
   - :doc:`utils`
   - :doc:`resources`
   - :doc:`../scientific_foundations/geoprior_subsnet`
   - :doc:`../scientific_foundations/losses_and_training`