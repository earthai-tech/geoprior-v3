Tuner API reference
===================

The GeoPrior-v3 tuning stack provides the hyperparameter-search
interface used to adapt ``GeoPriorSubsNet`` to a concrete training
setup. It is not just a thin wrapper around keras-tuner. The tuning
layer separates architecture hyperparameters from compile-only
hyperparameters, validates the scientific input contract expected by the
subsidence model, and persists search artifacts in a predictable project
layout.

This page documents the tuner stack at two complementary levels:

- the **public package surface** that application workflows use;
- the **implementation modules** that explain how the tuning system is
  organized internally.

The page also uses **fully qualified module names** throughout so that
Sphinx autosummary resolves modules directly, without relying on package
attribute lookup.

Overview
--------

The tuning code is split across a small public package and a lightweight
compatibility shim:

.. code-block:: text

   geoprior.models
   ├── tuners.py                       # public compatibility shim
   └── forecast_tuner/
       ├── __init__.py                 # public package surface
       ├── _base_tuner.py             # reusable tuner base class
       └── _geoprior_tuner.py         # GeoPrior-specific specialization

A useful mental model is the following:

- :class:`~geoprior.models.forecast_tuner.SubsNetTuner` is the main
  public tuner class exposed by the forecast-tuner package;
- :class:`~geoprior.models.forecast_tuner._base_tuner.PINNTunerBase`
  provides the reusable keras-tuner integration and generic search
  orchestration;
- :mod:`geoprior.models.tuners` is a compact compatibility shim that
  re-exports :class:`~geoprior.models.forecast_tuner.SubsNetTuner` from
  the public models namespace;
- :mod:`geoprior.models.forecast_tuner._geoprior_tuner` implements the
  GeoPrior-specific tuning logic that binds the search flow to
  :class:`~geoprior.models.GeoPriorSubsNet`.

This structure is reflected in the code you uploaded: the
``forecast_tuner`` package exports both ``SubsNetTuner`` and
``PINNTunerBase`` from its ``__init__.py``, while the top-level
``geoprior.models.tuners`` module simply re-exports
``SubsNetTuner`` as a compatibility surface. 

Module index
------------

The targets below are intentionally written as **fully qualified module
names**. This avoids the package-relative lookup behavior that often
causes autosummary to build malformed paths.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~geoprior.models.forecast_tuner
   ~geoprior.models.tuners
   ~geoprior.models.forecast_tuner._base_tuner
   ~geoprior.models.forecast_tuner._geoprior_tuner

Public package surface
----------------------

The main public package is :mod:`geoprior.models.forecast_tuner`. Its
``__init__.py`` exposes the tuning-availability flags together with the
public tuner classes, including ``SubsNetTuner`` and
``PINNTunerBase``.

.. automodule:: geoprior.models.forecast_tuner
   :members:
   :undoc-members:
   :show-inheritance:

Compatibility shim
------------------

The top-level module :mod:`geoprior.models.tuners` exists as a compact
compatibility layer. Rather than implementing the tuning logic itself,
it simply re-exports :class:`~geoprior.models.forecast_tuner.SubsNetTuner`
from the public models namespace. This is useful when existing code or
older imports expect a shorter path under ``geoprior.models``. 

.. automodule:: geoprior.models.tuners
   :members:
   :undoc-members:
   :show-inheritance:

Public tuner entry point
------------------------

The primary user-facing entry point is
:class:`~geoprior.models.forecast_tuner.SubsNetTuner`.

This class specializes the generic tuner workflow for
:class:`~geoprior.models.GeoPriorSubsNet`. Its class docstring explains
that it is designed for physics-informed hyperparameter search, that it
separates data-dependent fixed parameters from the search space, and
that ``create(...)`` is the recommended constructor when arrays are
already available in memory. 

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

Important public methods
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: geoprior.models.forecast_tuner._geoprior_tuner.SubsNetTuner.create

.. automethod:: geoprior.models.forecast_tuner._geoprior_tuner.SubsNetTuner.run

.. automethod:: geoprior.models.forecast_tuner._geoprior_tuner.SubsNetTuner.build

What ``SubsNetTuner`` is responsible for
----------------------------------------

The GeoPrior-specific tuner is responsible for the application-facing
part of the search pipeline. In particular, it:

- binds the tuning workflow to
  :class:`~geoprior.models.GeoPriorSubsNet`;
- infers and merges fixed data-shape parameters from array inputs when
  ``create(...)`` is used;
- canonicalizes target names such as ``subsidence`` and ``gwl`` to the
  model-facing names ``subs_pred`` and ``gwl_pred``;
- separates **model-init hyperparameters** from
  **compile-only hyperparameters**;
- constructs the per-trial model and compiles it with losses, metrics,
  and physics weights;
- validates GeoPrior input requirements before delegating the actual
  search loop to the shared base class.

These responsibilities are visible in the implementation itself. The
module defines explicit compile-only hyperparameter keys such as
``learning_rate``, ``lambda_gw``, ``lambda_cons``, ``lambda_prior``,
``lambda_smooth``, ``lambda_mv``, ``lambda_bounds``, ``lambda_q``, and
``lambda_offset``; it also binds ``GeoPriorSubsNet`` as the target model
class and provides a ``create(...)`` constructor that canonicalizes
training targets before inferring fixed parameters. 

This is why the tuner should be read as a model-aware scientific
component rather than as a generic high-level helper.

Shared base tuner
-----------------

The tuning stack also includes a reusable generic base class:
:class:`~geoprior.models.forecast_tuner._base_tuner.PINNTunerBase`.

This base class handles the broader search orchestration shared by PINN
or physics-aware tuning workflows. Its implementation includes:

- objective normalization;
- tuner backend validation;
- keras-tuner backend selection;
- project and directory management;
- repeated executions per trial;
- callback plumbing;
- validation-data handling;
- best-hyperparameter and best-model recovery.

The uploaded source shows that ``PINNTunerBase`` subclasses both
``HyperModel`` and ``BaseClass`` and centralizes the keras-tuner classes
through the ``KT_DEPS`` dependency container. It also normalizes tuner
backend names such as ``randomsearch``, ``bayesianoptimization``, and
``hyperband`` inside its validation logic. 

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

.. automethod:: geoprior.models.forecast_tuner._base_tuner.PINNTunerBase

How the tuning stack is layered
-------------------------------

A useful design map is:

.. code-block:: text

   PINNTunerBase
      ├── objective normalization
      ├── keras-tuner backend selection
      ├── trial orchestration
      ├── callback / validation wiring
      └── best-model recovery

   SubsNetTuner
      ├── binds GeoPriorSubsNet
      ├── infers data dimensions
      ├── checks required GeoPrior inputs
      ├── canonicalizes targets
      ├── separates init vs compile HPs
      └── builds the tuned GeoPrior model

This layering is one of the strengths of the current design. The base
class remains reusable and backend-oriented, while the specialized class
keeps the scientific contract of the subsidence model close to the
search logic.

GeoPrior-specific tuning logic
------------------------------

The specialization implemented in
:mod:`geoprior.models.forecast_tuner._geoprior_tuner` encodes several
assumptions that are specific to GeoPrior subsidence forecasting.

These include:

- required model inputs such as ``coords``, ``dynamic_features``, and
  ``H_field``;
- canonical target names ``subs_pred`` and ``gwl_pred``;
- GeoPrior architecture defaults and default physics settings;
- compile-time routing for loss weights and physics penalties;
- direct coupling to the ``GeoPriorSubsNet`` constructor and compile
  stage.

The class docstring also explains that the tuner is aware of the model's
physics objectives and loss balance, and not just its neural-network
shape. It explicitly describes the residual structure used during
training, including groundwater and consolidation residuals.

Compile-only hyperparameters
----------------------------

One especially important API concept in the tuner stack is the
separation between:

- **model-init hyperparameters**, which alter the instantiated model;
- **compile-only hyperparameters**, which alter the optimizer and loss
  weighting behavior without changing the instantiated architecture.

In the current implementation, keys such as the following are treated as
compile-only search parameters:

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

This split matters because it keeps the model definition conceptually
separate from the optimization regime and from the weighting of physics
terms. The implementation lists these keys explicitly in the
``_COMPILE_ONLY`` collection, which makes the behavior easy to trace in
the source. 

Supported tuner backends
------------------------

The shared base class currently normalizes and supports several
keras-tuner backends, including:

- random search,
- Bayesian optimization,
- Hyperband.

These are selected through the ``tuner_type`` argument and normalized by
``PINNTunerBase`` before the actual tuner object is instantiated.

Typical usage therefore looks like:

.. code-block:: python

   from geoprior.models.forecast_tuner import SubsNetTuner

   tuner = SubsNetTuner.create(
       inputs_data=inputs_np,
       targets_data=targets_np,
       search_space=space,
       tuner_type="randomsearch",
       max_trials=20,
   )

The example above matches the intended usage pattern described in the
``SubsNetTuner`` class documentation. 

Practical read order
--------------------

If you are new to the tuning code, a good order is:

1. :class:`~geoprior.models.forecast_tuner.SubsNetTuner`
2. :meth:`~geoprior.models.forecast_tuner._geoprior_tuner.SubsNetTuner.create`
3. :meth:`~geoprior.models.forecast_tuner._base_tuner.PINNTunerBase.search`
4. :meth:`~geoprior.models.forecast_tuner._geoprior_tuner.SubsNetTuner.build`
5. :mod:`geoprior.models.tuners`

This order moves from the public entry point to the specialized build
logic, then back to the generic search engine and the compatibility
surface.

Source listings with comments
-----------------------------

These source listings are included intentionally so that the inline
implementation comments remain visible during HTML rendering.

Public compatibility shim
~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/models/tuners.py
   :language: python
   :caption: geoprior/models/tuners.py

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

A few design details are worth keeping in mind while reading this API
page:

- ``SubsNetTuner.create(...)`` is the recommended user entry point when
  input and target arrays are already available, because it infers fixed
  dimensions and applies GeoPrior-aware canonicalization. 
- ``PINNTunerBase`` is generic in role, while
  ``SubsNetTuner`` provides the model-aware specialization. This is what
  keeps the tuning stack reusable without losing the scientific contract
  of GeoPrior. 
- The compatibility shim ``geoprior.models.tuners`` is a public import
  convenience, not a second implementation of the tuning logic.

- The tuning layer is tightly coupled to the GeoPrior forecasting and
  physics assumptions, so it should be read together with the subsidence
  model API and with the Stage-3 tuning workflow. 

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
