Parameters API
==============

.. currentmodule:: geoprior.params

The :mod:`geoprior.params` module provides compact, self-documenting
parameter descriptors for scalar physical quantities used in
physics-guided neural networks and related geoscientific workflows.
The module combines two complementary ideas:

- **Learnable scalar wrappers** for parameters that should be estimated
  during training, often through a log-space parametrisation that
  preserves positivity.
- **Fixed scalar wrappers** for constants and modelling datums that
  should remain non-trainable while still participating cleanly in
  configuration, serialization, and model assembly.

In the GeoPrior subsidence workflow, these wrappers are especially
useful for global physical hyperparameters such as effective
compressibility, consistency factors, the effective unit weight of
water, or a reference hydraulic head. They complement the spatially
varying effective fields learned elsewhere in the model stack rather
than replacing them. The module also retains a legacy family of
coefficient wrappers for backwards-compatible handling of a generic
positive physics coefficient ``C``. 

The implementation supports both a TensorFlow-backed mode and a NumPy
fallback mode. In TensorFlow mode, learnable parameters are represented
through trainable variables and can participate directly in optimization.
In fallback mode, the same wrappers degrade gracefully to numeric values
while preserving a consistent user-facing interface. 

Design overview
---------------

The API can be read as three layers.

1. **Legacy coefficient descriptors**
   These wrappers keep compatibility with earlier workflows that expose
   a generic positive coefficient ``C``.
2. **General learnable and fixed bases**
   These base classes provide the shared semantics for trainable versus
   fixed scalar descriptors, configuration handling, and value retrieval.
3. **Domain-facing parameter descriptors and resolver helpers**
   These are the classes and functions most users interact with,
   including hydraulic conductivity, specific storage, source/sink
   strength, compressibility, consistency factors, water unit weight,
   and reference-head configuration. 

Public module surface
---------------------

The following autosummary blocks use fully qualified targets so that the
API page remains stable during Sphinx builds and does not depend on
package-relative lookup.

Module reference
~~~~~~~~~~~~~~~~

.. automodule:: geoprior.params
   :no-members:
   :no-inherited-members:
   :show-inheritance:
   :no-index:

Legacy coefficient descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These wrappers preserve backwards compatibility for a generic physics
coefficient ``C``. They remain useful when older experiments or saved
configurations still rely on the original coefficient-based interface.

.. autosummary::
   :toctree: _autosummary/params/legacy
   :nosignatures:
   :no-index:

   ~geoprior.params.LearnableC
   ~geoprior.params.FixedC
   ~geoprior.params.DisabledC

General parameter bases
~~~~~~~~~~~~~~~~~~~~~~~

For advanced extension work, the module exposes abstract base classes
for learnable and fixed scalar descriptors. These bases are helpful when
implementing new parameter families that should follow the same
serialization and runtime conventions as the built-in descriptors.

.. autosummary::
   :toctree: _autosummary/params/bases
   :nosignatures:

   ~geoprior.params.BaseLearnable
   ~geoprior.params.BaseFixed

Learnable physical descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The learnable family wraps trainable scalar quantities used in the
physics-guided model. Positive parameters generally use a log-space
representation so that optimization proceeds in an unconstrained space
while the recovered physical parameter stays positive.

.. autosummary::
   :toctree: _autosummary/params/learnable
   :nosignatures:
   :no-index:

   ~geoprior.params.LearnableK
   ~geoprior.params.LearnableSs
   ~geoprior.params.LearnableQ
   ~geoprior.params.LearnableMV
   ~geoprior.params.LearnableKappa

Fixed physical descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~

The fixed family is intended for constants and modelling datums that
must remain non-trainable. Although fixed, these objects still provide a
uniform configuration and retrieval interface, which makes them easier
to integrate into model construction and serialization logic.

.. autosummary::
   :toctree: _autosummary/params/fixed
   :nosignatures:
   :no-index:

   ~geoprior.params.FixedGammaW
   ~geoprior.params.FixedHRef

Resolution and normalization helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The resolver helper provides a flexible bridge between raw numeric
values, wrapper instances, configuration dictionaries, and serialization
workflows. It is the most convenient entry point when a pipeline wants
to accept both simple user input and explicit descriptor objects.

.. autosummary::
   :toctree: _autosummary/params/helpers
   :nosignatures:
   :no-index:

   ~geoprior.params.resolve_physical_param

Interpretation in the GeoPrior workflow
---------------------------------------

The parameter wrappers documented here should be interpreted as
**global scalar descriptors**. In other parts of the GeoPrior modeling
stack, quantities such as hydraulic conductivity, specific storage,
compressible thickness, and relaxation time may also appear as
spatially varying effective fields learned from covariates and neural
representations. The purpose of :mod:`geoprior.params` is different: it
captures global scalar hyperparameters and modelling constants in a form
that remains explicit, serializable, and physically interpretable.

This distinction is particularly important for subsidence forecasting.
For example, a scalar compressibility wrapper such as
:class:`~geoprior.params.LearnableMV` does not replace the learned field
structure in the main subsidence model. Instead, it provides a compact
and interpretable scalar quantity that can be reused by equilibrium
relations, priors, or consistency terms. Likewise,
:class:`~geoprior.params.LearnableKappa` represents a scalar consistency
factor used in a relaxation-time prior, while
:class:`~geoprior.params.FixedGammaW` and
:class:`~geoprior.params.FixedHRef` act as physical constants or datum
choices that help make the governing relations explicit. 

Usage examples
--------------

The examples below illustrate the intended public usage pattern.
Because the public import surface is the module itself, users typically
write imports of the following form:

.. code-block:: python

   from geoprior.params import LearnableK
   from geoprior.params import LearnableMV
   from geoprior.params import FixedGammaW
   from geoprior.params import FixedHRef
   from geoprior.params import resolve_physical_param

A learnable conductivity descriptor:

.. code-block:: python

   from geoprior.params import LearnableK

   k_param = LearnableK(initial_value=1.0)
   value = k_param.get_value()

A fixed reference-head descriptor:

.. code-block:: python

   from geoprior.params import FixedHRef

   h_ref = FixedHRef(value=0.0, mode="auto")

A flexible resolution step from a raw numeric value:

.. code-block:: python

   from geoprior.params import resolve_physical_param

   mv = resolve_physical_param(
       1e-7,
       name="mv",
       status="learnable",
       param_type="MV",
   )

Notes
-----

- The module exports a compact public surface through ``__all__`` for
  the main descriptor families, including the legacy ``C`` wrappers,
  learnable scalar descriptors, and the fixed ``GammaW`` / ``HRef``
  descriptors.
- The helper :func:`~geoprior.params.resolve_physical_param` broadens
  that surface by normalizing numbers, wrappers, strings, and
  configuration dictionaries into a consistent descriptor workflow. 
- Private helpers such as ``_BaseC`` and the lower-level conversion
  utilities are intentionally omitted from the public API summary so
  that the page stays focused on the import patterns intended for end
  users.
