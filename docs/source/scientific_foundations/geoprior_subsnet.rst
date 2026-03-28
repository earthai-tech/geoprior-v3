.. _geoprior_subsnet:

GeoPriorSubsNet
===============

``GeoPriorSubsNet`` is the flagship model of GeoPrior-v3.

It is a **prior-regularized, physics-informed, multi-horizon
forecasting model** for land subsidence with groundwater
coupling. Architecturally, it combines an attentive
encoder-decoder forecasting backbone with a physics head that
learns effective hydrogeological fields and penalizes
violations of groundwater-flow and consolidation physics.

At a high level, GeoPriorSubsNet is designed to answer two
questions at once:

1. **Can we forecast future subsidence and groundwater
   behavior over a horizon?**
2. **Can we do so while keeping the learned dynamics
   physically interpretable and regularized?**

This makes the model suitable not only for prediction, but
also for parameter-oriented scientific analysis, transfer
studies, and physics-aware uncertainty workflows.

Scientific role in GeoPrior-v3
------------------------------

GeoPriorSubsNet is the **general coupled model** in the
current GeoPrior family.

It is the default choice when you want:

- groundwater-aware subsidence forecasting,
- physically constrained multi-horizon prediction,
- learned effective fields such as :math:`K`, :math:`S_s`,
  :math:`\tau`, and :math:`Q`,
- explicit physics residuals inside training,
- and uncertainty outputs organized around a physics-guided
  mean forecast.

Within the broader model family, it plays the role of the
most expressive current model, while
:class:`~geoprior.models.PoroElasticSubsNet` acts as a
consolidation-first preset with stronger prior-oriented
defaults.

What GeoPriorSubsNet combines
-----------------------------

GeoPriorSubsNet combines three layers of modeling logic.

**1. Multi-horizon forecasting**

The model predicts over a future horizon rather than at only
one next step. It uses an attentive sequence backbone with
static, dynamic, and future-known inputs, making it closer to
modern interpretable time-series forecasting architectures
than to a simple feedforward PINN.

**2. Physics-guided latent structure**

The model does not only predict observed targets. It also
learns effective physical fields and uses them inside
groundwater and consolidation residuals.

**3. Prior-regularized scientific training**

The learned fields are constrained by priors, smoothness, and
bounds logic, so the physics branch behaves as a structured
scientific regularizer rather than as a free latent head.

A concise model summary
-----------------------

A useful summary is:

.. code-block:: text

   attentive encoder-decoder
         +
   supervised forecasting heads
         +
   learned physics fields
         +
   residual-based physical constraints
         +
   explicit scaling and identifiability control

This combination is what distinguishes GeoPriorSubsNet from a
standard black-box forecaster.

Conceptual architecture
-----------------------

GeoPriorSubsNet builds on a shared attentive forecasting
backbone inherited from the GeoPrior neural stack.

The model processes:

- static features,
- dynamic historical features,
- future-known features,
- and coordinate-aware forecast positions.

The core encoder-decoder path may include:

- variable selection logic,
- dense preprocessing paths,
- hybrid LSTM or transformer-like sequence encoding,
- multi-scale temporal aggregation,
- attention-based decoding over the forecast horizon.

The forecasting backbone then supports two logically separate
heads:

- a **data head** for supervised outputs,
- a **physics head** for effective hydrogeological fields.

The resulting design keeps forecasting and physical
regularization tightly coupled without forcing them to be the
same computation.

Inputs
------

GeoPriorSubsNet follows the GeoPrior **dict-input** contract.

A typical batch contains:

- ``static_features`` with shape ``(B, S)``
- ``dynamic_features`` with shape ``(B, T_in, D)``
- ``future_features`` with shape ``(B, T_out, F)``
- ``coords`` with shape ``(B, T_out, 3)``

and, when required by the closure:

- ``H_field`` or a compatible thickness field.

The default coordinate order is:

.. code-block:: text

   ['t', 'x', 'y']

unless explicitly overridden through the scaling
configuration.

.. note::

   The coordinate tensor is not auxiliary decoration. It is
   part of the physics contract. The physics pathway computes
   derivatives with respect to these coordinates and then
   converts them into SI-consistent derivatives through the
   scaling machinery.

Outputs
-------

The public supervised outputs are intentionally simple:

- ``subs_pred``
- ``gwl_pred``

These are the model’s two user-facing output families.

``subs_pred``
    Multi-horizon subsidence forecast.

``gwl_pred``
    Multi-horizon groundwater-level forecast.

When quantiles are enabled, these outputs may carry a
quantile axis rather than only a single central prediction.

One important design detail is that the quantile head is
**centered on the physics-driven mean**, so uncertainty is
modeled around a physically informed baseline rather than
around a purely unconstrained neural forecast.

Learned physical fields
-----------------------

A defining feature of GeoPriorSubsNet is that it learns
effective physical fields in addition to the supervised
targets.

The core learned fields are:

- :math:`K(x,y)` — hydraulic conductivity
- :math:`S_s(x,y)` — specific storage
- :math:`\tau(x,y)` — effective relaxation / consolidation
  timescale
- :math:`Q(t,x,y)` — source / forcing term for the
  groundwater residual

These are effective grid-scale fields, not literal laboratory
parameters. They are produced by the physics head and then
mapped into physically meaningful ranges through scaling,
bounded transforms, and optional constraint logic.

The groundwater-flow residual
-----------------------------

When groundwater physics is active, GeoPriorSubsNet enforces
a groundwater-flow residual of the form:

.. math::

   R_{gw}
   =
   S_s \, \partial_t h
   -
   \nabla \cdot (K \nabla h)
   -
   Q

where:

- :math:`h` is hydraulic head,
- :math:`K` is hydraulic conductivity,
- :math:`S_s` is specific storage,
- :math:`Q` is the groundwater forcing term.

The implementation uses automatic differentiation with
respect to the same coordinate tensor seen by the forward
pass, then converts those derivatives into SI-consistent
forms before assembling the residual.

The consolidation residual
--------------------------

The subsidence pathway is governed by a relaxation-style
consolidation closure:

.. math::

   R_{cons}
   =
   \partial_t s
   -
   \frac{s_{eq}(h) - s}{\tau}

where:

- :math:`s` is subsidence,
- :math:`\tau` is an effective relaxation time,
- :math:`s_{eq}(h)` is an equilibrium settlement proxy.

A common approximation used by GeoPrior is:

.. math::

   s_{eq}(h)
   \approx
   S_s \, \Delta h \, H

with drawdown:

.. math::

   \Delta h = [h_{ref} - h]_+

and effective thickness :math:`H`.

In practice, the implementation prefers a step-consistent
consolidation update rather than a naive noisy finite
difference of :math:`\partial_t s`, which makes the approach
more stable at coarse temporal resolution.

The timescale prior
-------------------

One of the most important scientific ideas in GeoPriorSubsNet
is the prior linking learned :math:`\tau` to the learned
hydrogeological fields.

Conceptually, the model uses a closure of the form:

.. math::

   \tau_{prior}
   \approx
   \frac{H_d^2 \, S_s}{\pi^2 \, \kappa \, K}

where:

- :math:`H_d` is an effective drainage thickness,
- :math:`K` is hydraulic conductivity,
- :math:`S_s` is specific storage,
- :math:`\kappa` is a consistency factor,
- :math:`\tau` is the learned timescale.

The model then measures a prior mismatch in log space, which
helps stabilize training across large timescale ranges and
makes the penalty behave more like a relative error than an
absolute one.

Additional physics-oriented regularization
------------------------------------------

GeoPriorSubsNet is not limited to two PDE-style residuals.

It also supports additional scientific regularization terms,
including:

- a **timescale prior** on :math:`\tau`,
- a **smoothness prior** on learned fields,
- a **bounds residual** for plausible physical envelopes,
- an optional **:math:`m_v` prior** linking storage to
  compressibility and water unit weight,
- and optional regularization on the forcing term
  :math:`Q`.

This makes the model best understood as a **structured
physics-regularized forecaster**, not just as a PINN with one
equation.

PDE modes
---------

GeoPriorSubsNet supports several physics regimes through
``pde_mode``.

The main conceptual settings are:

``'both'``
    Activate both groundwater-flow and consolidation physics.

``'consolidation'``
    Use only the consolidation residual.

``'gw_flow'``
    Use only the groundwater-flow residual.

``'none'``
    Disable PDE residuals and behave as a data-driven
    encoder-decoder.

This is useful because not every dataset can support the same
physical regime. In some cases, a consolidation-first regime
is more trustworthy than a fully coupled groundwater-flow
setup.

Scaling is part of the model definition
---------------------------------------

Scaling in GeoPriorSubsNet is not a secondary preprocessing
choice. It is part of the model contract itself.

The model accepts scaling configuration through
``scaling_kwargs``, which are wrapped and resolved by a
Keras-serializable
:class:`~geoprior.models.subsidence.scaling.GeoPriorScalingConfig`.

This scaling layer governs, among other things:

- time-unit interpretation,
- coordinate normalization,
- degree-to-meter handling,
- groundwater semantics,
- SI conversion,
- alias consistency,
- bounds structure,
- feature naming consistency.

This is why a saved GeoPrior model should always be
interpreted together with its scaling context.

.. admonition:: Best practice

   Save the resolved scaling payload alongside the model
   artifacts.

   If two runs differ unexpectedly across cities or sites,
   compare the resolved scaling configuration first.

Identifiability regimes
-----------------------

GeoPriorSubsNet exposes **identifiability control** as a
first-class part of the public model design.

The current identifiability profiles include:

- ``base``
- ``anchored``
- ``closure_locked``
- ``data_relaxed``

These profiles exist because the poroelastic closure contains
ridge-like parameter trade-offs. Multiple combinations of
:math:`K`, :math:`S_s`, :math:`H_d`, and :math:`\tau` can
produce similar behavior unless the problem is constrained
carefully.

Different regimes change defaults such as:

- whether physics fields are frozen over time,
- whether subsidence residual freedom is allowed,
- bounds behavior,
- prior strength,
- warmup schedules,
- and optional locks on heads such as ``tau_head``.

This makes GeoPriorSubsNet useful not only for prediction,
but also for controlled scientific ablation and inverse-style
experiments.

Compile-time loss structure
---------------------------

GeoPriorSubsNet uses a custom training structure that blends
supervised and physics-aware losses.

The main compile-time scalar weights include:

- ``lambda_cons``
- ``lambda_gw``
- ``lambda_prior``
- ``lambda_smooth``
- ``lambda_mv``
- ``lambda_bounds``
- ``lambda_q``

These are assembled into a physics objective together with a
global physics multiplier controlled by ``lambda_offset`` and
``offset_mode``.

A useful conceptual split is:

- **data loss** for supervised forecast accuracy,
- **physics loss** for physical consistency,
- **global physics multiplier** for schedule-aware balancing.

This is one reason GeoPriorSubsNet can be trained in a more
controlled way than a monolithic single-objective network.

Numerical stability design
--------------------------

Because GeoPriorSubsNet learns positive physical fields and
differentiates through coordinate-sensitive residuals, it
includes explicit stability mechanisms.

Examples include:

- clipping physics logits,
- sanitizing residual scales,
- warmup and ramp schedules for physics enforcement,
- NaN-safe clipping constraints for log-parameters,
- filtered gradients for bad batches,
- soft vs hard bounds logic,
- finite-safe logging and diagnostics helpers.

These mechanisms are not incidental engineering details.
They are part of the scientific usability of the model.

Quantiles and uncertainty
-------------------------

GeoPriorSubsNet can operate in deterministic or probabilistic
mode.

When ``quantiles`` are provided, the model emits quantile
forecasts across the horizon. The uncertainty structure is
built around the physics-informed mean path rather than
around a detached uncertainty head.

This design has two advantages:

- the central forecast remains tied to the physics branch,
- the uncertainty export can be calibrated later without
  detaching it from the model’s scientific interpretation.

This is one of the reasons the model integrates naturally
with later interval calibration and forecast-export stages.

Diagnostics and payload export
------------------------------

GeoPriorSubsNet is designed to emit diagnostics, not only
predictions.

The model supports:

- forward passes with auxiliary outputs,
- epsilon-style physics summaries,
- physics payload export,
- physical-parameter extraction,
- identifiability audits,
- plot-oriented diagnostics.

In particular, the exported physics payload typically
contains arrays such as:

- ``tau``
- ``tau_prior`` or ``tau_closure``
- ``K``
- ``Ss``
- ``Hd``
- consolidation residual values
- summary metrics

This makes GeoPriorSubsNet especially suitable for
publication-grade and audit-oriented workflows.

Serialization and reconstruction
--------------------------------

GeoPriorSubsNet is designed to be reconstructable.

The class is Keras-serializable and supports reconstruction
through:

- ``get_config()``
- ``from_config(...)``
- and explicit ``custom_objects`` handling for the wrapped
  physics parameters and scaling configuration.

This matters because model reuse in GeoPrior is not only
about loading weights. It is also about reconstructing the
scientific configuration faithfully.

A minimal usage example
-----------------------

.. code-block:: python

   import tensorflow as tf
   from geoprior.models import GeoPriorSubsNet

   model = GeoPriorSubsNet(
       static_input_dim=3,
       dynamic_input_dim=8,
       future_input_dim=4,
       output_subsidence_dim=1,
       output_gwl_dim=1,
       forecast_horizon=3,
       quantiles=[0.1, 0.5, 0.9],
       pde_mode="both",
       scaling_kwargs={"time_units": "year"},
   )

   batch = {
       "static_features": tf.zeros([8, 3]),
       "dynamic_features": tf.zeros([8, 12, 8]),
       "future_features": tf.zeros([8, 3, 4]),
       "coords": tf.zeros([8, 3, 3]),
   }

   y = model(batch, training=False)
   print(sorted(y.keys()))

A second useful debugging pattern is to inspect both the
supervised outputs and the auxiliary physics outputs:

.. code-block:: python

   y_pred, aux = model.forward_with_aux(batch, training=False)
   print(y_pred["subs_pred"].shape)
   print(aux["phys_mean_raw"].shape)

When to use GeoPriorSubsNet
---------------------------

GeoPriorSubsNet is the best default when:

- you want the most general current GeoPrior model,
- you want coupled groundwater and consolidation behavior,
- you have a reasonable scaling and coordinate setup,
- you want a physics-aware probabilistic forecast model,
- you want later access to physics payloads and diagnostics.

When not to use it first
------------------------

You may prefer
:class:`~geoprior.models.PoroElasticSubsNet` first when:

- the groundwater PDE is poorly constrained,
- forcing proxies are weak or unavailable,
- you want a stronger prior-driven consolidation baseline,
- you are debugging units, bounds, or identifiability,
- you want a consolidation-first ablation.

That variant keeps the same general contract but changes the
default scientific posture.

See also
--------

.. seealso::

   - :doc:`models_overview`
   - :doc:`physics_formulation`
   - :doc:`poroelastic_background`
   - :doc:`residual_assembly`
   - :doc:`losses_and_training`
   - :doc:`data_and_units`
   - :doc:`scaling`
   - :doc:`identifiability`