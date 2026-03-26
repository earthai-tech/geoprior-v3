.. _geoprior_scaling:

=============================================
Scaling and conventions (``scaling_kwargs``)
=============================================

GeoPrior-v3 treats ``scaling_kwargs`` as part of the **model
contract**, not as optional preprocessing metadata.

This is one of the most important design choices in the
project. The scaling payload determines how the model
interprets:

- subsidence targets,
- groundwater variables,
- thickness fields,
- coordinate order and normalization,
- time-unit conversion,
- SI conversion,
- head-from-depth proxy logic,
- residual scaling,
- bounds and priors,
- and selected training / diagnostics policies.

The current code and model docstrings describe
``scaling_kwargs`` as the **single most important
configuration** for GeoPrior, precisely because it controls
how raw dataset values become the internal quantities used by
the physics core. If this payload changes, the *physics
problem definition itself* changes. 

This page explains:

- why scaling is scientifically central in GeoPrior,
- how ``GeoPriorScalingConfig`` works,
- what kinds of fields belong in ``scaling_kwargs``,
- and how scaling affects derivatives, priors, bounds,
  diagnostics, and saved-model reuse.

Why scaling is physics-critical
-------------------------------

GeoPrior is not a standard tabular forecaster.

Its physics core differentiates with respect to the forecast
coordinate tensor, converts those derivatives into
SI-consistent quantities, and assembles residuals for:

- groundwater flow,
- consolidation relaxation,
- timescale consistency,
- smoothness,
- and bounds.

That means the model is only scientifically meaningful if the
coordinate spans, time units, head conventions, and target
units are handled consistently. This is especially important
for groundwater-driven subsidence, where storage, drawdown,
drainage thickness, and delayed compaction must remain
physically interpretable :cite:p:`GallowayBurbey2011,Shirzaeietal2021,Ellisetal2023`.

A useful practical rule is:

.. code-block:: text

   if scaling is wrong,
   the physics branch is wrong

even when the supervised loss still appears to behave well.

A compact summary
-----------------

A useful summary is:

.. code-block:: text

   raw dataset values
        ↓
   scaling_kwargs defines meaning
        ↓
   GeoPriorScalingConfig resolves a canonical payload
        ↓
   model uses SI-consistent states and residuals
        ↓
   training, inference, and saved-model reuse remain aligned

This is the core role of scaling in GeoPrior.

The scaling configuration object
--------------------------------

GeoPrior wraps the scaling payload in a small
Keras-serializable container:

:class:`~geoprior.models.subsidence.scaling.GeoPriorScalingConfig`

The implementation describes this object as the container that
stores and reconstructs the **physics scaling and slicing
controls** used by GeoPriorSubsNet. This matters because a
reloaded model must preserve the same coordinate handling,
time-unit interpretation, groundwater conventions, bounds,
and residual scaling behavior it had during training. 

The class supports construction from:

- ``None`` (empty config),
- a mapping / dict-like payload,
- a filesystem path to a JSON or YAML config file,
- or an existing ``GeoPriorScalingConfig`` instance. 

The resolution pipeline
-----------------------

The scaling object does not assume the incoming payload is
already clean.

The current implementation resolves the scaling contract
through a four-step pipeline:

1. load the payload,
2. canonicalize keys and fill defaults,
3. enforce alias consistency,
4. validate values and required fields. 

This is important because it makes reloaded models behave the
same way as models built from a fresh training run.

A typical usage pattern is:

.. code-block:: python

   from geoprior.models.subsidence.scaling import (
       GeoPriorScalingConfig,
   )

   cfg = GeoPriorScalingConfig.from_any(
       "path/to/scaling_kwargs.json"
   )
   scaling_kwargs = cfg.resolve()

That pattern is also the right mental model for the docs:
the user may write an initial payload, but the *resolved*
payload is what actually defines the model’s physics meaning.

Where ``scaling_kwargs`` enters the workflow
--------------------------------------------

In the GeoPrior workflow, scaling is not introduced only at
model build time. It begins earlier and then persists through
the whole staged pipeline.

The older scaling guide already captured the right structure:
Stage-1 writes ``scaling_kwargs.json`` as a compact resolved
record of dataset conventions, feature naming, SI conversion
constants, coordinate mode, and physics-control knobs, and
Stage-2 then consumes that payload during training. 

In the current codebase, that logic goes further:

- the resolved scaling config is wrapped in
  ``GeoPriorScalingConfig``;
- the model stores it as a serialized Keras object in
  ``get_config()``;
- ``from_config()`` rehydrates it during model reload. 

This is why scaling belongs to the **scientific foundations**
section, not only to the user guide.

What ``scaling_kwargs`` governs
-------------------------------

The internal doc templates are very explicit about the role
of the scaling payload. The resolved payload is used by:

- target formatting for subsidence and head,
- coordinate handling (order, normalization, SI conversion),
- PDE derivative scaling and chain-rule correction,
- physics closures such as drawdown rules and head-from-depth
  proxy logic,
- priors and bounds for :math:`K`, :math:`S_s`,
  :math:`\tau`, and :math:`H`,
- training policies such as warmups, ramps, and diagnostics. 

So a good way to think about ``scaling_kwargs`` is:

- it defines **what the data mean**,
- not only **how the data are numerically scaled**.

Target scaling and conventions
------------------------------

The first role of ``scaling_kwargs`` is to define how the raw
targets become SI-consistent internal quantities.

Subsidence
~~~~~~~~~~

The payload can declare the semantic type of subsidence
through a field such as:

``subsidence_kind``
    Usually ``"cumulative"`` or ``"incremental"``. 

The corresponding linear conversion is:

.. math::

   s_{si}
   =
   s_{raw}\,subs\_scale\_si + subs\_bias\_si.

Typical interpretation:

- if raw subsidence is stored in millimeters, then
  ``subs_scale_si = 1e-3`` converts it to meters;
- if it is already in meters, ``subs_scale_si = 1.0`` is
  appropriate. 

Groundwater / head
~~~~~~~~~~~~~~~~~~

The payload may also include:

- ``head_scale_si``
- ``head_bias_si``

with the same general form:

.. math::

   h_{si}
   =
   h_{raw}\,head\_scale\_si + head\_bias\_si. 

This is especially important when the workflow mixes:

- already-physical head values,
- depth-to-water values later converted into head,
- or preprocessed standardized inputs that must be restored
  before physics is evaluated.

Thickness
~~~~~~~~~

Thickness enters both the equilibrium settlement closure and
the timescale prior. The scaling payload therefore also
supports:

- ``H_scale_si``
- ``H_bias_si``

so that:

.. math::

   H_{si}
   =
   H_{raw}\,H\_scale\_si + H\_bias\_si. 

This is a high-impact quantity, because incorrect thickness
scaling can distort:

- equilibrium settlement,
- effective drainage thickness,
- and the closure-based timescale prior.

Groundwater semantics and head proxies
--------------------------------------

GeoPrior explicitly tracks how groundwater is represented.

The scaling payload may define conventions such as:

- ``gwl_kind``
- ``gwl_sign``
- ``use_head_proxy``. 

This matters because different datasets may store:

- hydraulic head directly,
- depth to groundwater,
- or a head-like quantity reconstructed from surface
  elevation and groundwater depth.

The model docstring already makes this explicit: one of the
key ideas of GeoPrior is to **convert GWL to hydraulic head
using a scaling configuration** before computing physics
losses. 

A useful conceptual conversion is:

.. math::

   h = z_{surf} - z_{GWL}

when groundwater is stored as positive downward depth below
surface.

The important point for users is not only the formula, but
that the formula belongs to the **scaling contract** rather
than to ad hoc hand-written preprocessing.

Drawdown rules
--------------

The compaction side of GeoPrior depends on drawdown rather
than only on the raw groundwater variable.

The scaling payload can therefore also control:

- drawdown definition,
- sign policy,
- optional head-reference behavior,
- and whether head should be reconstructed from a proxy.

This is one reason ``scaling_kwargs`` belongs in the model
config itself: drawdown convention changes the meaning of the
consolidation residual and the equilibrium settlement
closure.

Coordinate handling
-------------------

The next major job of scaling is to define the derivative
frame.

GeoPrior expects the forecast coordinate tensor to follow a
declared order, typically:

.. code-block:: text

   ["t", "x", "y"]

This order is part of the scaling contract, not a hard-coded
assumption. The current model documentation explicitly states
that the coordinate axis order is expected to be
``['t','x','y']`` unless overridden by
``scaling_kwargs['coord_order']``. 

A wrong coordinate order may not cause a shape error, but it
makes the PDE residuals meaningless.

Normalized coordinates and chain-rule scaling
---------------------------------------------

One of the most important scaling roles is derivative
correction when coordinates are normalized.

The current doc templates describe this explicitly:
if coordinates are normalized and
``scale_pde_residuals=True``, GeoPrior applies the chain rule
using the coordinate ranges. For example, if

.. math::

   t' = \frac{t - t_0}{\Delta t},

then:

.. math::

   \partial_t
   =
   \frac{1}{\Delta t}\,\partial_{t'}.

Similarly for the spatial coordinates:

.. math::

   \partial_x
   =
   \frac{1}{\Delta x}\,\partial_{x'},
   \qquad
   \partial_y
   =
   \frac{1}{\Delta y}\,\partial_{y'}. 

This is one of the clearest examples of why scaling is
physics-critical. Raw autodiff operates in **model coordinate
space**, but the residuals must be interpreted in
**SI-consistent physical space**.

The corresponding payload fields typically include:

- ``coords_normalized``
- ``coord_ranges``
- ``coord_ranges_si``
- ``coord_order``. 

Time units
----------

Time handling is another central part of the scaling
contract.

The current docs for the model parameters define
``time_units`` as the name of the time unit used by the model
coordinates, for example:

- ``"year"``
- ``"month"``
- ``"day"``
- ``"second"``. 

This field influences:

- conversion from dataset time steps to SI seconds,
- the magnitude of time derivatives in PDE residuals,
- and the interpretation of :math:`\tau` priors. 

A useful practical rule is:

- always set ``time_units`` explicitly,
- even if the training setup feels obvious to you.

A wrong time-unit declaration silently corrupts both:

- the groundwater residual,
- and the consolidation timescale logic.

Longitude / latitude vs projected coordinates
---------------------------------------------

The scaling payload also controls whether spatial coordinates
are already metric or are still in angular units.

Typical fields include:

- ``coords_in_degrees``
- ``deg_to_m_lon``
- ``deg_to_m_lat``. 

This is essential because a longitude-latitude derivative is
not directly a meter-based spatial derivative. If the physics
core interprets raw degrees as meters, the spatial PDE terms
are distorted by large and site-dependent scaling errors.

A good rule is:

- projected coordinates are preferred for spatial PDE
  interpretation;
- if you must use lon/lat, the degree-to-meter conversion
  must be explicit in the scaling payload.

Residual scaling policy
-----------------------

GeoPrior also uses the scaling payload to control how
residuals themselves are nondimensionalized and audited.

The model docs already highlight:

- residual display/scaling,
- warmup/ramp gates,
- training policies,
- and diagnostics

as part of the scaling story. 

This is important because the same payload influences both:

- the **physical meaning** of the states and fields,
- and the **optimization behavior** of the physics losses.

In other words, scaling is not only about units. It is also
about keeping the loss geometry stable and comparable across
runs.

Bounds and priors
-----------------

One of the less obvious but very important jobs of
``scaling_kwargs`` is to define priors and plausible bounds.

The internal docs explicitly list priors and bounds for
:math:`K`, :math:`S_s`, :math:`\tau`, and :math:`H` among the
core responsibilities of the payload. 

This matters because GeoPrior learns effective physical
fields and then regularizes them. Without a stable scaling
contract, these fields can become:

- numerically unstable,
- physically implausible,
- or inconsistent across saved and reloaded models.

PoroElasticSubsNet even applies a “fill missing bounds”
policy on ``scaling_kwargs['bounds']`` by default, which
shows how central the payload is to the scientific posture of
the model family. 

Forcing-term meaning
--------------------

The forcing term :math:`Q` in GeoPrior’s groundwater residual
also depends on scaling conventions.

The current math helpers normalize the meaning of ``Q`` using
a field such as:

- ``Q_kind`` / ``q_kind`` / ``gw_q_kind``

and support meanings such as:

- ``per_volume``
- ``recharge_rate``
- ``head_rate``. 

The point is not that every user must expose a learnable
forcing term. The point is that when forcing is present, its
units and semantics must be declared explicitly, because the
groundwater residual expects an SI-consistent source term.

Serialization and reproducibility
---------------------------------

One of the strongest reasons GeoPrior uses a dedicated
scaling object is **saved-model reproducibility**.

The current model serialization logic stores
``scaling_kwargs`` as a serialized Keras object representing
the validated scaling configuration, and the model docs state
explicitly that this preserves the exact conventions
(units, coordinate normalization, bounds) used during
training and is critical for consistent inference. 

This means a saved GeoPrior model is not fully specified by:

- architecture weights,
- loss settings,
- and optimizer state

alone.

It also depends on the resolved scaling payload.

A useful rule is:

.. admonition:: Best practice

   Save the resolved scaling payload alongside all serious
   model artifacts.

   If two runs differ unexpectedly across sites, the first
   thing to compare is often the resolved scaling
   configuration. 

Why the scaling payload is JSON-safe
------------------------------------

The scaling helper includes defensive serialization through
``_jsonify(...)``, which converts nested objects, tuples,
sets, and NumPy scalars into stable JSON-safe Python types. 

This may look like an engineering detail, but it is
scientifically important: reproducibility is weaker if the
saved scaling contract depends on non-deterministic or
non-serializable objects.

A minimal example
-----------------

A compact test-style payload already used in the current
GeoPrior code looks like this:

.. code-block:: python

   scaling_kwargs = {
       "time_units": "year",
       "gwl_kind": "head",
       "gwl_sign": "up_positive",
       "use_head_proxy": False,
       "subsidence_kind": "cumulative",
       "coords_normalized": False,
       "track_aux_metrics": False,
       "allow_subs_residual": False,
   }

This is intentionally small, but it already shows the two
main roles of the payload:

- unit / meaning declaration,
- physics policy declaration. 

A richer payload may also include coordinate ranges, bounds,
feature indices, SI conversion factors, and diagnostics
settings.

Common mistakes
---------------

**Treating scaling as optional**

In GeoPrior, it is not optional. It defines the model’s
scientific interpretation. 

**Using normalized coordinates without range metadata**

This breaks the chain-rule conversion needed for PDE
derivatives. 

**Forgetting to declare time units**

This silently distorts time derivatives and timescale
priors. 

**Confusing head with depth-to-water**

This changes the meaning of drawdown and therefore the
consolidation closure. 

**Saving a model without its resolved scaling payload**

This makes later inference or transfer much harder to trust. 

A practical checklist
---------------------

Before trusting a GeoPrior run, confirm the following:

- ``coord_order`` matches the actual coordinate tensor;
- ``coords_normalized`` and coordinate spans are correct;
- ``time_units`` matches the time axis meaning;
- ``subs_scale_si`` converts subsidence to meters;
- head / groundwater semantics are explicit and correct;
- thickness scaling is correct and in meters internally;
- bounds and priors reflect the intended scientific regime;
- the saved model stores the same resolved scaling payload
  that was used during training.

If those items are correct, then later instability is more
likely to reflect:

- optimization difficulty,
- residual stiffness,
- or identifiability issues

rather than a basic contract error.

A compact scaling map
---------------------

The GeoPrior scaling story can be summarized as:

.. code-block:: text

   scaling_kwargs
      ├── target conventions
      │     subsidence, head, thickness
      ├── coordinate conventions
      │     order, normalization, SI conversion
      ├── groundwater semantics
      │     head, depth, drawdown, proxy logic
      ├── priors and bounds
      │     K, Ss, tau, H, closure controls
      ├── residual policies
      │     derivative scaling, diagnostics, warmups
      └── serialization
            preserved through GeoPriorScalingConfig

Relationship to the rest of the docs
------------------------------------

This page explains **what scaling controls**.

The companion pages explain related parts of the story:

- :doc:`data_and_units`
  explains the variable meanings and dataset contract;

- :doc:`physics_formulation`
  explains the governing equations and closures;

- :doc:`residual_assembly`
  explains how scaled quantities become residual maps and
  losses;

- :doc:`losses_and_training`
  explains how those scaled residuals affect optimization;

- :doc:`identifiability`
  explains how scaling and bounds interact with parameter
  trade-offs.

See also
--------

.. seealso::

   - :doc:`data_and_units`
   - :doc:`geoprior_subsnet`
   - :doc:`physics_formulation`
   - :doc:`residual_assembly`
   - :doc:`losses_and_training`
   - :doc:`identifiability`
   - :doc:`../user_guide/configuration`
   - :doc:`../user_guide/stage1`