.. _geoprior_data_and_units:

=============================
Data, units, and conventions
=============================

GeoPrior-v3 is highly sensitive to **units**, **sign
conventions**, and **coordinate semantics**.

This is not a cosmetic implementation detail. The flagship
model differentiates with respect to the forecast coordinate
tensor, converts those derivatives into SI-consistent form,
and then evaluates groundwater and consolidation residuals in
the physics core. If the data contract is inconsistent, the
model can appear numerically reasonable while the physics
branch becomes misleading or unstable. This is especially
important in groundwater-driven subsidence problems, where
drawdown, storage, drainage thickness, and delayed compaction
must all remain physically interpretable
:cite:p:`GallowayBurbey2011,Shirzaeietal2021,Ellisetal2023`.

This page defines:

- the **input batch contract** expected by GeoPrior,
- the meaning of the **scaling payload**
  (``scaling_kwargs``),
- the conventions used for head, drawdown, subsidence, and
  thickness,
- and the practical rules that keep the physics branch stable
  and scientifically meaningful.

A useful warning sign
---------------------

A very common failure mode is:

- the supervised loss looks fine,
- but physics diagnostics such as
  ``epsilon_cons_raw`` or ``epsilon_gw_raw`` explode.

When that happens, the first place to look is usually **this
page**, not the optimizer. In GeoPrior, derivative scales,
time units, coordinate spans, and head conventions are part
of the physical meaning of the model, not secondary metadata.

What this page covers
---------------------

This page covers four layers of the GeoPrior data contract:

1. **batch structure**
   what keys and shapes the model expects;

2. **physical variable semantics**
   what head, subsidence, and thickness mean in the physics
   core;

3. **coordinate and unit handling**
   how ``t``, ``x``, and ``y`` are interpreted and converted;

4. **scaling payload structure**
   how those conventions are recorded, validated, serialized,
   and reused.

See also
--------

.. seealso::

   - :doc:`models_overview`
   - :doc:`geoprior_subsnet`
   - :doc:`physics_formulation`
   - :doc:`residual_assembly`
   - :doc:`losses_and_training`
   - :doc:`scaling`
   - :doc:`../user_guide/stage1`
   - :doc:`../user_guide/stage2`

Input batch contract
====================

GeoPrior expects **dict inputs**.

At minimum, the physics-aware model path needs:

- ``coords`` for the spatio-temporal derivative frame;
- a thickness-like field such as ``H_field`` when the
  consolidation closure or timescale prior depends on it.

The most common input structure is:

.. code-block:: text

   inputs = {
       "static_features":  (B, S),
       "dynamic_features": (B, T_in, D),
       "future_features":  (B, T_out, F),
       "coords":           (B, T_out, 3),
       "H_field":          (B, T_out, 1),
   }

Here:

- ``B`` is batch size;
- ``T_in`` is the historical / encoder length;
- ``T_out`` is the forecast horizon;
- ``S``, ``D``, and ``F`` are the static, dynamic, and
  future-known feature dimensions.

A key point is that ``coords`` must align with the **forecast
horizon**, not with the encoder length. The physics residuals
are assembled on the forecast side of the model.

Required vs optional keys
-------------------------

In the standard GeoPrior workflow:

- ``static_features`` are expected by the forecasting
  backbone;
- ``dynamic_features`` provide the historical time-varying
  drivers;
- ``future_features`` provide known future covariates;
- ``coords`` are required for physics derivatives;
- ``H_field`` is required when the chosen physics closure
  needs thickness information.

If ``coords`` are missing, the model cannot construct the
derivative frame used by the physics branch. Likewise, if a
thickness field is needed but absent, the closure and prior
logic become underspecified.

A good practical rule is:

- if you want only a smoke test of the forward pass, keep the
  dict contract complete anyway;
- if you want physically meaningful diagnostics, the dict
  contract must be treated as mandatory.

Shapes and axis meaning
-----------------------

The most important shape rule is:

.. code-block:: text

   coords.shape == (B, T_out, 3)

The trailing axis contains the three coordinates in the order
specified by the scaling configuration, typically:

.. code-block:: text

   ["t", "x", "y"]

This means that the coordinate tensor is not a generic
“metadata array.” It is the actual differentiation frame for
the physics core.

.. admonition:: Best practice

   Keep the coordinate contract explicit in every custom
   dataset or NPZ bundle.

   The fastest way to break physics consistency is to silently
   change the coordinate order or horizon alignment.

Physical variables and internal SI meaning
==========================================

GeoPrior’s physics core is written in **SI-consistent
quantities**, even when the raw dataset is not.

The two main state variables are:

.. math::

   h(t,x,y) \quad \text{[m]}

.. math::

   s(t,x,y) \quad \text{[m]}

where:

- :math:`h` is hydraulic head, or a head-like proxy mapped to
  meters;
- :math:`s` is subsidence (settlement), in meters.

The learned effective physical fields are interpreted as:

.. math::

   K(x,y) \quad \text{[m/s]}

.. math::

   S_s(x,y) \quad \text{[1/m]}

.. math::

   \tau(x,y) \quad \text{[s]}

.. math::

   Q_{term}(t,x,y) \quad \text{[1/s in the residual form]}.

This is one of the most important facts about the GeoPrior
physics branch: **whatever the raw storage format of the data,
the internal residuals are built from SI-consistent
quantities**.

Why SI consistency matters
--------------------------

Without SI consistency, residual magnitudes become difficult
to interpret and impossible to compare across sites or runs.

For example:

- a head derivative in meters per second is not comparable to
  a derivative implicitly taken in “meters per year” unless
  that conversion is handled explicitly;
- a thickness in meters and a thickness in millimeters
  produce entirely different closure timescales if the
  scaling metadata is wrong;
- a longitude-latitude derivative treated as if it were
  already in meters can distort the groundwater residual by
  orders of magnitude.

This is why GeoPrior treats scaling and unit conversion as
part of the model contract rather than as passive
preprocessing metadata.

Subsidence conventions
======================

GeoPrior typically models **subsidence as cumulative
settlement**, but the scaling metadata allows the user to
declare what the dataset actually stores.

The key conceptual cases are:

``subsidence_kind = "cumulative"``
    The target stores total settlement up to that time.

``subsidence_kind = "incremental"``
    The target stores a per-step settlement increment.

The recommended SI conversion field is:

``subs_scale_si``
    Multiplicative factor that converts stored subsidence
    values to meters.

Examples:

- if the stored values are already in meters, use
  ``subs_scale_si = 1.0``;
- if the stored values are in millimeters, use
  ``subs_scale_si = 1e-3``.

An optional additive bias can also be recorded through a
field such as ``subs_bias_si``, though this is usually zero.

Practical interpretation
------------------------

The model’s supervised output is still called ``subs_pred``
regardless of storage convention. What changes is how the
dataset and the scaling payload define its physical meaning.

A useful rule is:

- the display unit in CSV exports may be convenient
  (for example, millimeters),
- but the physics side should be interpreted in SI meters.

Groundwater variable semantics
==============================

Groundwater data arrive in multiple forms, and GeoPrior
supports more than one convention.

The most important distinction is between:

- **hydraulic head**;
- **depth-to-water** (positive downward);
- and in some workflows, a **head proxy** reconstructed from
  a depth-like variable.

This is recorded in the scaling payload through fields such
as:

- ``gwl_kind``
- ``gwl_sign``
- ``use_head_proxy``.

Head-like data
--------------

If the dataset already stores **hydraulic head** in an
elevation-like form, then the model can use it directly once
it is converted to meters through fields such as:

- ``head_scale_si``
- ``head_bias_si``.

This is the cleanest case for the physics branch.

Depth-to-water data
-------------------

If the dataset stores **depth to groundwater** instead of
head, then GeoPrior needs surface elevation to construct a
head-like quantity.

The standard conceptual conversion is:

.. math::

   h = z_{surf} - z_{GWL},

where:

- :math:`z_{surf}` is surface elevation;
- :math:`z_{GWL}` is depth to groundwater, treated as
  positive downward.

This is why depth-like datasets must also provide access to
surface elevation through metadata such as:

- ``z_surf_col``
- ``z_surf_static_index``
- or equivalent stage-level bookkeeping.

.. warning::

   If groundwater is depth-like but surface elevation is
   missing, the physics meaning of the groundwater branch is
   incomplete.

Drawdown convention
===================

GeoPrior’s compaction logic is built from **drawdown**, not
just from whatever groundwater variable happened to exist in
the raw dataset.

Conceptually, drawdown begins from a reference rule such as:

.. math::

   \Delta h_{raw} = h_{ref} - h

or an equivalent convention declared by the scaling payload.
The physics core then applies a non-negative rule, so the
effective drawdown becomes something like:

.. math::

   \Delta h = [\Delta h_{raw}]_{+}.

This is important because the equilibrium settlement closure
is built from **non-negative head loss**, not from arbitrary
signed fluctuation.

A practical interpretation rule
-------------------------------

When diagnosing physics behavior, always ask:

- is the groundwater variable stored as head or depth?
- how is the reference head defined?
- what sign convention produces positive drawdown?
- is the non-negative gate behaving as expected?

Many apparent physics “bugs” are actually drawdown
convention mismatches.

Thickness and the ``H_field`` contract
======================================

GeoPrior uses an effective thickness field in several places,
especially:

- the equilibrium settlement proxy;
- the closure-based timescale prior;
- optional bounds and plausibility checks.

The most common input key is:

``H_field``
    Thickness-like input aligned with the forecast horizon.

A standard SI conversion field is:

``H_scale_si``
    Multiplicative factor converting stored thickness values
    to meters.

An optional bias field may also exist, though this is usually
zero:

``H_bias_si``

The model may also apply a small positive floor, conceptually:

``H_floor``
    Numerical safety floor for thickness in meters.

Why thickness is physically important
-------------------------------------

Thickness is not a cosmetic auxiliary feature.

In GeoPrior’s settlement and timescale logic, thickness
affects:

- equilibrium settlement
  :math:`s_{eq}(h) \approx S_s \,\Delta h \, H`;
- effective drainage thickness
  :math:`H_d`;
- and therefore the prior timescale
  :math:`\tau_{phys}`.

Because the closure timescale scales roughly with
:math:`H_d^2`, thickness mis-specification can strongly
distort the physics branch. This is one reason ``H_field``
and related metadata deserve careful auditing.

Coordinate contract
===================

The coordinate tensor ``coords`` is central to GeoPrior.

The physics branch differentiates with respect to the same
coordinate tensor that was passed into the forward call. So
the coordinate contract must remain consistent across:

- Stage-1 export,
- Stage-2 and Stage-3 training,
- Stage-4 inference,
- and any direct Python-side debugging.

The key coordinate settings in the scaling payload are
typically:

- ``coord_order``
- ``coords_normalized``
- ``coords_in_degrees``
- ``coord_ranges``
- ``coord_ranges_si``
- and, when needed, ``deg_to_m_lon`` / ``deg_to_m_lat``.

Coordinate order
----------------

The usual order is:

.. code-block:: text

   ["t", "x", "y"]

but GeoPrior allows this to be declared explicitly.

A wrong coordinate order can silently corrupt derivative
meaning. For example, swapping ``x`` and ``t`` does not
necessarily create a shape error, but it makes the physics
residuals meaningless.

Projected coordinates vs lon/lat
================================

GeoPrior supports two broad spatial coordinate modes.

Projected planar coordinates
----------------------------

This is the recommended mode.

If the coordinates are already in a projected CRS such as UTM,
then the usual configuration is:

.. code-block:: text

   coords_in_degrees = False

In this case, spatial derivatives can be interpreted directly
in meters after any normalization-related chain-rule scaling
has been applied.

Longitude / latitude coordinates
--------------------------------

If the coordinates are stored in degrees, then GeoPrior needs
explicit conversion factors:

- ``deg_to_m_lon``
- ``deg_to_m_lat``

and the scaling payload should declare:

.. code-block:: text

   coords_in_degrees = True

.. warning::

   Do not evaluate spatial PDE residuals on raw degree
   coordinates without a meters-per-degree conversion.

   The resulting residual scale will be physically wrong.

Normalized coordinates and chain-rule correction
================================================

Many GeoPrior datasets use normalized coordinates internally.

If the scaling payload declares:

.. code-block:: text

   coords_normalized = True

then the derivatives computed by autodiff are with respect to
the **normalized** coordinates, not the physical ones.

GeoPrior therefore applies a chain-rule correction using the
coordinate spans.

If, for example,

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

This becomes even more important for second derivatives and
divergence terms, where the span enters quadratically.

Coordinate span fields
----------------------

The most common span metadata are:

``coord_ranges``
    Coordinate spans in dataset units.

``coord_ranges_si``
    Preferred when available; coordinate spans already
    expressed in SI units.

A typical payload may contain something like:

.. code-block:: json

   {
     "coords_normalized": true,
     "coord_order": ["t", "x", "y"],
     "coord_ranges": {"t": 7.0, "x": 44494.875, "y": 39275.0},
     "time_units": "year"
   }

A useful rule is:

- if ``coords_normalized`` is false, the coordinate spans
  still matter less, but the units must still be correct;
- if ``coords_normalized`` is true, the spans become
  absolutely central to the physics interpretation.

Time units
==========

Time handling is one of the most sensitive parts of the
GeoPrior physics contract.

The key field is:

``time_units``
    Name of the time unit used in the coordinate tensor,
    such as ``"year"``, ``"month"``, ``"day"``, or
    ``"second"``.

Although the model may receive time in annual steps or other
dataset-native units, the physics core interprets temporal
derivatives in **seconds**.

Examples
--------

If the time coordinate increments by one per year, then the
correct declaration is:

.. code-block:: text

   time_units = "year"

If it increments by one per day:

.. code-block:: text

   time_units = "day"

If it is already expressed in seconds:

.. code-block:: text

   time_units = "second"

.. warning::

   A wrong ``time_units`` declaration silently corrupts
   :math:`\partial_t` and therefore affects both the
   groundwater and consolidation residuals.

This is one of the easiest ways to create apparently “bad
physics” from otherwise reasonable data.

The scaling payload
===================

GeoPrior uses a single audit-friendly configuration payload,
usually called:

.. code-block:: text

   scaling_kwargs

This payload is central to the model definition. The scaling
code itself describes it as the single most important
configuration for GeoPrior, because it defines how raw
dataset variables, coordinates, groundwater semantics,
physics closures, priors, bounds, and training policies are
interpreted and resolved. It is wrapped and serialized
through :class:`~geoprior.models.subsidence.scaling.GeoPriorScalingConfig`.

A helpful summary is:

- it governs **what the data mean**,
- not only **how the data are scaled**.

Typical fields in ``scaling_kwargs``
------------------------------------

Common payload fields include:

- ``time_units``
- ``coord_order``
- ``coords_normalized``
- ``coords_in_degrees``
- ``coord_ranges`` / ``coord_ranges_si``
- ``gwl_kind``
- ``gwl_sign``
- ``use_head_proxy``
- ``subsidence_kind``
- ``subs_scale_si``
- ``head_scale_si``
- ``H_scale_si``
- bounds-related metadata
- physics toggles such as
  ``allow_subs_residual`` or diagnostics settings

A minimal test-style example used in the current model tests
looks like:

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

Why serialization matters
-------------------------

GeoPrior serializes the scaling payload because a reloaded
model must preserve the same physical interpretation it had
during training.

If the scaling configuration is lost or altered, the saved
model may still reload successfully but no longer correspond
to the same effective physics.

That is why the scaling payload should be treated as part of
the **scientific identity** of the run.

Feature-index metadata
======================

GeoPrior often needs to know which dynamic channels correspond
to which physical roles.

Useful metadata may include:

- ``dynamic_feature_names``
- ``gwl_dyn_index``
- ``subs_dyn_index``

These are especially helpful for:

- drawdown diagnostics,
- consistent logging,
- manifest checks,
- and stage-to-stage handshake validation.

A run may still execute without every feature name recorded,
but the workflow becomes much easier to trust when feature
semantics are explicit rather than implicit.

Forcing term conventions
========================

GeoPrior’s groundwater residual can include a forcing term
:math:`Q_{term}`:

.. math::

   R_{gw}
   =
   S_s\,\partial_t h
   -
   \nabla\cdot(K\nabla h)
   -
   Q_{term}.

The important point is that the residual expects
:math:`Q_{term}` in **inverse-time units** compatible with the
groundwater equation used internally.

Common metadata fields may include:

- ``Q_in_per_second``
- ``Q_in_si``
- or more explicit forcing-type fields in extended payloads.

A safe practical rule is:

- declare the forcing meaning explicitly,
- and make sure the residual conversion matches that meaning.

If forcing semantics are ambiguous, keep the forcing branch
simple or disabled until the interpretation is clear.

Audit files and stage-level checking
====================================

The GeoPrior workflow is designed to expose these conventions
through stage-level audits and manifests.

Stage-1
-------

Stage-1 records the main preprocessing and scaling contract
through:

- manifest metadata,
- exported feature lists,
- split summaries,
- and scaling-side audit information.

Use this stage to verify:

- coordinate columns,
- unit assumptions,
- target semantics,
- thickness availability,
- and feature ordering.

Stage-2
-------

Stage-2 uses the Stage-1 contract as the authoritative input
boundary and checks:

- tensor shape agreement,
- feature dimension agreement,
- city / run identity,
- scaling-payload compatibility,
- and coordinate readiness for the physics core.

If a Stage-2 run shows physics anomalies, the first question
should be:

- did Stage-1 record the intended data-and-units contract?

A day-one stability checklist
=============================

Before trusting physics outputs, check the following.

Coordinates
-----------

- ``coord_order`` matches the actual tensor layout
- if ``coords_in_degrees=True``, meters-per-degree factors
  are present
- if ``coords_normalized=True``, coordinate spans are present
  and correct

Time
----

- ``time_units`` matches the meaning of the time axis
- the time increments are actually what you think they are

Targets
-------

- ``subs_scale_si`` converts stored subsidence values to
  meters
- if groundwater is depth-like, surface elevation metadata
  are present
- head and drawdown signs are consistent with the chosen
  closure

Thickness
---------

- ``H_field`` is positive after scaling
- thickness is expressed in meters inside the physics core
- thickness floors are not hiding a fundamentally bad input

Feature semantics
-----------------

- dynamic and future feature orders are explicit
- groundwater-related feature positions are known when needed
- the target-domain manifest matches the run being analyzed

If this checklist is satisfied, then later issues are much
more likely to be:

- architectural,
- optimization-related,
- or identifiability-related

rather than basic physics-meaning errors.

Common mistakes
===============

**Using the wrong time unit**

This silently corrupts temporal derivatives.

**Treating raw degrees as meters**

This distorts spatial derivatives and the groundwater
residual.

**Ignoring head vs depth-to-water semantics**

This makes drawdown and compaction physically ambiguous.

**Using thickness in the wrong unit**

This distorts both equilibrium settlement and the timescale
closure.

**Letting feature order drift between runs**

This is one of the easiest ways to break stage-to-stage
consistency without an obvious crash.

**Treating ``scaling_kwargs`` as optional decoration**

It is not optional. It is part of the model contract.

A compact data-and-units map
============================

The GeoPrior data-and-units story can be summarized as:

.. code-block:: text

   raw dataset values
        ↓
   scaling_kwargs defines meaning
        ↓
   coords define derivative frame
        ↓
   head / drawdown / thickness are mapped to SI
        ↓
   residuals are assembled in SI-consistent space
        ↓
   diagnostics are interpretable only if the contract is right

See also
--------

.. seealso::

   - :doc:`geoprior_subsnet`
   - :doc:`physics_formulation`
   - :doc:`residual_assembly`
   - :doc:`losses_and_training`
   - :doc:`scaling`
   - :doc:`identifiability`
   - :doc:`../user_guide/stage1`
   - :doc:`../user_guide/stage2`