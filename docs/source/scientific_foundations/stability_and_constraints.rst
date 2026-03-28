.. _geoprior_stability_and_constraints:

=========================
Stability and constraints
=========================

GeoPrior-v3 is a physics-guided forecasting system, not just
a standard neural regressor.

That creates a harder optimization problem than ordinary
supervised learning. The model must simultaneously keep track
of:

- multi-horizon forecast accuracy,
- physically plausible learned fields,
- stable coordinate derivatives,
- closure consistency,
- and interpretable uncertainty behavior.

Without explicit stability and constraint mechanisms, such a
system can fail in predictable ways:

- exponential overflow in positive fields,
- unstable residual magnitudes,
- exploding gradients,
- unphysical parameter drift,
- or visually good predictions with scientifically
  implausible internal states.

GeoPrior therefore includes a **layered stability strategy**
built directly into the model, the physics core, and the
training step.

Why this page matters
---------------------

A physics-guided model can fail even when the equations are
correct.

The practical difficulty is not only *what equations are
used*, but *how those equations are implemented and
optimized* inside a neural forecasting system.

GeoPrior addresses this through a multi-layer design:

1. **safe field construction**
2. **bounds and plausibility constraints**
3. **residual scaling**
4. **training warmup and gates**
5. **gradient sanitization and clipping**
6. **audit and diagnostics signals**

This page explains those mechanisms and how they relate to
the broader scientific logic of the model.

A compact view
--------------

A useful summary is:

.. code-block:: text

   raw neural outputs
        ↓
   constrained physical fields
        ↓
   bounded / scaled residuals
        ↓
   warmup-gated physics loss
        ↓
   sanitized + clipped gradients
        ↓
   stable training and interpretable fields

This is the main stability story in GeoPrior.

The main sources of instability
===============================

GeoPrior’s hardest numerical issues come from a combination
of factors.

Positive physical fields
------------------------

Several learned quantities must stay positive:

- hydraulic conductivity :math:`K`
- specific storage :math:`S_s`
- consolidation timescale :math:`\tau`
- effective thickness-like fields in some branches

Naively exponentiating unrestricted logits can create:

- overflow,
- underflow,
- or extreme gradients.

Coordinate-sensitive derivatives
--------------------------------

The groundwater and smoothness branches depend on autodiff
with respect to the coordinate tensor. If coordinate spans,
time units, or normalization metadata are wrong, residuals
can be off by orders of magnitude.

Residual scale mismatch
-----------------------

Different residual families naturally live on different
scales:

- groundwater flow,
- consolidation relaxation,
- log-timescale prior,
- smoothness,
- bounds,
- and optional forcing regularization.

Without explicit scaling, one branch can dominate the rest
for purely numerical reasons.

Inverse-style ambiguity
-----------------------

GeoPrior learns effective fields, not only outputs. That
means the model can sometimes exploit poorly constrained
parameter directions, creating large field drift without
obvious failure in the supervised loss.

Early-training stiffness
------------------------

Physics residuals are often the least reliable early in
training, because the forecasting backbone has not yet
learned a reasonable state representation. Enforcing physics
too strongly too early can make training unstable.

The GeoPrior stability strategy
===============================

GeoPrior addresses these issues through several coordinated
mechanisms.

The main ones are:

- **logit clamping**
- **safe floors and guarded transforms**
- **hard or soft bounds**
- **residual scaling**
- **warmup and ramp gates**
- **NaN / Inf gradient filtering**
- **global-norm clipping**
- **parameter-specific gradient scaling**
- **physics-off fallback behavior**

Each layer plays a different role.

Safe field construction
=======================

Before residuals are assembled, the raw physics channels are
mapped into physically interpretable fields.

This is already a stability mechanism, not only a semantic
one. The model does not directly trust raw head outputs or
physics logits as physical quantities.

Logit clamping
--------------

GeoPrior explicitly clamps the raw logits used to build
physics fields through a helper like
``clamp_physics_logits(...)``.

Conceptually:

.. code-block:: text

   K_logits, Ss_logits, dlogtau_logits ∈ [clip_min, clip_max]

with a typical default of:

.. code-block:: text

   [-15, 15]

before exponentiation.

This matters because:

.. math::

   \exp(-15) \approx 3\times 10^{-7}

.. math::

   \exp(15) \approx 3\times 10^{6}

So this range is already extremely wide physically, while
still avoiding catastrophic overflow or severe underflow.

The forcing channel :math:`Q` is treated differently because
it is usually linear rather than exponentiated, but GeoPrior
still clamps it to a finite range.

Why clamping matters
--------------------

Without this step, early training can produce logits so large
that:

- positive fields explode,
- timescale priors become meaningless,
- gradients become unstable,
- or one bad batch can corrupt the optimizer state.

Clamping therefore acts as the first stability barrier.

Floors and safe transforms
--------------------------

GeoPrior also uses explicit floors for sensitive physical
quantities.

Examples include:

- thickness floors such as ``H_floor_si``
- epsilon floors in range-normalization
- safe lower bounds before taking logs
- safe denominators in closure formulas

A good example is the thickness safeguard:

.. math::

   H_{safe} = \max(H, H_{floor})

which ensures that settlement and closure terms are never
built on exactly zero or negative thickness.

This may look simple, but it is essential because
:math:`H` enters both:

- equilibrium settlement
- and closure timescale formulas.

Hard and soft constraints
=========================

GeoPrior supports several constraint styles rather than only
one.

The most important distinction is between:

- **hard bounds**
- **soft bounds**

Hard bounds
-----------

In hard-bounds mode, parameters are clipped or projected into
a feasible region during field construction.

Conceptually:

.. math::

   z \leftarrow \operatorname{clip}(z, z_{min}, z_{max})

This approach is strict and can be very stabilizing, but it
can also hide how strongly the model is trying to move
outside the physically plausible range.

Soft bounds
-----------

In soft-bounds mode, the model keeps the raw field value but
adds a differentiable penalty through a bounds residual.

For a variable :math:`z` with bounds
:math:`z_{min} \le z \le z_{max}`, define the violation:

.. math::

   v(z)
   =
   \max(z_{min} - z, 0)
   +
   \max(z - z_{max}, 0)

and the normalized residual:

.. math::

   R(z)
   =
   \frac{v(z)}
   {\max(z_{max} - z_{min}, \varepsilon)}.

GeoPrior uses this logic in
``compute_bounds_residual(...)`` for quantities such as:

- thickness :math:`H` in linear space,
- :math:`K`, :math:`S_s`, and :math:`\tau` in log space.

Why soft bounds are valuable
----------------------------

Soft bounds provide two advantages:

- they preserve information about *how far* outside the
  plausible region the model wants to go;
- they let the optimizer learn its way back instead of
  abruptly flattening the gradient through clipping.

That makes them useful for scientific interpretation, not
only optimization.

Constraint targets
------------------

The main constrained quantities in GeoPrior are:

- :math:`H`
- :math:`K`
- :math:`S_s`
- :math:`\tau`

and, depending on the experiment, other auxiliary physical
parameters such as:

- :math:`m_v`
- :math:`\kappa`

This is a key point: GeoPrior’s constraints are not generic
weight decay. They are **parameter-aware scientific
constraints**.

Residual scaling as a stability mechanism
=========================================

One of the most important stability tools in GeoPrior is
**residual scaling**.

Without it, residual branches with naturally larger raw
magnitudes can dominate the optimization simply because of
unit scale, not because they are scientifically more
important.

GeoPrior therefore forms scaled residuals of the form:

.. math::

   R^{*}
   =
   \frac{R}{\max(c, c_{min})}

where :math:`c` is a characteristic scale and
:math:`c_{min}` is a safety floor.

The main beneficiaries are:

- the consolidation residual,
- the groundwater residual.

Why this helps
--------------

Residual scaling improves stability in two ways.

**1. It keeps the optimization geometry more balanced**

The physics weights become more meaningful across cities and
datasets.

**2. It reduces numerical dominance by one branch**

A large raw residual no longer automatically overwhelms the
rest of the objective if its scale is known and normalized.

Scale sanitization
------------------

GeoPrior also sanitizes the residual scale factors
themselves.

The helper ``sanitize_scales(...)``:

- stops gradients through scale tensors,
- replaces NaN / Inf scales with neutral values,
- clamps scales to a safe range.

This is crucial because a scale factor should stabilize the
loss, not become a new source of gradient pathology.

A useful conceptual form is:

.. math::

   c_{safe}
   =
   \operatorname{clip}
   \left(
   \operatorname{replace\_bad}(c, 1.0),
   c_{min},
   c_{max}
   \right).

Training warmup and gates
=========================

A major source of instability in physics-guided models is
**enforcing physics too strongly too early**.

GeoPrior addresses this with warmup and ramp gates.

Warmup gate
-----------

The helper ``compute_physics_warmup_gate(...)`` returns a
scalar multiplier between 0 and 1 based on optimizer step.

Conceptually:

- during warmup, the gate is 0;
- during ramp, it increases linearly;
- after ramp, it saturates at 1.

A piecewise view is:

.. math::

   g(step) =
   \begin{cases}
   0, & step < w \\
   \frac{step - w}{r}, & w \le step \le w+r \\
   1, & step > w+r
   \end{cases}

where:

- :math:`w` is warmup steps,
- :math:`r` is ramp steps.

This gate is then applied to the scaled physics loss during
training.

Why warmup is scientifically useful
-----------------------------------

Warmup is not only an engineering trick.

It reflects a reasonable scientific training logic:

- first, let the forecasting backbone learn a usable state
  representation from the data;
- then gradually enforce the physical structure on top of
  that representation.

This is especially useful for:

- coupled ``pde_mode="both"`` runs,
- noisy datasets,
- transfer experiments,
- regimes with stronger bounds or prior pressure.

Additional gates
----------------

GeoPrior also uses similar gate ideas for selected branches,
for example the forcing branch :math:`Q`.

A policy helper can control whether a term is:

- always on,
- warmed up,
- or ramped in.

This branch-specific gating is another reason the model can
be flexible without collapsing into an unstable training
problem.

Gradient stability
==================

Once the total loss is assembled, GeoPrior still applies
additional safety checks at the gradient level.

NaN / Inf filtering
-------------------

The helper ``filter_nan_gradients(...)`` replaces NaN or Inf
values in gradients with zeros before applying updates.

This includes support for both:

- dense gradients,
- sparse gradients such as ``IndexedSlices``.

This is extremely useful because a single bad batch should
not automatically destroy the model state.

A useful interpretation is:

- bad forward values are already a problem,
- but bad gradients are the final point where catastrophic
  optimizer corruption can still be prevented.

Global-norm clipping
--------------------

After sanitization, GeoPrior applies global-norm clipping to
the gradients before the optimizer update.

Conceptually:

.. math::

   g \leftarrow \operatorname{clip\_by\_global\_norm}(g, c)

with a default clip value around 1.0 in the current training
step.

This is especially important when:

- residuals spike early,
- stiffness appears in the physics branch,
- or closure-related terms create transient large updates.

Parameter-specific gradient scaling
-----------------------------------

GeoPrior also supports parameter-specific gradient scaling
through helpers like ``_scale_param_grads(...)``.

The main exposed multipliers are:

- ``mv_lr_mult``
- ``kappa_lr_mult``

These do not change the loss value itself. Instead, they
change how aggressively the optimizer updates selected
physics parameters.

This is useful because not all physical parameters learn well
at the same effective step size. Some may require slower or
faster adaptation than the main backbone.

A useful distinction is:

- loss weights decide **what matters**;
- gradient multipliers decide **how fast a parameter reacts**.

Physics-off fallback behavior
=============================

GeoPrior also has a clean way to disable physics constraints
without rewriting the whole training path.

When the model is effectively in:

.. code-block:: text

   pde_mode = "none"

the physics terms are short-circuited or neutralized and the
same train/eval code paths remain usable.

This is an important stability feature because it allows:

- ablation studies,
- pure data-driven bring-up,
- debugging of the forecasting backbone,
- stepwise activation of physics.

A good practical workflow is:

1. verify the supervised path in a physics-off regime;
2. enable a simpler constrained regime;
3. move to stronger coupled physics only after the core
   contract is stable.

Constraint-driven stability in the model family
===============================================

Not all stability mechanisms are purely numerical. Some are
scientific constraints that also improve optimization.

Examples include:

- freezing physics fields over time,
- suppressing free subsidence residual freedom,
- locking selected heads such as ``tau_head``,
- strengthening closure priors,
- switching between barrier-style and residual-style bounds.

These are described more fully in
:doc:`identifiability`, but they belong here too because they
are also part of the model’s stability strategy.

Why identifiability helps stability
-----------------------------------

A model with too many weakly constrained physical degrees of
freedom can drift into unstable or ambiguous solutions.

By constraining certain heads, freezing some branches, or
keeping :math:`\tau` closer to the closure, GeoPrior reduces
the freedom the optimizer can exploit. That can improve both:

- interpretability,
- and numerical stability.

This is an important point: **stability and identifiability
are related**, even though they are not identical.

Audit and debug signals
=======================

GeoPrior is designed so that stability is not hidden.

The diagnostics pathway already exposes several useful
signals, including:

- ``physics_loss_raw``
- ``physics_loss_scaled``
- ``physics_mult``
- ``epsilon_cons_raw``
- ``epsilon_cons``
- ``epsilon_gw_raw``
- ``epsilon_gw``
- ``epsilon_prior``
- bounds-loss terms
- run-level manifests and audit summaries

These let the user diagnose different kinds of instability.

A practical interpretation guide
--------------------------------

If you see:

**Good supervised loss, poor raw epsilons**

this often suggests unit or scaling issues rather than pure
optimizer failure.

**Large raw physics loss, moderate scaled physics loss**

residual scaling may be working correctly.

**Good fit, implausible fields**

the model may need stronger bounds, priors, or a stricter
identifiability regime.

**Early NaN gradients**

the likely causes include:

- bad coordinate scaling,
- extreme logits,
- missing floors,
- too aggressive early physics pressure.

**Very noisy training after enabling physics**

consider:

- lowering the global physics multiplier,
- using a longer warmup/ramp,
- starting from a simpler regime,
- checking bounds and unit contracts.

Common failure modes
====================

Overflow in positive fields
---------------------------

Usually caused by:

- unclamped logits,
- bad initialization,
- or an overly aggressive physics branch.

GeoPrior response:

- clamp logits,
- use log-space parameterization,
- keep explicit floors and bounds.

Residual domination
-------------------

Usually caused by:

- unscaled raw residuals,
- wrong unit declarations,
- or badly mismatched coordinate ranges.

GeoPrior response:

- residual scaling,
- scale sanitization,
- explicit time / coordinate conversion.

Pathological early training
---------------------------

Usually caused by:

- enforcing physics too early,
- or using strong constraints before the backbone is ready.

GeoPrior response:

- warmup and ramp gates,
- physics-off bring-up,
- regime-based constraint control.

Gradient corruption
-------------------

Usually caused by:

- NaNs or Infs in a stiff branch,
- bad derivative scales,
- or unstable exponentials.

GeoPrior response:

- filter bad gradients,
- clip global norm,
- optionally slow sensitive parameters.

Field drift outside plausible ranges
------------------------------------

Usually caused by:

- weak bounds,
- weak priors,
- or too much residual freedom.

GeoPrior response:

- soft or hard bounds,
- stronger closure prior,
- anchored or closure-locked regimes.

A practical stability checklist
===============================

Before trusting a run, check:

- logits are clamped or otherwise safely parameterized;
- ``H_field`` is positive after SI conversion and flooring;
- coordinate order and spans are correct;
- ``time_units`` matches the time axis;
- residual scaling is enabled for stiff PDE branches when
  needed;
- warmup and ramp are long enough for the chosen regime;
- bounds mode matches the scientific goal;
- gradient filtering and clipping remain active;
- raw and scaled epsilon diagnostics are both inspected.

If these items are in good shape, then remaining problems are
more likely to reflect:

- real data mismatch,
- insufficient model capacity,
- or deeper identifiability issues.

A compact stability map
=======================

The GeoPrior stability strategy can be summarized as:

.. code-block:: text

   raw physics channels
        ↓
   clamp logits + apply safe floors
        ↓
   construct bounded physical fields
        ↓
   assemble residuals
        ↓
   sanitize and scale residual magnitudes
        ↓
   apply bounds / priors / gates
        ↓
   compute total loss
        ↓
   scale selected parameter grads
        ↓
   filter NaN / Inf gradients
        ↓
   clip global norm
        ↓
   optimizer step
        ↓
   inspect raw and scaled diagnostics

Relationship to the rest of the docs
====================================

This page explains how GeoPrior stays **trainable and
plausible** while optimizing a hybrid physics-guided model.

The companion pages explain related pieces:

- :doc:`physics_formulation`
  explains the governing physical logic;

- :doc:`residual_assembly`
  explains how the residual bundle is built;

- :doc:`losses_and_training`
  explains how those residuals enter the optimization
  objective;

- :doc:`data_and_units`
  and :doc:`scaling`
  explain the conventions that must be correct before any
  stability mechanism can work;

- :doc:`identifiability`
  explains how regime-based constraints also help control
  ambiguous or unstable parameter freedom.

See also
========

.. seealso::

   - :doc:`physics_formulation`
   - :doc:`residual_assembly`
   - :doc:`losses_and_training`
   - :doc:`data_and_units`
   - :doc:`scaling`
   - :doc:`identifiability`
   - :doc:`../user_guide/diagnostics`