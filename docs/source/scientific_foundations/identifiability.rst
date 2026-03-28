.. _geoprior_identifiability:

================
Identifiability
================

GeoPrior-v3 treats **identifiability** as part of the model
design, not as a post hoc warning in the discussion section.

This matters because the flagship model does not only predict
subsidence and groundwater trajectories. It also learns
effective physical fields such as:

- :math:`K(x,y)`
- :math:`S_s(x,y)`
- :math:`\tau(x,y)`
- :math:`H_d(x,y)`

and uses them inside a coupled residual system. In problems
like this, multiple parameter combinations can often explain
similar observed behavior. That is a classic inverse-problem
difficulty, and it remains central in physics-informed and
poroelastic surrogate models :cite:p:`Donnelly2023,Roy2024,Sarmaetal2024IPINNs,Royetal2024HeteroPoroelasticPINNs`.

GeoPrior addresses this directly by exposing explicit
**identifiability regimes**, **parameter locks**, **closure
anchors**, and **diagnostic summaries**.

Why identifiability matters here
--------------------------------

A physics-guided model can fit observations well while still
learning physically ambiguous or weakly constrained internal
fields.

For example, in the GeoPrior closure

.. math::

   \tau_{phys}
   \approx
   \frac{H_d^2 \, S_s}{\pi^2 \, \kappa \, K},

different combinations of:

- :math:`K`
- :math:`S_s`
- :math:`H_d`
- :math:`\tau`

can produce similar effective timescales.

This creates **ridge-like trade-offs** in parameter space.
The supervised outputs may still look good, but the inferred
fields can become hard to interpret if the problem is not
constrained carefully.

That is the core identifiability issue in GeoPrior.

A useful intuition
------------------

A practical way to think about the problem is:

.. code-block:: text

   data fit alone
        ↓
   can constrain observable trajectories
        ↓
   but may not uniquely constrain K, Ss, tau, Hd
        ↓
   so the model needs explicit identifiability structure

This is why GeoPrior has a dedicated identifiability layer.

Where identifiability enters the model
--------------------------------------

Identifiability in GeoPrior is not handled in only one place.

It enters through:

- the **closure design** itself,
- the **scaling and bounds contract**,
- the **compile-time weight defaults**,
- the **training warmup and ramp schedules**,
- the **ability to freeze or lock selected heads**,
- and the **diagnostic audit layer**.

So the identifiability story is distributed across:

- model construction,
- scaling kwargs,
- compile defaults,
- training behavior,
- and post hoc diagnostics.

This is why the identifiability page belongs in the
scientific foundations section rather than in a narrow
debugging appendix.

The main source of non-identifiability
======================================

The most important structural source of non-identifiability
in GeoPrior is the interaction between:

- the closure-based timescale,
- the compaction equilibrium,
- and the learned physical fields.

Timescale ridge
---------------

The closure prior says, conceptually:

.. math::

   \tau_{phys}
   \propto
   \frac{H_d^2 S_s}{K}.

So increasing :math:`H_d`, increasing :math:`S_s`, or
decreasing :math:`K` can all lengthen the implied timescale.
Without additional constraints, several different field
combinations may generate similar effective dynamics.

Settlement ridge
----------------

The equilibrium settlement proxy is:

.. math::

   s_{eq}(h)
   \approx
   S_s \, \Delta h \, H.

So the same observed compaction tendency can also be traded
off among:

- storage-like response,
- drawdown magnitude,
- and thickness.

Together, these relations mean the model can have enough
freedom to fit the forecast targets while still remaining
ambiguous in how it explains them internally.

That is why identifiability must be designed into the model.

The GeoPrior strategy
=====================

GeoPrior uses three main strategies to improve
identifiability.

**1. Constrain by construction**

Use closure-based relationships so parameters are linked
instead of floating independently.

**2. Control training freedom**

Freeze or relax selected parts of the physics branch through
named regimes, locks, and warmup schedules.

**3. Audit what was actually learned**

Export JSON-safe audits and payload-based summaries so the
model’s internal physical interpretation can be inspected
after the run.

This combination is much stronger than only saying “inverse
problems can be ill-posed.”

Identifiability regimes
=======================

GeoPrior exposes identifiability through the public model
argument:

.. code-block:: python

   identifiability_regime="base"

The current model validation and constructor signature accept
four named regimes:

- ``base``
- ``anchored``
- ``closure_locked``
- ``data_relaxed``. 

These are not decorative presets. They change defaults in
both:

- the scaling / closure behavior,
- and the compile-time physics weights. 

How regimes are applied
-----------------------

The identifiability helper
``init_identifiability(...)``:

- normalizes the regime name,
- loads the matching profile,
- merges profile defaults into ``scaling_kwargs``,
- does **not** override user-provided keys,
- ensures the bounds-loss config exists in dict form. 

That “do not override user-provided keys” rule is important.
It means the regime behaves like a **scientific default
profile**, not like a hard-coded override that destroys
experiment-specific settings.

The four built-in regimes
=========================

Base
----

``base`` is the conservative default regime.

Its current profile sets:

- ``freeze_physics_fields_over_time = True``
- ``bounds_loss_kind = "barrier"``
- moderate physics warmup and ramp defaults
- compile defaults such as
  ``lambda_bounds = 1.0`` and
  ``lambda_prior = 1.0``. 

Conceptually, ``base`` says:

- keep the physics fields time-stable,
- apply a standard closure prior,
- and use moderate bounds / prior pressure.

This is the right regime when you want a scientifically
disciplined baseline without very aggressive locking.

Anchored
--------

``anchored`` makes the physics branch substantially more
constrained.

Its profile currently adds or strengthens settings such as:

- ``freeze_physics_fields_over_time = True``
- ``allow_subs_residual = False``
- barrier-style bounds
- stronger bounds and prior penalties
- ``stop_grad_ref = True``
- ``drawdown_zero_at_origin = True``
- shorter warmup but longer ramp
- stronger compile defaults, for example
  ``lambda_bounds = 10.0``,
  ``lambda_prior = 10.0``,
  ``lambda_cons = 50.0``. 

Conceptually, ``anchored`` says:

- keep the closure stable,
- stop the model from using overly flexible subsidence
  residual freedom,
- and strongly discourage parameter drift away from the prior
  structure.

This is useful when you want a more interpretable but less
free inverse solution.

Closure-locked
--------------

``closure_locked`` is the strongest closure-oriented regime in
the current built-in set.

It keeps many of the anchored constraints and additionally
locks the ``tau_head`` by setting:

.. code-block:: text

   "locks": {"tau_head": True}

in the profile. It also uses a very strong prior default,
currently:

.. code-block:: text

   lambda_prior = 100.0. 

Conceptually, this regime says:

- keep the learned timescale extremely close to the closure,
- strongly suppress free timescale drift,
- and interpret the model more like a closure-driven
  hydrogeological surrogate than a free inverse learner.

This is especially useful for:

- ablation studies,
- identifiability experiments,
- synthetic validation,
- and cases where you want to isolate the effect of the
  closure itself.

Data-relaxed
------------

``data_relaxed`` is the least constrained built-in regime.

Its profile currently sets:

- ``freeze_physics_fields_over_time = False``
- ``allow_subs_residual = True``
- a residual-style bounds loss instead of barrier-style
- weaker bounds and prior defaults
- ``stop_grad_ref = False``
- ``drawdown_zero_at_origin = False``
- much longer warmup and ramp
- compile defaults such as
  ``lambda_prior = 0.0`` and
  ``lambda_gw = 0.0``. 

Conceptually, this regime says:

- allow the model more freedom to follow the data,
- relax strong prior control,
- and tolerate greater flexibility in the subsidence pathway.

This is useful for exploratory forecasting or when the user
cares more about predictive fit than about aggressively
constrained internal fields.

A regime comparison map
-----------------------

A useful summary is:

.. code-block:: text

   base
     moderate prior + bounds, time-frozen fields

   anchored
     stronger prior + stronger bounds, no subs residual freedom

   closure_locked
     anchored-like + tau head locked close to closure

   data_relaxed
     weaker prior, more data freedom, slower physics ramp

Why ``tau`` is central
======================

In the current implementation, the most important parameter
for identifiability is usually :math:`\tau`.

That is because :math:`\tau` sits at the intersection of:

- settlement delay,
- hydraulic conductivity,
- storage,
- and effective drainage thickness.

If :math:`\tau` is left completely free, it can absorb
closure mismatch and make the inferred :math:`K` / :math:`S_s`
relationship harder to interpret.

That is why GeoPrior provides:

- closure-based :math:`\tau_{phys}`,
- log-space prior mismatch,
- ``closure_locked`` behavior,
- and even scenario utilities where :math:`K` is **derived**
  from :math:`\tau` instead of learned freely.

Locks and parameter freezing
============================

GeoPrior can apply explicit parameter-head locks through
``apply_ident_locks(...)``.

The current locking logic can freeze any of:

- ``tau_head``
- ``K_head``
- ``Ss_head``

when the active profile requests it. The built-in
``closure_locked`` regime currently freezes ``tau_head``. 

This is scientifically meaningful because it lets the user
separate two different questions:

- *Can the model fit the data with closure-driven fields?*
- *What changes when one field is allowed to move freely?*

That is the essence of a good identifiability ablation.

The tau-only scenario
=====================

The helper module also implements an explicit identifiability
scenario:

.. code-block:: text

   - learn tau only
   - derive K from tau via closure
   - freeze (or fix) Ss and Hd

This is implemented by
``scenario_tau_only_derive_K(...)``. The corresponding
closure is:

.. math::

   \tau
   =
   \frac{H_d^2\,S_s}{\pi^2\,\kappa_b\,K}

so that

.. math::

   K
   =
   \frac{H_d^2\,S_s}{\pi^2\,\kappa_b\,\tau}. 

This is one of the clearest examples of “breaking
non-identifiability ridges by construction,” which is exactly
how the module describes its purpose. 

Why this scenario is useful
---------------------------

The tau-only scenario is especially useful when you want to
ask:

- if :math:`S_s` and :math:`H_d` are fixed or trusted,
  what timescale is the data asking for?
- once that timescale is known, what effective
  conductivity :math:`K` does the closure imply?

This is a much more identifiable question than trying to
learn :math:`K`, :math:`S_s`, :math:`H_d`, and :math:`\tau`
all completely freely from the same observed trajectory.

Compile-time identifiability defaults
=====================================

Identifiability in GeoPrior also enters through the compile
defaults.

The helper ``resolve_compile_weights(...)`` merges the active
profile’s recommended compile weights with any user-specified
values. The selection rule is:

- if the user provides a value explicitly, that wins;
- otherwise the profile default is used;
- otherwise a fallback default is used. 

This is a very good design choice because it keeps the model
scientifically guided without making the built-in regimes
impossible to override.

A useful consequence is that identifiability is not only
about parameter constraints. It is also about **which
residuals are given enough optimization pressure to matter**.

How regimes affect training behavior
====================================

The current profiles do more than change lambdas.

They also change scaling-side and training-side behaviors
such as:

- ``freeze_physics_fields_over_time``
- ``allow_subs_residual``
- ``bounds_loss_kind``
- ``bounds_beta``
- ``bounds_guard``
- ``bounds_w``
- ``bounds_include_tau``
- ``bounds_tau_w``
- ``stop_grad_ref``
- ``drawdown_zero_at_origin``
- ``physics_warmup_steps``
- ``physics_ramp_steps``. 

These settings matter because identifiability is not only
about closed-form algebra. It is also about **how much
freedom the network is allowed to exploit during training**.

Examples
--------

A few especially important examples are:

``freeze_physics_fields_over_time``
    Prevents the model from explaining the same site with
    arbitrarily time-varying physical fields.

``allow_subs_residual = False``
    Reduces freedom in the settlement branch, pushing the
    model to follow the closure more closely.

``stop_grad_ref = True``
    Stops the model from “solving” drawdown inconsistencies by
    moving the reference head itself.

``drawdown_zero_at_origin = True``
    Anchors the drawdown convention more strongly near the
    reference point.

``physics_warmup_steps`` and ``physics_ramp_steps``
    Control how quickly physics pressure becomes active, which
    strongly affects whether the model finds a stable
    interpretable solution or a flexible but ambiguous one.

Identifiability audits
======================

GeoPrior includes a JSON-safe audit helper:

.. code-block:: python

   from geoprior.models.subsidence import ident_audit_dict

The audit records:

- the active regime name,
- whether identifiability is enabled,
- the profile keys used,
- active locks,
- the trainable status of heads such as ``tau_head``,
  ``K_head``, and ``Ss_head``,
- the effective lambda weights,
- selected scaling keys taken from the profile,
- selected scaling keys actually active on the model,
- and the current bounds-loss form. 

This is extremely useful for experiment logs, manifests, and
evaluation JSON because it lets the user confirm what regime
was *actually* active rather than what they intended to run.

.. admonition:: Best practice

   Save the output of ``ident_audit_dict(model)`` for every
   serious identifiability experiment.

   It is one of the cleanest ways to preserve the internal
   scientific posture of the run.

Payload-based identifiability diagnostics
=========================================

GeoPrior also exposes post hoc identifiability diagnostics
through the exported physics payload.

The public helpers include:

- ``identifiability_diagnostics_from_payload(...)``
- ``summarise_effective_params(...)``
- ``derive_K_from_tau_np(...)``. 

These are especially useful for synthetic or controlled
experiments where “true” effective parameters are known.

What the payload diagnostics measure
------------------------------------

The current payload diagnostics explicitly compute three
blocks:

1. **relative error in** :math:`\tau`
2. **log-timescale residual for the closure**
3. **log-offsets of** :math:`K`, :math:`S_s`, and
   :math:`H_d` **relative to true values and priors**. 

That is scientifically meaningful because it separates three
different questions:

- did the model recover the effective timescale?
- did the closure stay consistent with the learned fields?
- how far did the learned physical fields drift away from the
  truth or from the priors?

Summarising effective parameters
--------------------------------

For 1D or synthetic-column studies, the helper
``summarise_effective_params(...)`` collapses 1D arrays of
payload quantities such as:

- ``tau``
- ``tau_prior``
- ``K``
- ``Ss``
- ``Hd``

into representative scalar summaries, currently using the
median of finite values. 

This makes it much easier to compare runs when the real
scientific question is about one effective column-scale
parameter set rather than about a large spatial field.

A good synthetic identifiability workflow
-----------------------------------------

A useful synthetic workflow is:

.. code-block:: text

   train with a chosen identifiability regime
        ↓
   export physics payload
        ↓
   summarise effective params
        ↓
   compare tau, tau_prior, K, Ss, Hd with true and prior values
        ↓
   inspect closure_log_resid and parameter offsets

This is exactly the sort of loop that turns identifiability
from an abstract concern into a measurable property of the
model.

How to think about the regimes scientifically
=============================================

A good conceptual spectrum is:

- **data_relaxed**
  favors predictive freedom;
- **base**
  gives a balanced default;
- **anchored**
  favors interpretable closure-constrained behavior;
- **closure_locked**
  is closest to a closure-dominant identifiability test.

This means the regimes are not merely “more or less strict.”
They are really different scientific stances about **which
degrees of freedom are allowed to explain the data**.

When to use each regime
=======================

Use ``base`` when
-------------------

- you want the default scientific baseline;
- you want moderate prior and bounds pressure;
- you want time-frozen physical fields without very strong
  closure locking.

Use ``anchored`` when
---------------------

- you want stronger physical discipline;
- you want to suppress free subsidence residual behavior;
- you want a more interpretable but less flexible inverse
  solution.

Use ``closure_locked`` when
---------------------------

- you want to test closure-dominant behavior;
- you want :math:`\tau` to remain tightly linked to the
  closure;
- you are running synthetic recovery or ablation studies.

Use ``data_relaxed`` when
-------------------------

- predictive fit matters more than strict physical
  interpretability;
- you want more data freedom;
- you want a softer starting point for exploratory runs.

Common failure modes
====================

Good prediction, poor parameter identifiability
-----------------------------------------------

This is the most common failure mode.

The model can still predict well because the observable state
trajectory is easier to constrain than the internal fields.

A solution is often to:

- move from ``data_relaxed`` to ``base`` or ``anchored``;
- inspect the closure log residual;
- add or strengthen bounds and prior terms;
- freeze more of the physics structure.

Closure looks good, fields still drift
--------------------------------------

This can happen when the closure itself is satisfied but the
field decomposition remains ambiguous.

A solution is often to:

- compare offsets vs true values and vs prior values;
- use a tau-only or partially frozen scenario;
- inspect whether ``freeze_physics_fields_over_time`` should
  be enabled.

Bounds dominate too early
-------------------------

If bounds and prior penalties are too aggressive too early,
the model may become overly rigid before the forecasting
backbone has stabilized.

A solution is often to:

- lengthen warmup and ramp;
- reduce bounds strength;
- move from ``anchored`` to ``base`` first.

Data-relaxed solutions look better but are harder to trust
----------------------------------------------------------

This is expected. More flexibility can improve apparent fit
while reducing interpretability.

That is why identifiability should be reported explicitly,
not inferred from forecast quality alone.

A compact identifiability map
=============================

The GeoPrior identifiability story can be summarized as:

.. code-block:: text

   closure:
     tau ~ Hd^2 * Ss / (pi^2 * kappa * K)
        ↓
   creates ridge-like trade-offs among K, Ss, Hd, tau
        ↓
   identifiability regimes constrain those trade-offs
        ↓
   locks, bounds, priors, and warmups shape training freedom
        ↓
   audit helpers record the active regime
        ↓
   payload diagnostics measure recovery and closure consistency

Relationship to the rest of the docs
====================================

This page explains **how GeoPrior controls and audits
inverse-problem ambiguity**.

The companion pages explain related pieces:

- :doc:`physics_formulation`
  explains the closure and residual system that create the
  identifiability challenge;

- :doc:`residual_assembly`
  explains how those terms are built in the physics core;

- :doc:`losses_and_training`
  explains how the compile-time lambdas and physics multiplier
  affect optimization pressure;

- :doc:`scaling`
  explains the bounds and semantic conventions that support
  identifiability;

- :doc:`../user_guide/stage3`
  and :doc:`../user_guide/stage5`
  are especially relevant when identifiability is studied
  through tuning or transfer experiments.

See also
========

.. seealso::

   - :doc:`geoprior_subsnet`
   - :doc:`physics_formulation`
   - :doc:`residual_assembly`
   - :doc:`losses_and_training`
   - :doc:`data_and_units`
   - :doc:`scaling`
   - :doc:`../user_guide/stage3`
   - :doc:`../user_guide/stage5`