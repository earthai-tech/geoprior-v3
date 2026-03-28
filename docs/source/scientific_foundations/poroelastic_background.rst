.. _geoprior_poroelastic_background:

Poroelastic background
======================

GeoPrior-v3 is not a full continuum poromechanics solver.
Its flagship models are forecasting systems with
physics-guided structure. Even so, the scientific logic of
the current subsidence formulation is strongly motivated by
**poroelastic / consolidation-style thinking**.

This page explains that background.

The goal is not to reproduce the full theory of poroelastic
media. Instead, it is to clarify the physical intuition that
motivates the GeoPrior closure:

- groundwater decline changes effective stress,
- effective stress drives compaction,
- compaction evolves over a finite timescale,
- and that timescale depends on hydrogeological properties
  and drainage geometry.

This is the scientific foundation behind the
consolidation-first regime in GeoPrior and especially behind
:class:`~geoprior.models.PoroElasticSubsNet`.

Why this background matters
---------------------------

Land subsidence linked to groundwater extraction is not just
a statistical co-occurrence. In many aquifer systems, it is
the result of **hydro-mechanical coupling**: groundwater
decline changes pore pressure, the effective stress state of
compressible sediments changes, and the geologic system
compacts over time. Regional studies and groundwater-compaction
models have long emphasized this connection in subsiding
aquifer systems :cite:p:`GallowayBurbey2011,Hoffmannetal2000,Ellisetal2023`.

GeoPrior does not attempt to solve the full coupled mechanics
of a saturated deformable porous medium. Instead, it adopts a
forecasting-oriented surrogate that preserves the most
important causal structure:

.. code-block:: text

   groundwater / head decline
        ↓
   drawdown and stress change
        ↓
   tendency toward equilibrium compaction
        ↓
   delayed subsidence response over a finite timescale

That is the essential poroelastic idea carried into the
GeoPrior model family.

From pore pressure to effective stress
--------------------------------------

The classical physical intuition begins with **effective
stress**.

In a porous medium, the solid skeleton does not respond only
to total stress. It responds to the part of the stress that
is effectively carried by the grain framework after pore
pressure is accounted for. When pore pressure decreases,
effective stress increases, and compressible layers may
compact. This is the broad logic behind effective-stress
interpretations of groundwater-related settlement and
compaction :cite:p:`Bennethumetal1997,GallowayBurbey2011`.

In the simplest conceptual form:

- falling pore pressure
  :math:`\Rightarrow` rising effective stress
- rising effective stress
  :math:`\Rightarrow` compaction of compressible layers
- compaction integrated through the column
  :math:`\Rightarrow` measurable land subsidence

GeoPrior does not model full stress tensors directly, but it
preserves this causal chain through a head-driven settlement
closure.

Why head decline matters
------------------------

Groundwater models are often expressed in terms of hydraulic
head, and GeoPrior follows that convention internally.

Let:

.. math::

   h(t,x,y)

denote hydraulic head, or a head-like proxy that is mapped
consistently into meters inside the physics core.

A decline in head relative to a reference state can be
interpreted as **drawdown**:

.. math::

   \Delta h_{raw} = h_{ref} - h.

GeoPrior then uses a non-negative drawdown form, conceptually

.. math::

   \Delta h = [\Delta h_{raw}]_{+},

so that compaction is driven by head loss rather than by
head rise.

This is a practical but scientifically meaningful choice:
for the current application family, the model is intended to
learn the compaction side of groundwater depletion rather
than uplift due to the reverse sign of the same closure.

From drawdown to equilibrium settlement
---------------------------------------

The next scientific step is to connect drawdown to a
settlement tendency.

GeoPrior uses an effective equilibrium settlement
approximation of the form:

.. math::

   s_{eq}(h)
   \approx
   S_s\,\Delta h\,H,

where:

- :math:`S_s` is effective specific storage,
- :math:`\Delta h` is non-negative drawdown,
- :math:`H` is an effective compressible thickness.

This is not meant to be a full derivation of every
hydro-mechanical detail. It is an **effective closure** that
encodes a widely used physical intuition:

- more drawdown tends to cause more compaction,
- more compressible thickness tends to amplify settlement,
- higher effective storage / compressibility increases the
  settlement response.

The units are also consistent:

.. math::

   (1/\mathrm{m}) \cdot (\mathrm{m}) \cdot (\mathrm{m})
   = \mathrm{m}.

So even though the closure is simple, it remains
dimensionally meaningful and interpretable.

Why subsidence is delayed
-------------------------

A central reason poroelastic thinking matters is that
subsidence is often **delayed** relative to the forcing that
drives it.

In many real systems, the ground surface does not respond
instantaneously to drawdown. Compression and drainage can
unfold over months, years, or longer. This time-lagged
behavior is one of the reasons compaction models and
aquifer-system subsidence packages remain important in
hydrogeology and land-subsidence analysis
:cite:p:`Hoffmannetal2000,Ellisetal2023,ZapataNorbertoetal2025`.

GeoPrior captures this delayed response through a relaxation
timescale :math:`\tau` rather than through a full mechanical
stress-strain PDE.

The relaxation interpretation
-----------------------------

GeoPrior models the evolution of subsidence through a
relaxation-style ODE:

.. math::

   \frac{\partial s}{\partial t}
   =
   \frac{s_{eq}(h)-s}{\tau}.

This says that:

- :math:`s_{eq}(h)` is the settlement state suggested by the
  current hydrogeological forcing,
- :math:`s` is the current subsidence state,
- :math:`\tau` controls how quickly the system relaxes toward
  equilibrium.

This is one of the most important scientific simplifications
in GeoPrior.

Instead of directly solving a full coupled consolidation PDE,
the model uses a **forecasting-compatible effective law**
that still preserves the main poroelastic intuition: the
settlement response is delayed, state-dependent, and linked
to hydrogeological properties.

A useful interpretation of :math:`\tau`
---------------------------------------

The relaxation time :math:`\tau` should be interpreted as an
**effective drainage-compaction timescale**.

It absorbs several pieces of hydro-mechanical behavior,
including:

- how quickly pressure perturbations dissipate,
- how compressible the system is,
- how large the effective drainage path is,
- and how much unresolved structure is being represented at
  the grid scale.

This is why GeoPrior treats :math:`\tau` as a learned field
rather than as one global constant.

At the same time, GeoPrior does **not** let :math:`\tau`
float arbitrarily. It anchors it to a closure-based prior,
which is one of the strongest scientific features of the
model family.

Why drainage thickness matters
------------------------------

The consolidation timescale should depend on how difficult it
is for the system to drain.

GeoPrior encodes this through an effective drainage thickness
:math:`H_d`, often related to the main compressible thickness
:math:`H` by a factor such as

.. math::

   H_d = f_{Hd} \, H.

The central physical idea is simple:

- thicker drainage paths tend to imply slower equilibration,
- and therefore longer consolidation timescales.

This matters because the current GeoPrior prior scales
approximately like:

.. math::

   \tau_{phys}
   \approx
   \frac{H_d^2\,S_s}{\pi^2\,\kappa\,K},

where:

- :math:`H_d` is effective drainage thickness,
- :math:`S_s` is specific storage,
- :math:`K` is hydraulic conductivity,
- :math:`\kappa` is a consistency factor.

So thickness enters quadratically into the timescale prior.
That is one reason thickness handling is scientifically
important in GeoPrior rather than being a minor auxiliary
input.

How GeoPrior uses the timescale prior
-------------------------------------

GeoPrior uses the closure timescale as a **physical anchor**
for the learned :math:`\tau` field.

Conceptually, the model decomposes timescale behavior into:

- a closure-driven baseline,
- plus a learned residual adjustment.

This is expressed in log space through a prior mismatch such
as:

.. math::

   R_{prior}
   =
   \log(\tau_{learned})
   -
   \log(\tau_{phys}).

This choice is important for two reasons.

**1. It improves numerical stability**

Timescales can vary by orders of magnitude. Log-space
penalties are often better behaved than raw absolute
differences.

**2. It improves interpretability**

Instead of learning a completely free :math:`\tau`, the model
learns a timescale that remains tied to hydrogeological
fields and drainage structure.

Why this is called “poroelastic background”
-------------------------------------------

Strictly speaking, GeoPrior is not implementing the full
classical poroelastic continuum equations.

It is called *poroelastic background* because the model keeps
the **core physical narrative** of poroelastic compaction:

- pressure decline changes the effective state of the porous
  skeleton,
- compaction depends on storage-like and thickness-like
  properties,
- the response is delayed and drainage-controlled,
- and the resulting settlement can be studied through a
  physically meaningful time constant.

This makes the poroelastic label scientifically appropriate
for the closure, even though the implemented model is an
effective surrogate rather than a full geomechanical solver.
Related work on PINN-style surrogates for heterogeneous
poroelastic media highlights the value of this middle ground
between strict numerical simulation and unconstrained data
fitting :cite:p:`Roy2024,Royetal2024HeteroPoroelasticPINNs`.

What GeoPrior keeps from classical theory
-----------------------------------------

GeoPrior preserves several core ideas from poroelastic and
consolidation-style reasoning:

- head decline matters physically,
- settlement depends on drawdown and compressible thickness,
- response is delayed rather than instantaneous,
- storage and conductivity jointly affect timescale,
- drainage geometry matters,
- physically implausible field values should be constrained.

These are exactly the ingredients that make the consolidation
branch scientifically useful.

What GeoPrior simplifies
------------------------

GeoPrior also simplifies aggressively compared with a full
continuum poromechanics model.

It does **not** explicitly solve:

- full stress-strain tensor mechanics,
- elastic-plastic constitutive behavior,
- full vertical layering mechanics at continuum resolution,
- all boundary-condition details of real aquifer systems,
- exact forcing histories everywhere in space and time.

Instead, it uses:

- an effective settlement equilibrium,
- an effective relaxation timescale,
- learned effective fields,
- and residual penalties inside a forecasting model.

This is not a weakness by itself. It is a deliberate design
choice that makes the model more trainable and more useful
for forecasting in settings where classical simulation inputs
are incomplete or uncertain.

Why the consolidation-first preset is useful
--------------------------------------------

This background explains why
:class:`~geoprior.models.PoroElasticSubsNet` is a meaningful
preset rather than a cosmetic variant.

In some datasets:

- groundwater boundaries are poorly known,
- pumping or recharge proxies are uncertain,
- coordinate scaling is fragile,
- or derivative noise makes the full flow residual unstable
  early in training.

In those cases, the **consolidation-first** view is often the
best physically grounded starting point.

That is exactly the role of PoroElasticSubsNet:

- keep the same general architecture,
- keep the same output contract,
- but emphasize the poroelastic relaxation pathway more
  strongly than the full groundwater PDE.

This makes it especially useful for:

- ablation studies,
- identifiability analysis,
- early-stage debugging,
- prior-driven subsidence forecasting baselines.

Poroelastic interpretation of the learned fields
------------------------------------------------

Under the GeoPrior formulation, the learned fields can be
read through a poroelastic lens.

:math:`K(x,y)`
    Controls how easily head perturbations propagate and
    therefore influences the characteristic response time.

:math:`S_s(x,y)`
    Controls storage-like response and contributes both to
    groundwater dynamics and to equilibrium settlement.

:math:`\tau(x,y)`
    Encodes the delayed hydro-mechanical response time.

:math:`H` or :math:`H_d`
    Encodes how large the effective compressible / drainage
    column is.

This interpretation is one reason the model can be useful for
parameter-oriented scientific analysis rather than only for
forecast production.

How this connects to observed land subsidence
---------------------------------------------

The practical relevance of this closure is clear in real
subsidence systems, where sustained groundwater extraction is
often accompanied by measurable regional settlement,
infrastructure damage, and reduction in aquifer storage
capacity :cite:p:`GallowayBurbey2011,Hasanetal2023,Bagherietal2021`.

GeoPrior does not claim to resolve every process behind these
observations. Rather, it provides an effective forecasting
framework in which the dominant compaction pathway remains
physically interpretable.

That is especially useful for modern subsidence forecasting,
where the goal is often not only to fit historical data, but
also to understand how future settlement trajectories may
respond under hydrogeological constraints.

A compact conceptual map
------------------------

The poroelastic background of GeoPrior can be summarized as:

.. code-block:: text

   head decline
       ↓
   non-negative drawdown
       ↓
   equilibrium settlement proxy: Ss * Δh * H
       ↓
   delayed relaxation toward settlement equilibrium
       ↓
   timescale controlled by K, Ss, and Hd
       ↓
   prior-regularized learning of tau and related fields

This is the core physical story behind the consolidation side
of the model family.

What this page does not replace
-------------------------------

This page gives the scientific intuition, but it does not
replace the more technical pages that explain:

- the exact GeoPrior residual forms,
- the training-time loss construction,
- the scaling and SI conversion logic,
- and the identifiability controls.

Those pages should be read next once the physical intuition
here is clear.

See also
--------

.. seealso::

   - :doc:`geoprior_subsnet`
   - :doc:`physics_formulation`
   - :doc:`residual_assembly`
   - :doc:`losses_and_training`
   - :doc:`data_and_units`
   - :doc:`scaling`
   - :doc:`identifiability`