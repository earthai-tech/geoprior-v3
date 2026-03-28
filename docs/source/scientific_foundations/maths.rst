.. _geoprior_maths:

===========================================
GeoPrior maths: definitions and conventions
===========================================

:API reference:
   - :mod:`~geoprior.models.subsidence.maths`
   - :func:`~geoprior.models.subsidence.maths.compose_physics_fields`
   - :func:`~geoprior.models.subsidence.maths.tau_phys_from_fields`
   - :func:`~geoprior.models.subsidence.maths.equilibrium_compaction_si`
   - :func:`~geoprior.models.subsidence.maths.compute_mv_prior`
   - :func:`~geoprior.models.subsidence.maths.q_to_gw_source_term_si`
   - :func:`~geoprior.models.subsidence.maths.compute_bounds_residual`

This page defines the mathematical objects used by the
GeoPrior physics core: effective physical fields, settlement
closures, timescale priors, groundwater forcing conversion,
bounds penalties, residual scaling, and epsilon-style
diagnostics.

It complements the other scientific pages in a specific way:

- :doc:`physics_formulation` explains the governing
  scientific logic;
- :doc:`residual_assembly` explains how those objects become
  residual maps and loss terms;
- this page defines the **mathematical objects themselves**
  as GeoPrior uses them.

GeoPrior sits at the intersection of multi-horizon
forecasting and physics-guided surrogate modeling. That means
its math layer must remain both **scientifically meaningful**
and **numerically stable**, especially for groundwater-driven
subsidence where drawdown, compaction, and delayed response
interact across scales :cite:p:`GallowayBurbey2011,Ellisetal2023,Donnelly2023,Roy2024`.

Notation and SI conventions
===========================

GeoPrior couples two main state variables:

.. math::

   h(t,x,y) \quad \text{[m]}

.. math::

   s(t,x,y) \quad \text{[m]}

where:

- :math:`h` is hydraulic head, or a head-like proxy treated
  consistently as meters inside the physics core;
- :math:`s` is land subsidence / settlement in meters.

The coordinate system is:

.. math::

   t \quad \text{[s internally in the physics core]}

.. math::

   x,y \quad \text{[m internally in the physics core]}

The main learned effective physical fields are:

.. math::

   K(x,y) > 0 \quad \text{[m/s]}

.. math::

   S_s(x,y) > 0 \quad \text{[1/m]}

.. math::

   \tau(x,y) > 0 \quad \text{[s]}

.. math::

   Q_{term}(t,x,y) \quad \text{[1/s in the residual form]}.

These are **effective grid-scale fields**, not laboratory
constants. This is central to the GeoPrior formulation: the
model learns physically interpretable but coarse-grained
quantities that can explain observed subsidence and
groundwater dynamics without pretending to resolve all
subsurface heterogeneity :cite:p:`Roy2024,Royetal2024HeteroPoroelasticPINNs`.

A useful compact view
---------------------

.. code-block:: text

   model outputs h and s
       +
   physics head outputs raw latent channels
       ↓
   maths helpers convert them to K, Ss, tau, Q, Hd
       ↓
   residual assembly turns them into physical constraints

This page focuses on the **conversion and definition** step.

Effective fields from raw physics channels
==========================================

GeoPrior does not treat the raw physics head outputs as
physical fields directly.

Instead, the physics head emits base channels, conceptually:

- :math:`\ell_K^{base}`
- :math:`\ell_{S_s}^{base}`
- :math:`\Delta \log\tau`
- and optional raw forcing channels for :math:`Q`

which are then transformed into physical quantities by
:func:`~geoprior.models.subsidence.maths.compose_physics_fields`.

A useful conceptual summary is:

.. math::

   K = \exp(\ell_K)

.. math::

   S_s = \exp(\ell_{S_s})

.. math::

   \log\tau = \log\tau_{phys} + \Delta\log\tau

with positivity, bounds, floors, and stability guards applied
during composition.

Why logs are used
-----------------

The GeoPrior math layer uses log-space parameterization for
several important fields because:

- conductivity, storage, and timescales are positive;
- they often vary across orders of magnitude;
- log-space bounds are easier to interpret physically;
- optimization is usually more stable in log space.

This is especially important for:

- :math:`K`
- :math:`S_s`
- :math:`\tau`

which can otherwise become numerically fragile when optimized
directly in linear space.

Coordinate-aware corrections
----------------------------

GeoPrior can also add spatially varying corrections through
small coordinate-aware components evaluated on
time-zero coordinates. Conceptually, with
:math:`\tilde{\mathbf{c}} = (0,x,y)`:

.. math::

   \ell_K
   =
   \ell_K^{base}(t,x,y)
   +
   \Delta \ell_K(\tilde{\mathbf{c}})

.. math::

   \ell_{S_s}
   =
   \ell_{S_s}^{base}(t,x,y)
   +
   \Delta \ell_{S_s}(\tilde{\mathbf{c}})

This design encourages the effective fields to remain
primarily **spatial properties**, rather than arbitrary
time-varying noise absorbers.

The closure timescale :math:`\tau_{phys}`
=========================================

One of the most important quantities in GeoPrior is the
closure-based physical timescale returned by
:func:`~geoprior.models.subsidence.maths.tau_phys_from_fields`.

The implementation computes it in log space first, then
exponentiates only at the end if needed. This is a deliberate
stability choice because naive algebraic forms can behave
poorly when :math:`K` becomes small.

Effective drainage thickness
----------------------------

GeoPrior defines an effective drainage thickness
:math:`H_d` as either:

.. math::

   H_d = H \cdot f_{Hd}

when ``use_effective_thickness=True``, or

.. math::

   H_d = H

otherwise.

Here:

- :math:`H` is the compressible thickness field;
- :math:`f_{Hd}` is the configured drainage-thickness factor.

This is a pragmatic but physically meaningful knob that
allows the timescale prior to reflect unresolved drainage
geometry.

Two closure branches
--------------------

GeoPrior currently supports two main timescale closure
branches via ``kappa_mode``.

Bar mode
~~~~~~~~

When ``kappa_mode="bar"``, the closure is:

.. math::

   \tau_{phys}
   =
   \frac{\kappa \, H^2 \, S_s}{\pi^2 \, K}

with log form:

.. math::

   \log(\tau_{phys})
   =
   \log(\kappa)
   + 2\log(H)
   + \log(S_s)
   - \log(\pi^2)
   - \log(K).

Non-bar branch
~~~~~~~~~~~~~~

For the other branch, GeoPrior uses:

.. math::

   \tau_{phys}
   =
   \frac{H_d^2 \, S_s}{\pi^2 \, \kappa \, K}

with log form:

.. math::

   \log(\tau_{phys})
   =
   2\log(H_d)
   + \log(S_s)
   - \log(\pi^2)
   - \log(\kappa)
   - \log(K).

The second branch is often the more intuitive one because it
makes the effective drainage thickness explicit.

Why the closure is computed in log space
----------------------------------------

A log-first implementation is important because it reduces
the chance of unstable gradients from inverse powers of
conductivity or storage. The code effectively computes:

.. math::

   \log\tau_{phys}
   \rightarrow
   \tau_{phys} = \exp(\log\tau_{phys})

rather than directly assembling a potentially stiff algebraic
expression.

This is a good example of a general GeoPrior design rule:

- the math should remain physically meaningful,
- but it should also be implemented in a numerically stable
  way.

The learned timescale
=====================

GeoPrior does not learn :math:`\tau` completely freely.

Instead, it uses a closure-plus-residual form:

.. math::

   \log\tau
   =
   \log\tau_{phys}
   +
   \Delta\log\tau

and then

.. math::

   \tau
   =
   \exp(\log\tau) + \varepsilon_{\tau},

where :math:`\varepsilon_{\tau}` is a small positive floor for
stability.

This means the learned timescale is interpreted as:

- a physically anchored baseline,
- plus a learned correction for real-world mismatch.

This is one of the most important mathematical ideas in the
whole model family because it ties flexibility to
interpretability.

Equilibrium settlement
======================

GeoPrior uses an effective settlement equilibrium computed by
:func:`~geoprior.models.subsidence.maths.equilibrium_compaction_si`.

The core approximation is:

.. math::

   s_{eq}(h)
   \approx
   S_s\,\Delta h\,H.

The units are consistent:

.. math::

   (1/\mathrm{m})\cdot(\mathrm{m})\cdot(\mathrm{m})
   = \mathrm{m}.

This is not meant to be a full constitutive law for every
sedimentary basin. It is an effective closure that preserves
the key physical idea:

- more drawdown,
- more compressible thickness,
- and larger effective storage

should all tend to increase equilibrium settlement.

Drawdown definition
-------------------

GeoPrior builds compaction from drawdown rather than from the
raw groundwater variable directly.

The raw drawdown is formed by one of two rules:

.. math::

   \Delta h_{raw}
   =
   h_{ref} - h_{mean}

or

.. math::

   \Delta h_{raw}
   =
   h_{mean} - h_{ref}

depending on ``drawdown_rule``.

This is important because some datasets encode groundwater as
head-like variables, while others encode depth-like variables
that require the opposite sign convention.

Positive-part drawdown
----------------------

After the raw drawdown is formed, GeoPrior applies a
non-negative gate, conceptually:

.. math::

   \Delta h = [\Delta h_{raw}]_{+}.

The exact positive-part operator depends on
``drawdown_mode`` and may be implemented as:

- no gate,
- hard ReLU,
- softplus,
- or a smooth ReLU approximation.

This is an important numerical and scientific detail.
Compaction should respond to **head loss**, not to arbitrary
signed oscillations.

Reference-gradient control
--------------------------

GeoPrior can also stop gradients through the reference head:

.. math::

   h_{ref} := \operatorname{stop\_gradient}(h_{ref})

when ``stop_grad_ref=True``.

This prevents the network from trivially reducing drawdown by
moving the reference instead of learning a meaningful head
field. It is a small but important mathematical safeguard.

The exact-step consolidation update
===================================

The equilibrium settlement is not used in isolation. It feeds
the relaxation dynamics of subsidence.

Starting from the ODE

.. math::

   \frac{ds}{dt}
   =
   \frac{s_{eq} - s}{\tau},

GeoPrior uses a step-consistent exact update over a time step
:math:`\Delta t`:

.. math::

   s_{k+1}
   =
   s_k
   +
   (s_{eq,k} - s_k)\left(1 - e^{-\Delta t/\tau_k}\right).

So the physics-implied settlement increment is:

.. math::

   \Delta s_k^{phys}
   =
   (s_{eq,k} - s_k)\left(1 - e^{-\Delta t/\tau_k}\right).

This exact-step form is central to GeoPrior’s math because it
is much better behaved at coarse temporal resolution than a
naive Euler-style approximation :cite:p:`ZapataNorbertoetal2025`.

Why this matters
----------------

At annual or otherwise coarse time steps, directly estimating
:math:`\partial_t s` from noisy predictions can be unstable.
The exact-step formulation keeps the relaxation law explicit
while remaining compatible with sequence forecasting.

This is one of the clearest examples of GeoPrior’s general
mathematical philosophy:

- use physically interpretable equations,
- but implement them in the most stable forecast-compatible
  form available.

The :math:`m_v` prior
=====================

GeoPrior optionally adds a prior tying effective specific
storage to compressibility through water unit weight:

.. math::

   S_s \approx m_v \,\gamma_w.

This is implemented by
:func:`~geoprior.models.subsidence.maths.compute_mv_prior`.

The residual is formed in log space:

.. math::

   r
   =
   \log(S_s) - \log(m_v\,\gamma_w).

When used as a loss, GeoPrior penalizes both:

- the mean residual,
- and the dispersion of that residual.

A conceptual form is:

.. math::

   \bar{r} = \operatorname{mean}(r)

.. math::

   L_g = \operatorname{Huber}(\bar{r};\delta)

.. math::

   L_d = \operatorname{mean}
   \left(
   \operatorname{Huber}(r-\bar{r};\delta)
   \right)

.. math::

   L_{mv}
   =
   L_g + \alpha_{disp} L_d.

This is more robust than a plain MSE in log space because it
tolerates moderate mismatch while still discouraging large
and spatially inconsistent departures.

Gradient-flow modes
-------------------

The :math:`m_v` prior can operate in different modes that
change where gradients are allowed:

``calibrate``
    Stop gradients through :math:`S_s`; calibrate the prior
    without reshaping the learned storage field.

``field``
    Allow gradients through :math:`S_s` itself.

``logss``
    Allow gradients through :math:`\log S_s`, which is often
    more stable than the ``field`` mode.

This is a good example of how the GeoPrior math layer is not
only about equations, but also about **gradient semantics**.

Groundwater forcing conversion
==============================

GeoPrior supports an optional forcing term in the groundwater
residual through
:func:`~geoprior.models.subsidence.maths.q_to_gw_source_term_si`.

The groundwater residual expects:

.. math::

   R_{gw}
   =
   S_s\,\partial_t h
   -
   \nabla\cdot(K\nabla h)
   -
   Q_{term}

with :math:`Q_{term}` expressed in **inverse-time units**
compatible with the residual.

Supported ``Q_kind`` modes
--------------------------

GeoPrior can interpret the raw forcing channel in several
ways.

``per_volume``
~~~~~~~~~~~~~~

The raw forcing is already an inverse-time quantity. After
any normalized-time correction, it is converted to
:math:`1/\mathrm{s}` if needed.

``recharge_rate``
~~~~~~~~~~~~~~~~~

The raw forcing represents a length-rate recharge
:math:`R` in :math:`\mathrm{m}/\mathrm{s}` (or per model time
unit), and the residual source term becomes:

.. math::

   Q_{term}
   =
   \frac{R}{H_{safe}}.

``head_rate``
~~~~~~~~~~~~~

The raw forcing represents a head-rate forcing
:math:`q_h`, and the residual source term becomes:

.. math::

   Q_{term}
   =
   S_s\,q_h.

This is mathematically important because it keeps the
groundwater residual dimensionally consistent regardless of
how the network’s forcing head was parameterized.

Normalized-time correction
--------------------------

If coordinates are normalized, GeoPrior can also apply a
time-range correction before converting to SI seconds.

This mirrors the same chain-rule idea used for temporal
derivatives:

- the raw model output may live in normalized time;
- the residual source term must live in the physical time
  system.

This is another reason the math layer and the scaling layer
are inseparable in GeoPrior.

Bounds residuals
================

GeoPrior supports both:

- **field mapping bounds**
- and **soft differentiable bounds penalties**

The latter are implemented by
:func:`~geoprior.models.subsidence.maths.compute_bounds_residual`.

For a scalar variable :math:`z` with bounds
:math:`z_{min} \le z \le z_{max}`, define the non-negative
violation:

.. math::

   v(z)
   =
   \max(z_{min} - z, 0)
   +
   \max(z - z_{max}, 0).

The normalized residual is:

.. math::

   R(z)
   =
   \frac{v(z)}
   {\max(z_{max} - z_{min}, \varepsilon)}.

This form is used for:

- thickness :math:`H` in linear space,
- :math:`K`, :math:`S_s`, and :math:`\tau` in log space.

Why the residual is normalized
------------------------------

Range-normalization matters because otherwise the penalty for
a narrow-range variable and a wide-range variable would have
very different magnitudes for reasons unrelated to scientific
importance.

GeoPrior uses the normalized residual so that the bounds
penalty becomes more comparable across parameters.

Soft vs hard bounds
-------------------

Conceptually, GeoPrior supports two broad policies:

``bounds_mode="hard"``
    parameters are clipped into the feasible region during
    field construction;

``bounds_mode="soft"``
    raw values are preserved and penalized by a differentiable
    bounds residual.

The soft mode is often scientifically preferable during
training because it preserves information about *how far*
outside the plausible region a field wants to go.

Residual scaling and nondimensionalization
==========================================

GeoPrior can scale residuals before they become losses and
diagnostics.

For a residual :math:`R` and characteristic scale :math:`c`,
the scaled residual is conceptually:

.. math::

   R^{*}
   =
   \frac{R}{\max(c, c_{min})}.

This is important for two reasons:

- it improves training stability,
- it makes physics weights more transferable across sites.

The current implementation uses this idea particularly for
the consolidation and groundwater residual branches.

Raw vs scaled epsilons
----------------------

GeoPrior reports RMS-style diagnostics called epsilons. A
useful distinction is:

Raw epsilon
~~~~~~~~~~~

.. math::

   \epsilon_{raw}
   =
   \sqrt{\mathbb{E}[R^2]}

Scaled epsilon
~~~~~~~~~~~~~~

.. math::

   \epsilon
   =
   \sqrt{\mathbb{E}[(R^{*})^2]}.

This is why the diagnostics page distinguishes keys such as:

- ``epsilon_cons_raw``
- ``epsilon_cons``
- ``epsilon_gw_raw``
- ``epsilon_gw``
- ``epsilon_prior``

The raw versions are closer to physical-scale mismatch; the
scaled versions are closer to optimization-scale mismatch.

A practical interpretation rule
-------------------------------

If the raw epsilons are large but the scaled epsilons are
moderate, that does **not** necessarily mean something is
wrong. It may simply mean the residual scaling is doing its
job.

But if both raw and scaled epsilons remain large or unstable,
then the user should inspect:

- units,
- coordinate ranges,
- time conversion,
- bounds,
- or identifiability settings.

A compact math map
==================

The GeoPrior math layer can be summarized as:

.. code-block:: text

   raw physics channels
        ↓
   compose_physics_fields(...)
        ↓
   K, Ss, tau, Q, Hd, tau_phys
        ↓
   equilibrium_compaction_si(...)
        ↓
   exact-step consolidation update
        ↓
   q_to_gw_source_term_si(...)
        ↓
   compute_bounds_residual(...)
        ↓
   residual scaling
        ↓
   raw and scaled epsilon diagnostics

What this page does not replace
===============================

This page defines the mathematical objects, but it does not
replace the companion pages that explain:

- the governing scientific interpretation
  (:doc:`physics_formulation`);
- how these quantities are packed into residual maps
  (:doc:`residual_assembly`);
- how they enter optimization
  (:doc:`losses_and_training`);
- how units and coordinate conventions are declared
  (:doc:`data_and_units`, :doc:`scaling`).

See also
========

.. seealso::

   - :doc:`geoprior_subsnet`
   - :doc:`physics_formulation`
   - :doc:`poroelastic_background`
   - :doc:`residual_assembly`
   - :doc:`losses_and_training`
   - :doc:`data_and_units`
   - :doc:`scaling`
   - :doc:`identifiability`