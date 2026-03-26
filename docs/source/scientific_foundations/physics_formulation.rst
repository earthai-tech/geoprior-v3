.. _geoprior_physics_formulation:

Physics formulation
===================

GeoPrior-v3 is not built as a purely black-box forecasting
system. Its flagship model, :class:`~geoprior.models.GeoPriorSubsNet`,
combines a multi-horizon attentive forecasting backbone with a
physics-guided surrogate for groundwater-subsidence coupling.
The forecasting side is conceptually aligned with modern
interpretable multi-horizon sequence models :cite:p:`Limetal2021,Kouadio2025XTFT`,
while the physics side follows the broader philosophy of
physics-informed learning and hydro-mechanical surrogate
modeling :cite:p:`Donnelly2023,Roy2024`.

This page explains the **scientific equations and closures**
used by GeoPrior at the model level. It is the conceptual
bridge between the high-level model description and the
lower-level pages on residual assembly, losses, scaling, and
identifiability.

What GeoPrior is modeling
-------------------------

GeoPrior couples two primary state variables:

- :math:`h(t,x,y)` — hydraulic head, or a head-like proxy
  treated consistently as meters inside the physics core;
- :math:`s(t,x,y)` — land subsidence (settlement), treated in
  meters.

The model takes these states seriously in two ways:

1. it predicts them as supervised outputs over a forecast
   horizon;
2. it constrains them through physics residuals and priors.

This means GeoPrior is neither a pure PDE solver nor a pure
sequence forecaster. It is a **forecasting model with a
structured physical surrogate**.

Scope of the formulation
------------------------

The current formulation is designed for:

- multi-horizon subsidence forecasting,
- groundwater-aware settlement dynamics,
- learned effective physical fields,
- data-constrained residual regularization,
- and site-to-site transfer under explicit scaling and
  identifiability control.

It is important to interpret this as an **effective
grid-scale formulation**, not as a claim that every site
strictly obeys one idealized textbook PDE in all details.
That is why GeoPrior learns effective fields and also allows
closure mismatch to be absorbed through residual terms and
priors.

Primary variables, coordinates, and units
-----------------------------------------

The SI-consistent variables used by the physics core are:

.. math::

   h(t,x,y) \quad \text{[m]}

.. math::

   s(t,x,y) \quad \text{[m]}

Coordinates are:

.. math::

   t \quad \text{[s internally in the physics core]}

.. math::

   x,y \quad \text{[m internally in the physics core]}

The effective learned physical fields are:

.. math::

   K(x,y) > 0 \quad \text{[m/s]}

.. math::

   S_s(x,y) > 0 \quad \text{[1/m]}

.. math::

   \tau(x,y) > 0 \quad \text{[s]}

.. math::

   Q(t,x,y) \quad \text{[1/s in the GeoPrior residual form]}

The code and math guide treat these as **effective fields**
rather than laboratory-scale constants. This choice is
crucial for real geoscientific datasets, where the observed
behavior is often the result of unresolved heterogeneity,
coarse temporal resolution, and incomplete forcing
information. :cite:p:`Roy2024`

A compact conceptual view
-------------------------

A useful summary is:

.. code-block:: text

   forecasting backbone
       ↓
   predicts h and s over the forecast horizon
       ↓
   physics head learns K, Ss, tau, Q
       ↓
   residuals enforce groundwater flow and consolidation
       ↓
   priors, bounds, and scaling stabilize the inverse problem

This is the core scientific idea of GeoPrior.

The groundwater-flow residual
-----------------------------

When groundwater physics is enabled, GeoPrior enforces a
transient groundwater-flow residual in divergence form:

.. math::

   R_{gw}
   =
   S_s\,\frac{\partial h}{\partial t}
   -
   \nabla \cdot \left(K \nabla h\right)
   -
   Q_{term}.

In two spatial dimensions, the divergence term can be written
as:

.. math::

   \nabla \cdot (K \nabla h)
   =
   \frac{\partial}{\partial x}
   \left(K \frac{\partial h}{\partial x}\right)
   +
   \frac{\partial}{\partial y}
   \left(K \frac{\partial h}{\partial y}\right).

The divergence form is important because GeoPrior allows
:math:`K` to vary spatially. In heterogeneous media, this
form is more natural and robust than collapsing everything
into a constant-coefficient diffusion equation. It also
avoids over-reliance on noisy explicit derivatives of
:math:`K(x,y)` when the effective field is learned from data.

Interpretation of the groundwater term
--------------------------------------

Each term in the groundwater residual has a distinct role:

- :math:`S_s \,\partial_t h`
  represents transient storage response;
- :math:`\nabla \cdot (K \nabla h)`
  represents the divergence of hydraulic flux;
- :math:`Q_{term}`
  represents source or sink forcing in the residual form used
  internally by GeoPrior.

A good way to think about this equation is that GeoPrior does
not assume forcing is perfectly known. Instead, the network
can learn an effective :math:`Q` term when that option is
enabled, allowing unresolved recharge, pumping, or other
drivers to be absorbed into a physically structured forcing
channel rather than into arbitrary latent noise.

The consolidation residual
--------------------------

The settlement side is modeled through a relaxation-style
consolidation closure:

.. math::

   R_{cons}
   =
   \frac{\partial s}{\partial t}
   -
   \frac{s_{eq}(h)-s}{\tau}.

This says that subsidence evolves toward an equilibrium
settlement state :math:`s_{eq}(h)` on a learned timescale
:math:`\tau`.

This is one of the central modeling ideas in GeoPrior:

- groundwater and head behavior determine the equilibrium
  tendency,
- while :math:`\tau` determines how quickly subsidence relaxes
  toward that equilibrium.

This relaxation view is especially useful when observations
are available only at coarse time resolution, because it
gives the model a physically interpretable memory mechanism.

Equilibrium settlement
----------------------

GeoPrior uses a simple but physically interpretable
equilibrium settlement approximation:

.. math::

   s_{eq}(h)
   \approx
   S_s\,\Delta h\,H,

where:

- :math:`\Delta h` is drawdown relative to a reference level;
- :math:`H` is a compressible or drained thickness;
- :math:`S_s` is the effective specific storage.

The units are consistent:

.. math::

   (1/\mathrm{m}) \cdot (\mathrm{m}) \cdot (\mathrm{m})
   = \mathrm{m}.

This is a deliberately effective formulation. It is simple
enough to remain trainable and interpretable, but still
captures the central idea that settlement grows with storage,
drawdown, and compressible thickness.

Drawdown definition
-------------------

GeoPrior builds settlement from **non-negative drawdown**.

A raw drawdown can be defined by a rule such as:

.. math::

   \Delta h_{raw}
   =
   h_{ref} - h

or another equivalent sign convention, depending on the
configuration. The physics core then applies a non-negative
gate, conceptually:

.. math::

   \Delta h = [\Delta h_{raw}]_{+}.

In practice, GeoPrior may use a hard ReLU, softplus, or a
smooth approximation depending on the drawdown mode and
stability settings. This avoids unphysical “negative
drawdown-driven compaction” and keeps the subsidence closure
aligned with the intended physical interpretation.

The exact-step consolidation update
-----------------------------------

A key design choice in GeoPrior is that the consolidation
path is implemented in a **step-consistent** way.

Instead of using only a noisy finite-difference estimate of

.. math::

   \frac{\partial s}{\partial t},

the model compares the predicted settlement increment with the
increment implied by the relaxation law over one time step.

Starting from the ODE:

.. math::

   \frac{ds}{dt}
   =
   \frac{s_{eq} - s}{\tau},

the exact one-step solution over a time interval
:math:`\Delta t` is:

.. math::

   s_{k+1}
   =
   s_k
   +
   (s_{eq,k} - s_k)
   \left(1 - e^{-\Delta t/\tau_k}\right).

So the physics-implied increment is:

.. math::

   \Delta s_k^{phys}
   =
   (s_{eq,k} - s_k)
   \left(1 - e^{-\Delta t/\tau_k}\right).

If the model predicts a step increment

.. math::

   \Delta s_k^{pred}
   =
   s_{k+1}^{pred} - s_k^{pred},

GeoPrior forms a step residual:

.. math::

   r_k
   =
   \Delta s_k^{pred} - \Delta s_k^{phys}.

This exact-step construction is important for annual or other
coarse time steps, where naive finite differences can be both
noisy and physically misleading. It also aligns naturally
with one-dimensional subsidence and consolidation-style
simulation logic used in operational land-subsidence studies
:cite:p:`ZapataNorbertoetal2025`.

The learned physical fields
---------------------------

GeoPrior does not only learn :math:`h` and :math:`s`.
It also learns effective physical fields:

.. math::

   K(x,y), \quad S_s(x,y), \quad \tau(x,y), \quad Q(t,x,y).

These are produced by the physics head and then mapped into
positive SI-consistent quantities.

A useful way to think about the architecture is:

- the forecasting backbone predicts the observable state
  trajectories,
- the physics head predicts the hidden effective fields that
  explain those trajectories.

This is what turns GeoPrior into a structured hybrid of
forecasting and inverse learning.

Timescale as “closure + residual”
---------------------------------

GeoPrior does not learn :math:`\tau` completely freely.
Instead, it decomposes it conceptually into:

- a closure-driven physical baseline,
- plus a learned residual correction in log space.

That is,

.. math::

   \log \tau
   =
   \log \tau_{phys}
   +
   \Delta \log \tau.

This is an important design choice.

A fully free :math:`\tau` field can easily absorb too much
model mismatch and make the system hard to interpret. By
anchoring :math:`\tau` to a closure-based baseline and then
learning only a residual adjustment, GeoPrior keeps the
timescale physically meaningful while still allowing
real-world departure from a simple idealized closure.

The timescale closure
---------------------

The closure timescale is derived from the learned fields and
effective thickness. A common GeoPrior form is:

.. math::

   \tau_{phys}
   \approx
   \frac{H_d^2\,S_s}{\pi^2\,\kappa\,K},

where:

- :math:`H_d` is an effective drainage thickness,
- :math:`\kappa` is a consistency scalar,
- :math:`K` is hydraulic conductivity,
- :math:`S_s` is specific storage.

Depending on the chosen closure mode, GeoPrior may use a
closely related branch of this formula, but the central idea
is the same: the relaxation time must remain coupled to the
hydrogeological fields rather than floating freely.

The prior on this closure is expressed in log space as:

.. math::

   R_{prior}
   =
   \log(\tau_{learned})
   -
   \log(\tau_{phys}).

This log-space formulation is more stable numerically and
more meaningful scientifically, because timescales often span
orders of magnitude.

Effective drainage thickness
----------------------------

GeoPrior also introduces an effective drainage thickness
:math:`H_d`, often defined as:

.. math::

   H_d = f_{Hd}\,H,

where :math:`f_{Hd}` is a configurable factor. This is a
pragmatic way to reflect unresolved drainage geometry at
grid-scale resolution.

It is especially important because the closure prior scales
quadratically with thickness. That means thickness
mis-specification can strongly distort the implied timescale,
which is one reason GeoPrior keeps :math:`H`, :math:`H_d`,
and :math:`\tau` tightly linked in the physics branch.

The optional compressibility prior
----------------------------------

GeoPrior can also add an optional prior that links specific
storage to a compressibility-like coefficient:

.. math::

   S_s \approx m_v \,\gamma_w,

where:

- :math:`m_v` is a coefficient of compressibility,
- :math:`\gamma_w` is the unit weight of water.

This prior is also usually enforced in log space, which makes
it easier to stabilize and to interpret as a relative
consistency condition rather than a fragile raw-value
penalty.

Not every run needs this term, but it is useful when the user
wants a stronger physical coupling between learned storage
and compressibility assumptions.

The forcing term
----------------

The groundwater residual includes a forcing term
:math:`Q_{term}`, but GeoPrior treats this carefully.

The residual expects :math:`Q_{term}` in inverse-time units
(:math:`1/\mathrm{s}`), regardless of how the model’s raw
forcing channel was parameterized. The implementation can map
different conceptual forcing meanings into the residual form,
for example:

- already-inverse-time forcing,
- recharge-rate forcing divided by thickness,
- head-rate forcing multiplied by storage.

This matters because a source term that is physically
plausible in one representation may not have the correct
units for the residual. GeoPrior therefore keeps the
conversion explicit rather than hiding it inside an opaque
latent head.

Residual scaling and nondimensionalization
------------------------------------------

One of the most important implementation ideas in GeoPrior is
that raw residual magnitudes can vary dramatically across
datasets, cities, coordinate spans, and time units.

For that reason, GeoPrior can nondimensionalize a residual
map :math:`R` through a characteristic scale :math:`c`:

.. math::

   R^{*}
   =
   \frac{R}{\max(c, c_{min})}.

The physics loss is then built from the scaled residual:

.. math::

   \mathcal{L}
   =
   \mathbb{E}\left[(R^{*})^2\right].

This has two major benefits:

- it improves training stability,
- it makes the relative balance of physics penalties more
  transferable across sites.

This is why the diagnostics page distinguishes **raw
epsilons** from **scaled epsilons**. The scaled values are
closer to the optimization view, while raw values remain
important for physical interpretation.

Residual families used by GeoPrior
----------------------------------

At the model level, the main residual families are:

- groundwater flow,
- consolidation relaxation,
- timescale consistency prior,
- smoothness regularization,
- bounds regularization,
- and, optionally, the :math:`m_v` prior.

So a useful compact formulation is:

.. math::

   \mathcal{L}_{phys}
   =
   \lambda_{gw}\,\mathcal{L}_{gw}
   +
   \lambda_{cons}\,\mathcal{L}_{cons}
   +
   \lambda_{prior}\,\mathcal{L}_{prior}
   +
   \lambda_{smooth}\,\mathcal{L}_{smooth}
   +
   \lambda_{bounds}\,\mathcal{L}_{bounds}
   +
   \lambda_{mv}\,\mathcal{L}_{mv}
   +
   \lambda_{q}\,\mathcal{L}_{q\_reg}.

This is then blended with the supervised data loss under the
model’s overall physics scheduling logic.

Bounds and plausibility
-----------------------

GeoPrior uses bounds in two complementary ways:

1. to stabilize field construction itself,
2. to penalize implausible field values during training.

For a parameter :math:`z` with bounds
:math:`z_{min} \le z \le z_{max}`, a normalized violation
residual can be defined as:

.. math::

   v(z)
   =
   \max(z_{min} - z, 0)
   +
   \max(z - z_{max}, 0),

.. math::

   R(z)
   =
   \frac{v(z)}{\max(z_{max} - z_{min}, \varepsilon)}.

GeoPrior applies this logic directly in physical or log space
depending on the parameter. This is especially important for
log-bounded quantities such as :math:`K`, :math:`S_s`, and
:math:`\tau`, whose magnitudes can otherwise drift into
scientifically implausible ranges.

Physics regimes through ``pde_mode``
------------------------------------

GeoPrior exposes different physics regimes through
``pde_mode``. Conceptually, the main options are:

``'both'``
    Use groundwater and consolidation physics together.

``'consolidation'``
    Use only the consolidation residual.

``'gw_flow'``
    Use only the groundwater residual.

``'none'``
    Disable physics residuals and keep only the supervised
    forecasting path.

This is scientifically useful because not every dataset
supports the same level of physical coupling. In some cases,
the user may want a consolidation-first regime before
activating the full coupled system.

Why this is not a full PDE solver
---------------------------------

GeoPrior is inspired by PINN-style residual learning, but it
should not be mistaken for a classical PDE solver.

Its main purpose is still **forecasting under physical
regularization**. The physics branch acts as a structured
surrogate that encourages interpretable and plausible
behavior, not as a guarantee that the medium satisfies one
exact analytic PDE everywhere.

This is why the model:

- learns effective fields rather than fixed constants,
- allows closure residuals,
- uses priors and bounds,
- and relies on supervised data loss alongside physics loss.

This hybrid design is one of GeoPrior’s main strengths for
real geoscientific forecasting problems, where fully observed
boundary conditions and forcing histories are often not
available :cite:p:`Donnelly2023,Roy2024`.

Relationship to the forecasting backbone
----------------------------------------

GeoPrior’s physical formulation sits on top of a
multi-horizon attentive forecasting core inspired by
interpretable sequence models such as TFT
:cite:p:`Limetal2021`. The point is not to replace sequence
forecasting with PDEs, but to make the sequence forecaster
respect interpretable hydro-mechanical structure.

This is an important distinction:

- the forecasting backbone provides temporal representation
  power,
- the physics formulation provides scientific discipline.

That combination is also consistent with the broader XTFT
line of uncertainty-aware, interpretable, multi-horizon
forecasting in which GeoPrior is situated
:cite:p:`Kouadio2025XTFT`.

What this formulation enables
-----------------------------

Taken together, the GeoPrior formulation enables:

- physically regularized subsidence forecasting,
- learned effective parameter fields,
- uncertainty-aware outputs around a physics-guided mean,
- identifiability-aware scientific constraints,
- portable cross-city inference and transfer workflows,
- and post hoc analysis of physical payloads.

This makes the formulation useful for both:

- **prediction**, and
- **scientific interpretation**.

A compact equation map
----------------------

The GeoPrior physics formulation can be summarized as:

.. math::

   R_{gw}
   =
   S_s\,\partial_t h
   -
   \nabla\cdot(K\nabla h)
   -
   Q_{term}

.. math::

   R_{cons}
   =
   \partial_t s
   -
   \frac{s_{eq}(h)-s}{\tau}

.. math::

   s_{eq}(h)
   \approx
   S_s\,\Delta h\,H

.. math::

   \tau_{phys}
   \approx
   \frac{H_d^2\,S_s}{\pi^2\,\kappa\,K}

.. math::

   R_{prior}
   =
   \log(\tau_{learned})
   -
   \log(\tau_{phys})

with training governed by a weighted combination of
supervised loss and physics-consistency loss.

See also
--------

.. seealso::

   - :doc:`geoprior_subsnet`
   - :doc:`poroelastic_background`
   - :doc:`residual_assembly`
   - :doc:`losses_and_training`
   - :doc:`data_and_units`
   - :doc:`scaling`
   - :doc:`maths`
   - :doc:`identifiability`