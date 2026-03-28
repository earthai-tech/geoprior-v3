.. _geoprior_residual_assembly:

Residual assembly
=================

GeoPrior-v3 does not treat physics as a single scalar penalty
tacked onto a neural forecast. Instead, it builds a
**structured residual bundle** from the model’s forward pass,
its learned physical fields, the coordinate tensor, and the
resolved scaling configuration.

This page explains how those residuals are assembled.

It is the implementation-facing companion to
:doc:`physics_formulation`: that page explains the scientific
equations, while this page explains how the model turns them
into residual maps, scaled losses, and diagnostic epsilons
inside training and evaluation.

Why residual assembly matters
-----------------------------

In GeoPrior, the physics branch is not only about equations.
It is about turning those equations into a numerically stable,
traceable, and reusable training signal.

A good residual assembly design must answer several practical
questions:

- Which variables are differentiated?
- In what units are derivatives interpreted?
- Which residuals are maps and which are scalar priors?
- How are bounds handled?
- How are residuals scaled before contributing to training?
- How are diagnostics such as
  ``epsilon_cons_raw`` or ``epsilon_gw`` produced?
- How does the model keep the same logic across training and
  evaluation?

GeoPrior answers these questions through a shared physics
pipeline rather than through ad hoc stage-specific code. This
is consistent with the broader PINN and physics-surrogate
literature, where residual design and normalization often
matter as much as the governing equations themselves
:cite:p:`Donnelly2023,Roy2024,Sarmaetal2024IPINNs`.

A high-level view
-----------------

A useful summary is:

.. code-block:: text

   inputs + scaling config
        ↓
   forward pass
        ↓
   map raw physics channels to physical fields
        ↓
   differentiate with respect to coords
        ↓
   convert derivatives to SI-consistent form
        ↓
   assemble residual maps
        ↓
   optionally nondimensionalize residuals
        ↓
   reduce to scalar losses + epsilon diagnostics
        ↓
   apply global physics multiplier and warmup gate

This is the core residual assembly story in GeoPrior.

The shared physics pathway
--------------------------

The residual bundle is assembled in the shared physics core
used by both training and evaluation.

Conceptually, the core does the following:

1. prepare inputs and SI-aware quantities;
2. run ``forward_with_aux(...)``;
3. construct bounded physical fields;
4. compute head derivatives via autodiff;
5. convert derivatives into SI-consistent units;
6. assemble residual maps;
7. optionally scale those residuals;
8. reduce them into losses and diagnostics;
9. apply physics scheduling;
10. pack the physics bundle for logging and export.

This shared implementation is important because it keeps the
physics logic consistent across:

- training,
- evaluation,
- diagnostics,
- and exported physics payloads.

What enters residual assembly
-----------------------------

The residual assembly step depends on four classes of inputs.

**1. Supervised model outputs**

These are the user-facing predictions such as:

- ``subs_pred``
- ``gwl_pred``

**2. Auxiliary forward outputs**

The model’s forward pass also exposes raw intermediate
channels such as:

- ``data_mean_raw``
- ``phys_mean_raw``
- ``phys_features_raw_3d``

These are the latent ingredients used to build the physical
state and field quantities.

**3. Coordinate-aware inputs**

Residuals are differentiated with respect to the forecast
coordinate tensor:

.. math::

   \mathbf{c} = (t, x, y).

This is why the dict-input contract requires ``coords``.
Without coordinates, GeoPrior cannot construct the derivative
terms needed for the groundwater and smoothness branches.

**4. Scaling configuration**

The assembly logic also depends critically on the resolved
``scaling_kwargs`` contract, which determines:

- time-unit conversion,
- coordinate normalization,
- degree-to-meter conversion if needed,
- groundwater semantics,
- drawdown handling,
- physical bounds,
- and residual scaling policy.

This is why residual assembly cannot be understood without
the scaling page.

Stage 1: forward pass and auxiliary tensors
-------------------------------------------

Residual assembly begins with a standard model forward pass,
but GeoPrior uses a richer internal call than a plain output
dict.

Conceptually:

.. code-block:: python

   y_pred, aux = model.forward_with_aux(inputs, training=...)

The two most important things returned here are:

- the supervised predictions,
- the raw physics channels that will later be turned into
  bounded physical fields.

This separation is important. The residual bundle is not
assembled from arbitrary hidden states. It is assembled from
a specific auxiliary forward contract.

Stage 2: composing physical fields
----------------------------------

The raw physics channels are not used directly.

Instead, GeoPrior maps them into bounded, interpretable
physical fields such as:

- :math:`K(x,y)`
- :math:`S_s(x,y)`
- :math:`\tau(x,y)`
- :math:`Q(t,x,y)`
- effective drainage thickness :math:`H_d`
- closure timescale :math:`\tau_{phys}`

This composition step is where several important policies are
applied:

- positivity constraints,
- linear- vs log-space parameterization,
- soft or hard bounds,
- closure-based timescale construction,
- optional forcing conversion.

This is one of the most important design choices in the whole
model. The network does not emit “physics” directly. It emits
raw channels that are then transformed into a physically
meaningful bundle under explicit constraints.

Why this composition step exists
--------------------------------

Without a dedicated composition step, the model could easily
produce physically uninterpretable fields, for example:

- negative conductivity,
- negative specific storage,
- implausible timescales,
- forcing values with the wrong residual units,
- unstable exponentials in float32.

GeoPrior therefore treats field composition as part of the
physics assembly rather than as a post hoc cosmetic step.

This is especially important for:

- :math:`K`
- :math:`S_s`
- :math:`\tau`

which often live most naturally in log space.

Stage 3: derivative computation
-------------------------------

Once the physical fields are available, GeoPrior computes
derivatives of the head-like state with respect to the model
coordinates.

At a conceptual level, the core computes raw derivatives such
as:

.. math::

   \frac{\partial h}{\partial t},
   \qquad
   \frac{\partial h}{\partial x},
   \qquad
   \frac{\partial h}{\partial y}

and the second-derivative or divergence-related quantities
needed by the groundwater branch.

This step uses automatic differentiation with respect to the
same coordinate tensor that was fed into the model.

That is why the coordinates must have the expected shape:

.. code-block:: text

   (B, H, 3)

after coercion and preparation.

Stage 4: SI conversion of derivatives
-------------------------------------

Raw autodiff derivatives are not yet physically meaningful
unless the coordinate system is already expressed in SI form.

GeoPrior therefore converts derivatives into SI-consistent
quantities by applying the chain rule using the resolved
coordinate spans and conventions. This includes cases such
as:

- normalized coordinates that must be rescaled by actual
  spans,
- spatial coordinates originally expressed in degrees,
- time coordinates expressed in years or other non-SI units.

A useful mental summary is:

- autodiff happens in model coordinate space,
- residuals are assembled in SI-consistent space.

This is one of the most important reasons that coordinate and
unit handling are central to GeoPrior rather than peripheral
details.

The residual families
---------------------

After the field composition and derivative conversion steps,
GeoPrior assembles several residual families.

Core map-valued residuals
~~~~~~~~~~~~~~~~~~~~~~~~~

The main residual maps are:

- groundwater flow residual
- consolidation residual
- timescale prior residual
- smoothness residual
- bounds residual

Each serves a different scientific purpose.

Groundwater residual
~~~~~~~~~~~~~~~~~~~~

When the groundwater branch is active, GeoPrior builds:

.. math::

   R_{gw}
   =
   S_s\,\partial_t h
   -
   \nabla\cdot(K \nabla h)
   -
   Q_{term}.

This residual is assembled in SI-consistent units after the
derivative-conversion step.

A key detail is that the implementation works with the
**divergence term** rather than assuming constant
conductivity. That makes the residual compatible with
spatially varying learned :math:`K(x,y)` fields.

Consolidation residual
~~~~~~~~~~~~~~~~~~~~~~

The settlement side uses a **step-consistent** residual rather
than only a naive instantaneous derivative form.

Starting from the relaxation law:

.. math::

   \frac{ds}{dt}
   =
   \frac{s_{eq} - s}{\tau},

GeoPrior computes a one-step physics-implied settlement
increment:

.. math::

   \Delta s_k^{phys}
   =
   (s_{eq,k} - s_k)
   \left(1 - e^{-\Delta t/\tau_k}\right)

and compares it with the predicted settlement increment:

.. math::

   \Delta s_k^{pred}
   =
   s_{k+1}^{pred} - s_k^{pred}.

The step residual is then:

.. math::

   r_k
   =
   \Delta s_k^{pred} - \Delta s_k^{phys}.

GeoPrior can convert this step mismatch into a
rate-consistent consolidation residual map that participates
in the physics loss.

This exact-step design is especially important for coarse
time resolution, where a naive Euler-style differential view
can be unnecessarily noisy or unstable.

Timescale prior residual
~~~~~~~~~~~~~~~~~~~~~~~~

GeoPrior also builds a prior residual tying the learned
timescale to a closure-based physical baseline.

Conceptually:

.. math::

   R_{prior}
   =
   \log(\tau_{learned})
   -
   \log(\tau_{phys}).

This is not just another arbitrary regularizer. It is the
main mechanism that keeps the learned timescale connected to
the learned hydrogeological fields and effective drainage
geometry.

Smoothness residual
~~~~~~~~~~~~~~~~~~~

The smoothness branch regularizes the spatial variability of
the learned fields, especially:

- :math:`K(x,y)`
- :math:`S_s(x,y)`

Conceptually, it penalizes excessive spatial roughness using
gradients of those fields with respect to the spatial
coordinates.

This is important because effective fields should usually be
interpretable and spatially coherent rather than jagged,
pixel-level noise absorbers.

Bounds residual
~~~~~~~~~~~~~~~

Bounds are handled as a dedicated residual family rather than
only as a one-time clipping step.

GeoPrior computes violation residuals for quantities such as:

- thickness :math:`H`
- conductivity :math:`K`
- specific storage :math:`S_s`
- timescale :math:`\tau`

using linear or log-space bounds as appropriate.

These bound-violation residuals can then contribute to the
final physics objective in a controlled way.

Additional scalar priors
------------------------

Not every physics term is a spatial residual map.

GeoPrior can also include additional scalar penalties such as:

- an optional :math:`m_v` consistency prior,
- a forcing regularization term on :math:`Q`.

These terms are assembled alongside the main PDE-style terms
but are conceptually distinct from the map-valued residuals.

This distinction matters because some terms participate in the
global physics multiplier differently from others.

Residual scaling and nondimensionalization
------------------------------------------

Raw residual maps can have very different magnitudes across:

- cities,
- time spans,
- coordinate ranges,
- and field scales.

GeoPrior therefore supports optional residual scaling to form
dimensionless residuals of the form:

.. math::

   R^{*}
   =
   \frac{R}{\max(c, c_{min})},

where :math:`c` is a characteristic scale and :math:`c_{min}`
is a positive floor.

This scaling is applied separately to the consolidation and
groundwater branches when
``scale_pde_residuals=True``.

Why residual scaling matters
----------------------------

Residual scaling improves both:

- numerical stability,
- and transferability of physics weights.

Without it, the meaning of a physics weight such as
``lambda_cons`` or ``lambda_gw`` can change drastically from
one dataset to another simply because the raw residual units
or spans are different.

GeoPrior therefore distinguishes between:

- **raw residuals** for diagnostic interpretation,
- **scaled residuals** for optimization.

The characteristic scales
-------------------------

GeoPrior computes characteristic residual scales from the
current batch using representative physical magnitudes, then
sanitizes them and applies floors to avoid collapse.

Conceptually:

- a consolidation scale is built from the compaction-related
  state and closure magnitudes;
- a groundwater scale is built from storage, derivatives, and
  divergence-related terms.

These scales are treated as stop-gradient quantities so they
stabilize training without introducing unintended gradient
paths.

Scalar reduction to losses
--------------------------

Once the residual maps are assembled and optionally scaled,
GeoPrior reduces them into scalar losses.

For the main map-valued residuals:

.. math::

   L_{cons}
   =
   \mathbb{E}\!\left[(R_{cons}^{*})^2\right]

.. math::

   L_{gw}
   =
   \mathbb{E}\!\left[(R_{gw}^{*})^2\right]

.. math::

   L_{prior}
   =
   \mathbb{E}\!\left[(R_{prior})^2\right]

.. math::

   L_{smooth}
   =
   \mathbb{E}\!\left[(R_{smooth})^2\right]

and similarly for the bounds residual and optional auxiliary
priors.

This is the point where map-valued physical mismatch becomes
a scalar optimization term.

Epsilon diagnostics
-------------------

GeoPrior also turns the residuals into RMS-style diagnostic
quantities called **epsilons**.

For example:

- ``epsilon_cons_raw``
- ``epsilon_gw_raw``
- ``epsilon_cons``
- ``epsilon_gw``
- ``epsilon_prior``

Conceptually, GeoPrior distinguishes:

**raw epsilons**

.. math::

   \epsilon_{raw}
   =
   \sqrt{\mathbb{E}[R^2]}

**scaled epsilons**

.. math::

   \epsilon
   =
   \sqrt{\mathbb{E}[(R^{*})^2]}.

This distinction is essential for interpretation.

- raw epsilons retain the original scale of the residual;
- scaled epsilons are closer to the optimization view.

If raw epsilons are large but scaled epsilons remain moderate,
the scaling system may be doing exactly what it is supposed to
do.

Assembly of the total physics loss
----------------------------------

After the per-term losses are available, GeoPrior assembles
the overall physics loss using the weighted structure

.. math::

   T_{cons}   = \lambda_{cons}   L_{cons}

.. math::

   T_{gw}     = \lambda_{gw}     L_{gw}

.. math::

   T_{prior}  = \lambda_{prior}  L_{prior}

.. math::

   T_{smooth} = \lambda_{smooth} L_{smooth}

.. math::

   T_{bounds} = \lambda_{bounds} L_{bounds}

.. math::

   T_{mv}     = \lambda_{mv}     L_{mv}

.. math::

   T_{q}      = \lambda_{q}      L_{q}.

The PDE-style core is then:

.. math::

   L_{core}
   =
   T_{cons}
   +
   T_{gw}
   +
   T_{prior}
   +
   T_{smooth}
   +
   T_{bounds}.

GeoPrior defines an unscaled physics loss:

.. math::

   L_{phys,raw}
   =
   L_{core} + T_{mv} + T_{q}.

A global multiplier ``phys_mult`` is then applied to produce
the scaled physics loss actually added to the training
objective, with optional separate control over whether the
:math:`m_v` and :math:`Q` terms scale with the same offset.

This separation is one of the reasons the training system is
more flexible than a simple sum of residuals.

Warmup and ramp scheduling
--------------------------

During training, GeoPrior can apply a warmup/ramp gate to the
scaled physics loss.

Conceptually:

- early training focuses more on data fitting,
- the physics contribution ramps in gradually,
- per-term scaled physics contributions are gated together.

This is especially useful when:

- the physics branch is initially noisy,
- coordinates or scales make early derivatives unstable,
- the model needs a short data-driven warm start before
  strong physical coupling is enforced.

This warmup gate is applied only in training-time assembly,
not in the same way during pure evaluation diagnostics.

The packed physics bundle
-------------------------

After loss assembly, GeoPrior packs a canonical physics
bundle containing:

- physics loss values,
- the global multiplier,
- per-term losses,
- epsilon diagnostics,
- optional :math:`Q` regularization summaries,
- and, when requested, the raw or scaled residual maps and
  physical fields.

When ``return_maps=True``, the bundle can include quantities
such as:

- ``K_field``
- ``Ss_field``
- ``tau_field``
- ``tau_phys``
- ``Hd_eff``
- ``H_si``
- ``Q_si``
- ``R_cons``
- ``R_gw``
- ``R_prior``
- ``R_smooth``
- ``R_bounds``
- ``R_cons_scaled``
- ``R_gw_scaled``

This is what makes GeoPrior’s diagnostics and payload export
so rich: the residual assembly step already exposes the
intermediate scientific quantities in a consistent structure.

Physics-off behavior
--------------------

GeoPrior also supports a clean **physics-off** pathway.

When the model indicates that physics is disabled, the shared
core still performs the forward pass and returns predictions
and auxiliary outputs, but the physics bundle is replaced by
a safe empty or zero-style structure.

This allows the same train/eval code paths to operate in:

- full physics mode,
- partial physics mode,
- or purely data-driven mode

without rewriting the training loop from scratch.

Why this design is scientifically useful
----------------------------------------

This residual assembly design has several scientific
advantages.

**1. It is modular**

Each physical idea enters through a distinct residual family.

**2. It is interpretable**

The model can expose which part of the physics objective is
large or small.

**3. It is stable**

Scaling, bounds, and warmup logic make the training problem
more manageable.

**4. It is portable**

The same residual bundle logic works in training, evaluation,
inference diagnostics, and exported payloads.

**5. It is compatible with forecasting**

The system remains usable inside a multi-horizon attentive
forecasting backbone rather than demanding a separate PDE
solver.

Common misconceptions
---------------------

**“Residual assembly is just the equations.”**

No. In GeoPrior, residual assembly also includes:

- field composition,
- derivative conversion,
- bounds policy,
- scaling,
- scalar reduction,
- and diagnostics packing.

**“Scaled residuals replace raw residuals.”**

No. Both matter. Scaled residuals are mainly for
optimization, while raw residuals remain important for
physical interpretation.

**“The physics loss is a single number.”**

Not really. The single number is only the final aggregate.
Underneath it is a structured set of terms with different
roles and different scientific meanings.

A compact assembly map
----------------------

The GeoPrior residual assembly can be summarized as:

.. code-block:: text

   y_pred, aux = forward_with_aux(...)
        ↓
   compose physical fields: K, Ss, tau, Q, tau_phys, Hd
        ↓
   autodiff wrt coords
        ↓
   convert derivatives to SI-consistent form
        ↓
   build R_cons, R_gw, R_prior, R_smooth, R_bounds
        ↓
   optionally scale R_cons and R_gw
        ↓
   reduce to L_cons, L_gw, L_prior, L_smooth, L_bounds, ...
        ↓
   compute epsilon_cons_raw / epsilon_cons / ...
        ↓
   weight by lambda_* and phys_mult
        ↓
   optionally apply warmup gate
        ↓
   pack physics bundle for logs, eval, and export

See also
--------

.. seealso::

   - :doc:`physics_formulation`
   - :doc:`losses_and_training`
   - :doc:`data_and_units`
   - :doc:`scaling`
   - :doc:`maths`
   - :doc:`identifiability`