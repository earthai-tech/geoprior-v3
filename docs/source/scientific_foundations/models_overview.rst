Models overview
===============

GeoPrior-v3 currently exposes a compact but expressive
physics-guided model family for **land subsidence
forecasting**.

At the highest level, the model line is organized around two
public classes:

- :class:`~geoprior.models.GeoPriorSubsNet`
- :class:`~geoprior.models.PoroElasticSubsNet`

These are not unrelated architectures. They belong to the
same scientific family and share the same broad modeling
contract, but they serve slightly different scientific
purposes.

The goal of this page is to give a conceptual overview of the
model family before the later pages go deeper into the
physics formulation, residual assembly, scaling logic, and
identifiability machinery.

What the model family is designed to do
---------------------------------------

GeoPrior models are built for **multi-horizon subsidence
forecasting under physical constraints**.

More specifically, they are designed to combine:

- sequence forecasting over a future horizon,
- groundwater-aware latent dynamics,
- a physics-guided consolidation pathway,
- learnable effective physical fields,
- and explicit residual-based regularization.

This places the model family at the intersection of three
ideas:

1. **attentive multi-horizon forecasting**
2. **physics-informed residual learning**
3. **poroelastic / consolidation-aware geoscientific
   modeling**

The project documentation and model docstrings describe this
family as a prior-regularized, physics-guided forecasting
framework in which groundwater and subsidence are learned
jointly, while effective fields such as
:math:`K(x,y)`, :math:`S_s(x,y)`, :math:`\tau(x,y)`, and
:math:`Q(t,x,y)` are inferred and regularized during
training.

The two public models
---------------------

GeoPriorSubsNet
~~~~~~~~~~~~~~~

:class:`~geoprior.models.GeoPriorSubsNet` is the **general
coupled model**.

It is the main model when you want to couple:

- groundwater-flow behavior,
- consolidation / subsidence relaxation,
- learned physical fields,
- and supervised forecasting over a horizon.

Its default scientific posture is the most general one in the
current family:

- it supports coupled physics via ``pde_mode="both"``,
- it predicts both subsidence and groundwater outputs,
- it supports quantile forecasting,
- it uses a shared attentive encoder-decoder backbone,
- and it adds physics losses inside a custom training loop.

The model docstring explicitly frames it as a
**prior-regularized physics-informed network for multi-step
subsidence forecasting with groundwater coupling**.

PoroElasticSubsNet
~~~~~~~~~~~~~~~~~~

:class:`~geoprior.models.PoroElasticSubsNet` is the
**consolidation-first preset**.

It is not a separate architecture invented from scratch.
Instead, it is a specialized preset layered on top of
GeoPriorSubsNet for cases where the consolidation dynamics
are better constrained than the full groundwater PDE.

In practice, this variant changes the scientific default
regime by:

- setting ``pde_mode='consolidation'`` by default,
- enabling effective drainage thickness by default,
- using ``hd_factor < 1`` as a default prior convention,
- injecting missing physical bounds conservatively,
- and compiling with stronger prior and bounds defaults.

This makes it useful when:

- the groundwater forcing is poorly constrained,
- the head observations are noisy or sparse,
- you want a stable poroelastic baseline,
- or you are studying identifiability and prior structure
  before enabling the full coupled PDE.

A useful summary is:

- **GeoPriorSubsNet** = general coupled model
- **PoroElasticSubsNet** = consolidation-first strong-prior
  preset

One family, not two unrelated systems
-------------------------------------

Although the two models differ in their scientific defaults,
they are intentionally close.

They share the same broad contract:

- the same dict-input API,
- the same supervised output families,
- the same horizon-based forecasting logic,
- the same quantile layout when enabled,
- the same physics-aware scaling machinery,
- and the same general diagnostics pathway.

This is important for the documentation structure. The later
pages do not need to document two totally separate systems.
Instead, they can explain one common model family and then
highlight where the poroelastic preset changes the default
assumptions.

The forecasting backbone
------------------------

The current model family is built on top of an attentive
sequence backbone rather than a purely feedforward network.

In the code, :class:`~geoprior.models.GeoPriorSubsNet`
inherits from a shared attentive base and uses a hybrid
encoder-decoder path that can include:

- static features,
- dynamic historical features,
- future-known covariates,
- attention blocks,
- LSTM-based or hybrid temporal encoding,
- and optional variable-selection logic.

The broader intent is to combine **modern multi-horizon time
series forecasting structure** with a physics-guided head,
rather than replacing forecasting architecture with a purely
PDE-only design.

This is why the model family naturally connects to the
forecasting literature as well as the PINN literature.

What the models predict
-----------------------

The public output contract is stable and intentionally simple.

The two supervised output families are:

- ``subs_pred`` for subsidence forecasts
- ``gwl_pred`` for groundwater-level forecasts

These are the main user-facing outputs returned by the Keras
forward call.

When quantiles are enabled, the model can emit multi-quantile
forecast tensors instead of only central predictions. The
quantile head is centered around a physics-guided mean path,
so uncertainty is modeled around a physically informed
baseline rather than around an unconstrained black-box
forecast.

This is one of the main conceptual differences from a
standard multi-horizon neural forecaster.

The dict-input contract
-----------------------

The GeoPrior model family follows a consistent dict-input
API. A typical batch includes:

- ``static_features`` of shape ``(B, S)``
- ``dynamic_features`` of shape ``(B, T_in, D)``
- ``future_features`` of shape ``(B, T_out, F)``
- ``coords`` of shape ``(B, T_out, 3)``

and, when needed for the physics closure, a thickness-like
field such as ``H_field``.

The coordinate order is expected to follow
``['t', 'x', 'y']`` unless explicitly overridden by the
scaling configuration.

This matters because the physics pathway differentiates with
respect to these coordinates and then converts derivatives
into SI-consistent forms.

The learned physical fields
---------------------------

A defining feature of GeoPrior is that it does not only learn
forecast outputs.

It also learns **effective physical fields**, typically:

- :math:`K(x,y)` — hydraulic conductivity
- :math:`S_s(x,y)` — specific storage
- :math:`\tau(x,y)` — consolidation timescale
- :math:`Q(t,x,y)` — groundwater source / forcing term

These are not treated as laboratory constants. They are
effective, grid-scale fields produced by the physics head and
mapped into physically meaningful ranges through the scaling
and bounds logic.

This is one reason the model family is especially relevant
for geoscientific inverse-learning and interpretation tasks.

The core physics residuals
--------------------------

The current model family is organized around two main
residual families, plus several regularization terms.

Groundwater flow residual
~~~~~~~~~~~~~~~~~~~~~~~~~

The groundwater-flow residual is written conceptually as:

.. math::

   R_{gw}
   =
   S_s \,\partial_t h
   -
   \nabla \cdot (K \nabla h)
   -
   Q

This is the main flow-style PDE residual used when
groundwater physics is active.

Consolidation residual
~~~~~~~~~~~~~~~~~~~~~~

The consolidation pathway uses a relaxation-style closure:

.. math::

   R_{cons}
   =
   \partial_t s
   -
   \frac{s_{eq}(h) - s}{\tau}

with a common equilibrium approximation of the form:

.. math::

   s_{eq}(h)
   \approx
   S_s \,\Delta h \, H

This is what gives the model its poroelastic /
consolidation-aware character.

Additional regularization terms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond these two residuals, GeoPrior also uses additional
physics-oriented regularization, including:

- a **timescale prior** tying learned :math:`\tau` to a
  closure-based prior,
- a **smoothness prior** on learned fields,
- a **bounds residual** for field plausibility,
- and an optional **m_v prior** linking storage to
  compressibility and water unit weight.

This means the model family is more than a “PINN with one
equation.” It is a structured residual system with multiple
scientific constraints.

The timescale prior
-------------------

One of the most important model ideas is the prior on the
relaxation timescale :math:`\tau`.

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
- and :math:`\tau` is the effective consolidation timescale.

The prior is commonly enforced in log space, which helps
stabilize training and makes it easier to express
identifiability-oriented constraints.

This is one of the most important scientific ideas in the
entire model family, because it links learned fields to a
physically interpretable time scale.

Why scaling is central to the model family
------------------------------------------

GeoPrior does not treat scaling as a small preprocessing
detail.

The scaling configuration is part of the model contract
itself. It governs:

- coordinate normalization,
- time-unit interpretation,
- groundwater conventions,
- SI conversion,
- bounds,
- and field naming / alias consistency.

That is why the code uses a dedicated
:class:`~geoprior.models.subsidence.scaling.GeoPriorScalingConfig`
object that is Keras-serializable and preserved across model
saving and loading.

In practical terms, this means:

- you do not fully understand a GeoPrior model if you ignore
  ``scaling_kwargs``,
- and a saved model is not fully interpretable without its
  scaling context.

Identifiability is a model feature, not an afterthought
-------------------------------------------------------

Another distinctive feature of the current model family is
that **identifiability control is explicit**.

The code exposes named identifiability regimes such as:

- ``base``
- ``anchored``
- ``closure_locked``
- ``data_relaxed``

These regimes exist because the poroelastic closure contains
ridge-like parameter trade-offs. For example, different
combinations of :math:`K`, :math:`S_s`, :math:`H_d`, and
:math:`\tau` can explain similar behavior unless the problem
is constrained carefully.

Rather than hiding that issue, GeoPrior builds it into the
public model design.

This is scientifically important because it makes the model
family suitable not only for prediction, but also for
controlled inverse-learning and ablation studies.

The role of diagnostics
-----------------------

The model family also exposes diagnostics as part of the
public scientific interface.

In addition to supervised outputs, the codebase supports:

- auxiliary forward outputs,
- epsilon-style residual summaries,
- physics payload export,
- physical-parameter summaries,
- and identifiability-oriented diagnostics.

This is one of the reasons the GeoPrior model family works
well inside a staged workflow: the models are designed not
only to train, but also to emit artifacts that can be
audited, exported, and interpreted scientifically.

How to choose between the two models
------------------------------------

A practical choice rule is:

Choose :class:`~geoprior.models.GeoPriorSubsNet` when
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- you want the most general current model;
- you intend to use coupled groundwater and consolidation
  physics;
- you have reasonable confidence in forcing proxies,
  coordinate handling, and data coverage;
- you want the default flagship model for subsidence
  forecasting.

Choose :class:`~geoprior.models.PoroElasticSubsNet` when
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- you want a stronger prior-driven baseline;
- you trust the consolidation closure more than the full
  groundwater PDE;
- you are doing ablation, debugging, or identifiability
  studies;
- you want a stable consolidation-first regime before turning
  on the full coupled system.

This is often a more useful distinction than “complex model”
versus “simple model.” Both are scientifically meaningful,
but they emphasize different physical assumptions.

A conceptual model map
----------------------

The current model family can be summarized like this:

.. code-block:: text

   GeoPrior model family
        |
        +-- GeoPriorSubsNet
        |     general coupled subsidence + groundwater model
        |     - predicts subsidence and GWL
        |     - supports quantiles
        |     - supports groundwater + consolidation residuals
        |     - learns K, Ss, tau, Q
        |
        +-- PoroElasticSubsNet
              consolidation-first preset
              - same broad architecture
              - groundwater residual off by default
              - stronger prior / bounds defaults
              - effective drainage thickness enabled by default

Relationship to the rest of the scientific section
--------------------------------------------------

This page is only the conceptual entry point.

The rest of the scientific foundations section explains the
model family from different angles:

- :doc:`geoprior_subsnet`
  gives the main reference page for the flagship model;

- :doc:`physics_formulation`
  explains the scientific equations and closures more
  formally;

- :doc:`poroelastic_background`
  clarifies the consolidation-first interpretation that
  motivates the poroelastic preset;

- :doc:`residual_assembly`
  explains how the residual families are computed and packed;

- :doc:`losses_and_training`
  explains how supervised and physics losses are assembled;

- :doc:`data_and_units`
  explains the variable semantics and unit conventions;

- :doc:`scaling`
  explains the role of ``scaling_kwargs`` and the scaling
  configuration object;

- :doc:`identifiability`
  explains the identifiability regimes and why they matter.

See also
--------

.. seealso::

   - :doc:`geoprior_subsnet`
   - :doc:`physics_formulation`
   - :doc:`poroelastic_background`
   - :doc:`residual_assembly`
   - :doc:`losses_and_training`
   - :doc:`data_and_units`
   - :doc:`scaling`
   - :doc:`identifiability`