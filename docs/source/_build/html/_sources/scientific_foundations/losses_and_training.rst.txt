.. _geoprior_losses_and_training:

Losses and training
===================

GeoPrior-v3 is trained as a **hybrid forecasting and physics
optimization system**.

Its flagship model,
:class:`~geoprior.models.GeoPriorSubsNet`, does not optimize a
single monolithic loss. Instead, it combines:

- supervised forecasting losses,
- multiple physics-consistency losses,
- optional prior and bounds penalties,
- a global physics multiplier,
- and a training-time stability layer with gradient
  sanitization and clipping.

This page explains how those pieces fit together.

It is the natural continuation of
:doc:`residual_assembly`: that page explained how residuals
are built, while this page explains how they become training
objectives and how the optimizer uses them.

Why this page matters
---------------------

A physics-guided model can fail in two opposite ways:

- it can behave like a black-box forecaster if the physics
  terms are too weak;
- it can become numerically unstable or underfit the data if
  the physics terms are too strong, too early, or too poorly
  scaled.

GeoPrior addresses this by making the optimization structure
explicit and controllable. This is consistent with broader
work on physics-informed learning, where successful training
often depends on careful balancing between data fit,
residual-based regularization, and numerical stabilization
:cite:p:`Donnelly2023,Roy2024,Sarmaetal2024IPINNs`.

The high-level objective
------------------------

At the highest level, GeoPrior uses a custom training step
that optimizes

.. math::

   L_{\text{total}}
   =
   L_{\text{data}} + L_{\text{phys,scaled}}.

Here:

- :math:`L_{\text{data}}` is the supervised Keras loss
  computed from the configured output losses;
- :math:`L_{\text{phys,scaled}}` is the **scaled** physics
  loss returned by the shared physics core.

This distinction matters. The physics core already assembles
per-term weights, residual scaling, and global physics
multipliers before returning the physics loss seen by the
optimizer.

A compact view
--------------

A useful summary is:

.. code-block:: text

   supervised outputs
       ↓
   compiled Keras data loss
       +
   structured physics loss bundle
       ↓
   total objective
       ↓
   filtered and clipped gradients
       ↓
   optimizer step

This is the core training logic in GeoPrior.

The supervised data loss
------------------------

GeoPrior uses the normal Keras ``compile(...)`` interface for
the supervised part of the objective.

That means the model can inherit:

- per-output loss functions,
- per-output metrics,
- standard Keras regularization losses,
- and output-aware weighting behavior.

At the user-facing level, the main supervised outputs are:

- ``subs_pred``
- ``gwl_pred``

So the supervised loss may conceptually look like

.. math::

   L_{\text{data}}
   =
   L_{\text{subs}} + L_{\text{gwl}} + L_{\text{reg}},

where :math:`L_{\text{reg}}` includes Keras regularization
losses, if any.

The actual implementation relies on ``self.compiled_loss``
after aligning target shapes and output ordering to the
model’s output contract.

Why output alignment matters
----------------------------

GeoPrior is a dict-input, multi-output model, and some
outputs may be quantile-valued rather than only scalar.

Before supervised losses are computed, the training step
therefore:

- canonicalizes the target dictionary,
- reorders targets to match ``self.output_names``,
- aligns target shapes with prediction shapes,
- and then passes the aligned lists into
  ``self.compiled_loss``.

This is important because the optimization must remain stable
across:

- deterministic outputs,
- quantile outputs,
- and multi-output supervision.

The physics loss family
-----------------------

The physics side is built from several distinct terms rather
than one single PDE penalty.

At compile time, GeoPrior exposes the following scalar
weights:

- ``lambda_cons``
- ``lambda_gw``
- ``lambda_prior``
- ``lambda_smooth``
- ``lambda_mv``
- ``lambda_bounds``
- ``lambda_q``

These correspond to:

- consolidation residual loss
- groundwater-flow residual loss
- timescale prior loss
- smoothness regularization
- optional :math:`m_v` consistency loss
- bounds penalty
- forcing regularization on :math:`Q`

This makes the objective scientifically structured and
interpretable.

The per-term physics losses
---------------------------

Conceptually, the main physics loss terms are:

.. math::

   L_{\text{cons}}
   =
   \mathbb{E}\!\left[(R_{\text{cons}}^{*})^2\right]

.. math::

   L_{\text{gw}}
   =
   \mathbb{E}\!\left[(R_{\text{gw}}^{*})^2\right]

.. math::

   L_{\text{prior}}
   =
   \mathbb{E}\!\left[(R_{\text{prior}})^2\right]

.. math::

   L_{\text{smooth}}
   =
   \mathbb{E}\!\left[(R_{\text{smooth}})^2\right]

.. math::

   L_{\text{bounds}}
   =
   \mathbb{E}\!\left[(R_{\text{bounds}})^2\right].

Additional optional terms include:

.. math::

   L_{mv}
   \quad \text{and} \quad
   L_q.

These are then multiplied by their corresponding compile-time
weights.

Weighted physics terms
----------------------

A useful intermediate notation is:

.. math::

   T_{\text{cons}}
   =
   \lambda_{\text{cons}}\,L_{\text{cons}}

.. math::

   T_{\text{gw}}
   =
   \lambda_{\text{gw}}\,L_{\text{gw}}

.. math::

   T_{\text{prior}}
   =
   \lambda_{\text{prior}}\,L_{\text{prior}}

.. math::

   T_{\text{smooth}}
   =
   \lambda_{\text{smooth}}\,L_{\text{smooth}}

.. math::

   T_{\text{bounds}}
   =
   \lambda_{\text{bounds}}\,L_{\text{bounds}}

.. math::

   T_{mv}
   =
   \lambda_{mv}\,L_{mv}

.. math::

   T_q
   =
   \lambda_q\,L_q.

GeoPrior then defines a raw PDE-style core:

.. math::

   L_{\text{core}}
   =
   T_{\text{cons}}
   +
   T_{\text{gw}}
   +
   T_{\text{prior}}
   +
   T_{\text{smooth}}
   +
   T_{\text{bounds}}.

The full unscaled physics loss is:

.. math::

   L_{\text{phys,raw}}
   =
   L_{\text{core}} + T_{mv} + T_q.

This is useful for diagnostics because it is independent of
the global physics multiplier.

The global physics multiplier
-----------------------------

GeoPrior introduces a second layer of control through the
global physics multiplier ``lambda_offset``.

This is stored as a non-trainable scalar weight and converted
into the effective multiplier :math:`\alpha` according to the
selected ``offset_mode``:

``offset_mode='mul'``
    .. math::

       \alpha = \lambda_{\text{offset}}

``offset_mode='log10'``
    .. math::

       \alpha = 10^{\lambda_{\text{offset}}}

This design is extremely useful in practice because it lets
the user keep the per-term physics weights fixed while still
scheduling the overall strength of physics during training.

A useful interpretation is:

- the ``lambda_*`` values define the **relative structure**
  of the physics objective;
- ``lambda_offset`` defines the **overall physics level**.

Why the global multiplier exists
--------------------------------

The global multiplier solves an important practical problem.

If you only change each individual ``lambda_*`` by hand, then
it becomes hard to tell whether:

- the problem is the balance *within* physics,
- or the balance between data loss and physics loss.

By separating those two roles, GeoPrior makes experiments
cleaner and easier to interpret.

This is especially useful for:

- warmup schedules,
- ablation studies,
- transfer across cities,
- and early-stage debugging of stiff residual terms.

Scaled vs unscaled physics loss
-------------------------------

The final scaled physics loss is assembled as:

.. math::

   L_{\text{phys,scaled}}
   =
   \alpha\,L_{\text{core}}
   +
   s_{mv}\,T_{mv}
   +
   s_q\,T_q,

where:

- :math:`s_{mv} = \alpha` if
  ``scale_mv_with_offset=True``, else :math:`1`;
- :math:`s_q = \alpha` if
  ``scale_q_with_offset=True``, else :math:`1`.

This is a subtle but important design choice.

By default:

- PDE-style terms always scale with the global multiplier;
- the :math:`m_v` term is **not** forced to scale with the
  global multiplier unless explicitly requested;
- the :math:`Q` regularization term **does** scale with the
  global multiplier by default.

This lets calibration-style or auxiliary terms behave
differently from the main PDE core when needed.

A practical rule of thumb
-------------------------

A good way to think about the two physics losses is:

- ``physics_loss_raw`` tells you how large the underlying
  weighted physics mismatch is;
- ``physics_loss_scaled`` tells you what the optimizer is
  actually paying attention to in the total objective.

Both are useful, but they answer different questions.

Physics-off behavior
--------------------

GeoPrior also supports a clean **physics-off** regime.

When physics is disabled, the model neutralizes the physics
objective by forcing physics weights to safe values. In
practice, this means terms such as prior, smoothness,
:math:`m_v`, :math:`Q`, and bounds are set to zero
contribution, and the global offset is reset to a neutral
value.

This allows the same training and evaluation code paths to
work in:

- full physics mode,
- partial physics mode,
- purely data-driven mode.

That makes ablation and debugging much easier.

The compile API
---------------

GeoPrior extends the usual Keras ``compile(...)`` interface
with explicit physics-related arguments such as:

- ``lambda_cons``
- ``lambda_gw``
- ``lambda_prior``
- ``lambda_smooth``
- ``lambda_mv``
- ``lambda_bounds``
- ``lambda_q``
- ``lambda_offset``
- ``mv_lr_mult``
- ``kappa_lr_mult``
- ``scale_mv_with_offset``
- ``scale_q_with_offset``

The rest of the usual compile arguments, such as optimizer,
loss dictionary, and metrics, are still forwarded to the
underlying Keras machinery.

This means GeoPrior keeps the familiar Keras API while adding
its own structured physics controls.

A minimal compile example
-------------------------

.. code-block:: python

   from geoprior.models import GeoPriorSubsNet

   model = GeoPriorSubsNet(
       static_input_dim=3,
       dynamic_input_dim=8,
       future_input_dim=4,
       forecast_horizon=3,
       pde_mode="both",
   )

   model.compile(
       optimizer="adam",
       loss={"subs_pred": "mse", "gwl_pred": "mse"},
       lambda_cons=1.0,
       lambda_gw=1.0,
       lambda_prior=1.0,
       lambda_smooth=0.1,
       lambda_bounds=0.0,
       lambda_q=0.0,
       lambda_offset=0.1,
   )

This example already captures the main GeoPrior pattern:

- standard supervised losses,
- explicit physics weights,
- one global physics multiplier.

The training step
-----------------

GeoPrior overrides the default Keras ``train_step(...)`` to
implement its hybrid optimization logic.

Conceptually, one training step does the following:

1. unpack and canonicalize inputs and targets;
2. run the shared physics core inside a gradient tape;
3. compute aligned supervised data loss;
4. add the scaled physics loss;
5. compute gradients with respect to all trainable variables;
6. optionally rescale selected parameter gradients;
7. filter NaN / Inf gradients;
8. apply global-norm clipping;
9. apply the optimizer update;
10. update trackers and return a packed metrics dictionary.

This structure keeps data, physics, diagnostics, and
stability logic in one coherent training path.

The total training objective
----------------------------

Once the physics bundle has been returned by the shared core,
the total optimization target is simply:

.. math::

   L_{\text{total}}
   =
   L_{\text{data}} + L_{\text{phys,scaled}}.

This is a very important point:

GeoPrior does **not** recompute physics weighting inside the
training step. All of that is already handled inside the
shared physics bundle.

That keeps the training loop simpler and makes the physics
logic portable across training and evaluation.

Target alignment before loss computation
----------------------------------------

Because the model can emit deterministic or quantile-shaped
outputs, the training step aligns target tensors before
calling ``compiled_loss``.

This includes:

- ordering target keys consistently,
- matching output names,
- reconciling quantile layouts where needed.

This is especially important for robust multi-output training
under the dict-style GeoPrior API.

Gradient scaling for selected parameters
----------------------------------------

GeoPrior can apply parameter-specific gradient scaling during
training.

In particular, the compile interface exposes:

- ``mv_lr_mult``
- ``kappa_lr_mult``

These do **not** change the loss values themselves. Instead,
they rescale the gradients flowing into selected physical
parameters such as :math:`m_v` and :math:`\kappa`.

This is useful because some physics parameters are far more
sensitive than others and may otherwise update too quickly.

A useful interpretation is:

- the loss says **what** to optimize;
- gradient scaling changes **how aggressively** certain
  parameters respond.

Gradient sanitization
---------------------

GeoPrior includes explicit stability logic before applying
gradients.

The training step can:

- filter NaN gradients,
- filter Inf gradients,
- and optionally run extra gradient debug checks.

This is especially important in hybrid physics models, where
bad coordinate scaling, unstable exponential mappings, or
stiff residuals can produce pathological gradients early in
training.

Instead of letting one bad batch destroy the run, GeoPrior
tries to sanitize the update path first.

Global-norm clipping
--------------------

After gradient sanitization, GeoPrior applies global-norm
clipping before sending gradients to the optimizer.

The main purpose of this step is to reduce catastrophic
updates when:

- physics residuals spike,
- learned physical fields drift too fast,
- or coordinate-sensitive derivatives become temporarily
  unstable.

This is not a substitute for good scaling and good bounds,
but it provides an important safety layer during training.

Warmup and scheduling
---------------------

A recommended GeoPrior training pattern is to keep the
per-term ``lambda_*`` values fixed and schedule only the
global physics multiplier ``lambda_offset``.

This works well because:

- it preserves the internal balance of the physics terms;
- it lets the model start closer to a supervised forecaster;
- then it gradually increases physics discipline as training
  stabilizes.

This pattern is especially useful for:

- coupled ``pde_mode="both"`` runs,
- new datasets,
- transfer learning,
- identifiability experiments,
- and difficult cross-city training regimes.

Why warmup helps
----------------

A physics-guided model often faces a mismatch early in
training:

- the data head wants to learn quickly from supervised
  signals;
- the physics branch may initially produce poor fields,
  unstable derivatives, or raw residual spikes.

Warmup reduces the chance that the early optimization is
dominated by immature physics terms before the network has
learned a reasonable representation.

This is one of the most important practical reasons
``lambda_offset`` exists.

The evaluation step
-------------------

GeoPrior also overrides ``test_step(...)`` so that evaluation
uses the same broad objective structure as training, but
without weight updates.

This keeps evaluation consistent with training in terms of:

- supervised loss computation,
- physics-bundle assembly,
- target alignment,
- packed logging structure.

As a result, the logged evaluation metrics remain comparable
to the training-side ones.

What is logged
--------------

The packed step results typically include:

- total loss,
- per-output supervised losses and metrics,
- ``physics_loss_raw``
- ``physics_loss_scaled``
- ``physics_mult``
- ``lambda_offset``
- per-term physics losses such as consolidation, groundwater,
  prior, smoothness, bounds, :math:`m_v`, and :math:`Q`
- epsilon diagnostics.

This is why GeoPrior logs are much richer than the logs of a
standard black-box forecaster.

Epsilon metrics
---------------

GeoPrior optionally tracks epsilon-style physics diagnostics
during training and evaluation.

Typical keys include:

- ``epsilon_prior``
- ``epsilon_cons``
- ``epsilon_gw``
- ``epsilon_cons_raw``
- ``epsilon_gw_raw``

These are not primary optimization targets by themselves.
They are diagnostics derived from the residual bundle and are
especially useful for:

- debugging physics behavior,
- comparing runs across cities,
- judging whether residual scaling is doing its job,
- understanding whether training improvement is purely
  supervised or also physics-consistent.

What is **not** part of the training loss
-----------------------------------------

It is important not to confuse the model’s training loss with
the later calibration and export steps.

For example, interval calibration used later in Stage-2,
Stage-3, or Stage-4 is **not** part of the model’s internal
training objective. Calibration is a downstream procedure
applied to forecast intervals after the main training run.

So a useful distinction is:

- training loss = model optimization objective;
- calibration = post-training uncertainty adjustment.

This distinction keeps the scientific story cleaner.

How to choose physics weights
-----------------------------

There is no universal best set of weights, but a good
practical strategy is:

- start with moderate ``lambda_cons`` and ``lambda_prior``;
- keep ``lambda_smooth`` smaller than the core PDE terms;
- use ``lambda_bounds`` only when bounds matter scientifically
  and numerically;
- keep ``lambda_q`` at zero unless a learnable forcing head is
  part of the experiment;
- use ``lambda_offset`` to ramp the overall physics strength.

This is usually easier to reason about than constantly
changing every individual physics weight.

Common failure modes
--------------------

**Physics dominates too early**

Symptoms:

- total loss spikes,
- gradients become unstable,
- training stalls or becomes noisy.

Common fixes:

- lower ``lambda_offset``,
- use a warmup schedule,
- verify coordinate scaling and SI conversion,
- inspect raw vs scaled epsilons.

**Good data loss, poor physics diagnostics**

Symptoms:

- supervised metrics improve,
- but ``epsilon_cons`` or ``epsilon_gw`` remain large.

Common interpretation:

- the model is learning to fit the data,
- but the physical branch is still poorly constrained or
  inconsistently scaled.

**Physics looks good, supervised fit is weak**

Symptoms:

- residual diagnostics improve,
- but forecast performance remains poor.

Common interpretation:

- physics may be over-regularizing,
- or the data model needs more representation capacity.

**Unstable parameter fields**

Symptoms:

- implausible :math:`K`, :math:`S_s`, or :math:`\tau`,
- noisy training curves,
- bounds penalties or prior terms dominating.

Common fixes:

- inspect bounds mode,
- strengthen smoothness or prior terms,
- reduce gradient aggressiveness,
- verify Stage-1 and scaling assumptions.

A compact training map
----------------------

The GeoPrior loss and training flow can be summarized as:

.. code-block:: text

   inputs + targets
        ↓
   forward_with_aux(...)
        ↓
   residual assembly
        ↓
   per-term physics losses
        ↓
   weighted physics terms (lambda_*)
        ↓
   global physics multiplier (lambda_offset, offset_mode)
        ↓
   physics_loss_scaled
        +
   compiled supervised data loss
        ↓
   total_loss
        ↓
   gradients
        ↓
   optional param-specific grad scaling
        ↓
   NaN/Inf filtering + global norm clipping
        ↓
   optimizer step
        ↓
   packed logs + epsilon diagnostics

Best practices
--------------

.. admonition:: Best practice

   Treat ``lambda_offset`` as the main knob for physics
   warmup.

   Keep the relative ``lambda_*`` weights fixed unless you
   are intentionally changing the scientific balance within
   the physics objective.

.. admonition:: Best practice

   Compare ``physics_loss_raw`` and
   ``physics_loss_scaled`` together.

   They tell different stories: underlying mismatch versus
   effective optimization pressure.

.. admonition:: Best practice

   Use parameter-specific gradient scaling sparingly and only
   for clearly sensitive physical parameters.

   It is a stabilization tool, not a substitute for proper
   model formulation.

.. admonition:: Best practice

   Inspect epsilon diagnostics alongside supervised metrics.

   This is one of the best ways to see whether the model is
   improving physically as well as statistically.

.. admonition:: Best practice

   Do not confuse post-training calibration with the training
   objective.

   Calibration belongs to later workflow stages, not to the
   core optimization loop.

See also
--------

.. seealso::

   - :doc:`geoprior_subsnet`
   - :doc:`physics_formulation`
   - :doc:`residual_assembly`
   - :doc:`data_and_units`
   - :doc:`scaling`
   - :doc:`maths`
   - :doc:`identifiability`