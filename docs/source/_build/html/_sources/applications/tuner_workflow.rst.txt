.. _applications_tuner_workflow:

==============
Tuner workflow
==============

GeoPrior-v3 includes a dedicated tuning workflow for
systematic hyperparameter search over its flagship
subsidence-forecasting model family.

This page explains how tuning fits into the broader GeoPrior
application stack and how the current tuner is meant to be
used in practice.

The key point is that GeoPrior tuning is **not** only a
search over generic neural-network knobs such as hidden units
or dropout. It is also a search over **physics-aware model
controls**, including:

- the architecture of the attentive forecasting backbone,
- compile-time learning settings,
- PDE mode choices,
- and physics loss weights.

In the current codebase, the public tuner surface for this
application is centered on
:class:`~geoprior.models.forecast_tuner.SubsNetTuner`, which
is exported directly from the tuner package and specializes
the GeoPrior tuning base class for
:class:`~geoprior.models.GeoPriorSubsNet`. 

Why tuning matters in GeoPrior
------------------------------

GeoPrior is not a minimal one-loss neural predictor.

Its flagship model combines:

- multi-horizon sequence forecasting,
- multiple input groups,
- physics-guided residual losses,
- timescale priors,
- bounds penalties,
- and optional uncertainty outputs.

That means model quality depends on more than one family of
hyperparameters. A useful run may require good choices for:

- representational capacity,
- regularization,
- optimization rate,
- and physics-loss balance.

This is why tuning is a first-class application workflow in
GeoPrior-v3 rather than a small optional add-on.

Where tuning sits in the workflow
---------------------------------

The GeoPrior staged workflow naturally places tuning after the
first stable Stage-1 preprocessing contract is available.

A good mental model is:

.. code-block:: text

   Stage-1
     establish the tensor/manifest contract
        ↓
   Stage-2
     run one configured training baseline
        ↓
   Stage-3
     tune the model systematically
        ↓
   Stage-4
     reuse the tuned winner for inference/export
        ↓
   Stage-5
     optionally compare transfer behavior

In other words:

- Stage-2 answers “can this setup work?”
- Stage-3 answers “which setup works best under the current
  contract?”

This makes `tuner_workflow.rst` one of the most important
application pages.

The public tuner class
----------------------

The main public tuner class is:

.. code-block:: python

   from geoprior.models.forecast_tuner import SubsNetTuner

This class is a specialized tuner for
:class:`~geoprior.models.GeoPriorSubsNet`. In the current
implementation it inherits from a shared
``PINNTunerBase`` and binds the model class to
``GeoPriorSubsNet`` directly. 

That means the tuner is not generic over arbitrary model
families. It is intentionally built around the GeoPrior
subsidence model and its hybrid forecasting-plus-physics
training contract.

What the tuner is designed to search
------------------------------------

The current tuner supports search over three broad groups of
hyperparameters.

**1. Architecture hyperparameters**

Typical examples include:

- ``embed_dim``
- ``hidden_units``
- ``lstm_units``
- ``attention_units``
- ``num_heads``
- ``dropout_rate``
- ``vsn_units``
- ``use_vsn``
- ``use_batch_norm``

These control the forecasting backbone capacity and structure.
The default tuner configuration also already assumes a
GeoPrior-oriented architecture setup with a hybrid encoder,
attention stack, and variable-selection style feature
processing. 

**2. Physics/model hyperparameters**

Typical examples include:

- ``pde_mode``
- ``mv``
- ``kappa``
- ``use_effective_h``
- ``hd_factor``
- ``kappa_mode``
- ``scale_pde_residuals``

These control the physics-facing posture of the model rather
than only its representation capacity. 

**3. Compile-only hyperparameters**

The tuner also distinguishes a compile-only set of search
keys, including:

- ``learning_rate``
- ``lambda_gw``
- ``lambda_cons``
- ``lambda_prior``
- ``lambda_smooth``
- ``lambda_mv``
- ``lambda_bounds``
- ``lambda_q``
- ``lambda_offset``
- ``scale_mv_with_offset``
- ``scale_q_with_offset``
- ``mv_lr_mult``
- ``kappa_lr_mult``. 

This separation is scientifically important because it keeps
the model constructor and the optimizer/loss configuration
conceptually distinct.

Why compile-only parameters matter
----------------------------------

In a standard neural tuner, compile settings may feel like
secondary details.

In GeoPrior they are not secondary, because the compile step
controls the balance between:

- supervised forecasting loss,
- groundwater residual loss,
- consolidation residual loss,
- prior consistency,
- smoothness,
- bounds,
- and optional forcing regularization.

So a tuned GeoPrior model is not only “the best architecture.”
It is often also the best **data/physics compromise** found
under the search budget.

The `create(...)` entry point
-----------------------------

The recommended entry point is:

.. code-block:: python

   SubsNetTuner.create(...)

The class docstring explicitly describes this as the preferred
path, because it infers data dimensions from NumPy arrays,
merges them with robust defaults, and applies user overrides. 

This is one of the most convenient parts of the tuning
workflow, because it reduces the amount of hand-written model
dimension plumbing a user has to do.

Conceptually, the create path does three things:

1. canonicalize target names,
2. infer fixed input/output dimensions,
3. merge those inferred values with user-supplied fixed
   parameters.

Target canonicalization
-----------------------

The tuner normalizes the target dictionary so that the model
sees the expected GeoPrior output keys:

- ``subsidence`` → ``subs_pred``
- ``gwl`` → ``gwl_pred``. 

This is a small but important application detail because the
training and metrics stack are built around the model’s
public output contract, not around arbitrary dataset naming.

Required input structure
------------------------

The tuner docs also explicitly describe the required GeoPrior
inputs.

The expected keys include:

- ``coords``
- ``dynamic_features``
- ``H_field``

and the helper layer can canonicalize common aliases for
``H_field`` such as:

- ``soil_thickness``
- ``soil thickness``
- ``h_field``. 

This is important because tuning must still respect the same
physics contract as ordinary training. A search over
hyperparameters does not relax the requirement that the model
receive the correct physical inputs.

What stays fixed during tuning
------------------------------

The tuner separates **fixed parameters** from the searched
hyperparameters.

Typical fixed parameters include:

- ``static_input_dim``
- ``dynamic_input_dim``
- ``future_input_dim``
- ``output_subsidence_dim``
- ``output_gwl_dim``
- ``forecast_horizon``

as well as stable experiment-level settings such as a chosen
PDE regime, output shape conventions, or quantile layout. 

This design is important because a tuning run should search
under one stable Stage-1 contract rather than accidentally
mixing different data contracts inside one search job.

Search-space specification
--------------------------

The search space is intentionally flexible.

Each hyperparameter can be defined either as:

- a list of discrete choices,
- or a typed dict-based range.

The tuner docstring gives the intended patterns clearly.

Examples:

.. code-block:: python

   {
       "embed_dim": [32, 64, 96],
       "dropout_rate": {
           "type": "float",
           "min_value": 0.1,
           "max_value": 0.4,
       },
   }

Supported search-space types include:

- ``int``
- ``float``
- ``choice``
- ``bool``. 

This is a good balance for GeoPrior because it supports both:

- simple bounded experimental search,
- and more explicit discrete scientific comparisons.

Supported tuner backends
------------------------

The shared tuner base supports the following search engines:

- random search
- Bayesian optimization
- Hyperband. 

At the application level, this means the user can choose
between:

- simpler exploratory search,
- more sample-efficient model-based search,
- or more aggressive early-stopping style resource
  allocation.

A practical rule is:

- start with **random search** for the first stable tuning
  bring-up,
- then move to Bayesian or Hyperband only if the tuning
  budget and workflow maturity justify it.

How tuning interacts with GeoPrior losses
-----------------------------------------

The current tuner is deeply aware of the GeoPrior loss
structure.

The tuner docs explicitly note that GeoPrior adds residuals
consistent with:

- groundwater flow,
- and consolidation dynamics,

with weights controlled by compile-time lambda parameters. 

That makes the tuner scientifically different from an ordinary
sequence-model tuner. It is not searching only for predictive
fit, but for a configuration that behaves well under the
hybrid forecasting-plus-physics objective.

A useful conceptual view is:

.. code-block:: text

   tuner search
      ├── representation capacity
      ├── regularization strength
      ├── optimization rate
      └── physics balance

This is why tuned GeoPrior runs can behave very differently
even when the dataset is unchanged.

What a typical tuning experiment looks like
-------------------------------------------

A typical GeoPrior tuning workflow looks like:

1. preprocess one city with Stage-1,
2. choose a fixed forecast horizon and stable data contract,
3. define a compact but meaningful search space,
4. run `SubsNetTuner`,
5. inspect the best hyperparameters,
6. rebuild the best model,
7. evaluate and export the tuned winner.

This is the natural application meaning of Stage-3.

A compact Python example
------------------------

The tuner class docstring already gives a representative usage
pattern. Adapted to the documentation flow, a compact example
looks like this:

.. code-block:: python

   from geoprior.models.forecast_tuner import SubsNetTuner

   fixed = {
       "forecast_horizon": 7,
   }

   space = {
       "embed_dim": [32, 64],
       "num_heads": [2, 4],
       "dropout_rate": {
           "type": "float",
           "min_value": 0.1,
           "max_value": 0.3,
       },
       "learning_rate": [1e-3, 5e-4],
       "lambda_gw": {
           "type": "float",
           "min_value": 0.5,
           "max_value": 1.5,
       },
   }

   tuner = SubsNetTuner.create(
       inputs_data=inputs_np,
       targets_data=targets_np,
       search_space=space,
       fixed_params=fixed,
       max_trials=20,
       project_name="GeoPrior_HP_Search",
   )

   best_model, best_hps, kt = tuner.run(
       inputs=inputs_np,
       y=targets_np,
       validation_data=(val_inputs_np, val_targets_np),
       epochs=30,
       batch_size=32,
   )

This example is close to the current documented usage of the
class and captures the intended application lifecycle. 

What the base tuner adds
------------------------

The shared base tuner,
:class:`~geoprior.models.forecast_tuner._base_tuner.PINNTunerBase`,
provides the generic search orchestration:

- objective handling,
- tuner-type selection,
- directory/project management,
- Keras Tuner integration,
- early stopping during search,
- summary persistence,
- best-model recovery. 

This is valuable because the GeoPrior-specific tuner can stay
focused on:

- model-aware build logic,
- input/target canonicalization,
- and the distinction between searched and fixed parameters.

The objective and search logic
------------------------------

The base class lets the user specify an objective such as
``val_loss`` and infers the optimization direction when
needed. It also adds a default EarlyStopping callback during
search when one is not already provided. 

This is a sensible default for GeoPrior because tuning trials
can be expensive, especially when:

- the model is quantile-aware,
- the physics branch is active,
- and the forecast horizon is not tiny.

What the tuned result actually is
---------------------------------

A tuned result in GeoPrior should be interpreted as more than
just one best score.

At minimum, a useful tuned result consists of:

- the best hyperparameter set,
- the rebuilt best model,
- the search summary,
- and the Stage-3 evaluation/export artifacts associated with
  the tuned winner.

This is important because a good GeoPrior tuning outcome must
eventually be judged through:

- forecast quality,
- uncertainty behavior,
- and physics diagnostics,

not only through the tuner’s scalar objective.

Good search-space design
------------------------

A useful GeoPrior search space is usually **small but
scientifically meaningful**.

Good first candidates are:

- backbone size knobs such as hidden units or attention heads,
- dropout rate,
- learning rate,
- one or two physics weights,
- and perhaps one structural choice such as ``pde_mode`` or
  ``use_effective_h``.

A weak search strategy is to tune everything at once without
a stable Stage-2 baseline. That usually makes the result
harder to interpret scientifically.

A practical rule is:

- tune a few high-impact parameters first,
- then enlarge the search only if the workflow is already
  stable.

What not to tune too early
--------------------------

In early bring-up, avoid turning the search space into a
full scientific ablation study all at once.

For example, do **not** begin with a large space over:

- many architecture widths,
- many lambda weights,
- multiple PDE modes,
- multiple identifiability regimes,
- and multiple uncertainty layouts

unless the Stage-2 baseline is already trustworthy.

Tuning is powerful, but it should still be staged.

A practical tuning ladder
-------------------------

A useful progression is:

**First pass**

- tune learning rate,
- hidden units,
- dropout,
- number of heads.

**Second pass**

- tune one or two major physics lambdas,
- maybe compare one PDE regime change.

**Third pass**

- only after the workflow is stable, explore stronger
  scientific variations such as more physics options or
  transfer-aware tuning strategies.

This ladder makes the tuner easier to use scientifically.

Common mistakes
---------------

**Tuning before Stage-1 is stable**

The search then explores noise in the data contract rather
than real model improvement.

**Tuning too many physics weights at once**

This can make the best trial hard to interpret.

**Ignoring the fixed/search distinction**

A search should happen under one stable forecast horizon and
one stable Stage-1 contract.

**Reading only the best scalar objective**

The best trial still needs forecast, uncertainty, and physics
interpretation after tuning.

**Using a giant search space too early**

This wastes budget and often produces weak scientific
conclusions.

Best practices
--------------

.. admonition:: Best practice

   Always start from a trustworthy Stage-2 baseline.

   Tuning works best when it improves an already valid
   workflow rather than trying to rescue a broken contract.

.. admonition:: Best practice

   Keep the first search space compact.

   Tune only the highest-impact architecture and compile
   knobs before expanding the search.

.. admonition:: Best practice

   Separate model-init and compile-only parameters mentally.

   This makes the tuned result much easier to interpret.

.. admonition:: Best practice

   Preserve the best HPs, best model, and Stage-3 artifacts
   together.

   A scalar tuning objective is not the full scientific
   result.

.. admonition:: Best practice

   Inspect the tuned winner like a real model, not like a
   leaderboard entry.

   Forecast quality, uncertainty, and physics diagnostics all
   still matter.

A compact tuner-workflow map
----------------------------

The GeoPrior tuner workflow can be summarized as:

.. code-block:: text

   Stage-1 contract fixed
        ↓
   choose fixed params
        ↓
   define compact search space
        ↓
   SubsNetTuner.create(...)
        ↓
   search over architecture + physics + compile knobs
        ↓
   best_hps + best_model
        ↓
   Stage-3 tuned evaluation/export
        ↓
   Stage-4 reuse or Stage-5 transfer

Read next
---------

The strongest next pages after this one are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Calibration and uncertainty
      :link: calibration_and_uncertainty
      :link-type: doc
      :class-card: sd-shadow-sm

      Continue from the tuned model into interval-aware
      evaluation and uncertainty interpretation.

   .. grid-item-card:: Subsidence forecasting
      :link: subsidence_forecasting
      :link-type: doc
      :class-card: sd-shadow-sm

      Return to the main application story and see where
      tuning fits in the full forecasting lifecycle.

   .. grid-item-card:: Stage-3
      :link: ../user_guide/stage3
      :link-type: doc
      :class-card: sd-shadow-sm

      Read the user-guide view of the tuning stage itself.

   .. grid-item-card:: GeoPriorSubsNet
      :link: ../scientific_foundations/geoprior_subsnet
      :link-type: doc
      :class-card: sd-shadow-sm

      Revisit the flagship model being tuned.

.. seealso::

   - :doc:`subsidence_forecasting`
   - :doc:`calibration_and_uncertainty`
   - :doc:`../user_guide/stage3`
   - :doc:`../user_guide/stage2`
   - :doc:`../user_guide/stage4`
   - :doc:`../scientific_foundations/geoprior_subsnet`
   - :doc:`../scientific_foundations/losses_and_training`