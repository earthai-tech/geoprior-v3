.. _applications_subsidence_forecasting:

======================
Subsidence forecasting
======================

GeoPrior-v3 was built around a flagship application:
**physics-guided urban land subsidence forecasting**.

This page explains that application from the user and
scientific workflow perspective. It is not a full theory page
and it is not only a CLI tutorial. Instead, it shows how the
different parts of GeoPrior-v3 come together for one concrete
goal:

- prepare a city-scale subsidence dataset,
- train a physics-guided forecasting model,
- evaluate uncertainty-aware forecasts,
- reuse saved models for inference,
- and optionally study cross-city transfer behavior.

In the current codebase, the public model family for this
application is centered on
:class:`~geoprior.models.GeoPriorSubsNet` and
:class:`~geoprior.models.PoroElasticSubsNet`, with
GeoPriorSubsNet acting as the main coupled model and
PoroElasticSubsNet as a stronger consolidation-first preset. 

Why land subsidence is a strong GeoPrior application
----------------------------------------------------

Land subsidence is a particularly good fit for GeoPrior-v3
because it naturally combines:

- multi-horizon forecasting,
- groundwater-related delayed response,
- spatial heterogeneity,
- uncertainty-aware prediction,
- and physically interpretable closure structure.

In many urban and peri-urban systems, subsidence is not only
a curve-fitting problem. It is a coupled hydrogeological and
geomechanical forecasting problem in which the future
trajectory depends on:

- previous settlement history,
- groundwater evolution,
- effective thickness and drainage structure,
- rainfall and other external drivers,
- and local site conditions.

That is exactly the kind of setting GeoPrior is designed for.

What GeoPrior forecasts in this application
-------------------------------------------

The main supervised targets in the current model family are:

- subsidence,
- groundwater level or a head-related groundwater quantity.

At the model level, the main public outputs are
``subs_pred`` and ``gwl_pred``. The forecasting system can
also operate with quantiles, so the application naturally
supports both central forecasts and uncertainty-aware
forecast intervals. 

A practical mental model is:

.. code-block:: text

   historical site data
        ↓
   future-known drivers
        ↓
   GeoPrior forecasting backbone
        ↓
   subsidence and groundwater forecasts
        ↓
   physics-aware diagnostics and export

Typical forecasting questions
-----------------------------

This application layer is meant to support questions such as:

- How much subsidence is expected over the next
  1 to 5 years?
- Which zones are likely to experience stronger settlement?
- How uncertain are the forecast bands?
- Does a physically constrained model behave better than a
  purely data-driven baseline?
- Can a model trained in one city transfer to another city?

These are the kinds of questions that make land subsidence
forecasting more than a narrow regression benchmark.

What the data usually look like
-------------------------------

The GeoPrior config template already reflects the intended
structure of a subsidence forecasting dataset. It includes
city identity, file patterns, temporal windows, key columns,
groundwater conventions, optional feature groups, future
drivers, holdout strategy, model defaults, physics settings,
and training defaults. 

A typical application dataset contains:

- longitude and latitude,
- time or year,
- observed subsidence,
- groundwater level or groundwater depth,
- thickness-related fields,
- optional geological or lithological descriptors,
- optional urban-load or other anthropogenic indicators,
- rainfall or other future-known driver variables.

In the current default config template, GeoPrior explicitly
tracks fields such as:

- ``TIME_COL``
- ``LON_COL``
- ``LAT_COL``
- ``SUBSIDENCE_COL``
- ``GWL_COL``
- ``H_FIELD_COL_NAME``
- future-driver feature names
- optional numeric and categorical features. 

Why the groundwater convention matters
--------------------------------------

The same config template also makes groundwater semantics
explicit through settings such as:

- ``GWL_KIND``
- ``GWL_SIGN``
- ``USE_HEAD_PROXY``
- ``Z_SURF_COL``
- ``HEAD_COL``. 

This is especially important in subsidence forecasting,
because compaction is not built from an arbitrary raw
groundwater variable. It is built from a physically
interpreted head or drawdown quantity.

So the application story is not only:

- “we forecast subsidence”

but also:

- “we forecast subsidence under a declared groundwater
  interpretation.”

A complete application workflow
-------------------------------

The end-to-end subsidence forecasting workflow in GeoPrior-v3
can be summarized as:

.. code-block:: text

   Stage-1 preprocess city dataset
        ↓
   Stage-2 train one physics-guided model
        ↓
   Stage-3 optionally tune hyperparameters
        ↓
   Stage-4 reuse saved model for inference/export
        ↓
   Stage-5 optionally benchmark transfer across cities

This is the core application pathway for the package.

Stage-1 in this application
---------------------------

In the subsidence application, Stage-1 is where the raw city
dataset is turned into a stable modeling contract.

That includes:

- column resolution,
- feature grouping,
- coordinate handling,
- temporal window construction,
- train/validation/test splits,
- NPZ export,
- and manifest creation.

For this application, Stage-1 is especially important because
the later stages assume that subsidence, groundwater,
coordinates, and thickness-like fields have already been
organized into a coherent contract.

A good practical rule is:

- if the Stage-1 contract is wrong,
- the later subsidence forecasts may still run,
- but they become much harder to trust scientifically.

Stage-2 in this application
---------------------------

Stage-2 is the first true model-training stage.

In the subsidence application, it usually means:

- one city,
- one chosen configuration,
- one model family,
- one full training run,
- plus evaluation and export artifacts.

The current GeoPrior training stack is explicitly hybrid:
the supervised data loss is combined with a scaled physics
loss returned by the shared physics core. 

That makes Stage-2 scientifically important because it is not
only fitting a forecast model. It is also learning a
physically regularized representation of the subsidence
system.

Stage-3 in this application
---------------------------

Stage-3 extends the application from one configured run to a
systematic hyperparameter search.

In subsidence forecasting, this is where you ask:

- which architecture size works best?
- how much dropout is appropriate?
- do the physics weights need adjustment?
- does the tuned winner improve both forecast quality and
  physical consistency?

Stage-3 is therefore especially useful when you want to move
from a proof-of-concept city run to a more publication-grade
benchmark.

Stage-4 in this application
---------------------------

Stage-4 is the reusable inference stage.

In the subsidence application, this is where a saved model is
loaded and applied to a target contract to generate:

- evaluation forecasts,
- future forecasts,
- summary JSON files,
- and optional transfer-style diagnostics.

This is useful for operational or repeated forecasting
settings where the user does not want to retrain every time.

A typical use case is:

- preprocess once,
- train once,
- then run inference repeatedly as new forecast scenarios or
  reporting needs arise.

Stage-5 in this application
---------------------------

Stage-5 expands the application from one city to a transfer
study across two cities.

That is a very strong use case for urban subsidence
forecasting because different cities often share broad
hydrogeological mechanisms while still differing in forcing,
stratigraphy, urbanization pattern, and scale conventions.

In this stage, the application asks:

- how well does a model trained in city A work in city B?
- does target recalibration help?
- does warm-start adaptation improve transfer?
- do scaling choices explain some of the performance gap?

That is one of the reasons the subsidence application page
belongs early in the applications section.

Which model to use
------------------

For most urban land subsidence forecasting runs, the natural
starting point is:

:class:`~geoprior.models.GeoPriorSubsNet`

because it is the main coupled model in the current package. It
is publicly exported from the GeoPrior model layer together
with :class:`~geoprior.models.PoroElasticSubsNet`. 

Use GeoPriorSubsNet when
~~~~~~~~~~~~~~~~~~~~~~~~

- you want the flagship physics-guided model;
- you want coupled groundwater and consolidation behavior;
- you want quantile-aware subsidence forecasting;
- you want later access to physics payloads and diagnostics.

Use PoroElasticSubsNet when
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- you want a stronger consolidation-first baseline;
- the groundwater PDE is less reliable than the compaction
  pathway;
- you want a stricter prior-driven ablation or debugging
  setup.

A practical application data flow
---------------------------------

A useful way to think about the subsidence application is by
splitting the information into three groups:

**1. Static context**

Examples:

- geology or lithology class,
- stable site descriptors,
- long-lived spatial attributes.

**2. Dynamic historical context**

Examples:

- historical subsidence behavior,
- groundwater evolution,
- rainfall history,
- other past time-varying drivers.

**3. Future-known drivers**

Examples:

- forecast calendar positions,
- known driver scenarios,
- exogenous covariates assumed known over the horizon.

That structure matches the dict-input model contract used by
GeoPrior models, where the backbone combines static,
dynamic, future-known, and coordinate-aware inputs. 

What the default config already supports
----------------------------------------

The NATCOM config template already exposes most of the knobs
needed for this application.

Examples include:

- temporal layout such as
  ``TRAIN_END_YEAR``,
  ``FORECAST_START_YEAR``,
  ``FORECAST_HORIZON_YEARS``,
  ``TIME_STEPS``,
- holdout strategy settings,
- architecture defaults such as hidden units, LSTM units,
  attention units, heads, dropout, and quantiles,
- physics settings such as
  ``PDE_MODE_CONFIG``,
  ``SCALE_PDE_RESIDUALS``,
  ``LAMBDA_CONS``,
  ``LAMBDA_GW``,
  ``LAMBDA_PRIOR``,
  ``LAMBDA_SMOOTH``,
  ``LAMBDA_BOUNDS``,
- physical bounds,
- time-unit and coordinate-mode settings,
- training defaults such as epochs, batch size, and learning
  rate. 

That is why GeoPrior-v3 is best understood as a structured
application framework rather than a loose collection of
scripts.

Forecast horizons and practical use
-----------------------------------

The subsidence application is especially well suited to
**multi-year horizon forecasting**, where the user wants
future settlement trajectories rather than only one-step
predictions.

Examples include:

- annual forecasts over the next 3 years,
- medium-range subsidence scenarios over the next 5 years,
- benchmark comparisons across several horizon lengths.

The key point is that the model is not only learning a local
trend extrapolation. It is learning a multi-horizon response
under both data and physics constraints.

Uncertainty in this application
-------------------------------

Urban subsidence forecasting should not be presented as a
single deterministic curve whenever uncertainty matters.

GeoPrior’s model and application stack support quantile-based
forecasting, interval calibration, and downstream forecast
export. The public NN layer also exposes interval-calibration
helpers and related diagnostics utilities, which is why the
subsidence application naturally extends into calibrated
uncertainty workflows rather than stopping at point
prediction. 

So a practical application output is often not only:

- one median forecast,

but rather:

- a central forecast,
- a lower interval,
- an upper interval,
- plus diagnostics on coverage and sharpness.

Physics-aware interpretation in this application
------------------------------------------------

One reason GeoPrior is compelling for subsidence forecasting
is that the outputs can be interpreted together with learned
effective physical fields and physics diagnostics.

Depending on the run, later-stage artifacts can support
questions such as:

- is the learned timescale physically plausible?
- are the closure diagnostics stable?
- do the effective fields look smooth and bounded?
- do transfer runs break because of physics mismatch or only
  because of data shift?

That makes the application useful not only for operational
forecasting, but also for scientific interpretation.

A typical CLI-oriented workflow
-------------------------------

A practical end-to-end application flow is:

.. code-block:: bash

   geoprior-init --yes
   geoprior-run preprocess
   geoprior-run train
   geoprior-run tune
   geoprior-run infer --help
   geoprior-run transfer --help

The exact subcommand arguments should be taken from the
installed help pages, but this sequence captures the intended
application lifecycle. The config bootstrap command itself is
also designed around this staged flow. 

A compact application example
-----------------------------

Conceptually, a single-city subsidence forecasting experiment
looks like this:

.. code-block:: text

   choose city and dataset
        ↓
   define groundwater and coordinate conventions
        ↓
   export Stage-1 manifest and NPZ bundles
        ↓
   train GeoPriorSubsNet
        ↓
   inspect forecast metrics + physics diagnostics
        ↓
   export calibrated eval/future forecasts
        ↓
   optionally tune, reuse, or transfer

This is the most natural “first serious experiment” in
GeoPrior-v3.

When this application is especially valuable
--------------------------------------------

This application is especially strong when:

- subsidence has a delayed response to groundwater change;
- the user wants more than a black-box trend extrapolator;
- interpretability matters alongside predictive accuracy;
- multiple cities or regions need to be compared;
- future scenarios are needed rather than only retrospective
  fit.

It is less about “can a neural net predict a number?” and
more about “can a forecasting system remain useful under
scientific constraints?”

Common mistakes in subsidence forecasting workflows
---------------------------------------------------

**Treating Stage-1 as a minor preprocessing step**

In this application, Stage-1 defines the scientific contract.

**Ignoring groundwater semantics**

Head, depth-to-water, and drawdown are not interchangeable.

**Reporting only point forecasts**

For many planning or scientific uses, interval behavior also
matters.

**Reading model outputs without diagnostics**

A good fit can still hide poor physical consistency.

**Mixing artifacts from different runs**

This is especially dangerous when forecasting multiple cities
or comparing transfer regimes.

Best practices
--------------

.. admonition:: Best practice

   Start with one city and one well-audited Stage-1 contract.

   This is the cleanest way to validate the application
   workflow before tuning or transfer.

.. admonition:: Best practice

   Keep the groundwater convention explicit from the start.

   It affects drawdown, compaction, and physical
   interpretation.

.. admonition:: Best practice

   Use GeoPriorSubsNet as the default first model.

   Move to PoroElasticSubsNet when you intentionally want a
   more closure-dominant or consolidation-first regime.

.. admonition:: Best practice

   Inspect both forecast metrics and physics diagnostics.

   Subsidence forecasting in GeoPrior is both a predictive and
   a scientific workflow.

.. admonition:: Best practice

   Treat Stage-4 and Stage-5 as part of the application, not
   as afterthoughts.

   Reuse and transfer are major practical use cases.

A compact map of the application
--------------------------------

The GeoPrior subsidence forecasting application can be
summarized as:

.. code-block:: text

   city-scale subsidence dataset
        ↓
   Stage-1 preprocessing + manifest
        ↓
   GeoPriorSubsNet / PoroElasticSubsNet
        ↓
   Stage-2 training
        ↓
   Stage-3 tuning (optional)
        ↓
   Stage-4 inference and export
        ↓
   Stage-5 transfer benchmarking (optional)

Read next
---------

The strongest next pages after this one are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Tuner workflow
      :link: tuner_workflow
      :link-type: doc
      :class-card: sd-shadow-sm

      See how hyperparameter search fits into the application
      lifecycle.

   .. grid-item-card:: Calibration and uncertainty
      :link: calibration_and_uncertainty
      :link-type: doc
      :class-card: sd-shadow-sm

      Follow the subsidence application into interval
      calibration and uncertainty interpretation.

   .. grid-item-card:: Workflow overview
      :link: ../user_guide/workflow_overview
      :link-type: doc
      :class-card: sd-shadow-sm

      Revisit the five-stage workflow from the user guide.

   .. grid-item-card:: GeoPriorSubsNet
      :link: ../scientific_foundations/geoprior_subsnet
      :link-type: doc
      :class-card: sd-shadow-sm

      Go from the application page back into the flagship
      scientific model.

.. seealso::

   - :doc:`../user_guide/workflow_overview`
   - :doc:`../user_guide/stage1`
   - :doc:`../user_guide/stage2`
   - :doc:`../user_guide/stage3`
   - :doc:`../user_guide/stage4`
   - :doc:`../user_guide/stage5`
   - :doc:`../scientific_foundations/geoprior_subsnet`
   - :doc:`../scientific_foundations/physics_formulation`