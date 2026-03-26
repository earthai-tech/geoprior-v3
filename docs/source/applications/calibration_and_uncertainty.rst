.. _applications_calibration_and_uncertainty:

===========================
Calibration and uncertainty
===========================

GeoPrior-v3 does not treat uncertainty as an optional extra
added after the real modeling work is done. In the flagship
subsidence-forecasting workflow, uncertainty is part of the
application itself.

This page explains how uncertainty enters the GeoPrior
workflow, how quantile-based predictions are produced and
interpreted, where interval calibration happens, and how
calibrated uncertainty products are exported for later use.

The key idea is simple:

- the model can produce **raw quantile forecasts**;
- the workflow can then apply **post-training interval
  calibration**;
- downstream users should interpret **calibrated** and
  **uncalibrated** products differently.

Why uncertainty matters in subsidence forecasting
-------------------------------------------------

Urban land subsidence forecasting is rarely useful as a
single deterministic number alone.

Practical users often care about questions such as:

- How uncertain is the forecast at each horizon?
- Is the forecast interval too narrow or too wide?
- Are uncertainty bands stable across sites and years?
- Does transfer to another city widen uncertainty in a
  meaningful way?
- Does calibration improve interval reliability?

This is especially important in GeoPrior’s main application
domain, where delayed subsidence response, uncertain
groundwater evolution, and cross-city heterogeneity can all
affect forecast confidence.

A useful practical summary is:

.. code-block:: text

   point forecast tells you what is most likely
   interval forecast tells you how much confidence to place
   around it

What GeoPrior means by uncertainty
----------------------------------

In the current workflow, GeoPrior’s main uncertainty pathway
is **quantile forecasting**.

That means the model can produce multiple forecast levels
such as:

- lower quantile
- median or central quantile
- upper quantile

rather than only one central estimate.

A typical quantile set might be something like:

.. code-block:: text

   [0.1, 0.5, 0.9]

though the actual set is controlled by configuration.

The model therefore does not represent uncertainty through
only one global variance term. Instead, it represents
uncertainty through the shape of the forecasted quantiles.

Where uncertainty enters the model
----------------------------------

Uncertainty enters the GeoPrior stack at several levels.

**1. Model outputs**

The public model outputs such as ``subs_pred`` and
``gwl_pred`` can carry a quantile axis when quantiles are
enabled.

**2. Supervised training**

The training stack can use quantile-aware losses rather than
only ordinary point losses.

**3. Post-training calibration**

Later stages fit or reuse an interval calibrator on top of
the model’s raw quantile outputs.

**4. Export and diagnostics**

The workflow exports both forecast tables and interval-aware
diagnostics such as coverage and sharpness.

This means uncertainty is not a single module. It is an
application path through the whole workflow.

Raw quantile forecasts
----------------------

A **raw quantile forecast** is the direct output of the model
before interval calibration is applied.

For subsidence, this usually means that the model predicts a
set of quantile trajectories across the forecast horizon,
with the median or central quantile acting as the main point
forecast.

A useful interpretation is:

- the model provides the initial uncertainty structure;
- calibration later adjusts whether that structure is too
  narrow or too wide.

This distinction is important, because users often
incorrectly assume that exported quantile intervals are
always already calibrated.

Point forecasts vs interval forecasts
-------------------------------------

GeoPrior users should clearly separate two kinds of forecast
interpretation.

**Point interpretation**

This focuses on:

- median or central forecast,
- MAE,
- RMSE,
- horizon-wise point accuracy.

**Interval interpretation**

This focuses on:

- coverage,
- sharpness,
- calibration quality,
- interval width behavior over horizon and space.

A model can have:

- a good central forecast,
- but poorly calibrated intervals.

That is why this page is necessary even for users who already
understand the point-forecast workflow.

How quantile supervision works conceptually
-------------------------------------------

When quantiles are enabled, GeoPrior can train the forecast
head using quantile-aware supervision rather than only a
single regression loss.

At a conceptual level, if :math:`\hat{y}^{(q)}` is the
prediction at quantile :math:`q`, and :math:`y` is the true
target, then a standard quantile or pinball loss has the form

.. math::

   \ell_q(y, \hat{y}^{(q)})
   =
   \max\left(
   q (y - \hat{y}^{(q)}),
   (q-1)(y - \hat{y}^{(q)})
   \right).

The GeoPrior NN layer also exposes a public helper for
weighted pinball-style objectives:

.. code-block:: python

   from geoprior.models import make_weighted_pinball

This makes quantile-aware supervision part of the supported
public workflow rather than a hidden implementation detail.

Why quantile forecasting fits GeoPrior
--------------------------------------

Quantile forecasting is a natural match for GeoPrior because:

- it supports multi-horizon uncertainty directly;
- it aligns well with calibrated interval export;
- it remains compatible with a physics-guided central path;
- and it avoids forcing the uncertainty story into one rigid
  parametric distribution.

This is especially useful in land subsidence forecasting,
where uncertainty may be horizon-dependent and spatially
heterogeneous rather than well described by one simple
Gaussian assumption.

Coverage and sharpness
----------------------

Two of the most important interval diagnostics in GeoPrior
are:

- **coverage**
- **sharpness**

The public "models" package already exports metrics and helpers
such as:

- ``Coverage80``
- ``Sharpness80``
- ``coverage80_fn``
- ``sharpness80_fn``

through the public namespace. These are intended to support
interval-aware evaluation, not only internal debugging.

A practical interpretation rule is:

**Coverage**
   tells you how often the true target falls inside the
   predicted interval.

**Sharpness**
   tells you how narrow or wide that interval is.

A good uncertainty model should balance both:

- very wide intervals can achieve high coverage but be
  uninformative;
- very narrow intervals can look sharp but fail badly in
  reliability.

Why calibration is needed
-------------------------

Raw quantile outputs are often useful, but they are not
automatically reliable.

A model may systematically produce:

- intervals that are too narrow,
- intervals that are too wide,
- or intervals whose horizon-wise reliability drifts across
  the forecast window.

That is why GeoPrior includes **post-training interval
calibration** as part of the application workflow.

Calibration answers a different question from training:

- training asks how to fit the forecasting model;
- calibration asks how to adjust the interval structure so
  that uncertainty behaves more honestly.

Where calibration happens in the workflow
-----------------------------------------

Calibration appears in several stages of the GeoPrior
workflow.

**Stage-2**

After training, the workflow can reload a clean inference
model, fit interval calibration on validation outputs, and
then export calibrated forecast tables.

**Stage-3**

After tuning, the tuned winner can also be calibrated before
exporting tuned evaluation and future forecasts.

**Stage-4**

Saved-model reuse can apply one of several calibration
policies:

- no calibration,
- reuse a source calibrator,
- load an explicit calibrator,
- or fit a new target-side calibrator.

**Stage-5**

Cross-city transfer experiments can compare calibration
modes such as:

- none,
- source,
- target.

So calibration in GeoPrior is not confined to one single
stage. It is an application-facing concept that appears
wherever probabilistic forecasts are reused or compared.

The public calibration helpers
------------------------------

The public NN surface already exposes two calibration helpers:

.. code-block:: python

   from geoprior.models import (
       fit_interval_calibrator_on_val,
       apply_calibrator_to_subs,
   )

These are the main user-facing entry points for interval
calibration on the Python side.

A useful conceptual interpretation is:

- ``fit_interval_calibrator_on_val(...)``
  learns interval-width adjustment from validation outputs;

- ``apply_calibrator_to_subs(...)``
  applies the learned calibration to subsidence quantiles
  before export or downstream evaluation.

This is one of the cleanest signs that calibration is a
first-class application feature in GeoPrior.

How interval calibration works conceptually
-------------------------------------------

A useful conceptual view is:

1. the model emits a central forecast and outer quantiles;
2. the calibration stage measures interval behavior on
   validation data;
3. a scaling or adjustment factor is learned for the
   uncertainty width;
4. that adjusted interval is then exported as the calibrated
   forecast product.

This means calibration usually acts on the **interval width**
around the central tendency, not by retraining the model from
scratch.

That distinction matters because it keeps the scientific
story cleaner:

- forecasting model = learned from train data,
- interval calibration = fitted on validation behavior.

Calibration policies in practice
--------------------------------

Stage-4 and Stage-5 make calibration policy explicit, which
is one of the strongest parts of the GeoPrior design.

A useful practical summary is:

**No calibration**
   Export the raw quantile forecast as-is.

**Source calibrator reuse**
   Reuse the calibrator fitted near the original source model.

**Target recalibration**
   Fit a new calibrator on the target validation split.

These policies answer different scientific questions and
should not be mixed casually.

For example:

- source calibration tests whether uncertainty transfer is
  portable;
- target recalibration tests whether uncertainty improves
  after adapting the interval structure to the new domain.

Calibrated vs uncalibrated exports
----------------------------------

GeoPrior’s export flow may produce:

- uncalibrated evaluation forecast CSVs,
- calibrated evaluation forecast CSVs,
- uncalibrated future forecast CSVs,
- calibrated future forecast CSVs.

This is important because downstream users may otherwise
assume that the latest CSV is “the” forecast, without knowing
whether it reflects raw model uncertainty or post-hoc
calibrated uncertainty.

A good documentation and reporting rule is:

- always label exported uncertainty products as calibrated or
  uncalibrated.

Why future forecasts may also be calibrated
-------------------------------------------

Calibration is often fitted on validation behavior, but then
applied to future forecast intervals too.

That is useful because future forecasts have no observed
targets yet, but they still benefit from the best available
interval-width correction learned from held-out validation
behavior.

So a typical GeoPrior uncertainty flow is:

.. code-block:: text

   fit model on train data
        ↓
   estimate interval mismatch on validation data
        ↓
   learn calibration factor
        ↓
   apply it to future forecast intervals

This is one reason the calibration workflow belongs in the
applications section.

What gets exported
------------------

Depending on the stage and run, uncertainty-aware artifacts
can include:

- calibrated evaluation forecast CSV
- calibrated future forecast CSV
- calibration statistics JSON
- calibrated diagnostics JSON
- interval factor files
- inference summary JSON with calibration status
- transfer benchmark tables showing the calibration mode used

This is important because uncertainty in GeoPrior is not only
a tensor inside memory. It is exported as application-facing
artifacts meant for inspection, plotting, reporting, and
benchmarking.

How to interpret interval results
---------------------------------

A practical interpretation workflow is:

1. inspect the raw forecast intervals,
2. inspect the calibrated intervals,
3. compare coverage,
4. compare sharpness,
5. decide whether calibration improved reliability without
   making intervals uselessly wide.

This is a much better practice than looking only at one
coverage number in isolation.

A useful diagnostic interpretation rule is:

- **higher coverage** is not automatically better if sharpness
  collapses;
- **sharper intervals** are not automatically better if
  coverage becomes poor.

Good uncertainty modeling needs both.

Common use cases
----------------

GeoPrior’s uncertainty workflow is especially useful in the
following cases.

**Operational forecasting**

Users need future subsidence ranges, not only point
estimates.

**Scientific reporting**

Users want to show whether uncertainty bands remain reliable
across horizons or across cities.

**Transfer benchmarking**

Users want to compare whether source-calibrated or
target-calibrated transfer behaves better.

**Model comparison**

Users want to know whether a tuned model improved not only
point fit but also interval reliability.

How uncertainty interacts with tuning
-------------------------------------

A tuned model is not automatically a better uncertainty
model.

Stage-3 can improve:

- predictive accuracy,
- architecture fit,
- and hybrid loss balance,

but interval behavior still needs to be checked separately.

That is why the tuning workflow and the calibration workflow
should be seen as complementary:

- Stage-3 finds a better forecasting model;
- calibration-and-uncertainty analysis checks whether the
  resulting intervals are trustworthy.

A useful practical rule is:

- do not declare a tuning run “best” based only on one scalar
  objective if uncertainty is part of the application goal.

How uncertainty interacts with transfer
---------------------------------------

In cross-city transfer, uncertainty is often even more
important than point prediction.

A source-domain model may:

- transfer its point forecast moderately well,
- but produce poorly calibrated uncertainty in the target
  city.

That is why Stage-5 includes calibration modes explicitly.
It lets you ask:

- does zero-shot transfer preserve uncertainty reliability?
- does source-calibrator reuse help?
- does target-side recalibration make the transferred forecast
  more trustworthy?

This makes uncertainty a major part of the GeoPrior transfer
application, not a side metric.

Programmatic uncertainty workflow
---------------------------------

For users working directly from Python, a practical
uncertainty-oriented workflow is:

.. code-block:: python

   from geoprior.models import (
       fit_interval_calibrator_on_val,
       apply_calibrator_to_subs,
       coverage80_fn,
       sharpness80_fn,
   )

   calibrator = fit_interval_calibrator_on_val(
       y_true_val,
       q_low_val,
       q_med_val,
       q_high_val,
   )

   q_low_cal, q_high_cal = apply_calibrator_to_subs(
       q_low_future,
       q_med_future,
       q_high_future,
       calibrator,
   )

   cov = coverage80_fn(y_true_val, q_low_cal, q_high_cal)
   shp = sharpness80_fn(q_low_cal, q_high_cal)

This is only a conceptual sketch, but it reflects the public
workflow surface exposed by the package.

Common mistakes
---------------

**Treating calibrated and uncalibrated intervals as the same**

They answer different questions and should not be merged in
reporting.

**Looking only at point metrics**

A good MAE does not guarantee trustworthy uncertainty.

**Looking only at coverage**

Very wide intervals can look good on coverage alone.

**Skipping calibration in transfer studies**

Transferred uncertainty is often less reliable than
same-domain uncertainty and deserves explicit treatment.

**Assuming tuning automatically fixes uncertainty**

Tuning may help, but interval reliability still needs to be
checked separately.

Best practices
--------------

.. admonition:: Best practice

   Always label forecast outputs as calibrated or
   uncalibrated.

   This is essential for honest interpretation.

.. admonition:: Best practice

   Evaluate coverage and sharpness together.

   Neither metric is sufficient by itself.

.. admonition:: Best practice

   Use validation data for interval calibration.

   Keep calibration conceptually separate from the main
   training objective.

.. admonition:: Best practice

   Treat uncertainty as part of the application goal, not as
   a cosmetic add-on.

   This is especially important in urban risk and planning
   settings.

.. admonition:: Best practice

   In transfer experiments, make the calibration mode explicit
   in every table and figure.

   Source-calibrated and target-calibrated results are not
   the same scientific regime.

A compact calibration-and-uncertainty map
-----------------------------------------

The GeoPrior uncertainty workflow can be summarized as:

.. code-block:: text

   model emits raw quantile forecasts
        ↓
   evaluate raw interval behavior on validation data
        ↓
   fit interval calibrator
        ↓
   apply calibration to subsidence quantiles
        ↓
   export calibrated eval/future forecast products
        ↓
   compare coverage and sharpness
        ↓
   reuse or benchmark across domains

Read next
---------

The strongest next pages after this one are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Reproducibility scripts
      :link: reproducibility_scripts
      :link-type: doc
      :class-card: sd-shadow-sm

      See how calibrated forecasts, diagnostics, and payloads
      fit into the paper-facing workflow.

   .. grid-item-card:: Figure generation
      :link: figure_generation
      :link-type: doc
      :class-card: sd-shadow-sm

      Move from uncertainty outputs into plot-ready and
      figure-ready artifacts.

   .. grid-item-card:: Diagnostics
      :link: ../user_guide/diagnostics
      :link-type: doc
      :class-card: sd-shadow-sm

      Revisit how to interpret raw and calibrated outputs in
      the broader workflow.

   .. grid-item-card:: Inference and export
      :link: ../user_guide/inference_and_export
      :link-type: doc
      :class-card: sd-shadow-sm

      See how calibration-aware outputs are exported and
      reused.

.. seealso::

   - :doc:`subsidence_forecasting`
   - :doc:`tuner_workflow`
   - :doc:`../user_guide/stage2`
   - :doc:`../user_guide/stage3`
   - :doc:`../user_guide/stage4`
   - :doc:`../user_guide/stage5`
   - :doc:`../user_guide/diagnostics`
   - :doc:`../user_guide/inference_and_export`