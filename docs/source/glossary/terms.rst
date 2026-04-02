.. _key_terms:

Key terms and concepts
======================

This page explains the main modelling, forecasting, uncertainty,
and transfer concepts used throughout GeoPrior.


.. glossary::
   :sorted:

   Brier score
      Mean-squared error for probabilistic forecasts of a binary
      event, such as whether subsidence exceeds a chosen hazard
      threshold. Lower values indicate better probabilistic skill.

   calibration
      Agreement between stated predictive probabilities and observed
      frequencies. A calibrated forecast says “about 80%” and is
      correct about 80% of the time in repeated use.

   closure consistency
      Property of a solution in which the learned consolidation
      timescale remains compatible with the timescale implied by the
      effective closure variables, especially :math:`K`,
      :math:`S_s`, and :math:`H_d`. In GeoPrior, this idea is central
      to making physics guidance testable rather than merely
      decorative.

   coverage
      Fraction of observations that fall inside a stated prediction
      interval. For example, an 80% interval has good coverage when
      about 80% of held-out observations fall within it.

   cross-city transfer
      Evaluation setting in which a model trained on one city is
      applied to another city, allowing the project to test
      generalisation under basin-scale domain shift.

   data-dominated regime
      Regime in which the inferred timescale departs from the simple
      prior closure, yet the residual physics errors remain small.
      In practice, this suggests that the effective dynamics are being
      driven more by the data than by the prior timescale relation.

   dynamic-balance error
      Diagnostic measuring mismatch between the forecast mean-path
      settlement rate and the reduced consolidation law. Lower values
      indicate tighter agreement with the adopted relaxation dynamics.

   effective compressible thickness
      Censor-aware thickness quantity used by the physics pathway to
      represent the compressible part of the subsurface. It is built
      from mapped thickness with explicit handling of right-censored
      values, so it should be interpreted as an effective modelling
      input rather than a direct uncensored borehole-thickness
      estimate.

   effective drainage thickness
      Hydro-geomechanical drainage-path length controlling the reduced
      consolidation timescale. In the current implementation it is not
      learned as a fully free field; it is derived from externally
      supplied thickness information through the model configuration.

   exceedance probability
      Probability that subsidence exceeds a chosen threshold at a
      given location and time. This is the basis for threshold-risk
      maps and Brier-score evaluation.

   forecast horizon
      Number of future years predicted jointly from one historical
      input window.

   hotspot
      Location or spatial cluster identified as especially important
      for action because forecast severity, exceedance likelihood,
      and sometimes persistence jointly indicate elevated concern.

   hotspot stability
      Measure of whether the same high-priority locations remain
      important under transfer, recalibration, or repeated forecast
      years. In practice this helps answer the policy question
      “where should we act first?”

   identifiability
      Ability to recover or meaningfully separate effective model
      quantities from the available data and physics constraints.
      In GeoPrior, the effective timescale is usually more robustly
      recoverable than its full decomposition into
      :math:`K`, :math:`S_s`, and :math:`H_d`.

   identifiability regime
      Named configuration profile used to reduce ridge-like
      non-uniqueness in the closure. Current profiles include
      ``base``, ``anchored``, ``closure_locked``, and
      ``data_relaxed``. These profiles adjust how strongly the model
      is anchored by priors, bounds, or closure controls.

   physics-consistent forecast
      Forecast whose trajectories not only fit the data reasonably
      well, but also remain consistent with the adopted reduced
      groundwater and consolidation scaffold.

   prediction interval
      Interval-valued forecast, such as a central 80% interval,
      intended to capture predictive uncertainty rather than provide
      only a single deterministic value.

   sharpness
      Typical width of a predictive interval. For the same level of
      calibration and coverage, sharper forecasts are narrower and
      therefore more informative.

   warm-start adaptation
      Transfer strategy in which a model trained in one city is used
      to initialise the target-city model and is then fine-tuned on
      local target data. In the manuscript, this performs much better
      than direct zero-shot transfer.

   zero-shot transfer
      Direct application of a source-city model to a target city
      without local refitting. This is a stricter test of
      generalisation and is usually less reliable than warm-start
      adaptation under strong distribution shift.

How to read these terms
-----------------------

A useful reading order for new users is:

#. Start with **forecast horizon**, **prediction interval**,
   **calibration**, **coverage**, and **sharpness**.
#. Then read **closure consistency**, **dynamic-balance error**,
   **effective compressible thickness**, and
   **effective drainage thickness**.
#. Finish with **cross-city transfer**, **zero-shot transfer**,
   **warm-start adaptation**, and **hotspot stability**.

See also
--------

.. seealso::

   :doc:`abbreviations`
      Acronyms and short forms used across the project.

   :doc:`symbols`
      Mathematical notation, physical fields, and diagnostics.