.. _abbreviations:

Abbreviations and acronyms
==========================

This page collects the most common abbreviations used across the
GeoPrior documentation, examples, API reference, and manuscript-style
materials.

.. note::

   This page focuses on **abbreviations and acronyms** only.

   Mathematical notation and physics symbols such as
   :math:`K`, :math:`S_s`, :math:`H_d`, :math:`\tau`,
   :math:`\varepsilon_{cons}`, and :math:`\gamma_w`
   should be documented separately in :doc:`symbols`.

How to use this page
--------------------

- Use this page when you meet a short form such as ``PINN``,
  ``InSAR``, or ``PIT`` in the docs or examples.
- Use :ref:`symbols` later for equations, physical fields,
  and diagnostic symbols.
- Use :ref:`key_terms` later for longer conceptual explanations such as
  *calibration*, *sharpness*, *identifiability*, or
  *closure consistency*.

Core abbreviations
------------------

.. list-table::
   :header-rows: 1
   :widths: 16 34 50

   * - Abbreviation
     - Full term
     - Notes
   * - BS
     - Brier score
     - Probabilistic forecast metric used for binary or threshold
       exceedance evaluation.
   * - DTW
     - Dynamic time window
     - Refers to the moving or staged temporal input structure used in
       forecasting workflows.
   * - GeoPriorSubsNet
     - Geomechanical Prior-Informed Subsidence Network
     - Main physics-guided forecasting model introduced in GeoPrior.
   * - GRN
     - Gated residual network
     - Neural-network building block commonly used inside attentive
       forecasting architectures.
   * - HALNet
     - Hybrid Attentive LSTM Network
     - Backbone family or architectural reference used in the model
       lineage.
   * - InSAR
     - Interferometric Synthetic Aperture Radar
     - Remote-sensing technique used to derive land-subsidence
       observations.
   * - LSTM
     - Long short-term memory
     - Recurrent neural-network unit used for sequence modelling.
   * - MAE
     - Mean absolute error
     - Standard deterministic forecast error metric.
   * - MME
     - Multi-modal encoder
     - Encoder block used when combining multiple feature sources or
       modalities.
   * - MSE
     - Mean squared error
     - Standard deterministic forecast error metric that penalizes
       large deviations more strongly than MAE.
   * - MSLSTM
     - Multi-scale long short-term memory network
     - Multi-scale LSTM-style architecture used as a baseline or
       comparison family.
   * - PC1
     - First principal component
     - First latent axis from principal-component analysis.
   * - PI
     - Prediction interval
     - Interval forecast such as an 80% prediction interval used for
       uncertainty evaluation.
   * - PIT
     - Probability integral transform
     - Calibration diagnostic used to assess whether predictive
       distributions are statistically consistent with observations.
   * - PINN
     - Physics-informed neural network
     - Neural-network framework constrained by physical equations or
       residuals.
   * - PRD
     - Pearl River Delta
     - Regional setting containing the study areas used in the
       manuscript and examples.
   * - QC
     - Quality control
     - Data-screening and validation procedures applied before model
       training or evaluation.
   * - SBAS
     - Small Baseline Subset
     - InSAR time-series processing strategy used to recover ground
       deformation.
   * - SI
     - International System of Units
     - Conventional metric unit system used for physical quantities
       throughout the documentation.
   * - VSN
     - Variable selection network
     - Feature-selection block used in attention-based forecasting
       models.

Reading hints
-------------

A few abbreviations appear especially often across the project:

- **GeoPriorSubsNet** is the main forecasting model.
- **PINN** describes the broader modelling style.
- **InSAR** refers to the observation source for deformation.
- **PIT**, **BS**, and **PI** are central to uncertainty and
  probabilistic evaluation.
- **LSTM**, **GRN**, and **VSN** describe important neural-network
  components.

See also
--------

.. seealso::

   :doc:`symbols`
      Mathematical notation, physical fields, and diagnostic symbols.

   :doc:`terms`
      Plain-language explanations of key modelling and forecasting
      concepts.