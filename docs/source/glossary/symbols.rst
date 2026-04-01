Symbols and notation
====================

This page collects the main mathematical symbols, data symbols,
and diagnostic notation used throughout **GeoPrior-v3**.


Sign conventions
----------------

.. important::

   GeoPrior uses both depth-like and head-like quantities.

   - :math:`d(i,t)` is **groundwater depth below ground surface**
     and is positive **downward**.
   - :math:`h(i,t)` is **hydraulic head** and is positive
     **upward**.
   - Drawdown is typically written as
     :math:`\Delta h = h_{\mathrm{ref}} - h`,
     so positive :math:`\Delta h` corresponds to head loss.

Data and workflow symbols
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 16 66

   * - Symbol
     - Units
     - Meaning
   * - :math:`s(i,t)`
     - m
     - Cumulative vertical subsidence at grid cell :math:`i`
       and year :math:`t`.
   * - :math:`d(i,t)`
     - m
     - Groundwater depth below ground surface, positive downward.
   * - :math:`h(i,t)`
     - m
     - Hydraulic head, positive upward.
   * - :math:`R(i,t)`
     - mm yr\ :sup:`-1`
     - Annual rainfall.
   * - :math:`U^*(i,t)`
     - dimensionless
     - Normalised shallow urban-load proxy. This is a relative
       geospatial loading indicator, not a calibrated stress field.
   * - :math:`H(i)`
     - m
     - Mapped geological thickness in the harmonised data archive.
   * - :math:`H_{\mathrm{eff}}(i)` or :math:`H_{\mathrm{eff}}(x,y)`
     - m
     - Censor-aware effective compressible thickness. This is supplied
       by the data pipeline and is not learned as a free inversion field.
   * - :math:`z_{\mathrm{surf}}(i)`
     - m
     - Surface-elevation datum used in head construction and related
       preprocessing.
   * - :math:`c_H(i)`
     - dimensionless
     - Right-censor indicator for the thickness field.

Physics and closure symbols
---------------------------

.. math::

   \tau \approx \frac{H_d^2 \, S_s}{\pi^2 \, \kappa_b \, K}

The reduced closure above is central to GeoPrior and links the
effective drainage path, storage, anisotropy/leakage factor, and
hydraulic conductivity to the learned consolidation timescale. 

.. list-table::
   :header-rows: 1
   :widths: 20 16 64

   * - Symbol
     - Units
     - Meaning
   * - :math:`K(x,y)`
     - m s\ :sup:`-1`
     - Effective horizontal hydraulic conductivity controlling
       lateral head diffusion and entering the timescale closure.
   * - :math:`S_s(x,y)`
     - m\ :sup:`-1`
     - Specific storage. Links head changes to volumetric strain
       and enters both groundwater balance and equilibrium settlement.
   * - :math:`H_d(x,y)`
     - m
     - Effective drainage thickness or drainage path length used in
       the consolidation-timescale closure.
   * - :math:`\tau(x,y)`
     - s
     - Learned consolidation or relaxation timescale.
   * - :math:`\tau_{\mathrm{closure}}`
     - s
     - Closure-implied consolidation timescale computed from
       :math:`K`, :math:`S_s`, and :math:`H_d`.
   * - :math:`m_v(x,y)`
     - Pa\ :sup:`-1`
     - Vertical compressibility. Relates strain to effective stress
       and supports the approximation :math:`S_s \approx m_v \gamma_w`.
   * - :math:`\gamma_w`
     - N m\ :sup:`-3`
     - Unit weight of water, used to convert head change to
       effective-stress change.
   * - :math:`k_v`
     - m s\ :sup:`-1`
     - Effective vertical drainage conductivity.
   * - :math:`\kappa_b`
     - dimensionless
     - Effective anisotropy or leakage factor linking vertical
       conductivity to horizontal conductivity through
       :math:`k_v = \kappa_b K`.
   * - :math:`C_v`
     - m\ :sup:`2` s\ :sup:`-1`
     - Consolidation coefficient.
   * - :math:`\Delta h`
     - m
     - Drawdown, typically written as
       :math:`\Delta h = h_{\mathrm{ref}} - h`.
   * - :math:`h_{\mathrm{ref}}`
     - m
     - Reference head datum used to define drawdown.
   * - :math:`s_{\mathrm{eq}}`
     - m
     - Equilibrium settlement implied by the current head state.
   * - :math:`H_{\mathrm{phys}}`
     - m
     - Physical compressible-layer thickness appearing in the
       1-D consolidation derivation.
   * - :math:`w(z,t)`
     - m
     - Vertical displacement in the 1-D consolidation derivation,
       taken positive downward.
   * - :math:`\varepsilon_v(z,t)`
     - dimensionless
     - Vertical strain, commonly written as
       :math:`\varepsilon_v = \partial w / \partial z`.
   * - :math:`\Delta \sigma'`
     - Pa
     - Effective-stress change induced by drawdown.
   * - :math:`Q(t,x,y)`
     - context-dependent
     - Effective forcing term appearing in the groundwater-flow
       residual. In the code and docs it is treated as the forcing
       contribution in the reduced groundwater equation.

Residuals and diagnostic symbols
--------------------------------

.. math::

   R_{\mathrm{gw}}
   = S_s \, \partial_t h
   - \nabla \cdot \left(K \nabla h\right) - Q

.. math::

   R_{\mathrm{cons}}
   = \partial_t s - \frac{s_{\mathrm{eq}}(h) - s}{\tau}

These are the two main reduced residuals enforced by the GeoPrior
physics head. 

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Symbol
     - Units
     - Meaning
   * - :math:`R_{\mathrm{gw}}`
     - reduced residual
     - Groundwater-flow residual.
   * - :math:`R_{\mathrm{cons}}`
     - reduced residual
     - Consolidation residual in relaxation form.
   * - :math:`R_{\mathrm{prior}}`
     - log-residual
     - Prior-consistency residual comparing learned
       :math:`\tau` with closure-implied :math:`\tau`.
   * - :math:`\varepsilon_{\mathrm{cons}}`
     - dimensionless
     - RMS dynamic-balance or consolidation-error diagnostic.
   * - :math:`\varepsilon_{\mathrm{prior}}`
     - dimensionless
     - Prior-consistency error diagnostic for timescale mismatch.
   * - :math:`\Delta \log_{10}\tau`
     - decades
     - Closure-tension or identifiability-style mismatch in
       log-timescale space.

Forecasting and probabilistic symbols
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 16 66

   * - Symbol
     - Units
     - Meaning
   * - :math:`L`
     - years
     - Look-back window length.
   * - :math:`H`
     - years
     - Forecast horizon length.
   * - :math:`T_{\mathrm{eval}}`
     - time steps
     - Number of time steps in the evaluation window.
   * - :math:`Q`
     - set of probabilities
     - Forecast quantile set. In the main manuscript setting,
       :math:`Q = \{0.1, 0.5, 0.9\}`.
   * - :math:`\hat{s}^{(q)}(i,t_h)`
     - m
     - Predicted subsidence at quantile :math:`q`.
   * - :math:`\hat{h}^{(q)}(i,t_h)`
     - m
     - Predicted hydraulic head at quantile :math:`q`.
   * - :math:`PI_{0.8}(i,t_h)`
     - m
     - Nominal central 80% prediction interval, typically
       :math:`[\hat{s}^{(0.1)}, \hat{s}^{(0.9)}]`.
   * - :math:`T_s`
     - mm yr\ :sup:`-1`
     - Annual subsidence threshold used in exceedance tests.
   * - :math:`m_{i,t_h}`
     - binary mask
     - Availability mask for future head supervision when only a
       subset of windows has groundwater-head targets.

Model-dimension symbols
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 16 66

   * - Symbol
     - Units
     - Meaning
   * - :math:`d`
     - latent width
     - Stream embedding width in the predictive backbone.
   * - :math:`d_f`
     - latent width
     - Fused latent width after feature fusion.
   * - :math:`d_d`
     - latent width
     - Decoder width.
   * - :math:`n_{\mathrm{LSTM}}`
     - count
     - Number of LSTM layers.
   * - :math:`h_{\mathrm{LSTM}}`
     - count
     - Number of LSTM units per layer.
   * - :math:`n_{\mathrm{head}}`
     - count
     - Number of attention heads.
   * - :math:`p_{\mathrm{drop}}`
     - probability
     - Dropout rate.
   * - :math:`\eta`
     - learning rate
     - Optimizer learning rate.
   * - :math:`B`
     - batch size
     - Mini-batch size.
   * - :math:`N_{\max}`
     - epochs
     - Maximum number of training epochs.

Notes on interpretation
-----------------------

- :math:`H_{\mathrm{eff}}` is externally supplied and should be read
  as an effective, censor-aware thickness representation rather than
  as an uncensored borehole-thickness estimate.
- :math:`U^*` is a dimensionless proxy for shallow areal loading
  potential and should not be interpreted as a stress field in Pa.
- The most interpretable effective outputs are usually the relative
  spatial patterns, closure-consistent timescale structure, and
  thickness pathway, rather than a claim of unique local parameter
  recovery.

See also
--------

.. seealso::

   :doc:`abbreviations`
      Acronyms and short forms used across the project.

   :doc:`terms`
      Plain-language explanations of calibration, sharpness,
      identifiability, transfer, and other concepts.