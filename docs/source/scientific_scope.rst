Scientific scope and citation
===============================

GeoPrior-v3 is a physics-guided AI framework for geohazard modelling.
In the current documentation, its most fully developed scientific
application is **urban land subsidence forecasting** through
``GeoPriorSubsNet``.

Why this project exists
-----------------------

Urban land subsidence is a widespread geohazard that threatens
infrastructure, groundwater security, and coastal adaptation. It is
well documented in rapidly urbanising and groundwater-stressed regions,
including Jakarta, central Mexico, Iran, and many vulnerable coastal
settings worldwide :cite:p:`Abidinetal2011,Chaussardetal2014,
Motaghetal2008,GallowayBurbey2011,Shirzaeietal2021,
Nichollsetal2021,Fangetal2022`.

The scientific challenge is not only to detect subsidence, but also to
understand and forecast it under multiple interacting pressures.
Groundwater extraction, shallow anthropogenic loading, stratigraphic
heterogeneity, and coastal exposure can all interact in ways that are
difficult to separate in practice :cite:p:`CandelaKoster2022`.

The modelling gap
-----------------

Traditional physics-based simulators can provide strong mechanistic
detail, but they are expensive to parameterise and difficult to deploy
consistently over dense, city-scale geospatial archives. Purely
data-driven machine-learning models can capture correlations well, but
they often lack explicit physical accountability and may extrapolate in
ways that are difficult to audit for hazard management
:cite:p:`Liuetal2024,Limetal2021`.

GeoPrior-v3 is designed to address this gap: it aims to support forecasting
workflows that are not only predictive, but also physically interpretable,
diagnostically auditable, and usable for risk-aware decision support.

Why Nansha and Zhongshan matter
-------------------------------

The current land-subsidence application is built around two contrasting
urban regions in the Pearl River Delta: **Nansha** and **Zhongshan**.
This contrast is scientifically useful because the two settings do not
represent the same deformation regime.

In the current manuscript, Nansha is treated as a reclaimed coastal
delta with broader groundwater variability and more heterogeneous
compressible sediments, whereas Zhongshan is presented as a more
competent stratigraphic setting with clearer spatial organisation in
effective hydrogeological structure.

The two-city design therefore does more than provide two examples. It
tests whether one physics-guided framework can remain useful across
distinct subsidence regimes rather than only within a single local
basin :cite:p:`kouadio_geopriorsubsnet_nature_2025`.

What GeoPrior-v3 is
-------------------

GeoPrior-v3 is the broader software and documentation umbrella for
physics-guided AI workflows in geohazards. It is broader than a single 
urban land-subsidence problem. The present documentation
emphasises reproducible workflows, diagnostics, uncertainty analysis,
transfer evaluation, and interpretable physics-aware modelling.


What GeoPriorSubsNet brings
---------------------------

``GeoPriorSubsNet`` is the current application-specific model for urban
land subsidence. It combines an attentive spatio-temporal forecasting
backbone with reduced groundwater-flow and consolidation constraints,
while regularising the emergent relaxation timescale through a closure
based on effective hydraulic conductivity, specific storage, and
drainage thickness.

The goal is not only strong multi-horizon prediction. The framework is
also intended to deliver:

- physically auditable forecasts,
- calibrated uncertainty,
- interpretable effective fields,
- transfer-aware deployment diagnostics,
- and hotspot-oriented risk interpretation
  :cite:p:`kouadio_geopriorsubsnet_nature_2025`.

In that sense, GeoPrior is not only a model. It is a workflow for
linking geospatial data, reduced physics, uncertainty analysis,
diagnostics, and decision-facing interpretation in a single
reproducible framework :cite:p:`kouadio_geopriorsubsnet_nature_2025`.

How to cite the current land-subsidence application
---------------------------------------------------

If you use GeoPrior's current land-subsidence framework, the
GeoPriorSubsNet scientific concepts, or the Nansha/Zhongshan case
study, please cite the manuscript below.

.. admonition:: Preferred manuscript citation
   :class: note

   Kouadio, K. L., Liu, R., Jiang, S., Liu, Z., Kouamelan, S.,
   Liu, W., Qing, Z., and Zheng, Z. (2025).
   *Physics-Informed Deep Learning Reveals Divergent Urban Land
   Subsidence Regimes*.
   Unpublished manuscript. Submitted to *Nature Communications*.

BibTeX
------

.. code-block:: bibtex

   @unpublished{kouadio_geopriorsubsnet_nature_2025,
     author = {Kouadio, Kouao Laurent and Liu, Rong and Jiang, Shiyu and
               Liu, Zhuo and Kouamelan, Serge and Liu, Wenxiang and
               Qing, Zhanhui and Zheng, Zhiwen},
     title  = {Physics-Informed Deep Learning Reveals Divergent Urban Land Subsidence Regimes},
     note   = {Unpublished manuscript. Submitted to Nature Communications},
     year   = {2025}
   }

Related background in this documentation bibliography
-----------------------------------------------------

Readers who want a wider scientific background may find the following
themes especially useful:

- global and coastal subsidence risk
  :cite:p:`Shirzaeietal2021,Nichollsetal2021,Fangetal2022`
- groundwater extraction and regional subsidence
  :cite:p:`GallowayBurbey2011,Motaghetal2008,Chaussardetal2014,
  Abidinetal2011`
- anthropogenic and multi-driver subsidence interpretation
  :cite:p:`CandelaKoster2022`
- forecasting-model background
  :cite:p:`Liuetal2024,Limetal2021`

See also
--------

.. seealso::

   :doc:`references`
      Shared bibliography used across the documentation.