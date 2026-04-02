Utility API reference
=====================

GeoPrior-v3 exposes utilities across **three complementary layers**:

- :mod:`geoprior.utils`
  for **workflow-facing utilities** used across staged runs,
  preprocessing, forecasting, diagnostics, calibration,
  evaluation, export, and artifact handling;

- :mod:`geoprior.models.utils`
  for **model-facing utilities** used closer to forecasting
  models, sequence construction, PINN input preparation,
  tensor formatting, and PDE mode normalization;

- :mod:`geoprior.models.subsidence.utils`
  for **subsidence-physics utilities** used to canonicalize
  scaling metadata, convert between model and SI units,
  resolve groundwater and subsidence channels, and support
  head/depth transformations and physics-aware priors.

This page documents all three layers together because they are
not isolated in practice. A typical GeoPrior workflow uses them
as a stack:

- the workflow layer prepares data, resolves config, and drives
  staged execution;
- the model layer standardizes sequence handling, model inputs,
  forecast formatting, and PINN support functions;
- the subsidence-physics layer reconciles units, aliases,
  coordinate scaling, groundwater conventions, and model-side
  physical state extraction.

Overview
--------

A useful mental map is:

.. code-block:: text

   geoprior.utils
      ├── audits / handshakes
      ├── config and NAT helpers
      ├── data / IO / export helpers
      ├── forecast formatting and evaluation
      ├── holdout and split logic
      ├── spatial and geospatial utilities
      ├── sequence and shape helpers
      └── subsidence-oriented workflow conversions

   geoprior.models.utils
      ├── general model helpers
      ├── sequence construction
      ├── forecast formatting
      ├── input packing / unpacking
      ├── anomaly and evaluation helpers
      └── PINN-specific helper functions

   geoprior.models.subsidence.utils
      ├── scaling alias normalization
      ├── SI conversion helpers
      ├── coordinate scaling helpers
      ├── groundwater / head conversion
      ├── dynamic channel resolution
      └── history / reference-state extraction

Why three utility layers exist
------------------------------

The separation is architectural rather than cosmetic.

Workflow surface
~~~~~~~~~~~~~~~~

The top-level :mod:`geoprior.utils` package is the best entry
point when reading Stage-1 through Stage-5 workflows, figure
scripts, evaluation paths, and export code. Its public exports
already include config loaders, dataset builders, calibration
helpers, forecast formatters, holdout logic, geospatial helpers,
and subsidence-oriented postprocessing utilities.

Model surface
~~~~~~~~~~~~~

The :mod:`geoprior.models.utils` package complements the
workflow layer by handling lower-level model concerns such as
sequence construction, preparing input tuples,
forecast-to-DataFrame formatting, and PINN-oriented helper
functions. The associated module docstrings explicitly position
this layer in the context of sequence forecasting and
Temporal-Fusion-Transformer-style workflows, which is useful for
readers trying to connect GeoPrior's utilities to broader
multi-horizon forecasting practice.

Subsidence-physics surface
~~~~~~~~~~~~~~~~~~~~~~~~~~

The :mod:`geoprior.models.subsidence.utils` module is narrower
and more physics-aware. It is where scaling metadata is made
canonical, SI conversions are enforced, coordinate policies are
checked, groundwater depth can be reconciled with hydraulic head,
and dynamic-channel lookup is stabilized. This layer is central
when readers need to understand how raw workflow tensors become
physically interpretable model quantities.

Reading order
-------------

A practical order for new readers is:

1. :mod:`geoprior.utils.nat_utils`
2. :mod:`geoprior.utils.forecast_utils`
3. :mod:`geoprior.utils.subsidence_utils`
4. :mod:`geoprior.models.utils`
5. :mod:`geoprior.models.utils.pinn`
6. :mod:`geoprior.models.subsidence.utils`

That sequence moves from the workflow surface into the model
surface and finally into the more specialized subsidence-physics
support layer.

Module indexes
--------------

These indexes use **explicit fully qualified module names**
rather than recursive package discovery. That keeps autosummary
predictable and avoids ambiguous package-relative lookup.

Workflow utility modules
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/utils_workflow
   :nosignatures:

   ~geoprior.utils.audit_utils
   ~geoprior.utils.base_utils
   ~geoprior.utils.calibrate
   ~geoprior.utils.data_utils
   ~geoprior.utils.deps_utils
   ~geoprior.utils.forecast_utils
   ~geoprior.utils.generic_utils
   ~geoprior.utils.geo_utils
   ~geoprior.utils.holdout_utils
   ~geoprior.utils.io_utils
   ~geoprior.utils.nat_utils
   ~geoprior.utils.parallel_utils
   ~geoprior.utils.scale_metrics
   ~geoprior.utils.sequence_utils
   ~geoprior.utils.shapes
   ~geoprior.utils.spatial_utils
   ~geoprior.utils.split
   ~geoprior.utils.subsidence_utils
   ~geoprior.utils.sys_utils
   ~geoprior.utils.target_utils
   ~geoprior.utils.validator
   ~geoprior.utils.version

Model utility modules
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/utils_models
   :nosignatures:

   ~geoprior.models.utils._utils
   ~geoprior.models.utils.pinn

Subsidence-physics utility modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/utils_subsidence
   :nosignatures:

   ~geoprior.models.subsidence.utils

Top-level workflow utility package
----------------------------------

The public package surface below documents the aggregated export
layer of :mod:`geoprior.utils`.

.. automodule:: geoprior.utils
   :members:
   :undoc-members:
   :show-inheritance:

What this package covers
~~~~~~~~~~~~~~~~~~~~~~~~

The top-level utility package is intentionally broad because it
supports the staged application workflow end to end.

Its current public exports include helper groups such as:

- **audits and handshakes**
  including ``audit_stage1_scaling``,
  ``audit_stage2_handshake``, and ``should_audit``;

- **calibration**
  including ``calibrate_quantile_forecasts``;

- **forecast formatting and evaluation**
  including ``format_and_forecast``,
  ``evaluate_forecast``, and
  ``pivot_forecast_dataframe``;

- **configuration and stage helpers**
  including ``load_nat_config``,
  ``load_nat_config_payload``,
  ``make_tf_dataset``,
  ``map_targets_for_training``,
  ``resolve_hybrid_config``,
  ``resolve_si_affine``, and
  ``serialize_subs_params``;

- **geospatial and spatial workflow helpers**
  including ``spatial_sampling``,
  ``create_spatial_clusters``,
  ``augment_city_spatiotemporal_data``, and
  ``deg_to_m_from_lat``;

- **holdout logic**
  including ``compute_group_masks``,
  ``split_groups_holdout``, and
  ``filter_df_by_groups``;

- **subsidence-oriented workflow helpers**
  including ``convert_eval_payload_units``,
  ``cumulative_to_rate``,
  ``rate_to_cumulative``,
  ``resolve_gwl_for_physics``,
  ``resolve_head_column``, and
  ``make_txy_coords``.

Important workflow-facing modules
---------------------------------

Audit helpers
~~~~~~~~~~~~~

.. automodule:: geoprior.utils.audit_utils
   :members:
   :undoc-members:
   :show-inheritance:

These helpers matter because they validate Stage-1 scaling and
Stage-2 handoff assumptions before later model or physics-aware
runs rely on them.

Calibration helpers
~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.calibrate
   :members:
   :undoc-members:
   :show-inheritance:

The calibration layer is especially relevant to uncertainty
workflows spanning Stage-2 through Stage-5 and any evaluation
path that compares interval quality before and after calibration.

Forecast helpers
~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.forecast_utils
   :members:
   :undoc-members:
   :show-inheritance:

This is one of the most frequently visited workflow helper
modules because it turns saved or live outputs into reusable
forecast tables, aligned metrics, and export-ready structures.

Generic helpers
~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.generic_utils
   :members:
   :undoc-members:
   :show-inheritance:

This module contains reusable helpers for paths, result folders,
configuration display, and figure-saving behavior.

Geo/spatial helpers
~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.geo_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geoprior.utils.spatial_utils
   :members:
   :undoc-members:
   :show-inheritance:

Together, these modules support spatial augmentation,
sampling, clustering, dummy data generation, and degree-to-meter
conversion, which are all important in Stage-1 construction,
spatial forecasting, and map-oriented analysis.

Holdout helpers
~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.holdout_utils
   :members:
   :undoc-members:
   :show-inheritance:

These utilities are especially relevant when the workflow needs
group-aware, spatially constrained, or otherwise structured
holdout logic instead of purely random splitting.

IO, system, and runtime helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.io_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geoprior.utils.parallel_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geoprior.utils.sys_utils
   :members:
   :undoc-members:
   :show-inheritance:

These modules help manage serialization, joblib artifacts,
threading, runtime-device behavior, and larger workflow-side
assembly patterns.

Scale, shape, and sequence helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.scale_metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geoprior.utils.sequence_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geoprior.utils.shapes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geoprior.utils.target_utils
   :members:
   :undoc-members:
   :show-inheritance:

These helpers support inverse target scaling, sequence
construction, canonical layout handling, and target-aware
postprocessing.

Validation and helper infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.base_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geoprior.utils.deps_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geoprior.utils.split
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geoprior.utils.validator
   :members:
   :undoc-members:
   :show-inheritance:

These modules form the lighter infrastructure layer behind the
more visible staged helpers.

NAT/workflow contract helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.nat_utils
   :members:
   :undoc-members:
   :show-inheritance:

This is one of the most important workflow modules in the
GeoPrior stack. It includes config loading, dataset building,
target mapping, scaling resolution, prediction extraction,
artifact serialization, and model-evaluation support.

Subsidence-oriented workflow helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.subsidence_utils
   :members:
   :undoc-members:
   :show-inheritance:

This module is the workflow-side bridge back to the subsidence
application. It exposes unit conversion helpers,
cumulative/rate transforms, groundwater-resolution helpers,
head-column selection, and coordinate construction.

Selected workflow helpers
-------------------------

These functions are surfaced explicitly because they appear
frequently in staged workflows, export paths, and figure or
reproducibility scripts.

.. autofunction:: geoprior.utils.load_nat_config

.. autofunction:: geoprior.utils.load_nat_config_payload

.. autofunction:: geoprior.utils.make_tf_dataset

.. autofunction:: geoprior.utils.map_targets_for_training

.. autofunction:: geoprior.utils.resolve_si_affine

.. autofunction:: geoprior.utils.extract_preds

.. autofunction:: geoprior.utils.best_epoch_and_metrics

.. autofunction:: geoprior.utils.audit_stage1_scaling

.. autofunction:: geoprior.utils.audit_stage2_handshake

.. autofunction:: geoprior.utils.calibrate_quantile_forecasts

.. autofunction:: geoprior.utils.evaluate_forecast

.. autofunction:: geoprior.utils.format_and_forecast

.. autofunction:: geoprior.utils.create_spatial_clusters

.. autofunction:: geoprior.utils.spatial_sampling

.. autofunction:: geoprior.utils.make_txy_coords

Model-facing utility package
----------------------------

The model-side utility package complements the workflow
surface and is closer to data layout, model I/O contracts,
sequence generation, and PINN-oriented helper logic.

.. automodule:: geoprior.models.utils
   :members:
   :undoc-members:
   :show-inheritance:

What this package covers
~~~~~~~~~~~~~~~~~~~~~~~~

The public exports of :mod:`geoprior.models.utils` combine
**general model helpers** from :mod:`geoprior.models.utils._utils`
with **PINN-specific helpers** from
:mod:`geoprior.models.utils.pinn`.

The current export surface includes functions such as:

- ``prepare_model_inputs`` and ``prepare_model_inputs_in``
  for standardizing model input tuples;
- ``create_sequences`` and ``split_static_dynamic``
  for sequence preparation;
- ``forecast_single_step`` and ``forecast_multi_step``
  for rollout helpers;
- ``format_predictions_to_dataframe`` and
  ``format_pinn_predictions`` for output formatting;
- ``prepare_pinn_data_sequences`` and ``normalize_for_pinn``
  for PINN preparation;
- ``extract_txy`` and ``extract_txy_in`` for coordinate
  extraction;
- ``process_pde_modes`` and ``PDE_MODE_ALIASES`` for
  PDE-mode normalization.

General model helper module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.models.utils._utils
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

This module contains many of the reusable helpers for sequence
forecasting and model-side formatting. Its own module docstring
frames the utilities in the context of Temporal Fusion
Transformer-style preprocessing and multi-horizon forecasting,
which gives the page valuable methodological context.

PINN/model physics helper module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.models.utils.pinn
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

This module is the entry point for PINN-specific utility logic,
including PDE mode normalization, coordinate extraction,
PINN prediction formatting, and related helpers used near the
physics-informed model layer.

Selected model-facing helpers
-----------------------------

.. autofunction:: geoprior.models.utils.prepare_model_inputs
   :no-index:

.. autofunction:: geoprior.models.utils.prepare_model_inputs_in
   :no-index:

.. autofunction:: geoprior.models.utils.create_sequences
   :no-index:

.. autofunction:: geoprior.models.utils.split_static_dynamic
   :no-index:

.. autofunction:: geoprior.models.utils.forecast_single_step
   :no-index:

.. autofunction:: geoprior.models.utils.forecast_multi_step
   :no-index:

.. autofunction:: geoprior.models.utils.format_predictions_to_dataframe
   :no-index:

.. autofunction:: geoprior.models.utils.extract_batches_from_dataset
   :no-index:

.. autofunction:: geoprior.models.utils.get_tensor_from
   :no-index:

.. autofunction:: geoprior.models.utils.compute_anomaly_scores
   :no-index:

.. autodata:: geoprior.models.utils.PDE_MODE_ALIASES
   :no-index:

.. autofunction:: geoprior.models.utils.prepare_pinn_data_sequences
   :no-index:

.. autofunction:: geoprior.models.utils.normalize_for_pinn
   :no-index:

.. autofunction:: geoprior.models.utils.format_pinn_predictions
   :no-index:

.. autofunction:: geoprior.models.utils.extract_txy
   :no-index:

.. autofunction:: geoprior.models.utils.extract_txy_in
   :no-index:

.. autofunction:: geoprior.models.utils.process_pde_modes
   :no-index:

Subsidence-physics utility layer
--------------------------------

The subsidence-physics utility layer is more specialized than the
other two layers. It is not simply another helper bucket; it is a
coherence layer for unit systems, scaling metadata, coordinate
policies, historical states, and hydrogeological interpretation.

.. automodule:: geoprior.models.subsidence.utils
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

What this module covers
~~~~~~~~~~~~~~~~~~~~~~~

The current implementation includes several strongly related
subfamilies of helpers:

- **scaling canonicalization and policy enforcement**
  such as ``canonicalize_scaling_kwargs``,
  ``validate_scaling_kwargs``,
  ``enforce_scaling_alias_consistency``, and
  ``finalize_scaling_kwargs``;

- **SI conversion helpers**
  such as ``to_si_thickness``,
  ``to_si_head``,
  ``to_si_subsidence``, and
  ``from_si_subsidence``;

- **coordinate helpers**
  such as ``deg_to_m``,
  ``coord_ranges``, and
  ``coord_ranges_si``;

- **dynamic-channel resolution**
  such as ``resolve_gwl_dyn_index``,
  ``resolve_subs_dyn_index``,
  ``get_gwl_dyn_index_cached``, and
  ``get_subs_dyn_index_cached``;

- **groundwater/head reconciliation**
  such as ``gwl_to_head_m`` and
  ``get_h_ref_si``;

- **history and reference-state extraction**
  such as ``get_h_hist_si`` and
  ``get_s_init_si``.

Why this module matters
~~~~~~~~~~~~~~~~~~~~~~~

A large part of the difficulty in physics-informed forecasting
is not the PDE term alone, but the consistency of units,
metadata, and conventions. This module exists to make those
conventions explicit. It handles alias drift in scaling keys,
coordinates recorded in degrees versus projected meters,
depth-versus-head ambiguity for groundwater, and the lookup of
historical channels used to initialize or interpret model
states.

Selected subsidence-physics helpers
-----------------------------------

.. autofunction:: geoprior.models.subsidence.utils.canonicalize_scaling_kwargs
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.validate_scaling_kwargs
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.enforce_scaling_alias_consistency
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.to_si_thickness
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.to_si_head
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.to_si_subsidence
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.from_si_subsidence
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.deg_to_m
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.resolve_gwl_dyn_index
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.resolve_subs_dyn_index
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.gwl_to_head_m
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.get_h_hist_si
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.get_s_init_si
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.get_h_ref_si
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.policy_gate
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.finalize_scaling_kwargs
   :no-index:

.. autofunction:: geoprior.models.subsidence.utils.coord_ranges_si
   :no-index:

Connections to the scientific stack
-----------------------------------

These utility layers help explain how GeoPrior-v3 connects
workflow orchestration, forecasting abstractions, and
physics-aware subsidence modeling.

In particular:

- the workflow layer explains how artifacts, staged runs,
  evaluation tables, and calibrated outputs are prepared;
- the model layer explains how generic forecasting utilities
  interact with sequence models and PINN-style helper logic;
- the subsidence-physics layer explains how scaling metadata,
  hydraulic-head interpretation, and state initialization stay
  coherent when models are applied to land-subsidence problems.

For readers coming from a forecasting background, the
:mod:`geoprior.models.utils._utils` docstrings are useful because
they frame several helpers explicitly in the language of sequence
forecasting and Temporal Fusion Transformer workflows. For readers
coming from hydrogeology or subsidence physics, the
:mod:`geoprior.models.subsidence.utils` layer is the most direct
explanation of how physically meaningful quantities are recovered
from model-side tensors and metadata.

Source listings with comments
-----------------------------

These source listings are included intentionally so the inline
implementation comments remain visible in the documentation.

Top-level utility exports
~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../geoprior/utils/__init__.py
   :language: python
   :caption: geoprior/utils/__init__.py

Model utility exports
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../geoprior/models/utils/__init__.py
   :language: python
   :caption: geoprior/models/utils/__init__.py

Subsidence utility implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../geoprior/models/subsidence/utils.py
   :language: python
   :caption: geoprior/models/subsidence/utils.py

NAT package exports
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../geoprior/utils/nat_utils/__init__.py
   :language: python
   :caption: geoprior/utils/nat_utils/__init__.py

Optional: workflow/NAT helper module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../geoprior/utils/nat_utils/natutils.py
   :language: python
   :caption: geoprior/utils/nat_utils/natutils.py

Optional: model/PINN helper module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../geoprior/models/utils/pinn.py
   :language: python
   :caption: geoprior/models/utils/pinn.py

API notes
---------

A few practical notes help orient readers:

- :mod:`geoprior.utils` is the best starting point for staged
  workflow code, diagnostics, export, and reproducibility
  scripts.
- :mod:`geoprior.models.utils` is the better starting point for
  model-input preparation, sequence construction, and PINN-side
  formatting helpers.
- :mod:`geoprior.models.subsidence.utils` is the best place to
  inspect scaling, SI conversion, depth/head conventions,
  and reference-state extraction logic.
- The three layers are meant to complement one another rather
  than compete for the same role.

See also
--------

.. seealso::

   - :doc:`../user_guide/configuration`
   - :doc:`../user_guide/diagnostics`
   - :doc:`../user_guide/inference_and_export`
   - :doc:`../applications/reproducibility_scripts`
   - :doc:`../applications/figure_generation`
   - :doc:`subsidence`
   - :doc:`tuner`
   - :doc:`cli`
   - :doc:`resources`
