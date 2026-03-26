Utility API reference
=====================

GeoPrior-v3 exposes utilities at two complementary levels:

- :mod:`geoprior.utils`
  for **workflow-facing utilities** used across staged runs,
  preprocessing, forecasting, evaluation, diagnostics, and
  export;

- :mod:`geoprior.models.utils`
  for **model-facing helpers** used more directly by PINN and
  forecasting model code.

This page documents both layers, because in practice they
work together:

- the top-level utility package helps drive the staged
  workflow;
- the model utility package helps normalize inputs, forecast
  layouts, PDE mode handling, and model-side tensor logic.

Overview
--------

A useful mental map is:

.. code-block:: text

   geoprior.utils
      ├── workflow audits
      ├── config / IO helpers
      ├── forecast formatting
      ├── holdout / split logic
      ├── spatial utilities
      ├── calibration helpers
      └── subsidence-oriented conversions

   geoprior.models.utils
      ├── PINN helpers
      ├── model input preparation
      ├── PDE mode aliases
      ├── forecast formatting
      └── tensor / sequence helpers

The current top-level ``geoprior.utils`` package already
re-exports helpers spanning audits, calibration, forecast
evaluation, config loading, data generation, holdout logic,
parallel/runtime helpers, sequence building, and
subsidence-oriented postprocessing.
The current ``geoprior.models.utils`` package re-exports
model-side helpers such as ``PDE_MODE_ALIASES``,
``prepare_pinn_data_sequences``, ``process_pde_modes``,
``prepare_model_inputs``, and forecasting/formatting
utilities.

Module index
------------

.. autosummary::
   :toctree: generated/


   geoprior.utils
   geoprior.utils.audit_utils
   geoprior.utils.calibrate
   geoprior.utils.data_utils
   geoprior.utils.forecast_utils
   geoprior.utils.generic_utils
   geoprior.utils.geo_utils
   geoprior.utils.holdout_utils
   geoprior.utils.io_utils
   geoprior.utils.nat_utils
   geoprior.utils.parallel_utils
   geoprior.utils.scale_metrics
   geoprior.utils.sequence_utils
   geoprior.utils.shapes
   geoprior.utils.spatial_utils
   geoprior.utils.subsidence_utils

   geoprior.models.utils
   geoprior.models.utils._utils
   geoprior.models.utils.pinn

Top-level workflow utilities
----------------------------

The main utility package gathers reusable helpers for the
staged workflow and wider application layer.

.. automodule:: geoprior.utils
   :members:
   :undoc-members:
   :show-inheritance:

What this package covers
~~~~~~~~~~~~~~~~~~~~~~~~

The current public exports of :mod:`geoprior.utils` include
the following helper groups:

- **audits**
  such as ``audit_stage1_scaling``,
  ``audit_stage2_handshake``, and ``should_audit``;

- **calibration**
  such as ``calibrate_quantile_forecasts``;

- **forecast formatting and evaluation**
  such as ``format_and_forecast``,
  ``evaluate_forecast``, and
  ``pivot_forecast_dataframe``;

- **configuration and Stage helpers**
  such as ``load_nat_config``,
  ``load_nat_config_payload``,
  ``make_tf_dataset``,
  ``map_targets_for_training``,
  ``resolve_hybrid_config``,
  and ``resolve_si_affine``;

- **spatial and geospatial utilities**
  such as ``spatial_sampling``,
  ``create_spatial_clusters``,
  ``augment_city_spatiotemporal_data``,
  and ``deg_to_m_from_lat``;

- **holdout logic**
  such as ``compute_group_masks``,
  ``split_groups_holdout``,
  and ``filter_df_by_groups``;

- **subsidence-oriented helpers**
  such as ``convert_eval_payload_units``,
  ``cumulative_to_rate``,
  ``rate_to_cumulative``,
  ``resolve_gwl_for_physics``,
  ``resolve_head_column``,
  and ``make_txy_coords``.

Important workflow-facing modules
---------------------------------

Audit helpers
~~~~~~~~~~~~~

.. automodule:: geoprior.utils.audit_utils
   :members:
   :undoc-members:
   :show-inheritance:

These helpers are especially important for the staged
workflow because they validate Stage-1 scaling and Stage-2
handoff assumptions before later physics-aware runs rely on
them.

Calibration helpers
~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.calibrate
   :members:
   :undoc-members:
   :show-inheritance:

The top-level utility API includes quantile calibration
support through ``calibrate_quantile_forecasts``,
which makes this module relevant to Stage-2 through Stage-5
uncertainty workflows.

Forecast helpers
~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.forecast_utils
   :members:
   :undoc-members:
   :show-inheritance:

This module is the most directly relevant workflow utility
layer for turning saved or live outputs into usable forecast
tables and metrics. The public exports include
``evaluate_forecast``, ``format_and_forecast``, and
``pivot_forecast_dataframe``.

Generic helpers
~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.generic_utils
   :members:
   :undoc-members:
   :show-inheritance:

This module contains general reusable workflow helpers such
as default result-directory handling, path preparation,
configuration display, and figure saving.

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
sampling, clustering, dummy data generation, and degree to
meter conversion, all of which are useful for Stage-1 data
construction and spatial forecasting workflows.

Holdout helpers
~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.holdout_utils
   :members:
   :undoc-members:
   :show-inheritance:

These helpers are especially relevant when the workflow needs
group-aware or spatially structured holdout logic, rather
than only simple random splitting.

NAT/workflow contract helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.nat_utils
   :members:
   :undoc-members:
   :show-inheritance:

This is one of the most important utility modules in the
GeoPrior workflow layer. The public exports include config
loading, dataset building, target mapping, scaling resolution,
prediction extraction, and artifact serialization helpers
such as ``load_nat_config``, ``make_tf_dataset``,
``resolve_si_affine``, ``extract_preds``,
and ``save_ablation_record``.

Parallel/runtime helpers
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.parallel_utils
   :members:
   :undoc-members:
   :show-inheritance:

These helpers manage CPU/GPU environment settings, thread
allocation, and device resolution for workflow execution
across different runtime environments.

Scale and metric helpers
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.scale_metrics
   :members:
   :undoc-members:
   :show-inheritance:

These helpers support point forecast evaluation and inverse
target scaling, which are especially useful when exported
metrics and presentation units differ.

Sequence and shape helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.sequence_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: geoprior.utils.shapes
   :members:
   :undoc-members:
   :show-inheritance:

These modules provide reusable support for future-sequence
construction and canonical tensor layout handling, which
helps keep Stage-1 and later inference steps consistent.

Subsidence-oriented workflow helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.utils.subsidence_utils
   :members:
   :undoc-members:
   :show-inheritance:

This module is the workflow-side bridge back to the
subsidence application. Its public exports include helpers
for unit conversion, cumulative/rate transforms, groundwater
resolution for physics, head-column selection, and
coordinate construction.

Selected top-level helpers
--------------------------

These are especially important to surface explicitly because
they appear frequently in the staged workflow and related
scripts.

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
utility layer.

.. automodule:: geoprior.models.utils
   :members:
   :undoc-members:
   :show-inheritance:

What this package covers
~~~~~~~~~~~~~~~~~~~~~~~~

The current public exports of :mod:`geoprior.models.utils`
include two main groups:

- **general model helpers**
  from ``._utils``, such as
  ``prepare_model_inputs``,
  ``prepare_model_inputs_in``,
  ``create_sequences``,
  ``forecast_single_step``,
  ``forecast_multi_step``,
  ``format_predictions_to_dataframe``,
  ``extract_batches_from_dataset``,
  and ``make_dict_to_tuple_fn``;

- **PINN-specific helpers**
  from ``.pinn``, such as
  ``PDE_MODE_ALIASES``,
  ``prepare_pinn_data_sequences``,
  ``normalize_for_pinn``,
  ``format_pinn_predictions``,
  ``extract_txy``,
  ``extract_txy_in``,
  ``plot_hydraulic_head``,
  and ``process_pde_modes``.

General model helper module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.models.utils._utils
   :members:
   :undoc-members:
   :show-inheritance:

PINN/model physics helper module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.models.utils.pinn
   :members:
   :undoc-members:
   :show-inheritance:

Selected model-facing helpers
-----------------------------

These functions are especially worth surfacing because they
are the most reusable entry points when working with model
inputs and PDE-aware helpers directly.

.. autofunction:: geoprior.models.utils.prepare_model_inputs

.. autofunction:: geoprior.models.utils.prepare_model_inputs_in

.. autofunction:: geoprior.models.utils.create_sequences

.. autofunction:: geoprior.models.utils.forecast_single_step

.. autofunction:: geoprior.models.utils.forecast_multi_step

.. autofunction:: geoprior.models.utils.format_predictions_to_dataframe

.. autofunction:: geoprior.models.utils.extract_batches_from_dataset

.. autofunction:: geoprior.models.utils.PDE_MODE_ALIASES

.. autofunction:: geoprior.models.utils.prepare_pinn_data_sequences

.. autofunction:: geoprior.models.utils.normalize_for_pinn

.. autofunction:: geoprior.models.utils.format_pinn_predictions

.. autofunction:: geoprior.models.utils.extract_txy

.. autofunction:: geoprior.models.utils.extract_txy_in

.. autofunction:: geoprior.models.utils.process_pde_modes

Why both utility layers matter
------------------------------

The utility API is intentionally split because GeoPrior-v3 is
both:

- a **workflow framework**, with staged execution and
  artifact management;
- and a **model framework**, with PINN-aware input contracts,
  forecasting helpers, and PDE-mode normalization.

The top-level utility package supports the workflow surface
used in Stage-1 through Stage-5 and related figure or export
scripts. The model utility
package supports lower-level tensor preparation, sequence
construction, prediction formatting, and PINN mode handling
used closer to the neural model layer.

Suggested reading order
-----------------------

If you are new to the utility layer, a good order is:

1. :mod:`geoprior.utils.nat_utils`
2. :mod:`geoprior.utils.forecast_utils`
3. :mod:`geoprior.utils.subsidence_utils`
4. :mod:`geoprior.utils.audit_utils`
5. :mod:`geoprior.models.utils`
6. :mod:`geoprior.models.utils.pinn`

That gives a clean path from workflow-facing helpers to
model-facing tensor and PDE utilities.

Source listings with comments
-----------------------------

These source listings are included intentionally so the
inline implementation comments remain visible in the
documentation.

Top-level utility exports
~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/utils/__init__.py
   :language: python
   :caption: geoprior/utils/__init__.py

Model utility exports
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/models/utils/__init__.py
   :language: python
   :caption: geoprior/models/utils/__init__.py

Optional: key workflow helper module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/utils/nat_utils.py
   :language: python
   :caption: geoprior/utils/nat_utils.py

Optional: key model/PINN helper module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/models/utils/pinn.py
   :language: python
   :caption: geoprior/models/utils/pinn.py

API notes
---------

A few practical notes are worth keeping in mind when reading
this API page:

- :mod:`geoprior.utils` is the better starting point for
  staged workflow code, diagnostics, export, and scripts.
- :mod:`geoprior.models.utils` is the better starting point
  for low-level model-input preparation and PINN helper logic.
- The top-level utility package already re-exports many of
  the helpers most users need, so direct submodule imports are
  often optional.
- The model utility package is intentionally smaller and more
  focused on tensor layout, sequence handling, and PDE-aware
  helper logic.

See also
--------

.. seealso::

   - :doc:`../user_guide/configuration`
   - :doc:`../user_guide/diagnostics`
   - :doc:`../user_guide/inference_and_export`
   - :doc:`../applications/reproducibility_scripts`
   - :doc:`../applications/figure_generation`
   - :doc:`subsidence`
   - :doc:`cli`
   - :doc:`resources`