Quickstart
==========

This page gives the fastest practical path into GeoPrior-v3.

It is intentionally lightweight. The goal is not to explain
every scientific or architectural detail, but to help you:

- verify that GeoPrior-v3 is installed correctly,
- confirm that the CLI entry points are available,
- run a minimal Python sanity check,
- prepare for the first real workflow run.

If you have not installed the package yet, start with
:doc:`installation`. If you want the broader project
context, read :doc:`overview`.

What this page covers
---------------------

GeoPrior-v3 can be approached in two complementary ways:

- as a **Python package** for model development and
  experimentation;
- as a **workflow-oriented CLI** for configuration,
  staged runs, builds, and plotting.

For most new users, the fastest onboarding path is:

1. check the package import,
2. inspect the CLI help pages,
3. initialize or review configuration,
4. run a minimal model-level or workflow-level sanity check.

.. note::

   GeoPrior-v3 is built as a physics-guided framework for
   geohazards and risk analytics. The current flagship
   application is land subsidence through GeoPriorSubsNet
   v3.x, but the package is structured for broader growth
   over time.

1. Confirm the installation
---------------------------

After installation, verify that the package imports and that
the version is visible.

.. code-block:: bash

   python -c "import geoprior as gp; print(gp.__version__)"

A small interactive check also works:

.. code-block:: python

   import geoprior as gp

   print(gp.__version__)

If this succeeds, the base package is available in your
environment.

2. Inspect the CLI surface
--------------------------

GeoPrior-v3 provides a staged command-line workflow through
dedicated console entry points.

Start by checking that the commands resolve correctly:

.. code-block:: bash

   geoprior --help
   geoprior-run --help
   geoprior-build --help
   geoprior-plot --help
   geoprior-init --help

This is the quickest way to confirm that your environment is
ready for CLI-driven usage.

.. important::

   Do not guess the detailed subcommand syntax from memory.
   Use the help pages first. GeoPrior-v3 is organized around
   explicit staged commands and configuration-driven runs, so
   it is best to inspect the current command surface directly
   in your installed environment.

3. Initialize or inspect configuration
--------------------------------------

GeoPrior-v3 includes configuration resources and a dedicated
initialization entry point.

A common first move is to inspect the initialization help:

.. code-block:: bash

   geoprior-init --help

Depending on your local setup and current CLI design, this
command is the usual starting point for generating or copying
a configuration template before running the staged workflow.

After configuration initialization, the next pages to read
are usually:

- :doc:`../user_guide/configuration`
- :doc:`../user_guide/workflow_overview`
- :doc:`first_project_run`

4. Minimal Python smoke test
----------------------------

Before training a real model or running a full staged
pipeline, it is useful to do a minimal import-level smoke
test.

.. code-block:: python

   import geoprior as gp

   print("GeoPrior version:", gp.__version__)

This confirms that the package import works in the active
environment.

You can also check that warning suppression and optional
package behavior are accessible:

.. code-block:: python

   import geoprior as gp

   gp.suppress_warnings(True)
   print("GeoPrior is ready.")

5. Model-level bring-up with synthetic tensors
----------------------------------------------

Once the package import works, a useful next step is a small
model-level sanity check using synthetic data. The purpose is
not scientific validity. It is simply to verify that:

- dict inputs have the expected structure,
- the forward pass works,
- compilation and a short fit loop run end to end.

The current subsidence model family is organized under the
GeoPrior subsidence modules. A minimal synthetic example
looks like this:

.. code-block:: python

   import tensorflow as tf

   try:
       import keras
   except Exception:  # pragma: no cover
       from tensorflow import keras

   from geoprior.models import GeoPriorSubsNet

   # Example dimensions
   B = 8    # batch size
   T = 12   # past window length
   H = 3    # forecast horizon
   S = 4    # static feature dimension
   D = 10   # dynamic feature dimension
   F = 6    # future-known feature dimension

   x = {
       "static_features": tf.random.normal([B, S]),
       "dynamic_features": tf.random.normal([B, T, D]),
       "future_features": tf.random.normal([B, H, F]),
       "coords": tf.random.normal([B, H, 3]),
       "H_field": tf.ones([B, H, 1]) * 50.0,
   }

   y = {
       "subs_pred": tf.random.normal([B, H, 1]),
       "gwl_pred": tf.random.normal([B, H, 1]),
   }

   scaling_kwargs = {
       "time_units": "year",
       "coords_normalized": False,
       "coords_in_degrees": False,
       "coord_order": ["t", "x", "y"],
       "subs_scale_si": 1.0,
       "head_scale_si": 1.0,
       "H_scale_si": 1.0,
       "subsidence_kind": "cumulative",
   }

   model = GeoPriorSubsNet(
       static_input_dim=S,
       dynamic_input_dim=D,
       future_input_dim=F,
       forecast_horizon=H,
       max_window_size=T,
       pde_mode="none",
       scaling_kwargs=scaling_kwargs,
   )

   model.compile(
       optimizer=keras.optimizers.Adam(1e-3),
       loss={"subs_pred": "mse", "gwl_pred": "mse"},
   )

   history = model.fit(x, y, epochs=2, verbose=1)

   outputs = model(x, training=False)
   print("Output keys:", list(outputs.keys()))
   print("subs_pred shape:", outputs["subs_pred"].shape)
   print("gwl_pred shape:", outputs["gwl_pred"].shape)

This example starts with ``pde_mode="none"`` on purpose.
That is usually the best bring-up mode when you only want to
validate tensor plumbing and output structure first.

6. Enable physics after the shapes are correct
----------------------------------------------

Once the synthetic forward pass works, you can move toward
physics-guided execution.

A common next step is to switch from pure bring-up mode to a
physics-enabled setting such as:

- ``pde_mode="both"``
- explicit residual configuration
- physics loss weights at compile time

For example:

.. code-block:: python

   model_phys = GeoPriorSubsNet(
       static_input_dim=S,
       dynamic_input_dim=D,
       future_input_dim=F,
       forecast_horizon=H,
       max_window_size=T,
       pde_mode="both",
       residual_method="exact",
       scale_pde_residuals=True,
       scaling_kwargs=scaling_kwargs,
   )

   model_phys.compile(
       optimizer=keras.optimizers.Adam(1e-3),
       loss={"subs_pred": "mse", "gwl_pred": "mse"},
       lambda_cons=1.0,
       lambda_gw=1.0,
       lambda_prior=1.0,
       lambda_smooth=0.1,
       lambda_bounds=0.0,
       lambda_q=0.0,
       lambda_offset=0.1,
   )

   history = model_phys.fit(x, y, epochs=2, verbose=1)

At this point, the model is no longer just testing tensor
shapes. It is beginning to exercise the physics-guided loss
assembly.

.. important::

   Physics-guided runs depend strongly on correct scaling,
   coordinate conventions, and units. If these are wrong,
   physics residuals can become misleading or unstable even
   when supervised losses appear to decrease normally.

   Read :doc:`../scientific_foundations/data_and_units` and
   :doc:`../scientific_foundations/scaling` before trusting
   a physics-enabled run.

7. Move from synthetic data to real workflow artifacts
------------------------------------------------------

For real usage, you will typically not handcraft tensors as
in the synthetic example above. Instead, you will work from
prepared artifacts, configuration files, and stage outputs.

The practical pattern is usually:

1. initialize or load configuration,
2. prepare inputs through the staged workflow,
3. inspect diagnostics and scaling,
4. run training or inference,
5. export results and generate plots.

That is why the next pages after this one are usually more
important than going deeper into synthetic examples.

Recommended next pages
----------------------

If you have just finished the quickstart, the best next page
depends on what you want to do.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: First project run
      :link: first_project_run
      :link-type: doc
      :class-card: sd-shadow-sm

      See the first end-to-end path from installation and
      configuration toward a real workflow run.

   .. grid-item-card:: Workflow overview
      :link: ../user_guide/workflow_overview
      :link-type: doc
      :class-card: sd-shadow-sm

      Understand how GeoPrior-v3 is organized as a staged
      workflow rather than only as a model API.

   .. grid-item-card:: Configuration
      :link: ../user_guide/configuration
      :link-type: doc
      :class-card: sd-shadow-sm

      Learn how configuration files, templates, and run-time
      settings are organized.

   .. grid-item-card:: Models overview
      :link: ../scientific_foundations/models_overview
      :link-type: doc
      :class-card: sd-shadow-sm

      Move from the quickstart into the scientific logic of
      the model family and the physics-guided design.

.. seealso::

   - :doc:`installation`
   - :doc:`first_project_run`
   - :doc:`../user_guide/cli`
   - :doc:`../user_guide/configuration`
   - :doc:`../scientific_foundations/data_and_units`
   - :doc:`../scientific_foundations/scaling`