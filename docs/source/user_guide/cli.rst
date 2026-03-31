CLI in the workflow
===================

GeoPrior provides a **family-based command-line interface** for staged
execution, artifact building, and figure generation.

In practice, the CLI is the layer you use when you are ready to
**do something concrete**:

- run a stage,
- build an artifact,
- render a figure,
- initialize or adjust a configuration.

This page is here to help you place the CLI inside the broader workflow.

Where the CLI fits
------------------

A simple way to think about the documentation is:

- the **User Guide** explains the workflow and why each stage matters,
- the **CLI pages** show which command to run for a given task,
- the **Examples gallery** shows complete worked lessons with outputs,
  figures, and interpretation.

So when you already know you want to run a command, the best move is to
jump into the dedicated CLI pages below. When you want the reasoning,
context, and sequence behind the workflow, stay in the User Guide.

Start here for commands
-----------------------

.. container:: cta-tiles cli-tiles

   .. card:: CLI overview
      :link: ../cli/index
      :link-type: doc

      Start with the main CLI landing page for the overall structure,
      command families, and reading path.

   .. card:: CLI introduction
      :link: ../cli/introduction
      :link-type: doc

      Learn the mental model behind ``geoprior``,
      ``geoprior-run``, ``geoprior-build``, and ``geoprior-plot``.

   .. card:: Shared conventions
      :link: ../cli/shared_conventions
      :link-type: doc

      See the common CLI patterns once: config installation, runtime
      overrides, repeated data-loading behavior, and output handling.

   .. card:: Run family
      :link: ../cli/run_family
      :link-type: doc

      Go here when you want to execute workflow stages, transfer runs,
      sensitivity sweeps, or SM3 run-side diagnostics.

   .. card:: Build family
      :link: ../cli/build_family
      :link-type: doc

      Go here when you want to materialize datasets, payloads,
      summaries, metrics, or geospatial side products.

   .. card:: Plot family
      :link: ../cli/plot_family
      :link-type: doc

      Go here when you want to render figures, maps, uncertainty plots,
      transfer plots, or supporting diagnostics.

How the CLI maps onto the workflow
----------------------------------

The CLI follows the same staged logic as the rest of GeoPrior:

- **Stage 1** prepares the data and writes the artifact set used
  downstream.
- **Stage 2** trains the model from those prepared artifacts.
- **Stage 3** tunes hyperparameters from the same Stage-1 foundation.
- **Stage 4** runs inference, evaluation, calibration, and export.
- **Stage 5** handles transfer evaluation and related workflows.

That means the CLI is not separate from the workflow. It is the most
direct way to *operate* the workflow once you understand what each stage
is for.

A practical reading path
------------------------

A good way to move through the docs is:

#. read :doc:`workflow_overview` to understand the staged process,
#. open :doc:`../cli/index` when you are ready to choose a command,
#. move into the specific CLI family page that matches your task,
#. return to the stage pages when you want the scientific or workflow
   context behind that command,
#. use :doc:`../examples/index` when you want a full worked lesson.

Read next
---------

The best next pages after this one are:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc
      :class-card: sd-shadow-sm card--configuration
      :data-preview: configuration.png
      :data-preview-label: GeoPrior configuration system

      Learn how the config layer interacts with CLI
      installation, overrides, and runtime persistence.

   .. grid-item-card:: Workflow overview
      :link: workflow_overview
      :link-type: doc
      :class-card: sd-shadow-sm card--workflow
      :data-preview: workflow.png
      :data-preview-label: GeoPrior staged workflow

      See how the CLI maps onto the five-stage workflow.

   .. grid-item-card:: Stage-1
      :link: stage1
      :link-type: doc
      :class-card: sd-shadow-sm

      Move from the CLI overview into the first real workflow
      stage.

   .. grid-item-card:: Inference and export
      :link: inference_and_export
      :link-type: doc
      :class-card: sd-shadow-sm

      See how saved artifacts and CLI-driven outputs are
      reused downstream.

.. seealso::

   - :doc:`../cli/index`
   - :doc:`../cli/run_family`
   - :doc:`../cli/build_family`
   - :doc:`../cli/plot_family`
   - :doc:`workflow_overview`
   - :doc:`configuration`
   - :doc:`stage1`
   - :doc:`stage2`
   - :doc:`stage3`
   - :doc:`stage4`
   - :doc:`stage5`
   - :doc:`inference_and_export`