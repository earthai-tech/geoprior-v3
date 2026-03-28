Scientific foundations
======================

This section documents the scientific and mathematical scaffolding that
supports GeoPrior-v3, with particular emphasis on the subsidence model
family centered on GeoPriorSubsNet. It is written for readers who want
to understand not only *what* the package does, but also *why* the model
is structured the way it is, how the physical assumptions are expressed,
and where the main loss, scaling, and identifiability choices come from.

The pages collected here move from broad model positioning to more
specific technical components. Together, they explain:

- the role of the model family within the broader project,
- the poroelastic and consolidation background used as scientific
  scaffolding,
- how residuals, losses, and scaling are assembled,
- how constraints and identifiability shape the training problem,
- how units, quantities, and transformations are interpreted in
  practice.

This section complements the :doc:`../applications/index` pages, which
show how the scientific ideas are used in real workflows, and the
:doc:`../api/subsidence` reference, which documents the concrete module
interfaces that implement much of this logic.

.. tip::

   Read this section when you need scientific interpretability rather
   than only usage instructions. It is especially helpful for research
   writing, method comparison, debugging of physics-informed behavior,
   and contributor work that touches the model formulation.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Model overview
      :class-card: sd-shadow-sm card--workflow

      Begin with the overall positioning of the GeoPrior model family,
      including how the flagship subsidence workflow fits into the
      broader project roadmap.

      :doc:`models_overview`

   .. grid-item-card:: GeoPriorSubsNet
      :class-card: sd-shadow-sm card--physics

      Study the main subsidence model page and its role in linking data,
      learned fields, physics residuals, and uncertainty-aware outputs.

      :doc:`geoprior_subsnet`

   .. grid-item-card:: Physics formulation
      :class-card: sd-shadow-sm card--configuration

      Examine how the governing scientific assumptions are translated
      into the physics-guided structure used by the model.

      :doc:`physics_formulation`

   .. grid-item-card:: Poroelastic background
      :class-card: sd-shadow-sm

      Review the background concepts that motivate the consolidation and
      groundwater components of the formulation.

      :doc:`poroelastic_background`

   .. grid-item-card:: Residual assembly
      :class-card: sd-shadow-sm

      Understand how the main residual terms are constructed and how
      their different roles interact during model training.

      :doc:`residual_assembly`

   .. grid-item-card:: Losses and training
      :class-card: sd-shadow-sm

      Follow the relationship between data-fit terms, physics penalties,
      priors, and optimization strategy.

      :doc:`losses_and_training`

   .. grid-item-card:: Data and units
      :class-card: sd-shadow-sm

      Clarify the interpretation of variables, sign conventions, unit
      handling, and data-side assumptions used throughout the workflow.

      :doc:`data_and_units`

   .. grid-item-card:: Scaling
      :class-card: sd-shadow-sm

      See how nondimensionalization, normalization, and scale-aware
      choices stabilize training and interpretability.

      :doc:`scaling`

   .. grid-item-card:: Maths
      :class-card: sd-shadow-sm

      Explore the mathematical utilities and conceptual formulas that
      support the model implementation and its diagnostics.

      :doc:`maths`

   .. grid-item-card:: Identifiability
      :class-card: sd-shadow-sm

      Review how parameter coupling, prior structure, and closure logic
      affect what can realistically be inferred from the data.

      :doc:`identifiability`

   .. grid-item-card:: Stability and constraints
      :class-card: sd-shadow-sm

      Understand the safeguards that help keep the learned fields,
      residual terms, and optimization process physically credible.

      :doc:`stability_and_constraints`

Reading strategy
----------------

Readers new to the scientific material should usually begin with
:doc:`models_overview` and :doc:`geoprior_subsnet`, then continue to
:doc:`physics_formulation` and :doc:`poroelastic_background`. That order
provides the conceptual backbone before moving into residual details,
loss design, and training controls.

For method debugging or manuscript preparation, a more targeted route is
often better: read :doc:`residual_assembly`, :doc:`losses_and_training`,
:doc:`scaling`, :doc:`identifiability`, and
:doc:`stability_and_constraints` together. Those pages capture many of
the modeling decisions that matter most for interpretation, comparison,
and technical justification.

.. toctree::
   :maxdepth: 1
   :hidden:

   models_overview
   geoprior_subsnet
   physics_formulation
   poroelastic_background
   residual_assembly
   losses_and_training
   data_and_units
   scaling
   maths
   identifiability
   stability_and_constraints
