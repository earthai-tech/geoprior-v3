Scientific foundations
======================

This section presents the scientific and mathematical scaffolding behind
GeoPrior-v3, with particular emphasis on the subsidence model family
centered on GeoPriorSubsNet. It is intended for readers who want to
understand not only *what* the package does, but also *why* the model is
structured the way it is, how the physical assumptions are expressed,
and where the main loss, scaling, stability, and identifiability choices
come from.

Rather than serving as a purely theoretical appendix, this part of the
documentation connects the underlying scientific ideas to the practical
forecasting workflow implemented throughout the package. The pages
collected here move from broad model positioning to increasingly
specific technical components. Together, they explain:

- how the model family is positioned within the broader GeoPrior
  project,
- which poroelastic and consolidation ideas are used as scientific
  scaffolding,
- how residuals, losses, and scaling choices are assembled,
- how constraints and identifiability shape the training problem,
- how quantities, units, and transformations should be interpreted in
  practice.

This section complements the :doc:`../applications/index` pages, which
show how these ideas are used in complete workflows, and the
:doc:`../api/subsidence` reference, which documents the concrete module
interfaces that implement much of this logic.

.. note::

   Use this section when you need scientific interpretability rather
   than only usage instructions. It is especially useful for research
   writing, method comparison, debugging of physics-informed behavior,
   and contributor work that touches the model formulation directly.

.. container:: cta-tiles

   .. grid:: 1 2 2 3
      :gutter: 3

      .. grid-item-card:: Model overview
         :class-card: sd-shadow-sm card--workflow
         :link: models_overview
         :link-type: doc

         Begin with the overall positioning of the GeoPrior model family,
         including how the flagship subsidence workflow fits into the
         broader project roadmap.

      .. grid-item-card:: GeoPriorSubsNet
         :class-card: sd-shadow-sm card--physics
         :link: geoprior_subsnet
         :link-type: doc

         Study the main subsidence model page and its role in linking data,
         learned fields, physics residuals, and uncertainty-aware outputs.

      .. grid-item-card:: Physics formulation
         :class-card: sd-shadow-sm card--configuration
         :link: physics_formulation
         :link-type: doc

         Examine how the governing scientific assumptions are translated
         into the physics-guided structure used by the model.

      .. grid-item-card:: Poroelastic background
         :class-card: sd-shadow-sm
         :link: poroelastic_background
         :link-type: doc

         Review the background concepts that motivate the consolidation and
         groundwater components of the formulation.

      .. grid-item-card:: Residual assembly
         :class-card: sd-shadow-sm
         :link: residual_assembly
         :link-type: doc

         Understand how the main residual terms are constructed and how
         their different roles interact during model training.

      .. grid-item-card:: Losses and training
         :class-card: sd-shadow-sm
         :link: losses_and_training
         :link-type: doc

         Follow the relationship between data-fit terms, physics penalties,
         priors, and optimization strategy.

      .. grid-item-card:: Data and units
         :class-card: sd-shadow-sm
         :link: data_and_units
         :link-type: doc

         Clarify the interpretation of variables, sign conventions, unit
         handling, and data-side assumptions used throughout the workflow.

      .. grid-item-card:: Scaling
         :class-card: sd-shadow-sm
         :link: scaling
         :link-type: doc

         See how nondimensionalization, normalization, and scale-aware
         choices stabilize training and interpretability.

      .. grid-item-card:: Maths
         :class-card: sd-shadow-sm
         :link: maths
         :link-type: doc

         Explore the mathematical utilities and conceptual formulas that
         support the model implementation and its diagnostics.

      .. grid-item-card:: Identifiability
         :class-card: sd-shadow-sm
         :link: identifiability
         :link-type: doc

         Review how parameter coupling, prior structure, and closure logic
         affect what can realistically be inferred from the data.

      .. grid-item-card:: Stability and constraints
         :class-card: sd-shadow-sm
         :link: stability_and_constraints
         :link-type: doc

         Understand the safeguards that help keep the learned fields,
         residual terms, and optimization process physically credible.

How to read this section
------------------------

There are two good ways to use these pages.

The first is as a guided conceptual path. Readers who are new to the
scientific material should usually begin with
:doc:`models_overview` and :doc:`geoprior_subsnet`, then continue to
:doc:`physics_formulation` and :doc:`poroelastic_background`. That order
provides the main modeling picture before moving into residual details,
loss construction, and training controls.

The second is as a targeted technical reference. For debugging,
comparison, or manuscript preparation, it is often more useful to read
:doc:`residual_assembly`, :doc:`losses_and_training`,
:doc:`scaling`, :doc:`identifiability`, and
:doc:`stability_and_constraints` together. Those pages capture many of
the design choices that matter most for interpretation, justification,
and reproducible scientific discussion.

.. tip::

   If you are reading GeoPrior primarily as a scientific contribution,
   start with the model and physics pages first. If you are already
   familiar with the general formulation and want to understand training
   behavior in detail, move directly to the residual, loss, scaling,
   and identifiability pages.

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