Getting started
===============

This section is the best entry point for readers who are new to
GeoPrior-v3 and want to move from orientation to a first working run.
It gathers the introductory pages that explain what the project is,
how to install it, how to navigate the documentation, and how to launch
a minimal configuration-driven workflow without first studying the full
scientific or API layers.

GeoPrior is designed as more than a standalone model class. The project
combines a Python package interface, a staged command-line workflow,
packaged resources, and a documentation structure that mirrors the way
real subsidence studies are executed. Because of that, the getting
started material is organized around the practical questions most users
have at the beginning:

- what the package is for and where it fits scientifically,
- how to install it in a reproducible environment,
- how to obtain a first successful run quickly,
- where to go next for workflow, scientific, or API details.

If you are already familiar with the project goals and want the larger
workflow picture, continue afterward to the
:doc:`../user_guide/workflow_overview` guide. If you want the model-side
background behind the flagship subsidence workflow, see
:doc:`../scientific_foundations/models_overview`.

.. note::

   The goal of this section is to reduce startup friction. It focuses on
   the minimum conceptual and practical material required to begin using
   GeoPrior productively, while leaving the deeper physics, tuning, and
   API details to later sections of the documentation.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Project overview
      :class-card: sd-shadow-sm card--workflow

      Start with the broad project narrative, including the scientific
      problem, the role of GeoPriorSubsNet, and the documentation map.

      :doc:`overview`

   .. grid-item-card:: Installation
      :class-card: sd-shadow-sm card--configuration

      Set up a working Python environment, review the baseline
      requirements, and prepare the package for command-line or Python
      use.

      :doc:`installation`

   .. grid-item-card:: Quickstart
      :class-card: sd-shadow-sm card--cli

      Follow the fastest route from installation to a minimal working
      interaction with the package and its staged workflow.

      :doc:`quickstart`

   .. grid-item-card:: First project run
      :class-card: sd-shadow-sm card--physics

      Walk through a first structured execution path and understand what
      the main stages produce, save, and expose.

      :doc:`first_project_run`

Recommended reading path
------------------------

A natural reading order is:

1. :doc:`overview`
2. :doc:`installation`
3. :doc:`quickstart`
4. :doc:`first_project_run`

That sequence moves from project context to environment setup, then to a
minimal successful interaction, and finally to a slightly fuller
workflow picture. After that, most users should continue with
:doc:`../user_guide/workflow_overview` and the staged guides in the user
section.

For readers coming primarily from research rather than software, the
:doc:`overview` page is especially important because it clarifies how
GeoPrior relates the subsidence forecasting problem to the package
structure. For readers coming primarily from engineering or code use,
:doc:`quickstart` and :doc:`first_project_run` provide the most direct
path to hands-on use.

.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   installation
   quickstart
   first_project_run
