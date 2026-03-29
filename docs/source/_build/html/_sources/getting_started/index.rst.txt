Getting started
===============

This section is the main entry point for readers who are new to
GeoPrior-v3 and want to move from initial orientation to a first
successful run. It brings together the core introductory pages that
explain what the project is, how to install it, how to navigate the
documentation, and how to launch a minimal configuration-driven
workflow before diving into the deeper scientific or API layers.

GeoPrior is designed as more than a single model implementation. The
project combines a Python package, a staged command-line workflow,
packaged resources, and documentation that mirrors the structure of
real subsidence-forecasting studies. For that reason, the getting
started material is organized around the questions most users have at
the beginning:

- what the package is for and how it is positioned scientifically,
- how to install it in a stable and reproducible environment,
- how to obtain a first working run quickly,
- where to go next for workflow, scientific, or API details.

If you already understand the overall project goals and want the
larger execution picture, continue afterward to
:doc:`../user_guide/workflow_overview`. If you want the scientific
background behind the flagship subsidence workflow, see
:doc:`../scientific_foundations/models_overview`.

.. note::

   The purpose of this section is to reduce startup friction. It focuses
   on the minimum practical and conceptual material needed to begin
   using GeoPrior productively, while leaving deeper physics, tuning,
   and API details to later parts of the documentation.

.. container:: cta-tiles

   .. grid:: 1 1 2 2
      :gutter: 3

      .. grid-item-card:: Overview
         :link: overview
         :link-type: doc

         Understand what GeoPrior-v3 is built for, how the workflow is
         organized, and which stages matter most before you run the
         package.

      .. grid-item-card:: Installation
         :link: installation
         :link-type: doc

         Install the package, optional dependencies, and documentation
         tooling required for local use and reproducible builds.

      .. grid-item-card:: Quickstart
         :link: quickstart
         :link-type: doc

         Run the shortest practical path through the project so you can
         see the data flow, outputs, and expected artifacts.

      .. grid-item-card:: First project run
         :link: first_project_run
         :link-type: doc

         Walk through a more complete first execution, including the
         expected inputs, stage behavior, and exported results.

How to read this section
------------------------

These pages are meant to be read as a guided ramp into the project.

A natural path is:

#. start with :doc:`overview`,
#. continue to :doc:`installation`,
#. run through :doc:`quickstart`,
#. then follow :doc:`first_project_run`.

That sequence moves from project context to environment setup, then to
a minimal successful interaction, and finally to a slightly fuller
view of the workflow. After that, most readers should continue into
:doc:`../user_guide/workflow_overview` and the staged guides in the
user guide.

Different readers may enter from different directions. Readers coming
primarily from research often benefit from starting with
:doc:`overview`, because it explains how the forecasting problem maps
onto the package structure. Readers coming primarily from engineering
or code usage may want to move more quickly into :doc:`quickstart` and
:doc:`first_project_run`, where the practical execution path is more
immediate.

.. tip::

   If your goal is simply to get GeoPrior running as quickly as
   possible, focus first on installation and quickstart. If your goal
   is to understand how the package is organized before you run it,
   begin with the overview page and then move forward in order.

.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   installation
   quickstart
   first_project_run