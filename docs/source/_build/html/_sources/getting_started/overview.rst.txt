Overview
================

GeoPrior-v3 is a scientific Python framework for
**physics-guided geohazard modeling, forecasting, and risk
analytics**. It is designed for applications where purely
data-driven prediction is not enough and where scientific
structure, physical consistency, and reproducible workflows
must remain central to the modeling process.

The current generation of the project focuses on
**land subsidence** through **GeoPriorSubsNet v3.x**, while
the broader roadmap extends toward **landslides and other
geohazard modeling tasks**. GeoPrior-v3 combines model
development, scientific constraints, staged command-line
execution, configuration-driven runs, and reproducible
research scripts in a single framework. 

.. note::

   GeoPrior-v3 is both a Python package and a workflow-driven
   application framework. In addition to importable modules,
   it provides dedicated console entry points for
   initialization, staged runs, builds, and plotting:
   ``geoprior``, ``geoprior-run``, ``geoprior-build``,
   ``geoprior-plot``, and ``geoprior-init``. 

What GeoPrior-v3 is built for
-----------------------------

GeoPrior-v3 is intended for scientific settings in which
geophysical or geohazard processes should be modeled with
more structure than a standard black-box machine-learning
pipeline can provide. The framework therefore emphasizes:

- **physics-guided learning**, where physical reasoning is
  part of the modeling strategy rather than a post-hoc
  interpretation step;
- **configuration-driven execution**, so experiments and
  application runs can be organized in a controlled,
  reproducible manner;
- **staged workflows**, allowing the pipeline to be executed
  step by step through a consistent CLI;
- **scientific reproducibility**, including figure-oriented
  scripts and paper-facing workflows;
- **extensibility**, so the framework can grow beyond its
  current flagship use case.

In practice, GeoPrior-v3 is meant to support users who need
more than model training alone. It supports the full path
from project setup and configuration to staged execution,
diagnostics, inference, export, and reproducibility.

Current focus and roadmap
-------------------------

Today, GeoPrior-v3 is centered on **land subsidence**
modeling through the GeoPriorSubsNet v3.x family. This is
the current flagship application and the main scientific
context around which the present documentation is organized.

At the same time, the package identity already makes clear
that GeoPrior-v3 is not meant to remain limited to one
problem setting. The project roadmap points toward
**landslides and broader geohazard modeling**, which means
the framework is structured to support growth into a wider
set of physics-guided hazard applications over time. 

Design philosophy
-----------------

GeoPrior-v3 is organized around a few core ideas.

**1. Scientific structure matters**

Many geohazard problems involve latent physical processes,
partial observability, sparse field information, and strong
domain constraints. GeoPrior-v3 is therefore designed around
models that preserve scientific meaning rather than treating
prediction as a purely statistical mapping.

**2. Workflows should be reproducible**

A scientific result is only useful if it can be reproduced,
inspected, and rerun. GeoPrior-v3 favors explicit
configuration, staged execution, and documented artifacts so
that model runs are easier to understand and repeat.

**3. The CLI is a first-class interface**

The framework is not documented as an API library alone.
Its command-line interface is part of the intended user
experience, especially for stage-based execution,
configuration bootstrapping, and figure or result
generation. The project metadata explicitly registers
multiple console entry points to support that workflow. 

**4. Documentation should mirror real usage**

The documentation is organized around how GeoPrior-v3 is
actually used in practice: understanding the scientific
approach, preparing configuration, running staged workflows,
checking diagnostics, exporting outputs, and reproducing
research artifacts.

How users typically work with GeoPrior-v3
-----------------------------------------

A typical user workflow looks like the following:

1. install the package and review the project scope;
2. initialize or inspect a configuration;
3. run one or more stages of the workflow through the CLI;
4. inspect diagnostics, model behavior, and exported outputs;
5. generate figures or reproducibility artifacts for analysis
   or publication;
6. dive deeper into the scientific foundations and API when
   customization is needed.

This is why the documentation is divided into six main
sections:

- **Getting started**
- **User guide**
- **Scientific foundations**
- **Applications**
- **API reference**
- **Developer notes**

These sections are meant to support different kinds of
readers, from first-time users to contributors and
researchers who want to understand the deeper physical or
architectural logic of the framework.

Package and workflow surface
----------------------------

GeoPrior-v3 is distributed as a Python package named
``geoprior-v3`` and currently targets Python 3.10 and above.
The project metadata describes it as
``Physics-guided AI for geohazards`` and defines the package
keywords around geohazards, subsidence, physics-informed
modeling, forecasting, and risk. 

The package exposes command-line entry points for several
core workflow actions:

- ``geoprior`` for the main CLI surface;
- ``geoprior-run`` for execution-oriented commands;
- ``geoprior-build`` for build-oriented tasks;
- ``geoprior-plot`` for plotting-oriented tasks;
- ``geoprior-init`` for configuration initialization. 

At the documentation level, the Sphinx configuration is now
organized around a dedicated GeoPrior-v3 site using the
PyData Sphinx Theme, with support for BibTeX references,
design components, autosummary generation, and custom static
branding assets. :contentReference[oaicite:8]{index=8}

What to read next
-----------------

The best next page depends on how you want to approach the
project.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc
      :class-card: sd-shadow-sm

      Set up the package and the documentation environment,
      and review the core dependency expectations.

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc
      :class-card: sd-shadow-sm

      See the fastest way to begin using GeoPrior-v3 from
      Python and from the command line.

   .. grid-item-card:: Workflow overview
      :link: ../user_guide/workflow_overview
      :link-type: doc
      :class-card: sd-shadow-sm

      Understand the staged execution model and how the user
      guide is organized around it.

   .. grid-item-card:: Models overview
      :link: ../scientific_foundations/models_overview
      :link-type: doc
      :class-card: sd-shadow-sm

      Move from the project-level overview to the scientific
      and modeling foundations of the framework.

.. seealso::

   - :doc:`installation`
   - :doc:`quickstart`
   - :doc:`first_project_run`
   - :doc:`../user_guide/workflow_overview`
   - :doc:`../scientific_foundations/models_overview`