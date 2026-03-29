Developer notes
===============

This section documents the contributor-facing side of GeoPrior-v3. It
is intended for maintainers, contributors, and advanced users who need
more than end-user instructions: a clear view of how the package is
organized, how its current structure emerged, and how new changes can be
introduced without weakening the scientific, workflow, or documentation
contract.

GeoPrior is not just a collection of model classes. The project brings
together a staged command-line workflow, a subsidence-focused model
family, documentation-aware packaging, and reproducibility-oriented
scripts. Because of that, the developer documentation needs to explain
both software architecture and domain-specific design choices.

This section is therefore organized around three practical questions:

- how the package is structured,
- how legacy ideas from FusionLab map into GeoPrior-v3,
- how to contribute code and documentation in a way that stays
  consistent with the existing project design.

For broader user-facing guidance, see :doc:`../user_guide/faq` and the
:doc:`../api/index` reference. For the scientific context behind the
main model family, refer to
:doc:`../scientific_foundations/geoprior_subsnet`.

.. note::

   The purpose of this section is to preserve architectural clarity.
   When extending GeoPrior, prefer documented public modules, keep
   imports stable, and avoid introducing hidden import-time side effects
   that can weaken the CLI, the documentation build, or autosummary-
   based API generation.

.. container:: cta-tiles

   .. grid:: 1 2 2 3
      :gutter: 3

      .. grid-item-card:: Package structure
         :class-card: sd-shadow-sm
         :link: package_structure
         :link-type: doc

         Review the organization of the source tree, the role of the major
         subpackages, and the logic behind the current documentation and
         workflow layout.

      .. grid-item-card:: Migration from FusionLab
         :class-card: sd-shadow-sm
         :link: migration_from_fusionlab
         :link-type: doc

         Trace how concepts, workflow patterns, and module organization were
         adapted from the earlier FusionLab environment into GeoPrior-v3.

      .. grid-item-card:: Contributing
         :class-card: sd-shadow-sm
         :link: contributing
         :link-type: doc

         Read the contributor guidance for proposing changes, keeping the
         package style coherent, and maintaining documentation quality.

How to read this section
------------------------

This part of the documentation works best when read as a contributor
ramp rather than as a flat reference.

A natural reading path is:

#. begin with :doc:`package_structure`,
#. continue to :doc:`migration_from_fusionlab`,
#. finish with :doc:`contributing`.

That sequence gives you the current project layout first, then the
historical and architectural context behind it, and finally the working
rules for making changes that fit the existing codebase.

Readers will often come here with different goals. If your aim is to
understand where code should live and how the package is partitioned,
start with the structure page. If you are trying to understand naming,
workflow, or organizational choices inherited from earlier work, the
migration page provides the most useful background. If you already know
where you need to work and want the maintenance expectations directly,
move quickly to the contributing page.

.. tip::

   Before making a substantial change, it is usually worth reading the
   structure page first. That one step often prevents changes that work
   locally but cut across the intended CLI, documentation, or package
   boundaries.

.. toctree::
   :maxdepth: 1
   :hidden:

   package_structure
   migration_from_fusionlab
   contributing
