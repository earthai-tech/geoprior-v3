Developer notes
===============

The developer section documents the internal organization and maintenance
logic behind GeoPrior-v3. It is intended for contributors, maintainers,
and advanced users who need a clearer picture of how the package is laid
out, how the project evolved from earlier codebases, and how new changes
should be integrated without weakening the scientific or workflow
contract.

GeoPrior is not just a collection of model classes. It combines a staged
CLI workflow, a model family centered on subsidence forecasting,
documentation-driven packaging, and a set of reproducibility-oriented
scripts. Because of that, contributor-facing documentation needs to
cover both software architecture and domain-specific design choices.

This section therefore emphasizes three complementary questions:

- how the package is structured,
- how legacy concepts from FusionLab map into GeoPrior-v3,
- how to contribute new code and documentation in a consistent way.

For broader user-facing guidance, see the :doc:`../user_guide/faq` and
:doc:`../api/index` pages. For scientific context behind the main model
family, refer to
:doc:`../scientific_foundations/geoprior_subsnet`.

.. warning::

   The developer pages are written to preserve architectural clarity.
   When extending GeoPrior, prefer documented public modules, keep
   imports stable, and avoid introducing hidden import-time side effects
   that can weaken the CLI, documentation build, or autosummary-based
   API generation.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Package structure
      :class-card: sd-shadow-sm

      Review the organization of the source tree, the role of the major
      subpackages, and the logic behind the current documentation and
      workflow layout.

      :doc:`package_structure`

   .. grid-item-card:: Migration from FusionLab
      :class-card: sd-shadow-sm

      Trace how concepts, workflow patterns, and module organization were
      adapted from the earlier FusionLab environment into GeoPrior-v3.

      :doc:`migration_from_fusionlab`

   .. grid-item-card:: Contributing
      :class-card: sd-shadow-sm

      Read the contributor guidance for proposing changes, keeping the
      package style coherent, and maintaining documentation quality.

      :doc:`contributing`

Contributor focus
-----------------

GeoPrior-v3 benefits from contributions that improve one or more of the
following areas:

- scientific clarity of the model and physics documentation,
- stability of the CLI and staged workflow,
- reproducibility of scripts, figures, and packaged resources,
- robustness of the documented API surface,
- maintainability of the project structure as the package evolves.

If you are preparing a significant change, it is usually helpful to read
:doc:`package_structure` first, then the migration notes, and finally
:doc:`contributing`. That sequence provides the best context for making
changes that fit the existing architecture instead of working against it.

.. toctree::
   :maxdepth: 1
   :hidden:

   package_structure
   migration_from_fusionlab
   contributing
