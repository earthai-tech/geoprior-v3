.. _release_notes:

=============
Release Notes
=============

This section records **user-visible changes** across official
GeoPrior-v3 releases.

GeoPrior-v3 release notes are not only packaging history.
They also document changes that affect the project’s public
workflow and scientific contract, including:

- the CLI and staged execution flow,
- the flagship subsidence model family,
- tuning and calibration workflows,
- configuration and packaged defaults,
- exported artifacts, payloads, and summaries,
- documentation structure and migration notes.

This matters because GeoPrior-v3 is not just a single model
class. The project exposes a staged workflow, a physics-guided
subsidence model family, a dedicated tuner surface, top-level
utility helpers, and a packaged configuration template that
drives the runtime contract.

Why release notes are top-level
-------------------------------

Release notes live at the top documentation level because
they are useful to more than contributors alone.

They help:

- users upgrading workflows,
- researchers comparing runs across versions,
- maintainers tracking public behavior changes,
- and developers deciding whether a change affects backward
  compatibility.

This is especially important in GeoPrior-v3 because changes
to configuration defaults, stage outputs, model semantics,
payload schemas, or calibration behavior can affect both
scientific interpretation and downstream reproducibility
workflows.

What belongs in these notes
---------------------------

A release-note entry usually belongs here when a change
affects one or more of the following:

- **CLI behavior**
  such as commands, aliases, stage wrappers, or help surface;

- **public Python API**
  such as model, tuner, utility, or resource-facing imports;

- **scientific behavior**
  such as scaling, residuals, identifiability, uncertainty,
  or bounds logic in the flagship model family;

- **workflow contracts**
  such as config fields, manifests, saved summaries, payloads,
  exported CSV or JSON structure, or stage-to-stage handshake
  assumptions;

- **documentation navigation or migration**
  especially where older docs or import examples were updated
  from a previous hosting context.

For example, the public package surface already exposes
GeoPrior model classes, calibration helpers, payload and
identifiability utilities, top-level workflow helpers, and a
specialized tuner entry point, so changes to any of those
surfaces deserve clear release-note treatment.

How to read the notes
---------------------

Each version page is meant to answer three questions quickly:

- **What changed?**
- **Why does it matter?**
- **Do I need to update anything?**

To make that easier, version pages should prefer concrete,
user-facing sections such as:

- ``Added``
- ``Changed``
- ``Fixed``
- ``Deprecated``
- ``Breaking changes``

A good GeoPrior-v3 release note should make it obvious when a
change affects:

- model interpretation,
- config structure,
- stage outputs,
- tuned-model reuse,
- transfer workflows,
- or documentation migration.

Compatibility mindset
---------------------

When you read a version page, it helps to interpret changes
in one of three ways:

**No action required**
   additive or internal improvements that do not change how
   users run or read the workflow;

**Recommended update**
   an older pattern may still work, but a new preferred path
   is now documented;

**Migration required**
   users should update imports, config fields, scripts,
   artifact readers, or workflow assumptions.

In GeoPrior-v3, this distinction matters because the project
combines configuration bootstrap, staged execution, saved
artifacts, tuned-model selection, and transfer evaluation
rather than only a single training script.

Badge legend
------------

The documentation theme already defines reusable release-note
badges in ``conf.py``. These are intended to keep version
pages visually consistent across the project:

- |New|
- |Feature|
- |Enhancement|
- |Fix|
- |Docs|
- |Build|
- |Tests|
- |API Change|
- |Breaking|

Current release line
--------------------

The current docs configuration resolves the displayed release
from package metadata or ``pyproject.toml``, so the versioned
pages here should follow the actual package release line used
by the docs build.

Version index
-------------

.. toctree::
   :maxdepth: 1
   :caption: Versions

   release_notes/v3.2.0

Guidance for contributors
-------------------------

When a change is user-visible, contributors should update the
corresponding version page as part of the same work. This is
especially important for changes touching:

- the NATCOM config template and runtime config contract;
- the public model and tuner surfaces;
- top-level utility exports;
- documentation migration and navigation structure.

That keeps project history aligned with the actual public
behavior of the package.

See also
--------

.. seealso::

   - :doc:`developer/package_structure`
   - :doc:`developer/migration_from_fusionlab`
   - :doc:`developer/contributing`
   - :doc:`user_guide/workflow_overview`
   - :doc:`api/cli`
   - :doc:`api/subsidence`
   - :doc:`api/tuner`
   - :doc:`api/utils`
   - :doc:`api/resources`