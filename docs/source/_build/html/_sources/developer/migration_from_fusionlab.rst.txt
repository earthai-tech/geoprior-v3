Migration from FusionLab
========================

GeoPrior-v3 documentation was previously hosted inside a
broader **FusionLab-oriented documentation context**.

That older arrangement was useful during early development,
but it is no longer the right long-term structure for the
project. GeoPrior-v3 now has its **own documentation space**
and its own package identity, with a documentation target
aligned to the GeoPrior repository and workflow rather than
to a parent package or mixed project context.

This page explains what changed, why the migration matters,
and how developers should update older references, imports,
and documentation links.

Why the migration happened
--------------------------

GeoPrior-v3 has grown into a distinct scientific software
project with its own:

- model family,
- CLI surface,
- staged workflow,
- application layer,
- and documentation structure.

At that point, keeping the docs embedded in a FusionLab-style
context created several problems:

- project identity became blurred;
- API links could point to the wrong namespace;
- docs could imply dependencies or ownership that were no
  longer accurate;
- and contributors had to mentally translate between an old
  hosting context and the current repository structure.

A useful summary is:

.. code-block:: text

   early phase:
      GeoPrior lived inside a broader documentation context

   current phase:
      GeoPrior-v3 is documented as its own package and project

That transition is the core meaning of this migration page.

What changed
------------

The migration is not only about moving pages to a new docs
site. It also changes the expected documentation reference
model.

Before
~~~~~~

The old documentation context tended to assume a broader
FusionLab-style identity, where GeoPrior-related material
could be treated as part of a larger package ecosystem.

Typical consequences included:

- stale inter-page links,
- wrong module paths in API references,
- references to old package locations,
- mixed narrative language about which project owned the
  workflow.

Now
~~~

GeoPrior-v3 documentation is structured around the current
repository and package layout:

- :mod:`geoprior.cli`
- :mod:`geoprior.models`
- :mod:`geoprior.utils`
- :mod:`geoprior.resources`

with the flagship model family now clearly documented through
the GeoPrior model surface, including
:class:`~geoprior.models.GeoPriorSubsNet` and
:class:`~geoprior.models.PoroElasticSubsNet`, which are
publicly exported from the model package.

This means that documentation should now describe GeoPrior as
its own project, not as a feature nested inside a different
package identity.

What did **not** change
-----------------------

The migration does **not** mean the scientific ideas were
discarded.

The following remain continuous across the transition:

- the subsidence-forecasting model family,
- the staged workflow philosophy,
- the physics-guided forecasting logic,
- the tuner-based search workflow,
- the reproducibility and figure-generation layer.

What changed is mainly:

- ownership of the documentation surface,
- correctness of module paths and links,
- and clarity of the project boundary.

Why this matters for developers
-------------------------------

A documentation migration can leave behind subtle problems
even after the new site builds successfully.

For example:

- a page may still mention an old package path;
- an ``:mod:`` or ``:class:`` link may resolve to the wrong
  object;
- prose may still refer to FusionLab instead of GeoPrior-v3;
- examples may import from a deprecated or indirect location.

These problems are easy to miss because the text may still
*sound* correct even while the technical reference path is
wrong.

So the goal of this page is not only historical explanation.
It is also a **maintenance guide** for keeping the migrated
docs consistent.

Main migration principles
-------------------------

When updating old documentation, follow these rules.

**1. Prefer the GeoPrior namespace**

Pages should refer to ``geoprior`` modules and objects unless
there is a very specific reason to mention historical context.

**2. Treat FusionLab references as legacy context**

FusionLab can be mentioned to explain old links or migration
history, but it should not be presented as the current home
of GeoPrior-v3 docs.

**3. Update both prose and cross-references**

It is not enough to replace words in paragraphs. Sphinx
cross-references, import examples, API paths, and
``literalinclude`` paths also need to be updated.

**4. Keep current public exports authoritative**

Documentation should prefer the public import surfaces that
the package intentionally exposes today.

A compact migration view is:

.. code-block:: text

   old FusionLab-oriented wording
        ↓
   replace with GeoPrior-v3 project wording

   old module path
        ↓
   replace with current geoprior.* path

   old inter-doc link
        ↓
   replace with local page or current API target

What to update in old pages
---------------------------

When migrating an older page, check the following layers.

Project naming
~~~~~~~~~~~~~~

Look for wording like:

- “FusionLab package”
- “FusionLab docs”
- “the parent package”
- or language that treats GeoPrior as only a subsection of
  another project

and replace it with wording appropriate for GeoPrior-v3 as an
independent project.

Imports and API paths
~~~~~~~~~~~~~~~~~~~~~

Check all examples such as:

.. code-block:: python

   from fusionlab...

and replace them with the correct GeoPrior import surface.

For example, current public exports already include
GeoPrior-v3 model-facing objects directly from the model
package surface, such as:

.. code-block:: python

   from geoprior.models import GeoPriorSubsNet
   from geoprior.models import PoroElasticSubsNet

which reflects the current top-level model package exports.

Cross-references
~~~~~~~~~~~~~~~~

Check Sphinx roles such as:

- ``:mod:`...``
- ``:class:`...``
- ``:func:`...``
- ``:doc:`...``

A page can build while still pointing to a stale or wrong
target if the old reference pattern remains in place.

Literal include paths
~~~~~~~~~~~~~~~~~~~~~

Any ``literalinclude`` blocks copied from older docs should be
checked carefully.

A path that was valid inside the old documentation layout may
now be wrong relative to:

.. code-block:: text

   docs/source/...

This is one of the most common migration breakpoints.

Narrative assumptions
~~~~~~~~~~~~~~~~~~~~~

Check whether the page assumes:

- a different package layout,
- a different documentation site root,
- a different CLI entry strategy,
- or a different public API boundary.

These assumptions often survive even after links are fixed.

Typical old-to-new mapping patterns
-----------------------------------

The migration is easiest when contributors follow a small set
of standard replacement patterns.

Project identity
~~~~~~~~~~~~~~~~

Old pattern:

.. code-block:: text

   FusionLab documentation / package context

New pattern:

.. code-block:: text

   GeoPrior-v3 documentation / project context

Model imports
~~~~~~~~~~~~~

Old style:

.. code-block:: python

   from fusionlab....

New preferred style:

.. code-block:: python

   from geoprior.models import GeoPriorSubsNet
   from geoprior.models import PoroElasticSubsNet

Current model exports also include uncertainty and diagnostic
helpers through the same GeoPrior model surface, such as
interval calibration helpers and identifiability/payload
utilities.

Utility imports
~~~~~~~~~~~~~~~

Old style:

.. code-block:: python

   from fusionlab.utils....

New style:

.. code-block:: python

   from geoprior.utils import load_nat_config
   from geoprior.utils import make_tf_dataset
   from geoprior.utils import evaluate_forecast

These names are currently re-exported from the top-level
GeoPrior utility surface.

Tuner imports
~~~~~~~~~~~~~

Old style:

.. code-block:: python

   from fusionlab.... import some tuner

New style:

.. code-block:: python

   from geoprior.models.forecast_tuner import SubsNetTuner

which matches the current public tuner export surface
.

Config/template references
~~~~~~~~~~~~~~~~~~~~~~~~~~

Old style:

.. code-block:: text

   configuration template in previous doc context

New style:

.. code-block:: text

   geoprior/resources/natcom_config_template.py

The current template explicitly identifies itself as the
GeoPrior-v3 NATCOM config template and documents its intended
placement and bootstrap role.

How to rewrite migrated prose
-----------------------------

A good migration does more than replace import paths. It also
updates the voice of the documentation.

Old migrated prose often sounds like:

- “inside FusionLab we provide...”
- “this module belongs to the broader parent framework...”
- “the package extension used here is...”

A better GeoPrior-v3 style is:

- “GeoPrior-v3 provides...”
- “the GeoPrior workflow uses...”
- “the package exposes...”
- “the staged workflow writes...”

This makes the docs feel coherent rather than patched.

When to mention FusionLab explicitly
------------------------------------

FusionLab should still be mentioned in a few cases.

It is useful when:

- explaining why old links or imports may still appear in
  notebooks or draft docs;
- documenting a migration decision for contributors;
- clarifying that some historical text came from an earlier
  host documentation context.

It is **not** useful to mention FusionLab repeatedly inside
normal user-guide or API pages once the page has already been
fully migrated.

A good rule is:

- mention FusionLab for **history or migration**;
- use GeoPrior-v3 everywhere else.

Migration checklist for contributors
------------------------------------

When updating an older page, use this checklist.

Project identity
~~~~~~~~~~~~~~~~

- replace FusionLab-oriented project wording with GeoPrior-v3
  wording where appropriate
- keep historical references only when they help explain the
  migration

Imports and examples
~~~~~~~~~~~~~~~~~~~~

- update old imports to current ``geoprior`` imports
- prefer public re-export surfaces when they exist
- avoid documenting stale internal paths if a stable public
  path is available

Cross-references
~~~~~~~~~~~~~~~~

- check ``:mod:``, ``:class:``, ``:func:``, and ``:doc:``
  roles
- rebuild the docs and confirm references resolve correctly

Paths and includes
~~~~~~~~~~~~~~~~~~

- check all ``literalinclude`` paths
- check local page links relative to ``docs/source/``

Narrative consistency
~~~~~~~~~~~~~~~~~~~~~

- remove prose that assumes the old package boundary
- confirm the page now reads like a GeoPrior-v3 document,
  not like a migrated appendix

API surfaces to prefer after migration
--------------------------------------

As the docs migrate, contributors should prefer the current
public surfaces that GeoPrior intentionally exports.

Models
~~~~~~

Use:

.. code-block:: python

   from geoprior.models import GeoPriorSubsNet
   from geoprior.models import PoroElasticSubsNet

Utilities
~~~~~~~~~

Use top-level GeoPrior utility exports when they are
available, for example:

.. code-block:: python

   from geoprior.utils import load_nat_config
   from geoprior.utils import evaluate_forecast
   from geoprior.utils import make_txy_coords

The top-level utility package already re-exports many of the
workflow-facing helpers needed in documentation examples.

Tuner
~~~~~

Use:

.. code-block:: python

   from geoprior.models.forecast_tuner import SubsNetTuner

which matches the current forecast-tuner package export
surface .

Resources
~~~~~~~~~

Treat packaged resources as GeoPrior resources, especially:

.. code-block:: text

   geoprior/resources/natcom_config_template.py

which is the packaged NATCOM template for config bootstrap.

Common migration mistakes
-------------------------

**Replacing wording but not links**

The prose may look updated while Sphinx roles still point to
stale targets.

**Keeping historical imports in examples**

Users then copy code that no longer reflects the intended
public API.

**Mixing old and new project identities on one page**

This makes the documentation feel unstable and confuses new
contributors.

**Treating migrated pages as “good enough” once they build**

A successful build does not guarantee the page is correct,
coherent, or aligned with the new package boundary.

**Documenting old private paths instead of new public exports**

This makes future refactors harder.

A practical migration workflow
------------------------------

A good page-migration workflow is:

.. code-block:: text

   identify page with old FusionLab context
        ↓
   update prose to GeoPrior-v3 wording
        ↓
   replace stale imports with current geoprior imports
        ↓
   fix Sphinx refs and literalinclude paths
        ↓
   rebuild docs
        ↓
   verify rendered links and API targets
        ↓
   do one final prose pass for consistency

That order works better than only doing search-and-replace.

How to think about backward compatibility
-----------------------------------------

Migration does not require pretending the old context never
existed.

Instead, contributors should aim for:

- **historical honesty**
- plus
- **current technical correctness**

That means it is fine to say:

- “this page was migrated from an earlier FusionLab-hosted
  docs context”

but the page should still teach the user the **current**
GeoPrior-v3 names, APIs, and navigation paths.

Best practices
--------------

.. admonition:: Best practice

   Prefer current public GeoPrior import surfaces in examples.

   This makes the docs more stable and easier for users to
   follow.

.. admonition:: Best practice

   Mention FusionLab only where historical explanation is
   helpful.

   Do not let old hosting context dominate current docs.

.. admonition:: Best practice

   Check prose, refs, and includes together.

   A page is not fully migrated until all three are aligned.

.. admonition:: Best practice

   Keep migration edits explicit in pull requests.

   Reviewers should be able to see whether a change is a
   namespace fix, a wording fix, or a real behavior change.

.. admonition:: Best practice

   Use the current packaged template and utility surfaces as
   anchors when rewriting old setup pages.

   They are part of the clearest current GeoPrior-v3 contract
   .

A compact migration map
-----------------------

The migration from FusionLab can be summarized as:

.. code-block:: text

   old docs host/context
        ↓
   GeoPrior-v3 gets its own documentation space
        ↓
   old wording, refs, and imports become stale
        ↓
   migrate to current geoprior namespace and package boundary
        ↓
   rebuild links, examples, and API pages around GeoPrior-v3

Relationship to the rest of the developer docs
----------------------------------------------

This page explains **how to update old documentation context
into the new project boundary**.

The surrounding developer pages explain:

- :doc:`package_structure`
  where the current code and docs now live;

- :doc:`contributing`
  how to contribute cleanly to the current project;

- :doc:`../release_notes`
  how to record changes once the migration becomes part of a
  released docs history.

See also
--------

.. seealso::

   - :doc:`package_structure`
   - :doc:`contributing`
   - :doc:`../release_notes`
   - :doc:`../user_guide/configuration`
   - :doc:`../api/cli`
   - :doc:`../api/subsidence`
   - :doc:`../api/tuner`
   - :doc:`../api/utils`
   - :doc:`../api/resources`