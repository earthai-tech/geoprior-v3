CLI API reference
=================

The GeoPrior-v3 command-line interface is organized as a
**family-based dispatcher** with dedicated stage wrappers and
shared dispatch utilities.

This page documents the CLI package at two levels:

- the **public entry points** used by users and workflow
  automation;
- the **internal dispatcher and command registry** that make
  the CLI extensible and maintainable.

It also includes selected **source listings** so the inline
implementation comments remain visible in the documentation.

Overview
--------

The CLI package currently provides:

- a **root dispatcher** through ``geoprior``;
- family-specific entry points:
  ``geoprior-run``, ``geoprior-build``, ``geoprior-plot``;
- a dedicated bootstrap entry point:
  ``geoprior-init``;
- stage wrappers for:
  Stage-1 through Stage-5;
- shared dispatch utilities for command loading, alias
  handling, help rendering, and delegated execution.

A useful conceptual map is:

.. code-block:: text

   geoprior.cli
      ├── __init__.py
      ├── __main__.py
      ├── _dispatch.py
      ├── init_config.py
      ├── stage1.py
      ├── stage2.py
      ├── stage3.py
      ├── stage4.py
      └── stage5.py

The root dispatcher uses command families such as
``run``, ``build``, and ``plot`` to route subcommands to the
correct module entry point.

Module index
------------

.. autosummary::
   :toctree: generated/

   geoprior.cli
   geoprior.cli.__main__
   geoprior.cli._dispatch
   geoprior.cli.init_config
   geoprior.cli.stage1
   geoprior.cli.stage2
   geoprior.cli.stage3
   geoprior.cli.stage4
   geoprior.cli.stage5

Public CLI package
------------------

.. automodule:: geoprior.cli
   :members:
   :undoc-members:
   :show-inheritance:

Root dispatcher
---------------

The root dispatcher provides:

- the main ``geoprior`` entry point;
- family-specific wrappers:
  ``run_main()``, ``build_main()``, ``plot_main()``;
- the grouped help surface;
- command-family enforcement;
- alias resolution.

.. automodule:: geoprior.cli.__main__
   :members:
   :undoc-members:
   :show-inheritance:

Important public callables
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: geoprior.cli.__main__.main

.. autofunction:: geoprior.cli.__main__.run_main

.. autofunction:: geoprior.cli.__main__.build_main

.. autofunction:: geoprior.cli.__main__.plot_main

Shared dispatch utilities
-------------------------

The dispatcher helper module centralizes the reusable command
machinery.

Its responsibilities include:

- command specification through :class:`CommandSpec`;
- module and callable loading;
- delegated module execution;
- entrypoint calling conventions;
- alias-map creation;
- compact help-table rendering.

.. automodule:: geoprior.cli._dispatch
   :members:
   :undoc-members:
   :show-inheritance:

Key dispatcher objects
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: geoprior.cli._dispatch.CommandSpec
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: geoprior.cli._dispatch.load_module

.. autofunction:: geoprior.cli._dispatch.load_callable

.. autofunction:: geoprior.cli._dispatch.run_module

.. autofunction:: geoprior.cli._dispatch.call_entry

.. autofunction:: geoprior.cli._dispatch.public_items

.. autofunction:: geoprior.cli._dispatch.alias_map

.. autofunction:: geoprior.cli._dispatch.print_help_table

Stage and bootstrap wrappers
----------------------------

The stage modules provide thin CLI-facing wrappers around the
actual workflow stages. They are important because they keep
the public command surface stable even when the underlying
scientific pipeline evolves.

Configuration bootstrap
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: geoprior.cli.init_config
   :members:
   :undoc-members:
   :show-inheritance:

Stage-1 wrapper
~~~~~~~~~~~~~~~

.. automodule:: geoprior.cli.stage1
   :members:
   :undoc-members:
   :show-inheritance:

Stage-2 wrapper
~~~~~~~~~~~~~~~

.. automodule:: geoprior.cli.stage2
   :members:
   :undoc-members:
   :show-inheritance:

Stage-3 wrapper
~~~~~~~~~~~~~~~

.. automodule:: geoprior.cli.stage3
   :members:
   :undoc-members:
   :show-inheritance:

Stage-4 wrapper
~~~~~~~~~~~~~~~

.. automodule:: geoprior.cli.stage4
   :members:
   :undoc-members:
   :show-inheritance:

Stage-5 wrapper
~~~~~~~~~~~~~~~

.. automodule:: geoprior.cli.stage5
   :members:
   :undoc-members:
   :show-inheritance:

Related command registry
------------------------

The modern CLI also bridges to the paper- and figure-facing
``scripts`` registry so that plot and build commands can be
surfaced through the same family-based dispatcher.

If you want that relationship documented explicitly in the
API docs, include the registry modules here as well.

.. autosummary::
   :toctree: generated/

   scripts.__main__
   scripts.registry

.. automodule:: scripts.registry
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scripts.__main__
   :members:
   :undoc-members:
   :show-inheritance:

Command families and entry points
---------------------------------

The public CLI surface is organized around these entry points:

+------------------+---------------------------------------------+
| Entry point      | Purpose                                     |
+==================+=============================================+
| ``geoprior``     | Root dispatcher with explicit family token  |
+------------------+---------------------------------------------+
| ``geoprior-run`` | Workflow and stage execution                |
+------------------+---------------------------------------------+
| ``geoprior-build`` | Deterministic artifact builders          |
+------------------+---------------------------------------------+
| ``geoprior-plot`` | Figure and map rendering                  |
+------------------+---------------------------------------------+
| ``geoprior-init`` | Config bootstrap                          |
+------------------+---------------------------------------------+

The root dispatcher expects a family token first, for example:

.. code-block:: text

   geoprior run stage1-preprocess
   geoprior build full-inputs-npz
   geoprior plot physics-fields

The family-specific entry points remove the explicit family
token:

.. code-block:: text

   geoprior-run stage1-preprocess
   geoprior-build full-inputs-npz
   geoprior-plot physics-fields

Dispatcher conventions
----------------------

The dispatcher follows a few important design rules:

- command families are enforced explicitly;
- aliases resolve to one canonical public command name;
- unknown commands print grouped help and exit non-zero;
- family-specific entry points reject repeated family tokens;
- command entrypoints can be called in one of three styles:
  ``fn(argv, prog=...)``, ``fn(argv)``, or ``fn()`` with a
  patched ``sys.argv``.

That last point is especially important for backward
compatibility with older command modules.

Why the private dispatcher is documented
----------------------------------------

Normally, a module named ``_dispatch`` might be omitted from
public API docs. Here, it is worth documenting because it is
the core architectural piece that explains:

- how command specs are represented;
- how the same command family can support both modern CLI
  wrappers and legacy script entry points;
- how help tables and aliases are generated consistently;
- how delegated module execution preserves command identity.

So even though it is “private” by naming convention, it is
important enough to deserve full API documentation.

Source listings with comments
-----------------------------

The following source listings are included intentionally so
that inline comments remain visible in the documentation.

CLI package ``__init__.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/cli/__init__.py
   :language: python
   :caption: geoprior/cli/__init__.py

Root dispatcher ``__main__.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/cli/__main__.py
   :language: python
   :caption: geoprior/cli/__main__.py

Shared dispatcher ``_dispatch.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../geoprior/cli/_dispatch.py
   :language: python
   :caption: geoprior/cli/_dispatch.py

Optional: scripts registry bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../scripts/registry.py
   :language: python
   :caption: scripts/registry.py

Optional: legacy paper-scripts entry point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../scripts/__main__.py
   :language: python
   :caption: scripts/__main__.py

API notes
---------

A few implementation details are worth keeping in mind when
reading the CLI API:

- ``CommandSpec`` is the shared normalized representation used
  by the dispatcher layer.
- The root CLI registry mixes native GeoPrior commands with
  bridged commands from the ``scripts`` package.
- The dispatcher distinguishes between ``argv`` call style,
  ``sysargv``-style compatibility, and module-style
  execution.
- The help system is grouped by command family so the root
  entry point remains readable even as the command surface
  grows.

See also
--------

.. seealso::

   - :doc:`../user_guide/cli`
   - :doc:`../user_guide/configuration`
   - :doc:`subsidence`
   - :doc:`tuner`
   - :doc:`utils`
   - :doc:`resources`