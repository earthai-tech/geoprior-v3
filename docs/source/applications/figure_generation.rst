.. _applications_figure_generation:

=================
Figure generation
=================

GeoPrior-v3 is designed to support not only model training
and export, but also **paper-ready figure generation** from
saved workflow artifacts.

This page explains how figure generation fits into the
GeoPrior application stack, how the current ``scripts/``
package is organized, what kinds of figures it is meant to
produce, and how users should think about the relationship
between:

- workflow stages,
- deterministic artifact builders,
- and figure-facing scripts.

The central idea is simple:

- the staged workflow produces durable scientific artifacts;
- the scripts layer turns those artifacts into figures,
  panels, and tables suitable for papers, supplements, and
  reports.

Why figure generation is its own application layer
--------------------------------------------------

In a serious scientific project, figure generation should not
be buried inside the main training loop.

A good workflow separates:

- **model computation**, which may be expensive;
- **artifact generation**, which should be deterministic;
- **figure generation**, which should be lightweight and
  repeatable.

GeoPrior-v3 follows exactly this pattern. The staged
workflow creates the scientific results, while the dedicated
scripts layer consumes those results to produce visual
outputs. That is a much stronger design than mixing plotting
logic directly into training commands.

A useful summary is:

.. code-block:: text

   stages create artifacts
        ↓
   build commands materialize derived artifacts
        ↓
   scripts generate figures from those saved outputs

This is the core figure-generation philosophy of GeoPrior.

The ``scripts`` package
-----------------------

The current repository defines a dedicated ``scripts``
package whose own module docstring describes it as a
**paper scripts package** runnable with:

.. code-block:: bash

   python -m scripts <command> [args]

This is the legacy-but-supported direct entry point for the
figure layer. 

At the same time, GeoPrior-v3 now exposes a modern unified
CLI, so most figure commands are also available through the
root or family-specific entry points, for example:

.. code-block:: bash

   geoprior plot physics-fields -h
   geoprior-plot uncertainty -h

The CLI package explicitly documents the root/family split
and supports:

- ``geoprior run ...``
- ``geoprior build ...``
- ``geoprior plot ...``

along with dedicated family entry points such as
``geoprior-run``, ``geoprior-build``, and
``geoprior-plot``. 

This means the figure system is both:

- backward-compatible with legacy script use,
- and integrated into the main GeoPrior command-line design.

Output layout for figures
-------------------------

The scripts configuration already defines a clean filesystem
layout for paper-facing outputs:

- ``scripts/out`` for general output files,
- ``scripts/figs`` for figures.

These locations are centralized in the script config layer,
which also stores publication defaults such as figure DPI,
font size, unit labels, canonical city names, color choices,
and common artifact discovery patterns. 

This is an important design choice because it gives figure
generation a stable home instead of scattering outputs across
temporary directories.

A practical consequence is:

- if a figure command is given a relative output path,
  it can be resolved consistently under ``scripts/figs``.

What figure scripts are expected to consume
-------------------------------------------

GeoPrior figure scripts are designed to be
**artifact-driven**.

They are expected to consume saved workflow outputs such as:

- evaluation diagnostics JSON,
- calibrated or future forecast CSV files,
- physics payload NPZ files,
- transfer benchmark CSV / JSON summaries,
- stage manifests and run summaries.

The current script utilities already implement discovery
patterns for these artifact families, including:

- physics evaluation JSON,
- evaluation diagnostics JSON,
- calibrated evaluation forecasts,
- future forecasts,
- physics payload NPZ files,
- coordinate NPZ files,
- ablation-record JSONL files. 

This is exactly the right approach for a scientific software
project: figures should be generated from saved results,
not from hidden retraining.

Auto-detection of artifacts
---------------------------

The utility layer already provides artifact discovery helpers
such as:

- ``find_latest(...)``
- ``find_preferred(...)``
- ``detect_artifacts(...)``

which search a run directory for the most relevant saved
artifacts using known filename patterns. 

This is valuable for figure workflows because it reduces the
amount of manual path plumbing needed for common paper
scripts while still keeping the dependency chain explicit.

A practical use case is:

- point a figure script at one run directory,
- let it discover the best matching payload, JSON, or CSV,
- generate the figure from those saved results.

That is much lighter than asking users to manually specify
every single file for every panel.

The figure command registry
---------------------------

The scripts registry defines a structured catalog of
paper-facing commands, each with:

- a module,
- an entry function,
- a description,
- a family,
- a public CLI name,
- optional aliases.

The registry includes many plot commands, such as:

- ``physics-fields``
- ``uncertainty``
- ``litho-parity``
- ``spatial-forecasts``
- ``physics-sanity``
- ``physics-maps``
- ``transfer``
- ``transfer-impact``
- ``hotspot-analytics``
- ``external-validation``

as well as supplementary or SM3-related plotting commands. 

This matters because figure generation in GeoPrior is not a
single script. It is a structured figure application layer.

A useful command-family view
----------------------------

The modern CLI dispatcher maps those plotting commands into
the public ``plot`` family, while build-oriented commands are
exposed under the ``build`` family. The root dispatcher
documents both the family split and the grouped help layout
for pipeline, supplementary diagnostics, build commands, and
plot commands. 

A useful mental model is:

.. code-block:: text

   run family
      trains, tunes, infers, transfers

   build family
      materializes deterministic derived artifacts

   plot family
      renders figures from saved outputs

That is a clean, scalable scientific workflow design.

Typical figure categories
-------------------------

The current script surface suggests several recurring figure
categories in GeoPrior-v3.

**Forecast figures**

Examples:

- uncertainty panels,
- spatial forecast maps,
- cumulative geo curves.

**Physics figures**

Examples:

- physics fields,
- physics maps,
- physics sanity checks,
- physics profiles.

**Validation / parity figures**

Examples:

- lithology parity,
- external validation figures,
- driver-response plots.

**Ablation and sensitivity figures**

Examples:

- core ablation,
- ablations sensitivity,
- physics sensitivity.

**Transfer figures**

Examples:

- transferability,
- transfer impact,
- cross-city comparison visuals.

This is useful because it shows that figure generation is not
an afterthought. It is part of the package’s research-facing
application design.

Publication defaults and styling
--------------------------------

The figure utilities centralize publication-oriented style
choices.

The current configuration and utils layer already define:

- publication DPI,
- publication font size,
- city colors,
- unit labels,
- physics labels,
- closure strings,
- and a reusable paper plotting style function. 

This is scientifically useful because it helps keep figures:

- visually consistent,
- unit-aware,
- and easier to compare across panels and scripts.

A good figure workflow should not depend on every script
reinventing labels, units, and styling from scratch.

Saving figures in multiple formats
----------------------------------

The utility layer also provides a central figure-saving
helper:

- ``save_figure(...)``

which supports writing multiple output formats from one
figure, typically with publication-friendly defaults and
automatic resolution of relative output stems under the
figure directory. 

This is especially important for real paper workflows, where
different downstream needs often require different formats:

- PNG for quick sharing,
- SVG for editable vector workflows,
- PDF or EPS for manuscript pipelines.

So the figure layer is not only about plotting. It is also
about durable export.

Why figures should be artifact-driven
-------------------------------------

A GeoPrior figure script should ideally consume:

- one run directory,
- one manifest,
- one payload,
- one forecast CSV,
- or one benchmark summary

and then produce a figure deterministically.

This is a better design than a script that:

- silently retrains the model,
- silently changes configuration,
- or silently mixes multiple incompatible runs.

The utility layer strongly supports this artifact-driven
model through discovery helpers, robust JSON/CSV loading,
schema checking, and canonical label/unit handling. 

A practical figure-generation flow
----------------------------------

A useful end-to-end figure workflow looks like:

.. code-block:: text

   run GeoPrior stages
        ↓
   save model / forecast / payload / benchmark artifacts
        ↓
   optionally build derived artifacts
        ↓
   run one plot command against the saved outputs
        ↓
   export publication-ready figure files

This is the intended application meaning of the scripts
layer.

Examples of modern figure commands
----------------------------------

A few representative commands from the current registry are:

.. code-block:: bash

   geoprior plot physics-fields -h
   geoprior plot uncertainty -h
   geoprior plot litho-parity -h
   geoprior plot spatial-forecasts -h
   geoprior plot transfer -h
   geoprior plot transfer-impact -h
   geoprior plot external-validation -h

The corresponding legacy style remains available through the
scripts package, for example:

.. code-block:: bash

   python -m scripts plot-physics-fields -h
   python -m scripts plot-uncertainty -h

This dual surface is helpful during migration and keeps the
figure workflow flexible. 

A good figure script pattern
----------------------------

A strong GeoPrior figure script should usually:

1. accept explicit artifact or run-directory inputs;
2. auto-detect common saved outputs when possible;
3. validate that required fields exist;
4. harmonize labels and units;
5. render a deterministic figure;
6. save outputs in one or more standard formats.

This pattern already matches the structure of the shared
script utilities.

A compact summary is:

.. code-block:: text

   parse args
      ↓
   find artifacts
      ↓
   load JSON / CSV / NPZ
      ↓
   validate schema
      ↓
   render plot with paper style
      ↓
   save figure to scripts/figs

This is the best way to keep figure workflows reproducible.

Figure generation from evaluation forecasts
-------------------------------------------

Many GeoPrior figures naturally start from forecast CSVs.

Common cases include:

- forecast-vs-actual comparison panels,
- calibrated interval plots,
- horizon-wise uncertainty summaries,
- per-sample or per-city forecast behavior.

The shared utilities already include robust forecast-CSV
loaders with schema enforcement for key columns such as
forecast step, quantiles, and actual subsidence values. 

That is exactly the kind of reusable logic that a paper
workflow needs.

Figure generation from physics payloads
---------------------------------------

Physics payloads are another major figure input.

They make it possible to generate:

- effective-parameter maps,
- sanity plots for :math:`K`, :math:`S_s`, :math:`\tau`,
  and :math:`H_d`,
- identifiability diagnostics,
- closure-consistency panels,

without rerunning training.

This is one of the strongest parts of GeoPrior’s figure
generation story, because it separates heavy model fitting
from later scientific interpretation.

Figure generation from transfer results
---------------------------------------

Stage-5 outputs are also natural figure inputs.

Saved benchmark tables such as cross-city result CSV or JSON
files can drive:

- transferability plots,
- transfer-impact panels,
- calibration-mode comparisons,
- baseline-vs-transfer summaries.

This means a cross-city figure can be regenerated from the
saved benchmark outputs rather than by rerunning the entire
transfer stage.

Supplementary and ablation figures
----------------------------------

The current plotting layer also clearly supports
supplementary and sensitivity workflows.

For example, the ablation/sensitivity script surface already
contains options for:

- multi-metric heatmaps,
- contour or tricontour rendering,
- aligned lambda grids,
- Pareto trade-off views,
- city filtering,
- and output formatting. 

This is important because many scientific papers need more
than one flagship figure. They also need:

- supplementary diagnostics,
- sensitivity maps,
- and ablation summaries.

GeoPrior’s figure layer is already structured for that.

When to use build commands before plotting
------------------------------------------

Some figures require derived artifacts that are not part of
the initial training output.

In those cases, the clean workflow is:

- first run a deterministic build command,
- then run the figure command that consumes the built output.

This is especially appropriate for:

- full-input NPZ generation,
- physics payload materialization,
- external validation metrics,
- hotspot or zone summaries.

The CLI split between ``build`` and ``plot`` is exactly what
makes this clean. 

Common mistakes
---------------

**Putting figure logic inside model-training code**

This makes figures harder to rerun and harder to audit.

**Generating figures from mixed runs**

A figure should ideally consume artifacts from one coherent
run family unless the comparison itself is the point.

**Using unlabeled uncertainty products**

Calibrated and uncalibrated outputs should not be mixed
casually in figures.

**Ignoring saved summaries**

JSON summaries often explain what a plotted CSV or payload
actually represents.

**Hard-coding local paths**

Scripts should prefer explicit arguments or artifact
auto-discovery under a run directory.

Best practices
--------------

.. admonition:: Best practice

   Treat figure scripts as artifact consumers, not as hidden
   workflow rerunners.

   This keeps figures lightweight and reproducible.

.. admonition:: Best practice

   Use the modern CLI surface for new workflows.

   The legacy ``python -m scripts ...`` entry point remains
   useful, but the ``geoprior plot ...`` family is the main
   long-term interface.

.. admonition:: Best practice

   Save figures in more than one format when preparing paper
   outputs.

   Raster and vector formats often serve different needs.

.. admonition:: Best practice

   Keep run manifests and summary JSON files next to the
   artifacts that feed the figures.

   This makes later interpretation much easier.

.. admonition:: Best practice

   Standardize units, labels, and city naming through the
   shared script utilities whenever possible.

   This keeps the figure suite visually and scientifically
   consistent.

A compact figure-generation map
-------------------------------

The GeoPrior figure workflow can be summarized as:

.. code-block:: text

   saved stage artifacts
        ↓
   optional build commands
        ↓
   scripts registry selects plot command
        ↓
   utilities discover + validate inputs
        ↓
   paper style applied
        ↓
   figure exported to scripts/figs

Relationship to the rest of the docs
------------------------------------

This page explains how GeoPrior turns saved scientific
artifacts into publication-ready visuals.

The companion pages explain related parts of the workflow:

- :doc:`reproducibility_scripts`
  explains the broader script- and artifact-driven
  reproducibility layer;

- :doc:`calibration_and_uncertainty`
  explains one major family of uncertainty-aware figures;

- :doc:`../user_guide/inference_and_export`
  explains the forecast and summary artifacts that many
  figures consume;

- :doc:`../user_guide/diagnostics`
  explains the evaluation and physics diagnostics that often
  become plotted panels;

- :doc:`../user_guide/stage5`
  explains the transfer results that later feed transfer
  figures.

See also
--------

.. seealso::

   - :doc:`reproducibility_scripts`
   - :doc:`calibration_and_uncertainty`
   - :doc:`../user_guide/diagnostics`
   - :doc:`../user_guide/inference_and_export`
   - :doc:`../user_guide/stage4`
   - :doc:`../user_guide/stage5`
   - :doc:`../scientific_foundations/geoprior_subsnet`