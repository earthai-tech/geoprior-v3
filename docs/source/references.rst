.. _references:

==========
References
==========

This page collects the bibliography used throughout the
GeoPrior-v3 documentation.

The references listed here support the scientific
foundations, workflow design, uncertainty analysis, and
application context discussed across the documentation. They
are maintained centrally in the project BibTeX database:

.. code-block:: text

   docs/source/references.bib

How references are used
-----------------------

Citations are inserted throughout the documentation using
Sphinx BibTeX roles such as:

.. code-block:: rst

   :cite:p:`some_key`
   :cite:t:`some_key`

This page renders the full bibliography so readers can see
all cited works in one place.

Scope
-----

The bibliography may include references related to:

- subsidence and groundwater physics,
- poroelasticity and consolidation,
- physics-informed neural networks,
- time-series forecasting,
- uncertainty quantification and calibration,
- geospatial and environmental applications,
- benchmarking and reproducibility.

Notes for contributors
----------------------

When adding a new scientific claim, method comparison, or
historical background point that depends on prior work, add
the corresponding BibTeX entry to:

.. code-block:: text

   docs/source/references.bib

and cite it in the relevant page instead of pasting raw
bibliographic text directly into the prose.

A good practice is:

- keep citation keys stable,
- use complete bibliographic metadata,
- and prefer high-quality primary sources when possible.

Full bibliography
-----------------

.. bibliography::
   :style: unsrt