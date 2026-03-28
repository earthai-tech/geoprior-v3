# -*- coding: utf-8 -*-
"""Sphinx configuration for GeoPrior docs."""

from __future__ import annotations

import datetime
import importlib
import re
import sys
import warnings
from importlib.metadata import (
    PackageNotFoundError,
    version as pkg_version,
)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _read_release() -> str:
    """
    Resolve the package version with a safe fallback chain.

    Resolution order
    ----------------
    1. Import ``geoprior`` and read ``__version__``.
    2. Read installed package metadata for ``geoprior-v3``.
    3. Read version from ``pyproject.toml``.
    4. Fallback to ``0+unknown``.
    """
    try:
        gp = importlib.import_module("geoprior")
        ver = getattr(gp, "__version__", None)
        if isinstance(ver, str) and ver.strip():
            return ver.strip()
    except Exception:
        pass

    try:
        return pkg_version("geoprior-v3")
    except PackageNotFoundError:
        pass

    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8")
        match = re.search(
            r'^version\s*=\s*"([^"]+)"',
            text,
            flags=re.MULTILINE,
        )
        if match:
            return match.group(1)

    warnings.warn(
        (
            "Could not resolve GeoPrior version from import, "
            "installed metadata, or pyproject.toml; using "
            "0+unknown."
        ),
        stacklevel=2,
    )
    return "0+unknown"

# # -- Version Handling -------------------------------------------------------
# try:
#     import geoprior as gp

#     version = ".".join(
#         gp.__version__.split(".")[:2]
#     )  # Use the major.minor version
#     release = gp.__version__
# except Exception:
#     # Fallback: package not importable (e.g., building docs from source)
#     version = "0.0"
#     release = "0+unknown"
#     warnings.warn(
#         "GeoPrior not importable in docs environment; using 0+unknown"
#     )


release = _read_release()
version = ".".join(release.split(".")[:2])

project = "GeoPrior"
author = "Laurent Kouadio"
current_year = datetime.datetime.now().year
copyright = f"{current_year}"

html_title = f"{project} v{release}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"
language = "en"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autoclass_content = "class"
autosummary_generate = True
autosummary_imported_members = False
autodoc_mock_imports = ["tensorflow", "keras", "seaborn", "contextily"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
math_number_all = True

bibtex_bibfiles = ["references.bib"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": (
        "https://pandas.pydata.org/pandas-docs/stable/",
        None,
    ),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
intersphinx_cache_limit = 5
intersphinx_timeout = 10

html_theme = "pydata_sphinx_theme"
html_title = f"{project} v{release}"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/custom.js"]
html_logo = "_static/gp.logo.png"
html_favicon = "_static/gp.logo.ico"


html_theme_options = {
    "logo": {
        "image_light": "_static/geoprior-svg.svg",
        "image_dark": "_static/geoprior-svg.svg",
        "text": "",
        "alt_text": "GeoPrior-v3 documentation - Home",
        "link": "index",
    },

    "switcher": {
        "json_url": (
            "https://geoprior-v3.readthedocs.io/en/latest/"
            "_static/switcher.json"
        ),
        "version_match": release,
    },

    # keep disabled until the JSON is actually deployed
    "check_switcher": True,

    "navbar_start": [
        "navbar-logo",
        "version-switcher",
    ],
    "navbar_center": ["navbar-nav"],
    "navbar_end": [
        "theme-switcher",
        "navbar-icon-links",
    ],
    "navbar_persistent": ["search-button"],

    "icon_links_label": "External links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/earthai-tech/geoprior-v3",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "Website",
            "url": "https://geoprior-v3.readthedocs.io",
            "icon": "fa-solid fa-globe",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/geoprior-v3/",
            "icon": "_static/icons/pypi.svg",
            "type": "local",
        },
        {
            "name": "Stack Overflow",
            "url": "https://stackoverflow.com/questions/tagged/geoprior",
            "icon": "_static/icons/stackoverflow.svg",
            "type": "local",

        },
    ],

    "footer_start": [
        "author-copyright",
    ],
    "footer_center": [],
    "footer_end": [
        "theme-version",
    ],

    "use_edit_page_button": True,
    "header_links_before_dropdown": 5,
    "search_bar_text": "Search the GeoPrior docs ...",
    "navigation_with_keys": True,
    "show_prev_next": True,
    "collapse_navigation": False,
    "navigation_depth": 3,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "secondary_sidebar_items": [
        "page-toc",
        "edit-this-page",
        "sourcelink",
    ],
    "back_to_top_button": True,
}

html_context = {
    "github_user": "earthai-tech",
    "github_repo": "geoprior-v3",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "auto",
    "author_name": "Laurent Kouadio",
    "author_portfolio_url": "https://lkouadio.com",
    "current_year": current_year,
}

rst_epilog = """
.. |Feature| replace:: :bdg-success:`Feature`
.. |New| replace:: :bdg-success:`New`
.. |Fix| replace:: :bdg-info:`Fix`
.. |Enhancement| replace:: :bdg-info:`Enhancement`
.. |Breaking| replace:: :bdg-danger:`Breaking`
.. |API Change| replace:: :bdg-warning:`API Change`
.. |Docs| replace:: :bdg-secondary:`Docs`
.. |Build| replace:: :bdg-primary:`Build`
.. |Tests| replace:: :bdg-primary:`Tests`
"""

html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

copybutton_prompt_text = (
    r">>> |\.\.\. |\$ |In \[\d*\]: | "
    r" {2,5}\.\.\.: | {5,8}: "
)
copybutton_prompt_is_regexp = True

from sphinx_gallery.sorting import FileNameSortKey

extensions += [
    "sphinx_gallery.gen_gallery",
]

sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"plot_",
    "ignore_pattern": r"__init__\.py",
    "nested_sections": True,
    "within_subsection_order": FileNameSortKey,
    "download_all_examples": True,
    "remove_config_comments": True,
    "backreferences_dir": "generated/backreferences",
    "doc_module": ("geoprior",),
    "reference_url": {
        "geoprior": None,
    },
    "image_scrapers": ("matplotlib",),
    "matplotlib_animations": True,
    "capture_repr": ("_repr_html_", "__repr__"),
    "show_memory": False,
}
