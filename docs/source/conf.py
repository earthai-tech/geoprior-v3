# -*- coding: utf-8 -*-
"""Sphinx configuration for GeoPrior-v3 docs."""

from __future__ import annotations

import datetime
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
    """Resolve the package version without importing geoprior."""
    try:
        return pkg_version("geoprior-v3")
    except PackageNotFoundError:
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
                "Could not resolve GeoPrior-v3 version from "
                "installed metadata or pyproject.toml; "
                "using 0+unknown."
            ),
            stacklevel=2,
        )
        return "0+unknown"


release = _read_release()
version = ".".join(release.split(".")[:2])

project = "GeoPrior-v3"
author = "Laurent Kouadio"
current_year = datetime.datetime.now().year
copyright = f"{current_year}, {author}"

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
autodoc_mock_imports = ["tensorflow", "keras"]

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
html_css_files = ["css/custom.css"]
html_js_files = ["js/custom.js"]
html_logo = "_static/gp.logo.png"
html_favicon = "_static/gp.logo.ico"

html_theme_options = {
    "logo": {
        "image_light": "_static/geoprior-svg.svg",
        "image_dark": "_static/geoprior-svg.svg",
        "text": "GeoPrior-v3",
        "alt_text": "GeoPrior-v3 documentation - Home",
        "link": "index",
    },
    "icon_links_label": "External links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": (
                "https://github.com/earthai-tech/"
                "geoprior-v3"
            ),
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Website",
            "url": "https://geoprior-v3.readthedocs.io",
            "icon": "fas fa-globe",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": [
        "theme-switcher",
        "navbar-icon-links",
    ],
    "navbar_persistent": ["search-button-field"],
    "navbar_align": "content",
    "header_links_before_dropdown": 6,
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
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": ["theme-version"],
    "back_to_top_button": True,
}

html_context = {
    "github_user": "earthai-tech",
    "github_repo": "geoprior-v3",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "auto",
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
