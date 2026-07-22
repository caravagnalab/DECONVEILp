# Configuration file for the Sphinx documentation builder.

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


# Project information

project = "BDGDM"
author = "Katsiaryna Davydzenka"
copyright = "2026, Katsiaryna Davydzenka"

try:
    release = version("bdgdm")
except PackageNotFoundError:
    # Useful when the documentation is inspected before the package
    # has been installed in the current environment.
    release = "development"

version = release


# General configuration

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# AutoAPI

# Use this when the repository contains:
#
# BDGDM/
# ├── bdgdm/
# └── docs/
#
autoapi_type = "python"
autoapi_dirs = ["../bdgdm"]

autoapi_root = "api"

autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]

# Avoid documenting private implementation objects by default.
autoapi_member_order = "bysource"
autoapi_keep_files = True

# Optional exclusions.
autoapi_ignore = [
    "*/__pycache__/*",
    "*/tests/*",
]


# NumPy-style docstrings

napoleon_numpy_docstring = True
napoleon_google_docstring = False

napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

# MyST and notebook configuration

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "deflist",
]

# Use outputs already stored in the notebooks.
#
# This is appropriate for BDGDM because CmdStan compilation and NUTS
# sampling may take too long for a Read the Docs build.
nb_execution_mode = "off"

# Remove notebook input/output prompts such as In [1] and Out [1].
nb_remove_code_source = False
nb_remove_code_outputs = False

# Raise an error when a notebook contains a stored error output.
nb_execution_raise_on_error = True

# Files ignored by Sphinx

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
]


# HTML output

html_theme = "sphinx_rtd_theme"
html_title = "BDGDM documentation"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# Uncomment after creating docs/_static/.
# html_static_path = ["_static"]


