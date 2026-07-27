# Configuration file for the Sphinx documentation builder.

from importlib.metadata import (
    PackageNotFoundError,
    version as package_version,
)
from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DOCS_DIR = Path(__file__).resolve().parent
ROOT_DIR = DOCS_DIR.parent
PACKAGE_DIR = ROOT_DIR / "bdgdm"


# ---------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------

project = "BDGDM"
author = "Katsiaryna Davydzenka"
copyright = "2026, Katsiaryna Davydzenka"

try:
    release = package_version("bdgdm")
except PackageNotFoundError:
    # Useful when the documentation is inspected before the package
    # has been installed in the current environment.
    release = "development"

version = release

# The default is already "index", but making it explicit improves clarity.
root_doc = "index"


# ---------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]


# ---------------------------------------------------------------------
# AutoAPI
# ---------------------------------------------------------------------

autoapi_type = "python"

# Expected repository layout:
#
# DECONVEILp/
# ├── bdgdm/
# └── docs/
#
autoapi_dirs = [
    str(PACKAGE_DIR),
]

# Generated API pages are placed under docs/api/.
autoapi_root = "api"

# Automatically add the generated API index to the documentation tree.
autoapi_add_toctree_entry = True

# Show documented public members without displaying private or
# double-underscore implementation details.
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
]

autoapi_member_order = "bysource"

# Include both class-level and __init__ documentation when available.
autoapi_python_class_content = "both"

# Retain generated API source files during local development.
autoapi_keep_files = True

autoapi_ignore = [
    "*migrations*",
    "*/__pycache__/*",
    "*/tests/*",
    "*/.ipynb_checkpoints/*",
    "**/.ipynb_checkpoints/**",
    "*-checkpoint.py",
    "*_old.py",
    "*_old_ver.py",
]


# ---------------------------------------------------------------------
# NumPy-style docstrings
# ---------------------------------------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False

napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False


# ---------------------------------------------------------------------
# MyST and notebook configuration
# ---------------------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "deflist",
]

# Render outputs already saved in the notebooks.
#
# CmdStan compilation and NUTS sampling are intentionally not performed
# during Sphinx or Read the Docs builds.
nb_execution_mode = "off"

# Retain code cells and their saved outputs.
nb_remove_code_source = False
nb_remove_code_outputs = False

# Make long tables and text outputs vertically scrollable.
nb_scroll_outputs = True

# Keep stderr visible by default. Individual CmdStan fitting cells can
# receive the "remove-stderr" tag to suppress routine sampling logs.
nb_output_stderr = "show"


# ---------------------------------------------------------------------
# Files ignored by Sphinx
# ---------------------------------------------------------------------

exclude_patterns = [
    "_build",
    "api",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
]


# ---------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_title = (
    f"BDGDM {release} documentation"
    if release != "development"
    else "BDGDM documentation"
)

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# Enable these after creating docs/_static/custom.css.
#
# html_static_path = ["_static"]
# html_css_files = ["custom.css"]
