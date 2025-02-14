# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx_rtd_theme
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))

project = "pangeo-fish"
author = "pangeo-fish developers"
copyright = f"2023, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    # "sphinx.ext.doctest",
    # "autoapi.extension",
    "sphinx_copybutton",
    "myst_parser",
]

autosummary_generate = True
autosummary_imported_members = False

# Document Python Code
# autoapi_type = "python"
# autoapi_dirs = ["../pangeo_fish"]
# autoapi_options = [ "members", "undoc-members", "show-inheritance", "show-module-summary", "imported-members"]
# autoapi_ignore = ["*ipynb*", "*__main__.py*", "*/cli/*"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "sphinx_rtd_theme"
pygments_style = "sphinx"

# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = []
