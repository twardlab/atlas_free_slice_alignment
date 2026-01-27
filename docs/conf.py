# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'atlas_free_slice_alignment'
copyright = '2026, Daniel Tward, Andrew Bennecke, Gary Zhou'
author = 'Daniel Tward, Andrew Bennecke, Gary Zhou'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinxcontrib.htmlhelp',
    'sphinx.ext.viewcode'
]

# Jupyter Notebook configurations
nbsphinx_execute = 'never'
nbsphinx_execute_arguments = ["--InlineBackend.figure_formats={'svg', 'pdf'}",]

# Configuration for LaTeX output
latex_documents = [
    ('index', 'spine_registration.tex', project, author, 'manual'),
]

# Skip the following packages when building documentation
autodoc_mock_imports = ['torch','numpy','argparse','matplotlib','scipy']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

import sys
from pathlib import Path

sys.path.insert(0, str(Path('..', 'source').resolve()))
sys.path.insert(0, str(Path('..', 'source', 'scripts').resolve()))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bizstyle'

html_static_path = ['docs/_static']