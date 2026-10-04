"""Sphinx configuration for the Bioafferent ReadTheDocs site."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Bioafferent"
author = "Bioafferent Contributors"
version = "0.1.0"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True

# Optional integrations are mocked so API docs build without them.
autodoc_mock_imports = ["gymnasium", "mujoco"]
