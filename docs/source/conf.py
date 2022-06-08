# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import time
from source.k3d_directives.plot import K3D_Plot
import shutil
import json

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'K3D-jupyter'
author = u'Artur Trzęsiok, Marcin Kostur, Tomasz Gandor, Thomas Mattone'
copyright = time.strftime(
    '%Y') + ' ' + author

# The full version, including alpha/beta/rc tags
here = os.path.dirname(__file__)
repo = os.path.join(here, '..', '..')
_version_py = os.path.join(repo, 'package.json')
version_ns = {}

with open(_version_py) as f:
    version = json.load(f)["version"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
    'sphinx.ext.napoleon'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The root document.
root_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "show_prev_next": False,
    "google_analytics_id": 'UA-141840477-1',
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/K3D-tools/K3D-jupyter",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPi",
            "url": "https://pypi.org/project/k3d/",
            "icon": "fas fa-box-open",
            "type": "fontawesome",
        },
        {
            "name": "Conda",
            "url": "https://anaconda.org/conda-forge/k3d",
            "icon": "fas fa-circle-notch",
            "type": "fontawesome",
        }
    ]
}

html_sidebars = {
    "index": ["search-field", "sidebar_index"],
    "gallery/*": []
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app):
    shutil.copy('./../js/dist/standalone.js', './source/_static/standalone.js')
    shutil.copy('./../node_modules/requirejs/require.js', './source/_static/require.js')

    try:
        app.add_css_file('style.css')
        app.add_javascript('require.js')
        app.add_javascript('standalone.js?k3d')
    except AttributeError:
        app.add_css_file('style.css')
        app.add_js_file('require.js')
        app.add_js_file('standalone.js?k3d')

    app.add_directive('k3d_plot', K3D_Plot)
