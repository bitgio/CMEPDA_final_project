# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
<<<<<<< HEAD
# DEVE PUNTARE ALLA CARTELLA DOVE C'È IL CODICE PYTHON
sys.path.insert(0, os.path.abspath('../..'))

project = 'CMEPDA final project'
=======

sys.path.insert(0, os.path.abspath('../..'))

project = 'CMEPDA_final_project'
>>>>>>> 6ce09b68833834fe9c333f9dd8e596aef1be3a22
copyright = '2023, Giovanni Bitonti, Ana Pascual'
author = 'Giovanni Bitonti, Ana Pascual'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']
<<<<<<< HEAD
# source_suffix = [".rst", ".ipynb", ".md"]
=======

>>>>>>> 6ce09b68833834fe9c333f9dd8e596aef1be3a22
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
<<<<<<< HEAD
html_static_path = ['_static']
numpydoc_show_class_members = False
=======
html_static_path = []
>>>>>>> 6ce09b68833834fe9c333f9dd8e596aef1be3a22
