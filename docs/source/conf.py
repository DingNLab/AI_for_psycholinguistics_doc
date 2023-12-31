# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AI for psycholinguistics'
copyright = '2023, Honghua Chen'
author = 'Honghua Chen'
release = 'v0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'recommonmark',
    'sphinx.ext.mathjax',
    'furo.sphinxext',
    "sphinx_inline_tabs",
    # 'sphinx_markdown_tables'
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_title = ""
# html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
html_theme = "furo"
# html_theme = 'sphinx_book_theme'

html_static_path = ['_static']
