# -*- coding: utf-8 -*-
#
# LazyLLM documentation build configuration file

import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('../../'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode']

source_suffix = '.rst'
master_doc = 'index'

# General information about the project.
project = u'LazyLLM'
copyright = u'2024, SenseTime Research'
author = u'SenseTime Research'
version = u'0.1.0'
release = u'0.1.0 rc0'

language = 'cn'
exclude_patterns = []
pygments_style = 'sphinx'
todo_include_todos = True
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}
htmlhelp_basename = 'LazyLLMdoc'
latex_elements = {
}

latex_documents = [
    (master_doc, 'LazyLLM.tex', u'LazyLLM Documentation',
     u'SenseTime Research', 'manual'),
]


man_pages = [
    (master_doc, 'lazyllm', u'LazyLLM Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'LazyLLM', u'LazyLLM Documentation',
     author, 'LazyLLM', 'One line description of project.',
     'Miscellaneous'),
]
