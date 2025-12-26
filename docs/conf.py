import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'ImgRec'
copyright = '2024, Antigravity'
author = 'Antigravity'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'ja'
html_theme = 'alabaster'
html_static_path = ['_static']
