from datetime import datetime

import neuralop

year = datetime.now().year
copyright = f"{year}, Jean Kossaifi, David Pitt, Nikola Kovachki, Zongyi Li and Anima Anandkumar"
author = "Jean Kossaifi, David Pitt, Nikola Kovachki, Zongyi Li and Anima Anandkumar"
project = "neuraloperator"

release = neuralop.__version__
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "numpydoc.numpydoc",
    "sphinx_gallery.gen_gallery",
]
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
}
templates_path = ["_templates"]
exclude_patterns = []
numpydoc_class_members_toctree = False
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_flags = ["members"]
napoleon_google_docstring = False
napoleon_use_rtype = False
imgmath_image_format = "svg"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "tensorly_sphinx_theme"
html_logo = "_static/logos/neuraloperator_logo.png"
html_show_sphinx = False
html_theme_options = {
    "github_url": "https://github.com/neuraloperator/neuraloperator",
    "nav_links": [
        ("Install", "install"),
        ("User Guide", "user_guide/index"),
        ("API", "modules/api"),
        ("Examples", "auto_examples/index"),
    ],
}
html_static_path = ["_static"]
html_permalinks_icon = ""
