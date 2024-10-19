from datetime import datetime

import wsi

project = "wsi"
copyright = f"{datetime.now().year}, Rendeiro Lab"
author = "wsi contributors"
release = wsi.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]
autoclass_content = "class"
autodoc_docstring_signature = True
autodoc_default_options = {"members": None, "undoc-members": None}
autodoc_typehints = "none"
# setting autosummary
autosummary_generate = True
numpydoc_show_class_members = False
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = []


html_theme = "pydata_sphinx_theme"
html_sidebars = {"installation": [], "api": []}
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5," r"8}: "
copybutton_prompt_is_regexp = True

# Plot directive
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 200)]

intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/stable/", None),
}
