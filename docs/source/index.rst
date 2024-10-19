wsi
===

**wsi** is a Python library for processing whole slide images (WSI).

It is a simple library with an object-oriented interface where all operations are performed with a WholeSlideImage object.

It has the goal of doing basic processing of WSIs with reasonable defaults, but high customizability.

For example, to go from a slide remotely hosted to a torch geometric graph with ResNet50 features, the following lines suffice:

.. code-block:: python

   from wsi import WholeSlideImage
   slide = WholeSlideImage("https://brd.nci.nih.gov/brd/imagedownload/GTEX-O5YU-1426")
   slide.segment()
   slide.tile()
   data = slide.as_torch_geometric_data(model_name='resnet50')

Head over to the `Installation <install.html>`_ and `API reference <api.html>`_ pages to learn more.


.. toctree::
    :maxdepth: 1
    :hidden:

    install
    api


.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Installation
      :link: install
      :link-type: doc

      Instructions for installation

   .. grid-item-card:: API
      :link: api
      :link-type: doc

      API documentation
