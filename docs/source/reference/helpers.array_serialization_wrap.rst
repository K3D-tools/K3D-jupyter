.. _array_serialization_wrap:

========================
array_serialization_wrap
========================

.. autofunction:: k3d.helpers.array_serialization_wrap

.. note::

  This function is used along with the `ipywidgets` package,
  which provides custom object serializations.

  See `ipywidgets documentation <https://ipywidgets.readthedocs.io/en/7.x/examples/Widget%20Low%20Level.html?highlight=to_json#Serialization-of-widget-attributes>`_ for more informations.

--------
Examples
--------

.. code-block:: python3

    # Example from objects.VoxelChunk

    coord = Array(dtype=np.uint32).tag(sync=True, **array_serialization_wrap("coord"))