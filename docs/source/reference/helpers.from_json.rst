.. _from_json:

=========
from_json
=========

.. autofunction:: k3d.helpers.from_json

.. note::

  This function is used along with the `ipywidgets` package,
  which provides custom object serializations.

  See `ipywidgets documentation <https://ipywidgets.readthedocs.io/en/7.x/examples/Widget%20Low%20Level.html?highlight=to_json#Serialization-of-widget-attributes>`_ for more informations.

--------
Examples
--------

.. code-block:: python3

    # Example from helpers.array_serialization_wrap

    def array_serialization_wrap(name):
    return {
        "to_json": (lambda input, obj: to_json(name, input, obj)),
        "from_json": from_json,
    }