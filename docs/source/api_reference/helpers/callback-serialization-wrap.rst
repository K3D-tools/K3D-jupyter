.. _callback_serialization_wrap:

callback_serialization_wrap
===========================

.. autofunction:: k3d.helpers.callback_serialization_wrap

.. note::

  This function is used along with the `ipywidgets` package,
  which provides custom object serializations.

  See `ipywidgets documentation <https://ipywidgets.readthedocs.io/en/7.x/examples/Widget%20Low%20Level.html?highlight=to_json#Serialization-of-widget-attributes>`_ for more informations.

**Examples**

.. code-block:: python3

    # Example from objects.DrawableWithCallback

    click_callback = Any(default_value=None, allow_none=True).tag(sync=True, **callback_serialization_wrap("click_callback"))