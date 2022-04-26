.. _check_attribute_color_range:

check_attribute_color_range
===========================

.. autofunction:: k3d.helpers.check_attribute_color_range

**Examples**

Color range

.. code-block:: python3

    import k3d
    import numpy as np

    attribute = np.linspace(0, 0.5, 100)

    color_range = k3d.helpers.check_attribute_range(attribute, color_range=[0, 1])

    """
    (0, 1)
    """

No color range

.. code-block:: python3

    import k3d
    import numpy as np

    attribute = np.linspace(0, 0.5, 100)

    color_range = k3d.helpers.check_attribute_range(attribute)

    """
    (0.0, 0.5)
    """