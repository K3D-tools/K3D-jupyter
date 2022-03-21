.. _helpers.get_bounding_box_point:

helpers.get_bounding_box_point
==============================

.. autofunction:: k3d.helpers.get_bounding_box_point

**Examples**

.. code-block:: python3

    import k3d
    import numpy as np

    pos = np.array([1, 2, 3])

    box_point = k3d.helpers.get_bounding_box_point(pos)

    """
    array([1, 1, 2, 2, 3, 3])
    """