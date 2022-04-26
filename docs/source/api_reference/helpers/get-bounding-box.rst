.. _get_bounding_box:

get_bounding_box
================

.. autofunction:: k3d.helpers.get_bounding_box

**Examples**

Set boundary

.. code-block:: python3

    import k3d
    import numpy as np

    model_matrix = np.array([[1, 2, 3, 4]])

    box = k3d.helpers.get_bounding_box(model_matrix, boundary=[-3, 3, -1, 1, 0, 5])

    """
    array([-5, 20])
    """

No set boundaries

.. code-block:: python3

    import k3d
    import numpy as np

    model_matrix = np.array([[1, 2, 3, 4],
                             [2, 4, 6, 8]])

    box = k3d.helpers.get_bounding_box(model_matrix)

    """
    array([-3.,  3., -6.,  6.])
    """