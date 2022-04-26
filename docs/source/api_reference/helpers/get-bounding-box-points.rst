.. _get_bounding_box_points:

get_bounding_box_points
=======================

.. autofunction:: k3d.helpers.get_bounding_box_points

**Examples**

.. code-block:: python3

    import k3d
    import numpy as np

    arr = np.array([[-47,   0,  34],
                    [-34,  11,  30],
                    [ 13,  49, -48],
                    [-48,  30,   8],
                    [-10, -40, -12],
                    [-27,  40,   4],
                    [-40,  18,   3],
                    [-28,  43, -43],
                    [-35, -12, -21],
                    [-10, -18, -40]])

    model_matrix = np.identity(4)

    box_points = k3d.helpers.get_bounding_box_points(arr, model_matrix)

    """
    array([-48.,  13., -40.,  49., -48.,  34.])
    """