.. _helpers.bounding_corners:

helpers.bounding_corners
========================

.. autofunction:: k3d.helpers.bounding_corners

**Examples**

Z-bounds

.. code-block:: python3

    import k3d
    import numpy as np

    bounds = np.array([1, 2, 3, 4])

    corners_coordinates = k3d.helpers.bounding_corners(bounds, z_bounds=(-1, 1))

    """
    array([[ 1,  3, -1],
           [ 1,  3,  1],
           [ 1,  4, -1],
           [ 1,  4,  1],
           [ 2,  3, -1],
           [ 2,  3,  1],
           [ 2,  4, -1],
           [ 2,  4,  1]])
    """

No z-bounds

.. code-block:: python3

    import k3d
    import numpy as np

    bounds = np.array([1, 2, 3, 4, 5])

    corners_coordinates = k3d.helpers.bounding_corners(bounds)

    """
    array([[1, 3, 5],
           [1, 4, 5],
           [2, 3, 5],
           [2, 4, 5]])
    """