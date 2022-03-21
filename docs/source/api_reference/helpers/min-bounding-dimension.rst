.. _helpers.min_bounding_dimension:

helpers.min_bounding_dimension
==============================

.. autofunction:: k3d.helpers.min_bounding_dimension

**Examples**

.. code-block:: python3

    import k3d
    import numpy as np

    bounds = np.array([-4, 4, 0, 1, -2, 2, -3, 1])

    min_dim = k3d.helpers.min_bounding_dimension(bounds)

    """
    1
    """