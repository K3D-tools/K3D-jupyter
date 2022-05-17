.. _get_bounds_fit_matrix:

=====================
get_bounds_fit_matrix
=====================

.. autofunction:: k3d.transform.get_bounds_fit_matrix

--------
Examples
--------

.. code-block:: python3

    from k3d.transform import get_bounds_fit_matrix

    bfm = get_bounds_fit_matrix(0, 1, -1, 1, 2, 4)

    """
    array([[1. , 0. , 0. , 0.5],
           [0. , 2. , 0. , 0. ],
           [0. , 0. , 2. , 3. ],
           [0. , 0. , 0. , 1. ]], dtype=float32)
    """