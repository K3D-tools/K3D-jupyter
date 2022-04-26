.. _json_to_array:

json_to_array
=============

.. autofunction:: k3d.helpers.json_to_array

**Examples**

.. code-block:: python3

    import k3d
    import numpy as np

    ar = np.array([[1, 2, 3], [4, 5, 6]])
    data = k3d.helpers.array_to_json(ar, compression_level=7)

    ar_deserialized = k3d.helpers.json_to_array(data)

    """
    array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)
    """