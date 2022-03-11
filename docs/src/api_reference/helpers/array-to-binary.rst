.. heplers-array-to-binary:

helpers.array_to_binary
=======================

.. autofunction:: k3d.helpers.array_to_binary

**Examples**

Data compression

.. code-block:: python3

    import k3d
    import numpy as np

    array = np.array([[1, 2, 3], [4, 5, 6]])

    data = k3d.helpers.array_to_binary(ar, compression_level=2)

    """
    {
      'compressed_data': b'x^cd```\x02bf f\x01bV f\x03b\x00\x00\xf8\x00\x16',
      'dtype': 'int32',
      'shape': (2, 3)
    }
    """

No data compression

.. code-block:: python3

    import k3d
    import numpy as np

    array = np.array([[1, 2, 3], [4, 5, 6]])

    data = k3d.helpers.array_to_binary(ar)

    """
    {
      'data': <memory at 0x7faaae2a1e80>,
      'dtype': 'int32',
      'shape': (2, 3)
    }
    """