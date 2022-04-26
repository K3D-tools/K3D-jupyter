.. _transform:

transform
=========

.. autofunction:: k3d.transform.transform

**Examples**

.. code-block:: python3

    import numpy as np
    from k3d.transform import transform

    transform = transform(translation=[1, 3, 0],
                          rotation=[1, np.pi / 2, 0, 0],
                          scaling=[0.5, 1, 1.5])

    """
    Transform(bounds=None,
              translation=array([[1.], [3.], [0.]],
                                dtype=float32),
              rotation=array([0.87758255, 0.47942555, 0.        , 0.        ],
                             dtype=float32),
              scaling=array([0.5, 1. , 1.5],
                            dtype=float32))
    """