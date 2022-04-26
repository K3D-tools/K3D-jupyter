.. _quad:

quad
====

.. autofunction:: k3d.helpers.quad

**Examples**

.. code-block:: python3

    import k3d

    vertices, indices = k3d.helpers.quad(1, 3)

    """
    array([-0.5, -1.5, 0.,
            0.5, -1.5, 0.,
            0.5,  1.5, 0.,
           -0.5,  1.5, 0.] ,dtype=float32)

    array([0, 1, 2, 0, 2, 3], dtype=uint32)
    """