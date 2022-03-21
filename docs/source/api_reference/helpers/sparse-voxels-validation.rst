.. _helpers.sparse_voxels_validation:

helpers.sparse_voxels_validation
================================

.. autofunction:: k3d.helpers.sparse_voxels_validation

.. note::

  This validation function is used along with *traitlets* and *traittypes* packages,
  which are used to ensure the utilization of the right format and type within classes attributes.

  See `traittypes documentation <https://traittypes.readthedocs.io/>`_ for more informations.

**Examples**

Valid sparse voxels

.. code-block:: python3

    from k3d.helpers import sparse_voxels_validation
    import numpy as np
    from traitlets import HasTraits, TraitError
    from traittypes import Array

    class Foo(HasTraits):
        bar = Array(type=np.uint32).valid(sparse_voxels_validation())

    foo = Foo()

    foo.bar = np.array([[1, 2, 4, 5],
                        [6, 7, 8, 9]])

    """
    """

Unvalid sparse voxels - wrong shape

.. code-block:: python3

    from k3d.helpers import sparse_voxels_validation
    import numpy as np
    from traitlets import HasTraits, TraitError
    from traittypes import Array

    class Foo(HasTraits):
        bar = Array(type=np.uint32).valid(sparse_voxels_validation())

    foo = Foo()

    foo.bar = np.array([[1, 2, 3],
                        [4, 5, 6]])

    """
    TraitError: Expected an array of shape (N, 4) and got (2, 3)
    """

Unvalid sparse voxels - negative coordinates

.. code-block:: python3

    from k3d.helpers import sparse_voxels_validation
    import numpy as np
    from traitlets import HasTraits, TraitError
    from traittypes import Array

    class Foo(HasTraits):
        bar = Array(type=np.uint32).valid(sparse_voxels_validation())

    foo = Foo()

    foo.bar = np.array([[0, 0, 0, 0],
                        [1, 1, 1, 1],
                        [-2, -2, -2, -2]])

    """
    TraitError: Voxel coordinates and values must be non-negative
    """