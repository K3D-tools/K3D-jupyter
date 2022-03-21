.. _helpers.shape_validation:

helpers.shape_validation
==============================

.. autofunction:: k3d.helpers.shape_validation

.. note::

  This validation function is used along with *traitlets* and *traittypes* packages,
  which are used to ensure the utilization of the right format and type within classes attributes.

  See `traittypes documentation <https://traittypes.readthedocs.io/>`_ for more informations.

**Examples**

Valid shape

.. code-block:: python3

    from k3d.helpers import shape_validation
    import numpy as np
    from traitlets import HasTraits, TraitError
    from traittypes import Array

    class Foo(HasTraits):
        bar = Array(type=np.uint32).valid(shape_validation(3, 2))

    foo = Foo()

    foo.bar = np.array([[1, 2],
                        [3, 4],
                        [5, 6]])

    """
    """

Unvalid shape

.. code-block:: python3

    from k3d.helpers import shape_validation
    import numpy as np
    from traitlets import HasTraits, TraitError
    from traittypes import Array

    class Foo(HasTraits):
        bar = Array(type=np.uint32).valid(shape_validation(3, 2))

    foo = Foo()

    foo.bar = np.array([1, 2])

    """
    TraitError: Expected an array of shape (3, 2) and got (2,)
    """