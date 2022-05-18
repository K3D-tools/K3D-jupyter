.. _Tetrahedron:

===========
Tetrahedron
===========

.. autoclass:: k3d.platonic.Tetrahedron
    :members:
    :show-inheritance:

.. seealso::
    - :ref:`Cube`
    - :ref:`Dodecahedron`
    - :ref:`Icosahedron`
    - :ref:`Octahedron`

-------
Example
-------

.. code-block:: python3

    import k3d
    from k3d import platonic

    plot = k3d.plot()

    tetra_1 = platonic.Tetrahedron()
    tetra_2 = platonic.Tetrahedron(origin=[5, -2, 3], size=0.5)

    plot += tetra_1.mesh
    plot += tetra_2.mesh

    plot.display()

.. k3d_plot ::
  :filename: plots/factory/tetrahedron_plot.py