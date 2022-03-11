.. _platonic-icosahedron:

platonic.Icosahedron
======================

.. autoclass:: k3d.platonic.Icosahedron
    :members:
    :show-inheritance:

**Example**

.. code-block:: python3

    import k3d
    from k3d import platonic

    plot = k3d.plot()

    ico_1 = platonic.Icosahedron()
    ico_2 = platonic.Icosahedron(origin=[5, -2, 3], size=0.5)

    plot += ico_1.mesh
    plot += ico_2.mesh

    plot.display()

.. k3d_plot ::
  :filename: plots/icosahedron_plot.py