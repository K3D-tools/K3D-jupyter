.. _platonic.Cube:

platonic.Cube
=============

.. autoclass:: k3d.platonic.Cube
    :members:
    :show-inheritance:

**Example**

.. code-block:: python3

    import k3d
    from k3d import platonic

    plot = k3d.plot()

    cube_1 = platonic.Cube()
    cube_2 = platonic.Cube(origin=[5, -2, 3], size=0.5)

    plot += cube_1.mesh
    plot += cube_2.mesh

    plot.display()

.. k3d_plot ::
  :filename: plots/cube_plot.py