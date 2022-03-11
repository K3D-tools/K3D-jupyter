.. _platonic-dodecahedron:

platonic.Dodecahedron
======================

.. autoclass:: k3d.platonic.Dodecahedron
    :members:
    :show-inheritance:

**Example**

.. code-block:: python3

    import k3d
    from k3d import platonic

    plot = k3d.plot()

    dodec_1 = platonic.Dodecahedron()
    dodec_2 = platonic.Dodecahedron(origin=[5, -2, 3], size=0.5)

    plot += dodec_1.mesh
    plot += dodec_2.mesh

    plot.display()

.. k3d_plot ::
  :filename: plots/dodecahedron_plot.py