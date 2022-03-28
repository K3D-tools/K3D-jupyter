.. _factory.marching_cubes:

factory.marching_cubes
======================

.. autofunction:: k3d.factory.marching_cubes

**Examples**

Sinus cube

.. code-block:: python3

    # Example from https://graphics.stanford.edu/~mdfisher/MarchingCubes.html

    import k3d
    import numpy as np
    from numpy import sin

    t = np.linspace(-5, 5, 50, dtype=np.float32)
    x, y, z = np.meshgrid(t, t, t, indexing='ij')

    scalars = sin(x*y + x*z + y*z) + sin(x*y) + sin(y*z) + sin(x*z) - 1

    marching = k3d.marching_cubes(scalars, level=0.0,
                                  color=0x0e2763,
                                  opacity=0.25,
                                  xmin=0, xmax=1,
                                  ymin=0, ymax=1,
                                  zmin=0, zmax=1,
                                  compression_level=9,
                                  flat_shading=False)

    plot = k3d.plot()
    plot += marching
    plot.display()

.. k3d_plot ::
  :filename: plots/marching_cubes_sinus_cube.py