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

    t = np.linspace(-5, 5, 100, dtype=np.float32)
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

Levels

.. code-block:: python3

    import k3d
    import numpy as np

    t = np.linspace(-1.5, 1.5, 50, dtype=np.float32)
    x, y, z = np.meshgrid(t, t, t, indexing='ij')
    R = 1
    r = 0.5

    eq_heart = (x**2 + (9/4 * y**2) + z**2 - 1)**3 - (x**2 * z**3) - (9/200 * y**2 * z**3)
    eq_torus = (x**2 + y**2 + z**2 + R**2 - r**2)**2 - 4 * R**2 * (x**2 + y**2)

    plot = k3d.plot()

    for i in range(3):
        level = 0 + i * 1.5

        plt_heart = k3d.marching_cubes(eq_heart, level=level,
                                      color=0xe31b23,
                                      xmin=-1.5, xmax=1.5,
                                      ymin=-1.5, ymax=1.5,
                                      zmin=-1.5, zmax=1.5,
                                      translation=[i * 3.5, 0, 0])
        plt_torus = k3d.marching_cubes(eq_torus, level=level,
                                      color=0x5aabac,
                                      xmin=-1.5, xmax=1.5,
                                      ymin=-1.5, ymax=1.5,
                                      zmin=-1.5, zmax=1.5,
                                      translation=[i * 3.5, 0, -3.5])

        plot += plt_heart
        plot += plt_torus

    plot.display()

.. k3d_plot ::
  :filename: plots/marching_cubes_sinus_cube.py