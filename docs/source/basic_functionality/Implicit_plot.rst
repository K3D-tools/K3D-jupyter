Implicit plot
=============

Marching cubes is an example of using frontend (client side) for doing computations.
In this case a function of three variables is sampled on 3d equidistant grid and send to
an `k3d.marching_cubes` object which will do the visualization.

Note that:
    - the amount of data exchanged between the frontend and backend is big
    - the browser javascript does the mesh computations
    - `level` is a single scalar parameter which can be passed to the frontend for data exploration
    - it is possible to use `jslink <https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html#Linking-widgets-attributes-from-the-client-side>`_ for interaction without a Python kernel.

.. code::

    import k3d
    import numpy as np
    from numpy import sin,cos,pi

    plot = k3d.plot()

    T = 1.6
    r = 4.77
    zmin, zmax = -r, r
    xmin, xmax = -r, r
    ymin, ymax = -r, r
    Nx, Ny, Nz = 37, 37, 37

    x = np.linspace(xmin, xmax, Nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, Ny, dtype=np.float32)
    z = np.linspace(zmin, zmax, Nz, dtype=np.float32)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    p = 2 - (cos(x + T * y) + cos(x - T * y) + cos(y + T * z) +
             cos(y - T * z) + cos(z - T * x) + cos(z + T * x))
    plt_iso = k3d.marching_cubes(p, compression_level=9, xmin=xmin, xmax=xmax,
                                 ymin=ymin, ymax=ymax,
                                 zmin=zmin, zmax=zmax, level=0.0,
                                 flat_shading=False)
    plot += plt_iso
    plot.display()

.. k3d_plot ::
   :filename: implicit_plot/implicit_plot01.py