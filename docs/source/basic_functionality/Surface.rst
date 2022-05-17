Surface
=======

`k3d.surface` is an easy way to produce a plot of an explicit function of two variables
f(x,y) on a rectangular domain.

It takes a table of values, i.e. `f(x_i,y_i) = f[j,i]`. Proper scaling of x and y
axes can be done by specifying `xmin/xmax` parameters.

.. code::

    import k3d
    import numpy as np

    plot = k3d.plot()

    Nx, Ny = 50, 60
    xmin, xmax, ymin, ymax = -3, 3, 0, 3
    x = np.linspace(xmin, xmax, Nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, Ny, dtype=np.float32)
    x, y = np.meshgrid(x, y)
    f = np.sin(x ** 2 + y ** 2)
    plt_surface = k3d.surface(f, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    plot += plt_surface

    plot.display()

.. k3d_plot ::
   :filename: surface/Surface01.py
