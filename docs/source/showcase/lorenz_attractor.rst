Lorenz attractor
================

https://matplotlib.org/examples/mplot3d/lorenz_attractor.html

.. image:: https://matplotlib.org/_images/lorenz_attractor.png

.. code::

    import numpy as np
    import k3d

    plot = k3d.plot()

    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return x_dot, y_dot, z_dot

    dt = 0.01
    stepCnt = 10000

    # Need one more for the initial values
    xs = np.empty((stepCnt + 1,), dtype=np.float32)
    ys = np.empty((stepCnt + 1,), dtype=np.float32)
    zs = np.empty((stepCnt + 1,), dtype=np.float32)

    # Setting initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)

    # Stepping through "time".
    for i in range(stepCnt):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    line = k3d.line(np.vstack([xs, ys, zs]).T)

    plot += line
    plot.display()

.. k3d_plot ::
   :filename: lorenz_attractor_plot.py
