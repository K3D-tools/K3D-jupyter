Lines
=====

https://matplotlib.org/examples/mplot3d/lines3d_demo.html

.. image:: https://matplotlib.org/2.0.2/_images/lines3d_demo1.png

.. code::

    import numpy as np
    import k3d

    plot = k3d.plot()

    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100, dtype=np.float32)
    z = np.linspace(-2, 2, 100, dtype=np.float32)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    line = k3d.line(np.vstack([x,y,z]).T, width=0.2, scaling = [1,1,2])

    plot += line
    plot.display()

.. k3d_plot ::
   :filename: lines_plot.py