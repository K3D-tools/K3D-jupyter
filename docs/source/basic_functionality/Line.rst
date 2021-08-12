Line
====

Let us draw a trajectory of `N` steps of an approximation to the
[Wiener Process](https://en.wikipedia.org/wiki/Wiener_process) in three dimensions.

Below, a blue thin line is a trajectory and the total displacement is shown with red thick line.

Line takes data as an array of coordinates `[number_of_points,3]`.

.. code::

    import numpy as np
    import k3d

    plot = k3d.plot(name='Wiener process')
    N = 1000
    traj = np.cumsum(np.random.randn(N, 3).astype(np.float32), axis=0)
    plt_line = k3d.line(traj, shader='mesh', width=0.5)
    plt_line2 = k3d.line([traj[0], traj[-1]], shader='mesh', width=.5, color=0xff0000)
    plot += plt_line
    plot += plt_line2
    plot.display()

.. k3d_plot ::
   :filename: line/Line01.py