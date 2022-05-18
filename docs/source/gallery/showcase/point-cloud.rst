Point cloud
===========

.. admonition:: References

    - :ref:`plot`
    - :ref:`points`

:download:`point_cloud.npz <./assets/point_cloud.npz>`

.. code-block:: python3

    import k3d
    import numpy as np

    data = np.load('point_cloud.npz')['arr_0']

    plt_points = k3d.points(data[:, 0:3],
                            data[:, 4].astype(np.uint32),
                            point_size=0.15,
                            shader="flat")

    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False,
                    background_color=0x87ceeb)
    plot += plt_points
    plot.display()

    plot.camera = [20.84, -3.06, 6.96,
                   0.67, 0.84, 3.79,
                   0.0, 0.0, 1.0]

.. k3d_plot ::
  :filename: plots/point_cloud_plot.py