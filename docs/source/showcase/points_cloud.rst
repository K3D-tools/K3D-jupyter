Points cloud
============

.. code::

    import numpy as np
    import k3d

    plot = k3d.plot(camera_auto_fit=False)

    data = np.load('./assets/points_cloud.npz')['arr_0']
    plot += k3d.points(data[:, 0:3],
                       data[:, 4].astype(np.uint32), point_size=0.15, shader="flat")
    plot.camera = [20.84, -3.06, 6.96,
                   0.67, 0.84, 3.79,
                   0.0, 0.0, 1.0]
    plot.display()

.. k3d_plot ::
   :filename: points_cloud_plot.py