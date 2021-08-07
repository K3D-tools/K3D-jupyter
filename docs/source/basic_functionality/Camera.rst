Camera
======

Camera is 9-th vector:

.. code::

    [
     x1,y1,z1, # position of the camera in xyz space
     x2,y2,z2, # the point where camera is currently looking at
     x3,y3,z3  # orientation (up direction), this vector cannot be [0,0,0])
    ]


Is is synchronized between frontend and backend automatically.
Below there is an example of camera manipulation in Python backend.


.. code::

    import k3d
    import numpy as np
    from numpy import sin,cos,pi
    from k3d.platonic import Icosahedron

    plot = k3d.plot()
    plot += Icosahedron().mesh
    plot += Icosahedron((0,2,1),size=0.3).mesh
    plot.display()

.. k3d_plot ::
   :filename: camera/camera01.py

Look at bigger icosahedron from above (z>0)  and first quarter of xy plane:

.. code::

    plot.camera = [5, 5, 3] + \
                  [0, 0, 0] + \
                  [0, 0, 1]

.. k3d_plot ::
   :filename: camera/camera02.py

Look at smaller icosahedron from above (z>0)

.. code::

    plot.camera = [2, 2, 3] + \
                  [0, 2, 1] + \
                  [0, 0, 1]

.. k3d_plot ::
   :filename: camera/camera03.py

Look at larger icosahedron from a point above  its center orienting camera to have y-axis up.

.. code::

    plot.camera = [0.01, 0.01, 8] + \
                  [0, 0, 0] + \
                  [0, 1, 0]

.. k3d_plot ::
   :filename: camera/camera04.py