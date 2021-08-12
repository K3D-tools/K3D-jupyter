Interaction and timeseries
==========================

There are two ways of changing data in the plot:

 - send new data from backend
 - sent time series in the form of dictionary `{t1:data1,t2:data2,...}`

.. code::

     import k3d
     import numpy as np

     x = np.random.randn(100,3).astype(np.float32)
     plot = k3d.plot(name='points')
     plt_points = k3d.points(positions=x, point_size=0.2, shader='3d')
     plot += plt_points

     plot.display()

.. k3d_plot ::
   :filename: interaction/Interaction01.py

Using backend to send a data at each timestep
---------------------------------------------

The Python backend can update attribute of any plot object in K3D-jupyter.

.. code::

    from time import sleep
    for t in range(10):
        plt_points.positions = x - t/10*x/np.linalg.norm(x,axis=-1)[:,np.newaxis]
        sleep(0.5)

Sending a dictionary of all timesteps
-------------------------------------

In this case it is possible to play an animation using only frontend.
Time is a string denoting wall time.

.. code::

    plt_points.positions = {str(t):x - t/10*x/np.linalg.norm(x,axis=-1)[:,np.newaxis] for t in range(10)}

.. k3d_plot ::
   :filename: interaction/Interaction02.py

The animation can be controlled from GUI or by several attributes:

.. code::

    plot.start_auto_play()  # plot.stop_auto_play()

The number of frames which are played can be inspected or set with plot.fps attribute.

.. code::

    plot.fps

One can programatically change or read the time in the animation using:

.. code::

    plot.time = 0.5