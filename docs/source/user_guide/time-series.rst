.. _time_series:

Time series
===========

You have at your disposal two ways of changing data in a plot:

- send new data from the backend
- send time series in the form of a dictionary

Using the backend to send a data at each timestep
-------------------------------------------------

You can update a plot object attribute using the Python backend.

.. code-block:: python3

    import k3d
    import numpy as np
    import time

    np.random.seed(2022)

    x = np.random.randn(100,3).astype(np.float32)

    plt_points = k3d.points(x,
                            color=0x528881,
                            point_size=0.2)

    plot = k3d.plot()
    plot += plt_points
    plot.display()

    for t in range(20):
        plt_points.positions = x - t/10*x/np.linalg.norm(x,axis=-1)[:,np.newaxis]
        time.sleep(0.5)

Sending a dictionary of all timesteps
-------------------------------------

You can create an animation using only the frontend. |br|
Time is represented as a ``str`` denoting wall time.

.. code-block::

    import k3d
    import numpy as np

    np.random.seed(2022)

    x = np.random.randn(100,3).astype(np.float32)

    plt_points = k3d.points(x,
                            color=0x528881,
                            point_size=0.2)

    plot = k3d.plot()
    plot += plt_points
    plot.display()

    plt_points.positions = {str(t):x - t/5*x/np.linalg.norm(x,axis=-1)[:,np.newaxis] for t in range(10)}
    plot.start_auto_play()

.. k3d_plot ::
  :filename: plots/time_series_frontend_plot.py

You can control the animation from the GUI or through several attributes:

.. code-block::

    plot.start_auto_play() # Start the animation
    plot.stop_auto_play()  # Stop the animation

    plot.fps # Number of frame

    plot.time = O.5 # Read animation at a specific time

.. |br| raw:: html

   <br />