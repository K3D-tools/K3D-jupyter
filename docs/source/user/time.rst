.. _time_series:

===========
Time series
===========

You have at your disposal two ways of changing data in a plot:

- Send new data from the backend
- Send time series in the form of a dictionary

-----------------------------
Sending data at each timestep
-----------------------------

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

-------------------------------------
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

You can control the animation from the :ref:`K3D panel <panel>` or through several attributes:

.. code-block::
    
    plot.start_auto_play() # Start the animation
    plot.stop_auto_play()  # Stop the animation

    plot.fps # Number of frame

    plot.time = O.5 # Read animation at a specific time

------------------
Discrete timesteps
------------------

Between keyframes the frontend interpolates, which assumes the data has a fixed size and that a
point keeps its identity from one frame to the next. That is false for a lidar scan, per frame
detections or particles that appear and disappear, so interpolation can be turned off:

.. code-block::

    plot.time_interpolation = False  # hold each keyframe until the next one

Playback then steps rather than glides. Keyframes of unequal size are never blended in either
mode: the nearer one is shown whole.

-----------------------
Stepping through frames
-----------------------

Stepping starts from the keyframe nearest the current time, so it behaves the same whether time
sits exactly on a frame or between two, and it clamps at both ends rather than wrapping. Each
call returns the new time.

.. code-block::

    plot.next_frame()      # -> 0.3
    plot.previous_frame()  # -> 0.2
    plot.step_frame(3)     # three frames forward

The :ref:`K3D panel <panel>` has *Previous frame* and *Next frame* in its ``Animation`` section
for the same thing. With ``time_interpolation`` off, the time slider snaps to the nearest
keyframe, since a continuous slider would otherwise stop between two of them.

See ``examples/time_series_frame_stepping.ipynb`` for the whole flow, including wiring the steps
to your own ipywidgets buttons.

------------------------
Inspecting the keyframes
------------------------

Both of these are answered by the plot itself, so they work headless and before ``display()``:

.. code-block::

    plot.get_time_series_times()  # [0.0, 0.1, 0.2, ...]
    plot.get_time_series_range()  # [min, max]

Objects need not share keys - the times are their union - and ``camera_animation`` counts too.
The range always contains ``0.0``, because that is what ``time`` is clamped to.

---------------
From JavaScript
---------------

An external control needs the same numbers, plus notice when something moves. On the frontend
instance:

.. code-block:: javascript

    K3DInstance.getTimeSeriesInfo();  // {min, max, times}
    K3DInstance.stepFrame(1);         // returns the new time

    K3DInstance.on(K3DInstance.events.TIME_CHANGE, (time) => { ... });
    K3DInstance.on(K3DInstance.events.AUTO_PLAY_CHANGE, (playing) => { ... });

``TIME_CHANGE`` fires on every frame while playback runs, so its listeners have to be cheap. It
is a notification only and is never written back to the kernel.

The standalone bundle also exports ``timeSeries.interpolateTimeSeries``,
``timeSeries.getObjectsWithTimeSeriesAndMinMax`` and ``timeSeries.getTimeSeriesTimes``, so a time
series can be read or driven from outside without reimplementing the interpolation.

.. |br| raw:: html
    
    <br />