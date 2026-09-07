.. _headless:

=================
Headless plots
=================

A K3D plot is a browser widget, so anything that draws one needs a browser. ``k3d.headless``
gives you one without a notebook: it starts a small HTTP server, points a Selenium-driven
browser at it, and hands back an object that pushes the plot's state over and asks for
pictures.

.. code-block:: python3

    import k3d
    from k3d.headless import k3d_remote, get_headless_driver

    plot = k3d.plot(screenshot_scale=1)
    plot += k3d.points([[0, 0, 0], [1, 1, 1]], point_size=0.2)

    headless = k3d_remote(plot, get_headless_driver(), width=1280, height=720)
    headless.sync()

    with open('plot.png', 'wb') as f:
        f.write(headless.get_screenshot())

    headless.close()

This is what the library's own reference images and this documentation's plots are made
with, and it is the supported way to render a sequence of frames from a script.

.. warning::
    Neither of the two packages this needs is a dependency of ``k3d`` itself:
    ``pip install selenium flask``, or ``pip install k3d[dev]``, which brings them
    along with the test suite's own requirements. The Chrome driver additionally needs
    Chrome or Chromium on the machine; Selenium 4.6 and later fetches the matching
    ``chromedriver`` by itself.

-----------
The session
-----------

``k3d_remote(plot, driver, width=1280, height=720, port=8080)`` owns three things - the
server, the browser and the plot - and ``close()`` releases all of them. It quits the
driver rather than closing its window, so nothing is left behind; a script that forgets
it leaves a browser and a ``chromedriver`` running.

``width`` and ``height`` are the browser window, and the plot fills it. They are what
sets the output resolution, together with ``screenshot_scale``:

.. code-block:: python3

    plot = k3d.plot(screenshot_scale=2)
    headless = k3d_remote(plot, get_headless_driver(), width=1920, height=1080)
    # every get_screenshot() is now 3840x2160

Two drivers ship with the module. ``get_headless_driver()`` is Chrome, and by default it
runs with ``--enable-unsafe-swiftshader``, so it renders in software and works on a
machine with no usable GPU at all - a CI runner, a container without passthrough. Pass
``gpu=True`` for the real card, ``no_headless=True`` to watch the window, and
``extra_args=[...]`` for switches the library should not choose for everyone.
``get_headless_firefox_driver()`` is the Firefox equivalent.

.. note::
    Software rendering is correct and slow. Every image comes out right, so the only
    thing that tells you which one you got is the clock - see ``get_gl_info()`` below,
    and :ref:`renderers` for why the texture limits are not the tell.

-----------------
Pushing the state
-----------------

Nothing reaches the browser until you say so. ``sync()`` sends whatever changed since
the last call - a diff, so pushing a moved camera does not re-upload the volume - and
returns as soon as the page has been *asked* to apply it.

For a screenshot that is not enough:

.. code-block:: python3

    for i in range(frames):
        plot.camera = camera_for(i)
        headless.sync(hold_until_refreshed=True)

        with open('frame_%06d.png' % i, 'wb') as f:
            f.write(headless.get_screenshot(True))

The state travels over an asynchronous request, so a screenshot taken straight after a
bare ``sync()`` renders whichever scene the page happens to be holding - sometimes the
one you just set, sometimes the previous one. The symptom is two byte-identical files in
the middle of a sequence. ``hold_until_refreshed=True`` waits for the page to confirm it
has the new state.

``camera_reset(factor=1.5)`` frames the whole scene, the same as the panel's button, for
when you are not driving the camera yourself.

-------
Outputs
-------

``get_screenshot(only_canvas=False)`` returns the PNG as ``bytes``. It always renders the
full sample budget, so a ``cinematic`` frame is as clean as ``cinematic_samples`` allows
however long that takes. ``only_canvas=True`` gives you the 3D canvas alone, without the
HTML overlay - labels, the colour-map legend - rasterised on top of it, which is both
faster and what you want when the overlay would sit in the middle of a video frame.

The image is posted back over the session's own HTTP server rather than returned from the
browser, because every value returned from a browser script stays in the page's heap for
the life of that browser and nothing releases it. At 4K that was about 10 MB a frame, so
a few hundred frames used to end an animation with a heap exhaustion.

``get_browser_screenshot()`` is the driver's own capture of the whole window - the plot
as the browser composited it, at window resolution, with the panel in the corner. It is a
debugging view, not an output format.

``get_gltf()`` returns the scene geometry as a binary ``.glb``; see :ref:`gltf` for what a
glTF can and cannot carry from a K3D scene.

-----------
Diagnostics
-----------

.. code-block:: python3

    headless.get_gl_info()
    # {'vendor': 'Google Inc. (NVIDIA)', 'renderer': 'ANGLE (NVIDIA, ...)', ...}

What the browser actually got. A container that loses its GPU passthrough does not fail,
it falls back to software rendering, and the renderer string is where that shows.

.. code-block:: python3

    driver = get_headless_driver(extra_args=['--enable-precise-memory-info',
                                             '--js-flags=--expose-gc'])
    headless = k3d_remote(plot, driver)

    headless.get_memory()
    # {'used_mb': 878.1, 'total_mb': 906.5, 'limit_mb': 4192.0,
    #  'collected': True, 'precise': True}

The page's JS heap, for watching a long run. Both switches are needed and neither is
assumed: without ``--enable-precise-memory-info`` Chrome answers with a frozen constant
whatever the page allocates, and without ``--js-flags=--expose-gc`` a reading carries the
last frame's garbage. ``precise`` and ``collected`` report which of them you have, so a
flat line means a flat heap rather than a frozen counter. ``used_mb`` against ``limit_mb``
is the headroom; ``used_mb`` against ``total_mb`` is the fragmentation.

Volume data shows up in ``used_mb`` but cannot exhaust ``limit_mb``: a typed array's
backing store is counted in the reading and allocated outside the heap the limit applies
to. Only strings and objects - the PNGs a frame loop makes, for instance - are heap
resident.

--------
Logging
--------

A session says nothing about its own progress. It answers a ``/ping`` every few seconds
and ``sync()`` runs once per frame of an animation, so a line for each would bury whatever
the cell was asked to show. Warnings and errors are never suppressed. To get the rest
back:

.. code-block:: python3

    import logging

    logging.getLogger('k3d.headless').setLevel(logging.DEBUG)

A startup or a refresh that never completes raises with the reason and whatever the
browser console said, rather than hanging.
