.. _snapshots:

==================
Stand-alone HTML
==================

A snapshot is the whole plot as a single HTML file: geometry, colormaps, camera, panel and
renderer settings, in one document that opens in a browser with no Python behind it. Send it
to somebody, attach it to a paper, put it on a static site — they rotate the real thing rather
than looking at a picture of it.

This is also the answer to a question the other export paths do not cover. A PNG is flat, a
:ref:`glTF file <gltf>` is geometry without the scene, and a live widget needs a running
kernel. A snapshot is the viewer.

The gallery on this site is built out of snapshots, so every rotatable plot you have clicked
here is the same mechanism.

-----------------
From Python
-----------------

``get_snapshot`` builds the document without involving the browser at all, which makes it the
one export that works headless — in a script, in CI, from a notebook that is never displayed:

.. code:: python3

    import k3d
    import numpy as np

    plot = k3d.plot()
    plot += k3d.points(np.random.random((1000, 3)).astype(np.float32), point_size=0.05)

    with open('plot.html', 'w', encoding='utf-8') as f:
        f.write(plot.get_snapshot())

``plot.display()`` is not needed. Nothing has to be rendered first.

-------------------
Three kinds of file
-------------------

``plot.snapshot_type`` decides where the viewer's JavaScript comes from. The scene data is
identical in all three; only the size and the network dependency change.

.. list-table::
    :header-rows: 1
    :widths: 12 20 68

    * - Value
      - Needs network
      - What it is
    * - ``full``
      - no
      - The default. The whole K3D bundle is embedded in the file, so it opens offline and
        keeps working years from now regardless of what is on any CDN. Adds a few megabytes.
    * - ``online``
      - yes
      - A complete HTML document that pulls the bundle from unpkg. Small file, needs a
        connection, and pins the version it was written with.
    * - ``inline``
      - yes
      - Not a document but a fragment — a ``div`` and a ``script`` — meant to be pasted into a
        page you already have. Reuses that page's RequireJS if there is one.

Pick ``full`` when the file has to survive on its own, and ``online`` or ``inline`` when it is
going onto a page you control:

.. code:: python3

    plot.snapshot_type = 'online'

.. note::

    ``online`` and ``inline`` write the version of K3D that produced them into the CDN URL, so
    that version has to be published on npm for the file to load. A snapshot taken from an
    unreleased working tree will point at a URL that does not exist yet.

-------------------
From a Jupyter cell
-------------------

``fetch_snapshot`` asks the browser for the file instead of building it in Python, so it
captures the plot as it currently looks — including a camera you moved by hand. The answer
travels back over the widget comm, so it lands in ``plot.snapshot`` only after the current
cell has finished:

.. code:: python3

    plot.fetch_snapshot()

.. code:: python3

    with open('plot.html', 'w', encoding='utf-8') as f:
        f.write(plot.snapshot)

Unlike ``plot.screenshot`` and ``plot.gltf``, this trait holds the document as text — there is
nothing to base64-decode.

To keep it in one cell, ``yield_snapshots`` turns the round trip into a generator that resumes
once the file arrives:

.. code:: python3

    @plot.yield_snapshots
    def export():
        plot.fetch_snapshot()
        html = yield

        with open('plot.html', 'w', encoding='utf-8') as f:
            f.write(html)

    export()

------------------
From the K3D panel
------------------

``Snapshot HTML`` in the :ref:`Controls section <panel>` saves the same file straight from the
browser. It needs no kernel, so it works inside a snapshot as well: somebody who was sent one
can save their own from it, camera and all.

The plot area is also a drop target. Dropping a file on it loads that file into the running
plot:

* ``.html`` — a snapshot, replacing the current scene
* ``.stl`` — a mesh, added to it
* anything else is read as a binary snapshot

-------------------------------
Just the data, without the HTML
-------------------------------

``get_binary_snapshot`` returns the scene as zlib-compressed msgpack — the same bytes the HTML
carries, without the viewer around them. Useful when the plot is one part of something bigger:
a cache, a test fixture, a file your own page fetches and hands to K3D.

.. code:: python3

    with open('scene.k3d', 'wb') as f:
        f.write(plot.get_binary_snapshot())

``load_binary_snapshot`` reads it back into a plot, restoring both the objects and the plot
settings:

.. code:: python3

    plot = k3d.plot()

    with open('scene.k3d', 'rb') as f:
        plot.load_binary_snapshot(f.read())

    plot.display()

This is the format the panel accepts as a drop, so a ``.k3d`` file can be dragged onto any
plot, including one inside a snapshot.

--------------------
Running code on load
--------------------

``additional_js_code`` is JavaScript kept with the plot and executed once the viewer is ready,
with the instance in scope as ``K3DInstance``. It travels inside the snapshot, which makes it
the way to give a saved file behaviour it did not have in the notebook — hiding the panel,
refitting the camera when the file opens, wiring up a control of your own:

.. code:: python3

    plot = k3d.plot(additional_js_code='K3DInstance.setMenuVisibility(false);')

``get_snapshot`` also takes an ``additional_js_code`` argument, appended to the plot's own, for
code that belongs to one exported file rather than to the plot.

.. seealso::

    :ref:`glTF export <gltf>` for the geometry on its own, and the
    :ref:`Controls section <panel>` for the buttons described here.
