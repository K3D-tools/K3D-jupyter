.. _gltf:

===========
glTF export
===========

A plot leaves K3D either as a PNG or as a stand-alone HTML snapshot: a picture, or a whole
viewer. Neither of them is a model. glTF export produces the third thing, the geometry itself,
as a binary ``.glb`` file that opens in Blender, MeshLab, Windows 3D Viewer or a slicer for
3D printing.

The file holds the same triangles the renderer drew, so the result of a computation becomes
something you can keep working on: an asset to light and render elsewhere, a mesh to print, a
supplementary artifact for a publication.

-------------------
From a Jupyter cell
-------------------

``fetch_gltf`` asks the browser to build the file. The answer travels back over the widget
comm, so it lands in ``plot.gltf`` only after the current cell has finished, and the write has
to happen in the next one:

.. code:: python3

    plot.fetch_gltf()

.. code:: python3

    from base64 import b64decode

    with open('scene.glb', 'wb') as f:
        f.write(b64decode(plot.gltf))

``plot.gltf`` carries the ``.glb`` file base64-encoded, the same convention
``plot.screenshot`` uses for PNG.

To keep it in a single cell, ``yield_gltfs`` turns the round trip into a generator that
resumes once the file arrives:

.. code:: python3

    @plot.yield_gltfs
    def export():
        plot.fetch_gltf()
        glb = yield

        with open('scene.glb', 'wb') as f:
            f.write(glb)

    export()

------------------
From the K3D panel
------------------

The ``Export glTF`` button in the :ref:`Controls section <panel>` saves the same file straight
from the browser. It needs no kernel behind it, so it also works inside a stand-alone HTML
snapshot: somebody who was sent a snapshot can still pull the model out of it.

------------------
Without a notebook
------------------

For scripts and continuous integration the headless driver offers a synchronous equivalent.
This is a plain Python file with no notebook and no widget involved, although it does need a
working Chrome installation:

.. code:: python3

    import k3d
    from k3d.headless import k3d_remote, get_headless_driver

    plot = k3d.plot()
    plot += k3d.marching_cubes(scalar_field, level=0.0)

    headless = k3d_remote(plot, get_headless_driver())
    headless.sync(hold_until_refreshed=True)

    with open('scene.glb', 'wb') as f:
        f.write(headless.get_gltf())

    headless.close()

.. note::

    Every route goes through a browser. Unlike ``get_snapshot()``, which Python assembles on
    its own, glTF export cannot run without one: for most object types Python holds only the
    *input* -- a scalar field, a voxel array, a height map -- and the triangles are produced
    by the renderer. ``mesh`` is the exception that already carries its own vertices.

------------------------
What the format can hold
------------------------

glTF describes triangles and PBR materials. An object whose shape exists only while its shader
runs has nothing to hand over, so it is left out rather than exported as the proxy geometry
that shader consumes -- a volume would otherwise arrive as a plain cube. Whatever was skipped
is listed in the browser console.

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Status
      - Objects
    * - Exported
      - ``mesh``, ``surface``, ``stl``, ``marching_cubes``, ``voxels``, ``sparse_voxels``,
        ``voxels_group``, ``texture`` built from an image, ``points`` with ``shader='mesh'``,
        ``line`` and ``lines`` with ``shader='mesh'`` (tubes) or ``shader='simple'``, and the
        arrowheads of ``vectors`` and ``vector_field``
    * - Left out
      - ``volume``, ``mip``, ``volume_slice``, ``texture`` built from an ``attribute``,
        ``points`` with shader ``'3d'``, ``'dot'`` or ``'flat'``, ``line`` and ``lines`` with
        ``shader='thick'``, ``text``, ``text2d``, ``label``, ``texture_text``, and the shafts
        of ``vectors`` and ``vector_field``

Switching a points or lines object to ``shader='mesh'`` is enough to bring it into the export.
That shader builds real geometry, instanced spheres and tubes, where the others draw
camera-facing impostors that the vertex and fragment stages assemble on every frame.

Objects hidden with ``visible = False`` are skipped as well, so the panel doubles as a way of
choosing what goes into the file. The grid, axes, lights and color legend are not part of the
model and never travel.
