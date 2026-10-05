.. _gltf_factory:

====
gltf
====

.. autofunction:: k3d.factory.gltf

.. seealso::
    - :ref:`glb`
    - :ref:`mesh`
    - :ref:`gltf` (export)

--------
Examples
--------

A model with files beside it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``.gltf`` file is JSON. Here the geometry is in ``BoxTextured0.bin`` and the texture in
``CesiumLogoFlat.png``, both next to it, so the path is what is passed - the reader finds the
other two files relative to it.

:download:`BoxTextured.gltf <./assets/factory/BoxTextured/BoxTextured.gltf>`,
:download:`BoxTextured0.bin <./assets/factory/BoxTextured/BoxTextured0.bin>`,
:download:`CesiumLogoFlat.png <./assets/factory/BoxTextured/CesiumLogoFlat.png>`

.. code-block:: python3

    # Model: Box Textured, (c) 2017 Cesium, CC-BY 4.0,
    # https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BoxTextured

    import k3d

    box = k3d.gltf('BoxTextured/BoxTextured.gltf')

    plot = k3d.plot()
    plot += box
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/gltf_basic_plot.py

A self-contained .gltf
^^^^^^^^^^^^^^^^^^^^^^

When every buffer and image is inlined as a ``data:`` URI, the file needs nothing beside it,
and its bytes are read as well as its path. Bytes of a file that does refer to other files are
refused, with a message saying which one, because there is no directory to find it in.

.. code-block:: python3

    with open('model.gltf', 'rb') as f:
        model = k3d.gltf(f.read())
