.. _glb:

===
glb
===

.. autofunction:: k3d.factory.glb

.. seealso::
    - :ref:`gltf_factory`
    - :ref:`mesh`
    - :ref:`gltf` (export)
    - :doc:`../gallery/showcase/gltf-ring` - a Draco-compressed model with gems, path traced

--------
Examples
--------

A model from a file
^^^^^^^^^^^^^^^^^^^

Every part of the truck - body, wheels, glass - is its own object in the panel, in one folder
named after the file. `up` is left at 'y', glTF's own convention, so the model stands upright
in K3D's z-up scene.

.. code-block:: python3

    # Model: Cesium Milk Truck, (c) 2017 Cesium, CC-BY 4.0,
    # https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMilkTruck

    import k3d
    from k3d.helpers import download

    filename = download('https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/'
                        'main/Models/CesiumMilkTruck/glTF-Binary/CesiumMilkTruck.glb')

    truck = k3d.glb(filename)

    plot = k3d.plot(renderer='advanced')
    plot += truck
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/glb_basic_plot.py

The parts are ordinary objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

What comes back is a :class:`k3d.objects.Group` of plain :ref:`mesh` objects (and
:ref:`points` / :ref:`lines` for point and line primitives), so each part is changed like any
other K3D object. The group itself has no properties - it only holds them - with one
exception: its `model_matrix` moves the whole model, node hierarchy and all.

.. code-block:: python3

    for part in truck:
        print(part.name, part.vertices.shape)

    truck['Cesium_Milk_Truck/glass'].opacity = 0.3   # a part, by name

    truck.model_matrix = k3d.transform(translation=[0, 0, 2]).model_matrix

Materials from the file
^^^^^^^^^^^^^^^^^^^^^^^

Everything a glTF material carries becomes an ordinary :ref:`mesh` parameter: the colour
`texture`, the `normal_map` that gives the speaker grille its holes, the
`metalness_roughness_map` that makes the trim chrome and the body plastic, the `occlusion_map`
darkening the creases, and the `emissive_map` that lights the PLAY button. Each of them can be
read, replaced or switched off after reading like any other parameter.

The model is two centimetres wide, so `scaling` makes it a hundred times larger - the transform
arguments of every K3D factory place the whole model. The textures are halved here only because
this page carries every one of them inline; read from the file they are used at full size.

.. code-block:: python3

    # Model: BoomBox, Microsoft, CC0,
    # https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BoomBox

    import k3d
    from k3d.helpers import download

    filename = download('https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/'
                        'main/Models/BoomBox/glTF-Binary/BoomBox.glb')

    boombox = k3d.glb(filename, scaling=[100, 100, 100])

    part = boombox[0]
    print(part.alpha_mode, hex(part.emissive), len(part.normal_map))

    plot = k3d.plot(renderer='advanced', environment='studio', tone_mapping='aces',
                    grid_visible=False, background_color=0x1E2126)
    plot += boombox
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/glb_materials_plot.py

Bytes instead of a file
^^^^^^^^^^^^^^^^^^^^^^^

A ``.glb`` holds everything it needs, so its bytes are enough - from the network, from a
database, or from K3D's own export, which is z-up already:

.. code-block:: python3

    blob = plot.fetch_gltf()            # what this plot exports
    copy = k3d.glb(blob, up='z')
