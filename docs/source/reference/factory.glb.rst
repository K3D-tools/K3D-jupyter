.. _glb:

===
glb
===

.. autofunction:: k3d.factory.glb

.. seealso::
    - :ref:`gltf_factory`
    - :ref:`mesh`
    - :ref:`gltf` (export)

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

Bytes instead of a file
^^^^^^^^^^^^^^^^^^^^^^^

A ``.glb`` holds everything it needs, so its bytes are enough - from the network, from a
database, or from K3D's own export, which is z-up already:

.. code-block:: python3

    blob = plot.fetch_gltf()            # what this plot exports
    copy = k3d.glb(blob, up='z')
