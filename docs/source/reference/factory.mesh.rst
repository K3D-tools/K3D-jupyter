.. _mesh:

====
mesh
====

.. autofunction:: k3d.factory.mesh

.. seealso::
    - :ref:`surface`

--------
Examples
--------

Basic
^^^^^

.. code-block:: python3

    import k3d

    vertices = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    indices = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]]

    plt_tetra = k3d.mesh(vertices, indices,
                         colors=[0x32ff31, 0x37d3ff, 0xbc53ff, 0xffc700])

    plot = k3d.plot()
    plot += plt_tetra
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/mesh_basic_plot.py

Colormap
^^^^^^^^

.. attention::
    `color_map` must be used along with `attribute` and `color_range` in order to work correctly.

.. code-block:: python3

    import k3d
    import numpy as np
    from k3d.colormaps import matplotlib_color_maps
    from matplotlib.tri import Triangulation

    n_radii = 8
    n_angles = 36

    radii = np.linspace(0.125, 1.0, n_radii, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False, dtype=np.float32)[..., np.newaxis]

    x = np.append(np.float32(0), (radii * np.cos(angles)).flatten())
    y = np.append(np.float32(0), (radii * np.sin(angles)).flatten())
    z = np.sin(-x * y)

    vertices = np.vstack([x, y, z]).T
    indices = Triangulation(x, y).triangles.astype(np.uint32)

    plt_mesh = k3d.mesh(vertices, indices,
                        color_map=matplotlib_color_maps.Jet,
                        attribute=z,
                        color_range=[-1.1, 2.01])

    plot = k3d.plot()
    plot += plt_mesh
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/mesh_colormap_plot.py

Materials
^^^^^^^^^

.. versionadded:: 3.2.0

Besides a colour texture a mesh takes a normal map, a metalness-roughness map, an occlusion
map and an emissive colour with its own map - images given as their encoded bytes, read with
`uvs`. `alpha_mode` decides what the alpha of the texture and of `opacities` does: nothing
('opaque'), blending ('blend') or cutting out below `alpha_cutoff` ('mask'). `transmission`
with `ior`, `thickness` and the attenuation makes glass, water or gems, refracted for real by
the :ref:`cinematic` renderer.

.. code-block:: python3

    import io

    import numpy as np
    from PIL import Image

    import k3d


    def png(pixels):
        buffer = io.BytesIO()
        Image.fromarray(pixels.astype(np.uint8)).save(buffer, format='PNG')
        return buffer.getvalue()


    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)

    y, x = (np.mgrid[0:128, 0:128] + 0.5) / 32 % 1.0 - 0.5
    r = np.hypot(x, y)

    inside = r < 0.4
    nx, ny = np.where(inside, x / 0.4, 0), np.where(inside, -y / 0.4, 0)
    nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
    bumps = (np.stack([nx, ny, nz], -1) * 0.5 + 0.5) * 255

    rings = np.where(np.abs(r - 0.3) < 0.06, 255, 0)[..., None] * np.array([1.0, 0.6, 0.2])
    discs = np.dstack([np.full(r.shape, 60), np.full(r.shape, 140), np.full(r.shape, 230),
                       np.where(r < 0.35, 0, 255)])

    plot = k3d.plot(renderer='advanced', grid_visible=False)

    plot += k3d.mesh(vertices, indices, uvs=uvs, color=0xB0B8C0, metalness=0.6,
                     roughness=0.3, normal_map=png(bumps), flat_shading=False)
    plot += k3d.mesh(vertices + [1.1, 0, 0], indices, uvs=uvs, color=0x202020,
                     emissive=0xFFFFFF, emissive_map=png(rings), emissive_intensity=1.5)
    plot += k3d.mesh(vertices + [2.2, 0, 0], indices, uvs=uvs, texture=png(discs),
                     alpha_mode='mask', side='double')
    plot.display()

.. k3d_plot ::
  :filename: plots/factory/mesh_materials_plot.py

Colours multiply
^^^^^^^^^^^^^^^^

.. versionchanged:: 3.2.0

`color` multiplies `colors`, the colormap and the texture - the way a base colour does in other
renderers - where it used to be ignored next to any of them. Left out, it is white when one of
them is given, so a mesh coloured per vertex looks as it always did; given, it tints them.
