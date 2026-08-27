Plasma wind
===========

.. admonition:: References

    - :ref:`plot`
    - :ref:`volume`
    - :ref:`points`
    - :ref:`lines`

Five stars near a revolving pentagon - the symmetry deliberately broken -
blow spiral plasma arms into each other. The dye was advected through a
static wind field in the frame co-rotating with the stars; white streamlines
seeded on the source rings follow the same field, so they spiral along the
very filaments they created. With ``depth_peels`` the stars and the tubes
compose with the volume sample-accurately, and the advanced renderer adds
environment light and ambient occlusion.

The full simulation (pure numpy, a few minutes) lives in
``examples/volume_plasma_wind.ipynb``; the precomputed field below is enough
to render the scene.

:download:`plasma_wind.npz <./assets/plasma_wind.npz>`

.. code-block:: python3

    data = np.load('plasma_wind.npz')
    L = float(data['L'])

    plot = k3d.plot(grid_visible=False,
                    camera_auto_fit=False,
                    background_color=0x05060D)

    plot += k3d.volume(data['density'].astype(np.float32),
                       samples=512, alpha_coef=40,
                       color_map=k3d.matplotlib_color_maps.jet,
                       color_range=[50, 550],
                       bounds=[-L, L, -L, L, -L, L])

    plot += k3d.points(data['stars'], shader='mesh',
                       point_sizes=data['star_size'], mesh_detail=4,
                       colors=np.array([0x2040FF, 0x3060FF, 0x30A0FF,
                                        0x2080E0, 0x5050FF], dtype=np.uint32))

    plot += k3d.lines(data['line_vertices'], data['line_indices'],
                      indices_type='segment', shader='mesh',
                      width=0.006, color=0xFFFFFF)

    plot.depth_peels = 4
    plot.renderer = 'advanced'
    plot.environment = 'moonless_golf'
    plot.camera = [4.4, -4.4, 2.9, 0, 0, 0, 0, 0, 1]
    plot.display()

.. k3d_plot ::
  :filename: plots/plasma_wind_plot.py
