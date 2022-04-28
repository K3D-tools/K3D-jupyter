Streamlines
===========

:download:`streamlines_data.npz <./assets/streamlines_data.npz>`

.. code-block:: python3

    import k3d
    import numpy as np
    from k3d import matplotlib_color_maps

    data = np.load('streamlines_data.npz')

    v = data['v']
    lines = data['lines']
    vertices = data['vertices']
    indices = data['indices']

    plt_streamlines = k3d.line(lines,
                               width=0.00007,
                               attribute=v,
                               color_map=matplotlib_color_maps.Inferno,
                               color_range=[0, 0.5],
                               shader='mesh')

    plt_mesh = k3d.mesh(vertices, indices,
                        opacity=0.25,
                        wireframe=True,
                        color=0x0002)

    plot = k3d.plot(grid_visible=False)
    plot += plt_streamlines
    plot += plt_mesh
    plot.display()

    plot.camera = [0.0705, 0.0411, 0.0538,
                   0.0511, 0.0391, 0.0493,
                   -0.0798, 0.9872, 0.1265]

.. k3d_plot ::
  :filename: plots/streamlines_plot.py