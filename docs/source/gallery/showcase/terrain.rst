Terrain
=======

.. admonition:: References

    - :ref:`map_colors`
    - :ref:`paraview_color_maps`
    - :ref:`plot`
    - :ref:`surface`

.. code-block:: python3

    import numpy as np
    import k3d
    from pyvista import examples

    dem = examples.download_crater_topo()
    data = dem.get_array(0).reshape(dem.dimensions[::-1])[0, :, :].astype(np.float32)

    plot = k3d.plot()

    obj = k3d.surface(data,
                attribute=data,
                flat_shading=False,
                color_map = k3d.colormaps.matplotlib_color_maps.viridis,
                xmin=dem.bounds[0],
                xmax=dem.bounds[1],
                ymin=dem.bounds[2],
                ymax=dem.bounds[3])

    plot += obj
    plot.display()

.. k3d_plot ::
  :filename: plots/terrain_plot.py