Streamlines
===========

.. code::

    import numpy as np
    import k3d

    plot = k3d.plot()

    data = np.load('./assets/streamlines_data.npz')
    v = data['v']
    lines = data['lines']
    vertices = data['vertices']
    indices = data['indices']

    plt_streamlines = k3d.line(lines, attribute=v, width=0.00007,
                               color_map=k3d.matplotlib_color_maps.Inferno,
                               color_range=[0, 0.5], shader='mesh')

    plt_mesh = k3d.mesh(vertices, indices, opacity=0.25, wireframe=True, color=0x0002)

    plot.camera = [0.064, 0.043, 0.043, 0.051, 0.041, 0.049, -0.059, 0.993, 0.087]
    plot += plt_streamlines
    plot += plt_mesh

    plot.display()

.. k3d_plot ::
   :filename: streamlines_plot.py

Computing streamlines in VTK
----------------------------

Indices and vertices can be computed from e.g. stl file

.. code::

    import vtk
    reader = vtk.vtkSTLReader()
    reader.SetFileName('c0006.stl')
    reader.Update()
    geometry = reader.GetOutput()

    import pyacvd
    import pyvista as pv
    pv_temp = pv.PolyData(geometry)

    cluster = pyacvd.Clustering(pv_temp)
    cluster.cluster(1234)
    remesh = cluster.create_mesh()
    remesh_vtk = vtk.vtkPolyData()
    remesh_vtk.SetPoints(remesh.GetPoints())
    remesh_vtk.SetVerts(remesh.GetVerts())
    remesh_vtk.SetPolys(remesh.GetPolys())

    plt_mesh = k3d.vtk_poly_data(remesh_vtk, opacity=0.831, wireframe=True, color=0xaaaaaa)

    np.savez('streamlines_data.npz',v=v, lines=lines, \
        vertices = plt_mesh.vertices, indices=plt_mesh.indices)
