Stanford dragon
===============

.. admonition:: References

    - :ref:`vtk_poly_data`
    - :ref:`mesh`
    - :ref:`plot`
    - :ref:`cinematic`

The classic scan - 871k triangles - read with VTK and path traced. The model is
the reason to reach for :ref:`cinematic` here: the scales and the coiled tail are
all concavity, so the shadow under the belly, the light the floor throws back
into the flank, and the sheen running along the spine come out of the simulation
rather than from a screen-space approximation.

Two details do most of the work. The scan is authored Y-up while K3D is Z-up, so
it is rotated in VTK rather than through the mesh's model matrix - that keeps the
bounds honest, and the camera and the floor below are derived from them. And the
floor is not decoration: it is where the bounced light comes from, and without it
a path traced model floats in the environment and reads flatter than it is.

Both images are the same scene at 256 samples: the gallery thumbnail accumulates
them once, when these pages are built, the plot below in your browser.
``tone_mapping='aces'`` matters at this budget - bounced light between the gold
and the floor genuinely exceeds 1.0, and without a curve it clips.

.. code-block:: python3

    import zipfile

    import numpy as np
    import vtk

    import k3d
    from k3d.helpers import download

    filename = download('https://casual-effects.com/g3d/data10/research/model/dragon/dragon.zip')

    with zipfile.ZipFile(filename) as archive:
        archive.extract('dragon.obj')

    reader = vtk.vtkOBJReader()
    reader.SetFileName('dragon.obj')

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(reader.GetOutputPort())

    to_z_up = vtk.vtkTransform()
    to_z_up.RotateX(90)

    upright = vtk.vtkTransformPolyDataFilter()
    upright.SetTransform(to_z_up)
    upright.SetInputConnection(triangles.GetOutputPort())
    upright.Update()

    dragon = upright.GetOutput()

    bounds = np.array(dragon.GetBounds()).reshape(3, 2)
    centre = bounds.mean(axis=1)
    size = float((bounds[:, 1] - bounds[:, 0]).max())

    plot = k3d.plot(renderer='cinematic',
                    environment='venice_sunset',
                    tone_mapping='aces',
                    grid_visible=False,
                    camera_auto_fit=False,
                    cinematic_samples=256,
                    cinematic_bounces=6)

    plot += k3d.vtk_poly_data(dragon,
                              color=0xC8A15A,
                              flat_shading=False,
                              roughness=0.2,
                              metalness=0.7,
                              compression_level=5,
                              name='dragon')

    floor_z = float(bounds[2, 0])
    span = 0.9 * size

    plot += k3d.mesh(np.array([[centre[0] - span, centre[1] - span, floor_z],
                               [centre[0] + span, centre[1] - span, floor_z],
                               [centre[0] + span, centre[1] + span, floor_z],
                               [centre[0] - span, centre[1] + span, floor_z]], np.float32),
                     np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
                     color=0x9AA0A6,
                     roughness=0.2,
                     metalness=0.7,
                     name='floor')

    eye = centre + np.array([1.02, 0.245, 0.38]) * size
    plot.camera = [*eye, *centre, 0, 0, 1]

    plot.display()

The same model under ``advanced``, and the rest of the walkthrough, is in
``examples/stanford_dragon.ipynb``.

.. k3d_plot ::
  :filename: plots/stanford_dragon_plot.py
