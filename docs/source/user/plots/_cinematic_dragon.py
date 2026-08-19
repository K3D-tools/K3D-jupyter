"""One dragon, rendered once per environment for the cinematic page.

The polydata is cached at module level: the six scripts using it share one sphinx process,
so the archive is fetched and the OBJ parsed once.
"""
import os
import zipfile

import numpy as np
import vtk

import k3d
from k3d.headless import get_headless_driver, k3d_remote
from k3d.helpers import download

# the directive caches each PNG, so only a cold build pays for these six renders
SAMPLES = 256
WIDTH = 560
HEIGHT = 360

_scan = None


def _dragon():
    global _scan

    if _scan is not None:
        return _scan

    archive = download(
        'https://casual-effects.com/g3d/data10/research/model/dragon/dragon.zip')

    if not os.path.isfile('dragon.obj'):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extract('dragon.obj')

    reader = vtk.vtkOBJReader()
    reader.SetFileName('dragon.obj')

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(reader.GetOutputPort())

    # the scan is authored Y-up, K3D is Z-up
    to_z_up = vtk.vtkTransform()
    to_z_up.RotateX(90)

    upright = vtk.vtkTransformPolyDataFilter()
    upright.SetTransform(to_z_up)
    upright.SetInputConnection(triangles.GetOutputPort())
    upright.Update()

    _scan = upright.GetOutput()

    return _scan


def screenshot(environment):
    dragon = _dragon()

    bounds = np.array(dragon.GetBounds()).reshape(3, 2)
    centre = bounds.mean(axis=1)
    size = float((bounds[:, 1] - bounds[:, 0]).max())

    plot = k3d.plot(renderer='cinematic',
                    environment=environment,
                    tone_mapping='aces',
                    grid_visible=False,
                    camera_auto_fit=False,
                    # flat backdrop, not the environment map: the comparison is about light
                    background_color=0x2A2C30,
                    screenshot_scale=1,
                    axes_helper=0,
                    cinematic_samples=SAMPLES,
                    cinematic_bounces=6)

    plot += k3d.vtk_poly_data(dragon,
                              color=0xC8A15A,
                              flat_shading=False,
                              roughness=0.2,
                              metalness=0.7,
                              compression_level=5)

    # the polished floor is what makes the environment visible on the model
    floor_z = float(bounds[2, 0])
    span = 0.9 * size

    plot += k3d.mesh(np.array([[centre[0] - span, centre[1] - span, floor_z],
                               [centre[0] + span, centre[1] - span, floor_z],
                               [centre[0] + span, centre[1] + span, floor_z],
                               [centre[0] - span, centre[1] + span, floor_z]], np.float32),
                     np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
                     color=0x9AA0A6,
                     roughness=0.3,
                     metalness=0.5)

    plot.camera = [*(centre + np.array([1.02, 0.245, 0.30]) * size), *centre, 0, 0, 1]

    headless = k3d_remote(plot, get_headless_driver(), width=WIDTH, height=HEIGHT)
    headless.sync(hold_until_refreshed=True)

    png = headless.get_screenshot()
    headless.close()

    return png
