"""Probe, not a test: does the rendered colour depend on the output resolution?

The denoiser has a fixed radius in PIXELS, so at a quarter of the width it covers sixteen times
the area of the subject. If the difference disappears with the filter off, the filter owns it.
screenshot_scale drives the output size through the same path a real screenshot uses.
"""
import os
from io import BytesIO

import numpy as np
import pytest
import vtk
from PIL import Image
from vtk.util import numpy_support

import k3d

from .plot_compare import prepare

SAMPLES = int(os.environ.get("K3D_PROBE_SAMPLES", "32"))
SCALES = [float(s) for s in os.environ.get("K3D_PROBE_SCALES", "1.0,0.25").split(",")]


def _shot(scale, denoise):
    pytest.plot.screenshot_scale = scale
    pytest.plot.cinematic_denoise = denoise
    pytest.headless.sync(hold_until_refreshed=True)

    png = pytest.headless.get_screenshot(True)

    return np.asarray(Image.open(BytesIO(png)).convert("RGB"), dtype=np.float64)


def _stats(rgb):
    """Mean channels over pixels carrying something, and how red they are."""
    mask = rgb.sum(axis=2) > 0

    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0

    r, g, b = (rgb[..., i][mask].mean() for i in range(3))

    return r, g, b, r / max((g + b) / 2.0, 1e-6)


def test_probe_resolution_colour():
    prepare()

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName("./test/assets/volume.vti")
    reader.Update()
    vti = reader.GetOutput()

    x, y, _ = vti.GetDimensions()
    volume_data = (
        numpy_support.vtk_to_numpy(vti.GetPointData().GetArray(0))
        .reshape(-1, y, x)
        .astype(np.float32)
    )

    pytest.plot += k3d.volume(volume_data, alpha_coef=200, samples=128,
                              color_map=k3d.matplotlib_color_maps.Reds_r)
    pytest.plot.renderer = "cinematic"
    pytest.plot.cinematic_seed = 1
    pytest.plot.cinematic_samples = SAMPLES

    try:
        print("\n  denoise  scale   piksele        R        G        B    czerwonosc")

        for denoise in (0.0, 2.0):
            for scale in SCALES:
                rgb = _shot(scale, denoise)
                r, g, b, redness = _stats(rgb)

                print("  %7.1f  %5.2f  %5dx%-5d %7.3f  %7.3f  %7.3f   %9.4f"
                      % (denoise, scale, rgb.shape[1], rgb.shape[0], r, g, b, redness))

        print("")
        print("  Jesli czerwonosc zmienia sie ze skala TYLKO przy denoise 2.0, wlascicielem")
        print("  jest filtr i jego staly promien w pikselach.")
    finally:
        pytest.plot.screenshot_scale = 1.0
        pytest.plot.cinematic_denoise = 0.0
        pytest.plot.cinematic_samples = 64
        pytest.plot.renderer = "simple"
        pytest.headless.sync(hold_until_refreshed=True)
