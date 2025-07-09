import math
import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

import k3d
from .plot_compare import compare, prepare


def test_texture():
    prepare()

    # Y-Z flip
    model_matrix = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]

    with open("./test/assets/texture.png", "br") as file:
        texture = k3d.texture(
            file.read(),
            "png",
            rotation=[math.radians(90), 1, 0, 0],
            model_matrix=model_matrix,
            name="Photo",
        )

    pytest.plot += texture

    compare("texture")

    with open("./test/assets/mandelbrot.jpg", "br") as file:
        texture.binary = file.read()
    texture.name = "Fractal"

    compare("texture_dynamic_change")


def test_texture_attribute():
    prepare()

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName("./test/assets/volume.vti")
    reader.Update()
    vti = reader.GetOutput()

    x, y, z = vti.GetDimensions()
    volume_data = (
        numpy_support.vtk_to_numpy(vti.GetPointData().GetArray(0))
        .reshape(-1, y, x)
        .astype(np.float32)
    )

    texture = k3d.texture(
        attribute=volume_data[64],
        transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]),
        color_map=k3d.basic_color_maps.Jet,
        opacity_function=[-1, 1, 1, 1],
        name="Slice",
    )

    pytest.plot += texture

    compare("texture_attribute")

    texture.interpolation = False

    compare("texture_attribute_dynamic_interpolation")
