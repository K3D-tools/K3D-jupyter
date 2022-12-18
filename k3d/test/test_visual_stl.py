import numpy as np
import pytest

import k3d
from k3d.helpers import download
from .plot_compare import prepare, compare


def test_stl():
    prepare()

    filename = download(
        'https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl')

    with open(filename, 'rb') as f:
        mesh = k3d.stl(f.read(), color=0x222222, flat_shading=True,
                       transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))

    pytest.plot += mesh

    compare('stl')


def test_stl_color():
    prepare()

    filename = download(
        'https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl')

    with open(filename, 'rb') as f:
        mesh = k3d.stl(f.read(), color=0xff00ff,
                       transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))

    pytest.plot += mesh

    compare('stl_color')


def test_stl_wireframe():
    prepare()

    filename = download(
        'https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl')

    with open(filename, 'rb') as f:
        mesh = k3d.stl(f.read(), wireframe=True,
                       transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))

    pytest.plot += mesh

    compare('stl_wireframe')


def test_stl_smooth():
    prepare()

    filename = download(
        'https://github.com/To-Fujita/Babylon.js_3D_Graphics/raw/master/scenes/stl/Cute%20Darth%20Vader.stl')

    with open(filename, 'rb') as f:
        mesh = k3d.stl(f.read(), flat_shading=False, color=0x222222,
                       transform=k3d.transform(rotation=[np.pi / 2, 1, 0, 0]))

    pytest.plot += mesh

    compare('stl_smooth')
