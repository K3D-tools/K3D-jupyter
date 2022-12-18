import pytest

import k3d
from .plot_compare import prepare, compare


def test_texture_text():
    prepare()

    position1 = [0.5, 1.2, 0.0]
    position2 = [-0.5, -1.2, 0.0]

    pytest.plot += k3d.texture_text('K3D Jupyter', position1, color=0xff0000, font_face='Arial',
                                    font_weight=300, size=2.0)
    pytest.plot += k3d.texture_text('K3D Jupyter', position2, color=0xff00ff, font_face='Arial',
                                    font_weight=700, size=1.0)

    compare('texture_text', camera_factor=0.5)
