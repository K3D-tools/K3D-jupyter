import numpy as np
import unittest

from ..colormaps.basic_color_maps import Binary, Rainbow
from ..helpers import map_colors


class TestPythonColorMapping(unittest.TestCase):
    # Rainbow: for 0 - blue, for 1 - red

    def test_zeros(self):
        # given
        attribute = np.zeros(5)
        # when
        colors = map_colors(attribute, Rainbow)
        # then
        self.assertTrue((colors == np.ones(5, dtype=np.int32) * 0xFF).all())

    def test_ones(self):
        # given
        attribute = np.ones(5)
        # when
        # color range needed here, if uniform, range inferred as [1., 2.], which also gives all blue
        colors = map_colors(attribute, Rainbow, color_range=(0.0, 1.0))
        # then
        self.assertTrue((colors == np.ones(5, dtype=np.int32) * 0xFF0000).all())

    def test_gradient(self):
        # given
        attribute = np.array([0, 0.5, 1])
        # when
        colors = map_colors(attribute, Binary)
        # then
        self.assertTrue((colors == [0xFFFFFF, 0x7F7F7F, 0]).all())


if __name__ == "__main__":
    unittest.main()
