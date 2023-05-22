import numpy as np
import unittest

from ..colormaps.basic_color_maps import Rainbow, Binary
from ..helpers import map_colors


class TestPythonColorMapping(unittest.TestCase):
    # Rainbow: for 0 - blue, for 1 - red

    def test_zeros(self):
        # given
        attribute = np.zeros(5)
        # when
        colors = map_colors(attribute, Rainbow)
        # then
        self.assertTrue((colors == np.ones(5, dtype=np.int32) * 0xff).all())

    def test_ones(self):
        # given
        attribute = np.ones(5)
        # when
        # color range needed here, if uniform, range inferred as [1., 2.], which also gives all blue
        colors = map_colors(attribute, Rainbow, color_range=(0., 1.))
        # then
        self.assertTrue((colors == np.ones(5, dtype=np.int32) * 0xff0000).all())

    def test_gradient(self):
        # given
        attribute = np.array([0, 0.5, 1])
        # when
        colors = map_colors(attribute, Binary)
        # then
        self.assertTrue((colors == [0xffffff, 0x7f7f7f, 0]).all())


if __name__ == '__main__':
    unittest.main()
