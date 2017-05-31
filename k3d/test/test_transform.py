import unittest
import numpy as np
from ..transform import Transform
from ..factory import Factory


class TestTransform(unittest.TestCase):
    def test_recompute_translation(self):
        # given
        transform = Transform()
        # when
        transform.translation = [1., 2., 3.]
        # then
        self.assertTrue((transform.model_matrix == np.array([
            [1., 0., 0., 1.],
            [0., 1., 0., 2.],
            [0., 0., 1., 3.],
            [0., 0., 0., 1.]
        ])).all())

    def test_recompute_scaling(self):
        # given
        transform = Transform()
        # when
        transform.scaling = [1., 2., 3.]
        # then
        self.assertTrue((transform.model_matrix == np.array([
            [1., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 3., 0.],
            [0., 0., 0., 1.]
        ])).all())

    def test_drawable_notification(self):
        # given
        transform = Transform()
        points = Factory.points([0, 0, 1])
        transform._add_drawable(points)
        # when
        transform.scaling = [1., 2., 3.]
        # then
        self.assertTrue((points.model_matrix == np.array([
            [1., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 3., 0.],
            [0., 0., 0., 1.]
        ])).all())


if __name__ == '__main__':
    unittest.main()

