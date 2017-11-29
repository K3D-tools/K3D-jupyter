import unittest
from ..objects import Drawable


class TestDrawable(unittest.TestCase):
    def setUp(self):
        self.obj = Drawable()

    def test_iteration_returns_object(self):
        for obj in self.obj:
            self.assertTrue(isinstance(obj, Drawable))

    def test_can_add_another_object(self):
        self.obj += Drawable()

        for obj in self.obj:
            self.assertTrue(isinstance(obj, Drawable))

    def test_can_add_many_objects(self):
        obj = Drawable()
        self.obj += obj + obj

        for obj in self.obj:
            self.assertTrue(isinstance(obj, Drawable))


if __name__ == '__main__':
    unittest.main()
