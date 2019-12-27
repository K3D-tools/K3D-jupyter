import unittest

from traitlets import TraitError
import numpy as np

from ..objects import Drawable
from ..k3d import text


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


class TestText(unittest.TestCase):
    def test_position_accepts_numpy(self):
        text_ = text('test', [0, 0, 0])
        text_.position = np.arange(3)


class TestSTL(unittest.TestCase):
    def test_creation(self):
        from ..objects import STL
        s = STL(text='''
solid
    facet normal 0 0 0
        outer loop
            vertex -1.000000 1.000000 -1.000000
            vertex -1.000000 -1.000000 -1.000000
            vertex -1.000000 -1.000000 1.000000
        endloop
    endfacet
endsolid
        '''.strip())

    def test_creation_named(self):
        from ..objects import STL
        STL(text='''
solid named_solid
    facet normal 0 0 0
        outer loop
            vertex -1.000000 1.000000 -1.000000
            vertex -1.000000 -1.000000 -1.000000
            vertex -1.000000 -1.000000 1.000000
        endloop
    endfacet
endsolid named_solid
        '''.strip())

    def test_invalid(self):
        from ..objects import STL
        s = STL(text='''
solid
    facet normal 0 0 0
        outer loop
            vertex -1.000000 1.000000 -1.000000
            vertex -1.000000 -1.000000 -1.000000
            vertex -1.000000 -1.000000 1.000000
        endloop
    endfacet
endsolid
        '''.strip())

        def assign_bad():
            # missing endsolid, gibberish after facet normal:
            s.text = '''solid
    facet normal 0 0 0 bebebe
        outer loop
            vertex -1.000000 1.000000 -1.000000
            vertex -1.000000 -1.000000 -1.000000
            vertex -1.000000 -1.000000 1.000000
        endloop
    endfacet
        '''

        self.assertRaises(TraitError, assign_bad)

if __name__ == '__main__':
    unittest.main()
