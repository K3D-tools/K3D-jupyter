import unittest

import numpy as np
from traitlets import TraitError

from ..factory import text
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


class TestText(unittest.TestCase):
    def test_position_accepts_numpy(self):
        text_ = text("test", [0, 0, 0])
        text_.position = np.arange(3)


class TestSTL(unittest.TestCase):
    def test_creation(self):
        from ..objects import STL

        STL(
            text="""
solid
    facet normal 0 0 0
        outer loop
            vertex -1.000000 1.000000 -1.000000
            vertex -1.000000 -1.000000 -1.000000
            vertex -1.000000 -1.000000 1.000000
        endloop
    endfacet
endsolid
        """.strip()
        )

    def test_creation_named(self):
        from ..objects import STL

        STL(
            text="""
solid named_solid
    facet normal 0 0 0
        outer loop
            vertex -1.000000 1.000000 -1.000000
            vertex -1.000000 -1.000000 -1.000000
            vertex -1.000000 -1.000000 1.000000
        endloop
    endfacet
endsolid named_solid
        """.strip()
        )

    def test_invalid(self):
        from ..objects import STL

        s = STL(
            text="""
solid
    facet normal 0 0 0
        outer loop
            vertex -1.000000 1.000000 -1.000000
            vertex -1.000000 -1.000000 -1.000000
            vertex -1.000000 -1.000000 1.000000
        endloop
    endfacet
endsolid
        """.strip()
        )

        def assign_bad():
            # missing endsolid, gibberish after facet normal:
            s.text = """solid
    facet normal 0 0 0 bebebe
        outer loop
            vertex -1.000000 1.000000 -1.000000
            vertex -1.000000 -1.000000 -1.000000
            vertex -1.000000 -1.000000 1.000000
        endloop
    endfacet
        """

        self.assertRaises(TraitError, assign_bad)

    def test_ascii_bounding_box(self):
        import warnings

        import k3d

        obj = k3d.stl(
            """
solid
    facet normal 0 0 0
        outer loop
            vertex 10 20 30
            vertex 11 20 30
            vertex 10 21 30
        endloop
    endfacet
endsolid
            """.strip()
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bbox = obj.get_bounding_box()

        self.assertFalse([w for w in caught if "bounding box" in str(w.message)])
        np.testing.assert_allclose(bbox, [10, 11, 20, 21, 30, 30], atol=1e-5)

    def test_binary_bounding_box(self):
        import struct

        import k3d

        from ..validation.stl import vertices_from_binary

        tri = np.array([[10.0, 20.0, 30.0], [11.0, 20.0, 30.0], [10.0, 21.0, 30.0]], dtype=np.float32)
        buf = bytearray(80 + 4 + 50)
        struct.pack_into("<I", buf, 80, 1)
        struct.pack_into("<12fH", buf, 84, 0, 0, 0, *tri.ravel(), 0)
        raw = bytes(buf)

        obj = k3d.stl(raw)
        np.testing.assert_allclose(obj.get_bounding_box(), [10, 11, 20, 21, 30, 30], atol=1e-5)

        verts = vertices_from_binary(np.frombuffer(raw, dtype=np.uint8))
        np.testing.assert_allclose(verts, tri)

    def test_stl_bounding_box_feeds_auto_grid(self):
        import k3d

        plot = k3d.plot()
        plot += k3d.stl(
            """
solid
    facet normal 0 0 0
        outer loop
            vertex 10 20 30
            vertex 11 20 30
            vertex 10 21 30
        endloop
    endfacet
endsolid
            """.strip()
        )

        np.testing.assert_allclose(plot.get_auto_grid(), [10, 11, 20, 21, 30, 30], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
