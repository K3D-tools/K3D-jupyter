"""A factory has to take every parameter the object it builds takes.

The factories are the documented entry point, so a trait they do not name is a trait the
user cannot set at creation - either loudly, with a TypeError, or silently, because the
leftover keyword lands in **kwargs and process_transform_arguments ignores what it does
not recognise.
"""

import inspect
import unittest

import numpy as np

import k3d
from k3d._widget import K3DAnyWidget
from k3d.objects import Drawable
from k3d.plot.plot_base import PlotBase

# the widget machinery's own traits (layout, tooltip, _model_name, ...) are not k3d parameters
WIDGET_TRAITS = set(K3DAnyWidget.class_traits())

# k3d traits that are not creation parameters, and why
NOT_A_PARAMETER = {
    # assigned by k3d itself
    "id",
    "type",
    # written by the browser and read back, never given at creation
    "screenshot",
    "snapshot",
    "gltf",
    "object_ids",
    # process_transform_arguments builds it out of bounds/translation/rotation/scaling
    "model_matrix",
}

# traits a particular factory derives from an argument of its own
DERIVED_BY_FACTORY = {
    "stl": {"binary", "text"},
    "vtk_poly_data": {
        "attribute",
        "colors",
        "indices",
        "normals",
        "texture",
        "texture_file_format",
        "triangles_attribute",
        "uvs",
        "vertices",
    },
}

_STL = (
    "solid a\nfacet normal 0 0 1\nouter loop\n"
    "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
    "endloop\nendfacet\nendsolid a"
)

# every public object factory, with the smallest input it accepts
FACTORIES = {
    "line": lambda **kw: k3d.line([[0, 0, 0], [1, 1, 1]], **kw),
    "lines": lambda **kw: k3d.lines([[0, 0, 0], [1, 1, 1]], [[0, 1]], **kw),
    "label": lambda **kw: k3d.label("a", **kw),
    "marching_cubes": lambda **kw: k3d.marching_cubes(
        np.zeros((4, 4, 4), np.float32), level=0.5, **kw
    ),
    "mesh": lambda **kw: k3d.mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], **kw),
    "mip": lambda **kw: k3d.mip(np.zeros((2, 2, 2), np.float32), **kw),
    "points": lambda **kw: k3d.points([[0, 0, 0]], **kw),
    "sparse_voxels": lambda **kw: k3d.sparse_voxels(
        np.zeros((0, 4), np.uint16), [2, 2, 2], **kw
    ),
    "stl": lambda **kw: k3d.stl(_STL, **kw),
    "surface": lambda **kw: k3d.surface(np.zeros((4, 4), np.float32), **kw),
    "text": lambda **kw: k3d.text("a", **kw),
    "text2d": lambda **kw: k3d.text2d("a", **kw),
    "texture": k3d.texture,  # takes no positional argument, so the factory is the builder
    "texture_text": lambda **kw: k3d.texture_text("a", **kw),
    "vector_field": lambda **kw: k3d.vector_field(np.zeros((2, 2, 2, 3), np.float32), **kw),
    "vectors": lambda **kw: k3d.vectors([0, 0, 0], [1, 1, 1], **kw),
    "volume": lambda **kw: k3d.volume(np.zeros((2, 2, 2), np.float32), **kw),
    "volume_slice": lambda **kw: k3d.volume_slice(np.zeros((2, 2, 2), np.float32), **kw),
    "voxels": lambda **kw: k3d.voxels(np.zeros((2, 2, 2), np.uint8), **kw),
    "voxels_group": lambda **kw: k3d.voxels_group([2, 2, 2], voxels=[], **kw),
}


def _parameters(factory):
    return set(inspect.signature(factory).parameters)


def _settable_traits(cls):
    traits = {name for name, t in cls.class_traits().items() if t.metadata.get("sync")}
    traits -= {name for name in traits if name.startswith("_")}
    return traits - WIDGET_TRAITS - NOT_A_PARAMETER


class TestPlotFactory(unittest.TestCase):
    def test_takes_every_plot_parameter(self):
        missing = sorted(_settable_traits(PlotBase) - _parameters(k3d.plot))

        self.assertEqual([], missing, "k3d.plot() does not take: %s" % ", ".join(missing))

    def test_mode_reaches_the_plot(self):
        self.assertEqual("callback", k3d.plot(mode="callback").mode)
        self.assertEqual("manipulate", k3d.plot(mode="manipulate").mode)

    def test_values_arrive(self):
        given = {
            "rendering_steps": 4,
            "colorbar_scientific": True,
            "camera": [1, 2, 3, 0, 0, 0, 0, 0, 1],
            "clipping_planes": [[1, 0, 0, 0]],
            "hidden_object_ids": [7],
            "slice_viewer_object_id": 3,
            "slice_viewer_direction": "x",
            "slice_viewer_mask_object_ids": [1, 2],
        }
        plot = k3d.plot(**given)

        for name, value in given.items():
            self.assertEqual(value, getattr(plot, name), name)

    def test_defaults_are_unchanged(self):
        plot, bare = k3d.plot(), PlotBase()

        for name in _settable_traits(PlotBase) & _parameters(k3d.plot):
            self.assertEqual(getattr(bare, name), getattr(plot, name), name)


class TestObjectFactories(unittest.TestCase):
    def test_take_every_object_parameter(self):
        gaps = {}

        for name, build in FACTORIES.items():
            cls = type(build())
            missing = _settable_traits(cls) - _parameters(getattr(k3d, name))
            missing -= DERIVED_BY_FACTORY.get(name, set())
            if missing:
                gaps[name] = sorted(missing)

        self.assertEqual({}, gaps)

    def test_visible_reaches_the_object(self):
        # it used to be swallowed by **kwargs, so the object came back visible anyway
        for name, build in FACTORIES.items():
            self.assertFalse(build(visible=False).visible, name)
            self.assertTrue(build().visible, name)

    def test_callbacks_reach_the_object(self):
        def callback(params):
            return params

        for name, build in FACTORIES.items():
            if "click_callback" not in _parameters(getattr(k3d, name)):
                continue
            obj = build(click_callback=callback, hover_callback=callback)
            self.assertIs(callback, obj.click_callback, name)
            self.assertIs(callback, obj.hover_callback, name)

    def test_every_factory_builds_a_drawable(self):
        for name, build in FACTORIES.items():
            self.assertIsInstance(build(), Drawable, name)


if __name__ == "__main__":
    unittest.main()
