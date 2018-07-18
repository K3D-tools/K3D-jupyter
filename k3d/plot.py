from __future__ import print_function
import warnings
import types
import codecs

import ipywidgets as widgets
from traitlets import Unicode, Bool, Int, List, Float
from IPython.display import display

from ._frontend import EXTENSION_SPEC_VERSION
from .objects import Drawable, ListOrArray
from ._version import __version__ as BACKEND_VERSION


class Plot(widgets.DOMWidget):
    """
    Main K3D widget.

    Attributes:
        antialias: `bool`: Enable antialiasing in WebGL renderer, changes have no effect after displaying.
        height: `int`: Height of the Widget in pixels, changes have no effect after displaying.
        background_color: `int`.  Packed RGB color of the plot background (0xff0000 is red, 0xff is blue).
        camera_auto_fit: `bool`. Enable automatic camera setting after adding, removing or changing a plot object.
        grid_auto_fit: `bool`. Enable automatic adjustment of the plot grid to contained objects.
        screenshot_scale: `Float`. Multipiler to screenshot resolution.
        voxel_paint_color: `int`. The (initial) int value to be inserted when editing voxels.
        grid: `array_like`. 6-element tuple specifying the bounds of the plot grid (x0, y0, z0, x1, y1, z1).
        camera: `array_like`. 9-element list or array specifying camera position.
        objects: `list`. List of `k3d.objects.Drawable` currently included in the plot, not to be changed directly.
    """

    _view_name = Unicode('PlotView').tag(sync=True)
    _model_name = Unicode('PlotModel').tag(sync=True)
    _view_module = Unicode('k3d').tag(sync=True)
    _model_module = Unicode('k3d').tag(sync=True)

    _view_module_version = Unicode(EXTENSION_SPEC_VERSION).tag(sync=True)
    _model_module_version = Unicode(EXTENSION_SPEC_VERSION).tag(sync=True)

    _backend_version = Unicode(BACKEND_VERSION).tag(sync=True)

    # readonly (specified at creation)
    antialias = Bool().tag(sync=True)
    height = Int().tag(sync=True)

    # readonly (not to be modified directly)
    object_ids = List().tag(sync=True)

    # read-write
    camera_auto_fit = Bool(True).tag(sync=True)
    lighting = Float().tag(sync=True)
    grid_auto_fit = Bool(True).tag(sync=True)
    fps_meter = Bool(True).tag(sync=True)
    screenshot_scale = Float().tag(sync=True)
    grid = ListOrArray((-1, -1, -1, 1, 1, 1), minlen=6, maxlen=6).tag(sync=True)
    background_color = Int().tag(sync=True)
    voxel_paint_color = Int().tag(sync=True)
    camera = ListOrArray(minlen=9, maxlen=9, empty_ok=True).tag(sync=True)
    clipping_planes = ListOrArray(empty_ok=True).tag(sync=True)
    screenshot = Unicode().tag(sync=True)

    objects = []

    def __init__(self, antialias=True, background_color=0xFFFFFF, camera_auto_fit=True, grid_auto_fit=True, height=512,
                 voxel_paint_color=0, grid=(-1, -1, -1, 1, 1, 1), screenshot_scale=2.0, lighting=1.0, fps_meter=False,
                 *args, **kwargs):
        super(Plot, self).__init__()

        self.antialias = antialias
        self.camera_auto_fit = camera_auto_fit
        self.grid_auto_fit = grid_auto_fit
        self.fps_meter = fps_meter
        self.grid = grid
        self.background_color = background_color
        self.voxel_paint_color = voxel_paint_color
        self.screenshot_scale = screenshot_scale
        self.height = height
        self.lighting = lighting

        self.object_ids = []
        self.objects = []

        self.outputs = []

        self._screenshot_handler = None
        self.observe(self._screenshot_changed, names=['screenshot'])

    def __iadd__(self, objs):
        assert isinstance(objs, Drawable)

        for obj in objs:
            if obj.id not in self.object_ids:
                self.object_ids = self.object_ids + [obj.id]
                self.objects.append(obj)

        return self

    def __isub__(self, objs):
        assert isinstance(objs, Drawable)

        for obj in objs:
            self.object_ids = [id_ for id_ in self.object_ids if id_ != obj.id]
            if obj in self.objects:
                self.objects.remove(obj)

        return self

    def display(self, **kwargs):
        output = widgets.Output()

        with output:
            display(self, **kwargs)

        self.outputs.append(output)

        display(output)

    def close(self):
        for output in self.outputs:
            output.clear_output()

        self.outputs = []

    def fetch_screenshot(self, handler=None):
        self._screenshot_handler = handler
        if isinstance(self._screenshot_handler, types.GeneratorType):
            if handler is not self._screenshot_handler:
                # start (only new) generator
                next(self._screenshot_handler)
        self.send({'msg_type': 'fetch_screenshot'})

    def _screenshot_changed(self, change):
        if self._screenshot_handler is not None:
            data = codecs.decode(change['new'].encode('ascii'), 'base64')
            if isinstance(self._screenshot_handler, types.GeneratorType):
                try:
                    self._screenshot_handler.send(data)
                except StopIteration:
                    # unregister used up generator
                    self._screenshot_handler = None
            elif callable(self._screenshot_handler):
                self._screenshot_handler(data)
            else:
                raise TypeError('Screenshot handler of wrong type')
