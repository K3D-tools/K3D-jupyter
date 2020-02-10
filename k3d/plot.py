from __future__ import print_function

import base64
import ipywidgets as widgets
from IPython.display import display
from functools import wraps
from traitlets import Unicode, Bool, Int, List, Float

from ._version import __version__ as version
from .objects import Drawable, ListOrArray


class Plot(widgets.DOMWidget):
    """
    Main K3D widget.

    Attributes:
        antialias: `int`:
            Enable antialiasing in WebGL renderer, changes have no effect after displaying.
        height: `int`:
            Height of the Widget in pixels, changes have no effect after displaying.
        background_color: `int`.
            Packed RGB color of the plot background (0xff0000 is red, 0xff is blue), -1 is for transparent.
        camera_auto_fit: `bool`.
            Enable automatic camera setting after adding, removing or changing a plot object.
        grid_auto_fit: `bool`.
            Enable automatic adjustment of the plot grid to contained objects.
        grid_visible: `bool`.
            Enable or disable grid.
        screenshot_scale: `Float`.
            Multipiler to screenshot resolution.
        voxel_paint_color: `int`.
            The (initial) int value to be inserted when editing voxels.
        lighting: `Float`.
            Lighting factor.
        grid: `array_like`.
            6-element tuple specifying the bounds of the plot grid (x0, y0, z0, x1, y1, z1).
        camera: `array_like`.
            9-element list or array specifying camera position.
        camera_no_rotate: `Bool`.
            Lock for camera rotation.
        camera_no_zoom: `Bool`.
            Lock for camera zoom.
        camera_no_pan: `Bool`.
            Lock for camera pan.
        camera_fov: `Float`.
            Camera Field of View.
        axes: `list`.
            Axes labels for plot.
        time: `list`.
            Time value (used in TimeSeries)
        name: `string`.
            Name of the plot. Used to filenames of snapshot/screenshot etc.
        mode: `str`.
            Mode of K3D viewer.

            Legal values are:

            :`view`: No interaction with objects,

            :`add`: On voxels objects adding mode,

            :`change`: On voxels objects edit mode,

            :`callback`: Handling click_callback and hover_callback on some type of objects.
        auto_rendering: `Bool`.
            State of auto rendering.
        fps: `Float`.
            Fps of animation.
        objects: `list`.
            List of `k3d.objects.Drawable` currently included in the plot, not to be changed directly.
    """

    _view_name = Unicode('PlotView').tag(sync=True)
    _model_name = Unicode('PlotModel').tag(sync=True)
    _view_module = Unicode('k3d').tag(sync=True)
    _model_module = Unicode('k3d').tag(sync=True)

    _view_module_version = Unicode(version).tag(sync=True)
    _model_module_version = Unicode(version).tag(sync=True)
    _backend_version = Unicode(version).tag(sync=True)

    # readonly (specified at creation)
    antialias = Int(min=0, max=5).tag(sync=True)
    height = Int().tag(sync=True)

    # readonly (not to be modified directly)
    object_ids = List().tag(sync=True)

    # read-write
    camera_auto_fit = Bool(True).tag(sync=True)
    auto_rendering = Bool(True).tag(sync=True)
    lighting = Float().tag(sync=True)
    fps = Float().tag(sync=True)
    grid_auto_fit = Bool(True).tag(sync=True)
    grid_visible = Bool(True).tag(sync=True)
    fps_meter = Bool(True).tag(sync=True)
    menu_visibility = Bool(True).tag(sync=True)
    screenshot_scale = Float().tag(sync=True)
    time = Float().tag(sync=True)
    grid = ListOrArray((-1, -1, -1, 1, 1, 1), minlen=6, maxlen=6).tag(sync=True)
    background_color = Int().tag(sync=True)
    voxel_paint_color = Int().tag(sync=True)
    camera = ListOrArray(minlen=9, maxlen=9, empty_ok=True).tag(sync=True)
    camera_no_rotate = Bool(False).tag(sync=True)
    camera_no_zoom = Bool(False).tag(sync=True)
    camera_no_pan = Bool(False).tag(sync=True)
    clipping_planes = ListOrArray(empty_ok=True).tag(sync=True)
    colorbar_object_id = Int(-1).tag(sync=True)
    rendering_steps = Int(1).tag(sync=True)
    screenshot = Unicode().tag(sync=True)
    snapshot = Unicode().tag(sync=True)
    camera_fov = Float().tag(sync=True)
    name = Unicode(default_value=None, allow_none=True).tag(sync=True)
    axes = List(minlen=3, maxlen=3, default_value=['x', 'y', 'z']).tag(sync=True)
    axes_helper = Float().tag(sync=True)
    mode = Unicode().tag(sync=True)

    objects = []

    def __init__(self, antialias=3, background_color=0xFFFFFF, camera_auto_fit=True, grid_auto_fit=True,
                 grid_visible=True, height=512, voxel_paint_color=0, grid=(-1, -1, -1, 1, 1, 1), screenshot_scale=2.0,
                 lighting=1.5, time=0.0, fps_meter=False, menu_visibility=True, colorbar_object_id=-1,
                 rendering_steps=1, axes=['x', 'y', 'z'], camera_no_rotate=False, camera_no_zoom=False,
                 camera_no_pan=False, camera_fov=45.0, axes_helper=1.0, name=None, mode='view',
                 auto_rendering=True, fps=25.0, *args, **kwargs):
        super(Plot, self).__init__()

        self.antialias = antialias
        self.camera_auto_fit = camera_auto_fit
        self.grid_auto_fit = grid_auto_fit
        self.fps_meter = fps_meter
        self.fps = fps
        self.grid = grid
        self.grid_visible = grid_visible
        self.background_color = background_color
        self.voxel_paint_color = voxel_paint_color
        self.screenshot_scale = screenshot_scale
        self.height = height
        self.lighting = lighting
        self.time = time
        self.menu_visibility = menu_visibility
        self.colorbar_object_id = colorbar_object_id
        self.rendering_steps = rendering_steps
        self.camera_no_rotate = camera_no_rotate
        self.camera_no_zoom = camera_no_zoom
        self.camera_no_pan = camera_no_pan
        self.camera_fov = camera_fov
        self.axes = axes
        self.axes_helper = axes_helper
        self.name = name
        self.mode = mode
        self.auto_rendering = auto_rendering
        self.camera = [4.5, 4.5, 4.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

        self.object_ids = []
        self.objects = []

        self.outputs = []

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

    def render(self):
        self.send({'msg_type': 'render'})

    def start_auto_play(self):
        self.send({'msg_type': 'start_auto_play'})

    def stop_auto_play(self):
        self.send({'msg_type': 'stop_auto_play'})

    def close(self):
        for output in self.outputs:
            output.clear_output()

        self.outputs = []

    def camera_reset(self, factor=1.5):
        self.send({'msg_type': 'reset_camera', 'factor': factor})

    def fetch_screenshot(self, only_canvas=False):
        self.send({'msg_type': 'fetch_screenshot', 'only_canvas': only_canvas})

    def yield_screenshots(self, generator_function):
        """Decorator for a generator function receiving screenshots via yield."""

        @wraps(generator_function)
        def inner():
            generator = generator_function()

            def send_new_value(change):
                try:
                    generator.send(base64.b64decode(change.new))
                except StopIteration:
                    self.unobserve(send_new_value, 'screenshot')

            self.observe(send_new_value, 'screenshot')
            # start the decorated generator
            generator.send(None)

        return inner

    def fetch_snapshot(self, compression_level=9):
        self.send({'msg_type': 'fetch_snapshot', 'compression_level': compression_level})

    def yield_snapshots(self, generator_function):
        """Decorator for a generator function receiving snapshots via yield."""

        @wraps(generator_function)
        def inner():
            generator = generator_function()

            def send_new_value(change):
                try:
                    generator.send(base64.b64decode(change.new))
                except StopIteration:
                    self.unobserve(send_new_value, 'snapshot')

            self.observe(send_new_value, 'snapshot')
            # start the decorated generator
            generator.send(None)

        return inner

    def get_snapshot(self, compression_level=9):
        import os
        import io
        import msgpack
        import zlib
        import numpy as np
        from base64 import b64encode
        from .helpers import to_json

        dir_path = os.path.dirname(os.path.realpath(__file__))

        snapshot = {
            "objects": [],
            "chunkList": []
        }

        for o in self.objects:
            obj = {}
            for k, v in o.traits().items():
                if 'sync' in v.metadata:
                    if isinstance(o[k], np.ndarray):
                        obj[k] = to_json(k, o[k], o, compression_level)
                    else:
                        obj[k] = o[k]

            snapshot['objects'].append(obj)

        data = msgpack.packb(snapshot)
        data = b64encode(zlib.compress(data, compression_level))

        f = io.open(os.path.join(dir_path, 'static', 'snapshot.txt'), mode="r", encoding="utf-8")
        template = f.read()
        f.close()

        f = io.open(os.path.join(dir_path, 'static', 'standalone.js'), mode="r", encoding="utf-8")
        template = template.replace('[K3D_SOURCE]',
                                    b64encode(zlib.compress(f.read().encode(), compression_level)).decode("utf-8")
                                    )
        f.close()

        f = io.open(os.path.join(dir_path, 'static', 'require.js'), mode="r", encoding="utf-8")
        template = template.replace('[REQUIRE_JS]', f.read())
        f.close()

        f = io.open(os.path.join(dir_path, 'static', 'pako_inflate.min.js'), mode="r", encoding="utf-8")
        template = template.replace('[PAKO_JS]', f.read())
        f.close()

        template = template.replace('[DATA]', data.decode("utf-8"))
        template = template.replace('[PARAMS]', "{}")
        template = template.replace('[CAMERA]', "[1,0,0,0,0,0,0,1,0]")

        return template
