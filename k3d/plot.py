from __future__ import print_function

import base64
import ipywidgets as widgets
import numpy as np
from IPython.display import display
from functools import wraps
from traitlets import Unicode, Bool, Int, List, Float, Dict

from ._version import __version__ as version
from .objects import (ListOrArray, Drawable, TimeSeries, create_object)


class Plot(widgets.DOMWidget):
    """
    Main K3D widget.

    Attributes:
        antialias: `int`:
            Enable antialiasing in WebGL renderer, changes have no effect after displaying.
        logarithmic_depth_buffer: `bool`.
            Enables logarithmic_depth_buffer in WebGL renderer.
        height: `int`:
            Height of the Widget in pixels, changes have no effect after displaying.
        background_color: `int`.
            Packed RGB color of the plot background (0xff0000 is red, 0xff is blue), -1 is for transparent.
        camera_auto_fit: `bool`.
            Enable automatic camera setting after adding, removing or changing a plot object.
        grid_auto_fit: `bool`.
            Enable automatic adjustment of the plot grid to contained objects.
        grid_color: `int`.
            Packed RGB color of the plot grids (0xff0000 is red, 0xff is blue).
        grid_visible: `bool`.
            Enable or disable grid.
        screenshot_scale: `Float`.
            Multipiler to screenshot resolution.
        voxel_paint_color: `int`.
            The (initial) int value to be inserted when editing voxels.
        label_color: `int`.
            Packed RGB color of the labels (0xff0000 is red, 0xff is blue).
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
        camera_rotate_speed: `Float`.
            Speed of camera rotation.
        camera_zoom_speed: `Float`.
            Speed of camera zoom.
        camera_pan_speed: `Float`.
            Speed of camera pan.
        camera_fov: `Float`.
            Camera Field of View.
        camera_damping_factor: `Float`.
            Defines the intensity of damping. Default is 0 (disabled).
        camera_up_axis: `str`.
            Fixed up axis for camera.

            Legal values are:

            :`x`: x axis,

            :`y`: y axis,

            :`z`: z axis,

            :`none`: Handling click_callback and hover_callback on some type of objects.
        snapshot_type: `string`.
            Can be 'full', 'online' or 'inline'.
        axes: `list`.
            Axes labels for plot.
        axes_helper: `Float`.
            Axes helper size.
        axes_helper_colors: `List`.
            List of triple packed RGB color of the axes helper (0xff0000 is red, 0xff is blue).
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

            :`callback`: Handling click_callback and hover_callback on some type of objects,

            :`manipulate`: Enable object transform widget.
        camera_mode: `str`.
            Mode of camera movement.

            Legal values are:

            :`trackball`: orbit around point with dynamic up-vector of camera,

            :`orbit`: orbit around point with fixed up-vector of camera,

            :`fly`: orbit around point with dynamic up-vector of camera, mouse wheel also moves target point.
        manipulate_mode: `str`.
            Mode of manipulate widgets.

            Legal values are:

            :`translate`: Translation widget,

            :`rotate`: Rotation widget,

            :`scale`: Scaling widget.
        depth_peels: `int`.
            Set the maximum number of peels to use. Disabled if zero.
        auto_rendering: `Bool`.
            State of auto rendering.
        fps: `Float`.
            Fps of animation.
        minimum_fps: `Float`.
            If negative then disabled. Set target FPS to adaptative resolution.
        objects: `list`.
            List of `k3d.objects.Drawable` currently included in the plot, not to be changed directly.
    """

    _view_name = Unicode("PlotView").tag(sync=True)
    _model_name = Unicode("PlotModel").tag(sync=True)
    _view_module = Unicode("k3d").tag(sync=True)
    _model_module = Unicode("k3d").tag(sync=True)

    _view_module_version = Unicode(version).tag(sync=True)
    _model_module_version = Unicode(version).tag(sync=True)
    _backend_version = Unicode(version).tag(sync=True)

    # readonly (specified at creation)
    antialias = Int(min=0, max=5).tag(sync=True)
    logarithmic_depth_buffer = Bool(True).tag(sync=True)
    height = Int().tag(sync=True)

    # readonly (not to be modified directly)
    object_ids = List().tag(sync=True)

    # read-write
    camera_auto_fit = Bool(True).tag(sync=True)
    auto_rendering = Bool(True).tag(sync=True)
    lighting = Float().tag(sync=True)
    fps = Float().tag(sync=True)
    minimum_fps = Float().tag(sync=True)
    grid_auto_fit = Bool(True).tag(sync=True)
    grid_visible = Bool(True).tag(sync=True)
    fps_meter = Bool(True).tag(sync=True)
    menu_visibility = Bool(True).tag(sync=True)
    screenshot_scale = Float().tag(sync=True)
    time = Float().tag(sync=True)
    grid = ListOrArray((-1, -1, -1, 1, 1, 1), minlen=6, maxlen=6).tag(sync=True)
    grid_color = Int().tag(sync=True)
    label_color = Int().tag(sync=True)
    background_color = Int().tag(sync=True)
    voxel_paint_color = Int().tag(sync=True)
    camera = ListOrArray(minlen=9, maxlen=9, empty_ok=True).tag(sync=True)
    camera_animation = TimeSeries(List()).tag(sync=True)
    camera_no_rotate = Bool(False).tag(sync=True)
    camera_no_zoom = Bool(False).tag(sync=True)
    camera_no_pan = Bool(False).tag(sync=True)
    camera_rotate_speed = Float().tag(sync=True)
    camera_zoom_speed = Float().tag(sync=True)
    camera_pan_speed = Float().tag(sync=True)
    camera_damping_factor = Float().tag(sync=True)
    camera_up_axis = Unicode().tag(sync=True)
    clipping_planes = ListOrArray(empty_ok=True).tag(sync=True)
    slice_viewer_mask_object_ids = ListOrArray(empty_ok=True).tag(sync=True)
    slice_viewer_object_id = Int(-1).tag(sync=True)
    slice_viewer_direction = Unicode().tag(sync=True)
    colorbar_object_id = Int(-1).tag(sync=True)
    colorbar_scientific = Bool(False).tag(sync=True)
    rendering_steps = Int(1).tag(sync=True)
    screenshot = Unicode().tag(sync=True)
    snapshot = Unicode().tag(sync=True)
    snapshot_type = Unicode().tag(sync=True)
    camera_fov = Float().tag(sync=True)
    name = Unicode(default_value=None, allow_none=True).tag(sync=True)
    axes = List(minlen=3, maxlen=3, default_value=["x", "y", "z"]).tag(sync=True)
    axes_helper = Float().tag(sync=True)
    axes_helper_colors = List(minlen=3, maxlen=3,
                              default_value=[0xff0000, 0x00ff00, 0x0000ff]).tag(sync=True)
    mode = Unicode().tag(sync=True)
    depth_peels = Int().tag(sync=True)
    camera_mode = Unicode().tag(sync=True)
    manipulate_mode = Unicode().tag(sync=True)
    hidden_object_ids = List().tag(sync=True)
    custom_data = Dict(default_value=None, allow_none=True).tag(sync=True)

    objects = []

    def __init__(
            self,
            antialias=3,
            logarithmic_depth_buffer=True,
            background_color=0xFFFFFF,
            camera_auto_fit=True,
            grid_auto_fit=True,
            grid_visible=True,
            height=512,
            voxel_paint_color=0,
            grid=(-1, -1, -1, 1, 1, 1),
            screenshot_scale=2.0,
            lighting=1.5,
            time=0.0,
            fps_meter=False,
            menu_visibility=True,
            colorbar_object_id=-1,
            rendering_steps=1,
            axes=["x", "y", "z"],
            camera_no_rotate=False,
            camera_no_zoom=False,
            camera_rotate_speed=1.0,
            camera_zoom_speed=1.2,
            camera_pan_speed=0.3,
            camera_up_axis='none',
            snapshot_type='full',
            camera_no_pan=False,
            camera_fov=45.0,
            camera_damping_factor=0.0,
            axes_helper=1.0,
            axes_helper_colors=[0xff0000, 0x00ff00, 0x0000ff],
            name=None,
            mode="view",
            camera_mode="trackball",
            manipulate_mode="translate",
            auto_rendering=True,
            fps=25.0,
            minimum_fps=-1,
            grid_color=0xE6E6E6,
            label_color=0x444444,
            custom_data=None,
            slice_viewer_object_id=-1,
            slice_viewer_mask_object_ids=[],
            slice_viewer_direction='z',
            depth_peels=0,
            *args,
            **kwargs
    ):
        super(Plot, self).__init__()

        self.antialias = antialias
        self.logarithmic_depth_buffer = logarithmic_depth_buffer
        self.camera_auto_fit = camera_auto_fit
        self.grid_auto_fit = grid_auto_fit
        self.fps_meter = fps_meter
        self.fps = fps
        self.minimum_fps = minimum_fps
        self.grid = grid
        self.grid_visible = grid_visible
        self.background_color = background_color
        self.grid_color = grid_color
        self.label_color = label_color
        self.voxel_paint_color = voxel_paint_color
        self.screenshot_scale = screenshot_scale
        self.height = height
        self.lighting = lighting
        self.time = time
        self.menu_visibility = menu_visibility
        self.colorbar_object_id = colorbar_object_id
        self.slice_viewer_object_id = slice_viewer_object_id
        self.slice_viewer_mask_object_ids = slice_viewer_mask_object_ids
        self.slice_viewer_direction = slice_viewer_direction
        self.rendering_steps = rendering_steps
        self.camera_no_rotate = camera_no_rotate
        self.camera_no_zoom = camera_no_zoom
        self.camera_no_pan = camera_no_pan
        self.camera_rotate_speed = camera_rotate_speed
        self.camera_zoom_speed = camera_zoom_speed
        self.camera_pan_speed = camera_pan_speed
        self.camera_damping_factor = camera_damping_factor
        self.camera_fov = camera_fov
        self.camera_up_axis = camera_up_axis
        self.axes = axes
        self.axes_helper = axes_helper
        self.axes_helper_colors = axes_helper_colors
        self.name = name
        self.mode = mode
        self.snapshot_type = snapshot_type
        self.camera_mode = camera_mode
        self.manipulate_mode = manipulate_mode
        self.auto_rendering = auto_rendering
        self.camera = []
        self.depth_peels = depth_peels
        self.custom_data = custom_data

        self.object_ids = []
        self.objects = []
        self.hidden_object_ids = []

        self.outputs = []

    def __iadd__(self, objs):
        """Add Drawable to plot."""
        assert isinstance(objs, Drawable)

        for obj in objs:
            if obj.id not in self.object_ids:
                self.object_ids = self.object_ids + [obj.id]
                self.objects.append(obj)

        return self

    def __isub__(self, objs):
        """Remove Drawable from plot."""
        assert isinstance(objs, Drawable)

        for obj in objs:
            self.object_ids = [id_ for id_ in self.object_ids if id_ != obj.id]
            if obj in self.objects:
                self.objects.remove(obj)

        return self

    def display(self, **kwargs):
        """Show plot inside ipywidgets.Output()."""
        output = widgets.Output()

        with output:
            display(self, **kwargs)

        self.outputs.append(output)

        display(output)

    def render(self):
        """Trigger rendering on demand.

        Useful when self.auto_rendering == False."""
        self.send({"msg_type": "render"})

    def start_auto_play(self):
        """Start animation of plot with objects using TimeSeries."""
        self.send({"msg_type": "start_auto_play"})

    def stop_auto_play(self):
        """Stop animation of plot with objects using TimeSeries."""
        self.send({"msg_type": "stop_auto_play"})

    def close(self):
        """Remove plot from all its ipywidgets.Output()-s."""
        for output in self.outputs:
            output.clear_output()

        self.outputs = []

    def camera_reset(self, factor=1.5):
        """Trigger auto-adjustment of camera.

        Useful when self.camera_auto_fit == False."""
        self.send({"msg_type": "reset_camera", "factor": factor})

    def get_auto_grid(self):
        if len(self.objects) == 0:
            return np.array([self.grid[0], self.grid[3],
                             self.grid[1], self.grid[4],
                             self.grid[2], self.grid[5]])

        d = np.stack([o.get_bounding_box() for o in self.objects])

        return np.dstack(
            [np.nanmin(d[:, 0::2], axis=0), np.nanmax(d[:, 1::2], axis=0)]
        ).flatten()

    def get_auto_camera(self, factor=1.5, yaw=25, pitch=15, bounds=None):
        """ Compute the camera vector from the specified parameters. If `bounds`
        is not provided, then the algorithm will obtain it from the available
        meshes.
        """
        if bounds is None:
            bounds = self.get_auto_grid()
        center = (bounds[::2] + bounds[1::2]) / 2.0
        radius = 0.5 * np.sum(np.abs(bounds[::2] - bounds[1::2]) ** 2) ** 0.5
        cam_distance = radius * factor / np.sin(np.deg2rad(self.camera_fov / 2.0))

        x = np.sin(np.deg2rad(pitch)) * np.cos(np.deg2rad(yaw))
        y = np.sin(np.deg2rad(pitch)) * np.sin(np.deg2rad(yaw))
        z = np.cos(np.deg2rad(pitch))

        if pitch not in [0, 180]:
            up = [0, 0, 1]
        else:
            up = [0, 1, 1]

        return [
            center[0] + x * cam_distance,
            center[1] + y * cam_distance,
            center[2] + z * cam_distance,
            center[0],
            center[1],
            center[2],
            up[0],
            up[1],
            up[2],
        ]

    def fetch_screenshot(self, only_canvas=False):
        """Request creating a PNG screenshot on the JS side and saving it in self.screenshot

        The result is a string of a PNG file in base64 encoding.
        This function requires a round-trip of websocket messages. The result will
        be available after the current cell finishes execution."""
        self.send({"msg_type": "fetch_screenshot", "only_canvas": only_canvas})

    def yield_screenshots(self, generator_function):
        """Decorator for a generator function receiving screenshots via yield."""

        @wraps(generator_function)
        def inner():
            generator = generator_function()

            def send_new_value(change):
                try:
                    generator.send(base64.b64decode(change.new))
                except StopIteration:
                    self.unobserve(send_new_value, "screenshot")

            self.observe(send_new_value, "screenshot")
            # start the decorated generator
            generator.send(None)

        return inner

    def fetch_snapshot(self, compression_level=9):
        """Request creating a HTML snapshot on the JS side and saving it in self.snapshot

        The result is a string: a HTML document with this plot embedded.
        This function requires a round-trip of websocket messages. The result will
        be available after the current cell finishes execution."""
        self.send(
            {"msg_type": "fetch_snapshot", "compression_level": compression_level}
        )

    def yield_snapshots(self, generator_function):
        """Decorator for a generator function receiving snapshots via yield."""

        @wraps(generator_function)
        def inner():
            generator = generator_function()

            def send_new_value(change):
                try:
                    generator.send(base64.b64decode(change.new))
                except StopIteration:
                    self.unobserve(send_new_value, "snapshot")

            self.observe(send_new_value, "snapshot")
            # start the decorated generator
            generator.send(None)

        return inner

    def get_binary_snapshot(self, compression_level=9, voxel_chunks=[]):
        import zlib
        import msgpack

        snapshot = self.get_binary_snapshot_objects(voxel_chunks)
        snapshot['plot'] = self.get_plot_params()

        data = msgpack.packb(snapshot, use_bin_type=True)

        return zlib.compress(data, compression_level)

    def load_binary_snapshot(self, data):
        import zlib
        import msgpack

        data = msgpack.unpackb(zlib.decompress(data))
        self.voxel_chunks = []

        if 'objects' in data.keys():
            for o in data['objects']:
                self += create_object(o)

        if 'chunkList' in data.keys():
            for o in data['chunkList']:
                self.voxel_chunks.append(create_object(o, True))

        return data, self.voxel_chunks

    def get_binary_snapshot_objects(self, voxel_chunks=[]):
        snapshot = {"objects": [], "chunkList": []}

        for name, l in [('objects', self.objects), ('chunkList', voxel_chunks)]:
            for o in l:
                snapshot[name].append(o.get_binary())

        return snapshot

    def get_plot_params(self):
        return {
            "cameraAutoFit": self.camera_auto_fit,
            "viewMode": self.mode,
            "menuVisibility": self.menu_visibility,
            "gridAutoFit": self.grid_auto_fit,
            "gridVisible": self.grid_visible,
            "grid": self.grid,
            "gridColor": self.grid_color,
            "labelColor": self.label_color,
            "antialias": self.antialias,
            "logarithmicDepthBuffer": self.logarithmic_depth_buffer,
            "screenshotScale": self.screenshot_scale,
            "clearColor": self.background_color,
            "clippingPlanes": self.clipping_planes,
            "lighting": self.lighting,
            "time": self.time,
            "fpsMeter": self.fps_meter,
            "cameraMode": self.camera_mode,
            "depthPeels": self.depth_peels,
            "colorbarObjectId": self.colorbar_object_id,
            "sliceViewerObjectId": self.slice_viewer_object_id,
            "sliceViewerMaskObjectIds": self.slice_viewer_mask_object_ids,
            "sliceViewerDirection": self.slice_viewer_direction,
            "hiddenObjectIds": self.hidden_object_ids,
            "axes": self.axes,
            "camera": self.camera,
            "cameraNoRotate": self.camera_no_rotate,
            "cameraNoZoom": self.camera_no_zoom,
            "cameraNoPan": self.camera_no_pan,
            "cameraRotateSpeed": self.camera_rotate_speed,
            "cameraZoomSpeed": self.camera_zoom_speed,
            "cameraPanSpeed": self.camera_pan_speed,
            "cameraDampingFactor": self.camera_damping_factor,
            "cameraUpAxis": self.camera_up_axis,
            "name": self.name,
            "height": self.height,
            "cameraFov": self.camera_fov,
            "axesHelper": self.axes_helper,
            "axesHelperColors": self.axes_helper_colors,
            "cameraAnimation": self.camera_animation,
            "customData": self.custom_data,
            "fps": self.fps,
            "minimumFps": self.minimum_fps,
        }

    def get_snapshot(self, compression_level=9, voxel_chunks=[], additional_js_code=""):
        """Produce on the Python side a HTML document with the current plot embedded."""
        import os
        import io
        import zlib

        dir_path = os.path.dirname(os.path.realpath(__file__))

        data = self.get_binary_snapshot(compression_level, voxel_chunks)

        if self.snapshot_type == 'full':
            f = io.open(
                os.path.join(dir_path, "static", "snapshot_standalone.txt"),
                mode="r",
                encoding="utf-8",
            )
            template = f.read()
            f.close()

            f = io.open(
                os.path.join(dir_path, "static", "standalone.js"),
                mode="r",
                encoding="utf-8",
            )
            template = template.replace(
                "[K3D_SOURCE]",
                base64.b64encode(
                    zlib.compress(f.read().encode(), compression_level)
                ).decode("utf-8"),
            )
            f.close()

            f = io.open(
                os.path.join(dir_path, "static", "require.js"),
                mode="r",
                encoding="utf-8",
            )
            template = template.replace("[REQUIRE_JS]", f.read())
            f.close()

            f = io.open(
                os.path.join(dir_path, "static", "fflate.js"),
                mode="r",
                encoding="utf-8",
            )
            template = template.replace("[FFLATE_JS]", f.read())
            f.close()
        else:
            if self.snapshot_type == 'online':
                template_file = 'snapshot_online.txt'
            elif self.snapshot_type == 'inline':
                template_file = 'snapshot_inline.txt'
            else:
                raise Exception('Unknown snapshot_type')

            f = io.open(
                os.path.join(dir_path, "static", template_file),
                mode="r",
                encoding="utf-8",
            )
            template = f.read()
            f.close()

            template = template.replace("[VERSION]", self._view_module_version)
            template = template.replace("[HEIGHT]", str(self.height))
            template = template.replace("[ID]", str(id(self)))

        template = template.replace("[DATA]", base64.b64encode(data).decode("utf-8"))
        template = template.replace("[ADDITIONAL]", additional_js_code)

        return template

    def get_static_path(self):
        import os

        dir_path = os.path.dirname(os.path.realpath(__file__))

        return os.path.join(dir_path, "static")
    
    def __getstate__(self):            
        return self.get_binary_snapshot()
    
    def __setstate__(self,data):
        self.__init__()
        self.load_binary_snapshot(data)
