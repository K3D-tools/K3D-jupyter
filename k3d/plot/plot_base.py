import ipywidgets as widgets
from traitlets import Bool, Dict, Float, Int, List, Unicode
from typing import Any
from typing import Dict as TypingDict
from typing import List as TypingList
from typing import Optional

from .._version import __version__ as version
from ..objects import Drawable, ListOrArray, TimeSeries


class PlotBase(widgets.DOMWidget):
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
    object_ids = List(default_value=[]).tag(sync=True)

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
    time_speed = Float().tag(sync=True)
    grid = ListOrArray((-1, -1, -1, 1, 1, 1), minlen=6, maxlen=6).tag(sync=True)
    grid_color = Int().tag(sync=True)
    label_color = Int().tag(sync=True)
    background_color = Int().tag(sync=True)
    voxel_paint_color = Int().tag(sync=True)
    camera = ListOrArray(minlen=9, maxlen=9, empty_ok=True).tag(sync=True)
    camera_animation = TimeSeries(List(default_value=[])).tag(sync=True)
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
    axes_helper_colors = List(
        minlen=3, maxlen=3, default_value=[0xFF0000, 0x00FF00, 0x0000FF]
    ).tag(sync=True)
    mode = Unicode().tag(sync=True)
    depth_peels = Int().tag(sync=True)
    camera_mode = Unicode().tag(sync=True)
    manipulate_mode = Unicode().tag(sync=True)
    hidden_object_ids = List(default_value=[]).tag(sync=True)
    custom_data = Dict(default_value=None, allow_none=True).tag(sync=True)

    objects: TypingList[Drawable] = []

    def __init__(
            self,
            antialias: int = 3,
            logarithmic_depth_buffer: bool = True,
            background_color: int = 0xFFFFFF,
            camera_auto_fit: bool = True,
            grid_auto_fit: bool = True,
            grid_visible: bool = True,
            height: int = 512,
            voxel_paint_color: int = 0,
            grid: tuple = (-1, -1, -1, 1, 1, 1),
            screenshot_scale: float = 2.0,
            lighting: float = 1.5,
            time: float = 0.0,
            time_speed: float = 1.0,
            fps_meter: bool = False,
            menu_visibility: bool = True,
            colorbar_object_id: int = -1,
            rendering_steps: int = 1,
            axes: TypingList[str] = None,
            camera_no_rotate: bool = False,
            camera_no_zoom: bool = False,
            camera_rotate_speed: float = 1.0,
            camera_zoom_speed: float = 1.2,
            camera_pan_speed: float = 0.3,
            camera_up_axis: str = "none",
            snapshot_type: str = "full",
            camera_no_pan: bool = False,
            camera_fov: float = 45.0,
            camera_damping_factor: float = 0.0,
            axes_helper: float = 1.0,
            axes_helper_colors: TypingList[int] = None,
            name: Optional[str] = None,
            mode: str = "view",
            camera_mode: str = "trackball",
            manipulate_mode: str = "translate",
            auto_rendering: bool = True,
            fps: float = 25.0,
            minimum_fps: float = -1,
            grid_color: int = 0xE6E6E6,
            label_color: int = 0x444444,
            custom_data: Optional[TypingDict[str, Any]] = None,
            slice_viewer_object_id: int = -1,
            slice_viewer_mask_object_ids: TypingList[int] = None,
            slice_viewer_direction: str = "z",
            depth_peels: int = 0,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        super().__init__()

        if axes is None:
            axes = ["x", "y", "z"]
        if axes_helper_colors is None:
            axes_helper_colors = [0xFF0000, 0x00FF00, 0x0000FF]
        if slice_viewer_mask_object_ids is None:
            slice_viewer_mask_object_ids = []

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
        self.time_speed = time_speed
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

        self.outputs: TypingList[widgets.Output] = []
