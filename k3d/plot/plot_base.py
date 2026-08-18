import ipywidgets as widgets
from traitlets import Any as TraitAny
from traitlets import Bool, Dict, Float, Int, List, TraitError, Unicode, validate
from typing import Any
from typing import Dict as TypingDict
from typing import List as TypingList
from typing import Optional

from .._version import __version__ as version
from .._widget import K3DAnyWidget
from ..environments import load as load_environment
from ..helpers import environment_from_json, environment_to_json, json_to_array
from ..objects import Drawable, ListOrArray, TimeSeries


class PlotBase(K3DAnyWidget):
    _kind = Unicode("plot").tag(sync=True)
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
    time_interpolation = Bool(True).tag(sync=True)
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
    renderer = Unicode(default_value="simple").tag(sync=True)
    environment = TraitAny(default_value="neutral").tag(
        sync=True, to_json=environment_to_json, from_json=environment_from_json
    )

    @validate("environment")
    def _resolve_environment(self, proposal):
        value = proposal["value"]
        # the resolved array loses the catalog name - remembered here so the wire
        # dict can carry it and the GUI shows the name instead of 'custom'
        self._environment_catalog_name = None
        # a snapshot round-trip carries the wire dict
        if isinstance(value, dict):
            self._environment_catalog_name = value.get("name")
            return json_to_array(value)
        # catalog names resolve to their arrays; procedural preset names pass through to JS
        if isinstance(value, str):
            catalog = load_environment(value)
            if catalog is not None:
                self._environment_catalog_name = value
                return catalog
        return value
    environment_rotation = Float(default_value=0.0).tag(sync=True)
    tone_mapping = Unicode(default_value="none").tag(sync=True)
    ao_radius = Float(default_value=0.07).tag(sync=True)
    ao_strength = Float(default_value=1.8).tag(sync=True)

    @validate("ao_radius")
    def _validate_ao_radius(self, proposal):
        value = float(proposal["value"])
        if not 0.0 < value <= 1.0:
            raise TraitError(
                "ao_radius is a fraction of the scene diagonal and must be in (0, 1], "
                "got %s" % value
            )
        return value

    @validate("ao_strength")
    def _validate_ao_strength(self, proposal):
        value = float(proposal["value"])
        if not 0.0 <= value <= 10.0:
            raise TraitError("ao_strength must be in [0, 10], got %s" % value)
        return value
    cinematic_samples = Int(default_value=64).tag(sync=True)
    cinematic_bounces = Int(default_value=6).tag(sync=True)

    @validate("cinematic_samples")
    def _validate_cinematic_samples(self, proposal):
        value = int(proposal["value"])
        if not 1 <= value <= 4096:
            raise TraitError("cinematic_samples must be in [1, 4096], got %s" % value)
        return value

    @validate("cinematic_bounces")
    def _validate_cinematic_bounces(self, proposal):
        value = int(proposal["value"])
        if not 1 <= value <= 32:
            raise TraitError("cinematic_bounces must be in [1, 32], got %s" % value)
        return value
    camera_mode = Unicode().tag(sync=True)
    additional_js_code = Unicode().tag(sync=True)
    manipulate_mode = Unicode().tag(sync=True)
    hidden_object_ids = List(default_value=[]).tag(sync=True)
    custom_data = Dict(default_value=None, allow_none=True).tag(sync=True)

    objects: TypingList[Drawable] = []

    def _handle_custom_msg(self, content, buffers):
        # the anywidget module lives under a blob: URL, so the HTML-snapshot button
        # cannot locate standalone.js by script path - the kernel serves it instead
        if content.get("msg_type") == "fetch_snapshot_source":
            import zlib
            from pathlib import Path

            source = (Path(__file__).parent.parent / "static" / "standalone.js").read_bytes()
            self.send({"msg_type": "snapshot_source"}, buffers=[zlib.compress(source, 9)])
        elif content.get("msg_type") == "fetch_objects":
            self._relay_send_state(content.get("ids", []))
        elif content.get("msg_type") == "object_change":
            self._relay_apply_change(buffers)

    # Colab-style frontends materialise widget models lazily, per output frame:
    # the plot model exists there, the object models never do (object_ids are
    # plain integers, not model references). The plot comm relays their state
    # instead, in the .k3d binary encoding (zlib over msgpack).

    def _relay_send_state(self, ids):
        import zlib

        import msgpack

        wanted = set(ids)
        state = {
            "objects": [o.get_binary() for o in self.objects if o.id in wanted],
            "chunkList": [
                c.get_binary() for c in getattr(self, "voxel_chunks", [])
            ],
        }

        self._relay_wire_observers()
        self.send(
            {"msg_type": "objects_state"},
            buffers=[zlib.compress(msgpack.packb(state, use_bin_type=True), 1)],
        )

    def _relay_wire_observers(self):
        if not hasattr(self, "_relay_observed"):
            self._relay_observed = set()
            self.observe(lambda change: self._relay_wire_observers(), "object_ids")

        for o in self.objects:
            if o.id not in self._relay_observed:
                self._relay_observed.add(o.id)
                o.observe(self._relay_forward)

    def _relay_forward(self, change):
        import zlib

        import msgpack

        from ..helpers import to_json

        obj = change.owner

        if change.name not in obj._synced_props or change.name in ("id", "type"):
            return

        patch = {
            "id": obj.id,
            "key": change.name,
            "value": to_json(change.name, change.new, obj, obj["compression_level"]),
        }
        self.send(
            {"msg_type": "object_patch"},
            buffers=[zlib.compress(msgpack.packb(patch, use_bin_type=True), 1)],
        )

    def _relay_apply_change(self, buffers):
        import zlib

        import msgpack

        from ..helpers import from_json

        patch = msgpack.unpackb(zlib.decompress(buffers[0]), strict_map_key=False)
        obj = next((o for o in self.objects if o.id == patch["id"]), None)

        if obj is not None and patch["key"] in obj._synced_props:
            setattr(obj, patch["key"], from_json(patch["value"]))

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
            time_interpolation: bool = True,
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
            camera_fov: float = 60.0,  # matches the k3d.plot() factory default
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
            renderer: str = "simple",
            environment: str = "neutral",
            environment_rotation: float = 0.0,
            tone_mapping: str = "none",
            ao_radius: float = 0.07,
            ao_strength: float = 1.8,
            cinematic_samples: int = 64,
            cinematic_bounces: int = 6,
            additional_js_code: str = '',
            *args: Any,
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

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
        self.time_interpolation = time_interpolation
        self.menu_visibility = menu_visibility
        self.colorbar_object_id = colorbar_object_id
        self.slice_viewer_object_id = slice_viewer_object_id
        self.slice_viewer_mask_object_ids = slice_viewer_mask_object_ids
        self.slice_viewer_direction = slice_viewer_direction
        self.rendering_steps = rendering_steps
        self.camera_no_rotate = camera_no_rotate
        self.camera_no_zoom = camera_no_zoom
        self.camera_no_pan = camera_no_pan

        self.on_msg(self._handle_custom_msg)
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
        if "camera" not in kwargs:
            self.camera = []
        self.depth_peels = depth_peels
        self.renderer = renderer
        self.environment = environment
        self.environment_rotation = environment_rotation
        self.tone_mapping = tone_mapping
        self.ao_radius = ao_radius
        self.ao_strength = ao_strength
        self.cinematic_samples = cinematic_samples
        self.cinematic_bounces = cinematic_bounces
        self.custom_data = custom_data
        self.additional_js_code = additional_js_code

        self.object_ids = []
        self.objects = []
        if "hidden_object_ids" not in kwargs:
            self.hidden_object_ids = []

        self.outputs: TypingList[widgets.Output] = []
        # Populated by load_binary_snapshot. Initialised here so it can be read before any
        # load and so get_binary_snapshot can default to it.
        self.voxel_chunks: TypingList[Any] = []
