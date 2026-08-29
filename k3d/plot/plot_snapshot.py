import base64
from functools import wraps
from typing import Any, Callable, Generator, List, Optional
from typing import Dict as TypingDict

import numpy as np

from .._version import __version__ as version
from ..helpers import environment_to_json
from ..objects import create_object

# Snapshot key -> trait name. One mapping used for both saving and restoring, so the JS-facing
# key names cannot drift apart from the Python side.
_PLOT_PARAMS = (
    ("cameraAutoFit", "camera_auto_fit"),
    ("viewMode", "mode"),
    ("menuVisibility", "menu_visibility"),
    ("gridAutoFit", "grid_auto_fit"),
    ("gridVisible", "grid_visible"),
    ("grid", "grid"),
    ("gridColor", "grid_color"),
    ("labelColor", "label_color"),
    ("antialias", "antialias"),
    ("logarithmicDepthBuffer", "logarithmic_depth_buffer"),
    ("screenshotScale", "screenshot_scale"),
    ("clearColor", "background_color"),
    ("clippingPlanes", "clipping_planes"),
    ("lighting", "lighting"),
    ("time", "time"),
    ("timeSpeed", "time_speed"),
    ("timeInterpolation", "time_interpolation"),
    ("fpsMeter", "fps_meter"),
    ("cameraMode", "camera_mode"),
    ("depthPeels", "depth_peels"),
    ("renderer", "renderer"),
    ("environment", "environment"),
    ("environmentRotation", "environment_rotation"),
    ("toneMapping", "tone_mapping"),
    ("colorbarObjectId", "colorbar_object_id"),
    ("sliceViewerObjectId", "slice_viewer_object_id"),
    ("sliceViewerMaskObjectIds", "slice_viewer_mask_object_ids"),
    ("sliceViewerDirection", "slice_viewer_direction"),
    ("hiddenObjectIds", "hidden_object_ids"),
    ("axes", "axes"),
    ("camera", "camera"),
    ("cameraNoRotate", "camera_no_rotate"),
    ("cameraNoZoom", "camera_no_zoom"),
    ("cameraNoPan", "camera_no_pan"),
    ("cameraRotateSpeed", "camera_rotate_speed"),
    ("cameraZoomSpeed", "camera_zoom_speed"),
    ("cameraPanSpeed", "camera_pan_speed"),
    ("cameraDampingFactor", "camera_damping_factor"),
    ("cameraUpAxis", "camera_up_axis"),
    ("name", "name"),
    ("height", "height"),
    ("cameraFov", "camera_fov"),
    ("axesHelper", "axes_helper"),
    ("axesHelperColors", "axes_helper_colors"),
    ("aoRadius", "ao_radius"),
    ("aoStrength", "ao_strength"),
    ("cinematicSamples", "cinematic_samples"),
    ("cinematicBounces", "cinematic_bounces"),
    ("cinematicGlossyFilter", "cinematic_glossy_filter"),
    ("cameraAnimation", "camera_animation"),
    ("customData", "custom_data"),
    ("fps", "fps"),
    ("minimumFps", "minimum_fps"),
    ("additionalJsCode", "additional_js_code"),
)


def _msgpack_safe(value: Any) -> Any:
    """Return `value` with numpy types replaced by plain Python equivalents.

    The grid/camera/clipping_planes traits are ListOrArray and accept ndarrays, and casting one
    to a list leaves numpy scalars behind, which msgpack cannot pack.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _msgpack_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_msgpack_safe(v) for v in value]
    return value


class PlotSnapshotMixin:
    def fetch_screenshot(self, only_canvas: bool = False) -> None:
        """Request creating a PNG screenshot on the JS side and saving it in self.screenshot

        The result is a string of a PNG file in base64 encoding.
        This function requires a round-trip of websocket messages. The result will
        be available after the current cell finishes execution."""
        self.send({"msg_type": "fetch_screenshot", "only_canvas": only_canvas})

    def yield_screenshots(
            self, generator_function: Callable[[], Generator[bytes, None, None]]
    ) -> Callable[[], None]:
        """Decorator for a generator function receiving screenshots via yield."""

        @wraps(generator_function)
        def inner() -> None:
            generator = generator_function()

            def send_new_value(change: Any) -> None:
                try:
                    generator.send(base64.b64decode(change.new))
                except StopIteration:
                    self.unobserve(send_new_value, "screenshot")

            self.observe(send_new_value, "screenshot")
            generator.send(None)

        return inner

    def fetch_snapshot(self, compression_level: int = 9) -> None:
        """Request creating a HTML snapshot on the JS side and saving it in self.snapshot

        The result is a string: a HTML document with this plot embedded.
        This function requires a round-trip of websocket messages. The result will
        be available after the current cell finishes execution."""
        self.send(
            {"msg_type": "fetch_snapshot", "compression_level": compression_level}
        )

    def yield_snapshots(
            self, generator_function: Callable[[], Generator[str, None, None]]
    ) -> Callable[[], None]:
        """Decorator for a generator function receiving snapshots via yield.

        The generator receives the HTML document as a `str`: unlike screenshots, the frontend
        stores this trait as raw HTML (js/src/k3d.js sets it from getHTMLSnapshot), so there is
        nothing to base64-decode.
        """

        @wraps(generator_function)
        def inner() -> None:
            generator = generator_function()

            def send_new_value(change: Any) -> None:
                try:
                    generator.send(change.new)
                except StopIteration:
                    self.unobserve(send_new_value, "snapshot")

            self.observe(send_new_value, "snapshot")
            generator.send(None)

        return inner

    def fetch_gltf(self) -> None:
        """Request exporting the scene on the JS side and saving it in self.gltf

        The result is a string of a binary glTF (.glb) file in base64 encoding.
        This function requires a round-trip of websocket messages. The result will
        be available after the current cell finishes execution.

        glTF describes triangles and PBR materials, so an object whose shape only exists while
        its shader runs cannot be handed over, and is left out rather than exported as the proxy
        geometry that shader consumes - a volume would otherwise arrive as a plain cube.

        Exported: mesh, surface, stl, marching_cubes, voxels, sparse_voxels, voxels_group,
        texture built from an image, points with shader='mesh', line and lines with
        shader='mesh' (tubes) or shader='simple' (glTF line primitives), and the arrowheads of
        vectors and vector_field.

        Left out: volume, mip, volume_slice, texture built from an attribute, points with
        shader '3d', 'dot' or 'flat', line and lines with shader='thick', text, text2d, label,
        texture_text, and the shafts of vectors and vector_field. Switching a points or lines
        object to shader='mesh' is enough to bring it into the export.

        The grid, axes, lights and color legend are not part of the model and never exported.
        """
        self.send({"msg_type": "fetch_gltf"})

    def yield_gltfs(
            self, generator_function: Callable[[], Generator[bytes, None, None]]
    ) -> Callable[[], None]:
        """Decorator for a generator function receiving glTF exports via yield.

        The generator receives the .glb file as `bytes`, ready to be written to disk.
        """

        @wraps(generator_function)
        def inner() -> None:
            generator = generator_function()

            def send_new_value(change: Any) -> None:
                try:
                    generator.send(base64.b64decode(change.new))
                except StopIteration:
                    self.unobserve(send_new_value, "gltf")

            self.observe(send_new_value, "gltf")
            generator.send(None)

        return inner

    def get_binary_snapshot(
            self, compression_level: int = 9, voxel_chunks: Optional[List[Any]] = None
    ) -> bytes:
        import zlib

        import msgpack

        if voxel_chunks is None:
            voxel_chunks = getattr(self, "voxel_chunks", [])
        snapshot = self.get_binary_snapshot_objects(voxel_chunks)
        snapshot["plot"] = self.get_plot_params()
        snapshot["version"] = version
        data = msgpack.packb(snapshot, use_bin_type=True)
        return zlib.compress(data, compression_level)

    def load_binary_snapshot(self, data: bytes) -> tuple:
        import zlib

        import msgpack

        data = msgpack.unpackb(zlib.decompress(data))
        self.voxel_chunks = []
        if "plot" in data:
            self.set_plot_params(data["plot"])
        if "objects" in data:
            for o in data["objects"]:
                self += create_object(o)
        if "chunkList" in data:
            for o in data["chunkList"]:
                self.voxel_chunks.append(create_object(o, True))
        return data, self.voxel_chunks

    def get_binary_snapshot_objects(
            self, voxel_chunks: Optional[List[Any]] = None
    ) -> TypingDict[str, List[Any]]:
        if voxel_chunks is None:
            voxel_chunks = []
        snapshot = {"objects": [], "chunkList": []}
        for name, drawables in [("objects", self.objects), ("chunkList", voxel_chunks)]:
            for o in drawables:
                snapshot[name].append(o.get_binary())
        return snapshot

    def get_snapshot(
            self,
            compression_level: int = 9,
            voxel_chunks: Optional[List[Any]] = None,
            additional_js_code: str = "",
    ) -> str:
        """Produce on the Python side a HTML document with the current plot embedded."""
        import os
        import zlib

        if voxel_chunks is None:
            voxel_chunks = getattr(self, "voxel_chunks", [])
        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/../'

        def read_static(name):
            with open(os.path.join(dir_path, "static", name), encoding="utf-8") as f:
                return f.read()

        data = self.get_binary_snapshot(compression_level, voxel_chunks)
        if self.snapshot_type == "full":
            template = read_static("snapshot_standalone.txt")
            template = template.replace(
                "[K3D_SOURCE]",
                base64.b64encode(
                    zlib.compress(read_static("standalone.js").encode(), compression_level)
                ).decode("utf-8"),
            )
            template = template.replace("[REQUIRE_JS]", read_static("require.js"))
            template = template.replace("[FFLATE_JS]", read_static("fflate.js"))
        else:
            if self.snapshot_type == "online":
                template_file = "snapshot_online.txt"
            elif self.snapshot_type == "inline":
                template_file = "snapshot_inline.txt"
            else:
                raise Exception("Unknown snapshot_type")
            template = read_static(template_file)
            template = template.replace("[VERSION]", self._view_module_version)
            template = template.replace("[HEIGHT]", str(self.height))
            template = template.replace("[ID]", str(id(self)))
        template = template.replace("[DATA]", base64.b64encode(data).decode("utf-8"))
        return template.replace("[ADDITIONAL]",
                                    self.additional_js_code + '\n' + additional_js_code)

    def get_plot_params(self) -> dict:
        """Plot settings in the wire format shared with the JS side.

        Values are normalised to plain Python so they can be msgpack-packed directly.
        """
        params = {}
        for key, trait in _PLOT_PARAMS:
            value = getattr(self, trait)
            if key == "environment" and not isinstance(value, str):
                # a user map goes as a typed array, not a nested list
                params[key] = environment_to_json(value, self)
            else:
                params[key] = _msgpack_safe(value)
        return params

    def set_plot_params(self, params: TypingDict[str, Any]) -> None:
        """Apply settings produced by get_plot_params. Unknown keys are ignored."""
        by_key = dict(_PLOT_PARAMS)
        # Snapshots written by older versions carry the snake_case spelling of timeSpeed.
        by_key.setdefault("time_speed", "time_speed")

        for key, value in params.items():
            trait = by_key.get(key)
            if trait is not None:
                setattr(self, trait, value)
