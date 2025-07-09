import base64
from functools import wraps
from typing import Any, Callable
from typing import Dict as TypingDict
from typing import Generator, List, Optional

from ..objects import create_object


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
            self, generator_function: Callable[[], Generator[bytes, None, None]]
    ) -> Callable[[], None]:
        """Decorator for a generator function receiving snapshots via yield."""

        @wraps(generator_function)
        def inner() -> None:
            generator = generator_function()

            def send_new_value(change: Any) -> None:
                try:
                    generator.send(base64.b64decode(change.new))
                except StopIteration:
                    self.unobserve(send_new_value, "snapshot")

            self.observe(send_new_value, "snapshot")
            generator.send(None)

        return inner

    def get_binary_snapshot(
            self, compression_level: int = 9, voxel_chunks: Optional[List[Any]] = None
    ) -> bytes:
        import zlib

        import msgpack

        if voxel_chunks is None:
            voxel_chunks = []
        snapshot = self.get_binary_snapshot_objects(voxel_chunks)
        snapshot["plot"] = self.get_plot_params()
        data = msgpack.packb(snapshot, use_bin_type=True)
        return zlib.compress(data, compression_level)

    def load_binary_snapshot(self, data: bytes) -> tuple:
        import zlib

        import msgpack

        data = msgpack.unpackb(zlib.decompress(data))
        self.voxel_chunks = []
        if "objects" in data.keys():
            for o in data["objects"]:
                self += create_object(o)
        if "chunkList" in data.keys():
            for o in data["chunkList"]:
                self.voxel_chunks.append(create_object(o, True))
        return data, self.voxel_chunks

    def get_binary_snapshot_objects(
            self, voxel_chunks: Optional[List[Any]] = None
    ) -> TypingDict[str, List[Any]]:
        if voxel_chunks is None:
            voxel_chunks = []
        snapshot = {"objects": [], "chunkList": []}
        for name, l in [("objects", self.objects), ("chunkList", voxel_chunks)]:
            for o in l:
                snapshot[name].append(o.get_binary())
        return snapshot

    def get_snapshot(
            self,
            compression_level: int = 9,
            voxel_chunks: Optional[List[Any]] = None,
            additional_js_code: str = "",
    ) -> str:
        """Produce on the Python side a HTML document with the current plot embedded."""
        import io
        import os
        import zlib

        if voxel_chunks is None:
            voxel_chunks = []
        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/../'
        data = self.get_binary_snapshot(compression_level, voxel_chunks)
        if self.snapshot_type == "full":
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
            if self.snapshot_type == "online":
                template_file = "snapshot_online.txt"
            elif self.snapshot_type == "inline":
                template_file = "snapshot_inline.txt"
            else:
                raise Exception("Unknown snapshot_type")
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

    def get_plot_params(self) -> dict:
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
            "time_speed": self.time_speed,
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
