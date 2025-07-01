import numpy as np
from typing import Optional, List


class PlotCameraMixin:
    def camera_reset(self, factor: float = 1.5) -> None:
        """Trigger auto-adjustment of camera.

        Useful when self.camera_auto_fit == False."""
        self.send({"msg_type": "reset_camera", "factor": factor})

    def get_auto_grid(self) -> np.ndarray:
        if len(self.objects) == 0:
            return np.array([self.grid[0], self.grid[3],
                             self.grid[1], self.grid[4],
                             self.grid[2], self.grid[5]])

        d = np.stack([o.get_bounding_box() for o in self.objects])

        return np.dstack(
            [np.nanmin(d[:, 0::2], axis=0), np.nanmax(d[:, 1::2], axis=0)]
        ).flatten()

    def get_auto_camera(self, factor: float = 1.5, yaw: float = 25, pitch: float = 15,
                        bounds: Optional[np.ndarray] = None) -> List[float]:
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
