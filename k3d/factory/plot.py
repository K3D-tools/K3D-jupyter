"""Factory functions for creating Plot widgets."""

import numpy as np
from typing import Any
from typing import Dict as TypingDict
from typing import List as TypingList
from typing import Optional, Tuple, Union

from ..plot import Plot

# Type aliases for better readability
ArrayLike = Union[TypingList, np.ndarray, Tuple]


def plot(
        height: int = 512,
        antialias: int = 3,
        logarithmic_depth_buffer: bool = True,
        background_color: int = 0xFFFFFF,
        camera_auto_fit: bool = True,
        grid_auto_fit: bool = True,
        grid_visible: bool = True,
        screenshot_scale: float = 2.0,
        grid: Tuple[float, float, float, float, float, float] = (-1, -1, -1, 1, 1, 1),
        grid_color: int = 0xE6E6E6,
        label_color: int = 0x444444,
        lighting: float = 1.5,
        menu_visibility: bool = True,
        voxel_paint_color: int = 0,
        colorbar_object_id: int = -1,
        camera_fov: float = 60.0,
        time: float = 0.0,
        depth_peels: int = 0,
        axes: TypingList[str] = None,
        axes_helper: float = 1.0,
        axes_helper_colors: TypingList[int] = None,
        camera_mode: str = "trackball",
        snapshot_type: str = "full",
        auto_rendering: bool = True,
        camera_no_zoom: bool = False,
        camera_no_rotate: bool = False,
        camera_no_pan: bool = False,
        camera_rotate_speed: float = 1.0,
        camera_zoom_speed: float = 1.2,
        camera_pan_speed: float = 0.3,
        camera_damping_factor: float = 0.0,
        camera_up_axis: str = "none",
        fps: float = 25.0,
        minimum_fps: float = -1,
        fps_meter: bool = False,
        name: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
) -> Plot:
    if axes is None:
        axes = ["x", "y", "z"]
    if axes_helper_colors is None:
        axes_helper_colors = [0xFF0000, 0x00FF00, 0x0000FF]

    return Plot(
        antialias=antialias,
        logarithmic_depth_buffer=logarithmic_depth_buffer,
        background_color=background_color,
        lighting=lighting,
        time=time,
        colorbar_object_id=colorbar_object_id,
        camera_auto_fit=camera_auto_fit,
        grid_auto_fit=grid_auto_fit,
        grid_visible=grid_visible,
        grid_color=grid_color,
        label_color=label_color,
        height=height,
        menu_visibility=menu_visibility,
        voxel_paint_color=voxel_paint_color,
        grid=grid,
        depth_peels=depth_peels,
        axes=axes,
        axes_helper=axes_helper,
        axes_helper_colors=axes_helper_colors,
        screenshot_scale=screenshot_scale,
        camera_fov=camera_fov,
        name=name,
        camera_mode=camera_mode,
        snapshot_type=snapshot_type,
        camera_no_zoom=camera_no_zoom,
        camera_no_rotate=camera_no_rotate,
        camera_no_pan=camera_no_pan,
        camera_rotate_speed=camera_rotate_speed,
        camera_zoom_speed=camera_zoom_speed,
        camera_damping_factor=camera_damping_factor,
        camera_pan_speed=camera_pan_speed,
        camera_up_axis=camera_up_axis,
        auto_rendering=auto_rendering,
        fps=fps,
        minimum_fps=minimum_fps,
        fps_meter=fps_meter,
        custom_data=custom_data,
    )
