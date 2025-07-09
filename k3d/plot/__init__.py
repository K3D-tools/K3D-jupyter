from .plot_base import PlotBase
from .plot_camera import PlotCameraMixin
from .plot_display import PlotDisplayMixin
from .plot_objects import PlotObjectsMixin
from .plot_serialization import PlotSerializationMixin
from .plot_snapshot import PlotSnapshotMixin


class Plot(
    PlotBase,
    PlotObjectsMixin,
    PlotDisplayMixin,
    PlotCameraMixin,
    PlotSnapshotMixin,
    PlotSerializationMixin,
):
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
            Multiplier to screenshot resolution.
        voxel_paint_color: `int`.
            The (initial) integer value to be inserted when editing voxels.
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
        time: `float`.
            Time value (used in TimeSeries)
        time_speed: `float`.
            Time speed (used in TimeSeries)
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

    pass

__all__ = ["Plot"]
