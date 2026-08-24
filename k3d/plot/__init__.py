from .plot_base import PlotBase
from .plot_camera import PlotCameraMixin
from .plot_display import PlotDisplayMixin
from .plot_objects import PlotObjectsMixin
from .plot_serialization import PlotSerializationMixin
from .plot_snapshot import PlotSnapshotMixin
from .plot_time import PlotTimeMixin


class Plot(
    PlotObjectsMixin,
    PlotDisplayMixin,
    PlotCameraMixin,
    PlotSnapshotMixin,
    PlotSerializationMixin,
    PlotTimeMixin,
    PlotBase,
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
            Lighting factor - the exposure knob. In the advanced renderer the
            environment carries the shape of the light, lighting scales its energy.
        renderer: `str`.
            Rendering pipeline of the plot.

            Legal values are:

            :`simple`: the classic rasteriser with a fixed light rig (default),

            :`advanced`: image-based lighting from the environment map, physically
             based materials and ambient occlusion,

            :`cinematic`: progressive path tracing with global illumination.
             Requires WebGL2 with renderable float textures; when the browser
             cannot run it, the switch fails with an error instead of falling
             back to another renderer.
        environment: `str` or `array_like`.
            The light environment of the advanced renderer.

            Legal values are:

            :`neutral`: procedural achromatic gradient with a soft key light (default),

            :`studio`: procedural gradient with two soft studio lights,

            :`outdoor`: procedural sky with a sun disc and ground,

            :name from k3d.environments.available(): a photographic HDRI shipped
             with the package (Poly Haven, CC0),

            :array_like: a custom (height, width, 3) float32 equirectangular
             radiance map. Every map is energy-normalised.
        environment_rotation: `float`.
            Rotation of the environment map around the scene's up axis, in radians.
        tone_mapping: `str`.
            Tone curve applied by the advanced renderer.

            Legal values are:

            :`none`: linear output (default),

            :`agx`: AgX filmic curve,

            :`aces`: ACES filmic curve.
        ao_radius: `float`.
            Occlusion radius of the advanced renderer's ambient occlusion, as a
            fraction of the scene's bounding-box diagonal, in (0, 1]. Default 0.07.
            Dense point clouds and closed interiors usually want a smaller radius.
        ao_strength: `float`.
            Exponent deepening the ambient occlusion shadows, in [0, 10].
            0 disables the darkening, default 1.8.
        cinematic_samples: `int`.
            Sample budget of the cinematic renderer, in [1, 100000]. Default 64,
            which settles in a moment; raise it for a final render. The
            interactive view accumulates one sample per animation frame up to
            this budget, then parks itself; any change to the camera, the scene
            or the lighting restarts the accumulation from zero. Screenshots
            always render the full budget.
        cinematic_bounces: `int`.
            Light bounce count of the cinematic renderer's path tracing,
            in [1, 32]. Default 6.
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
            With peeling on, volumes compose correctly with intersecting
            meshes (the ray march is split at the layer depths); use
            depth_peels >= 3, below that the effect is unpredictable.
        render_on_change: `Bool`.
            Whether adding or updating an object draws a frame on its own. With it off,
            call plot.render() yourself. It has never controlled a render loop - K3D
            draws only when something changed. Named auto_rendering before 3.0.0.
        fps: `Float`.
            Fps of animation.
        minimum_fps: `Float`.
            If negative then disabled. Set target FPS to adaptative resolution.
        objects: `list`.
            List of `k3d.objects.Drawable` currently included in the plot, not to be changed directly.
        additional_js_code: `str`.
            Additional Js code that will be run after plot is initialized
    """

    pass

__all__ = ["Plot"]
