"""Factory functions for creating Plot widgets."""

import warnings
from typing import Any, Optional, Tuple, Union
from typing import Dict as TypingDict
from typing import List as TypingList

import numpy as np

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
        renderer: str = "simple",
        environment: str = "neutral",
        environment_rotation: float = 0.0,
        tone_mapping: str = "none",
        ao_radius: float = 0.07,
        ao_strength: float = 1.8,
        cinematic_samples: int = 64,
        cinematic_bounces: int = 6,
        cinematic_glossy_filter: float = 0.25,
        cinematic_seed: Optional[int] = None,
        cinematic_denoise: float = 0.0,
        cinematic_bokeh_size: float = 0.0,
        cinematic_focus_distance: float = 0.0,
        cinematic_aperture_blades: int = 0,
        axes: TypingList[str] = None,
        axes_helper: float = 1.0,
        axes_helper_colors: TypingList[int] = None,
        camera_mode: str = "trackball",
        manipulate_mode: str = "translate",
        snapshot_type: str = "full",
        render_on_change: bool = True,
        auto_rendering: Optional[bool] = None,
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
        time_speed: float = 1.0,
        time_interpolation: bool = True,
        additional_js_code: str = '',
        custom_data: Optional[TypingDict[str, Any]] = None,
) -> Plot:
    """
    Create a Plot widget, the canvas every drawable is added to.

    Parameters
    ----------
    height : int, optional
        Height of the Widget in pixels, changes have no effect after displaying. Default is 512.
    antialias : int, optional
        Enable antialiasing in WebGL renderer, changes have no effect after displaying. Default is
        3.
    logarithmic_depth_buffer : bool, optional
        Enables logarithmic_depth_buffer in WebGL renderer. Default is True.
    background_color : int, optional
        Packed RGB color of the plot background (0xff0000 is red, 0xff is blue), -1 is for
        transparent. Default is 16777215.
    camera_auto_fit : bool, optional
        Enable automatic camera setting after adding, removing or changing a plot object. Default
        is True.
    grid_auto_fit : bool, optional
        Enable automatic adjustment of the plot grid to contained objects. Default is True.
    grid_visible : bool, optional
        Enable or disable grid. Default is True.
    screenshot_scale : float, optional
        Multiplier to screenshot resolution. A screenshot is the plot's own width and height times
        this, whatever resolution the interactive view happens to be drawing at. Default is 2.0.
    grid : array_like, optional
        6-element tuple specifying the bounds of the plot grid (x0, y0, z0, x1, y1, z1). Default
        is (-1, -1, -1, 1, 1, 1).
    grid_color : int, optional
        Packed RGB color of the plot grids (0xff0000 is red, 0xff is blue). Default is 15132390.
    label_color : int, optional
        Packed RGB color of the labels (0xff0000 is red, 0xff is blue). Default is 4473924.
    lighting : float, optional
        Lighting factor - the exposure knob. In the advanced renderer the environment carries the
        shape of the light, lighting scales its energy. Default is 1.5.
    menu_visibility : Bool, optional
        Whether the panel in the top right corner is shown. Default is True.
    voxel_paint_color : int, optional
        The (initial) integer value to be inserted when editing voxels. Default is 0.
    colorbar_object_id : int, optional
        Id of the object whose color map the colorbar shows. -1 picks the first object that has a
        color range. Default is -1.
    camera_fov : float, optional
        Camera Field of View. Default is 60.0.
    time : float, optional
        Time value (used in TimeSeries) Default is 0.0.
    depth_peels : int, optional
        Set the maximum number of peels to use. Disabled if zero. With peeling on, volumes compose
        correctly with intersecting meshes (the ray march is split at the layer depths); use
        depth_peels >= 3, below that the effect is unpredictable. Default is 0.
    renderer : str, optional
        Rendering pipeline of the plot. Legal values are: 'simple' the classic rasteriser with a
        fixed light rig (default), 'advanced' image-based lighting from the environment map,
        physically based materials and ambient occlusion, 'cinematic' progressive path tracing
        with global illumination. Requires WebGL2 with renderable float textures; when the browser
        cannot run it, the switch fails with an error instead of falling back to another renderer.
        Default is 'simple'.
    environment : str or array_like, optional
        The light environment of the advanced renderer. Legal values are: 'neutral' procedural
        achromatic gradient with a soft key light (default), 'studio' procedural gradient with two
        soft studio lights, 'outdoor' procedural sky with a sun disc and ground, 'name from
        k3d.environments.available()' a photographic HDRI shipped with the package (Poly Haven,
        CC0), 'array_like' a custom (height, width, 3) float32 equirectangular radiance map. Every
        map is energy-normalised. Default is 'neutral'.
    environment_rotation : float, optional
        Rotation of the environment map around the scene's up axis, in radians. Default is 0.0.
    tone_mapping : str, optional
        Tone curve applied by the advanced renderer. Legal values are: 'none' linear output
        (default), 'agx' AgX filmic curve, 'aces' ACES filmic curve. Default is 'none'.
    ao_radius : float, optional
        Occlusion radius of the advanced renderer's ambient occlusion, as a fraction of the
        scene's bounding-box diagonal, in (0, 1]. Default 0.07. Dense point clouds and closed
        interiors usually want a smaller radius. Default is 0.07.
    ao_strength : float, optional
        Exponent deepening the ambient occlusion shadows, in [0, 10]. 0 disables the darkening,
        default 1.8. Default is 1.8.
    cinematic_samples : int, optional
        Sample budget of the cinematic renderer, in [1, 100000]. Default 64, which settles in a
        moment; raise it for a final render. The interactive view accumulates one sample per
        animation frame up to this budget, then parks itself; any change to the camera, the scene
        or the lighting restarts the accumulation from zero. Screenshots always render the full
        budget. Default is 64.
    cinematic_bounces : int, optional
        Light bounce count of the cinematic renderer's path tracing, in [1, 32]. Default 6.
        Default is 6.
    cinematic_glossy_filter : float, optional
        How much the cinematic renderer widens a glossy lobe in proportion to the roughness
        already gathered along a path, in [0, 1]. Default 0.25. It removes fireflies where they
        live and leaves a specular seen directly untouched; 0 disables it. Default is 0.25.
    cinematic_seed : int or None, optional
        Seed of the path tracer's sample sequence, an int in the range [1, 2**31 - 1]. With None
        every accumulation starts from fresh noise; with a seed the same scene renders the same
        image every time. 0 is refused, so that "unset" and "seeded" never blur. Default is None.
    cinematic_denoise : float, optional
        How hard the cinematic renderer filters Monte Carlo noise out of the traced image,
        measured in standard deviations of the noise it estimates per pixel. Default 0 is off, and
        the only value that leaves the image exactly as it was traced. Around 2 removes most of
        the grain a moderate sample budget leaves behind; around 4 bone in a CT scan starts to
        look waxy, because the grain and the trabecular texture under it go together. The filter
        is guided by the spread between two halves of the accumulation and runs in linear space
        before tone mapping, over a five by five kernel and no wider - which is where the grain is
        and where almost nothing else is. It is not a substitute for samples: it is worth roughly
        four times as many of them on a volume, and nothing at all once the render has converged.
        Default is 0.0.
    cinematic_bokeh_size : float, optional
        Diameter of the cinematic renderer's aperture, in scene units. Default 0, a pinhole -
        everything in focus, and the only value that leaves the image identical to the other
        renderers. Anything above it defocuses whatever is not at the focus distance, and costs a
        shader recompile the first time it leaves zero. Default is 0.0.
    cinematic_focus_distance : float, optional
        How far in front of the camera the cinematic renderer focuses, in scene units. Default 0
        means the camera's own target, so the plot is sharp where you are looking. Ignored while
        cinematic_bokeh_size is 0. Default is 0.0.
    cinematic_aperture_blades : int, optional
        How many blades the cinematic renderer's iris has: 0, the default, is a perfect circle,
        and 3 to 16 give an aperture of that many sides, which is what makes an out-of-focus
        highlight read as hexagonal rather than round. Ignored while cinematic_bokeh_size is 0.
        Default is 0.
    axes : list, optional
        Axes labels for plot. Default is None.
    axes_helper : float, optional
        Axes helper size. Default is 1.0.
    axes_helper_colors : List, optional
        List of triple packed RGB color of the axes helper (0xff0000 is red, 0xff is blue).
        Default is None.
    camera_mode : str, optional
        Mode of camera movement. Legal values are: 'trackball' orbit around point with dynamic up-
        vector of camera, 'orbit' orbit around point with fixed up-vector of camera, 'fly' orbit
        around point with dynamic up-vector of camera, mouse wheel also moves target point.
        Default is 'trackball'.
    manipulate_mode : str, optional
        Mode of manipulate widgets. Legal values are: 'translate' Translation widget, 'rotate'
        Rotation widget, 'scale' Scaling widget. Default is 'translate'.
    snapshot_type : string, optional
        Can be 'full', 'online' or 'inline'. Default is 'full'.
    render_on_change : Bool, optional
        Whether adding or updating an object draws a frame on its own. With it off, call
        plot.render() yourself. It has never controlled a render loop - K3D draws only when
        something changed. Named auto_rendering before 3.0.0. Default is True.
    auto_rendering : bool, optional
        Renamed to render_on_change in 3.0.0; passing it warns and sets that instead. Default is
        None.
    camera_no_zoom : Bool, optional
        Lock for camera zoom. Default is False.
    camera_no_rotate : Bool, optional
        Lock for camera rotation. Default is False.
    camera_no_pan : Bool, optional
        Lock for camera pan. Default is False.
    camera_rotate_speed : float, optional
        Speed of camera rotation. Default is 1.0.
    camera_zoom_speed : float, optional
        Speed of camera zoom. Default is 1.2.
    camera_pan_speed : float, optional
        Speed of camera pan. Default is 0.3.
    camera_damping_factor : float, optional
        Defines the intensity of damping. Default is 0 (disabled). Default is 0.0.
    camera_up_axis : str, optional
        Fixed up axis for camera. Legal values are: 'x' x axis, 'y' y axis, 'z' z axis, 'none'
        Handling click_callback and hover_callback on some type of objects. Default is 'none'.
    fps : float, optional
        Fps of animation. Default is 25.0.
    minimum_fps : float, optional
        If negative then disabled. Set target FPS to adaptative resolution. Default is -1.
    fps_meter : Bool, optional
        Whether to show the frames-per-second counter. It measures animation frames, not draws: an
        idle plot draws nothing and the meter still ticks. Default is False.
    name : str, optional
        A name of the object. Default is None.
    time_speed : float, optional
        Time speed (used in TimeSeries) Default is 1.0.
    time_interpolation : Bool, optional
        Whether a time series blends between the two nearest keyframes. With it off, playback
        steps from frame to frame. Default is True.
    additional_js_code : str, optional
        Additional Js code that will be run after plot is initialized Default is ''.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.

    Returns
    -------
    Plot
        The created Plot object.
    """
    if auto_rendering is not None:
        warnings.warn(
            "auto_rendering was renamed to render_on_change in 3.0.0",
            DeprecationWarning,
            stacklevel=2,
        )

        render_on_change = auto_rendering

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
        renderer=renderer,
        environment=environment,
        environment_rotation=environment_rotation,
        tone_mapping=tone_mapping,
        ao_radius=ao_radius,
        ao_strength=ao_strength,
        cinematic_samples=cinematic_samples,
        cinematic_bounces=cinematic_bounces,
        cinematic_glossy_filter=cinematic_glossy_filter,
        cinematic_seed=cinematic_seed,
        cinematic_denoise=cinematic_denoise,
        cinematic_bokeh_size=cinematic_bokeh_size,
        cinematic_focus_distance=cinematic_focus_distance,
        cinematic_aperture_blades=cinematic_aperture_blades,
        axes=axes,
        axes_helper=axes_helper,
        axes_helper_colors=axes_helper_colors,
        screenshot_scale=screenshot_scale,
        camera_fov=camera_fov,
        name=name,
        camera_mode=camera_mode,
        manipulate_mode=manipulate_mode,
        snapshot_type=snapshot_type,
        camera_no_zoom=camera_no_zoom,
        camera_no_rotate=camera_no_rotate,
        camera_no_pan=camera_no_pan,
        camera_rotate_speed=camera_rotate_speed,
        camera_zoom_speed=camera_zoom_speed,
        camera_damping_factor=camera_damping_factor,
        camera_pan_speed=camera_pan_speed,
        camera_up_axis=camera_up_axis,
        render_on_change=render_on_change,
        fps=fps,
        minimum_fps=minimum_fps,
        time_speed=time_speed,
        time_interpolation=time_interpolation,
        additional_js_code=additional_js_code,
        fps_meter=fps_meter,
        custom_data=custom_data,
    )
