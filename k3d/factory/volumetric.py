"""Factory functions for volumetric and voxel-based objects."""

import warnings
from typing import Any, Callable, Optional, Tuple, Union
from typing import Dict as TypingDict
from typing import List as TypingList

import numpy as np

from ..helpers import check_attribute_color_range, rgb_volume_channels
from ..objects import MIP, MarchingCubes, SparseVoxels, Volume, VolumeSlice, VoxelChunk, Voxels, VoxelsGroup
from ..transform import process_transform_arguments
from .common import _default_color, default_colormap, nice_colors

# Type aliases for better readability
ArrayLike = Union[TypingList, np.ndarray, Tuple]
ColorMap = Union[TypingList[TypingList[float]], TypingDict[str, Any], np.ndarray]
ColorRange = TypingList[float]
OpacityFunction = TypingList[float]


def _warn_rgb_ignores(color_map, color_range):
    """An RGB volume maps nothing, so a colormap or a range would only look like it works."""
    if color_map is not None and len(color_map) > 0:
        warnings.warn(
            "color_map is ignored for an RGB volume: the colour is in the data",
            stacklevel=3,
        )

    if color_range is not None and len(color_range) > 0:
        warnings.warn(
            "color_range is ignored for an RGB volume: there is no scalar to window",
            stacklevel=3,
        )


def volume(
        volume: ArrayLike,
        color_map: Optional[ColorMap] = None,
        opacity_function: Optional[OpacityFunction] = None,
        color_range: ColorRange = None,
        samples: float = 512.0,
        alpha_coef: float = 50.0,
        gradient_step: float = 0.005,
        roughness: float = 0.25,
        metalness: float = 0.0,
        light_scale: float = 1.0,
        shadow: str = "off",
        interpolation: bool = True,
        shadow_delay: int = 500,
        shadow_res: int = 128,
        mask: ArrayLike = None,
        mask_opacities: ArrayLike = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        visible: bool = True,
        **kwargs: Any,
) -> Volume:
    """
    Create a Volume drawable for direct volume rendering of a scalar field.

    Parameters
    ----------
    volume : array_like
        3D array of `float`, indexed as [z, y, x]. A 4D array of `uint8` shaped [z, y, x, 3] or
        [z, y, x, 4] is colour per voxel instead - the measurement itself, the way a photographic
        or RGB-encoded scan carries it - and is drawn without a colormap: color_map and
        color_range are ignored, and the alpha comes from opacity_function over Rec. 709
        luminance.
    color_map : list, optional
        A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The
        first quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in
        the range 0.0 to 1.0. Default is None.
    opacity_function : list, optional
        A list of float tuples (attribute value, opacity), sorted by attribute value. The first
        tuple should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0. Default is
        None.
    color_range : list, optional
        A pair [min_value, max_value], which determines the levels of color attribute mapped to 0
        and 1 in the color map respectively. Default is None.
    samples : float, optional
        Number of iteration per 1 unit of space. Default is 512.0.
    alpha_coef : float, optional
        Alpha multiplier. Default is 50.0.
    gradient_step : float, optional
        Distance the finite differences of the shading gradient are taken over, as a fraction of
        the mean edge of the volume's box. Default is 0.005.
    roughness : float, optional
        Roughness of the specular highlight of the isodensity surface (GGX), 0.0-1.0. Default is
        0.25.
    metalness : float, optional
        Metalness of the specular highlight: 0.0 dielectric, 1.0 metal tinted by the transfer-
        function colour. Default is 0.0.
    light_scale : float, optional
        Multiplies the light the medium collects, so the volume can be exposed without touching
        the rest of the scene. Only the cinematic renderer reads it. Every ratio in the image
        survives, self-shadowing included - unlike a brighter environment, which lifts the
        geometry around the volume as well. Default is 1.0.
    shadow : str, optional
        Type of shadow on volume. Legal values are: 'off' shadow disabled, 'on_demand' update
        shadow map on demand ( self.shadow_map_update() ), 'dynamic' update shadow map
        automaticaly every shadow_delay. Default is 'off'.
    interpolation : bool, optional
        Whether volume raycasting should interpolate data or not. Default is True.
    shadow_delay : float, optional
        Minimum number of miliseconds between shadow map updates. Default is 500.
    shadow_res : int, optional
        Resolution of shadow map. Default is 128.
    mask : array_like, optional
        3D array of `int` in range (0, 255), indexed as [z, y, x]. Default is None.
    mask_opacities : array_like, optional
        List of opacity values for mask. Default is None.
    name : str, optional
        A name of the object. Default is None.
    group : str, optional
        A name of a group. Default is None.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.
    compression_level : int, optional
        Level of compression [-1, 9]. Default is 0.
    visible : bool, optional
        Whether the object is drawn. Default is True.
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    Volume
        The created Volume object.
    """
    if color_range is None:
        color_range = []
    if mask is None:
        mask = []
    if mask_opacities is None:
        mask_opacities = []

    if rgb_volume_channels(volume):
        _warn_rgb_ignores(color_map, color_range)

        color_map = []
        color_range = []

        # the colour needs no ramp, the alpha does: without one the box is opaque and all you
        # see is its front face. Luminance from 0 to 1 is the axis this ramp runs along.
        if opacity_function is None:
            opacity_function = [0.0, 0.0, 1.0, 1.0]
    else:
        if color_map is None:
            color_map = default_colormap

        color_range = (
            check_attribute_color_range(volume, color_range)
            if type(color_range) is not dict
            else color_range
        )

        if opacity_function is None:
            # ravel first: a colormap given as (N, 4) slices by row here, and the ramp then
            # spans whatever the sampled rows happen to hold instead of the first column
            if type(color_map) is dict:
                values = np.concatenate(
                    [np.asarray(frame, np.float32).ravel()[::4] for frame in color_map.values()]
                )
            else:
                values = np.asarray(color_map, np.float32).ravel()[::4]
            opacity_function = [np.min(values), 0.0, np.max(values), 1.0]

    return process_transform_arguments(
        Volume(
            volume=volume,
            color_map=color_map,
            opacity_function=opacity_function,
            color_range=color_range,
            compression_level=compression_level,
            visible=visible,
            samples=samples,
            alpha_coef=alpha_coef,
            gradient_step=gradient_step,
            roughness=roughness,
            metalness=metalness,
            light_scale=light_scale,
            interpolation=interpolation,
            shadow=shadow,
            shadow_delay=shadow_delay,
            shadow_res=shadow_res,
            mask=mask,
            mask_opacities=mask_opacities,
            name=name,
            group=group,
            custom_data=custom_data,
        ),
        **kwargs,
    )


def mip(
        volume: ArrayLike,
        color_map: Optional[ColorMap] = None,
        opacity_function: Optional[OpacityFunction] = None,
        color_range: ColorRange = None,
        samples: float = 512.0,
        gradient_step: float = 0.005,
        roughness: float = 0.25,
        metalness: float = 0.0,
        interpolation: bool = True,
        mask: ArrayLike = None,
        mask_opacities: ArrayLike = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        visible: bool = True,
        **kwargs: Any,
) -> MIP:
    """
    Create a MIP drawable, a maximum intensity projection of a scalar field.

    Parameters
    ----------
    volume : array_like
        3D array of `float`, indexed as [z, y, x]. A 4D array of `uint8` shaped [z, y, x, 3] or
        [z, y, x, 4] is colour per voxel; the maximum along the ray is then taken over Rec. 709
        luminance and the pixel keeps the colour of the voxel that reached it, so color_map and
        color_range are ignored and opacity_function runs along that luminance.
    color_map : list, optional
        A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The
        first quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in
        the range 0.0 to 1.0. Default is None.
    opacity_function : list, optional
        A list of float tuples (attribute value, opacity), sorted by attribute value. The first
        tuple should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0. Default is
        None.
    color_range : list, optional
        A pair [min_value, max_value], which determines the levels of color attribute mapped to 0
        and 1 in the color map respectively. Default is None.
    samples : float, optional
        Number of iteration per 1 unit of space. Default is 512.0.
    gradient_step : float, optional
        Distance the finite differences of the shading gradient are taken over, as a fraction of
        the mean edge of the volume's box. Default is 0.005.
    roughness : float, optional
        Roughness of the specular highlight of the isodensity surface (GGX), 0.0-1.0. Default is
        0.25.
    metalness : float, optional
        Metalness of the specular highlight: 0.0 dielectric, 1.0 metal tinted by the transfer-
        function colour. Default is 0.0.
    interpolation : bool, optional
        Whether the ray march should interpolate the data or read the nearest voxel. Default is
        True.
    mask : array_like, optional
        3D array of `int` in range (0, 255), indexed as [z, y, x]. Default is None.
    mask_opacities : array_like, optional
        List of opacity values for mask. Default is None.
    name : str, optional
        A name of the object. Default is None.
    group : str, optional
        A name of a group. Default is None.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.
    compression_level : int, optional
        Level of compression [-1, 9]. Default is 0.
    visible : bool, optional
        Whether the object is drawn. Default is True.
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    MIP
        The created MIP object.
    """
    if color_range is None:
        color_range = []
    if mask is None:
        mask = []
    if mask_opacities is None:
        mask_opacities = []

    if rgb_volume_channels(volume):
        _warn_rgb_ignores(color_map, color_range)

        color_map = []
        color_range = []

        # the brightest voxel along the ray still needs an alpha, and it comes from the same
        # luminance that decided which voxel that was
        if opacity_function is None:
            opacity_function = [0.0, 0.0, 1.0, 1.0]
    else:
        if color_map is None:
            color_map = default_colormap

        color_range = (
            check_attribute_color_range(volume, color_range)
            if type(color_range) is not dict
            else color_range
        )

        if opacity_function is None:
            # color_map may be a TimeSeries dict, which cannot be sliced; derive the default
            # ramp from the union of its frames in that case.
            # ravel first: a colormap given as (N, 4) slices by row here, and the ramp then
            # spans whatever the sampled rows happen to hold instead of the first column
            if type(color_map) is dict:
                values = np.concatenate(
                    [np.asarray(frame, np.float32).ravel()[::4] for frame in color_map.values()]
                )
            else:
                values = np.asarray(color_map, np.float32).ravel()[::4]
            opacity_function = [np.min(values), 0.0, np.max(values), 1.0]

    return process_transform_arguments(
        MIP(
            volume=volume,
            color_map=color_map,
            opacity_function=opacity_function,
            color_range=color_range,
            samples=samples,
            gradient_step=gradient_step,
            roughness=roughness,
            metalness=metalness,
            interpolation=interpolation,
            mask=mask,
            mask_opacities=mask_opacities,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
            visible=visible,
        ),
        **kwargs,
    )


def volume_slice(
        volume: ArrayLike = None,
        color_map: Optional[ColorMap] = None,
        color_range: ColorRange = None,
        opacity_function: OpacityFunction = None,
        opacity: float = 1.0,
        mask: ArrayLike = None,
        active_masks: ArrayLike = None,
        color_map_masks: Optional[ColorMap] = None,
        mask_opacity: float = 0.5,
        slice_x: int = -1,
        slice_y: int = -1,
        slice_z: int = 0,
        interpolation: int = 1,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        visible: bool = True,
        click_callback: Optional[Callable] = None,
        hover_callback: Optional[Callable] = None,
        **kwargs: Any,
) -> VolumeSlice:
    """
    Create a VolumeSlice drawable showing axis-aligned slices of a scalar field.

    Parameters
    ----------
    volume : array_like, optional
        3D array of `float`, indexed as [z, y, x]. A list of two such arrays is a two-channel
        slice, read through a 2D colormap. A 4D array of `uint8` shaped [z, y, x, 3] or
        [z, y, x, 4] is colour per voxel and is drawn as it stands, without a colormap.
        Default is None.
    color_map : list, optional
        A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The
        first quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in
        the range 0.0 to 1.0. Default is None.
    color_range : list, optional
        A pair [min_value, max_value], which determines the levels of color attribute mapped to 0
        and 1 in the color map respectively. Default is None.
    opacity_function : list, optional
        A list of float tuples (attribute value, opacity), sorted by attribute value. The first
        tuple should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0. Default is
        None.
    opacity : float, optional
        Opacity of slice. Default is 1.0.
    mask : array_like, optional
        3D array of `int` in range (0, 255), indexed as [z, y, x]. Default is None.
    active_masks : array_like, optional
        List of values from mask. Default is None.
    color_map_masks : list, optional
        Flat array of `int` packed RGB colors (0xff0000 is red, 0xff is blue). The color defined
        at index i is for voxel value (i+1), e.g.: Default is None.
    mask_opacity : float, optional
        Mask enhanced coefficient. Default is 0.5.
    slice_x : int, optional
        Index of the slice along x, or -1 for none. Default is -1.
    slice_y : int, optional
        Index of the slice along y, or -1 for none. Default is -1.
    slice_z : int, optional
        Index of the slice along z, or -1 for none. Default is 0.
    interpolation : int, optional
        0 - no interpolation, 1 - linear, 2 - cubic. Default is 1.
    name : str, optional
        A name of the object. Default is None.
    group : str, optional
        A name of a group. Default is None.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.
    compression_level : int, optional
        Level of compression [-1, 9]. Default is 0.
    visible : bool, optional
        Whether the object is drawn. Default is True.
    click_callback : callable, optional
        Called with the picking parameters when the object is clicked, while the plot is
        in mode='callback'. Default is None.
    hover_callback : callable, optional
        Called with the picking parameters when the cursor is over the object, while the
        plot is in mode='callback'. Default is None.
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    VolumeSlice
        The created VolumeSlice object.
    """
    if volume is None:
        volume = []
    if color_range is None:
        color_range = []
    if opacity_function is None:
        opacity_function = []
    if mask is None:
        mask = []
    if active_masks is None:
        active_masks = []

    if color_map is None and not rgb_volume_channels(volume):
        color_map = default_colormap

    if color_map_masks is None:
        color_map_masks = nice_colors

    if rgb_volume_channels(volume):
        _warn_rgb_ignores(color_map, color_range)

        if len(opacity_function) > 0:
            warnings.warn(
                "opacity_function is ignored for an RGB volume_slice: a slice is opaque and "
                "its alpha is the object's opacity",
                stacklevel=2,
            )

        color_map = []
        color_range = []
        opacity_function = []
    else:
        color_map = (
            np.array(color_map, np.float32) if type(color_map) is not dict else color_map
        )

        if len(volume) > 0:
            color_range = check_attribute_color_range(volume, color_range, channels=True)

    # createCanvasGradient2d writes a fixed alpha, so a transfer function has no channel to
    # apply to once there are two. Saying so beats dropping it without a word.
    if isinstance(volume, (list, tuple)) and len(volume) > 1 and len(opacity_function) > 0:
        warnings.warn(
            "opacity_function is ignored for a multi-channel volume_slice: the 2D colormap "
            "carries colour only",
            stacklevel=2,
        )

    return process_transform_arguments(
        VolumeSlice(
            volume=volume,
            color_map=color_map,
            color_range=color_range,
            opacity_function=opacity_function,
            opacity=opacity,
            slice_x=slice_x,
            slice_y=slice_y,
            slice_z=slice_z,
            interpolation=interpolation,
            mask=mask,
            mask_opacity=mask_opacity,
            active_masks=active_masks,
            color_map_masks=color_map_masks,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
            visible=visible,
            click_callback=click_callback,
            hover_callback=hover_callback,
        ),
        **kwargs,
    )


def voxels(
        voxels: ArrayLike,
        color_map: Optional[ColorMap] = None,
        wireframe: bool = False,
        outlines: bool = True,
        outlines_color: int = 0,
        opacity: float = 1.0,
        roughness: float = 0.4,
        metalness: float = 0.0,
        bounds: Optional[ArrayLike] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        visible: bool = True,
        **kwargs: Any,
) -> Voxels:
    """
    Create a Voxels drawable from a dense array of voxel ids.

    Parameters
    ----------
    voxels : array_like
        3D array of `int` in range (0, 255), indexed as [z, y, x]. 0 means empty voxel, 1 and
        above refer to consecutive color_map entries.
    color_map : list, optional
        A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The
        first quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in
        the range 0.0 to 1.0. Default is None.
    wireframe : bool, optional
        Whether mesh should display as wireframe. Default is False.
    outlines : bool, optional
        Whether mesh should display with outlines. Default is True.
    outlines_color : int, optional
        Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue) Default is 0.
    opacity : float, optional
        Opacity of voxels. Default is 1.0.
    roughness : float, optional
        Roughness of the material. Default is 0.4.
    metalness : float, optional
        Metalness of the material. Default is 0.0.
    bounds : array_like, optional
        Bounding box [xmin, xmax, ymin, ymax, zmin, zmax] the object is scaled into; derived from
        the data when omitted. Default is None.
    name : str, optional
        A name of the object. Default is None.
    group : str, optional
        A name of a group. Default is None.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.
    compression_level : int, optional
        Level of compression [-1, 9]. Default is 0.
    visible : bool, optional
        Whether the object is drawn. Default is True.
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    Voxels
        The created Voxels object.
    """
    if color_map is None:
        color_map = nice_colors

    if bounds is not None:
        kwargs["bounds"] = bounds
    else:
        max_z, max_y, max_x = np.shape(voxels)
        kwargs["bounds"] = np.array([0, max_x, 0, max_y, 0, max_z])

    return process_transform_arguments(
        Voxels(
            voxels=voxels,
            color_map=color_map,
            wireframe=wireframe,
            outlines=outlines,
            outlines_color=outlines_color,
            opacity=opacity,
            roughness=roughness,
            metalness=metalness,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
            visible=visible,
        ),
        **kwargs,
    )


def sparse_voxels(
        sparse_voxels: ArrayLike,
        space_size: ArrayLike,
        color_map: Optional[ColorMap] = None,
        wireframe: bool = False,
        outlines: bool = True,
        outlines_color: int = 0,
        opacity: float = 1.0,
        roughness: float = 0.4,
        metalness: float = 0.0,
        bounds: Optional[ArrayLike] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        visible: bool = True,
        **kwargs: Any,
) -> SparseVoxels:
    """
    Create a SparseVoxels drawable from a list of voxel coordinates and ids.

    Parameters
    ----------
    sparse_voxels : array_like
        2D array of `coords` in format [[x,y,z,v],[x,y,z,v]]. v = 0 means empty voxel, 1 and above
        refer to consecutive color_map entries.
    space_size : array_like
        Width, height and length of the voxel space, in voxels.
    color_map : list, optional
        A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The
        first quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in
        the range 0.0 to 1.0. Default is None.
    wireframe : bool, optional
        Whether mesh should display as wireframe. Default is False.
    outlines : bool, optional
        Whether mesh should display with outlines. Default is True.
    outlines_color : int, optional
        Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue) Default is 0.
    opacity : float, optional
        Opacity of voxels. Default is 1.0.
    roughness : float, optional
        Roughness of the material. Default is 0.4.
    metalness : float, optional
        Metalness of the material. Default is 0.0.
    bounds : array_like, optional
        Bounding box [xmin, xmax, ymin, ymax, zmin, zmax] the object is scaled into; derived from
        the data when omitted. Default is None.
    name : str, optional
        A name of the object. Default is None.
    group : str, optional
        A name of a group. Default is None.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.
    compression_level : int, optional
        Level of compression [-1, 9]. Default is 0.
    visible : bool, optional
        Whether the object is drawn. Default is True.
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    SparseVoxels
        The created SparseVoxels object.
    """
    if color_map is None:
        color_map = nice_colors

    assert (
            isinstance(space_size, (tuple, list, np.ndarray))
            and np.shape(space_size) == (3,)
            and all(d > 0 for d in space_size)
    )

    # a named parameter never reaches **kwargs, which is the only thing
    # process_transform_arguments reads - so bounds was accepted and dropped
    if bounds is not None:
        kwargs["bounds"] = bounds

    return process_transform_arguments(
        SparseVoxels(
            sparse_voxels=sparse_voxels,
            space_size=space_size,
            color_map=color_map,
            wireframe=wireframe,
            outlines=outlines,
            outlines_color=outlines_color,
            opacity=opacity,
            roughness=roughness,
            metalness=metalness,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
            visible=visible,
        ),
        **kwargs,
    )


def voxels_group(
        space_size: ArrayLike,
        voxels_group: TypingList[TypingDict[str, Any]] = None,
        chunks_ids: TypingList[int] = None,
        color_map: Optional[ColorMap] = None,
        wireframe: bool = False,
        outlines: bool = True,
        outlines_color: int = 0,
        opacity: float = 1.0,
        roughness: float = 0.4,
        metalness: float = 0.0,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        visible: bool = True,
        **kwargs: Any,
) -> VoxelsGroup:
    """
    Create a VoxelsGroup drawable, a voxel space assembled from chunks.

    Parameters
    ----------
    space_size : array_like
        Width, height and length of the voxel space, in voxels.
    voxels_group : array_like, optional
        List of `chunks` in format {voxels 'np.array, coord' [x,y,z], multiple: number}. Default
        is None.
    chunks_ids : list, optional
        Ids of the VoxelChunk objects the group is assembled from. Default is None.
    color_map : list, optional
        A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The
        first quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in
        the range 0.0 to 1.0. Default is None.
    wireframe : bool, optional
        Whether mesh should display as wireframe. Default is False.
    outlines : bool, optional
        Whether mesh should display with outlines. Default is True.
    outlines_color : int, optional
        Packed RGB color of the resulting outlines (0xff0000 is red, 0xff is blue) Default is 0.
    opacity : float, optional
        Opacity of voxels. Default is 1.0.
    roughness : float, optional
        Roughness of the material. Default is 0.4.
    metalness : float, optional
        Metalness of the material. Default is 0.0.
    name : str, optional
        A name of the object. Default is None.
    group : str, optional
        A name of a group. Default is None.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.
    compression_level : int, optional
        Level of compression [-1, 9]. Default is 0.
    visible : bool, optional
        Whether the object is drawn. Default is True.
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    VoxelsGroup
        The created VoxelsGroup object.
    """
    if voxels_group is None:
        voxels_group = []
    if chunks_ids is None:
        chunks_ids = []

    if color_map is None:
        color_map = nice_colors

    for g in voxels_group:
        g["coord"] = np.array(g["coord"])
        g["voxels"] = np.array(g["voxels"])

        if "multiple" not in g:
            g["multiple"] = 1

    return process_transform_arguments(
        VoxelsGroup(
            voxels_group=voxels_group,
            chunks_ids=chunks_ids,
            space_size=space_size,
            color_map=color_map,
            wireframe=wireframe,
            outlines=outlines,
            outlines_color=outlines_color,
            opacity=opacity,
            roughness=roughness,
            metalness=metalness,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
            visible=visible,
        ),
        **kwargs,
    )


def marching_cubes(
        scalar_field: ArrayLike,
        level: float,
        color: int = _default_color,
        attribute: ArrayLike = None,
        color_map: Optional[ColorMap] = None,
        color_range: ColorRange = None,
        opacity_function: OpacityFunction = None,
        wireframe: bool = False,
        flat_shading: bool = True,
        roughness: float = 0.4,
        metalness: float = 0.0,
        shininess: float = None,
        opacity: float = 1.0,
        spacings_x: ArrayLike = None,
        spacings_y: ArrayLike = None,
        spacings_z: ArrayLike = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        visible: bool = True,
        click_callback: Optional[Callable] = None,
        hover_callback: Optional[Callable] = None,
        **kwargs: Any,
) -> MarchingCubes:
    """
    Create a MarchingCubes drawable, an isosurface of a scalar field.

    Parameters
    ----------
    scalar_field : array_like
        3D array of the scalar field, indexed as [z, y, x]. The surface faces outwards where the
        field is negative inside it.
    level : float
        Value at the computed isosurface.
    color : int, optional
        Packed RGB color of the isosurface (0xff0000 is red, 0xff is blue). Default is 255.
    attribute : array_like, optional
        3D array of float sampled on the same grid as scalar_field, from which the surface colour
        is read at each vertex. A flat, per-vertex array is not accepted: it cannot be sampled at
        a position and is ignored, with a warning in the browser console. Default is None.
    color_map : list, optional
        A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The
        first quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in
        the range 0.0 to 1.0. Default is None.
    color_range : list, optional
        A pair [min_value, max_value], which determines the levels of color attribute mapped to 0
        and 1 in the color map respectively. Default is None.
    opacity_function : list, optional
        A list of float tuples (attribute value, opacity), sorted by attribute value. The first
        tuple should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0. Default is
        None.
    wireframe : bool, optional
        Whether mesh should display as wireframe. Default is False.
    flat_shading : bool, optional
        Whether mesh should display with flat shading. Default is True.
    roughness : float, optional
        Roughness of object material. Default is 0.4.
    metalness : float, optional
        Metalness of object material. Default is 0.0.
    shininess : float, optional
        Removed in 3.0.0; passing it raises. Use roughness and metalness. Default is None.
    opacity : float, optional
        Opacity of mesh. Default is 1.0.
    spacings_x : array_like, optional
        Distances between consecutive samples along x: one shorter than that axis of scalar_field.
        Any other length is ignored and the axis falls back to even spacing. Default is None.
    spacings_y : array_like, optional
        Distances between consecutive samples along y, one shorter than that axis. Default is
        None.
    spacings_z : array_like, optional
        Distances between consecutive samples along z, one shorter than that axis. Default is
        None.
    name : str, optional
        A name of the object. Default is None.
    group : str, optional
        A name of a group. Default is None.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.
    compression_level : int, optional
        Level of compression [-1, 9]. Default is 0.
    visible : bool, optional
        Whether the object is drawn. Default is True.
    click_callback : callable, optional
        Called with the picking parameters when the object is clicked, while the plot is
        in mode='callback'. Default is None.
    hover_callback : callable, optional
        Called with the picking parameters when the cursor is over the object, while the
        plot is in mode='callback'. Default is None.
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    MarchingCubes
        The created MarchingCubes object.
    """
    if attribute is None:
        attribute = []
    if color_range is None:
        color_range = []
    if opacity_function is None:
        opacity_function = []
    if spacings_x is None:
        spacings_x = []
    if spacings_y is None:
        spacings_y = []
    if spacings_z is None:
        spacings_z = []

    if color_map is None:
        color_map = default_colormap

    attribute = (
        np.array(attribute, np.float32) if type(attribute) is not dict else attribute
    )
    color_range = check_attribute_color_range(attribute, color_range)

    return process_transform_arguments(
        MarchingCubes(
            scalar_field=scalar_field,
            spacings_x=spacings_x,
            spacings_y=spacings_y,
            spacings_z=spacings_z,
            color=color,
            attribute=attribute,
            color_map=color_map,
            color_range=color_range,
            opacity_function=opacity_function,
            level=level,
            wireframe=wireframe,
            flat_shading=flat_shading,
            roughness=roughness,
            metalness=metalness,
            shininess=shininess,
            opacity=opacity,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
            visible=visible,
            click_callback=click_callback,
            hover_callback=hover_callback,
        ),
        **kwargs,
    )


def voxel_chunk(
        voxels: ArrayLike, coord: ArrayLike, multiple: int = 1, compression_level: int = 0
) -> VoxelChunk:
    """Create a VoxelChunk object for selective updating voxels.

    Parameters
    ----------
    voxels : array_like
        Array of voxel data.
    coord : array_like
        Coordinates for the chunk.
    multiple : int, optional
        Multiple factor, by default 1.
    compression_level : int, optional
        Compression level for the chunk, by default 0.

    Returns
    -------
    VoxelChunk
        VoxelChunk object.
    """
    return VoxelChunk(
        voxels=np.array(voxels, np.uint8),
        coord=np.array(coord, np.uint32),
        multiple=multiple,
        compression_level=compression_level,
    )
