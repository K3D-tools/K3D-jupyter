"""Factory functions for volumetric and voxel-based objects."""

import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple

from ..helpers import check_attribute_color_range
from ..objects import (
    Volume, MIP, VolumeSlice, Voxels, SparseVoxels, VoxelsGroup, MarchingCubes, VoxelChunk
)
from ..transform import process_transform_arguments
from .common import _default_color, nice_colors, default_colormap

# Type aliases for better readability
ArrayLike = Union[List, np.ndarray, Tuple]
ColorMap = Union[List[List[float]], Dict[str, Any], np.ndarray]
ColorRange = List[float]
OpacityFunction = List[float]


def volume(
        volume: ArrayLike,
        color_map: Optional[ColorMap] = None,
        opacity_function: Optional[OpacityFunction] = None,
        color_range: ColorRange = None,
        samples: float = 512.0,
        alpha_coef: float = 50.0,
        gradient_step: float = 0.005,
        shadow: str = "off",
        interpolation: bool = True,
        shadow_delay: int = 500,
        shadow_res: int = 128,
        focal_length: float = 0.0,
        focal_plane: float = 100.0,
        ray_samples_count: int = 16,
        mask: ArrayLike = None,
        mask_opacities: ArrayLike = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> Volume:
    if color_range is None:
        color_range = []
    if mask is None:
        mask = []
    if mask_opacities is None:
        mask_opacities = []
        
    if color_map is None:
        color_map = default_colormap

    color_range = (
        check_attribute_color_range(volume, color_range)
        if type(color_range) is not dict
        else color_range
    )

    if opacity_function is None:
        opacity_function = [np.min(color_map[::4]), 0.0, np.max(color_map[::4]), 1.0]

    return process_transform_arguments(
        Volume(
            volume=volume,
            color_map=color_map,
            opacity_function=opacity_function,
            color_range=color_range,
            compression_level=compression_level,
            samples=samples,
            alpha_coef=alpha_coef,
            gradient_step=gradient_step,
            interpolation=interpolation,
            shadow=shadow,
            shadow_delay=shadow_delay,
            shadow_res=shadow_res,
            focal_plane=focal_plane,
            focal_length=focal_length,
            mask=mask,
            mask_opacities=mask_opacities,
            name=name,
            group=group,
            custom_data=custom_data,
            ray_samples_count=ray_samples_count,
        ),
        **kwargs
    )


def mip(
        volume: ArrayLike,
        color_map: Optional[ColorMap] = None,
        opacity_function: Optional[OpacityFunction] = None,
        color_range: ColorRange = None,
        samples: float = 512.0,
        gradient_step: float = 0.005,
        interpolation: bool = True,
        mask: ArrayLike = None,
        mask_opacities: ArrayLike = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> MIP:
    if color_range is None:
        color_range = []
    if mask is None:
        mask = []
    if mask_opacities is None:
        mask_opacities = []
        
    if color_map is None:
        color_map = default_colormap

    color_range = (
        check_attribute_color_range(volume, color_range)
        if type(color_range) is not dict
        else color_range
    )

    if opacity_function is None:
        opacity_function = [np.min(color_map[::4]), 0.0, np.max(color_map[::4]), 1.0]

    return process_transform_arguments(
        MIP(
            volume=volume,
            color_map=color_map,
            opacity_function=opacity_function,
            color_range=color_range,
            samples=samples,
            gradient_step=gradient_step,
            interpolation=interpolation,
            mask=mask,
            mask_opacities=mask_opacities,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
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
        custom_data: Optional[Dict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> VolumeSlice:
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
        
    if color_map is None:
        color_map = default_colormap

    if color_map_masks is None:
        color_map_masks = nice_colors

    color_map = np.array(color_map, np.float32) if type(color_map) is not dict else color_map

    if len(volume) > 0:
        color_range = check_attribute_color_range(volume, color_range)

    return process_transform_arguments(
        VolumeSlice(volume=volume,
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
                    compression_level=compression_level),
        **kwargs
    )


def voxels(
        voxels: ArrayLike,
        color_map: Optional[ColorMap] = None,
        wireframe: bool = False,
        outlines: bool = True,
        outlines_color: int = 0,
        opacity: float = 1.0,
        bounds: Optional[ArrayLike] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> Voxels:
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
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def sparse_voxels(
        sparse_voxels: ArrayLike,
        space_size: ArrayLike,
        color_map: Optional[ColorMap] = None,
        wireframe: bool = False,
        outlines: bool = True,
        outlines_color: int = 0,
        opacity: float = 1.0,
        bounds: Optional[ArrayLike] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> SparseVoxels:
    if color_map is None:
        color_map = nice_colors

    assert (
            isinstance(space_size, (tuple, list, np.ndarray))
            and np.shape(space_size) == (3,)
            and all(d > 0 for d in space_size)
    )

    return process_transform_arguments(
        SparseVoxels(
            sparse_voxels=sparse_voxels,
            space_size=space_size,
            color_map=color_map,
            wireframe=wireframe,
            outlines=outlines,
            outlines_color=outlines_color,
            opacity=opacity,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def voxels_group(
        space_size: ArrayLike,
        voxels_group: List[Dict[str, Any]] = None,
        chunks_ids: List[int] = None,
        color_map: Optional[ColorMap] = None,
        wireframe: bool = False,
        outlines: bool = True,
        outlines_color: int = 0,
        opacity: float = 1.0,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> VoxelsGroup:
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
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
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
        shininess: float = 50.0,
        opacity: float = 1.0,
        spacings_x: ArrayLike = None,
        spacings_y: ArrayLike = None,
        spacings_z: ArrayLike = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> MarchingCubes:
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
            shininess=shininess,
            opacity=opacity,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def voxel_chunk(
        voxels: ArrayLike,
        coord: ArrayLike,
        multiple: int = 1,
        compression_level: int = 0
) -> VoxelChunk:
    """Create a VoxelChunk object for selective updating voxels.
    
    Args:
        voxels: Array of voxel data
        coord: Coordinates for the chunk
        multiple: Multiple factor
        compression_level: Compression level for the chunk
        
    Returns:
        VoxelChunk object
    """
    return VoxelChunk(
        voxels=np.array(voxels, np.uint8),
        coord=np.array(coord, np.uint32),
        multiple=multiple,
        compression_level=compression_level,
    ) 