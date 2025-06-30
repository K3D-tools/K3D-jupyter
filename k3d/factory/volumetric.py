"""Factory functions for volumetric and voxel-based objects."""

import numpy as np
from ..helpers import check_attribute_color_range
from ..objects import (
    Volume, MIP, VolumeSlice, Voxels, SparseVoxels, VoxelsGroup, MarchingCubes, VoxelChunk
)
from ..transform import process_transform_arguments
from .common import _default_color, nice_colors, default_colormap


def volume(
        volume,
        color_map=None,
        opacity_function=None,
        color_range=[],
        samples=512.0,
        alpha_coef=50.0,
        gradient_step=0.005,
        shadow="off",
        interpolation=True,
        shadow_delay=500,
        shadow_res=128,
        focal_length=0.0,
        focal_plane=100.0,
        ray_samples_count=16,
        mask=[],
        mask_opacities=[],
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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
        volume,
        color_map=None,
        opacity_function=None,
        color_range=[],
        samples=512.0,
        gradient_step=0.005,
        interpolation=True,
        mask=[],
        mask_opacities=[],
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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


def volume_slice(volume=[], color_map=None, color_range=[], opacity_function=[],
                 opacity=1.0, mask=[], active_masks=[], color_map_masks=None,
                 mask_opacity=0.5, slice_x=-1, slice_y=-1, slice_z=0, interpolation=1, name=None,
                 group=None,
                 custom_data=None, compression_level=0, **kwargs):
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
        voxels,
        color_map=None,
        wireframe=False,
        outlines=True,
        outlines_color=0,
        opacity=1.0,
        bounds=None,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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
        sparse_voxels,
        space_size,
        color_map=None,
        wireframe=False,
        outlines=True,
        outlines_color=0,
        opacity=1.0,
        bounds=None,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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
        space_size,
        voxels_group=[],
        chunks_ids=[],
        color_map=None,
        wireframe=False,
        outlines=True,
        outlines_color=0,
        opacity=1.0,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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
        scalar_field,
        level,
        color=_default_color,
        attribute=[],
        color_map=None,
        color_range=[],
        opacity_function=[],
        wireframe=False,
        flat_shading=True,
        shininess=50.0,
        opacity=1.0,
        spacings_x=[],
        spacings_y=[],
        spacings_z=[],
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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


def voxel_chunk(voxels, coord, multiple=1, compression_level=0):
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