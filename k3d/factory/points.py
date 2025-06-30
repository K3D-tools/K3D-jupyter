"""Factory function for point cloud objects."""

import numpy as np
from ..helpers import check_attribute_color_range
from ..objects import Points
from ..transform import process_transform_arguments
from .common import _default_color, default_colormap


def points(
        positions,
        colors=[],
        color=_default_color,
        point_size=1.0,
        point_sizes=[],
        shininess=50.0,
        shader="3dSpecular",
        opacity=1.0,
        opacities=[],
        attribute=[],
        color_map=None,
        color_range=[],
        opacity_function=[],
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        mesh_detail=2,
        **kwargs
):
    if color_map is None:
        color_map = default_colormap

    attribute = (
        np.array(attribute, np.float32) if type(attribute) is not dict else attribute
    )
    color_range = check_attribute_color_range(attribute, color_range)

    return process_transform_arguments(
        Points(
            positions=positions,
            colors=colors,
            color=color,
            point_size=point_size,
            point_sizes=point_sizes,
            shininess=shininess,
            shader=shader,
            opacity=opacity,
            opacities=opacities,
            mesh_detail=mesh_detail,
            attribute=attribute,
            color_map=color_map,
            color_range=color_range,
            opacity_function=opacity_function,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    ) 