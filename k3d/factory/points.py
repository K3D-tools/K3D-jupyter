"""Factory function for point cloud objects."""

import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple

from ..helpers import check_attribute_color_range
from ..objects import Points
from ..transform import process_transform_arguments
from .common import _default_color, default_colormap

# Type aliases for better readability
ArrayLike = Union[List, np.ndarray, Tuple]
ColorMap = Union[List[List[float]], Dict[str, Any], np.ndarray]
ColorRange = List[float]
OpacityFunction = List[float]


def points(
        positions: ArrayLike,
        colors: List[int] = None,
        color: int = _default_color,
        point_size: float = 1.0,
        point_sizes: ArrayLike = None,
        shininess: float = 50.0,
        shader: str = "3dSpecular",
        opacity: float = 1.0,
        opacities: ArrayLike = None,
        attribute: ArrayLike = None,
        color_map: Optional[ColorMap] = None,
        color_range: ColorRange = None,
        opacity_function: OpacityFunction = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        compression_level: int = 0,
        mesh_detail: int = 2,
        **kwargs: Any
) -> Points:
    if colors is None:
        colors = []
    if point_sizes is None:
        point_sizes = []
    if opacities is None:
        opacities = []
    if attribute is None:
        attribute = []
    if color_range is None:
        color_range = []
    if opacity_function is None:
        opacity_function = []
        
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