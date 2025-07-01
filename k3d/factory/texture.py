"""Factory function for texture objects."""

import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple

from ..helpers import check_attribute_color_range
from ..objects import Texture
from ..transform import process_transform_arguments
from .common import default_colormap

# Type aliases for better readability
ArrayLike = Union[List, np.ndarray, Tuple]
ColorMap = Union[List[List[float]], Dict[str, Any], np.ndarray]
ColorRange = List[float]
OpacityFunction = List[float]


def texture(
        binary: Optional[bytes] = None,
        file_format: Optional[str] = None,
        color_map: Optional[ColorMap] = None,
        color_range: ColorRange = None,
        attribute: ArrayLike = None,
        puv: ArrayLike = None,
        opacity_function: OpacityFunction = None,
        interpolation: bool = True,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> Texture:
    if color_range is None:
        color_range = []
    if attribute is None:
        attribute = []
    if puv is None:
        puv = []
    if opacity_function is None:
        opacity_function = []
        
    if color_map is None:
        color_map = default_colormap
    color_map = np.array(color_map, np.float32)
    attribute = np.array(attribute, np.float32)
    color_range = check_attribute_color_range(attribute, color_range)

    return process_transform_arguments(
        Texture(
            binary=binary,
            file_format=file_format,
            color_map=color_map,
            color_range=color_range,
            attribute=attribute,
            opacity_function=opacity_function,
            puv=puv,
            interpolation=interpolation,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    ) 