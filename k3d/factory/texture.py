"""Factory function for texture objects."""

import numpy as np
from ..helpers import check_attribute_color_range
from ..objects import Texture
from ..transform import process_transform_arguments
from .common import default_colormap


def texture(
        binary=None,
        file_format=None,
        color_map=None,
        color_range=[],
        attribute=[],
        puv=[],
        opacity_function=[],
        interpolation=True,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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