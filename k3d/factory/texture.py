"""Factory function for texture objects."""

from typing import Any, Callable, Optional, Tuple, Union
from typing import Dict as TypingDict
from typing import List as TypingList

import numpy as np

from ..helpers import check_attribute_color_range
from ..objects import Texture
from ..transform import process_transform_arguments
from .common import default_colormap

# Type aliases for better readability
ArrayLike = Union[TypingList, np.ndarray, Tuple]
ColorMap = Union[TypingList[TypingList[float]], TypingDict[str, Any], np.ndarray]
ColorRange = TypingList[float]
OpacityFunction = TypingList[float]


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
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        visible: bool = True,
        click_callback: Optional[Callable] = None,
        hover_callback: Optional[Callable] = None,
        **kwargs: Any,
) -> Texture:
    """
    Create a Texture drawable from an encoded image or from data and a colormap.

    Parameters
    ----------
    binary : bytes, optional
        Image file contents, as read from disk. Default is None.
    file_format : str, optional
        Format of the image in `binary`, without the dot, for example 'png' or 'jpg'. The browser
        has to be able to decode it. Default is None.
    color_map : list, optional
        A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The
        first quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in
        the range 0.0 to 1.0. Default is None.
    color_range : list, optional
        A pair [min_value, max_value], which determines the levels of color attribute mapped to 0
        and 1 in the color map respectively. Default is None.
    attribute : array_like, optional
        Array of float attribute for the color mapping, coresponding to each pixels. Default is
        None.
    puv : array_like, optional
        Origin and two edge vectors [P, U, V] of the plane the texture is drawn on, nine floats in
        all. Default is None.
    opacity_function : list, optional
        A list of float tuples (attribute value, opacity), sorted by attribute value. The first
        tuple should have value 0.0, the last 1.0; opacity is in the range 0.0 to 1.0. Default is
        None.
    interpolation : bool, optional
        Whether data should be interpolatedor not. Default is True.
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
    Texture
        The created Texture object.
    """
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
            visible=visible,
            click_callback=click_callback,
            hover_callback=hover_callback,
        ),
        **kwargs,
    )
