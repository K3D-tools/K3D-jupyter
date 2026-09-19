"""Factory function for point cloud objects."""

from typing import Any, Optional, Tuple, Union
from typing import Dict as TypingDict
from typing import List as TypingList

import numpy as np

from ..helpers import check_attribute_color_range
from ..objects import Points
from ..transform import process_transform_arguments
from .common import _default_color, default_colormap

# Type aliases for better readability
ArrayLike = Union[TypingList, np.ndarray, Tuple]
ColorMap = Union[TypingList[TypingList[float]], TypingDict[str, Any], np.ndarray]
ColorRange = TypingList[float]
OpacityFunction = TypingList[float]


def points(
        positions: ArrayLike,
        colors: TypingList[int] = None,
        color: int = _default_color,
        point_size: float = 1.0,
        point_sizes: ArrayLike = None,
        roughness: float = 0.4,
        metalness: float = 0.0,
        shininess: float = None,
        shader: str = "3d",
        opacity: float = 1.0,
        opacities: ArrayLike = None,
        attribute: ArrayLike = None,
        color_map: Optional[ColorMap] = None,
        color_range: ColorRange = None,
        opacity_function: OpacityFunction = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        mesh_detail: int = 2,
        **kwargs: Any,
) -> Points:
    """
    Create a Points drawable representing a point cloud.

    Parameters
    ----------
    positions : array_like
        Array with (x, y, z) coordinates of the points.
    colors : array_like, optional
        Same-length array of (`int`) packed RGB color of the points (0xff0000 is red, 0xff is
        blue). Default is None.
    color : int, optional
        Packed RGB color of the points (0xff0000 is red, 0xff is blue) when `colors` is empty.
        Default is 255.
    point_size : float, optional
        Diameter of the balls representing the points in 3D space. Default is 1.0.
    point_sizes : array_like, optional
        Same-length array of `float` sizes of the points. Default is None.
    roughness : float, optional
        Roughness of object material. Default is 0.4.
    metalness : float, optional
        Metalness of object material. Default is 0.0.
    shininess : float, optional
        Removed in 3.0.0; passing it raises. Use roughness and metalness. Default is None.
    shader : str, optional
        Display style (name of the shader used) of the points. Legal values are: 'flat' simple
        circles with uniform color, 'dot' simple dot with uniform color, '3d' little 3D balls
        (impostors) with full PBR lighting - the highlights are driven by `roughness` and
        `metalness` (`3dSpecular` is accepted as a legacy alias), 'mesh' high precision triangle
        mesh of a ball (high quality and GPU load). Default is '3d'.
    opacity : float, optional
        Opacity of the points, in the range 0.0 to 1.0. Default is 1.0.
    opacities : array_like, optional
        Same-length array of `float` opacities of the points, used instead of `opacity`. Default
        is None.
    attribute : array_like, optional
        Array of float attribute for the color mapping, coresponding to each point. Default is
        None.
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
    name : str, optional
        A name of the object. Default is None.
    group : str, optional
        A name of a group. Default is None.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.
    compression_level : int, optional
        Level of compression [-1, 9]. Default is 0.
    mesh_detail : int, optional
        Subdivision level of the ball mesh; only used by shader='mesh'. Default is 2.
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    Points
        The created Points object.
    """
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

    # pre-2.19 alias: '3dSpecular' folded into '3d' - the highlights are driven
    # by roughness/metalness now
    if shader == "3dSpecular":
        shader = "3d"

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
            roughness=roughness,
            metalness=metalness,
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
        **kwargs,
    )
