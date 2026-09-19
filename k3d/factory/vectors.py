"""Factory functions for vector and vector field objects."""

from typing import Any, Optional, Tuple, Union
from typing import Dict as TypingDict
from typing import List as TypingList

import numpy as np

from ..objects import VectorField, Vectors
from ..transform import process_transform_arguments
from .common import _default_color

# Type aliases for better readability
ArrayLike = Union[TypingList, np.ndarray, Tuple]


def vector_field(
        vectors: ArrayLike,
        colors: TypingList[int] = None,
        origin_color: Optional[int] = None,
        head_color: Optional[int] = None,
        color: int = _default_color,
        use_head: bool = True,
        head_size: float = 1.0,
        scale: float = 1.0,
        line_width: float = 0.01,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any,
) -> VectorField:
    """
    Create a VectorField drawable for plotting a regular grid of arrows.

    Parameters
    ----------
    vectors : array_like
        Array of (dx, dy, dz) components on a 2D or 3D grid.
    colors : array_like, optional
        Twice the length of vectors array of int: packed RGB colors (0xff0000 is red, 0xff is
        blue). The array has consecutive pairs (origin_color, head_color) for vectors in row-major
        order. Default is None.
    origin_color : int, optional
        Packed RGB color of the origins (0xff0000 is red, 0xff is blue) when `colors` is empty.
        Default is None.
    head_color : int, optional
        Packed RGB color of the vector heads (0xff0000 is red, 0xff is blue) when `colors` is
        empty. Default is None.
    color : int, optional
        Packed RGB color of the vectors (0xff0000 is red, 0xff is blue), used for whichever of
        `origin_color` and `head_color` is not given. Default is 255.
    use_head : bool, optional
        Whether vectors should display an arrow head. Default is True.
    head_size : float, optional
        The size of the arrow heads. Default is 1.0.
    scale : float, optional
        Scale factor applied to every vector. Default is 1.0.
    line_width : float, optional
        Width of the vector segments. Default is 0.01.
    name : str, optional
        A name of the object. Default is None.
    group : str, optional
        A name of a group. Default is None.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.
    compression_level : int, optional
        Level of compression [-1, 9]. Default is 0.
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    VectorField
        The created VectorField object.
    """
    if colors is None:
        colors = []

    return process_transform_arguments(
        VectorField(
            vectors=vectors,
            colors=colors,
            use_head=use_head,
            head_size=head_size,
            line_width=line_width,
            head_color=head_color if head_color is not None else color,
            origin_color=origin_color if origin_color is not None else color,
            scale=scale,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs,
    )


def vectors(
        origins: ArrayLike,
        vectors: Optional[ArrayLike] = None,
        colors: TypingList[int] = None,
        origin_color: Optional[int] = None,
        head_color: Optional[int] = None,
        color: int = _default_color,
        use_head: bool = True,
        head_size: float = 1.0,
        labels: TypingList[str] = None,
        label_size: float = 1.0,
        line_width: float = 0.01,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any,
) -> Vectors:
    """
    Create a Vectors drawable for plotting arrows from explicit origins.

    Parameters
    ----------
    origins : array_like
        Array of (x, y, z) coordinates the vectors start at.
    vectors : array_like, optional
        Array of (dx, dy, dz) components, one per origin. Default is None.
    colors : array_like, optional
        Twice the length of vectors array of int: packed RGB colors (0xff0000 is red, 0xff is
        blue). The array has consecutive pairs (origin_color, head_color) for vectors in row-major
        order. Default is None.
    origin_color : int, optional
        Packed RGB color of the origins (0xff0000 is red, 0xff is blue), default: same as color.
        Default is None.
    head_color : int, optional
        Packed RGB color of the vector heads (0xff0000 is red, 0xff is blue), default: same as
        color. Default is None.
    color : int, optional
        Packed RGB color of the vectors (0xff0000 is red, 0xff is blue), used for whichever of
        `origin_color` and `head_color` is not given. Default is 255.
    use_head : bool, optional
        Whether vectors should display an arrow head. Default is True.
    head_size : float, optional
        The size of the arrow heads. Default is 1.0.
    labels : list, optional
        Array of strings displayed at the middle of each vector. Default is None.
    label_size : float, optional
        Font size of the labels in em HTML units. Default is 1.0.
    line_width : float, optional
        Width of the vector segments. Default is 0.01.
    name : str, optional
        A name of the object. Default is None.
    group : str, optional
        A name of a group. Default is None.
    custom_data : dict, optional
        An object with custom data attached to object. Default is None.
    compression_level : int, optional
        Level of compression [-1, 9]. Default is 0.
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    Vectors
        The created Vectors object.
    """
    if colors is None:
        colors = []
    if labels is None:
        labels = []

    return process_transform_arguments(
        Vectors(
            vectors=vectors if vectors is not None else origins,
            origins=origins
            if vectors is not None
            else np.zeros_like(origins, dtype=np.float32),
            colors=colors,
            origin_color=origin_color if origin_color is not None else color,
            head_color=head_color if head_color is not None else color,
            use_head=use_head,
            head_size=head_size,
            labels=labels,
            label_size=label_size,
            line_width=line_width,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs,
    )
