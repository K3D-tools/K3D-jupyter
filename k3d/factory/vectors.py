"""Factory functions for vector and vector field objects."""

import numpy as np
from typing import Any
from typing import Dict as TypingDict
from typing import List as TypingList
from typing import Optional, Tuple, Union

from .common import _default_color
from ..objects import VectorField, Vectors
from ..transform import process_transform_arguments

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
    if colors is None:
        colors = []
    if labels is None:
        labels = []

    return process_transform_arguments(
        Vectors(
            vectors=vectors if vectors is not None else origins,
            origins=origins if vectors is not None else np.zeros_like(vectors),
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
