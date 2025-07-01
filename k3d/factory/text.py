"""Factory functions for text and label objects."""

from typing import Union, List as TypingList, Optional, Dict as TypingDict, Any, Tuple

from .common import _default_color
from ..objects import Text, Text2d, Label, TextureText
from ..transform import process_transform_arguments

# Type aliases for better readability
ArrayLike = Union[TypingList, Tuple]


def text(
        text: str,
        position: ArrayLike = None,
        color: int = _default_color,
        reference_point: str = "lb",
        on_top: bool = True,
        size: float = 1.0,
        label_box: bool = True,
        is_html: bool = False,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> Text:
    if position is None:
        position = [0, 0, 0]

    return process_transform_arguments(
        Text(
            position=position,
            reference_point=reference_point,
            text=text,
            size=size,
            color=color,
            on_top=on_top,
            is_html=is_html,
            label_box=label_box,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def text2d(
        text: str,
        position: Tuple[float, float] = (0, 0),
        color: int = _default_color,
        size: float = 1.0,
        reference_point: str = "lt",
        label_box: bool = True,
        is_html: bool = False,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
) -> Text2d:
    return Text2d(
        position=position,
        reference_point=reference_point,
        text=text,
        size=size,
        color=color,
        is_html=is_html,
        label_box=label_box,
        name=name,
        group=group,
        custom_data=custom_data,
        compression_level=compression_level,
    )


def label(
        text: str,
        position: Tuple[float, float, float] = (0, 0, 0),
        color: int = _default_color,
        on_top: bool = True,
        size: float = 1.0,
        max_length: float = 0.8,
        mode: str = "dynamic",
        is_html: bool = False,
        label_box: bool = True,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> Label:
    return process_transform_arguments(
        Label(
            position=position,
            text=text,
            size=size,
            color=color,
            on_top=on_top,
            max_length=max_length,
            mode=mode,
            is_html=is_html,
            label_box=label_box,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )


def texture_text(
        text: str,
        position: Tuple[float, float, float] = (0, 0, 0),
        color: int = _default_color,
        font_weight: int = 400,
        font_face: str = "Courier New",
        font_size: int = 68,
        size: float = 1.0,
        name: Optional[str] = None,
        group: Optional[str] = None,
        custom_data: Optional[TypingDict[str, Any]] = None,
        compression_level: int = 0,
        **kwargs: Any
) -> TextureText:
    return process_transform_arguments(
        TextureText(
            text=text,
            position=position,
            color=color,
            size=size,
            font_face=font_face,
            font_size=font_size,
            font_weight=font_weight,
            name=name,
            group=group,
            custom_data=custom_data,
            compression_level=compression_level,
        ),
        **kwargs
    )
