"""Factory functions for text and label objects."""

from ..objects import Text, Text2d, Label, TextureText
from ..transform import process_transform_arguments
from .common import _default_color


def text(
        text,
        position=[0, 0, 0],
        color=_default_color,
        reference_point="lb",
        on_top=True,
        size=1.0,
        label_box=True,
        is_html=False,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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
        text,
        position=(0, 0),
        color=_default_color,
        size=1.0,
        reference_point="lt",
        label_box=True,
        is_html=False,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
):
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
        text,
        position=(0, 0, 0),
        color=_default_color,
        on_top=True,
        size=1.0,
        max_length=0.8,
        mode="dynamic",
        is_html=False,
        label_box=True,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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
        text,
        position=(0, 0, 0),
        color=_default_color,
        font_weight=400,
        font_face="Courier New",
        font_size=68,
        size=1.0,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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