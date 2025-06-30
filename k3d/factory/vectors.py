"""Factory functions for vector and vector field objects."""

import numpy as np
from ..objects import VectorField, Vectors
from ..transform import process_transform_arguments
from .common import _default_color


def vector_field(
        vectors,
        colors=[],
        origin_color=None,
        head_color=None,
        color=_default_color,
        use_head=True,
        head_size=1.0,
        scale=1.0,
        line_width=0.01,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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
        **kwargs
    )


def vectors(
        origins,
        vectors=None,
        colors=[],
        origin_color=None,
        head_color=None,
        color=_default_color,
        use_head=True,
        head_size=1.0,
        labels=[],
        label_size=1.0,
        line_width=0.01,
        name=None,
        group=None,
        custom_data=None,
        compression_level=0,
        **kwargs
):
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
        **kwargs
    ) 