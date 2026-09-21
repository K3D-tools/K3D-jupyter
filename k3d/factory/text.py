"""Factory functions for text and label objects."""

from typing import Any, Optional, Tuple, Union
from typing import Dict as TypingDict
from typing import List as TypingList

from ..objects import Label, Text, Text2d, TextureText
from ..transform import process_transform_arguments
from .common import _default_color

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
        visible: bool = True,
        **kwargs: Any,
) -> Text:
    """
    Create a Text drawable, rendered with KaTeX at a position in space.

    Parameters
    ----------
    text : str or list of str
        Content of the text.
    position : list, optional
        (x, y, z) coordinates of text position, by default (0, 0, 0). If n text is pass position
        should contain 3*n elements . Default is None.
    color : int, optional
        Packed RGB color of the text (0xff0000 is red, 0xff is blue). Default is 255.
    reference_point : str, optional
        Two-letter string representing the text's alignment. First letter ''l', 'c' or 'r'' left,
        center or right Second letter ''t', 'c' or 'b'' top, center or bottom. Default is 'lb'.
    on_top : bool, optional
        Render order with 3d object Default is True.
    size : float, optional
        Font size in 'em' HTML units. Default is 1.0.
    label_box : bool, optional
        Label background box. Default is True.
    is_html : bool, optional
        Whether text should be interpreted as HTML insted of KaTeX. Default is False.
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
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    Text
        The created Text object.
    """
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
            visible=visible,
        ),
        **kwargs,
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
        visible: bool = True,
) -> Text2d:
    """
    Create a Text2d drawable, rendered with KaTeX at a fixed place on the canvas.

    Parameters
    ----------
    text : str or list of str
        Content of the text.
    position : list, optional
        Ratios (r_x, r_y) of the text's position in range (0, 1) - relative to canvas size. If n
        text is pass position should contain 2*n elements . Default is (0, 0).
    color : int, optional
        Packed RGB color of the text (0xff0000 is red, 0xff is blue). Default is 255.
    size : float, optional
        Font size in 'em' HTML units. Default is 1.0.
    reference_point : str, optional
        Two-letter string representing the text's alignment. First letter ''l', 'c' or 'r'' left,
        center or right Second letter ''t', 'c' or 'b'' top, center or bottom. Default is 'lt'.
    label_box : bool, optional
        Label background box. Default is True.
    is_html : bool, optional
        Whether text should be interpreted as HTML insted of KaTeX. Default is False.
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

    Returns
    -------
    Text2d
        The created Text2d object.
    """
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
        visible=visible,
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
        visible: bool = True,
        **kwargs: Any,
) -> Label:
    """
    Create a Label drawable, a text with a line pointing at a position in space.

    Parameters
    ----------
    text : str or list of str
        Content of the text.
    position : list, optional
        (x, y, z) coordinates of text position, by default (0, 0, 0). If n text is pass position
        should contain 3*n elements . Default is (0, 0, 0).
    color : int, optional
        Packed RGB color of the text (0xff0000 is red, 0xff is blue). Default is 255.
    on_top : bool, optional
        Render order with 3d object Default is True.
    size : float, optional
        Font size in 'em' HTML units. Default is 1.0.
    max_length : float, optional
        Maximum length of the line connecting the label to its position, as a fraction of the
        canvas. Default is 0.8.
    mode : str, optional
        How the label is placed: 'dynamic', 'local' or 'side'. Default is 'dynamic'.
    is_html : bool, optional
        Whether text should be interpreted as HTML insted of KaTeX. Default is False.
    label_box : bool, optional
        Label background box. Default is True.
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
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    Label
        The created Label object.
    """
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
            visible=visible,
        ),
        **kwargs,
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
        visible: bool = True,
        **kwargs: Any,
) -> TextureText:
    """
    Create a TextureText drawable, a text drawn into a texture in the scene.

    Parameters
    ----------
    text : str or list of str
        Content of the text.
    position : list, optional
        (x, y, z) coordinates of text position, by default (0, 0, 0). If n text is pass position
        should contain 3*n elements . Default is (0, 0, 0).
    color : int, optional
        Packed RGB color of the text (0xff0000 is red, 0xff is blue). Default is 255.
    font_weight : int, optional
        Thickness of the characters in HTML-like units from the range (100, 900), where 400 is
        normal and 600 is bold font. Default is 400.
    font_face : str, optional
        Name of the font to use for rendering the text. Default is 'Courier New'.
    font_size : int, optional
        The font size inside the sprite texture in px units. This does not affect the size of the
        text in the scene, only the accuracy and raster size of the texture. Default is 68.
    size : float, optional
        Size of the texture sprite containing the text. Default is 1.0.
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
    **kwargs
        Additional keyword arguments passed to process_transform_arguments.

    Returns
    -------
    TextureText
        The created TextureText object.
    """
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
            visible=visible,
        ),
        **kwargs,
    )
