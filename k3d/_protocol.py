import sys

__all__ = ["get_protocol", "switch_to_text_protocol", "switch_to_binary_protocol"]

if sys.version_info >= (3, 8):
    from typing import Literal

    ProtocolName = Literal["text", "binary"]
else:
    ProtocolName = str

_protocol: ProtocolName = "binary"


def switch_to_text_protocol() -> None:
    # Deprecated since 3.0.0: the anywidget transport carries binary buffers in
    # every frontend (Colab included), so the base64 fallback is never needed.
    global _protocol

    import warnings

    warnings.warn(
        "switch_to_text_protocol() is deprecated since 3.0.0 - the binary "
        "protocol works everywhere, including Google Colab.",
        DeprecationWarning,
        stacklevel=2,
    )

    _protocol = "text"


def switch_to_binary_protocol() -> None:
    global _protocol

    _protocol = "binary"


def get_protocol() -> ProtocolName:
    global _protocol

    return _protocol
