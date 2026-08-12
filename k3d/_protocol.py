import sys

__all__ = ["get_protocol", "switch_to_text_protocol", "switch_to_binary_protocol"]

if sys.version_info >= (3, 8):
    from typing import Literal

    ProtocolName = Literal["text", "binary"]
else:
    ProtocolName = str

_protocol: ProtocolName = "binary"


def switch_to_text_protocol() -> None:
    global _protocol

    _protocol = "text"


def switch_to_binary_protocol() -> None:
    global _protocol

    _protocol = "binary"


def get_protocol() -> ProtocolName:
    global _protocol

    return _protocol
