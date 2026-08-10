import sys

__all__ = ["get_protocol", "switch_to_text_protocol", "switch_to_binary_protocol"]

if sys.version_info >= (3, 8):
    from typing import Literal

    ProtocolName = Literal["text", "binary"]
else:
    # typing.Literal was added in Python 3.8; on 3.7 fall back to plain str so that
    # `import k3d` keeps working. Type checkers run on 3.8+ and still get the narrowing.
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
