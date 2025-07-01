from typing import Literal

__all__ = ["get_protocol", "switch_to_text_protocol", "switch_to_binary_protocol"]

_protocol: Literal['text', 'binary'] = 'binary'


def switch_to_text_protocol() -> None:
    global _protocol

    _protocol = 'text'


def switch_to_binary_protocol() -> None:
    global _protocol

    _protocol = 'binary'


def get_protocol() -> Literal['text', 'binary']:
    global _protocol

    return _protocol
