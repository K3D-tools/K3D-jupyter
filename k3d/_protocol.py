__all__ = ["get_protocol", "switch_to_text_protocol", "switch_to_binary_protocol"]

_protocol = 'binary'


def switch_to_text_protocol():
    global _protocol

    _protocol = 'text'


def switch_to_binary_protocol():
    global _protocol

    _protocol = 'binary'


def get_protocol():
    global _protocol

    return _protocol
