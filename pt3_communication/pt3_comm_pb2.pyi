from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Array_message(_message.Message):
    __slots__ = ("datatype", "width", "height", "bytesarray")
    DATATYPE_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    BYTESARRAY_FIELD_NUMBER: _ClassVar[int]
    datatype: str
    width: int
    height: int
    bytesarray: bytes
    def __init__(self, datatype: _Optional[str] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., bytesarray: _Optional[bytes] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
