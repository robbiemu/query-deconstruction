from typing import Sequence, TypedDict
from langchain_core.messages import AnyMessage


type Messages = Sequence[AnyMessage]


class MyTypedDict(TypedDict):
    __module__ = 'typing'
    __origin__ = TypedDict

def typeddict_from_dict(d: dict) -> type:
    new_type = type('MyTypedDict', (MyTypedDict,), d)
    return new_type