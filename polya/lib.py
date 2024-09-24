from typing import TypedDict
from langchain_core.messages import AIMessage
import yaml

from models import Understanding


class MyTypedDict(TypedDict):
    __module__ = 'typing'
    __origin__ = TypedDict

def typeddict_from_dict(d: dict) -> type:
    new_type = type('MyTypedDict', (MyTypedDict,), d)
    return new_type

def understanding_to_message(understanding: Understanding) -> AIMessage:
    content = yaml.dump(understanding, sort_keys=False, default_flow_style=False)
    return AIMessage(content=content, label="context")
