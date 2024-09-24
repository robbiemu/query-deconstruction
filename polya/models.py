from typing import Dict, List, Optional, Sequence, TypedDict
from pydantic import BaseModel
from langchain_core.messages import AnyMessage


type Messages = Sequence[AnyMessage]

class Terms(TypedDict, total=False):
    doubts: List[str]
    term_definitions: Dict[str, str]

class Understanding(TypedDict, total=False):
    rephrasal: str
    goals: List[str]
    information: List[str]
    terms: Terms

class State(BaseModel):
    problem: Optional[str] = None
    understanding: Optional[Understanding] = None
    plan: Optional[str] = None
    execution: Optional[str] = None
    reflection: Optional[str] = None
    messages: Messages
