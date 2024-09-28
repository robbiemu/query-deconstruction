from typing import Dict, List, Optional, Sequence, TypedDict
from pydantic import BaseModel
from langchain_core.messages import AnyMessage


type Messages = Sequence[AnyMessage]

# node 1
class Rephrasal(TypedDict, total=False):
    rephrasal: str

class Goals(TypedDict, total=False):
    goals: List[str]

class Information(TypedDict, total=False):
    information: List[str]

class Doubts(TypedDict, total=False):
    doubts: List[str]

class Terms(TypedDict, total=False):
    doubts: List[str]
    term_definitions: Dict[str, str]

class Understanding(TypedDict, total=False):
    rephrasal: str
    goals: List[str]
    information: List[str]
    terms: Terms

# node 2
class Strategy(TypedDict, total=False):
    strategy: str
    evaluation: str
    tried: bool

class Plan(TypedDict, total=False):
    strategies: Sequence[Strategy]
    selected_strategy: Strategy
    plan_for_obstacles: str

def get_type(type:str):
    match type:
        case "Rephrasal":
            return Rephrasal
        case "Goals":
            return Goals
        case "Information":
            return Information
        case "Doubts":
            return Doubts
        case "Terms":
            return Terms
        case "Understanding":
            return Understanding
        case "Strategy":
            return Strategy
        case "Plan":
            return Plan
        case _:
            raise ValueError("Invalid type")

# overall
class State(BaseModel):
    problem: Optional[str] = None
    understanding: Optional[Understanding] = None
    plan: Optional[str] = None
    execution: Optional[str] = None
    reflection: Optional[str] = None
    messages: Messages
