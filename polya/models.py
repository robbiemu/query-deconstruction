from typing import Dict, List, Optional, Sequence, TypedDict
from pydantic import BaseModel
from langchain_core.messages import AnyMessage


type Messages = Sequence[AnyMessage]

# node 1
class Rephrasal(BaseModel):
    rephrasal: str

class Goals(BaseModel):
    goals: List[str]

class information(BaseModel):
    information: List[str]

class Terms(BaseModel):
    doubts: Optional[List[str]] = None
    term_definitions: Optional[Dict[str, str]] = None

class Understanding(BaseModel):
    rephrasal: Optional[str] = None
    goals: Optional[List[str]] = None
    information: Optional[List[str]] = None
    terms: Optional[Terms] = None

# node 2
class Strategy(BaseModel):
    strategy: Optional[str] = None
    evaluation: Optional[str] = None
    tried: Optional[bool] = None

class Plan(BaseModel):
    strategies: Optional[Sequence[Strategy]] = None
    selected_strategy: Optional[Strategy] = None
    plan_for_obstacles: Optional[str] = None
    messages: Optional[Messages] = None


def get_type(type:str):
    match type:
        case "Rephrasal":
            return Rephrasal
        case "Goals":
            return Goals
        case "Information":
            return information
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
    understanding: Optional[Understanding] = None
    plan: Optional[str] = None
    execution: Optional[str] = None
    reflection: Optional[str] = None
    messages: Messages
