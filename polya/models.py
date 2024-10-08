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
    description: str
    evaluation: str
    outcome: str

class Plan(TypedDict, total=False):
    strategies: Sequence[Strategy]
    selected_strategy: Strategy
    plan_for_obstacles: str

# node 3   
class AdjustedStrategy(TypedDict, total=False):
    is_adjusted: bool
    recommendation_from_plan_for_obstacles: str
    previous_progress: str
    original: str
    description: str

class ExecutionSummary(TypedDict, total=False):
    summary: str

class StepAction(TypedDict, total=False):
    action: str

class StepResult(TypedDict, total=False):
    result: str

class Step(TypedDict, total=False):
    action: str
    result: str
    is_verified: bool

class StrategySteps(TypedDict, total=False):
    steps: Sequence[str]

class Execution(TypedDict, total=False):
    steps: Sequence[Step]
    should_change_strategy: bool
    previous_adjustments: Sequence[AdjustedStrategy]
    result: str

# node 4
class Summary(TypedDict, total=False):
    summary_of_work: str

class Analysis(TypedDict, total=False):
    analysis: str

class Reflections(TypedDict, total=False):
    select_strategy: bool
    summary_of_work: str
    solution: str

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
        case "AdjustedStrategy":
            return AdjustedStrategy
        case "ExecutionSummary":
            return ExecutionSummary
        case "StrategySteps":
            return StrategySteps
        case "Step":
            return Step
        case "StepAction":
            return StepAction
        case "StepResult":
            return StepResult
        case "Execution":
            return Execution
        case "Summary":
            return Summary
        case "Analysis":
            return Analysis
        case "Reflection":
            return Reflections
        case _:
            raise ValueError("Invalid type")

# overall
class State(BaseModel):
    problem: Optional[str] = None
    understanding: Optional[Understanding] = None
    plan: Optional[Plan] = None
    execution: Optional[Execution] = None
    reflection: Optional[str] = None
    messages: Messages
