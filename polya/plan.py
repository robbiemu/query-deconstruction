from typing import Optional, TypedDict, Sequence, Dict, Callable
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, \
    SystemMessage
from langchain_core.language_models import BaseChatModel
import yaml

from lib import Messages

"""
// Node 2: Devising a Plan
Node DevisePlan:
    Input: parsedProblemData
    Process:
        Generate multiple potential strategies.
        Evaluate the feasibility and potential effectiveness of each.
        Select the most promising strategy.
        Plan for potential obstacles.
    Output: selectedStrategy
    NextNode: CarryOutPlan
"""


class Strategy(TypedDict):
    strategy: str
    plan: Optional[str]
    evaluation: Optional[str]
    tried: Optional[bool]

class Plan(BaseModel):
    strategies: Sequence[Strategy]
    evaluation: str
    strategy: Strategy
    messages: Messages

class DevicePlan():
    pass