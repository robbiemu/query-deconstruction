from typing import TypedDict, Sequence, List, Dict, Callable
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


class Plan(BaseModel):
    messages: Messages

class DevicePlan():