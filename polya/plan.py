from typing import List
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import yaml

from lib import PolyaNode
from models import Plan, State, Strategy
from prompts import config

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


def plan_to_message(plan: Plan) -> AIMessage:
    content = yaml.dump(plan, sort_keys=False, default_flow_style=False)
    return AIMessage(content=content, label="context")

class PlanPreparation(PolyaNode):
    def __init__(self, prompt_key: str, llm: BaseChatModel) -> None:
        super().__init__(prompt_key, llm)

    def _ideate_strategies(self, state: State, review: bool) -> List[Strategy]:
        """Generate multiple potential strategies"""
        print("\n--- #[internal step] Ideating strategies...\n")
        response = self._default_step(
            template="Plan",
            system=config["plan"][self.prompt_key]["system_prompt"],
            messages=state["messages"], 
            context=plan_to_message(state["plan"]) 
                if review else None, 
            prompts=[
                config["plan"][self.prompt_key]["generate_strategies"], 
                config["plan"][self.prompt_key]["revise_generated_strategies"]], 
            review=review)
        
        return response["strategies"]

    def _evaluate_strategies(self, state: State, review: bool) -> List[Strategy]:
        """Evaluate the feasibility and potential effectiveness of each"""
        print("\n--- #[internal step] Evaluating strategies...\n")
        response = self._default_step(
            template="Plan",
            system=config["plan"][self.prompt_key]["system_prompt"],
            messages=state["messages"], 
            context=plan_to_message(state["plan"]), 
            prompts=[
                config["plan"][self.prompt_key]["evaluate_strategies"], 
                config["plan"][self.prompt_key]["revise_strategy_evaluations"]], 
            review=review)

        return response["strategies"]

    def _select_strategy(self, state: State, review: bool) -> Strategy:
        """Select the most promising strategy"""
        print("\n--- #[internal step] Selecting strategy...\n")
        response = self._default_step(
            template="Plan",
            system=config["plan"][self.prompt_key]["system_prompt"],
            messages=state["messages"], 
            context=plan_to_message(state["plan"]), 
            prompts=[
                config["plan"][self.prompt_key]["select_strategy"], 
                config["plan"][self.prompt_key]["select_new_strategy"]], 
            review=review)

        return response["selected_strategy"]

    def _plan_for_obstacles(self, state: State, review: bool) -> None:
        """Plan for potential obstacles"""
        print("\n--- #[internal step] Planning for obstacles...\n")
        response = self._default_step(
            system=config["plan"][self.prompt_key]["system_prompt"],
            template="Plan",
            messages=state["messages"], 
            context=plan_to_message(state["plan"]), 
            prompts=[
                config["plan"][self.prompt_key]["plan_for_obstacles"], 
                config["plan"][self.prompt_key]["revise_plan_for_obstacles"]], 
            review=review)

        return response["plan_for_obstacles"]

    def devise_plan(self, state: State, review: bool) -> State:
        working_state = state.model_dump()
        working_state["messages"] = state.messages.copy()
        if not review:
            working_state["plan"] = Plan()

        working_state["plan"]["strategies"] = \
            self._ideate_strategies(working_state, review)
        working_state["plan"]["strategies"] = \
            self._evaluate_strategies(working_state, review)
        working_state["plan"]["selected_strategy"] = \
            self._select_strategy(working_state, review)
        working_state["plan"]["plan_for_obstacles"] = \
            self._plan_for_obstacles(working_state, review)
            
        return { 
            "plan": working_state["plan"] 
        }
