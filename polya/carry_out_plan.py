from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
import math
import json
import yaml

from lib import PolyaNode
from models import AdjustedStrategy, Execution, Step, State
from prompts import config

"""
// Node 3: Carrying Out the Plan
Node CarryOutPlan:
    Input: selectedStrategy
    Process:
        Implement the strategy step by step.
        Monitor progress and verify each step.
        Adjust the plan as needed when encountering obstacles.
    Output: solutionAttempt, resultStatus (success or failure)
    NextNode: ReflectOnSolution
"""


def execution_to_message(execution: Execution) -> AIMessage:
    content = yaml.dump(execution, sort_keys=False, default_flow_style=False)
    return AIMessage(content=content, label="context") \
        if content else None

class CarryOutPlan(PolyaNode):
    def __init__(self, prompt_key: str, llm: BaseChatModel) -> None:
        super().__init__(prompt_key, llm)

    def _convert_plan_to_step_by_step_actions(self, 
                                              state: State) -> List[Step]:
        """Create a plan to implement the strategy step by step."""
        print("\n--- #[internal step] Creating plan...\n")
        instruction = str.format(
            config["carry_out_plan"][self.prompt_key]["convert_plan_to_steps"],
            strategy=state["plan"]["selected_strategy"]["description"])
        response = self._default_step(
            template="StrategySteps",
            system=config["carry_out_plan"][self.prompt_key]["system_prompt"],
            messages=state["messages"], 
            context=execution_to_message(state["execution"]) 
                if state["execution"].values() else None, 
            prompts=[
                instruction, 
                None])
        
        return response["steps"]

    def _do_step(self, step: str, state: State) -> str:
        """Implement the strategy step by step."""
        print(f"\n--- #[internal step] Stepping through plan...")
        instruction = str.format(
            config["carry_out_plan"][self.prompt_key]["do_step"], step=step)
        response = self._default_step(
            template='StepResult',
            system=config["carry_out_plan"][self.prompt_key]["system_prompt"],
            messages=state["messages"], 
            context=execution_to_message(state["execution"]), 
            prompts=[instruction, None])
        
        return response["result"]
    
    def _verify_step(self, step: str, state: State) -> bool:
        """Monitor progress and verify each step."""
        print("\n--- #[internal step] Monitoring progress...\n")
        instruction = str.format(
            config["carry_out_plan"][self.prompt_key]["verify_step"], step=step)
        response = self._default_step(
            template="Step",
            system=config["carry_out_plan"][self.prompt_key]["system_prompt"],
            messages=state["messages"], 
            context=execution_to_message(state["execution"]), 
            prompts=[instruction, None])
        
        return response["is_verified"]

    def _adjust_plan(self, state: State) -> AdjustedStrategy:
        """Adjust the plan as needed when encountering obstacles."""
        print("\n--- #[internal step] Adjusting plan...\n")
        instruction = str.format(
            config["carry_out_plan"][self.prompt_key]["adjust_plan"],
            strategy=state["plan"]["selected_strategy"]["description"])
        response = self._default_step(
            template="AdjustedStrategy",
            system=config["carry_out_plan"][self.prompt_key]["system_prompt"],
            messages=state["messages"], 
            context=execution_to_message(state["execution"]), 
            prompts=[instruction, None])

        return response
    
    def _summary_results(self, state: State) -> str:
        """Summarize the results of carrying out the plan."""
        print("\n--- #[internal step] Summarizing results...\n")
        response = self._default_step(
            template="ExecutionSummary", 
            system=config["carry_out_plan"][self.prompt_key]["system_prompt"],
            messages=state["messages"], 
            context=execution_to_message(state["execution"]), 
            prompts=[
                config["carry_out_plan"][self.prompt_key]["summarize_results"], 
                None])
        
        return response["summary"]

    def carry_out_plan(
            self, 
            state: State, 
            recursion_limit=math.inf, 
            adjustment_limit=math.inf
    ) -> State:
        working_state = state.model_dump()
        working_state["execution"] = Execution()
        working_state["messages"] = state.messages.copy()
        working_state["plan"] = state.plan.copy()

        execution = working_state["execution"]
        plan = working_state["plan"]

        adjustment_counter = 0
        while recursion_limit > 0:
            recursion_limit -= 1

            # Convert plan into step-by-step actions
            steps = self._convert_plan_to_step_by_step_actions(working_state)
            execution["steps"] = [Step(action=action) for action in steps]

            should_restart = False

            for i, action in enumerate(steps):
                # Execute the step
                step_result = self._do_step(action, working_state)
                execution["steps"][i]["result"] = step_result

                # Verify the step
                verification_passed = self._verify_step(
                    yaml.dump(execution["steps"][i]), 
                    working_state)
                execution["steps"][i]["is_verified"] = verification_passed
                if not verification_passed:
                    # Adjust the plan and restart
                    adjustment_counter += 1
                    if adjustment_counter > adjustment_limit:
                        # Exceeded adjustment limit, cannot adjust plan further
                        execution["result"] = config\
                            ["carry_out_plan"]["execution-failed-messages"]\
                                ["impasse"] + f"{json.dumps(plan)}"
                        execution["should_change_strategy"] = True
                        return { "execution": execution, "plan": plan }
                    
                    new_strategy = self._adjust_plan(working_state)
                    if new_strategy["is_adjusted"]:
                        adj = execution["previous_adjustments"] \
                            if "previous_adjustments" in execution else []
                        execution["previous_adjustments"] = adj + [new_strategy]

                        plan["selected_strategy"] = new_strategy

                        should_restart = True
                        execution["steps"] = []
                    break  # Break out of the for loop to restart the while loop

            if should_restart:
                continue  # Restart the while loop with the adjusted plan

            # If all steps completed successfully
            execution["result"], execution["should_change_strategy"] = self._summary_results(working_state)
            break  # Exit the while loop

        if execution["should_change_strategy"] is None:
            # Handle failure due to recursion limit reached
            execution["result"] = config\
                ["carry_out_plan"]["execution-failed-messages"]["recursion"]
            execution["should_change_strategy"] = True

        return { "execution": execution, "plan": plan }

