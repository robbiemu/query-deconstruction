import yaml

from lib import PolyaNode
from models import Analysis, Reflections, State, Summary
from prompts import config

"""
// Node 4: Reflecting on the Solution
Node ReflectOnSolution:
    Input: solutionAttempt, resultStatus
    Process:
        Verify that the solution addresses the problem fully.
        Analyze the effectiveness of the strategy and execution.
        If unsuccessful, determine reasons and consider alternative strategies.
        If successful, consider how the approach can be applied elsewhere.
    DecisionBranches:
        if resultStatus == success:
            Output: finalSolution
            NextNode: Terminate (End the process)
        else:
            Update parsedProblemData with new insights.
            RedirectTo: DevisePlan
"""


class Reflection(PolyaNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _verify_solution(self, state: State) -> bool:
        """Verify (and log) that the strategy found a solution."""
        description = state.plan['selected_strategy']['original'] \
            if 'original' in state.plan['selected_strategy'] \
                else state.plan['selected_strategy']['description']
        for strategy in state.plan['strategies']:
            if strategy['description'] == description:
                    strategy['outcome'] = state.execution['result']
                    break

        return not state.execution['should_change_strategy']

    def _analyze_effectiveness(self, state: State) -> bool:
        """Analze the effectiveness of the strategy and execution."""
        print('\n--- #[internal step] Analyzing effectiveness ...')
        response: Analysis = self._default_step(
            template="Analysis",
            system=config["reflection"][self.prompt_key]["system_prompt"],
            messages=state.messages, 
            prompts=[
                config["reflection"][self.prompt_key]["analyze_effectiveness"], 
                None])
        
        return response["analysis"]

    def _select_next_course(self, state: State) -> bool:
        """If unsuccessful, determine reasons and consider alternative strategies."""
        state.plan['selected_strategy']['outcome'] = state.execution['result']

        for strategy in state.plan['strategies']:
            if 'outcome' in strategy: # outcome was just set during _verify_solution
                continue
            return True
        return False

    def _summarize_work(self, state: State) -> str:
        """If successful, summarize the work and consider how the approach can be applied elsewhere"""
        print('\n--- #[internal step] Summarizing work ...')
        response: Summary = self._default_step(
            template="Summary",
            system=config["reflection"][self.prompt_key]["system_prompt"],
            messages=state.messages, 
            prompts=[
                config["reflection"][self.prompt_key]["summarize_work"], 
                None])
        
        return response["summary_of_work"]
    
    def reflect_on_solution(self, state: State) -> State:
        should_keep_strategy = self._verify_solution(state) \
            and self._analyze_effectiveness(state)
        if should_keep_strategy:
            summary = self._summarize_work(state)
            return {
                "reflections": { 
                    'summary_of_work' : summary, 
                    'solution' : state.execution['result'] 
                }}
        
        if not self._select_next_course(state):
            # if we have no solution and no next strategy
            summary = self._summarize_work(state)
            return { "reflection": { 'summary_of_work' : summary }}
