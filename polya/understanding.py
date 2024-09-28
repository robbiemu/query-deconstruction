from typing import List, Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
import math
import yaml

from lib import PolyaNode
from models import Doubts, State, Terms, Understanding
from prompts import config

"""
// Node 1: Understanding the Problem
Node UnderstandProblem:
    Input: problemDescription
    Process:
        Rephrase the problem in own words.
        Identify the goal and what is being asked.
        List all knowns, unknowns, and constraints.
        Define any ambiguous terms.
    Output: parsedProblemData
    NextNode: DevisePlan
"""


def understanding_to_message(understanding: Understanding) -> AIMessage:
    content = yaml.dump(understanding, sort_keys=False, default_flow_style=False)
    return AIMessage(content=content, label="context")

class ProblemUnderstanding(PolyaNode):
    def __init__(
            self, 
            prompt_key: str,
            llm: BaseChatModel, 
            human_feedback: Callable[[Terms], Terms]
    ) -> None:
        super().__init__(prompt_key=prompt_key, llm=llm)
        self.human_feedback = human_feedback

    def _rephrase_problem(self, state: State, review: bool) -> str:
        """ Rephrase the Problem: Encourage rephrasing the problem in one's own words to ensure deep understanding. """
        print("\n--- #[internal step] Rephrasing the problem...\n")
        response = self._default_step(
            template="Rephrasal",
            messages=state["messages"], 
            context=understanding_to_message(state["understanding"]) 
                if review else None, 
            system = config["understanding"][self.prompt_key]["system_prompt"],
            prompts=[
                config["understanding"][self.prompt_key]["rephrase_problem"], 
                config["understanding"][self.prompt_key]["revise_problem_statement"]], 
            review=review)
        
        return response["rephrasal"]
    
    def _identify_goals(self, state: State, review: bool) -> List[str]:
        """ Identify Goals: Explicitly state what needs to be achieved. """
        print("\n--- #[internal step] Identifying goals...")
        response = self._default_step(
            template="Goals",
            messages=state["messages"], 
            context=understanding_to_message(state["understanding"]), 
            system = config["understanding"][self.prompt_key]["system_prompt"],
            prompts=[
                config["understanding"][self.prompt_key]["identify_goals"], 
                config["understanding"][self.prompt_key]["revise_goals"]], 
            review=review)

        return response["goals"]
    
    def _list_information(self, state: State, review: bool) -> List[str]:
        """ List Information: Make a detailed list of all given data and constraints. """
        print("\n--- #[internal step] Listing information..." )
        response = self._default_step(
            template="Information",
            messages=state["messages"], 
            context=understanding_to_message(state["understanding"]), 
            system = config["understanding"][self.prompt_key]["system_prompt"],
            prompts=[
                config["understanding"][self.prompt_key]["list_information"], 
                config["understanding"][self.prompt_key]["revise_information"]], 
            review=review)

        return response["information"]
    
    def _define_terms(self, state: State, review: bool, 
                      recursion_limit: int = 2) -> Terms:
        """Define any ambiguous terms"""
        print("\n--- #[internal step] Defining terms...")
        # find unknowns

        prompt_key = "identify_doubts" if not review \
            else "identify_further_doubts"
        prompt = config["understanding"][self.prompt_key][prompt_key]

        messages = self._prepare_messages_for_tool_call(
            system = config["understanding"][self.prompt_key]["system_prompt"],
            prompt = prompt, 
            conversation = state["messages"],
            context = understanding_to_message(state["understanding"]))

        for m in messages:
            m.pretty_print()

        doubts = self.llm.with_structured_output(Doubts).invoke(messages)
        if doubts is None:
            return
        
        terms = {"term_definitions": {}}
        terms["doubts"] = doubts.get("doubts")
        if self.human_feedback is None:
            return terms

        understanding = state["understanding"].copy()
        understanding["terms"] = terms
        
        print("doubts\n",len(understanding["terms"]["doubts"]))
        # while we have unknowns
        while self.human_feedback and recursion_limit > 0 and terms["doubts"]:
            recursion_limit -= 1
            print("asking user to clarify...\n", yaml.dump(terms))
            # request user to define unkowns
            definitions = self.human_feedback(terms)

            print("definitions", definitions, terms)

            for term, definition in definitions["term_definitions"].items():
                terms["doubts"].remove(term)
                terms["term_definitions"][term] = definition

            print("remaining doubts\n",len(terms["doubts"]))

            # verify no additional terms come up
            messages = self._prepare_messages_for_tool_call(
                system = config["understanding"][self.prompt_key]["system_prompt"],
                prompt = config["understanding"][self.prompt_key]\
                    ["identify_further_doubts"], 
                conversation = state["messages"],
                context = understanding_to_message(understanding))
        
            print("verifying we have no new doubts immediately ...\n", yaml.dump(terms))
            next_terms = self.llm.with_structured_output(Terms)\
                .invoke(messages)
            
            if next_terms is None:
                return

            terms["doubts"].extend([term for term in next_terms["doubts"] 
                                    if term not in terms["term_definitions"]])

            for key in terms["term_definitions"]:
                if key in terms["doubts"]:
                    del terms["doubts"][key]

        return terms

    def understand_problem(
            self, 
            state: State, 
            terms_recursion_limit: int = 2,
            recursion_limit: int = math.inf
    ) -> State:
        working_state = state.model_dump()
        working_state["messages"] = state.messages.copy()
        working_state["understanding"] = Understanding()

        items = 0
        is_revising = True
        while is_revising and recursion_limit:
            recursion_limit -= 1
            working_state["understanding"]["rephrasal"] = \
                self._rephrase_problem(working_state, bool(items))
            working_state["understanding"]["goals"] = \
                self._identify_goals(working_state, bool(items))
            working_state["understanding"]["information"] = \
                self._list_information(working_state, bool(items))
            working_state["understanding"]["terms"] = \
                self._define_terms(working_state, bool(items), 
                                   terms_recursion_limit)
            
            is_revising = len(
                working_state["understanding"]["terms"]["doubts"] or []) > items
            items = len(working_state["understanding"]["terms"]["doubts"] or [])

        return { "understanding": working_state["understanding"] }
