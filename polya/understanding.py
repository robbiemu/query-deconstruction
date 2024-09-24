from typing import TypedDict, Optional, List, Dict, Callable
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, \
    SystemMessage
from langchain_core.language_models import BaseChatModel
import yaml

from lib import Messages

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


class Rephrasal(TypedDict):
    new: str

class Goals(TypedDict):
    items: List[str]

class Information(TypedDict):
    items: List[str]

class Terms(TypedDict):
    doubts: List[str]
    term_definitions: Optional[Dict[str, str]]

class Understanding(TypedDict):
    rephrasal: Rephrasal
    goals: Goals
    information: Information
    terms: Terms

class State(BaseModel):
    problem: Optional[str] = None
    understanding: Optional[Understanding] = None
    plan: Optional[str] = None
    execution: Optional[str] = None
    reflection: Optional[str] = None
    messages: Messages

def understanding_to_message(understanding: Understanding) -> AIMessage:
    content = yaml.dump(understanding, sort_keys=False, default_flow_style=False)
    return AIMessage(content=content, label="context")


class ProblemUnderstanding():
    rephrase_problem_system_message_content = """You are tasked with rephrasing problems to ensure a deep understanding before solving them. When presented with a problem, follow these steps:

    Restate the problem in clear and simple terms, breaking down complex language or ideas.
    Focus on the core elements and conditions of the problem.
    Ensure that the rephrased version is easy to understand and aligns with the original intent of the problem. Don't express your doubts here, simply rephrase the question.
    Ask yourself: What is the problem really asking for? What conditions must be met?
    Reflect on any ambiguities in the problem statement and clarify them in your rephrasing.
    Your goal is to ensure that the problem is fully understood before attempting any solution."""
    revise_problem_statement_system_message_content="""You are tasked with ensuring accurate rephrasing of problems. These rephrasals were made to ensure a deep understanding before solving them. When presented with a problem, follow these steps:

    Review the current rephrasal to ensure that it is correct. Pay special attention to the terms as these were ambiguities that have been resolved. If the current rephrasing remains accurate, simply return that. If the rephrasing does not fully capture the problem statement, revise it follwing the remaining steps:

    Restate the problem in clear and simple terms, breaking down complex language or ideas.
    Focus on the core elements and conditions of the problem.
    Ensure that the rephrased version is easy to understand and aligns with the original intent of the problem.
    Ask yourself: What is the problem really asking for? What conditions must be met?
    Reflect on any ambiguities in the problem statement and clarify them in your rephrasing.
    Your goal is to ensure that the problem is fully understood before attempting any solution."""
    identify_goals_system_message_content = """You are tasked with identifying the goals (each independent goal is considered an item) to guide a problem solver in solving the problem. These goals should clarify what needs to be achieved. When presented with a problem, follow these steps:

Clearly state the primary objective or outcome that the problem is asking to achieve.
Break down any sub-goals or intermediate steps necessary to reach the main goal.
Distinguish between what is known (facts, conditions) and what is unknown (what must be solved or determined).
Ensure the goals are specific and measurable, focusing on what success looks like in the context of the problem.
Consider whether there may be multiple possible goals or solutions, and clarify each one if applicable.
Your objective is to make the goals explicit and clear so that the path toward solving the problem is well-defined."""
    revise_goals_system_message_content = """You are tasked with revising a list of goals (each independent goal is considered an item) to ensure that they are relevant, clear and concise. Pay special attention to the terms as these were ambiguities that have been resolved.
    
When presented with a problem, follow these steps:

Start by identifying the goals that were not effected by ambiguity in the newly defined terms: simply preserve these in your response unless they no longer make sense, otherwise omit them. For the ones that mentioned the terms, edit them as necessary to ensure that they are accurate. If new goals need to be added, add them following these steps:
Clearly state the primary objective or outcome that the problem is asking to achieve.
Break down any sub-goals or intermediate steps necessary to reach the main goal.
Distinguish between what is known (facts, conditions) and what is unknown (what must be solved or determined).
Ensure the goals are specific and measurable, focusing on what success looks like in the context of the problem.
Consider whether there may be multiple possible goals or solutions, and clarify each one if applicable.
Your objective is to make the goals explicit and clear so that the path toward solving the problem is well-defined."""
    list_information_system_message_content = """Your task is to make a detailed list of all the information provided in the problem. Do not list the goals themselves in this step! Follow these steps:

1. Identify and list all the given data points (numbers, conditions, facts, relationships) explicitly stated in the problem.
2. Note any constraints or limitations that must be followed (e.g., boundaries, rules, assumptions).
3. Organize this information clearly, categorizing relevant data and constraints separately if needed. Each item of information should be its own entry.
4. Avoid interpreting or solving at this stage; focus purely on collecting all the information provided.
Ensure that no key information is missed, and all relevant details are listed comprehensively.
Your objective is to create a thorough inventory of all the given information to form a strong foundation for problem-solving."""
    revise_information_system_message_content = """You are tasked with ensuring the accuracy of the list of inforation about a given problem. The information was made to ensure a deep understanding before solving them. Pay special attention to the terms as these were ambiguities that have been resolved, and of course to the information items themselves. Correct or if more approapriate remove any information that was inaccurate because of a misunderstanding of a term."""
    identify_doubts_system_message_content = """Your task is to carefully examine the problem for any terms or concepts that could have multiple meanings or that may not be immediately clear. Follow these steps:

1. Identify any terms or concepts in the problem that could lead to confusion due to multiple possible interpretations or lack of clarity.
2. List each ambiguous term or concept without attempting to resolve it yet. Focus only on detecting potential issues with understanding the problem.
3. Ensure that each term or concept identified is sufficiently distinct that the user will not sense repetition in the questions.
Do not proceed to explanations or clarifications at this stage—simply highlight the problematic terms or concepts."""
    identify_further_doubts_system_message_content = """Your task is to identify any further doubts or ambiguous terms that may have emerged after the initial clarifications. Pay special attention to the term definitions as these were ambiguities that have been resolved. But only modify the doubts section. Follow these steps:

1. Re-examine the problem, along with the newly defined terms, to identify any further terms or concepts that could still cause confusion.
2. Highlight any new terms or concepts that might have multiple interpretations or are not fully understood.
3. List these further ambiguous terms for clarification, but do not resolve them in this step. Ensure that each new term or concept identified is sufficiently distinct that the user will not sense repetition in the questions.
4. Ensure that all key terms and concepts are fully understood before proceeding.
Do not proceed to explanations or clarifications at this stage—simply highlight the problematic terms or concepts."""


    def __init__(
            self, 
            llm: BaseChatModel, 
            human_feedback: Callable[[Terms], Terms]
    ) -> None:
        self.llm = llm
        self.human_feedback = human_feedback

    def _prepare_messages_for_tool_call(
            self, content: str, 
            conversation: Messages, 
            context: AIMessage = None
    ) -> Messages:
        sys = SystemMessage(content)
        messages = []
        messages.append(sys)
        if context:
            messages.append(context)

        etc = [msg for msg in conversation
                        if not isinstance(msg, SystemMessage)]
        if isinstance(etc[-1], AIMessage):
            etc[-1] = HumanMessage(content=messages[-1], source="llm")

        messages += etc

        return messages

    def _rephrase_problem(self, state: State, review: bool) -> Rephrasal:
        """ Rephrase the Problem: Encourage rephrasing the problem in one's own words to ensure deep understanding. """
        AIMessage(content="Rephrasing the problem...\n" + yaml.dump(state["understanding"])).pretty_print()
        if not review:
            messages = self._prepare_messages_for_tool_call(
            self.rephrase_problem_system_message_content, 
            state["messages"])
        else:
            messages = self._prepare_messages_for_tool_call(
                content = self.revise_problem_statement_system_message_content, 
                conversation = state["messages"],
                context = understanding_to_message(state["understanding"]))

        return self.llm.with_structured_output(Rephrasal).invoke(messages)
    
    def _identify_goals(self, state: State, review: bool) -> Goals:
        """ Identify Goals: Explicitly state what needs to be achieved. """
        AIMessage(content="Identifying goals...\n" + yaml.dump(state["understanding"])).pretty_print()
        if not review:
            messages = self._prepare_messages_for_tool_call(
                content = self.identify_goals_system_message_content, 
                conversation = state["messages"],
                context = understanding_to_message(state["understanding"]))
        else:
            messages = self._prepare_messages_for_tool_call(
                content = self.revise_goals_system_message_content, 
                conversation = state["messages"],
                context = understanding_to_message(state["understanding"]))

        return self.llm.with_structured_output(Goals).invoke(messages)
    
    def _list_information(self, state: State, review: bool) -> Information:
        """ List Information: Make a detailed list of all given data and constraints. """
        AIMessage(content="Listing information...\n" + yaml.dump(state["understanding"])).pretty_print()
        if not review:
            messages = self._prepare_messages_for_tool_call(
                content = self.list_information_system_message_content, 
                conversation = state["messages"],
                context = understanding_to_message(state["understanding"]))
        else:
            messages = self._prepare_messages_for_tool_call(
                content = self.revise_information_system_message_content, 
                conversation = state["messages"],
                context = understanding_to_message(state["understanding"]))

        return self.llm.with_structured_output(Information).invoke(messages)
    
    def _define_terms(self, state: State, review: bool) -> Terms:
        """Define any ambiguous terms"""
        AIMessage(content="Defining terms...\n" + yaml.dump(state["understanding"])).pretty_print()
        # find unknowns
        if not review:
            messages = self._prepare_messages_for_tool_call(
                content = self.identify_doubts_system_message_content, 
                conversation = state["messages"],
                context = understanding_to_message(state["understanding"]))
        else:
            messages = self._prepare_messages_for_tool_call(
                content = self.identify_further_doubts_system_message_content, 
                conversation = state["messages"],
                context = understanding_to_message(state["understanding"]))
        
        terms = self.llm.with_structured_output(Terms).invoke(messages)
        if terms is None:
            return
        
        understanding = state["understanding"].copy()
        understanding["terms"] = terms
        
        print("doubts\n",len(understanding["terms"]["doubts"]))
        # while we have unknowns
        while terms["doubts"]:
            print("asking user to clarify...\n", yaml.dump(terms))
            # request user to define unkowns
            definitions = self.human_feedback(terms)
            for term, definition in definitions["term_definitions"].items():
                terms["doubts"].remove(term)
                terms["term_definitions"][term] = definition

            print("remaining doubts\n",len(terms["doubts"]))

            # verify no additional terms come up
            messages = self._prepare_messages_for_tool_call(
                content = self.identify_further_doubts_system_message_content, 
                conversation = state["messages"],
                context = understanding_to_message(understanding))
        
            print("verifying we have no new doubts immediately ...\n", yaml.dump(terms))
            next_terms = self.llm.with_structured_output(Terms).invoke(messages)
            terms["doubts"].extend([term for term in next_terms["doubts"] if term not in terms["term_definitions"]])

        return terms

    def understand_problem(self, state: State) -> State:
        working_state = state.model_dump()
        working_state["understanding"] = Understanding()

        items = 0
        is_revising = True
        while is_revising:
            working_state["understanding"]["rephrasal"] = \
                self._rephrase_problem(working_state, not items)
            working_state["problem"] = working_state["understanding"]["rephrasal"]
            working_state["understanding"]["goals"] = \
                self._identify_goals(working_state, not items)
            working_state["understanding"]["information"] = \
                self._list_information(working_state, not items)
            working_state["understanding"]["terms"] = \
                self._define_terms(working_state, not items)
            
            is_revising = len(working_state["understanding"]["terms"]["doubts"] or []) > items
            items = len(working_state["understanding"]["terms"]["doubts"] or [])

        return { 
            "problem": working_state["problem"], 
            "understanding": working_state["understanding"] 
        }
