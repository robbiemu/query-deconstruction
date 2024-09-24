from typing import TypedDict, Sequence, List, Dict, Callable
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, \
    SystemMessage
from langchain_core.language_models import BaseChatModel
import yaml

"""
Polya's How to solve it

Agent: ProblemSolver

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

// Node 3: Carrying Out the Plan
Node CarryOutPlan:
    Input: selectedStrategy
    Process:
        Implement the strategy step by step.
        Monitor progress and verify each step.
        Adjust the plan as needed when encountering obstacles.
    Output: solutionAttempt, resultStatus (success or failure)
    NextNode: ReflectOnSolution

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

// Node 5: End of Process
Node Terminate:
    Input: finalSolution
    Process:
        Present the final solution to the problem.
"""


type Messages = Sequence[AnyMessage]

class Rephrasal(TypedDict):
    original: str
    new: str

class Goals(TypedDict):
    goals: List[str]

class Information(TypedDict):
    information: List[str]

class Terms(TypedDict):
    doubts: List[str]
    term_definitions: Dict[str, str]

class Understanding(TypedDict):
    rephrasal: Rephrasal
    goals: Goals
    information: Information
    terms: Terms

class State(TypedDict):
    problem: str
    understanding: Understanding
    plan: str
    execution: str
    reflection: str
    messages: Messages


def understanding_to_message(understanding: Understanding) -> AIMessage:
    content = yaml.dump(understanding, sort_keys=False, default_flow_style=False)
    return AIMessage(content=content, label="context")


class ProblemUnderstanding():
    rephrase_problem_system_message_content = """You are tasked with rephrasing problems to ensure a deep understanding before solving them. When presented with a problem, follow these steps:

    Restate the problem in clear and simple terms, breaking down complex language or ideas.
    Focus on the core elements and conditions of the problem.
    Ensure that the rephrased version is easy to understand and aligns with the original intent of the problem.
    Ask yourself: What is the problem really asking for? What conditions must be met?
    Reflect on any ambiguities in the problem statement and clarify them in your rephrasing.
    Your goal is to ensure that the problem is fully understood before attempting any solution."""
    identify_goals_system_message_content = """You are tasked with identifying the goals of a problem to clarify what needs to be achieved. When presented with a problem, follow these steps:

Clearly state the primary objective or outcome that the problem is asking to achieve.
Break down any sub-goals or intermediate steps necessary to reach the main goal.
Distinguish between what is known (facts, conditions) and what is unknown (what must be solved or determined).
Ensure the goals are specific and measurable, focusing on what success looks like in the context of the problem.
Consider whether there may be multiple possible goals or solutions, and clarify each one if applicable.
Your objective is to make the goals explicit and clear so that the path toward solving the problem is well-defined."""
    list_information_system_message_content = """Your task is to make a detailed list of all the information provided in the problem. Follow these steps:

Identify and list all the given data points (numbers, conditions, facts, relationships) explicitly stated in the problem.
Note any constraints or limitations that must be followed (e.g., boundaries, rules, assumptions).
Organize this information clearly, categorizing relevant data and constraints separately if needed.
Avoid interpreting or solving at this stage; focus purely on collecting all the information provided.
Ensure that no key information is missed, and all relevant details are listed comprehensively.
Your objective is to create a thorough inventory of all the given information to form a strong foundation for problem-solving."""
    define_terms_system_message_content="""Your task is to clarify any ambiguous terms or concepts in the problem by engaging the user. Follow these steps:

Identify any terms or concepts in the problem that could have multiple meanings or that may not be immediately clear.
If an ambiguous term is detected, ask the user for clarification or more information about how they interpret the term.
Provide suggestions or examples to help guide the user toward a clear understanding of the term, if needed.
Your objective is to make sure all key terms and concepts are clearly defined to avoid confusion during problem-solving."""

    def __init__(
            self, 
            llm: BaseChatModel, 
            human_feedback: Callable[[Terms], Terms]
    ) -> None:
        llm.bind_tools([self.rephrase_problem])
        self.llm = llm
        self.human_feedback = human_feedback

    def _prepare_messages_for_tool_call(self, content: str, conversation: Messages, 
                                        context:str) -> Messages:
        sys = SystemMessage(content)
        messages = []
        messages.append(sys)
        if context:
            messages.append(AIMessage(content=context, label="context"))

        etc = [msg for msg in conversation
                        if not isinstance(msg, SystemMessage)]
        if isinstance(etc[-1], AIMessage):
            etc[-1] = HumanMessage(content=messages[-1], source="llm")

        messages += etc

        return messages

    def _rephrase_problem(self, state: State) -> Rephrasal:
        """ Rephrase the Problem: Encourage rephrasing the problem in one's own words to ensure deep understanding. """
        messages = self._prepare_messages_for_tool_call(
            content = self.rephrase_problem_system_message_content, 
            conversation = state["messages"],
        )

        self.understand.rephrasal = self.llm.with_structured_output(Rephrasal).invoke(messages)
        return self.understand.rephrasal
    
    def _identify_goals(self, state: State) -> Goals:
        """ Identify Goals: Explicitly state what needs to be achieved. """
        messages = self._prepare_messages_for_tool_call(
            content = self.identify_goals_system_message_content, 
            conversation = state["messages"],
            context = understanding_to_message(state["understanding"])
        )

        return self.llm.with_structured_output(Goals).invoke({
            "messages": messages, 
            "rephrasal": self.understanding.rephrasal 
        })
    
    def _list_information(self, state: State) -> Information:
        """ List Information: Make a detailed list of all given data and constraints. """
        messages = self._prepare_messages_for_tool_call(
            content = self.list_information_system_message_content, 
            conversation = state["messages"],
            context = understanding_to_message(state["understanding"])
        )

        return self.llm.with_structured_output(Information).invoke(messages)
    
    def _define_terms(self, state: State) -> Terms:
        messages = self._prepare_messages_for_tool_call(
            content = self.identify_doubts_system_message_content, 
            conversation = state["messages"],
            context = understanding_to_message(state["understanding"])
        )
        
        terms = self.llm.with_structured_output(Terms).invoke(messages)

        messages = self._prepare_messages_for_tool_call(
            self.rewrite_terms_system_message_content, messages)
        self.llm.with_structured_output(Information).invoke({"messages": messages, "terms": terms})

    def understand_problem(self, state: State) -> State:
        state["understanding"] = Understanding()
        state["understanding"]["rephrasal"] = self._identify_rephrasals(state)
        state["understanding"]["goals"] = self._identify_goals(state)
        state["understanding"]["doubts"] = self._identify_doubts(state)
        state["understanding"]["terms"] = self._identify_terms(state)

        return state
