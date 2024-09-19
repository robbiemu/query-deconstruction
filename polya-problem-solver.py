from typing import Dict, TypedDict, Annotated
from langgraph.graph import Graph, StateType
from langgraph.prebuilt import LanguageModelChat

# Define our state
class ProblemState(TypedDict):
    problem: str
    understanding: str
    plan: str
    solution: str
    reflection: str
    complete: bool

# Initialize our language model
llm = LanguageModelChat()

# Define our nodes
def understand_problem(state: ProblemState) -> ProblemState:
    response = llm.invoke(
        f"Given the problem: '{state['problem']}', provide a clear understanding of what the problem is asking. Include any key information, constraints, or goals."
    )
    state["understanding"] = response
    return state

def devise_plan(state: ProblemState) -> ProblemState:
    response = llm.invoke(
        f"Based on the understanding: '{state['understanding']}', devise a step-by-step plan to solve the problem. Be specific about the approach and methods you'll use."
    )
    state["plan"] = response
    return state

def carry_out_plan(state: ProblemState) -> ProblemState:
    response = llm.invoke(
        f"Following the plan: '{state['plan']}', solve the problem step by step. Show your work and explain each step."
    )
    state["solution"] = response
    return state

def reflect(state: ProblemState) -> ProblemState:
    response = llm.invoke(
        f"Reflect on the solution: '{state['solution']}'. Is it correct? Can it be improved? Are there alternative approaches? What did we learn from this problem?"
    )
    state["reflection"] = response
    state["complete"] = True
    return state

# Define our graph
workflow = Graph()

# Add nodes to the graph
workflow.add_node("understand", understand_problem)
workflow.add_node("plan", devise_plan)
workflow.add_node("solve", carry_out_plan)
workflow.add_node("reflect", reflect)

# Connect the nodes
workflow.add_edge("understand", "plan")
workflow.add_edge("plan", "solve")
workflow.add_edge("solve", "reflect")

# Set the entrypoint
workflow.set_entry_point("understand")

# Compile the graph
app = workflow.compile()

# Function to run the solver
def solve_problem(problem: str) -> Dict:
    initial_state: ProblemState = {
        "problem": problem,
        "understanding": "",
        "plan": "",
        "solution": "",
        "reflection": "",
        "complete": False
    }
    for state in app.stream(initial_state):
        if state["complete"]:
            return state
    return state  # This line should never be reached

# Example usage
problem = "Find the area of a circle with radius 5 cm."
solution = solve_problem(problem)

print("Problem:", solution["problem"])
print("\nUnderstanding:", solution["understanding"])
print("\nPlan:", solution["plan"])
print("\nSolution:", solution["solution"])
print("\nReflection:", solution["reflection"])
