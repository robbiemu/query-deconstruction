from typing import TypedDict, List, Dict, Callable
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import ChatMessage, AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

"""
# Task Deconstruction

The purpose of the task deconstruction graph is to allow users to deconstruct a 
task into its constituent parts, and then execute them in order. This is useful 
for tasks that are too large or complex to be executed all at once, and can be 
serialized. When results have been gathered they will be compiled into a single 
result and returned.

## Design goals

- **Modular Design**: The code is split into clear components (break_down_task, 
execute_instruction, compile_final_result), following the Single Responsibility
Principle (SRP).
- **Documentation and typing**: The code is meant to be well documented and 
typed to facilitate usage.
"""


class StartState(TypedDict):
    """ the initial input from the user. """
    query: str


class WorkState(TypedDict): 
    """ the intermediate and final states during task execution. """
    query: str
    task_instructions: List[ChatMessage]
    instruction_execution_results: List[str]
    context: List[Dict]
    result: str


""" break_down_task_graph
This graph starts the task deconstruction process. It takes in the query and
returns a List of task instructions representing subproblems, which are later 
executed by the "execute_instruction" node.
"""

class Instructions(TypedDict):
    """ Collector class for the task instructions. """
    task_instructions: List[str]


def factory_break_down_task(llm: BaseChatModel) \
-> CompiledStateGraph:
    """
    A factory function that creates a function using the provided LLM to break 
    down the user query into smaller tasks.
    """
    def break_down_task(state: StartState) -> WorkState:
        # Use an LLM to break down the user query into a list of instructions
        query = state["query"]
        result = llm.with_structured_output(Instructions).invoke(f"Divide this query into smaller tasks so many can work on it together (they may work in serial or in parellel): {query}")
        print("STEP break down task:", query, result)
        
        instructions = [ ChatMessage(content=task, role="instructor")
                        for task in result["task_instructions"] ]
        
        return {"query": query, "task_instructions": instructions}
    
    tasker_graph_builder = StateGraph(WorkState, input=StartState, output=WorkState)
    tasker_graph_builder.add_node("tasker", break_down_task)

    tasker_graph_builder.add_edge(START, "tasker")
    tasker_graph_builder.add_edge("tasker", END)
    
    return tasker_graph_builder.compile()


""" Solver Graph 
The solver graph is meant to be called iteratively in the "execute_instruction" 
node to solve subproblems from the original query.

If you've not optimized the task breakdown, it might result in repeated 
instructions with the same work, producing frequent calls to the LLM. this could
introduce latency. The code may benefit from memoization or caching mechanisms 
if this execute_instruction processes similar instructions repeatedly.
"""

class Task(TypedDict):
    instructions: List[ChatMessage]
    context: List[Dict]
    result: ChatMessage


def factory_solver(llm: BaseChatModel) -> CompiledStateGraph: 
    """
    A factory function that creates a solver node using the provided LLM to 
    execute a single instruction.
    note: A more production ready version of this graph might make use of 
    context, a system message, etc.
    """
    def execute_instruction(state: Task) -> Task:
        print("STEP execute instruction:", state)
        """ executes an instruction using an LLM and returns its result. """
        messages = [ AIMessage(content=instruction.content) \
                    for instruction in state["instructions"][:-1] 
                 ] + [ HumanMessage(content=state["instructions"][-1].content) ]
        results = llm.invoke(messages)
        
        return { "result": results }

    solver_graph_builder = StateGraph(Task) 
    solver_graph_builder.add_node("execute_instruction", execute_instruction) 

    solver_graph_builder.add_edge(START, "execute_instruction") 
    solver_graph_builder.add_edge("execute_instruction", END)

    return solver_graph_builder.compile()


""" Task Scheduler
The task scheduler recieves a List of tasks representing subproblems from 
the original query, and executes them in parallel using the provided solver at 
the "execute_instruction" node.

The user is free to substitute in  custom task scheduler if desired. Examples 
uses would be to pre- or post- process the state or tasks before or after the 
tasks are sent.
"""

def factory_scheduler(solver_graph: CompiledStateGraph) \
-> Callable[[WorkState], WorkState]:
    """ Couple an external solver_graph to the scheduler """

    def scheduler(state: WorkState):
        """ 
        The default scheduler recursively calls the sovler_graph with 
        a compiled list of instructions (including the current task 
        instruction), and context.
        """
        context = []
        instructions = []
        for s in state["task_instructions"]:
            instructions.append(s)
            task = {
                "instructions": instructions,
                "context": context
            }
            print("STEP scheduler:", task)
            result = solver_graph.invoke(task)

            context.append(result["result"])

        return {"instruction_execution_results": [solution.content for solution in context]}
    
    return scheduler


""" Compiler Graph
The compiler graph receives a list of results from the subproblems and combines 
them to produce an answer.
"""

def compile_final_result(state: WorkState):
    # Combine the results from each instruction into a single string.
    print("STEP compiler:", state)

    task_results = "\n".join([ result for result in 
                              state["instruction_execution_results" ]])
    
    return { "result": task_results }

# Default compiler graph
compiler_graph_builder = StateGraph(WorkState)
compiler_graph_builder.add_node("compile_final_result", compile_final_result)
compiler_graph_builder.add_edge(START, "compile_final_result")
compiler_graph_builder.add_edge("compile_final_result", END)
compiler_graph = compiler_graph_builder.compile()


""" High level functions
These functions ease the construction of task deconstruction graphs.
"""

def deconstruct_task(
    break_down_task: Callable[[StartState], Instructions],
    scheduler: Callable[[WorkState], WorkState],
    compiler_graph=compiler_graph, 
) -> CompiledStateGraph:
    """
    creates a LangGraph that breaks down a task into smaller steps, executes each 
    step in its own subgraph, and compiles the final result.
    """ 
    graph_builder = StateGraph(WorkState, input=StartState, output=WorkState) 

    graph_builder.add_node("break_down_task", break_down_task) 
    graph_builder.add_node("execute_task", scheduler) 
    graph_builder.add_node("compile_result", compiler_graph) 

    graph_builder.add_edge(START, "break_down_task") 
    graph_builder.add_edge("break_down_task", "execute_task")
    graph_builder.add_edge("execute_task", "compile_result")
    graph_builder.add_edge("compile_result", END)

    return graph_builder.compile()


def factory_deconstruct_task(llm: BaseChatModel) -> CompiledStateGraph:
    """ Factory method to produce a default task deconstructor """
    break_down_task = factory_break_down_task(llm)
    solver_graph = factory_solver(llm)
    scheduler = factory_scheduler(solver_graph)

    return deconstruct_task(break_down_task, scheduler)


if __name__ == "__main__":
    """ happy path test """
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model = "mistral-nemo:12b-instruct-2407-q8_0",
        temperature = 0.2,
        num_ctx=8192,
        num_predict = 4096,
    )

    graph = factory_deconstruct_task(llm)

    for s in graph.stream({"query": "tell me 20 differnt types of animals"}):
        print(s)
