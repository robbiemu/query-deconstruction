'''
Flow diagram:

```mermaid
graph TD
    A[Problem Description] --> B(Rephrase Problem)
    B --> C(List Information)
    C --> D(Define Terms)
    D --> E(Identify Goals)
    E --> F(Generate Strategies)
    F --> G(Evaluate Strategies)
    G --> H(Select Strategy & Plan for Obstacles)
    H --> I(Carry Out Plan)
    I --> J(Monitor Progress and Verify Steps)
    J --> K[Adjust Plan if Necessary]
    K -->|No Adjustment Needed| L(Summarize Results of Execution)
    K -->|Adjustment Needed| M(select_alternate_strategy)
    M --> N(Select New Strategy Due to Changes in Understanding)
    N --> O(Plan for Obstacles with the New Strategy)
    O --> I
    L --> P(Reflect on Solution's Effectiveness)
    P --> Q{Solution Found?}
    Q -->|Yes| R(Final Solution)
    Q -->|No| S(Update Problem Understanding with Insights)
    S --> F

subgraph UnderstandProblem
    B
    C
    D
    E
end

subgraph DevisePlan
    F
    G
    H
    M(select_alternate_strategy)
    N(Select New Strategy Due to Changes in Understanding)
    O(Plan for Obstacles with the New Strategy)
end

subgraph CarryOutPlan
    I
    J
    K
    L
end

subgraph ReflectOnSolution
    P
    Q
    R(Final Solution)
    S(Update Problem Understanding with Insights)
end
```

Sample flow.

from langchain_ollama import ChatOllama


llm = ChatOllama(
    #model = "mistral-small:22b-instruct-2409-q6_K",
    model = "qwen2.5:32b-instruct-q6_K",
    #model = "finalend/hermes-3-llama-3.1:70b-q3_K_M",
    temperature = 0.666,
    top_p=0.8,
    repeat_penalty = 1.05,
    num_ctx=32678,
    num_predict = 8192,
)
problem = """Five people (A, B, C, D, and E) are in a room. A is watching TV with B, D is sleeping, B is eating chow min, and E is playing table tennis. Suddenly, a call comes on the telephone. B goes out of the room to pick up the call. What is C doing?"""

# node 1

from understanding import ProblemUnderstanding
from langchain_core.messages import HumanMessage
from models import State, Understanding

import yaml


initial_state = State(messages=[HumanMessage(content=problem)])

problem_understanding = ProblemUnderstanding("simple_prompts", llm, None)
interim = problem_understanding.understand_problem(initial_state, terms_recursion_limit=0, recursion_limit=1)

saved_understanding_yaml = understanding = yaml.dump(interim["understanding"])
saved_understanding:Understanding = yaml.safe_load(saved_understanding_yaml)


# node 2

from langchain_core.messages import AIMessage

from devise_plan import DevisePlan
from models import Plan


plan_preparation = DevisePlan("prompts", llm)

initial_state = State(
    messages=[AIMessage(content=saved_understanding_yaml), HumanMessage(content=problem)],
    understanding=saved_understanding
)

result = plan_preparation.devise_plan(state=initial_state, review=False)

saved_plan_yaml = yaml.dump(result["plan"])
saved_plan: Plan = yaml.safe_load(saved_plan_yaml)

# node 3

from carry_out_plan import CarryOutPlan


executor = CarryOutPlan("prompts", llm)


initial_state: State = State(
    messages=[
        AIMessage(content=saved_understanding_yaml), 
        AIMessage(content=saved_plan_yaml), 
        HumanMessage(content=problem)
    ],
    understanding=saved_understanding,
)
initial_state.plan = saved_plan

result = executor.carry_out_plan(initial_state, adjustment_limit=1)
saved_execution_yaml = yaml.dump(result["execution"])
saved_execution = yaml.safe_load(saved_execution_yaml)

saved_plan_yaml = yaml.dump(result["plan"])
saved_plan: Plan = yaml.safe_load(saved_plan_yaml)

# node 4

from models import Reflections

initial_state: State = State(
    messages=[
        AIMessage(content=saved_understanding_yaml),
        AIMessage(content=saved_plan_yaml),
        AIMessage(content=saved_execution_yaml),
        HumanMessage(content=problem)
    ],
    understanding=saved_understanding,
    plan=saved_plan,
    execution=saved_execution,
)

# Initialize the Reflection node.
from reflection import Reflection

reflector = Reflection("prompts", llm)

# Perform reflection.
result: Reflections = reflector.reflect_on_solution(initial_state)

#print(yaml.dump(result))

# if result is empty, the plan may be changed with Plan.select_alternate_strategy()
# otherwise we are ready to output to the user.
# if the result->reflection is missing solution, no solution was found, and we have a summary of work. We should output our summary to the user.
# if the result->reflection has a solution, we should output the solution to the user.
'''

from typing import TypeVar, Generic, Callable, Dict, Any
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph

from models import State, Reflections
from understanding import ProblemUnderstanding
from devise_plan import DevisePlan
from carry_out_plan import CarryOutPlan
from reflection import Reflection

T = TypeVar('T')

class GraphFactory(Generic[T]):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def create_graph(
        self,
        problem_understanding_config: Dict[str, Any] = None,
        devise_plan_config: Dict[str, Any] = None,
        carry_out_plan_config: Dict[str, Any] = None,
        reflection_config: Dict[str, Any] = None,
        problem_understanding_method_config: Dict[str, Any] = None,
        devise_plan_method_config: Dict[str, Any] = None,
        carry_out_plan_method_config: Dict[str, Any] = None,
        reflection_method_config: Dict[str, Any] = None,
    ) -> StateGraph[State]:
        # Helper function to ensure 'prompt_key' and 'llm' are set
        def ensure_config(config, default_prompt_key='prompts'):
            if 'prompt_key' not in config:
                config['prompt_key'] = default_prompt_key
            if 'llm' not in config:
                config['llm'] = self.llm
            return config

        # Ensure class configs are dicts and set default 'prompt_key' and 'llm'
        class_configs = {
            'problem_understanding_config': problem_understanding_config,
            'devise_plan_config': devise_plan_config,
            'carry_out_plan_config': carry_out_plan_config,
            'reflection_config': reflection_config
        }

        for key in class_configs:
            if class_configs[key] is None:
                class_configs[key] = {}
            ensure_config(class_configs[key], default_prompt_key='prompts')

        # Ensure method configs are dicts
        method_configs = {
            'problem_understanding_method_config': problem_understanding_method_config,
            'devise_plan_method_config': devise_plan_method_config,
            'carry_out_plan_method_config': carry_out_plan_method_config,
            'reflection_method_config': reflection_method_config
        }

        for key in method_configs:
            if method_configs[key] is None:
                method_configs[key] = {}

        # Unpack configs
        problem_understanding_config = class_configs['problem_understanding_config']
        devise_plan_config = class_configs['devise_plan_config']
        carry_out_plan_config = class_configs['carry_out_plan_config']
        reflection_config = class_configs['reflection_config']

        problem_understanding_method_config = method_configs['problem_understanding_method_config']
        devise_plan_method_config = method_configs['devise_plan_method_config']
        carry_out_plan_method_config = method_configs['carry_out_plan_method_config']
        reflection_method_config = method_configs['reflection_method_config']

        # Initialize nodes using the configurations
        problem_understanding_node = ProblemUnderstanding(**problem_understanding_config)
        devise_plan_node = DevisePlan(**devise_plan_config)
        carry_out_plan_node = CarryOutPlan(**carry_out_plan_config)
        reflection_node = Reflection(**reflection_config)

        # Define the graph with nodes and their corresponding functions
        graph = StateGraph[State](
            {
                "understand_problem": lambda state: problem_understanding_node.understand_problem(
                    state,
                    **problem_understanding_method_config
                ),
                "devise_plan": lambda state: devise_plan_node.devise_plan(
                    state,
                    **devise_plan_method_config
                ),
                "carry_out_plan": lambda state: carry_out_plan_node.carry_out_plan(
                    state,
                    **carry_out_plan_method_config
                ),
                "reflect_on_solution": lambda state: reflection_node.reflect_on_solution(
                    state,
                    **reflection_method_config
                )
            }
        )

        # Define edges to connect nodes based on the flow described
        graph.add_edge("understand_problem", "devise_plan")
        graph.add_edge("devise_plan", "carry_out_plan")
        graph.add_edge("carry_out_plan", "reflect_on_solution")

        # Single conditional edge from "reflect_on_solution"
        graph.add_conditional_edges(
            source="reflect_on_solution",
            path=self.determine_next_node,
            path_map={"change_strategy": "devise_plan", "proceed": "produce_final_answer"},
            then=None
        )

        # Set entry and finish points
        graph.set_entry_point("understand_problem")
        graph.set_finish_point("produce_final_answer")

        return graph

    def determine_next_node(self, state: State) -> str:
        """
        Determine the next node based on the reflection result.

        Args:
            state (State): The current state containing the solution or summary.

        Returns:
            str: Name of the next node.
        """
        if state.reflection.get("should_change_strategy", False):
            return "change_strategy"
        else:
            return "proceed"

    def update_state_with_reflections(self, state: State) -> State:
        """
        Update the state with reflections before re-devising a plan.

        Args:
            state (State): The current state containing reflection details.

        Returns:
            State: Updated state.
        """
        # Assuming reflection updates understanding or other relevant parts
        if "summary_of_work" in state.reflection:
            state.understanding["reflections"] = state.reflection["summary_of_work"]
        
        return state

    def produce_final_answer(self, state: State) -> Reflections:
        """
        Generates the final answer based on the solution or summary in the state.
        This function updates the state with the final reflection.

        Args:
            state (State): The current state containing the solution or summary.

        Returns:
            Reflections: Final reflections to return.
        """
        # Implement the logic to produce the final reflection
        if "solution" in state.reflection and state.reflection["solution"]:
            # Format the final answer using the solution
            final_reflection = {
                "summary_of_work": f"Here is the solution:\n\n{state.reflection['solution']}"
            }
        else:
            # Provide a summary or a message indicating no solution was found
            final_reflection = {
                "summary_of_work": "Unfortunately, a solution could not be found."
            }

        return Reflections(**final_reflection)


def create_langgraph_agent(
    llm: BaseChatModel,
    problem_understanding_config: Dict[str, Any] = None,
    devise_plan_config: Dict[str, Any] = None,
    carry_out_plan_config: Dict[str, Any] = None,
    reflection_config: Dict[str, Any] = None,
    problem_understanding_method_config: Dict[str, Any] = None,
    devise_plan_method_config: Dict[str, Any] = None,
    carry_out_plan_method_config: Dict[str, Any] = None,
    reflection_method_config: Dict[str, Any] = None,
) -> StateGraph[State]:
    factory = GraphFactory(llm)

    # Helper function to set defaults similar to 'ensure_config' in 'create_graph'
    def ensure_config(config, default_config):
        if config is None:
            return default_config
        else:
            # Update the default config with the provided config
            updated_config = default_config.copy()
            updated_config.update(config)
            return updated_config

    # Set default configurations
    default_problem_understanding_method_config = {'terms_recursion_limit': 0, 'recursion_limit': 1}
    default_devise_plan_method_config = {'review': False}
    default_carry_out_plan_method_config = {'adjustment_limit': 1}

    # Ensure configurations are set with defaults if not provided
    problem_understanding_method_config = ensure_config(
        problem_understanding_method_config, default_problem_understanding_method_config
    )
    devise_plan_method_config = ensure_config(
        devise_plan_method_config, default_devise_plan_method_config
    )
    carry_out_plan_method_config = ensure_config(
        carry_out_plan_method_config, default_carry_out_plan_method_config
    )

    # Other configs can remain as provided or be set to empty dicts if None
    devise_plan_config = devise_plan_config or {}
    carry_out_plan_config = carry_out_plan_config or {}
    reflection_config = reflection_config or {}
    reflection_method_config = reflection_method_config or {}

    return factory.create_graph(
        problem_understanding_config=problem_understanding_config,
        devise_plan_config=devise_plan_config,
        carry_out_plan_config=carry_out_plan_config,
        reflection_config=reflection_config,
        problem_understanding_method_config=problem_understanding_method_config,
        devise_plan_method_config=devise_plan_method_config,
        carry_out_plan_method_config=carry_out_plan_method_config,
        reflection_method_config=reflection_method_config,
    )

"""
# Example usage:
from langchain_ollama import ChatOllama

# Initialize LLM
llm = ChatOllama(
    model="qwen2.5:32b-instruct-q6_K",
    temperature=0.666,
    top_p=0.8,
    repeat_penalty=1.05,
    num_ctx=32678,
    num_predict=8192
)

# Create the graph
graph = create_langgraph_agent(llm)

# Define initial state
initial_state = State(
    messages=[
        {"role": "user", "content": "Five people (A, B, C, D, and E) are in a room..."}
    ],
    understanding={},
    plan={},
    execution={},
    reflection={}
)

# Run the graph with the initial state
final_reflection = graph.run(initial_state)
print(final_reflection)
"""
