import operator
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. Define State & Schemas
# ==========================================
class PlanExecuteState(TypedDict):
    objective: str
    plan: List[str]
    # operator.add ensures that returning a new list appends to the existing one, rather than overwriting it
    past_steps: Annotated[List[tuple], operator.add]
    final_response: str

class Plan(BaseModel):
    """Structured output schema for the planner."""
    steps: List[str] = Field(description="The steps required to achieve the objective.")

# ==========================================
# 2. Define Nodes
# ==========================================
def planner_node(state: PlanExecuteState):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("Missing 'OPENAI_API_KEY' in your .env file.")
    
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="openai/gpt-4o",
        temperature=0
    )
    structured_llm = llm.with_structured_output(Plan)
    
    prompt = f"Create a step-by-step plan to achieve this objective: {state['objective']}"
    plan_output = structured_llm.invoke(prompt)
    
    print(f"--- PLAN CREATED ---\n{plan_output.steps}\n")
    return {"plan": plan_output.steps}

def executor_node(state: PlanExecuteState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Identify the next step by checking how many steps we've already completed
    step_index = len(state.get("past_steps", []))
    current_step = state["plan"][step_index]
    
    print(f"--- EXECUTING STEP {step_index + 1} ---")
    print(f"Task: {current_step}")
    
    prompt = f"""
    Objective: {state['objective']}
    Past steps and results: {state.get("past_steps", [])}
    
    Current Step to execute: {current_step}
    
    Perform the step and provide the result.
    """
    
    # Note: In a production agent, you would provide the LLM with tools here (e.g., search, DB access)
    result = llm.invoke(prompt)
    
    return {"past_steps": [(current_step, result.content)]}

def finalizer_node(state: PlanExecuteState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = f"Objective: {state['objective']}\nResults: {state.get('past_steps', [])}\n\nProvide the final answer."
    result = llm.invoke(prompt)
    
    return {"final_response": result.content}

# ==========================================
# 3. Define Conditional Routing
# ==========================================
def route_execution(state: PlanExecuteState):
    """Determine whether to continue executing or finalize."""
    if len(state.get("past_steps", [])) >= len(state.get("plan", [])):
        return "finalize"
    return "execute"

# ==========================================
# 4. Build and Compile the Graph
# ==========================================
workflow = StateGraph(PlanExecuteState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("finalizer", finalizer_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")

# After execution, route based on whether steps remain
workflow.add_conditional_edges(
    "executor",
    route_execution,
    {"execute": "executor", "finalize": "finalizer"}
)
workflow.add_edge("finalizer", END)

app = workflow.compile()

# ==========================================
# 5. Run the Agent
# ==========================================
if __name__ == "__main__":
    inputs = {
        "objective": "Explain the difference between LangChain and LangGraph, then write a 2-sentence summary.",
        "past_steps": [] # Initialize to avoid operator.add errors on the first run
    }
    
    for output in app.stream(inputs):
        for key, value in output.items():
            if key == "finalizer":
                print(f"\n--- FINAL RESPONSE ---\n{value['final_response']}")