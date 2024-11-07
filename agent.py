from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import AnyMessage, add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated, Literal, Dict
from langchain_core.runnables import RunnableConfig

load_dotenv()

# Initialize ChatGroq (or whichever model you're using) with tools bound to it
chatbot_with_tools = ChatGroq(model="llama3-groq-8b-8192-tool-use-preview")  # Replace with actual model ID or other config

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

async def call_model(state: GraphState, config: RunnableConfig) -> Dict[str, AnyMessage]:
    messages = state["messages"]
    response = await chatbot_with_tools.ainvoke(messages, config)
    return {"messages": response}

def get_runnable():
    workflow = StateGraph(GraphState)
    workflow.add_node("agent", call_model)
    workflow.set_entry_point("agent")
    memory = MemorySaver()  # Use in-memory saver instead
    return workflow.compile(checkpointer=memory)
