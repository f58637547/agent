from langgraph.graph import END, StateGraph
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated, Literal, Dict

load_dotenv()

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
    memory = AsyncSqliteSaver.from_conn_string(":memory:")
    return workflow.compile(checkpointer=memory)
