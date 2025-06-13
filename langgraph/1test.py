from typing import Annotated
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import tempfile

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
os.environ['TAVILY_API_KEY'] = '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'
os.environ['GROQ_API_KEY'] = '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'
tool = TavilySearchResults(max_results=2, )
tools = [tool]
llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

try:
    png_data = graph.get_graph().draw_mermaid_png()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        temp_file.write(png_data)
        temp_file_path = temp_file.name
    os.system(f'xdg-open {temp_file_path}')
except Exception as e:
    print(f"An error occurred while generating or displaying the graph: {e}")
