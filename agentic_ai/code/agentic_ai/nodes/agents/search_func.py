from langchain_community.tools import DuckDuckGoSearchRun
from agentic_ai.utils.agent_state import AgentState


def search_func(state: AgentState):
    search = DuckDuckGoSearchRun()
    result = search.invoke({"query": state["messages"][0]})
    return {"messages": [result]}
