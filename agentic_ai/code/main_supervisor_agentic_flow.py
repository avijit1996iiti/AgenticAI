from langgraph.graph import StateGraph, END, START


# node related imports
from agentic_ai.utils.agent_state import AgentState
from agentic_ai.nodes.supervisor import supervisor
from agentic_ai.nodes.agents.rag import CustomRag
from agentic_ai.models.models import Models
from agentic_ai.nodes.agents.llm_call import llm_call
from agentic_ai.nodes.agents.search_func import search_func
from agentic_ai.nodes.agents.stock_price import stock_price_tool
from agentic_ai.nodes.agents.validation import validation

# routers
from agentic_ai.routers.router import router, validation_router_1

# utils
from agentic_ai.utils.utils import run_agentic_app_from_file


# If you want to add new tools
# step1. add node
# step2. add to conditional edges
# step3. add a edge with validation
models = Models()
custom_rag = CustomRag(models=models)
workflow = StateGraph(AgentState)
workflow.add_node("Supervisor", supervisor)
workflow.add_node("RAG", custom_rag.rag)
workflow.add_node("LLM", llm_call)
workflow.add_node("WEB", search_func)
workflow.add_node("STOCK", stock_price_tool)
workflow.add_node("VALIDATION", validation)
workflow.set_entry_point("Supervisor")


workflow.add_conditional_edges(
    "Supervisor",
    router,
    {
        "RAG Call": "RAG",
        "LLM Call": "LLM",
        "WEB Call": "WEB",
        "STOCK Call": "STOCK",
    },
)

workflow.add_edge("RAG", "VALIDATION")
workflow.add_edge("LLM", "VALIDATION")
workflow.add_edge("WEB", "VALIDATION")
workflow.add_edge("STOCK", "VALIDATION")

workflow.add_conditional_edges("VALIDATION", validation_router_1, {"yes": END, "no": "Supervisor"})


app = workflow.compile() 
# convert a defined workflow or agent system into a runnable application or executable graph.



# from IPython.display import Image, display
# display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# with open("graph.png", "wb") as f:
#     f.write(app.get_graph(xray=True).draw_mermaid_png())


## LLM Call
# run_agentic_app_from_file(
#     app= app,
#     input_path= "agentic_ai/data/input/llm_message.txt",
#     output_path= "agentic_ai/data/output/llm_message.json"
# )



### RAG Example
# run_agentic_app_from_file(
#     app= app,
#     input_path= "agentic_ai/data/input/rag_message.txt",
#     output_path= "agentic_ai/data/output/rag_message.json"
# )



# Stock Price Checker 
# run_agentic_app_from_file(
#     app= app,
#     input_path= "agentic_ai/data/input/stock_price_message.txt",
#     output_path= "agentic_ai/data/output/stock_price_message.json"
# )



# Web
run_agentic_app_from_file(
    app= app,
    input_path= "agentic_ai/data/input/web_message.txt",
    output_path= "agentic_ai/data/output/web_message.json"
)
