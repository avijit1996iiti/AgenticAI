from agentic_ai.utils.agent_state import AgentState


def router(state: AgentState):
    print("-> ROUTER ->")

    last_message = state["messages"][-1]
    print("last_message:", last_message)
    if "avijit" in last_message.lower():
        return "RAG Call"
    elif any(
        word in last_message.lower()
        for word in [
            "stock price",
            "stock",
            "share price",
            "market",
            "ticker",
            "nasdaq",
            "nse",
            "bse",
        ]
    ):
        return "STOCK Call"
    elif "latest" in last_message.lower():
        return "WEB Call"
    else:
        return "LLM Call"


def validation_router_1(state: AgentState):
    print("-> Validation ROUTER ->")

    last_message = state["messages"][-1]
    print("last_message:", last_message)

    if "yes" in last_message.lower():
        return "yes"
    else:
        return "no"
