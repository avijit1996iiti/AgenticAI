from agentic_ai.utils.agent_state import AgentState
from agentic_ai.models.models import Models


def llm_call(state: AgentState):
    print("-> LLM Call ->")
    question = state["messages"][0]

    # Normal LLM call
    complete_query = (
        "Answer the follow question with you knowledge of the real world. Following is the user question: "
        + question
    )

    models = Models()
    response = models.gpt_model.invoke(complete_query)
    return {"messages": [response.content]}
