from langchain.prompts import PromptTemplate

from agentic_ai.validations.validation import parser
from agentic_ai.utils.agent_state import AgentState
from agentic_ai.models.models import Models


def validation(state: AgentState):

    question = state["messages"][0]
    answer = state["messages"][-1]

    print("answer", answer)

    template = """
    Your task is to check if the response is related to the user question. 
    Respond in JSON format with fields "Topic" and "Reasoning".

    - "Topic" should be either "Yes" or "No".
    - "Reasoning" should briefly justify why.

    User query: {question}
    response: {answer}
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variable=["question", "answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = Models().gpt_model
    chain = prompt | model | parser

    response = chain.invoke({"question": question, "answer": answer})

    print("Parsed response:", response)

    return {"messages": [response.Topic]}
