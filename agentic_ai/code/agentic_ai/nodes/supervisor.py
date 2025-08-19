from agentic_ai.utils.agent_state import AgentState
from agentic_ai.validations.validation import parser
from agentic_ai.models.models import Models
from langchain_core.prompts import PromptTemplate


def supervisor(state: AgentState):
    question = state["messages"][-1]

    print("Question", question)

    template = """
    Your task is to classify the given user query into one of the following related categories: 
    [Avijit, LLM, Latest, Stock Price]. If question is about 
    Avijit then Avijit,
    if question is generic then LLM and
    if question is about stock price then Stock Price and
    if question is about some recent thing which you don't know then Latest 
    Only respond with the category name and nothing else.

    User query: {question}
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variable=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = Models().gpt_model
    chain = prompt | model | parser

    response = chain.invoke({"question": question})

    print("Parsed response:", response)

    return {"messages": [response.Topic]}
