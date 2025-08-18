from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class TopicSelectionParser(BaseModel):
    Topic: str = Field(description="selected topic")
    Reasoning: str = Field(description="Reasoning behind topic selection")


parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)

