from dotenv import load_dotenv
import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


@dataclass
class Models:
    gpt_model_name: str = "gpt-4o"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self):
        self.gpt_model = ChatOpenAI(
            model=self.gpt_model_name, api_key=os.getenv("OPENAI_API_KEY")
        )

        self.hf_embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
