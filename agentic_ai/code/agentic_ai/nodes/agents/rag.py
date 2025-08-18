from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from agentic_ai.utils.agent_state import AgentState
from agentic_ai.models.models import Models
from agentic_ai.utils.utils import format_docs


@dataclass
class CustomRag:
    models: Models

    def __post_init__(self):
        # Load a PDF and validate the page count
        file_path = "agentic_ai/data/Avijit_Bhattacharjee__Senior_Machine_Learning_Engineer__L3__.pdf"
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # if len(pages) < 200:
        #     raise ValueError("The PDF must have at least 200 pages.")

        # Using semantic chunking with recursive text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        chunks = text_splitter.split_documents(pages)

        # Hugging Face embedding
        # embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        embedding_model = self.models.hf_embeddings

        faiss_flat = FAISS.from_documents(chunks, embedding_model)
        self.retriever_flat = faiss_flat.as_retriever()

    def rag(self, state: AgentState):
        print("-> RAG Call ->")

        question = state["messages"][0]

        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
              If you don't know the answer, just say that you don't know.
                Keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""",
            input_variables=["context", "question"],
        )

        rag_chain = (
            {
                "context": self.retriever_flat | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.models.gpt_model
            | StrOutputParser()
        )
        result = rag_chain.invoke(question)
        return {"messages": [result]}
