import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    groq_api_key: str = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    langchain_tracing: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "BigBasket-RAG")
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")

    
    top_k: int = int(os.getenv("TOP_K", 5))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
    rag_prompt: str = """
You are a helpful assistant providing information about BigBasket products.

Category: {category}

Use only the context below to answer the question. 
If the context does not contain the exact answer, provide the best possible guidance based on the available information. 
Do not make up details that are not supported by the context. 
If unsure, politely indicate that the information is not fully available.

Context:
{context}

Question:
{question}

Provide a clear, concise, and accurate answer:
"""
settings = Settings()
