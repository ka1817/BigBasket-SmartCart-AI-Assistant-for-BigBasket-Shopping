# src/retrival_generation.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

# LangChain
from langchain.schema import Document, BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Sentence Transformers Reranker
from sentence_transformers import CrossEncoder

# Your modules
from src.data_preprocessing import preprocess_bigbasket_docs
from src.query_rewritting import query_rewriting

import warnings
warnings.filterwarnings('ignore')
# 🔐 GROQ API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🔁 CrossEncoder Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query: str, retrieved_docs: List[Document]) -> List[Document]:
    docs_texts = [doc.page_content for doc in retrieved_docs]
    pairs = [(query, doc_text) for doc_text in docs_texts]
    scores = reranker.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)]
    return sorted_docs

class RerankRetriever(BaseRetriever, BaseModel):
    base_retriever: BaseRetriever
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        initial_docs = self.base_retriever.invoke(query)
        reranked_docs = rerank_documents(query, initial_docs)
        return reranked_docs[:self.top_k]


def ingest_bigbasket_data():
    """Preprocess and embed product docs, store in FAISS vectorstore."""
    docs = preprocess_bigbasket_docs()
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore


def generate_bigbasket_chain(vectorstore):
    """Create RAG chain using custom reranker-based retriever."""

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    custom_retriever = RerankRetriever(base_retriever=base_retriever, top_k=5)

    PRODUCT_BOT_TEMPLATE = """
    You are a helpful assistant that provides answers to user queries based on BigBasket product information.

    Instructions:
    - Base your answer only on the provided context.
    - Include details like price, rating, and brand.
    - Summarize multiple products if they match.
    - Say "Sorry, no matching product information was found in the dataset." if nothing matches.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    """

    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")

    chain = (
        {"context": custom_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

