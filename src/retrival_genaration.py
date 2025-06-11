import os
import sys
from dotenv import load_dotenv
import warnings
from typing import List
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from langchain.schema import Document, BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.data_preprocessing import preprocess_bigbasket_docs
from src.query_rewritting import query_rewriting
from src.utils.logger import setup_logger

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logger = setup_logger("retrieval_generation", "retrieval_generation.log")

warnings.filterwarnings('ignore')

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query: str, retrieved_docs: List[Document]) -> List[Document]:
    logger.info(f"Reranking {len(retrieved_docs)} documents for query: '{query}'")
    docs_texts = [doc.page_content for doc in retrieved_docs]
    pairs = [(query, doc_text) for doc_text in docs_texts]
    scores = reranker.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)]
    logger.info("Reranking complete")
    return sorted_docs

class RerankRetriever(BaseRetriever, BaseModel):
    base_retriever: BaseRetriever
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        logger.info(f"Retrieving documents for: '{query}'")
        initial_docs = self.base_retriever.invoke(query)
        reranked_docs = rerank_documents(query, initial_docs)
        return reranked_docs[:self.top_k]

def ingest_bigbasket_data():
    logger.info("Starting preprocessing and embedding for BigBasket data")
    docs = preprocess_bigbasket_docs()
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    logger.info("Vectorstore creation complete")
    return vectorstore

def generate_bigbasket_chain(vectorstore):
    logger.info("Generating RAG chain with custom retriever")

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

    logger.info("RAG chain successfully created")
    return chain

if __name__ == "__main__":
    logger.info("Running standalone test of retrieval + generation")
    vectorstore = ingest_bigbasket_data()
    chain = generate_bigbasket_chain(vectorstore)
    sample_query = "best organic basmati rice under ₹200"
    rewritten_query = query_rewriting(sample_query)
    response = chain.invoke(rewritten_query)
    print(f"\n 🔁 Rewritten Query: {rewritten_query}\n🧠 Response:\n{response}")
