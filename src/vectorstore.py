# src/vectorstore.py
import os
import logging
from typing import List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import warnings
warnings.filterwarnings('ignore')
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("VectorStore")

load_dotenv()


class VectorStoreManager:

    def __init__(self, index_name: str = "bigbasket-products", dimension: int = 768):
        self.index_name = index_name
        self.dimension = dimension

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("Missing PINECONE_API_KEY in environment variables")

        self.pc = Pinecone(api_key=self.pinecone_api_key)

        self.embeddings = HuggingFaceEmbeddings()

        self.index = None
        self.vector_store: Optional[PineconeVectorStore] = None

    def create_or_load_index(self) -> None:
        if not self.pc.has_index(self.index_name):
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            logger.info(f"Using existing Pinecone index: {self.index_name}")

        self.index = self.pc.Index(self.index_name)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            logger.warning("No documents provided for insertion into vector store.")
            return

        if not self.vector_store:
            self.create_or_load_index()

        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Inserted {len(documents)} documents into index '{self.index_name}'")
        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            raise

    def load_vectorstore(self) -> PineconeVectorStore:
        if not self.vector_store:
            self.create_or_load_index()

        logger.info(f"Vector store loaded for index: {self.index_name}")
        return self.vector_store

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get retriever for use in QA pipelines.
        """
        if not self.vector_store:
            self.create_or_load_index()

        search_kwargs = search_kwargs or {"k": 5}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vector_store:
            self.create_or_load_index()

        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise
