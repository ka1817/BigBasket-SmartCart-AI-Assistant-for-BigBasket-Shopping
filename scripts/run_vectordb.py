#!/usr/bin/env python
import logging
from src.vectorstore import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("RunVectorDB")

def main():
    try:
        logger.info("Loading existing vector store...")
        vs_manager = VectorStoreManager()
        vectorstore = vs_manager.load_vectorstore()
        logger.info("Vector store loaded successfully.")

        query = "cold drinks"
        results = vectorstore.similarity_search(query, k=3)

        logger.info(f"Top {len(results)} results for query: '{query}'")
        for idx, r in enumerate(results, 1):
            print(f"Result {idx}:\n{r.page_content[:200]}...\n---")

    except Exception as e:
        logger.error(f"Vector store operation failed: {e}")

if __name__ == "__main__":
    main()
