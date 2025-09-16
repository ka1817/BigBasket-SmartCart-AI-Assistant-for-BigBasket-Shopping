#!/usr/bin/env python
import logging
from src.retrival_genaration import QueryRouter
from src.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("RunGeneration")
def main():
    try:
        logger.info("Initializing RAG QueryRouter...")
        router = QueryRouter()
        logger.info("QueryRouter initialized successfully.")

        test_query = "Give me Best hair oil"
        logger.info(f"Routing test query: {test_query}")
        result = router.route(test_query)

        print("\n=== RAG Answer ===")
        print(result)

    except Exception as e:
        logger.error(f"RAG generation failed: {e}")

if __name__ == "__main__":
    main()
