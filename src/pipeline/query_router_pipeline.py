# src/pipeline/query_router_pipeline.py
import logging
from src.reranking_genaration import QueryRouter

logger = logging.getLogger("QueryRouterPipeline")

class QueryRouterPipeline:
    def run(self, query: str) -> str:
        logger.info(">>> Running Query Router Pipeline...")
        router = QueryRouter()
        answer = router.route(query)
        logger.info("Query Router completed.")
        return answer
