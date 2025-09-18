import logging

from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_preprocessing import DataPreprocessingPipeline
from src.pipeline.query_classification_pipeline import QueryClassificationPipeline
from src.pipeline.vectorstore_pipeline import VectorStorePipeline
from src.pipeline.query_router_pipeline import QueryRouterPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("MainPipeline")

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting BigBasket RAG Pipeline...")

        df = DataIngestionPipeline().run()

        clean_df, docs = DataPreprocessingPipeline().run(df)

        classifier = QueryClassificationPipeline().run(clean_df)

        vs_manager = VectorStorePipeline().run()

        query = "What are the best organic rice products?"
        answer = QueryRouterPipeline().run(query)
        logger.info(f"\n🔎 Sample Query: {query}\n💡 Answer: {answer}")

        logger.info("✅ Pipeline executed successfully.")

    except Exception as e:
        logger.exception(f"❌ Pipeline failed: {e}")
        raise
