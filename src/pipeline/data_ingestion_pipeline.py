import logging
from src.data_ingestion import DataIngestion
import pandas as pd

logger = logging.getLogger("DataIngestionPipeline")

class DataIngestionPipeline:
    def run(self) -> pd.DataFrame:
        logger.info(">>> Running Data Ingestion Pipeline...")
        ingestor = DataIngestion()
        df = ingestor.load_data()
        logger.info(f"Data Ingestion complete. Shape: {df.shape}")
        return df
