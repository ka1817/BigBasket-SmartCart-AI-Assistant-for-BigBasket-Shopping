import logging
import pandas as pd
from langchain_core.documents import Document
from src.data_preprocessing import BigBasketPreprocessor

logger = logging.getLogger("DataPreprocessingPipeline")

class DataPreprocessingPipeline:
    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Document]]:
        logger.info(">>> Running Data Preprocessing Pipeline...")
        preprocessor = BigBasketPreprocessor()
        preprocessor.df = df   # pass ingested data instead of reloading
        clean_df = preprocessor.clean_data()
        docs = preprocessor.generate_documents()
        logger.info(f"Data Preprocessing complete. Clean shape: {clean_df.shape}, Docs: {len(docs)}")
        return clean_df, docs
