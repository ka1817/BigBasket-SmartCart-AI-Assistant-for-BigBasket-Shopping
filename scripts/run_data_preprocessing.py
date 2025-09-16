#!/usr/bin/env python
import logging
from src.data_preprocessing import BigBasketPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("RunPreprocessing")

def main():
    try:
        logger.info("Starting data preprocessing...")
        preprocessor = BigBasketPreprocessor()
        preprocessor.clean_data()
        documents = preprocessor.generate_documents()
        logger.info(f"Preprocessing completed. {len(documents)} Document objects generated.")
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")

if __name__ == "__main__":
    main()
