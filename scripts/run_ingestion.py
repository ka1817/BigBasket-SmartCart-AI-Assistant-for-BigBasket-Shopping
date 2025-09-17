import logging
from src.data_ingestion import DataIngestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("RunIngestion")

def main():
    try:
        logger.info("Starting data ingestion...")
        ingestor = DataIngestion()
        df = ingestor.load_data()
        logger.info(f"Data ingestion completed. Rows loaded: {df.shape[0]}")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        

if __name__ == "__main__":
    main()
