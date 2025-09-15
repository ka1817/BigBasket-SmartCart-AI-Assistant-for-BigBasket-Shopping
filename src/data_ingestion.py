import pandas as pd
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("DataIngestion")


class DataIngestion:
    def __init__(self, base_dir: str = None):
        try:
            self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_file = os.path.join(self.base_dir, "data", "BigBasket Products.csv")

            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"File not found at path: {self.data_file}")

            logger.info(f"DataIngestion initialized with data file: {self.data_file}")

        except Exception as e:
            logger.error(f"Error during DataIngestion initialization: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.data_file)
            logger.info(f"Data loaded successfully from {self.data_file}")
            logger.info(f"Dataset shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error occurred during data ingestion: {e}")
            raise

