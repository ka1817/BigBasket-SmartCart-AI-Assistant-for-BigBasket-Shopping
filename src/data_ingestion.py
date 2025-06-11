import pandas as pd
import os
from src.utils.logger import setup_logger

logger = setup_logger("data_ingestion", "data_ingestion.log")

def load_bigbasket_data():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_dir, "data", "BigBasket Products.csv")
        
        df = pd.read_csv(train_path)
        logger.info(f"Data loaded successfully from {train_path}")
        logger.info(f"Dataset shape: {df.shape}")
        
        return df

    except FileNotFoundError:
        logger.error(f"File not found at path: {train_path}")
        raise
    except Exception as e:
        logger.error(f"Error occurred during data ingestion: {e}")
        raise

if __name__ == '__main__':
    df = load_bigbasket_data()
    print(df.head())
