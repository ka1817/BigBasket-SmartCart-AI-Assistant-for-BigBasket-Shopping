from src.data_ingestion import DataIngestion
if __name__ == "__main__":
    try:
        data_ingestor = DataIngestion()
        df = data_ingestor.load_data()
        print(df.head(3))
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
