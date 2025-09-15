import logging
import pandas as pd
from langchain_core.documents import Document
from src.data_ingestion import DataIngestion 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("DataPreprocessing")


class BigBasketPreprocessor:

    def __init__(self, base_dir: str = None):
        logger.info("Initializing BigBasket Preprocessor...")
        self.data_ingestor = DataIngestion(base_dir=base_dir)
        self.df = self.data_ingestor.load_data()
        logger.info(f"Loaded dataset with shape: {self.df.shape}")

    def clean_data(self) -> pd.DataFrame:
        
        logger.info("Starting data cleaning...")
        df = self.df.copy()

        essential_cols = ["product", "description", "sale_price", "market_price", "rating"]
        df.dropna(subset=essential_cols, inplace=True)
        logger.info(f"After dropping NA in essential columns: {df.shape}")

        numeric_cols = ["sale_price", "market_price", "rating"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=numeric_cols, inplace=True)
        logger.info(f"After cleaning numeric columns: {df.shape}")

        df["category_tag"] = df["category"] + " > " + df["sub_category"]
        self.df = df
        return df

    def generate_documents(self) -> list[Document]:
        if self.df.empty:
            logger.warning("DataFrame is empty. Ensure `clean_data()` was called.")
            return []

        docs = []
        logger.info("Generating Document objects from dataset...")
        for _, row in self.df.iterrows():
            text = (
                f"Product: {row['product']}\n"
                f"Brand: {row['brand']}\n"
                f"Category: {row['category_tag']}\n"
                f"Type: {row['type']}\n"
                f"Sale Price: ₹{row['sale_price']}\n"
                f"Market Price: ₹{row['market_price']}\n"
                f"Rating: {row['rating']} / 5.0\n"
                f"Description: {row['description']}"
            )

            metadata = {
                "product": row["product"],
                "brand": row["brand"],
                "category": row["category"],
                "sub_category": row["sub_category"],
                "type": row["type"],
                "sale_price": float(row["sale_price"]),
                "market_price": float(row["market_price"]),
                "rating": float(row["rating"]),
                "category_tag": row["category_tag"]
            }

            docs.append(Document(page_content=text, metadata=metadata))

        logger.info(f"Generated {len(docs)} Document objects for RAG.")
        return docs


