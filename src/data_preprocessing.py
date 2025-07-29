from langchain_core.documents import Document
import pandas as pd
from src.data_ingestion import load_bigbasket_data
from src.utils.logger import setup_logger

logger = setup_logger("data_preprocessing", "data_preprocessing.log")

def preprocess_bigbasket_docs():
    logger.info("Starting BigBasket data preprocessing...")

    df = load_bigbasket_data()
    original_shape = df.shape
    logger.info(f"Original dataset shape: {original_shape}")

    df.dropna(subset=["product", "description", "sale_price", "market_price", "rating"], inplace=True)
    logger.info(f"After dropping NA in essential columns: {df.shape}")

    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
    df["market_price"] = pd.to_numeric(df["market_price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.dropna(subset=["sale_price", "market_price", "rating"], inplace=True)
    logger.info(f"After cleaning price/rating columns: {df.shape}")

    df["category_tag"] = df["category"] + " > " + df["sub_category"]

    docs = []
    for _, row in df.iterrows():
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

        doc = Document(page_content=text, metadata=metadata)
        docs.append(doc)

    logger.info(f"Generated {len(docs)} documents for RAG")
    return docs

if __name__ == '__main__':
    docs = preprocess_bigbasket_docs()
    print(f"Sample Document:\n{docs[0].page_content}\n")

