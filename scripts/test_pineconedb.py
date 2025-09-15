from src.data_preprocessing import BigBasketPreprocessor
from src.vectorstore import VectorStoreManager

if __name__ == "__main__":

    vs = VectorStoreManager(index_name="bigbasket-products")
    vectorstore = vs.load_vectorstore()

    query = "cold drinks"

    metadata_filter = {"category": "Beverages"}

    results = vectorstore.similarity_search(
        query,
        k=3,
        filter=metadata_filter  
    )

    for r in results:
        print(f"Product: {r.metadata['product']}")
        print(f"Category: {r.metadata['category']}")
        print(f"Description: {r.page_content[:200]}")
        print("---")
