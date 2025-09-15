from src.data_preprocessing import BigBasketPreprocessor
from src.vectorstore import VectorStoreManager

if __name__ == "__main__":
    preprocessor = BigBasketPreprocessor()
    df = preprocessor.clean_data()
    documents = preprocessor.generate_documents()

    vs = VectorStoreManager(index_name="bigbasket-products")

    vs.add_documents(documents)

    vectorstore = vs.load_vectorstore()

    results = vectorstore.similarity_search("cold drinks", k=3)
    for r in results:
        print(r.page_content[:200], "\n---")
