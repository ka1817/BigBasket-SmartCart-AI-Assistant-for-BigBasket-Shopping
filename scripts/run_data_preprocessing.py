from src.data_preprocessing import BigBasketPreprocessor
if __name__ == "__main__":
    preprocessor = BigBasketPreprocessor()
    preprocessor.clean_data()
    documents = preprocessor.generate_documents()
    if documents:
        print(f"Sample Document:\n{documents[0].page_content}\n")
