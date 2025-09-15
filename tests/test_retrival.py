import pytest
from src.vectorstore import VectorStoreManager

@pytest.mark.skipif(
    not bool(__import__("os").environ.get("PINECONE_API_KEY")),
    reason="PINECONE_API_KEY not set"
)
def test_similarity_search():
    vs_manager = VectorStoreManager(index_name="bigbasket-products")
    
    vectorstore = vs_manager.load_vectorstore()
    assert vectorstore is not None, "Vector store failed to load"

    query = "cold drinks"
    results = vectorstore.similarity_search(query, k=3)
    
    assert isinstance(results, list), "Results should be a list"
    for doc in results:
        assert hasattr(doc, "page_content"), "Each result should have page_content"
    
    print("Top 3 results for query '{}':".format(query))
    for doc in results:
        print(doc.page_content[:200], "\n---")
