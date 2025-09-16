import pytest
from src.vectorstore import VectorStoreManager
@pytest.fixture
def vectorstore_manager():
    return VectorStoreManager(index_name="bigbasket-products")

def test_load_vectorstore(vectorstore_manager):
    vs = vectorstore_manager.load_vectorstore()
    assert vs is not None

def test_similarity_search(vectorstore_manager):
    vs = vectorstore_manager.load_vectorstore()
    query = "cold drinks"
    
    results = vs.similarity_search(query, k=3)
    
    assert isinstance(results, list)
    assert len(results) <= 3
    
    for doc in results:
        assert hasattr(doc, "page_content")
