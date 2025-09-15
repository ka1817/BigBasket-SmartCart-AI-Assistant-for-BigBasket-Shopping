# tests/test_generation.py
import pytest
from src.retrival_genaration import QueryRouter

@pytest.mark.skipif(
    not bool(__import__("os").environ.get("PINECONE_API_KEY")),
    reason="PINECONE_API_KEY not set"
)
def test_query_router_initialization():
    """Test that the QueryRouter initializes without errors."""
    router = QueryRouter(top_k=2, confidence_threshold=0.5)
    assert router.global_retriever is not None, "Global retriever should be initialized"
    assert isinstance(router.retrievers, dict), "Retrievers should be a dictionary"
    print(f"Available categories: {list(router.retrievers.keys())}")


@pytest.mark.skipif(
    not bool(__import__("os").environ.get("PINECONE_API_KEY")),
    reason="PINECONE_API_KEY not set"
)
def test_route_query():
    """Test routing a sample query."""
    router = QueryRouter(top_k=2, confidence_threshold=0.5)
    sample_query = "Which beauty products are the most popular? based on rating"
    
    result = router.route(sample_query)
    
    assert isinstance(result, str), "Result should be a string (from StrOutputParser)"
    print(f"Query: {sample_query}\nAnswer: {result}")
