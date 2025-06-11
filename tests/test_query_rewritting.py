from src.query_rewritting import query_rewriting

def test_query_rewriting():
    original_query = "which is the best skin care product with high rating"
    rewritten_query = query_rewriting(original_query)

    assert isinstance(rewritten_query, str)
    assert len(rewritten_query) > 0
