from src.retrival_genaration import RerankRetriever, rerank_documents
from langchain.schema import Document
from sentence_transformers import CrossEncoder

def test_rerank_documents():
    query = "organic rice under ₹200"
    dummy_docs = [
        Document(page_content="Organic Basmati Rice, ₹180, Rating 4.5"),
        Document(page_content="Regular Rice, ₹90, Rating 3.0"),
        Document(page_content="Organic Brown Rice, ₹195, Rating 4.0"),
    ]

    reranked = rerank_documents(query, dummy_docs)

    assert isinstance(reranked, list)
    assert all(isinstance(doc, Document) for doc in reranked)
    assert len(reranked) == 3
    assert reranked[0].page_content in [doc.page_content for doc in dummy_docs]

