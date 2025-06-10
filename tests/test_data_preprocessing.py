from src.data_preprocessing import preprocess_bigbasket_docs
from langchain_core.documents import Document

def test_preprocess_bigbasket_docs():
    docs = preprocess_bigbasket_docs()
    assert docs is not None
    assert isinstance(docs, list)
    assert len(docs) > 0

    first_doc = docs[0]
    assert isinstance(first_doc, Document)
    assert first_doc.page_content is not None
    assert isinstance(first_doc.metadata, dict)
    required_metadata_keys = {
        "product", "brand", "category", "sub_category", "type",
        "sale_price", "market_price", "rating", "category_tag"
    }
    assert required_metadata_keys.issubset(first_doc.metadata.keys())
