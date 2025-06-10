from src.retrival_genaration import ingest_bigbasket_data, generate_bigbasket_chain

def test_ingest_bigbasket_data():
    vectorstore = ingest_bigbasket_data()
    assert vectorstore is not None

    retriever = vectorstore.as_retriever()
    results = retriever.invoke("skin products")
    assert isinstance(results, list)
    assert all(hasattr(doc, "page_content") for doc in results)

def test_generate_bigbasket_chain():
    vectorstore = ingest_bigbasket_data()
    chain = generate_bigbasket_chain(vectorstore)
    assert chain is not None

    answer = chain.invoke("which is the best skin care product with highest rating")
    assert isinstance(answer, str)
    assert len(answer) > 0
