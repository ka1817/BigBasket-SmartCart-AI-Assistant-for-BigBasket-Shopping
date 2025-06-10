from src.data_ingestion import load_bigbasket_data

def test_load_bigbasket_data():
    df = load_bigbasket_data()
    assert df is not None
    assert not df.empty
    expected_columns = {"product", "description", "sale_price", "market_price", "rating", "category", "sub_category"}
    assert expected_columns.issubset(set(df.columns))
