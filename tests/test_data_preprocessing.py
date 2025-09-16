import pytest
import pandas as pd
from src.data_preprocessing import BigBasketPreprocessor
from src.data_ingestion import DataIngestion

@pytest.fixture
def sample_csv(tmp_path):
    """Fixture to create a sample CSV file for testing."""
    data = [
        {
            "index": 0,
            "product": "Apple",
            "category": "Fruits",
            "sub_category": "Apples",
            "brand": "FarmFresh",
            "sale_price": 100,
            "market_price": 120,
            "type": "Food",
            "rating": 4.5,
            "description": "Fresh apple"
        },
        {
            "index": 1,
            "product": "Banana",
            "category": "Fruits",
            "sub_category": "Bananas",
            "brand": "Tropical",
            "sale_price": 50,
            "market_price": 60,
            "type": "Food",
            "rating": 4.0,
            "description": "Ripe banana"
        },
        {
            "index": 2,
            "product": "Garlic Oil - Vegetarian Capsule 500 mg",
            "category": "Beauty & Hygiene",
            "sub_category": "Hair Care",
            "brand": "Sri Sri Ayurveda",
            "sale_price": 220,
            "market_price": 220,
            "type": "Hair Oil & Serum",
            "rating": 4.1,
            "description": "This Product contains Garlic Oil that is known to help proper digestion, maintain proper cholesterol levels, support cardiovascular and also build immunity."
        },
        {
            "index": 3,
            "product": "Water Bottle - Orange",
            "category": "Kitchen, Garden & Pets",
            "sub_category": "Storage & Accessories",
            "brand": "Mastercook",
            "sale_price": 180,
            "market_price": 180,
            "type": "Water & Fridge Bottles",
            "rating": 2.3,
            "description": "Each product is microwave safe, refrigerator safe, dishwasher safe and can also be used for re-heating food."
        },
    ]
    df = pd.DataFrame(data)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "BigBasket Products.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


def test_clean_data(sample_csv):
    preprocessor = BigBasketPreprocessor(base_dir=str(sample_csv))
    df_clean = preprocessor.clean_data()

    assert not df_clean.empty
    assert "category_tag" in df_clean.columns

    assert pd.api.types.is_numeric_dtype(df_clean["sale_price"])
    assert pd.api.types.is_numeric_dtype(df_clean["market_price"])
    assert pd.api.types.is_numeric_dtype(df_clean["rating"])

    row = df_clean[df_clean["product"] == "Apple"].iloc[0]
    assert row["category_tag"] == "Fruits > Apples"


def test_generate_documents(sample_csv):
    preprocessor = BigBasketPreprocessor(base_dir=str(sample_csv))
    preprocessor.clean_data()
    docs = preprocessor.generate_documents()

    assert isinstance(docs, list)
    assert len(docs) == 4
    first_doc = docs[0]

    assert hasattr(first_doc, "page_content")
    assert hasattr(first_doc, "metadata")

    for key in [
        "product", "brand", "category", "sub_category",
        "type", "sale_price", "market_price", "rating", "category_tag"
    ]:
        assert key in first_doc.metadata

    assert "Product:" in first_doc.page_content
    assert "Sale Price:" in first_doc.page_content
    assert "Description:" in first_doc.page_content


def test_generate_documents_empty_df(sample_csv):
    preprocessor = BigBasketPreprocessor(base_dir=str(sample_csv))
    preprocessor.df = pd.DataFrame()
    docs = preprocessor.generate_documents()
    assert docs == []
