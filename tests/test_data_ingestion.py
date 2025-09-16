import pandas as pd
import pytest
from src.data_ingestion import DataIngestion
@pytest.fixture
def sample_csv(tmp_path):
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
            "brand": "Sri Sri Ayurveda ",
            "sale_price": 220,
            "market_price": 220,
            "type": "Hair Oil & Serum",
            "rating": 4.1,
            "description": "This Product contains Garlic Oil that is known to help proper digestion, maintain proper cholesterol levels, support cardiovascular and also build immunity.  For Beauty tips, tricks & more visit https://bigbasket.blog/"
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
            "description": "Each product is microwave safe (without lid), refrigerator safe, dishwasher safe and can also be used for re-heating food and not for cooking. All containers come with airtight lids and a wide variety of attractive colours. Stack these stylish and colourful containers in your kitchen with ease and for a look-good factor."
        },
    ]
    df = pd.DataFrame(data)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "BigBasket Products.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path

def test_load_data(sample_csv):
    ingestion = DataIngestion(base_dir=str(sample_csv))
    df = ingestion.load_data()
    assert df.shape[0] == 4
    assert "product" in df.columns
    assert "Garlic Oil - Vegetarian Capsule 500 mg" in df["product"].values
    assert "Water Bottle - Orange" in df["product"].values

def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataIngestion(base_dir=str(tmp_path))
