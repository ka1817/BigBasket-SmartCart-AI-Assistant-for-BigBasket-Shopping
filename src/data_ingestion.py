import pandas as pd
import os

def load_bigbasket_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))  
    file_path = os.path.join(base_dir, "data", "BigBasket Products.csv")  

    df = pd.read_csv(file_path)
    return df
