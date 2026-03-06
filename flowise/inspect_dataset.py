from phoenix.client import Client
import json
import logging

logging.basicConfig(level=logging.INFO)
client = Client()

SOURCE_DATASET_NAME = "demo 4"

try:
    ds = client.datasets.get_dataset(dataset=SOURCE_DATASET_NAME)
    df = ds.to_dataframe()
    
    if not df.empty:
        row = df.iloc[0]
        print(f"\n--- FULL ROW 0 ---")
        for col in df.columns:
            val = row[col]
            print(f"[{col}]: {val}")
            
except Exception as e:
    import traceback
    traceback.print_exc()
