from phoenix.client import Client
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
client = Client()

SOURCE_DATASET_NAME = "demo 4"

try:
    ds = client.datasets.get_dataset(dataset=SOURCE_DATASET_NAME)
    df = ds.to_dataframe()
    
    print("\n--- Dataframe Overview ---")
    print(f"Columns: {df.columns.tolist()}")
    
    if not df.empty:
        print("\n--- Row 0 Sample ---")
        row = df.iloc[0]
        if 'response' in df.columns:
            print(f"Response: {str(row['response'])[:200]}...")
        else:
            print("ERROR: 'response' column still missing!")
            
        print(f"Input: {row['input']}")
        print(f"Output: {row['output']}")

except Exception as e:
    import traceback
    traceback.print_exc()
