from phoenix.client import Client
import logging
import pandas as pd

logging.basicConfig(level=logging.ERROR)
client = Client()

try:
    ds = client.datasets.get_dataset(dataset="demo 4")
    # Calling as_dataframe directly to pass drop_empty_columns=False
    df = ds.as_dataframe(drop_empty_columns=False)
    
    print(f"COLUMNS: {df.columns.tolist()}")
    print(f"RESPONSE_COLUMN_EXISTS: {'response' in df.columns}")
    
    if 'response' in df.columns:
        non_null = df['response'].dropna()
        print(f"NON_NULL_COUNT: {len(non_null)}")
        if not non_null.empty:
            print(f"SAMPLE: {non_null.iloc[0]}")

except Exception as e:
    print(f"ERROR: {e}")
