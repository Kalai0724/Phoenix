from phoenix.client import Client
import pandas as pd

client = Client()

try:
    ds = client.datasets.get_dataset(dataset="demo 4")
    df = ds.to_dataframe()
    
    print("Columns found:", df.columns.tolist())
    
    if "response" in df.columns:
        print("SUCCESS: 'response' column exists.")
        # Check if it has data
        non_null = df["response"].dropna()
        print(f"Number of non-null responses: {len(non_null)}")
        if len(non_null) > 0:
            print("Sample response:", non_null.iloc[0])
    else:
        print("FAILURE: 'response' column not found.")
        print("Available columns:", df.columns.tolist())

except Exception as e:
    print(f"ERROR: {e}")
