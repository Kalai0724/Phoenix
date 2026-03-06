from phoenix.client import Client
import logging

logging.basicConfig(level=logging.ERROR)
client = Client()

try:
    ds = client.datasets.get_dataset(dataset="demo 4")
    df = ds.to_dataframe()
    
    print(f"COLUMNS_EXIST: {'response' in df.columns}")
    if 'response' in df.columns:
        # Get first non-null response if possible
        resp = df['response'].dropna().iloc[0] if not df['response'].dropna().empty else "EMPTY"
        print(f"RESPONSE_SAMPLE: {str(resp)[:100]}")
    else:
        print("COLUMNS_FOUND: " + str(df.columns.tolist()))

except Exception as e:
    print(f"ERROR: {e}")
