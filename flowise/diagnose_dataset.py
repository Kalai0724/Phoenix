from phoenix.client import Client
import logging

logging.basicConfig(level=logging.ERROR)
client = Client()

try:
    ds = client.datasets.get_dataset(dataset="demo 4")
    print(f"TYPE: {type(ds)}")
    print(f"DIR: {sorted([m for m in dir(ds) if not m.startswith('_')])}")
    
    # Try to_dataframe if it exists
    if hasattr(ds, 'to_dataframe'):
        df = ds.to_dataframe()
        print(f"COLUMNS via to_dataframe: {df.columns.tolist()}")
    
    # Try as_dataframe if it exists
    if hasattr(ds, 'as_dataframe'):
        df = ds.as_dataframe()
        print(f"COLUMNS via as_dataframe: {df.columns.tolist()}")

except Exception as e:
    print(f"ERROR: {e}")
