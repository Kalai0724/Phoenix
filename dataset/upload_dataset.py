import phoenix as px
from phoenix.client import Client

# ✅ import dataset_df
from create_dataset import dataset_df

px_client = Client()

px_client.datasets.create_dataset(
    dataframe=dataset_df,
    name="support-ticket-queries",
    input_keys=["query"],
    output_keys=["expected_category"],
)

print("✅ Dataset uploaded successfully")