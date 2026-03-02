import pandas as pd
from phoenix.client import Client

client = Client()

df = pd.DataFrame({
    "question": [
        "What is the refund policy?",
        "How can customers contact support?",
        "What is the uptime guarantee?"
    ],
    "reference_answer": [
        "Customers can request a full refund within 30 days of purchase.",
        "Support is available 24/7 via email and chat, with phone support on business days.",
        "The platform guarantees 99.9% uptime."
    ]
})

client.datasets.create_dataset(
    name="company-policy-rag-dataset",
    dataframe=df,
    input_keys=["question"],
    output_keys=["reference_answer"]
)

print("✅ Evaluation dataset created")