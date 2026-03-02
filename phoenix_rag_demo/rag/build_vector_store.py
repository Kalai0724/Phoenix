import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load CSV knowledge base
df = pd.read_csv("../data/company_policies.csv")

texts = df["content"].tolist()
metadatas = df[["company_name", "category"]].to_dict(orient="records")

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

docs = splitter.create_documents(texts, metadatas=metadatas)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="../vectorstore"
)

print("✅ Vector store built")