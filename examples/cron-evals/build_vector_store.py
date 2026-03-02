
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings

loader = WebBaseLoader("https://docs.arize.com/arize/")
documents = loader.load()
print(f"Loaded {len(documents)} documents")
if not documents:
    raise RuntimeError("No documents loaded. Please check the documentation URL or try a different loader.")

# Use Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
Qdrant.from_documents(
    documents,
    embeddings,
    path="./vector-store",
    collection_name="arize-documentation",
)
