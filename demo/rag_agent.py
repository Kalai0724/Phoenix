from google import genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="vectorstore/company_policy_chroma",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

genai_client = genai.Client()

def rag_agent(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
You are a company support assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text