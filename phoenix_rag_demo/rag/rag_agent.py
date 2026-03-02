from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

# Gemini setup
genai.configure()

# Load vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="../vectorstore",
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def rag_agent(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a company support assistant.

Use ONLY the following company policy to answer the question.
If the answer is not present, say "I don't know".

Company Policy:
{context}

Question:
{question}
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text