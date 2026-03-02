import os
import json

# ================= LangChain =================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================= Phoenix =================
from phoenix.otel import register
from phoenix.client import Client
from openinference.instrumentation import capture_span_context

# ================= CONFIG =================
PHOENIX_ENDPOINT = "http://localhost:6006"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ================= Phoenix Tracing =================
register(auto_instrument=True)
phoenix_client = Client(base_url=PHOENIX_ENDPOINT)

print(f"✅ Connected to Phoenix at {PHOENIX_ENDPOINT}")
print(f"📊 View traces at: {PHOENIX_ENDPOINT}")

# ================= RAG SETUP =================
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

if not os.path.exists("documents"):
    os.makedirs("documents")
    with open("documents/sample.txt", "w") as f:
        f.write(
            "Our company offers 30-day refunds on all products. "
            "Contact support@company.com for help."
        )

documents = []
for file in os.listdir("documents"):
    if file.endswith(".txt"):
        documents.extend(TextLoader(f"documents/{file}").load())

chunks = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ================= LLMs =================
answer_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 🔐 JSON-ENFORCED JUDGE LLM (KEY FIX)
judge_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    response_format={"type": "json_object"},
)

# ================= RAG CHAIN =================
prompt = ChatPromptTemplate.from_template("""
Use the context to answer the question.
If unsure, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    | prompt
    | answer_llm
    | StrOutputParser()
)

# ================= SINGLE JUDGE PROMPT =================
JUDGE_PROMPT = """
Evaluate the RAG answer using ONLY the reference text.

Return ONLY valid JSON in this exact structure:

{{
  "qa_correctness": {{
    "label": "correct | incorrect",
    "explanation": "short reason"
  }},
  "faithfulness": {{
    "label": "faithful | unfaithful",
    "explanation": "short reason"
  }},
  "relevance": {{
    "label": "relevant | irrelevant",
    "explanation": "short reason"
  }},
  "conciseness": {{
    "label": "concise | verbose",
    "explanation": "short reason"
  }},
  "hallucination": {{
    "label": "hallucinated | grounded",
    "explanation": "short reason"
  }}
}}

Rules:
- Use ONLY the reference text
- No markdown
- No extra text

Question: {question}
Reference: {context}
Answer: {answer}
"""

LABEL_SCORES = {
    "correct": 1, "incorrect": 0,
    "faithful": 1, "unfaithful": 0,
    "relevant": 1, "irrelevant": 0,
    "concise": 1, "verbose": 0,
    "hallucinated": 1, "grounded": 0,
}

# ================= INTERACTIVE LOOP =================
print("\n🤖 Agent ready! Type 'exit' to quit")

while True:
    query = input("\nYou: ").strip()
    if query.lower() == "exit":
        break

    try:
        with capture_span_context() as capture:

            # 1️⃣ Retrieve context
            docs = retriever.invoke(query)
            context = docs[0].page_content if docs else ""

            # 2️⃣ Generate answer
            answer = rag_chain.invoke(
                query,
                config={"metadata": {"question": query, "reference_answer": context}},
            )

            print(f"\n🤖: {answer}")

            span_id = capture.get_last_span_id()
            if not span_id:
                continue

            # 3️⃣ SINGLE JUDGE CALL
            judge_response = judge_llm.invoke(
                JUDGE_PROMPT.format(
                    question=query,
                    context=context,
                    answer=answer,
                )
            )

            results = json.loads(judge_response.content)

            # 4️⃣ ADD PHOENIX ANNOTATIONS (WITH EXPLANATIONS)
            for metric, data in results.items():
                phoenix_client.annotations.add_span_annotation(
                    annotation_name=metric,
                    annotator_kind="LLM",
                    span_id=span_id,
                    label=data["label"],
                    score=LABEL_SCORES[data["label"]],
                    explanation=data["explanation"],
                )

    except Exception as e:
        print(f"❌ Error: {e}")