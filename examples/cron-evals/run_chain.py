# type: ignore
"""
Loads a pre-built Qdrant vector store and runs retrieval QA with Gemini.
"""

from itertools import cycle
import pandas as pd
from langchain_qdrant import QdrantVectorStore
try:
    from openinference.instrumentation.langchain_instrumentor import LangChainInstrumentor
    TRACING_AVAILABLE = True
except ImportError:
    print("[WARNING] OpenInference tracing is not available. Skipping tracing integration.")
    TRACING_AVAILABLE = False
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def get_retriever():
    qdrant_client = QdrantClient(path="./vector-store")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="arize-documentation",
        embedding=embeddings,
    )
    return vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 2}, enable_limit=True
    )

def instrument_langchain():
    endpoint = "http://127.0.0.1:6006/v1/traces"
    tracer_provider = trace_sdk.TracerProvider(
        resource=Resource.create({"service.name": "phoenix-cron-evals"})
    )
    trace_api.set_tracer_provider(tracer_provider)
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    if TRACING_AVAILABLE:
        LangChainInstrumentor().instrument()
    else:
        # TODO: Revisit tracing integration when openinference-instrumentation is updated
        pass

def load_queries():
    return pd.read_parquet(
        "http://storage.googleapis.com/arize-phoenix-assets/datasets/unstructured/llm/context-retrieval/langchain-pinecone/langchain_pinecone_query_dataframe_with_user_feedbackv2.parquet"
    ).text.to_list()

if __name__ == "__main__":
        # Minimal test span for Phoenix evaluation
        instrument_langchain()  # Set up tracing as the very first step
        tracer = trace_api.get_tracer(__name__)
        with tracer.start_as_current_span("retrieval-qa-query") as span:
            span.set_attribute("input", "What is Phoenix?")
            span.set_attribute("context", "Phoenix is an open-source AI observability platform.")
            span.set_attribute("output", "Phoenix is an AI observability platform built for tracing, evaluation, and monitoring.")
            span.set_attribute("retrieved_documents", ["Phoenix documentation", "Phoenix overview article"])
            span.set_attribute("reference_answer", "Phoenix is an AI observability platform.")
            print("Minimal Test Span Attributes:")
            print("input: What is Phoenix?")
            print("context: Phoenix is an open-source AI observability platform.")
            print("output: Phoenix is an AI observability platform built for tracing, evaluation, and monitoring.")
            print("retrieved_documents: [Phoenix documentation, Phoenix overview article]")
            print("reference_answer: Phoenix is an AI observability platform.")
        queries = load_queries()
        retriever = get_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        for query in cycle(queries):
            with tracer.start_as_current_span("retrieval-qa-query") as span:
                docs = retriever._get_relevant_documents(query, run_manager=None)
                context = "\n".join([doc.page_content for doc in docs])
                prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
                response = llm.invoke(prompt)
            # Phoenix evaluation fields as direct attributes
            span.set_attribute("input", query)
            span.set_attribute("context", context)
            span.set_attribute("output", str(response))
            span.set_attribute("retrieved_documents", [doc.page_content for doc in docs])
            # If you have a reference answer, set it here:
            # span.set_attribute("reference_answer", expected_answer)
            # Add useful attributes to the span for observability
            span.set_attribute("query", query)
            span.set_attribute("context_length", len(context))
            span.set_attribute("response", str(response))
                print("Query")
                print("=====")
                print(query)
            print()
            print("Response")
            print("========")
            print(str(response))
            print()
            print("Span Attributes:")
            print("question:", query)
            print("context:", context)
            print("answer:", str(response))
            print("retrieved_documents:", [doc.page_content for doc in docs])
            print(response)
            print()
