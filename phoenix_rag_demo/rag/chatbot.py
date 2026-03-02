import os
from phoenix.otel import register
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry import trace

from rag_agent import rag_agent

# --------------------------------------------------
# Phoenix setup
# --------------------------------------------------
os.environ["PHOENIX_BASE_URL"] = "http://localhost:6006"

tracer_provider = register()
GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

tracer = trace.get_tracer(__name__)

print("🤖 Company Policy Chatbot")
print("Type 'exit' to quit.\n")

# --------------------------------------------------
# Chat loop WITH tracing
# --------------------------------------------------
while True:
    user_query = input("You: ").strip()

    if user_query.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    # 🔑 THIS IS THE KEY FIX
    with tracer.start_as_current_span(
        "company-policy-chatbot",
        attributes={
            "input.question": user_query,
        },
    ) as span:
        answer = rag_agent(user_query)

        # store output explicitly
        span.set_attribute("output.answer", answer)

    print(f"\nBot: {answer}\n")