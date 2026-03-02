import os
from getpass import getpass

# ---- Phoenix Localhost ----
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = (
    globals().get("PHOENIX_COLLECTOR_ENDPOINT")
    or "http://localhost:6006"
)

os.environ["OPENAI_API_KEY"] = (
    os.getenv("OPENAI_API_KEY") or getpass("🔑 Enter your OpenAI API Key: ")
)

os.environ["TAVILY_API_KEY"] = (
    os.getenv("TAVILY_API_KEY") or getpass("🔑 Enter your Tavily API Key: ")
)

from phoenix.otel import register
from opentelemetry import trace

tracer_provider = register(
    auto_instrument=True,
    project_name="python-phoenix-tutorial",
)

tracer = trace.get_tracer(__name__)