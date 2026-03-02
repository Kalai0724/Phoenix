from phoenix.client import Client
from phoenix.otel import register
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from phoenix.evals import ClassificationEvaluator, LLM
from rag_agent import rag_agent

# Enable tracing
tracer_provider = register()
GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = Client()

dataset = client.datasets.get_dataset(
    dataset="company-policy-rag-dataset"
)

def task(input):
    return rag_agent(input["question"])

EVAL_PROMPT = """
Judge whether the answer is correct based on the reference.

Question:
{{input}}

Reference Answer:
{{reference}}

Model Answer:
{{output}}

Respond with ONE label:
- correct
- partially_correct
- incorrect
"""

correctness_eval = ClassificationEvaluator(
    name="rag_correctness",
    prompt_template=EVAL_PROMPT,
    llm=LLM(provider="google", model="gemini-2.5-flash"),
    choices={
        "correct": 1,
        "partially_correct": 0.5,
        "incorrect": 0
    }
)

experiment = client.experiments.run_experiment(
    dataset=dataset,
    task=task,
    evaluators=[correctness_eval],
    experiment_name="company-rag-evaluation"
)

print("✅ Evaluation completed")