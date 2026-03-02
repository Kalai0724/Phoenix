import os
from phoenix.client import Client
from phoenix.otel import register
from phoenix.evals import ClassificationEvaluator, LLM
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

from rag_agent import rag_agent

# --------------------------------------------------
# Phoenix tracing
# --------------------------------------------------
os.environ["PHOENIX_BASE_URL"] = "http://localhost:6006"

tracer_provider = register()
GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# --------------------------------------------------
# Phoenix client
# --------------------------------------------------
client = Client()

# Dataset already uploaded via Phoenix UI
DATASET_NAME = "company_policy_eval_v2"
dataset = client.datasets.get_dataset(dataset=DATASET_NAME)

print(f"📦 Loaded dataset: {dataset.name}")

# --------------------------------------------------
# Task = RAG bot
# --------------------------------------------------
def task(input):
    question = input["question"]
    return rag_agent(question)

# --------------------------------------------------
# LLM-as-a-Judge evaluator
# --------------------------------------------------
correctness_eval = ClassificationEvaluator(
    name="correctness",
    prompt_template="""
Judge whether the model answer correctly answers the question
based on the reference answer.

Question:
{{input.question}}

Reference Answer:
{{reference}}

Model Answer:
{{output}}

Respond with ONE label only:
- correct
- partially_correct
- incorrect
""",
    llm=LLM(provider="google", model="gemini-2.5-flash"),
    choices={
        "correct": 1.0,
        "partially_correct": 0.5,
        "incorrect": 0.0,
    },
)

# --------------------------------------------------
# Run experiment
# --------------------------------------------------
experiment = client.experiments.run_experiment(
    dataset=dataset,
    task=task,
    evaluators=[correctness_eval],
    experiment_name="company-policy-rag-eval",
)

print("\n✅ Experiment completed")
print("🔗 Experiment ID:", experiment.get("experiment_id"))

# --------------------------------------------------
# Structured logs (IMPORTANT PART)
# --------------------------------------------------
print("\n📋 STRUCTURED EVALUATION LOGS\n")

task_runs = experiment["task_runs"]
evaluation_runs = experiment["evaluation_runs"]

# Map experiment_run_id → evaluation result
eval_by_run_id = {
    eval_run.experiment_run_id: eval_run
    for eval_run in evaluation_runs
}

for idx, task_run in enumerate(task_runs, start=1):
    eval_run = eval_by_run_id.get(task_run["id"])
    example = dataset.examples[idx - 1]

    # 🔐 SAFE reference extraction (handles ALL Phoenix versions)
    if "reference" in example:
        reference_answer = example["reference"]
    elif "expected" in example:
        reference_answer = example["expected"].get("reference_answer")
    else:
        reference_answer = "<REFERENCE NOT FOUND>"

    print("─" * 70)
    print(f"🧪 Test Case {idx}")

    print("\n❓ Question:")
    print(example["input"]["question"])

    print("\n📌 Reference Answer:")
    print(reference_answer)

    print("\n🤖 RAG Output:")
    print(task_run["output"])

    if eval_run:
        result = eval_run.result
        print("\n📊 Evaluation:")
        print(f"Label : {result['label']}")
        print(f"Score : {result['score']}")

        print("\n🧠 Explanation:")
        print(result["explanation"])
    else:
        print("\n⚠️ No evaluation found")

    print("─" * 70)