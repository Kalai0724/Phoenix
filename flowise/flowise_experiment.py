from phoenix.client import Client
from phoenix.experiments import run_experiment
from phoenix.evals import create_classifier, LLM
from phoenix.experiments.types import EvaluationResult
from phoenix.evals.rate_limiters import RateLimitError
import time

# -----------------------------------
# 1. LOAD DATASET
# -----------------------------------
DATASET_NAME = "test1"

px_client = Client()
dataset = px_client.datasets.get_dataset(dataset=DATASET_NAME)

print(f"📦 Using dataset: {dataset.name}")

# -----------------------------------
# 2. TASK
# -----------------------------------
def flowise_task(input: dict):
    return input.get("output") or input

# -----------------------------------
# 3. HALLUCINATION JUDGE
# -----------------------------------
HALLUCINATION_PROMPT = """
You are evaluating a chatbot response for hallucinations.

Mark GROUNDED if the response is fully supported by the input.
Mark HALLUCINATED if it adds facts not present in the input.

Conversation / Context:
{input}

Assistant Response:
{output}

Return only one label: grounded or hallucinated.
"""

hallucination_judge = create_classifier(
    name="flowise-hallucination-judge",
    prompt_template=HALLUCINATION_PROMPT,
    llm=LLM(model="gpt-3.5-turbo", provider="openai"),
    choices={
        "grounded": 1.0,
        "hallucinated": 0.0,
    },
)

# -----------------------------------
# 4. FAIL-SAFE EVALUATOR (NO CRASH)
# -----------------------------------
def hallucination(input, output):
    for attempt in range(3):
        try:
            result = hallucination_judge.evaluate({
                "input": input,
                "output": output
            })[0]

            # slow down between calls
            time.sleep(2)

            return EvaluationResult(
                score=result.score,
                label=result.label,
                explanation=result.explanation
            )

        except RateLimitError:
            wait = 3
            print(f"⚠️ Rate limited, waiting {wait}s...")
            time.sleep(wait)

    # ✅ NEVER crash Phoenix — return neutral result
    return EvaluationResult(
        score=0.0,
        label="hallucinated",
        explanation="Evaluation skipped due to repeated rate limits"
    )

# -----------------------------------
# 5. RUN EXPERIMENT (LEGACY SAFE)
# -----------------------------------
run_experiment(
    dataset,
    flowise_task,
    evaluators=[hallucination],
    experiment_name="flowise-hallucination-eval",
    experiment_description="Hallucination evaluation for Flowise chatbot responses",
)

print("✅ Flowise hallucination evaluation completed successfully")