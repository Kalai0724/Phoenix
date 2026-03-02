import os
from phoenix.client import Client
from phoenix.client.experiments import run_experiment
from phoenix.evals import create_classifier, LLM
from phoenix.experiments.types import EvaluationResult


# Point to Phoenix
os.environ["PHOENIX_BASE_URL"] = "http://localhost:6006"

client = Client()

# Load your dataset
dataset = client.datasets.get_dataset(
    dataset="company_policies_eval"
)

# Use Flowise output
def flowise_task(example):
    return example.output

# Hallucination judge
hallucination_prompt = """
Ground truth answer:
{reference_output}

Chatbot response:
{output}

If the chatbot response adds information not present in the ground truth,
mark it as HALLUCINATED.

If it is fully supported by the ground truth,
mark it as GROUNDED.

Return only one label: grounded or hallucinated.
"""

hallucination_judge = create_classifier(
    name="policy-hallucination-eval",
    prompt_template=hallucination_prompt,
    llm=LLM(model="gpt-3.5-turbo", provider="openai"),
    choices={"grounded": 1.0, "hallucinated": 0.0},
)



def hallucination_eval(input, output, reference):
    result = hallucination_judge.evaluate({
        "output": output,
        "reference_output": reference
    })[0]

    return EvaluationResult(
        score=result.score,
        label=result.label,
        explanation=result.explanation,
    )

# Run experiment
run_experiment(
    dataset=dataset,
    task=flowise_task,
    evaluators=[hallucination_eval],
    experiment_name="flowise-vs-policy-hallucination",
)

print("✅ Flowise hallucination evaluation completed")