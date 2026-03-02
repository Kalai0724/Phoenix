from google import genai

from phoenix.client import Client
from phoenix.otel import register
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

from phoenix.evals import ClassificationEvaluator, LLM


# =====================================================
# 1. Phoenix tracing
# =====================================================
tracer_provider = register()
GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)


# =====================================================
# 2. Phoenix client
# =====================================================
client = Client()


# =====================================================
# 3. Load existing dataset
# =====================================================
dataset = client.datasets.get_dataset(
    dataset="dataset-eval-demo"
)

print("Loaded dataset:", dataset)


# =====================================================
# 4. Gemini client (TASK model)
# =====================================================
genai_client = genai.Client()


def task(input):
    """
    Runs Gemini to answer the dataset question
    """
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=input["question"],
    )
    return response.text


# =====================================================
# 5. LLM-BASED EVALUATORS
# =====================================================

# ---------- 5.1 Correctness ----------
CORRECTNESS_PROMPT = """
You are an evaluator.

Judge whether the model answer correctly answers the question
based on the reference answer.

Question:
{{input}}

Reference Answer:
{{reference}}

Model Answer:
{{output}}

Respond with ONE label only:
- correct
- partially_correct
- incorrect
"""

correctness_evaluator = ClassificationEvaluator(
    name="correctness",
    prompt_template=CORRECTNESS_PROMPT,
    llm=LLM(provider="google", model="gemini-2.5-flash"),
    choices={
        "correct": 1,
        "partially_correct": 0.5,
        "incorrect": 0,
    },
)


# ---------- 5.2 Relevance ----------
RELEVANCE_PROMPT = """
You are an evaluator.

Determine whether the model answer is relevant to the question.
Ignore factual correctness; focus only on relevance.

Question:
{{input}}

Model Answer:
{{output}}

Respond with ONE label only:
- relevant
- partially_relevant
- irrelevant
"""

relevance_evaluator = ClassificationEvaluator(
    name="relevance",
    prompt_template=RELEVANCE_PROMPT,
    llm=LLM(provider="google", model="gemini-2.5-flash"),
    choices={
        "relevant": 1,
        "partially_relevant": 0.5,
        "irrelevant": 0,
    },
)


# ---------- 5.3 Conciseness ----------
CONCISENESS_PROMPT = """
You are an evaluator.

Evaluate whether the model answer is concise.
An answer is concise if it answers the question clearly
without unnecessary verbosity.

Question:
{{input}}

Model Answer:
{{output}}

Respond with ONE label only:
- concise
- acceptable
- verbose
"""

conciseness_evaluator = ClassificationEvaluator(
    name="conciseness",
    prompt_template=CONCISENESS_PROMPT,
    llm=LLM(provider="google", model="gemini-2.5-flash"),
    choices={
        "concise": 1,
        "acceptable": 0.5,
        "verbose": 0,
    },
)


# =====================================================
# 6. Run experiment WITH MULTIPLE EVALUATORS
# =====================================================
experiment = client.experiments.run_experiment(
    dataset=dataset,
    task=task,
    evaluators=[
        correctness_evaluator,
        relevance_evaluator,
        conciseness_evaluator,
    ],
    experiment_name="dataset-gemini-multi-llm-eval",
)

print("Experiment completed:", experiment)