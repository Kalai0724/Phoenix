from phoenix.evals import create_classifier
from phoenix.evals.llm import LLM

REFERENCE_PROMPT = """
You are evaluating a travel itinerary.

Compare the agent output to the reference output.

Check:
- Same trip duration
- Same destinations
- Budget consistency
- Required elements included

[Query]
{input}

[Reference]
{reference_output}

[Agent Output]
{output}

Return ONLY:
correct
or
incorrect
"""

llm = LLM(provider="openai", model="gpt-5")

reference_evaluator = create_classifier(
    name="REFERENCE_MATCH",
    llm=llm,
    prompt_template=REFERENCE_PROMPT,
    choices={"correct": 1.0, "incorrect": 0.0},
)