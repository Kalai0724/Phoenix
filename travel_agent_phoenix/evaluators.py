from phoenix.evals import create_evaluator
from phoenix.evals.llm import LLM

llm = LLM(provider="openai", model="gpt-5")

from phoenix.evals import create_classifier
from phoenix.evals.llm import LLM

llm = LLM(provider="openai", model="gpt-5")

ANSWER_RELEVANCE_PROMPT_TEMPLATE = """
You are an impartial evaluator.

Assess how well the answer satisfies the user's query.

Use the following scale:

- excellent → 0.9 to 1.0 (almost perfect)
- good → 0.6 to 0.8 (mostly correct)
- partial → 0.3 to 0.5 (somewhat correct)
- poor → 0.0 to 0.2 (mostly incorrect)

[Query]: {input}
[Answer]: {output}

Return ONLY ONE WORD:
excellent
good
partial
poor
"""

relevancy_evaluator = create_classifier(
    name="ANSWER RELEVANCE",
    llm=llm,
    prompt_template=ANSWER_RELEVANCE_PROMPT_TEMPLATE,
    choices={
        "excellent": 0.9,
        "good": 0.6,
        "partial": 0.4,
        "poor": 0.0,
    },
)

# ------------------------------------------------------------------
# 2. BUDGET CONSISTENCY (Math check, NO reference output)
# ------------------------------------------------------------------
BUDGET_CONSISTENCY_PROMPT_TEMPLATE = """
Evaluate budget consistency.

Use the following scale:

- excellent → fully consistent
- good → minor rounding issues
- partial → noticeable gaps
- poor → clearly incorrect

[Query]: {input}
[Answer]: {output}

Return ONLY ONE WORD:
excellent
good
partial
poor
"""

budget_evaluator = create_classifier(
    name="BUDGET CONSISTENCY",
    llm=llm,
    prompt_template=BUDGET_CONSISTENCY_PROMPT_TEMPLATE,
    choices={
        "excellent": 0.9,
        "good": 0.6,
        "partial": 0.4,
        "poor": 0.0,
    },
)

# ------------------------------------------------------------------
# 3. REFERENCE RELEVANCE (Agent output vs Reference output)
# ------------------------------------------------------------------
REFERENCE_RELEVANCE_PROMPT_TEMPLATE = """
You are evaluating a travel-planning assistant using a reference answer.

User Query:
{input}

Reference Answer (Gold Standard):
{reference_output}

Agent Answer:
{output}

Judge whether the agent answer sufficiently matches the reference answer.

Criteria:
- Covers the same destination and duration
- Includes key sections present in the reference (itinerary, budget, tips)
- Does not miss important information from the reference
- Does not introduce contradictory details

Explain briefly.

Final LABEL (one word only):
- correct
- incorrect
"""

reference_relevance_evaluator = create_classifier(
    name="REFERENCE RELEVANCE",
    llm=llm,
    prompt_template=REFERENCE_RELEVANCE_PROMPT_TEMPLATE,
    choices={"correct": 1.0, "incorrect": 0.0},
)

# ------------------------------------------------------------------
# 4. REFERENCE BUDGET MATCH (Optional but recommended)
# ------------------------------------------------------------------
REFERENCE_BUDGET_PROMPT_TEMPLATE = """
You are evaluating budget alignment between an agent answer and a reference answer.

Reference Budget:
{reference_output}

Agent Budget:
{output}

Determine whether:
- Budget totals align with the reference ranges
- No mathematical contradictions exist
- Major cost categories are not missing

Final LABEL (one word only):
- correct
- incorrect
"""

reference_budget_evaluator = create_classifier(
    name="REFERENCE BUDGET MATCH",
    llm=llm,
    prompt_template=REFERENCE_BUDGET_PROMPT_TEMPLATE,
    choices={"correct": 1.0, "incorrect": 0.0},
)

# ------------------------------------------------------------------
# Export evaluators (import these in experiment.py)
# ------------------------------------------------------------------
EVALUATORS = [
    relevancy_evaluator,
    budget_evaluator,
]