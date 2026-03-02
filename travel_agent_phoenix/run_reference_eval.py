from phoenix.client import Client
from openai import OpenAI

client = Client()
llm = OpenAI()

REFERENCE_EVAL_PROMPT = """
You are evaluating an AI travel itinerary.

User Query:
{query}

Reference Itinerary:
{reference}

Agent Itinerary:
{output}

Tasks:
1. Decide if the agent answer aligns with the reference in destination, duration, theme, and budget.
2. Minor differences are acceptable.
3. Major omissions or contradictions are incorrect.

Return JSON only:
{{
  "label": "correct" or "incorrect",
  "score": 1 or 0,
  "explanation": "short explanation"
}}
"""

def evaluate_against_reference(
    span_id: str,
    query: str,
    reference: str,
    output: str,
):
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": REFERENCE_EVAL_PROMPT.format(
                query=query,
                reference=reference,
                output=output,
            ),
        }],
        temperature=0,
    )

    result = eval(response.choices[0].message.content)

    eval_df = [{
        "span_id": span_id,
        "label": result["label"],
        "score": result["score"],
        "explanation": result["explanation"],
    }]

    client.spans.log_span_annotations_dataframe(
        dataframe=eval_df,
        annotation_name="Reference Match",
        annotator_kind="LLM",
    )