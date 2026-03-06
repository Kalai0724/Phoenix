import phoenix as px
from phoenix.client import Client
from phoenix.client.experiments import run_experiment, create_evaluator
import logging
import pandas as pd
import json
from typing import Any
import re

# 1. Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = Client()

# This is the dataset already loaded in Phoenix with 'input' and 'output'
SOURCE_DATASET = "evaluation_dataset_1"

# 3. DATASET & TRACE PREPARATION
try:
    # 1. Fetch the Dataset
    dataset = client.datasets.get_dataset(dataset=SOURCE_DATASET)
    df_golden = dataset.to_dataframe()

    def extract_text(val):
        if isinstance(val, dict): return val.get("input", str(val))
        return str(val)

    # 2. Fetch Trace History from Phoenix Project
    # Correct API call: client.spans.get_spans_dataframe(project_name="default")
    spans_df = client.spans.get_spans_dataframe(project_name="default")
    
    global TRACE_LOOKUP
    TRACE_LOOKUP = {}
    
    if not spans_df.empty:
        # Create a mapping of Input -> Actual Agent Response from Traces
        for _, span in spans_df.iterrows():
            s_in = extract_text(span.get('attributes.input.value', ""))
            s_out = extract_text(span.get('attributes.output.value', ""))
            if s_in and s_out:
                TRACE_LOOKUP[s_in] = s_out

    logger.info(f"✅ Evaluation ready for dataset '{SOURCE_DATASET}'.")
except Exception as e:
    logger.error(f"❌ Setup Error: {e}")
    raise e

# 4. TASK FUNCTION
def task(input_data):
    """Retrieves the pre-existing trace response matching the input."""
    # Phoenix passes the dataset row. Extract query from the 'input' column.
    # The 'input' key from input_data maps directly to the dataset's 'input' column.
    raw_input = input_data.get('input', "")
    query = extract_text(raw_input)
    return str(TRACE_LOOKUP.get(query, "No matching trace found in Phoenix project"))

# 5. EVALUATORS

# A. Jaccard Similarity (Percent Match)
def word_match_percent(output: str, expected: dict[str, Any]) -> float:
    if output is None: return 0.0
    reference_text = str(expected.get("output", "")).lower()
    actual_text = str(output).lower()
    actual_words = set(re.findall(r'\w+', actual_text))
    expected_words = set(re.findall(r'\w+', reference_text))
    if not expected_words: return 0.0
    intersection = actual_words.intersection(expected_words)
    union = actual_words.union(expected_words)
    return len(intersection) / len(union)

# B. Correctness (LLM Judge)
eval_prompt = """
Compare the PREDICTED ANSWER with the REFERENCE.
PREDICTED ANSWER: {output}
REFERENCE: {reference}

Provide your response in JSON format with two keys:
1. "label": either "correct" or "incorrect"
2. "explanation": a brief reason for your decision.
"""

@create_evaluator(kind="llm")
def llm_correctness(output: str, expected: dict[str, Any]) -> dict[str, Any]:
    from openai import OpenAI
    openai_client = OpenAI()
    ref = expected.get("output", "")
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": eval_prompt.format(output=output, reference=ref)}],
        response_format={ "type": "json_object" }
    )
    res_data = json.loads(response.choices[0].message.content)
    return {
        "label": res_data.get("label", "incorrect"),
        "score": 1.0 if res_data.get("label", "").lower() == "correct" else 0.0,
        "explanation": res_data.get("explanation", "")
    }

# 6. RUN EXPERIMENT ON EXISTING DATASET
logger.info(f"🚀 Running evaluation on tracks within '{SOURCE_DATASET}'...")
run_experiment(
    dataset=dataset, 
    task=task, 
    experiment_name="Trace_v_Dataset_Evaluation",
    evaluators=[llm_correctness, word_match_percent],
)

logger.info("✨ Process Finished. Check Phoenix UI.")
