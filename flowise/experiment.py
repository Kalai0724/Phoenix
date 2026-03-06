import phoenix as px
from phoenix.client import Client
from phoenix.client.experiments import run_experiment, create_evaluator
from phoenix.otel import register
import logging
import pandas as pd
import json
from typing import Any
from openai import OpenAI

# 1. Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

register(auto_instrument=True)
client = Client()
openai_client = OpenAI()

SOURCE_DATASET_NAME = "demo 1" 
TARGET_DATASET_NAME = "Final_Eval_Dataset_v15"

# 2. REFERENCE OUTPUT 
reference_map = {
    "capital of france?": "Paris",
    "capital of india?": "New Delhi",
    "what is RAG": "RAG (Retrieval-Augmented Generation) is a technique where an AI model retrieves relevant information from a knowledge source to generate accurate responses.",
    "Hi": "Hi there! How can I help you today?"
}

# 3. DATASET PREPARATION
try:
    old_ds = client.datasets.get_dataset(dataset=SOURCE_DATASET_NAME)
    df = old_ds.to_dataframe()

    # Log columns to help verify the current dataset structure
    logger.info(f"Available columns: {df.columns.tolist()}")

    # Find columns even if they are prefixed (e.g., 'metadata.response')
    def find_col(possible_names):
        for name in possible_names:
            if name in df.columns: return name
            for col in df.columns:
                if col.endswith(f".{name}") or col.endswith(f"_{name}"):
                    return col
        return None

    input_col = find_col(['input', 'Input']) or 'input'
    # Use the newly added 'agent_response' column from to_dataframe()
    response_col = 'agent_response' if 'agent_response' in df.columns else (find_col(['response', 'Response']) or 'agent_response')

    logger.info(f"Using input column: '{input_col}', response column: '{response_col}'")

    def extract_query(val):
        if not isinstance(val, str): return str(val)
        val = val.strip()
        if val.startswith('{'):
            try:
                # Handle cases like "{'input': '...'}"
                clean_json = val.replace("'", '"')
                parsed = json.loads(clean_json)
                if isinstance(parsed, dict):
                    return parsed.get("input", val)
            except: pass
        return val

    df['input_clean'] = df[input_col].apply(extract_query)
    df['actual_response'] = df[response_col].fillna("")
    
    # Step C: Reference Output Mapping
    df['reference_output'] = df['input_clean'].map(reference_map).fillna("Paris")

    # Mapping for target dataset
    df['input'] = df['input_clean']
    
    # Refresh Dataset
    try:
        existing_ds = client.datasets.get_dataset(dataset=TARGET_DATASET_NAME)
        client.delete_dataset(dataset_id=existing_ds.id)
    except: pass

    target_dataset = client.datasets.create_dataset(
        name=TARGET_DATASET_NAME,
        dataframe=df,
        input_keys=["input"], 
        output_keys=["reference_output"],
        response_keys=["actual_response"] 
    )
    logger.info("✅ Dataset created with extracted agent responses.")

except Exception as e:
    import traceback
    logger.error(f"❌ Error: {e}")
    traceback.print_exc()
    exit()

# 4. TASK FUNCTION
def task(input, metadata, example):
    # Returns a dict so UI shows {"output": "..."}
    # Use the 'response' field from the Example object
    res = example.response or ""
    return {"output": str(res)}

# 5. EVALUATORS
def match_percentage(input, output, expected) -> float:
    # output is {"output": "..."}
    # expected is {"reference_output": "..."}
    ref = str(expected.get("reference_output", "")).lower()
    act = str(output.get("output", "") if isinstance(output, dict) else output).lower()
    
    act_w, ref_w = set(act.split()), set(ref.split())
    if not ref_w: return 0.0
    intersection = act_w.intersection(ref_w)
    union = act_w.union(ref_w)
    return (len(intersection) / len(union)) * 100 if union else 0.0

@create_evaluator(kind="llm")
def correctness(input, output, expected) -> float:
    # input: {"input": "..."}
    # output: {"output": "..."}
    query = input.get("input", "")
    ref = expected.get("reference_output", "")
    actual = output.get("output", "") if isinstance(output, dict) else output
    
    prompt = (
        f"Query: {query}\n"
        f"Reference Answer: {ref}\n"
        f"Agent Response: {actual}\n\n"
        "Question: Does the Agent Response accurately match the Reference Answer? "
        "Answer ONLY 'Correct' or 'Incorrect'."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}]
    )
    result = resp.choices[0].message.content.strip().lower()
    return 1.0 if "correct" in result else 0.0

# 6. RUN
run_experiment(
    dataset=target_dataset, 
    task=task, 
    experiment_name="UI_Final_Fix_v4",
    evaluators=[match_percentage, correctness],
)