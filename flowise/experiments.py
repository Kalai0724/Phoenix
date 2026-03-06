import phoenix as px
from phoenix.client import Client
from phoenix.client.experiments import run_experiment
from phoenix.client.experiments import create_evaluator
import logging
import pandas as pd
import json
from typing import Any
import re
import time
from datetime import datetime, timedelta, timezone

# 1. Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = Client()

# The dataset already loaded in Phoenix with 'input' and 'output' (reference) columns
SOURCE_DATASET = "evaluation_dataset_1"

def extract_text(val):
    if isinstance(val, dict):
        return val.get("input", val.get("output", str(val)))
    return str(val)

# 2. EVALUATORS

# A. Word Match Percentage (Regex implementation)
# A. Word Match Percentage (Regex implementation)
def word_match_percent_fn(output: str, expected_output: str) -> float:
    if not output or not expected_output: return 0.0
    actual_words = set(re.findall(r'\w+', str(output).lower()))
    expected_words = set(re.findall(r'\w+', str(expected_output).lower()))
    if not expected_words: return 0.0
    intersection = actual_words.intersection(expected_words)
    union = actual_words.union(expected_words)
    return len(intersection) / len(union)

# Create the Phoenix Evaluators for use in run_experiment
word_match_percent_obj = create_evaluator(word_match_percent_fn)

# B. Correctness (LLM Judge)
eval_prompt = """
Compare the PREDICTED ANSWER with the REFERENCE.
PREDICTED ANSWER: {output}
REFERENCE: {reference}

Provide your response in JSON format with two keys:
1. "label": either "correct" or "incorrect"
2. "explanation": a brief reason for your decision.
"""

def llm_correctness_fn(output: str, reference: str) -> str:
    # Use the evaluation logic defined later
    res = evaluate_correctness(output, reference)
    return res["label"]

llm_correctness_obj = create_evaluator(llm_correctness_fn)

# Note: We call LLM directly in live loop to avoid @create_evaluator overhead for stream
def evaluate_correctness(output: str, reference: str) -> dict[str, Any]:
    from openai import OpenAI
    openai_client = OpenAI()
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": eval_prompt.format(output=output, reference=reference)}],
            response_format={ "type": "json_object" }
        )
        res_data = json.loads(response.choices[0].message.content)
        return {
            "label": res_data.get("label", "incorrect"),
            "explanation": res_data.get("explanation", "")
        }
    except Exception as e:
        logger.error(f"LLM Eval Error: {e}")
        return {"label": "error", "explanation": str(e)}

# 3. LIVE MONITORING LOOP
def start_live_evaluation():
    logger.info(f"📡 Initializing LIVE evaluation against '{SOURCE_DATASET}'...")
    
    # Load dataset once to build a lookup for reference outputs
    try:
        px_dataset = client.datasets.get_dataset(dataset=SOURCE_DATASET)
        df_golden = px_dataset.to_dataframe()
        # Map input query to reference output
        REFERENCE_LOOKUP = {extract_text(row['input']): extract_text(row['output']) for _, row in df_golden.iterrows()}
        logger.info(f"✅ Loaded {len(REFERENCE_LOOKUP)} references for matching.")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return

    processed_span_ids = set()
    logger.info("👀 Watching for new traces in 'default' project... (Press Ctrl+C to stop)")

    while True:
        try:
            # Look for spans from the last 60 seconds
            start_time = datetime.now(timezone.utc) - timedelta(seconds=60)
            spans_df = client.spans.get_spans_dataframe(project_name="default", start_time=start_time)
            
            if not spans_df.empty:
                for span_id, span in spans_df.iterrows():
                    if span_id in processed_span_ids:
                        continue
                    
                    # Extract input/output from the live trace
                    trace_input = extract_text(span.get('attributes.input.value', ""))
                    trace_output = extract_text(span.get('attributes.output.value', ""))
                    
                    # Try to find a match in our Golden Dataset
                    reference = REFERENCE_LOOKUP.get(trace_input)
                    
                    if reference:
                        logger.info(f"🔍 Match found for input: '{trace_input[:50]}...'")
                        
                        # Run Evaluators
                        correctness = evaluate_correctness(trace_output, reference)
                        similarity = word_match_percent_fn(trace_output, reference)
                        
                        # 4. LOG EVALUATIONS BACK TO PHOENIX
                        # This logs to the Span for observability
                        from phoenix.client.__generated__.v1 import SpanAnnotationData
                        client.spans.log_span_annotations(
                            span_annotations=[
                                SpanAnnotationData(
                                    span_id=span_id,
                                    name="correctness",
                                    annotator_kind="LLM",
                                    label=correctness["label"],
                                    explanation=correctness["explanation"]
                                )
                            ]
                        )
                        
                        # 5. TRIGGER A LIVE EXPERIMENT RUN
                        # This will create an entry in the "Experiments" tab
                        logger.info(f"🧪 Triggering Experiment entry for: {trace_input[:30]}...")
                        
                        def task_fn(input_data):
                            return trace_output
                        
                        run_experiment(
                            dataset=px_dataset,
                            task=task_fn, 
                            evaluators=[llm_correctness_obj, word_match_percent_obj],
                            experiment_name=f"Live_Eval_{datetime.now().strftime('%H:%M:%S')}"
                        )
                        
                        logger.info(f"✅ Logged to both Traces and Experiments.")
                        processed_span_ids.add(span_id)
            
        except Exception as e:
            logger.error(f"⚠️ Loop error: {e}")
        
        time.sleep(5)  # Refresh every 5 seconds

if __name__ == "__main__":
    start_live_evaluation()
