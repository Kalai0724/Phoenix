import pandas as pd
import os
from phoenix.client import Client
from phoenix.evals import OpenAIModel, llm_classify

# 1. Setup Environment
os.environ["OPENAI_API_KEY"] = ""
client = Client(base_url="http://localhost:6006")

def run_advanced_evals():
    print("🔄 Fetching root traces from Phoenix...")
    # Using root_spans_only=True ensures we only evaluate the main 2 traces in your UI
    spans_df = client.spans.get_spans_dataframe(project_identifier="default", root_spans_only=True)

    if spans_df.empty:
        print("❌ No traces found. Make sure you have interacted with the bot.")
        return

    # Prepare input/output for the judge
    eval_data = spans_df.copy()
    eval_data['input'] = eval_data['attributes.input.value'].fillna("N/A")
    eval_data['output'] = eval_data['attributes.output.value'].fillna("N/A")
    
    model = OpenAIModel(model="gpt-4o-mini")

    # --- DEFINE EVALUATOR CONFIGURATIONS ---
    # We define labels and their corresponding numeric scores here
    metrics = {
        "bot_helpfulness": {
            "prompt": "Evaluate helpfulness: EXCELLENT (1.0), PARTIAL (0.5), UNHELPFUL (0.0).",
            "choices": {"EXCELLENT": 1.0, "PARTIAL": 0.5, "UNHELPFUL": 0.0}
        },
        "tone": {
            "prompt": "Rate the tone of the response. Choices: PROFESSIONAL (1.0), FRIENDLY (0.8), NEUTRAL (0.5), ROBOTIC (0.3), RUDE (0.0).",
            "choices": {"PROFESSIONAL": 1.0, "FRIENDLY": 0.8, "NEUTRAL": 0.5, "ROBOTIC": 0.3, "RUDE": 0.0}
        },
        "coherence": {
            "prompt": "Does the response follow logically from the input? Choices: PERFECT (1.0), MINOR_GAP (0.6), ILLOGICAL (0.0).",
            "choices": {"PERFECT": 1.0, "MINOR_GAP": 0.6, "ILLOGICAL": 0.0}
        },
        "conciseness": {
            "prompt": "Is the response the right length? Choices: BALANCED (1.0), WORDY (0.7), TOO_SHORT (0.4), REPETITIVE (0.0).",
            "choices": {"BALANCED": 1.0, "WORDY": 0.7, "TOO_SHORT": 0.4, "REPETITIVE": 0.0}
        }
    }

    print(f"🧠 Running 3 advanced evaluators on {len(eval_data)} traces...")

    for name, config in metrics.items():
        print(f"--- Evaluating {name} ---")
        
        # Build the dynamic template
        template = f"""
        {config['prompt']}
        
        User Input: {{input}}
        Chatbot Response: {{output}}
        
        Return exactly one label from the choices.
        """

        # Run LLM judge
        eval_results = llm_classify(
            data=eval_data,
            template=template,
            model=model,
            rails=list(config['choices'].keys()),
            provide_explanation=True
        )
        eval_results.index = eval_data.index

        # Build annotation list
        annotations = []
        for span_id, row in eval_results.iterrows():
            label = str(row.get('label', '')).upper().strip()
            # Map the text label back to our partial numeric score
            score = config['choices'].get(label, 0.0) 
            
            annotations.append({
                "span_id": span_id,
                "name": name,
                "label": label,
                "score": score,
                "explanation": row.get('explanation', ''),
                "annotator_kind": "LLM"
            })

        # Log to Phoenix
        if annotations:
            client.spans.log_span_annotations_dataframe(
                dataframe=pd.DataFrame(annotations).set_index("span_id"),
                annotation_name=name
            )

    print("🚀 All evaluations complete! Refresh your Phoenix UI.")

if __name__ == "__main__":
    run_advanced_evals()