import pandas as pd
import os
import time
import asyncio
from phoenix.client import Client
from phoenix.evals import OpenAIModel, llm_classify

# 1. Setup Environment
# PRO TIP: It's safer to set this in your terminal: set OPENAI_API_KEY=sk-...
os.environ["OPENAI_API_KEY"] = "" 

client = Client(base_url="http://localhost:6006")

def run_smart_live_evaluator():
    print("🚀 Smart Live Evaluator Started... (Press Ctrl+C to stop)")
    model = OpenAIModel(model="gpt-4o-mini")
    
    metrics = {
        "bot_helpfulness": {
            "prompt": "Evaluate helpfulness: EXCELLENT (1.0), PARTIAL (0.5), UNHELPFUL (0.0).",
            "choices": {"EXCELLENT": 1.0, "PARTIAL": 0.5, "UNHELPFUL": 0.0}
        }
    }

    try:
        while True:
            # 1. Fetch root traces
            spans_df = client.spans.get_spans_dataframe(project_identifier="default", root_spans_only=True)
            
            if spans_df.empty:
                time.sleep(5)
                continue

            # 2. Filter logic to skip already-evaluated traces
            try:
                existing_annotations = client.spans.get_span_annotations_dataframe(project_identifier="default")
                if not existing_annotations.empty:
                    # Filter for traces that DO NOT have 'bot_helpfulness' yet
                    done_ids = existing_annotations[existing_annotations['name'] == 'bot_helpfulness'].index.unique()
                    new_spans = spans_df[~spans_df.index.isin(done_ids)].copy()
                else:
                    new_spans = spans_df.copy()
            except Exception:
                new_spans = spans_df.copy()

            if new_spans.empty:
                print("😴 No new traces. Waiting...", end="\r")
                time.sleep(10) # Longer sleep to prevent event loop congestion
                continue

            print(f"\n✨ Found {len(new_spans)} NEW trace(s). Evaluating...")
            
            new_spans['input'] = new_spans['attributes.input.value'].fillna("N/A")
            new_spans['output'] = new_spans['attributes.output.value'].fillna("N/A")

            # 3. Process each metric
            for name, config in metrics.items():
                template = f"{config['prompt']}\nUser Input: {{input}}\nChatbot Response: {{output}}\nReturn ONLY the label."
                
                # We wrap the evaluation to ensure it finishes before moving on
                eval_results = llm_classify(
                    data=new_spans,
                    template=template,
                    model=model,
                    rails=list(config['choices'].keys()),
                    provide_explanation=True
                )
                eval_results.index = new_spans.index

                annotations = []
                for span_id, row in eval_results.iterrows():
                    label = str(row.get('label', '')).upper().strip()
                    score = config['choices'].get(label, 0.0) 
                    annotations.append({
                        "span_id": span_id,
                        "name": name,
                        "label": label,
                        "score": score,
                        "explanation": row.get('explanation', ''),
                        "annotator_kind": "LLM"
                    })

                client.spans.log_span_annotations_dataframe(
                    dataframe=pd.DataFrame(annotations).set_index("span_id"),
                    annotation_name=name
                )
            
            print(f"✅ Sync complete for {len(new_spans)} traces.")
            # Give the system time to close async connections properly
            time.sleep(5) 

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"\n⚠️ An error occurred: {e}")
        print("Retrying in 10 seconds...")
        time.sleep(10)
        run_smart_live_evaluator() # Restart on crash

if __name__ == "__main__":
    run_smart_live_evaluator()