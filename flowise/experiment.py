import pandas as pd
from tqdm import tqdm
from phoenix.client import Client
from phoenix.evals import create_classifier, LLM

# --- CORRECT IMPORT FOR LOGGING ---
# This is the modern location for the logging function
from phoenix.trace.evaluations import log_evaluations

# 1. Initialize Phoenix
px_client = Client()

# 2. Get the dataset and turn it into a list of dictionaries
dataset = px_client.datasets.get_dataset(dataset="flowise")
df = dataset.to_dataframe()

# 3. Setup LLM Judge
llm = LLM(model="gpt-4o", provider="openai")
CORRECTNESS_TEMPLATE = """
Evaluate the agent response.
User Query: {input}
Agent Response: {output}
Respond with only "correct" or "incorrect".
"""
classifier = create_classifier(
    name="correctness",
    prompt_template=CORRECTNESS_TEMPLATE,
    llm=llm,
    choices={"correct": 1.0, "incorrect": 0.0}
)

# 4. RUN EVALUATION MANUALLY
print("Evaluating Dataset Rows...")
eval_results = []

for example_id, row in tqdm(df.iterrows(), total=len(df)):
    # We pass the data directly to the classifier
    # 'output' comes from your dataset's Reference Output column
    eval_df = pd.DataFrame([{"output": row['output'], "input": row['input']}])
    
    try:
        results = classifier.evaluate(eval_df)
        res = results[0] 
        
        # We prepare the evaluation record
        eval_results.append({
            "name": "correctness",
            "example_id": example_id,  # This links it to the row
            "label": getattr(res, 'label', 'incorrect'),
            "score": getattr(res, 'score', 0.0),
            "explanation": getattr(res, 'explanation', "")
        })
    except Exception as e:
        print(f"Row {example_id} failed: {e}")

# 5. LOG DIRECTLY
if eval_results:
    # This pushes the scores directly into the dataset's 'Evaluators' tab
    log_evaluations(eval_results, client=px_client)
    print("\n✅ Success! Check the 'Evaluators' tab in the flowise dataset.")
else:
    print("\n❌ No results to log.")