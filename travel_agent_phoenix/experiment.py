import pandas as pd
from phoenix.client import Client
from opentelemetry.trace import get_current_span, format_span_id

# Internal Imports
from dataset import dataset 
from evaluators import relevancy_evaluator, budget_evaluator
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from tools import essential_info, budget_basics, local_flavor

# Initialize the Phoenix Client
client = Client()

# 1. Setup the Agent
updated_trip_agent = Agent(
    name="TripPlanner",
    role="AI Travel Assistant",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=(
        "You are a friendly travel planner. "
        "Ensure budget totals are mathematically consistent."
    ),
    markdown=True,
    tools=[essential_info, budget_basics, local_flavor],
)

# 2. Agent Task with Correct ID Capture
captured_runs = []

def agent_task(input):
    query = input["input"]
    response = updated_trip_agent.run(query, stream=False)
    print("Agent response:\n", response.content)  # Print agent response in terminal
    # Get the current span to extract the ID
    current_span = get_current_span()
    span_context = current_span.get_span_context()
    span_id = format_span_id(span_context.span_id)
    # Store it for later feedback
    captured_runs.append({
        "span_id": span_id,
        "input": query
    })
    return response.content

# 3. Run the Experiment
from phoenix.client.experiments import run_experiment

print("🚀 Running experiment...")
experiment_iteration = run_experiment(
    dataset=dataset,
    task=agent_task,
    experiment_name="budget_fix_with_feedback_v2",
    evaluators=[
        relevancy_evaluator,
        budget_evaluator,
    ],
)

# 4. Log Programmatic Feedback (The Correct Way)
print(f"✅ Logging feedback for {len(captured_runs)} traces...")

# Phoenix Client expects a specific list of dictionaries for log_span_annotations
annotations_to_log = []

for run in captured_runs:
    annotations_to_log.append({
        "span_id": run["span_id"],
        "name": "user_feedback",
        "annotator_kind": "HUMAN",
        "result": {
            "label": "thumbs-up",
            "score": 1,
            "explanation": "Simulated user feedback"
        },
        "metadata": {"source": "experiment_script"}
    })

# Use the client directly to log annotations
client.annotations.log_span_annotations(span_annotations=annotations_to_log)

print("✨ Done! Check your Phoenix UI. The feedback is now linked to your experiment results.")