import phoenix as px
from phoenix.client import Client
from phoenix.experiments import run_experiment

from task import classify_ticket_task
from evaluators import tool_call_accuracy

px_client = Client()

dataset = px_client.datasets.get_dataset(
    dataset="support-ticket-queries"
)

run_experiment(
    dataset,
    classify_ticket_task,
    evaluators=[tool_call_accuracy],
    experiment_name="ticket classification accuracy",
    experiment_description="Evaluating classification accuracy using code-based evaluator"
)