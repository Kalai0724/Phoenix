import os
from getpass import getpass

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = globals().get("PHOENIX_COLLECTOR_ENDPOINT") or getpass(
    "🔑 Enter your Phoenix Endpoint: "
)

os.environ["PHOENIX_API_KEY"] = globals().get("PHOENIX_API_KEY") or getpass(
    "🔑 Enter your Phoenix API Key: "
)

os.environ["OPENAI_API_KEY"] = globals().get("OPENAI_API_KEY") or getpass(
    "🔑 Enter your OpenAI API Key: "
)

os.environ["TAVILY_API_KEY"] = globals().get("TAVILY_API_KEY") or getpass(
    "🔑 Enter your Tavily API Key: "
)

from phoenix.otel import register

tracer_provider = register(auto_instrument=True, project_name="python-phoenix-tutorial")

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

# --- Helper functions for tools ---
import httpx


@tracer.chain(name="search-api")
def _search_api(query: str) -> str | None:
    """Try Tavily search first, fall back to None."""
    api_key = os.getenv("TAVILY_API_KEY")

    resp = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": 3,
            "search_depth": "basic",
            "include_answer": True,
        },
        timeout=8,
    )
    data = resp.json()
    answer = data.get("answer", "") or ""
    snippets = [item.get("content", "") for item in data.get("results", [])]

    combined = " ".join([answer] + snippets).strip()
    return combined[:400] if combined else None


@tracer.chain(name="weather-api")
def _weather(dest):
    g = httpx.get(f"https://geocoding-api.open-meteo.com/v1/search?name={dest}")
    if g.status_code != 200 or not g.json().get("results"):
        return ""
    lat, lon = g.json()["results"][0]["latitude"], g.json()["results"][0]["longitude"]
    w = httpx.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    ).json()
    cw = w.get("current_weather", {})
    return f"Weather now: {cw.get('temperature')}°C, wind {cw.get('windspeed')} km/h."

from agno.tools import tool


@tool
def essential_info(destination: str) -> str:
    """Get essential info using Search and Weather APIs"""
    parts = []

    q = f"{destination} travel essentials weather best time top attractions etiquette"
    s = _search_api(q)
    if s:
        parts.append(f"{destination} essentials: {s}")
    else:
        parts.append(
            f"{destination} is a popular travel destination. Expect local culture, cuisine, and landmarks worth exploring."
        )

    weather = _weather(destination)
    if weather:
        parts.append(weather)

    return f"{destination} essentials:\n" + "\n".join(parts)


@tool
def budget_basics(destination: str, duration: str) -> str:
    """Summarize travel cost categories."""
    q = f"{destination} travel budget average daily costs {duration}"
    s = _search_api(q)
    if s:
        return f"{destination} budget ({duration}): {s}"
    return f"Budget for {duration} in {destination} depends on lodging, meals, transport, and attractions."


@tool
def local_flavor(destination: str, interests: str = "local culture") -> str:
    """Suggest authentic local experiences."""
    q = f"{destination} authentic local experiences {interests}"
    s = _search_api(q)
    if s:
        return f"{destination} {interests}: {s}"
    return f"Explore {destination}'s unique {interests} through markets, neighborhoods, and local eateries."

from agno.agent import Agent
from agno.models.openai import OpenAIChat

# --- Main Agent ---
trip_agent = Agent(
    name="TripPlanner",
    role="AI Travel Assistant",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=(
        "You are a friendly and knowledgeable travel planner. "
        "Combine multiple tools to create a trip plan including essentials, budget, and local flavor. "
        "Keep the tone natural, clear, and under 1000 words."
    ),
    markdown=True,
    tools=[essential_info, budget_basics, local_flavor],
)

# --- Example usage ---
destination = "Tokyo"
duration = "5 days"
interests = "food, culture"

query = f"""
Plan a {duration} trip to {destination}.
Focus on {interests}.
Include essential info, budget breakdown, and local experiences.
"""


def agent_task(query):
    trip_agent.print_response(query, stream=True)


agent_task(query)

from phoenix.client import Client

client = Client()

from phoenix.client import Client

client = Client()

import pandas as pd

# --- Example queries ---
queries = [
    "Plan a 7-day trip to Italy focused on art, history, and local food. Include essential travel info, a budget estimate, and key attractions in Rome, Florence, and Venice.",
    "Create a 4-day itinerary for Seoul centered on K-pop, fashion districts, and street food. Include transportation tips and a mid-range budget.",
    "Plan a romantic 5-day getaway to Paris with emphasis on museums, wine tasting, and scenic walks. Provide cost estimates and essential travel notes.",
    "Design a 3-day budget trip to Mexico City focusing on food markets, archaeological sites, and nightlife. Include daily cost breakdowns.",
    "Prepare a 6-day itinerary for New Zealand’s South Island with a focus on outdoor adventure, hikes, and photography spots. Include travel logistics and gear essentials.",
    "Plan a 10-day trip across Spain, hitting Barcelona, Madrid, and Seville. Focus on architecture, tapas, and cultural festivals. Include a detailed budget.",
    "Create a 5-day family-friendly itinerary for Singapore with theme parks, nature activities, and kid-friendly dining. Include entry fees and transit costs.",
    "Plan a 4-day luxury spa and relaxation trip to Bali. Include premium resorts, wellness activities, and a high-end budget.",
    "Design a 7-day solo backpacking trip through Thailand with hostels, street food, and cultural attractions. Provide safety essentials and budget breakdown.",
    "Create a 3-day weekend itinerary for New York City focusing on art galleries, rooftop restaurants, and iconic attractions. Include estimated costs.",
]

dataset_df = pd.DataFrame(data={"input": queries})

dataset = client.datasets.create_dataset(
    dataframe=dataset_df, name="travel-questions", input_keys=["input"]
)https://storage.googleapis.com/arize-phoenix-assets/assets/images/end-to-end-python-tutorial-dataset.png

ANSWER_RELEVANCE_PROMPT_TEMPLATE = """
You will be given a travel-planning query and an itinerary answer. Your task is to decide whether
the answer correctly follows the user's instructions. An answer is "incorrect" if it contradicts,
ignores, or fails to include required elements from the query (such as trip length, destination,
themes, budget details, or essential info). It is also "incorrect" if it adds irrelevant or
contradictory details.

    [BEGIN DATA]
    ************
    [Query]: {{input}}
    ************
    [Answer]: {{output}}
    ************
    [END DATA]

Explain step-by-step how you determined your judgment. Then provide a final LABEL:
- Use "correct" if the answer follows the query accurately and fully.
- Use "incorrect" if it deviates from the query or omits required information.

Your final output must be only one word: "correct" or "incorrect".
"""

from phoenix.evals import create_classifier
from phoenix.evals.llm import LLM

llm = LLM(provider="openai", model="gpt-5")

relevancy_evaluator = create_classifier(
    name="ANSWER RELEVANCE",
    llm=llm,
    prompt_template=ANSWER_RELEVANCE_PROMPT_TEMPLATE,
    choices={"correct": 1.0, "incorrect": 0.0},
)

BUDGET_CONSISTENCY_PROMPT_TEMPLATE = """
You will be given a travel-planning query and an itinerary answer. Your task is to determine whether
the answer provides a consistent and mathematically coherent budget. An answer is "incorrect" if:
- The summed minimum costs of all listed budget categories exceed the stated minimum total estimate.
- The summed maximum costs of all listed budget categories exceed the stated maximum total estimate.
- The total estimate claims a range that cannot be derived from (or contradicted by) the itemized ranges.
- The answer lists budget items but provides a total that is not numerically aligned with them.
- The answer contradicts itself regarding pricing or cost ranges.

    [BEGIN DATA]
    ************
    [Query]: {{input}}
    ************
    [Answer]: {{output}}
    ************
    [END DATA]

Explain step-by-step how you evaluated the itemized costs and the final total, including whether
the ranges mathematically match. Then provide a final LABEL:
- Use "correct" if the budget totals are consistent with the itemized values.
- Use "incorrect" if the totals contradict or cannot be derived from the itemized values.

Your final output must be only one word: "correct" or "incorrect".
"""

llm = LLM(provider="openai", model="gpt-5")

budget_evaluator = create_classifier(
    name="BUDGET CONSISTENCY",
    llm=llm,
    prompt_template=BUDGET_CONSISTENCY_PROMPT_TEMPLATE,
    choices={"correct": 1.0, "incorrect": 0.0},
)

def agent_task(input):
    query = input["input"]
    response = trip_agent.run(query, stream=False)
    return response.content

from phoenix.client.experiments import run_experiment

experiment = run_experiment(
    dataset=dataset,
    task=agent_task,
    experiment_name="inital run",
    evaluators=[relevancy_evaluator, budget_evaluator],
)

#Run Experiment

def agent_task(input):
    query = input["input"]
    response = trip_agent.run(query, stream=False)
    return response.content


from phoenix.client.experiments import run_experiment

experiment = run_experiment(
    dataset=dataset,
    task=agent_task,
    experiment_name="inital run",
    evaluators=[relevancy_evaluator, budget_evaluator],
)

#Iterate and Re-Run Experiment


# --- Main Agent with Updated Instructions ---
trip_agent = Agent(
    name="TripPlanner",
    role="AI Travel Assistant",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=(
        "You are a friendly and knowledgeable travel planner. "
        "Combine multiple tools to create a trip plan including essentials, budget, and local flavor. "
        "Keep the tone natural, clear, and under 1000 words. "
        "When providing budget details: Ensure the final total budget range is mathematically consistent with the sum of the itemized ranges."
    ),
    markdown=True,
    tools=[essential_info, budget_basics, local_flavor],
)

experiment = run_experiment(
    dataset=dataset,
    task=agent_task,
    experiment_name="updated agent prompt to improve budget",
    evaluators=[relevancy_evaluator, budget_evaluator],
)