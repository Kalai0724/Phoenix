from agno.agent import Agent
from agno.models.openai import OpenAIChat
from tools import essential_info, budget_basics, local_flavor

# ---- ORIGINAL AGENT ----
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