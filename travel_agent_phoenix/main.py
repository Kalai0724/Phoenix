from agent import trip_agent

query = """
Plan a 5-day trip to Tokyo.
Focus on food and culture.
Include essential info, budget breakdown, and local experiences.
"""

trip_agent.print_response(query, stream=True)