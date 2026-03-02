import os
import httpx
from agno.tools import tool
from config import tracer


@tracer.chain(name="search-api")
def _search_api(query: str) -> str | None:
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
    g = httpx.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={dest}"
    )
    if g.status_code != 200 or not g.json().get("results"):
        return ""

    lat = g.json()["results"][0]["latitude"]
    lon = g.json()["results"][0]["longitude"]

    w = httpx.get(
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    ).json()

    cw = w.get("current_weather", {})
    return f"Weather now: {cw.get('temperature')}°C, wind {cw.get('windspeed')} km/h."


@tool
def essential_info(destination: str) -> str:
    parts = []

    q = f"{destination} travel essentials weather best time top attractions etiquette"
    s = _search_api(q)

    if s:
        parts.append(f"{destination} essentials: {s}")
    else:
        parts.append(
            f"{destination} is a popular travel destination."
        )

    weather = _weather(destination)
    if weather:
        parts.append(weather)

    return f"{destination} essentials:\n" + "\n".join(parts)


@tool
def budget_basics(destination: str, duration: str) -> str:
    q = f"{destination} travel budget average daily costs {duration}"
    s = _search_api(q)

    if s:
        return f"{destination} budget ({duration}): {s}"

    return (
        f"Budget for {duration} in {destination} depends on lodging, meals, transport, and attractions."
    )


@tool
def local_flavor(destination: str, interests: str = "local culture") -> str:
    q = f"{destination} authentic local experiences {interests}"
    s = _search_api(q)

    if s:
        return f"{destination} {interests}: {s}"

    return (
        f"Explore {destination}'s unique {interests} through markets, neighborhoods, and local eateries."
    )