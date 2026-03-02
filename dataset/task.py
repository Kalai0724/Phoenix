def classify_ticket(query: str) -> str:
    """
    Replace this with your real LLM / agent logic
    """
    query = query.lower()

    if "charge" in query or "billing" in query:
        return "billing"
    if "crash" in query or "error" in query:
        return "technical"
    if "account" in query or "email" in query or "login" in query:
        return "account"

    return "other"


def classify_ticket_task(input: dict) -> str:
    """
    Task function used by Phoenix
    """
    query = input.get("query")
    return classify_ticket(query)