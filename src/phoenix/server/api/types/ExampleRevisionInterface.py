import strawberry
from typing import Optional
from strawberry.scalars import JSON


@strawberry.interface
class ExampleRevision:
    """
    Represents an example revision for generative tasks.
    For example, you might have text -> text, text -> labels, etc.
    """

    input: JSON
    output: JSON
    metadata: JSON
    agent_response: Optional[str] = strawberry.field(name="agent_response", default=None)
