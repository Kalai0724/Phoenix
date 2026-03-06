from typing import Optional

import strawberry
from strawberry import UNSET
from strawberry.relay import GlobalID
from strawberry.scalars import JSON


@strawberry.input
class DatasetExampleInput:
    input: JSON
    output: JSON
    metadata: JSON
    agent_response: Optional[str] = UNSET
    span_id: Optional[GlobalID] = UNSET
