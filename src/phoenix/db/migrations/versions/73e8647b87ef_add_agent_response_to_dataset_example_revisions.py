"""add agent_response to dataset_example_revisions

Revision ID: 73e8647b87ef
Revises: f1a6b2f0c9d5
Create Date: 2026-03-04 14:40:00.000000

"""

import ast
import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "73e8647b87ef"
down_revision: Union[str, None] = "f1a6b2f0c9d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add the column
    op.add_column(
        "dataset_example_revisions",
        sa.Column("agent_response", sa.String(), nullable=True),
    )

    # 2. Data migration
    connection = op.get_bind()
    
    # We use a broad select and filter in Python to handle JSON parsing safely across dialects
    results = connection.execute(
        sa.text("SELECT id, input FROM dataset_example_revisions")
    ).fetchall()

    for row_id, input_data in results:
        try:
            if isinstance(input_data, str):
                input_dict = json.loads(input_data)
            else:
                input_dict = dict(input_data) if input_data else {}
        except (ValueError, TypeError):
            continue

        if "raw_agent_response" in input_dict:
            raw_response = input_dict.pop("raw_agent_response")
            agent_response = None
            
            if isinstance(raw_response, str):
                try:
                    # Attempt to parse as Python literal (it often has single quotes in user examples)
                    resp_data = ast.literal_eval(raw_response)
                    if isinstance(resp_data, dict):
                        agent_response = resp_data.get("output")
                except (ValueError, SyntaxError):
                    # Fallback to plain string if it's not a dict-like string
                    agent_response = raw_response
            elif isinstance(raw_response, dict):
                agent_response = raw_response.get("output")

            # Update the row: set agent_response and clean up input column
            connection.execute(
                sa.text(
                    "UPDATE dataset_example_revisions "
                    "SET agent_response = :agent_response, input = :input "
                    "WHERE id = :id"
                ),
                {
                    "agent_response": agent_response,
                    "input": json.dumps(input_dict),
                    "id": row_id,
                },
            )


def downgrade() -> None:
    # Restore raw_agent_response to input if possible (best effort)
    connection = op.get_bind()
    results = connection.execute(
        sa.text("SELECT id, input, agent_response FROM dataset_example_revisions WHERE agent_response IS NOT NULL")
    ).fetchall()

    for row_id, input_data, agent_response in results:
        try:
            if isinstance(input_data, str):
                input_dict = json.loads(input_data)
            else:
                input_dict = dict(input_data) if input_data else {}
        except (ValueError, TypeError):
            continue

        input_dict["raw_agent_response"] = json.dumps({"output": agent_response})
        
        connection.execute(
            sa.text("UPDATE dataset_example_revisions SET input = :input WHERE id = :id"),
            {"input": json.dumps(input_dict), "id": row_id},
        )

    op.drop_column("dataset_example_revisions", "agent_response")
