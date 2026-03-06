import sqlite3
import os
import json

db_path = os.path.expanduser('~/.phoenix/phoenix.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('SELECT id, input FROM dataset_example_revisions')
rows = cursor.fetchall()
count = 0

for row_id, input_json in rows:
    try:
        data = json.loads(input_json)
        if 'raw_agent_response' in data:
            resp = data.pop('raw_agent_response')
            new_input = json.dumps(data)
            cursor.execute(
                'UPDATE dataset_example_revisions SET input = ?, agent_response = ? WHERE id = ?',
                (new_input, resp, row_id)
            )
            count += 1
    except Exception as e:
        print(f"Error processing row {row_id}: {e}")

conn.commit()
print(f'Successfully migrated {count} rows')
conn.close()
