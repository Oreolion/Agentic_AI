import json
import utils
import pandas as pd
from dotenv import load_dotenv
import aisuite as ai


_ = load_dotenv()

client = ai.Client()


utils.create_transactions_db()
# utils.print_html(utils.get_schema('products.db'))
def generate_sql(question: str, schema: str, model: str) -> str:
    prompt = f"""
    You are a SQL assistant. Given the schema and the user's question, write a SQL query for SQLite.

    Schema:
    {schema}

    User question:
    {question}

    Respond with the SQL only.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# Example usage of generate_sql

# We provide the schema as a string
schema = """
Table name: transactions
id (INTEGER)
product_id (INTEGER)
product_name (TEXT)
brand (TEXT)
category (TEXT)
color (TEXT)
action (TEXT)
qty_delta (INTEGER)
unit_price (REAL)
notes (TEXT)
ts (DATETIME)
"""

# We ask a question about the data in natural language
question = "Which color of product has the highest total sales?"

utils.print_html(question, title="User Question")

# Generate the SQL query using the specified model
sql_V1 = generate_sql(question, schema, model="openai:gpt-4.1")

# Display the generated SQL query
utils.print_html(sql_V1, title="SQL Query V1")


# Execute the generated SQL query (sql_V1) against the products.db database.
# The result is returned as a pandas DataFrame.
df_sql_V1 = utils.execute_sql(sql_V1, db_path='products.db')

# Render the DataFrame as an HTML table in the notebook.
# This makes the query output easier to read and interpret.
utils.print_html(df_sql_V1, title="Output of SQL Query V1 - ❌ Does NOT fully answer the question")

