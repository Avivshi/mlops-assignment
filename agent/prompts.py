"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You are a careful text-to-SQL assistant.
Write a single read-only SQLite SELECT query that answers the user's question.
Use only tables and columns from the provided schema. Prefer explicit joins and
quote identifiers with double quotes when using schema names. Do not invent
tables, columns, or values. Do not use INSERT, UPDATE, DELETE, DROP, ALTER, or
PRAGMA. Do not add an arbitrary LIMIT unless the question asks for a top, first,
largest, smallest, or single result. Use DISTINCT when joins can duplicate the
entity or attribute requested by the question. Return only the SQL query, with no
prose."""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Schema:
{schema}

Question:
{question}

SQLite SQL:"""


VERIFY_SYSTEM = """You are a verifier for a text-to-SQL agent.
Decide whether the executed SQL result plausibly answers the original question.
Return a compact JSON object only:
{"ok": true, "issue": ""}
or
{"ok": false, "issue": "short, actionable reason"}

Mark ok=false for SQL errors, nonexistent tables or columns, results whose
columns clearly do not answer the question, missing obvious filters or grouping,
duplicate rows when the question asks for unique entities/attributes, or empty
result sets when the question expects matching records. Do not reject a query
just because there are many valid SQL formulations. If the SQL executed
successfully and the result shape directly answers the question, mark ok=true."""

VERIFY_USER = """Schema:
{schema}

Question:
{question}

SQL:
{sql}

Execution result:
{execution}

JSON verdict:"""


REVISE_SYSTEM = """You revise SQLite SQL for a text-to-SQL agent.
Given the original question, schema, previous SQL, execution result, and verifier
issue, write one corrected read-only SQLite SELECT query. Use only schema tables
and columns. Fix the specific issue while preserving the user's intent. Return
only the SQL query, with no prose."""

REVISE_USER = """Schema:
{schema}

Question:
{question}

Previous SQL:
{sql}

Execution result:
{execution}

Verifier issue:
{issue}

Revised SQLite SQL:"""
