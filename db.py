"""
db.py
-----
Handles all SQLite connection and query-execution logic, including the
safety validation that ensures only read-only SELECT statements are
ever executed against the database.
"""

import logging
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("ai_analytics.db")

# Path to the SQLite database file (created by load_data.py)
DB_PATH = Path(__file__).parent / "analytics.db"

# The schema is exposed to the LLM prompt builder so the model always
# knows exactly what tables/columns are available.
DB_SCHEMA = """
Table: sales
Columns:
  - product     (TEXT)   -- name of the product sold
  - revenue     (REAL)   -- revenue generated in USD
  - clicks      (INTEGER)-- number of ad clicks
  - impressions (INTEGER)-- number of ad impressions
  - date        (TEXT)   -- date of the record, format YYYY-MM-DD
"""

# Keywords that are never allowed in a generated query. This is a defense
# in depth measure on top of the "must start with SELECT" check.
FORBIDDEN_KEYWORDS = [
    "delete", "drop", "update", "insert", "alter", "truncate",
    "create", "replace", "attach", "detach", "pragma", "vacuum",
    "--", ";--", "grant", "revoke",
]


class UnsafeQueryError(Exception):
    """Raised when a generated SQL query fails safety validation."""


class QueryExecutionError(Exception):
    """Raised when a syntactically-valid but failing SQL query is run."""


@contextmanager
def get_connection():
    """Context manager yielding a SQLite connection with row access by name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def validate_sql(sql: str) -> str:
    """
    Validates that a SQL string is a safe, read-only SELECT statement.

    Rules enforced:
      1. The query must not be empty.
      2. The query must start with SELECT (case-insensitive, ignoring
         leading whitespace/comments).
      3. The query must not contain any forbidden keywords anywhere
         (DELETE, DROP, UPDATE, INSERT, ALTER, etc.).
      4. Only a single statement is allowed (no stacked queries via ';').

    Returns the cleaned SQL string if valid, otherwise raises
    UnsafeQueryError.
    """
    if not sql or not sql.strip():
        raise UnsafeQueryError("Generated SQL is empty.")

    cleaned = sql.strip().strip("`").strip()
    # Strip a leading "sql" language tag some models add (```sql ... ```)
    cleaned = re.sub(r"^sql\s*", "", cleaned, flags=re.IGNORECASE).strip()

    # Disallow multiple statements (only one trailing semicolon allowed)
    no_trailing = cleaned.rstrip(";").strip()
    if ";" in no_trailing:
        raise UnsafeQueryError("Multiple SQL statements are not allowed.")

    if not re.match(r"(?is)^\s*select\b", no_trailing):
        raise UnsafeQueryError("Only SELECT queries are permitted.")

    lowered = no_trailing.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        # word-boundary match to avoid false positives (e.g. "updated_at")
        pattern = r"\b" + re.escape(keyword) + r"\b" if keyword.isalpha() else re.escape(keyword)
        if re.search(pattern, lowered):
            raise UnsafeQueryError(f"Query contains forbidden keyword: '{keyword}'")

    return no_trailing


def run_query(sql: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    Executes a validated SELECT query and returns (rows_as_dicts, row_count).
    Raises QueryExecutionError on SQLite failures.
    """
    validated_sql = validate_sql(sql)
    logger.info("Executing validated SQL: %s", validated_sql)

    try:
        with get_connection() as conn:
            cursor = conn.execute(validated_sql)
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            return result, len(result)
    except sqlite3.Error as exc:
        logger.error("SQLite execution error: %s", exc)
        raise QueryExecutionError(str(exc)) from exc
