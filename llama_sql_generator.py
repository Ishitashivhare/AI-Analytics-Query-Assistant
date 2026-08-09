"""
llama_sql_generator.py
-----------------------
Builds the prompt sent to the local Ollama LLaMA 3 model and handles
both the non-streaming and streaming calls to the Ollama API
(http://localhost:11434/api/generate).
"""

import json
import logging
from typing import AsyncGenerator

import httpx

from db import DB_SCHEMA

logger = logging.getLogger("ai_analytics.llm")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
REQUEST_TIMEOUT = 60.0


def build_sql_prompt(question: str) -> str:
    """
    Builds a strict, rule-enforcing prompt that instructs the model to
    convert a natural-language question into a single safe SQL SELECT
    statement against the known schema.
    """
    prompt = f"""You are a senior data analyst who writes precise SQLite SQL queries.

DATABASE SCHEMA:
{DB_SCHEMA}

STRICT RULES (follow all of them):
1. ONLY generate a single SELECT statement. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or any other data-modifying statement.
2. Use ONLY the table and columns listed in the schema above. Do not invent columns or tables.
3. Return ONLY the raw SQL query. No explanations, no markdown formatting, no code fences, no comments — just the SQL statement itself.
4. The query must be valid SQLite syntax.
5. If the question cannot be answered with the given schema, return exactly: SELECT 'UNSUPPORTED_QUESTION' AS error;

USER QUESTION:
{question}

SQL QUERY:"""
    return prompt


async def generate_sql(question: str) -> str:
    """
    Calls the Ollama /api/generate endpoint (non-streaming) and returns
    the raw SQL text produced by the model.
    Raises httpx.HTTPError / httpx.RequestError on connectivity failure.
    """
    prompt = build_sql_prompt(question)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }

    logger.info("Requesting SQL generation from Ollama for question: %s", question)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        raw_sql = data.get("response", "").strip()

    # Strip common markdown/code-fence wrapping the model might still add
    cleaned = raw_sql.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("sql\n", "", 1).replace("sql\r\n", "", 1)
    cleaned = cleaned.strip().strip("`").strip()

    logger.info("Generated SQL: %s", cleaned)
    return cleaned


async def stream_llm_response(question: str) -> AsyncGenerator[str, None]:
    """
    Streams tokens from Ollama for a general natural-language answer
    (used by the /stream endpoint to simulate a typing effect).
    Yields text chunks as they arrive from Ollama.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": (
            "You are a helpful data analytics assistant. "
            f"Answer the following question clearly and concisely:\n\n{question}"
        ),
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("response", "")
                    if token:
                        yield token

                    if chunk.get("done"):
                        break
    except httpx.RequestError as exc:
        logger.error("Ollama streaming connection failed: %s", exc)
        yield f"\n[ERROR] Could not reach Ollama at {OLLAMA_URL}: {exc}\n"
    except httpx.HTTPStatusError as exc:
        logger.error("Ollama returned an error status: %s", exc)
        yield f"\n[ERROR] Ollama returned status {exc.response.status_code}\n"


async def check_ollama_health() -> bool:
    """Quick check to see if the Ollama server is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            return resp.status_code == 200
    except httpx.RequestError:
        return False
