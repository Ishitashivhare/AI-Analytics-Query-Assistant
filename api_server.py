"""
api_server.py
--------------
FastAPI application exposing the AI Analytics Query Assistant.

Endpoints:
  GET  /health  -> service + Ollama connectivity check
  POST /ask     -> natural language question -> SQL -> execute -> results
  POST /stream  -> streams a natural-language LLM answer token by token

Run with:
    uvicorn api_server:app --reload --port 8000
"""

import logging

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from db import QueryExecutionError, UnsafeQueryError, run_query
from llama_sql_generator import (
    OLLAMA_MODEL,
    check_ollama_health,
    generate_sql,
    stream_llm_response,
)
from models import AskRequest, AskResponse, HealthResponse, StreamRequest

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_analytics.api")

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Analytics Query Assistant",
    description="Converts natural language questions into SQL, executes them "
                 "against a SQLite sales database, and returns structured results.",
    version="1.0.0",
)

# Allow local frontends / tools to call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Reports API status and whether the Ollama server is reachable."""
    ollama_ok = await check_ollama_health()
    return HealthResponse(
        status="ok",
        ollama_reachable=ollama_ok,
        model=OLLAMA_MODEL,
    )


@app.post("/ask", response_model=AskResponse, tags=["Query"])
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Full pipeline:
      1. Send the question + schema to the LLM to generate SQL.
      2. Validate the SQL is a safe, read-only SELECT statement.
      3. Execute it against the SQLite database.
      4. Return the question, generated SQL, and results as JSON.
    """
    question = request.question.strip()
    logger.info("Received /ask request: %s", question)

    # Step 1: Generate SQL via the LLM
    try:
        sql = await generate_sql(question)
    except httpx.RequestError as exc:
        logger.error("LLM request failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach the LLM service (Ollama). Is it running? Details: {exc}",
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.error("LLM returned error status: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"LLM service returned an error: {exc.response.status_code}",
        ) from exc

    if not sql:
        return AskResponse(
            question=question,
            sql=None,
            result=None,
            error="The LLM did not return any SQL for this question.",
        )

    # Step 2 + 3: Validate and execute the SQL safely
    try:
        result, row_count = run_query(sql)
    except UnsafeQueryError as exc:
        logger.warning("Rejected unsafe SQL: %s | reason: %s", sql, exc)
        return AskResponse(
            question=question,
            sql=sql,
            result=None,
            error=f"Query rejected for safety reasons: {exc}",
        )
    except QueryExecutionError as exc:
        logger.error("SQL execution failed: %s | reason: %s", sql, exc)
        return AskResponse(
            question=question,
            sql=sql,
            result=None,
            error=f"Query execution failed: {exc}",
        )

    if row_count == 0:
        logger.info("Query executed successfully but returned no rows.")

    # Step 4: Return structured response
    return AskResponse(
        question=question,
        sql=sql,
        result=result,
        row_count=row_count,
        error=None,
    )


@app.post("/stream", tags=["Query"])
async def stream_answer(request: StreamRequest) -> StreamingResponse:
    """
    Streams the LLM's natural-language answer token by token, simulating
    a typing effect on the client side. Uses FastAPI's StreamingResponse
    with an async generator fed by the Ollama streaming API.
    """
    question = request.question.strip()
    logger.info("Received /stream request: %s", question)

    async def token_generator():
        async for token in stream_llm_response(question):
            # Each chunk is sent as plain text; the client appends it
            # to the UI as it arrives, simulating typing.
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.get("/", tags=["System"])
async def root():
    """Simple root endpoint with basic usage info."""
    return {
        "service": "AI Analytics Query Assistant",
        "endpoints": {
            "GET /health": "Check API and Ollama status",
            "POST /ask": "Ask a natural language question, get SQL + results",
            "POST /stream": "Stream a natural language LLM answer",
        },
    }
