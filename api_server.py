"""
api_server.py
--------------
FastAPI application exposing the AI Analytics Query Assistant.

Endpoints:
  GET  /health                         -> service + Ollama connectivity check
  POST /data-sources/sqlite/upload     -> upload a SQLite database file
  POST /data-sources/csv/upload        -> upload a CSV and load it into temp SQLite
  POST /data-sources/mysql/connect     -> connect to MySQL and inspect schema
  POST /ask                            -> natural language question -> SQL -> execute -> results
  POST /stream                         -> streams a natural-language LLM answer token by token

Run with:
    uvicorn api_server:app --reload --port 8000
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from db import (
    DB_PATH,
    CSVAdapter,
    DatabaseAdapter,
    DatabaseSchema,
    MySQLAdapter,
    QueryExecutionError,
    SQLiteAdapter,
    UnsafeQueryError,
    get_database_adapter,
)
from file_utils import ALLOWED_CSV_EXTENSIONS, ALLOWED_SQLITE_EXTENSIONS, delete_path, store_upload_bytes
from llama_sql_generator import OLLAMA_MODEL, check_ollama_health, generate_sql, stream_llm_response
from models import (
    AskRequest,
    AskResponse,
    DataSourceResponse,
    DatabaseSchemaResponse,
    HealthResponse,
    MySQLConnectRequest,
    StreamRequest,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_analytics.api")


@dataclass
class RegisteredSource: #represents one connected data source in your application such as sql or csv.
    source_id: str
    source_type: str
    adapter: DatabaseAdapter
    schema: DatabaseSchema
    display_name: str


REGISTERED_SOURCES: dict[str, RegisteredSource] = {}


def _schema_to_response(schema: DatabaseSchema) -> DatabaseSchemaResponse:
    return DatabaseSchemaResponse.model_validate(schema.to_dict())

#Take a newly connected data source, give it a unique ID, create a RegisteredSource object, store it in the registry, and return it.
def _register_source(source_type: str, adapter: DatabaseAdapter, schema: DatabaseSchema, display_name: str) -> RegisteredSource:
    source_id = uuid.uuid4().hex   #uuid.uuid4() generates a random UUID, and .hex converts it into a hexadecimal string.
    registered = RegisteredSource(   #creating object
        source_id=source_id,
        source_type=source_type,
        adapter=adapter,
        schema=schema,
        display_name=display_name,
    )
    REGISTERED_SOURCES[source_id] = registered
    return registered

#Given a source_id, find and return the corresponding RegisteredSource from the registry.
def _get_registered_source(source_id: str | None) -> RegisteredSource | None:
    if not source_id:
        return None
    return REGISTERED_SOURCES.get(source_id)

#Remove a data source from the registry and close its adapter/connection to release resources.
def _close_registered_source(source_id: str) -> None:
    registered = REGISTERED_SOURCES.pop(source_id, None)
    if not registered:
        return
    try:
        registered.adapter.close()
    except Exception:
        logger.warning("Failed to close data source %s", source_id, exc_info=True)

#Close and remove every currently registered data source.
def _close_all_registered_sources() -> None:
    for source_id in list(REGISTERED_SOURCES):
        _close_registered_source(source_id)

#To convert technical MySQL exceptions into meaningful messages that can be returned to the user.
def _mysql_error_message(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    errno = getattr(exc, "errno", None)

    if errno in {1045} or "access denied" in lowered or "using password" in lowered:
        return "MySQL connection failed: invalid username or password."
    if errno in {1049} or "unknown database" in lowered:
        return "MySQL connection failed: the requested database does not exist."
    if errno in {2003, 2005} or "can't connect to mysql server" in lowered or "host" in lowered and "unknown" in lowered:
        return "MySQL connection failed: could not reach the host or port."
    if errno in {2006, 2013} or "lost connection" in lowered:
        return "MySQL connection failed: the server closed the connection unexpectedly."
    return f"MySQL connection failed: {message}"

#This function creates the structured response that the API sends back after a data source is connected.
def _build_data_source_response(registered: RegisteredSource, message: str) -> DataSourceResponse:
    return DataSourceResponse(
        source_id=registered.source_id,
        source_type=registered.source_type,  # type: ignore[arg-type]
        message=message,
        database_schema=_schema_to_response(registered.schema),
    )

#This function creates the default SQLite data source for the application.
def _default_sqlite_source() -> RegisteredSource:
    adapter = SQLiteAdapter(DB_PATH, source_name=DB_PATH.name)
    schema = adapter.get_schema()
    return RegisteredSource(
        source_id="legacy-analytics-db",
        source_type="sqlite",
        adapter=adapter,
        schema=schema,
        display_name=DB_PATH.name,
    )

#Checks that the file has a filename.Reads the uploaded file.Passes the file data to store_upload_bytes() for validation/storage.Converts storage/validation errors into a FastAPI HTTPException.
async def _save_uploaded_file(upload_file: UploadFile, allowed_extensions: set[str], prefix: str) -> Path:
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    data = await upload_file.read()
    try:
        return store_upload_bytes(upload_file.filename, data, allowed_extensions, prefix=prefix)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------
#This creates the FastAPI application object.
app = FastAPI(
    title="AI Analytics Query Assistant",
    description="Converts natural language questions into SQL, executes them against the selected data source, and returns structured results.",
    version="1.0.0",
)

# Allow local frontends / tools to call the API during development
#CORS (Cross-Origin Resource Sharing) for your API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#When the application shuts down, close all registered data-source connections.
@app.on_event("shutdown")
def shutdown_cleanup() -> None:
    _close_all_registered_sources()


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Reports API status and whether the Ollama server is reachable."""
    ollama_ok = await check_ollama_health()
    return HealthResponse(
        status="ok",
        ollama_reachable=ollama_ok,
        model=OLLAMA_MODEL,
    )


@app.post("/data-sources/sqlite/upload", response_model=DataSourceResponse, tags=["Data Source"])
async def upload_sqlite_database(file: UploadFile = File(...)) -> DataSourceResponse:
    """Upload a SQLite database file and introspect its schema."""
    upload_path = await _save_uploaded_file(file, ALLOWED_SQLITE_EXTENSIONS, prefix="sqlite_upload")
    adapter: SQLiteAdapter | None = None

    try:
        adapter = SQLiteAdapter(upload_path, managed=True, source_name=file.filename)
        schema = adapter.get_schema()
        registered = _register_source("sqlite", adapter, schema, file.filename or upload_path.name)
        return _build_data_source_response(registered, "Connected successfully to the uploaded SQLite database.")
    except QueryExecutionError as exc:
        if adapter is not None:
            adapter.close()
        else:
            delete_path(upload_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        delete_path(upload_path)
        raise HTTPException(status_code=400, detail=f"Could not load the uploaded SQLite database: {exc}") from exc


@app.post("/data-sources/csv/upload", response_model=DataSourceResponse, tags=["Data Source"])
async def upload_csv_file(file: UploadFile = File(...)) -> DataSourceResponse:
    """Upload a CSV file and load it into a temporary SQLite database."""
    upload_path = await _save_uploaded_file(file, ALLOWED_CSV_EXTENSIONS, prefix="csv_upload")

    try:
        adapter = CSVAdapter(upload_path, source_name=file.filename)
        schema = adapter.get_schema()
        registered = _register_source("csv", adapter, schema, file.filename or upload_path.name)
        return _build_data_source_response(registered, "Connected successfully to the uploaded CSV file.")
    except QueryExecutionError as exc:
        delete_path(upload_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        delete_path(upload_path)
        raise HTTPException(status_code=400, detail=f"Could not load the uploaded CSV file: {exc}") from exc

#Unlike SQLite, there is no file upload here. The user provides MySQL connection details through MySQLConnectRequest
@app.post("/data-sources/mysql/connect", response_model=DataSourceResponse, tags=["Data Source"])
async def connect_mysql_database(request: MySQLConnectRequest) -> DataSourceResponse:
    """Connect to a MySQL database and introspect its schema."""
    try:
        adapter = MySQLAdapter(
            host=request.host,
            port=request.port,
            database=request.database,
            username=request.username,
            password=request.password,
            source_name=request.database,
        )
        schema = adapter.get_schema()
        registered = _register_source("mysql", adapter, schema, request.database)
        return _build_data_source_response(registered, "Connected successfully to MySQL.")
    except QueryExecutionError as exc:
        raise HTTPException(status_code=400, detail=_mysql_error_message(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_mysql_error_message(exc)) from exc

#The /ask endpoint takes a natural-language question, finds the selected data source, sends the question + schema to Llama through Ollama to generate SQL, validates and executes that SQL, and returns the results.
@app.post("/ask", response_model=AskResponse, tags=["Query"])
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Full pipeline:
      1. Select the correct source and get its dynamic schema.
      2. Send the question + schema + dialect to the LLM to generate SQL.
      2. Validate the SQL is a safe, read-only SELECT statement.
      3. Execute it against the selected database.
      4. Return the question, generated SQL, and results as JSON.
    """
    question = request.question.strip()
    logger.info("Received /ask request: %s | source_id=%s", question, request.source_id or "legacy")

    registered = _get_registered_source(request.source_id)
    if request.source_id and not registered:
        raise HTTPException(status_code=404, detail="The selected data source is no longer available. Please reconnect.")

    if registered is None:
        registered = _default_sqlite_source()

    # Step 1: Generate SQL via the LLM
    try:
        sql = await generate_sql(question, registered.schema, registered.source_type)
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
    #The LLM did not return any SQL for this question.
    if not sql:
        return AskResponse(
            question=question,
            source_id=registered.source_id if registered.source_id != "legacy-analytics-db" else None,
            source_type=registered.source_type,  # type: ignore[arg-type]
            sql=None,
            result=None,
            error="The LLM did not return any SQL for this question.",
        )
#if the question cannot be answered using the selected database.
    if "UNSUPPORTED_QUESTION" in sql.upper():
        return AskResponse(
            question=question,
            source_id=registered.source_id if registered.source_id != "legacy-analytics-db" else None,
            source_type=registered.source_type,  # type: ignore[arg-type]
            sql=sql,
            result=None,
            error="This question cannot be answered from the selected data source.",
        )

    # Step 2 + 3: Validate and execute the SQL safely
    try:
        result, row_count = registered.adapter.execute_query(sql)
    except UnsafeQueryError as exc:
        logger.warning("Rejected unsafe SQL: %s | reason: %s", sql, exc)
        return AskResponse(
            question=question,
            source_id=registered.source_id if registered.source_id != "legacy-analytics-db" else None,
            source_type=registered.source_type,  # type: ignore[arg-type]
            sql=sql,
            result=None,
            error=f"Query rejected for safety reasons: {exc}",
        )
    except QueryExecutionError as exc:
        logger.error("SQL execution failed: %s | reason: %s", sql, exc)
        return AskResponse(
            question=question,
            source_id=registered.source_id if registered.source_id != "legacy-analytics-db" else None,
            source_type=registered.source_type,  # type: ignore[arg-type]
            sql=sql,
            result=None,
            error=f"Query execution failed: {exc}",
        )

    if row_count == 0:
        logger.info("Query executed successfully but returned no rows.")

    # Step 4: Return structured response
    return AskResponse(
        question=question,
        source_id=registered.source_id if registered.source_id != "legacy-analytics-db" else None,
        source_type=registered.source_type,  # type: ignore[arg-type]
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
