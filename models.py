"""
models.py
----------
Pydantic models used for request validation and response serialization
across the AI Analytics Query Assistant API.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

DataSourceType = Literal["sqlite", "mysql", "csv"]


class ColumnSchema(BaseModel):
    """A single column discovered from a database schema."""

    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: Optional[str] = None


class TableSchema(BaseModel):
    """A table discovered from a database schema."""

    name: str
    columns: List[ColumnSchema]


class DatabaseSchemaResponse(BaseModel):
    """Schema payload returned to the frontend and used for prompting."""

    source_type: DataSourceType
    sql_dialect: Literal["sqlite", "mysql"]
    tables: List[TableSchema]


class DataSourceResponse(BaseModel):
    """Response returned after a successful SQLite/CSV upload or MySQL connect."""

    source_id: str
    source_type: DataSourceType
    message: str
    database_schema: DatabaseSchemaResponse


class MySQLConnectRequest(BaseModel):
    """Connection details for a MySQL database."""

    host: str = Field(..., min_length=1)
    port: int = Field(3306, ge=1, le=65535)
    database: str = Field(..., min_length=1)
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        description="Natural language question about the selected data source.",
        examples=["What is the total revenue by product?"],
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Identifier for the selected data source. If omitted, the legacy analytics.db is used.",
    )


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    question: str
    source_id: Optional[str] = None
    source_type: Optional[DataSourceType] = None
    sql: Optional[str] = None
    result: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = None
    error: Optional[str] = None


class StreamRequest(BaseModel):
    """Request body for the /stream endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        description="Natural language question to stream an LLM answer for.",
    )


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    ollama_reachable: bool
    model: str
