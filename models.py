"""
models.py
----------
Pydantic models used for request validation and response serialization
across the AI Analytics Query Assistant API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""
    question: str = Field(
        ...,
        min_length=3,
        description="Natural language question about the sales data.",
        examples=["What is the total revenue by product?"],
    )


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""
    question: str
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
