"""
file_utils.py
-------------
Small helpers for upload validation, temporary file handling, and safe
SQL identifier normalization.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Iterable

UPLOAD_ROOT = Path(__file__).parent / "temporary" / "uploads"
ALLOWED_SQLITE_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}
ALLOWED_CSV_EXTENSIONS = {".csv"}


def ensure_upload_root() -> Path:
    """Create the temporary upload directory if it does not already exist."""
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    return UPLOAD_ROOT


def normalize_identifier(value: str, default: str = "data", max_length: int = 63) -> str:
    """Normalize arbitrary text into a safe SQL identifier."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = default
    if cleaned[0].isdigit():
        cleaned = f"{default}_{cleaned}"
    return cleaned[:max_length]


def validate_extension(filename: str, allowed_extensions: Iterable[str], label: str) -> None:
    """Raise ValueError if the file extension is not allowed."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {ext.lower() for ext in allowed_extensions}:
        allowed = ", ".join(sorted({ext.lower() for ext in allowed_extensions}))
        raise ValueError(f"Unsupported {label} file type '{suffix or '[none]'}'. Allowed extensions: {allowed}")


def unique_upload_path(original_filename: str, prefix: str | None = None) -> Path:
    """Generate a unique file path inside the temporary upload directory."""
    ensure_upload_root()
    stem = normalize_identifier(Path(original_filename).stem, default=prefix or "upload")
    suffix = Path(original_filename).suffix.lower()
    return UPLOAD_ROOT / f"{stem}_{uuid.uuid4().hex}{suffix}"


def store_upload_bytes(original_filename: str, data: bytes, allowed_extensions: Iterable[str], prefix: str | None = None) -> Path:
    """Validate and store uploaded bytes inside the temporary upload directory."""
    validate_extension(original_filename, allowed_extensions, "uploaded")
    destination = unique_upload_path(original_filename, prefix=prefix)
    destination.write_bytes(data)
    return destination


def delete_path(path: Path | None) -> None:
    """Best-effort removal of a file or directory."""
    if path is None:
        return
    try:
        if path.is_dir():
            for child in path.iterdir():
                delete_path(child)
            path.rmdir()
        elif path.exists():
            path.unlink()
    except OSError:
        pass