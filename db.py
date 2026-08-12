"""
db.py
-----
Database adapters and schema introspection for SQLite, MySQL, and CSV.

The application keeps a single question -> SQL -> validation -> execution
flow, but the adapter underneath can now switch between data sources.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from file_utils import delete_path, normalize_identifier

try:
    import mysql.connector as mysql_connector
except ImportError:  # pragma: no cover - handled in runtime validation
    mysql_connector = None

logger = logging.getLogger("ai_analytics.db")

# Path to the legacy SQLite database file (created by load_data.py).
DB_PATH = Path(__file__).parent / "analytics.db"
UPLOAD_ROOT = Path(__file__).parent / "temporary" / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# Keywords that are never allowed in a generated query. This is a defense
# in depth measure on top of the "must start with SELECT" check.
FORBIDDEN_KEYWORDS = [
    "delete",
    "drop",
    "update",
    "insert",
    "alter",
    "truncate",
    "create",
    "replace",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "--",
    ";--",
    "grant",
    "revoke",
]


class UnsafeQueryError(Exception):
    """Raised when a generated SQL query fails safety validation."""


class QueryExecutionError(Exception):
    """Raised when a syntactically-valid but failing SQL query is run."""


@dataclass
class ColumnSchema:
    """Column metadata used for schema display and prompt construction."""

    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "primary_key": self.primary_key,
            "foreign_key": self.foreign_key,
        }


@dataclass
class TableSchema:
    """Table metadata used for schema display and prompt construction."""

    name: str
    columns: List[ColumnSchema] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": [column.to_dict() for column in self.columns],
        }


@dataclass
class DatabaseSchema:
    """Structured schema description for the current data source."""

    source_type: str
    sql_dialect: str
    tables: List[TableSchema] = field(default_factory=list)
    source_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "sql_dialect": self.sql_dialect,
            "source_name": self.source_name,
            "tables": [table.to_dict() for table in self.tables],
        }

    def to_prompt_text(self) -> str:
        """Render the schema in a compact, LLM-friendly format."""
        lines = [f"Data Source Type: {self.source_type}", f"SQL Dialect: {self.sql_dialect}"]
        if self.source_name:
            lines.append(f"Source Name: {self.source_name}")

        if not self.tables:
            lines.append("No tables were found in the selected data source.")
            return "\n".join(lines)

        for table in self.tables:
            lines.append(f"\nTable: {table.name}")
            lines.append("Columns:")
            for column in table.columns:
                flags: List[str] = []
                if column.primary_key:
                    flags.append("PRIMARY KEY")
                if column.foreign_key:
                    flags.append(f"FOREIGN KEY -> {column.foreign_key}")
                suffix = f" [{', '.join(flags)}]" if flags else ""
                lines.append(f"- {column.name}: {column.data_type}{suffix}")

        return "\n".join(lines)


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _quote_mysql_identifier(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


def _format_foreign_key_reference(table: str, column: str) -> str:
    return f"{table}.{column}"


def _unique_column_names(column_names: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    result: List[str] = []
    for original_name in column_names:
        normalized = normalize_identifier(str(original_name), default="column")
        count = seen.get(normalized, 0)
        if count:
            normalized = f"{normalized}_{count + 1}"
        seen[normalize_identifier(str(original_name), default="column")] = count + 1
        result.append(normalized)
    return result


def _infer_sqlite_type_from_series(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "INTEGER"
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TEXT"
    return "TEXT"


def _normalize_dataframe_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe.columns = _unique_column_names(list(dataframe.columns))
    return dataframe


def validate_sql(sql: str, database_type: str | None = None) -> str:
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
    cleaned = re.sub(r"^sql\s*", "", cleaned, flags=re.IGNORECASE).strip()

    # Disallow multiple statements (only one trailing semicolon allowed)
    no_trailing = cleaned.rstrip(";").strip()
    if ";" in no_trailing:
        raise UnsafeQueryError("Multiple SQL statements are not allowed.")

    if not re.match(r"(?is)^\s*select\b", no_trailing):
        dialect = f" for {database_type}" if database_type else ""
        raise UnsafeQueryError(f"Only SELECT queries are permitted{dialect}.")

    lowered = no_trailing.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        pattern = r"\b" + re.escape(keyword) + r"\b" if keyword.isalpha() else re.escape(keyword)
        if re.search(pattern, lowered):
            raise UnsafeQueryError(f"Query contains forbidden keyword: '{keyword}'")

    return no_trailing


class DatabaseAdapter(ABC):
    """Common interface for the application's supported data sources."""

    source_type: str
    sql_dialect: str

    @abstractmethod
    def get_schema(self) -> DatabaseSchema:
        raise NotImplementedError

    @abstractmethod
    def execute_query(self, sql: str) -> Tuple[List[Dict[str, Any]], int]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class SQLiteAdapter(DatabaseAdapter):
    """Adapter for SQLite database files."""

    def __init__(self, db_path: Path, *, managed: bool = False, source_name: Optional[str] = None):
        self.db_path = Path(db_path)
        self.managed = managed
        self.source_type = "sqlite"
        self.sql_dialect = "sqlite"
        self.source_name = source_name or self.db_path.name

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_schema(self) -> DatabaseSchema:
        try:
            with self._connection() as conn:
                table_rows = conn.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                ).fetchall()

                tables: List[TableSchema] = []
                for table_row in table_rows:
                    table_name = table_row[0]
                    column_rows = conn.execute(f"PRAGMA table_info({_quote_sqlite_identifier(table_name)})").fetchall()
                    foreign_key_rows = conn.execute(
                        f"PRAGMA foreign_key_list({_quote_sqlite_identifier(table_name)})"
                    ).fetchall()

                    foreign_key_map = {
                        row[3]: _format_foreign_key_reference(row[2], row[4])
                        for row in foreign_key_rows
                    }

                    columns = [
                        ColumnSchema(
                            name=row[1],
                            data_type=row[2] or "TEXT",
                            nullable=not bool(row[3]),
                            primary_key=bool(row[5]),
                            foreign_key=foreign_key_map.get(row[1]),
                        )
                        for row in column_rows
                    ]
                    tables.append(TableSchema(name=table_name, columns=columns))

                return DatabaseSchema(
                    source_type=self.source_type,
                    sql_dialect=self.sql_dialect,
                    tables=tables,
                    source_name=self.source_name,
                )
        except sqlite3.Error as exc:
            logger.error("SQLite schema introspection failed for %s: %s", self.db_path, exc)
            raise QueryExecutionError(f"Could not inspect SQLite database: {exc}") from exc

    def execute_query(self, sql: str) -> Tuple[List[Dict[str, Any]], int]:
        validated_sql = validate_sql(sql, self.sql_dialect)
        logger.info("Executing validated SQLite SQL against %s", self.db_path)

        try:
            with self._connection() as conn:
                cursor = conn.execute(validated_sql)
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]
                return result, len(result)
        except sqlite3.Error as exc:
            logger.error("SQLite execution error: %s", exc)
            raise QueryExecutionError(str(exc)) from exc

    def close(self) -> None:
        if self.managed and self.db_path.exists() and self.db_path != DB_PATH:
            delete_path(self.db_path)


class MySQLAdapter(DatabaseAdapter):
    """Adapter for MySQL databases using mysql-connector-python."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        *,
        source_name: Optional[str] = None,
        connect_timeout: int = 10,
    ):
        if mysql_connector is None:
            raise QueryExecutionError("MySQL support requires the mysql-connector-python package.")

        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.connect_timeout = connect_timeout
        self.source_type = "mysql"
        self.sql_dialect = "mysql"
        self.source_name = source_name or database

    def _connect(self):
        return mysql_connector.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.username,
            password=self.password,
            connection_timeout=self.connect_timeout,
        )

    def get_schema(self) -> DatabaseSchema:
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """,
                (self.database,),
            )
            table_rows = cursor.fetchall()

            cursor.execute(
                """
                SELECT
                    kcu.table_name,
                    kcu.column_name,
                    kcu.referenced_table_name,
                    kcu.referenced_column_name
                FROM information_schema.key_column_usage kcu
                JOIN information_schema.table_constraints tc
                    ON tc.constraint_name = kcu.constraint_name
                   AND tc.table_schema = kcu.table_schema
                   AND tc.table_name = kcu.table_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND kcu.table_schema = %s
                """,
                (self.database,),
            )
            foreign_key_rows = cursor.fetchall()

            foreign_key_map: Dict[Tuple[str, str], str] = {}
            for row in foreign_key_rows:
                referenced = _format_foreign_key_reference(
                    row["referenced_table_name"], row["referenced_column_name"]
                )
                foreign_key_map[(row["table_name"], row["column_name"])] = referenced

            tables: List[TableSchema] = []
            for row in table_rows:
                table_name = row["table_name"]
                cursor.execute(
                    """
                    SELECT column_name, data_type, is_nullable, column_key, column_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (self.database, table_name),
                )
                column_rows = cursor.fetchall()
                columns = [
                    ColumnSchema(
                        name=column_row["column_name"],
                        data_type=column_row["column_type"] or column_row["data_type"] or "TEXT",
                        nullable=(column_row["is_nullable"].upper() == "YES"),
                        primary_key=(column_row["column_key"] == "PRI"),
                        foreign_key=foreign_key_map.get((table_name, column_row["column_name"])),
                    )
                    for column_row in column_rows
                ]
                tables.append(TableSchema(name=table_name, columns=columns))

            return DatabaseSchema(
                source_type=self.source_type,
                sql_dialect=self.sql_dialect,
                tables=tables,
                source_name=self.source_name,
            )
        except Exception as exc:
            logger.error("MySQL schema introspection failed for %s.%s: %s", self.host, self.database, exc)
            raise QueryExecutionError(str(exc)) from exc
        finally:
            if conn is not None:
                conn.close()

    def execute_query(self, sql: str) -> Tuple[List[Dict[str, Any]], int]:
        validated_sql = validate_sql(sql, self.sql_dialect)
        logger.info("Executing validated MySQL SQL against %s.%s", self.host, self.database)

        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(validated_sql)
            rows = cursor.fetchall()
            return [dict(row) for row in rows], len(rows)
        except Exception as exc:
            logger.error("MySQL execution error: %s", exc)
            raise QueryExecutionError(str(exc)) from exc
        finally:
            if conn is not None:
                conn.close()

    def close(self) -> None:
        return None


class CSVAdapter(SQLiteAdapter):
    """Adapter for CSV files loaded into a temporary SQLite database."""

    def __init__(self, csv_path: Path, *, source_name: Optional[str] = None):
        self.csv_path = Path(csv_path)
        self.csv_source_name = source_name or self.csv_path.name
        self.table_name = normalize_identifier(self.csv_path.stem, default="csv_data")
        self._temp_db_path = UPLOAD_ROOT / f"{self.table_name}_{self.csv_path.stem}_{Path(self.csv_path).stat().st_mtime_ns}.sqlite3"
        self._schema = self._load_csv_into_sqlite()
        super().__init__(self._temp_db_path, managed=True, source_name=self.csv_source_name)
        self.source_type = "csv"
        self.sql_dialect = "sqlite"

    def _load_csv_into_sqlite(self) -> DatabaseSchema:
        try:
            dataframe = pd.read_csv(self.csv_path)
        except pd.errors.EmptyDataError as exc:
            raise QueryExecutionError("The uploaded CSV file is empty.") from exc
        except pd.errors.ParserError as exc:
            raise QueryExecutionError(f"The uploaded CSV file is malformed: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise QueryExecutionError("The uploaded CSV file could not be decoded as text.") from exc

        if dataframe.columns.empty:
            raise QueryExecutionError("The uploaded CSV file does not contain any columns.")

        dataframe = _normalize_dataframe_columns(dataframe)
        unique_columns = list(dataframe.columns)

        try:
            with sqlite3.connect(self._temp_db_path) as conn:
                dataframe.to_sql(self.table_name, conn, if_exists="replace", index=False)
        except sqlite3.Error as exc:
            delete_path(self._temp_db_path)
            raise QueryExecutionError(f"Could not create a temporary database from the CSV file: {exc}") from exc

        columns = [
            ColumnSchema(
                name=column_name,
                data_type=_infer_sqlite_type_from_series(dataframe[column_name]),
                nullable=bool(dataframe[column_name].isna().any()) or dataframe.empty,
            )
            for column_name in unique_columns
        ]

        return DatabaseSchema(
            source_type="csv",
            sql_dialect="sqlite",
            tables=[TableSchema(name=self.table_name, columns=columns)],
            source_name=self.csv_source_name,
        )

    def get_schema(self) -> DatabaseSchema:
        return self._schema

    def close(self) -> None:
        delete_path(self._temp_db_path)
        delete_path(self.csv_path)


def build_legacy_schema() -> str:
    """Return the legacy analytics.db schema as a prompt string."""
    try:
        return SQLiteAdapter(DB_PATH).get_schema().to_prompt_text()
    except Exception:
        return ""


DB_SCHEMA = build_legacy_schema()


def get_database_adapter(source_type: str, source_config: Dict[str, Any]) -> DatabaseAdapter:
    """Create the correct adapter for the selected source type."""
    normalized = (source_type or "sqlite").strip().lower()
    if normalized == "sqlite":
        return SQLiteAdapter(Path(source_config["file_path"]), managed=bool(source_config.get("managed", False)))
    if normalized == "mysql":
        return MySQLAdapter(
            host=source_config["host"],
            port=int(source_config.get("port", 3306)),
            database=source_config["database"],
            username=source_config["username"],
            password=source_config["password"],
        )
    if normalized == "csv":
        return CSVAdapter(Path(source_config["file_path"]))
    raise ValueError(f"Unsupported data source type: {source_type}")


def run_query(sql: str) -> Tuple[List[Dict[str, Any]], int]:
    """Backwards-compatible helper that still runs against analytics.db."""
    adapter = SQLiteAdapter(DB_PATH)
    return adapter.execute_query(sql)
