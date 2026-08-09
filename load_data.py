"""
load_data.py
------------
Creates the SQLite database (analytics.db) and populates the `sales`
table with sample data. Run this once before starting the API server:

    python load_data.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "analytics.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sales (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product     TEXT    NOT NULL,
    revenue     REAL    NOT NULL,
    clicks      INTEGER NOT NULL,
    impressions INTEGER NOT NULL,
    date        TEXT    NOT NULL
);
"""

SAMPLE_ROWS = [
    ("Wireless Mouse",      1250.50, 320, 8400, "2024-01-05"),
    ("Mechanical Keyboard", 3420.00, 540, 12100, "2024-01-06"),
    ("USB-C Hub",            980.75, 210, 5600, "2024-01-07"),
    ("Laptop Stand",        1675.20, 275, 7300, "2024-01-08"),
    ("Webcam 1080p",        2210.00, 410, 9800, "2024-01-09"),
    ("Noise Cancelling Headphones", 5430.90, 610, 15400, "2024-01-10"),
    ("Bluetooth Speaker",   1890.40, 350, 8900, "2024-01-11"),
    ("Monitor 27in",        6720.00, 480, 11200, "2024-01-12"),
    ("Ergonomic Chair",     8950.00, 190, 4200, "2024-01-13"),
    ("Desk Lamp",            540.30, 150, 3100, "2024-01-14"),
]


def load_sample_data() -> None:
    """Creates the sales table (if needed) and inserts sample rows."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_SQL)

        # Avoid duplicate inserts if the script is run multiple times
        cursor.execute("SELECT COUNT(*) FROM sales")
        existing_count = cursor.fetchone()[0]

        if existing_count == 0:
            cursor.executemany(
                """
                INSERT INTO sales (product, revenue, clicks, impressions, date)
                VALUES (?, ?, ?, ?, ?)
                """,
                SAMPLE_ROWS,
            )
            conn.commit()
            print(f"Inserted {len(SAMPLE_ROWS)} sample rows into 'sales'.")
        else:
            print(f"'sales' table already has {existing_count} rows. Skipping insert.")

    finally:
        conn.close()


if __name__ == "__main__":
    load_sample_data()
    print(f"Database ready at: {DB_PATH}")
