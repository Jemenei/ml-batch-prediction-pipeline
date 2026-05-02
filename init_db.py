"""
init_db.py
----------
Sets up the SQLite database with the required tables and seeds
the input_data table with synthetic rows for testing.

Run this once before starting the pipeline:
    python init_db.py
"""

import logging
import random
import sqlite3

from config import DATABASE_PATH, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with foreign-key support enabled."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """Create input_data and predictions tables if they don't already exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS input_data (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_1 REAL NOT NULL,
            feature_2 REAL NOT NULL,
            feature_3 REAL NOT NULL,
            feature_4 REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                   INTEGER PRIMARY KEY,
            prediction           INTEGER NOT NULL,
            prediction_timestamp TEXT    NOT NULL
        );
    """)
    conn.commit()
    logger.info("Tables created (or already exist).")


def seed_input_data(conn: sqlite3.Connection, n_rows: int = 50) -> None:
    """
    Populate input_data with random dummy rows.
    Skips seeding if the table already has data.
    """
    existing = conn.execute("SELECT COUNT(*) FROM input_data").fetchone()[0]
    if existing > 0:
        logger.info("input_data already has %d row(s) — skipping seed.", existing)
        return

    random.seed(42)
    rows = [
        (
            round(random.uniform(0.0, 10.0), 4),   # feature_1
            round(random.uniform(-5.0, 5.0), 4),   # feature_2
            round(random.uniform(0.0, 1.0), 4),    # feature_3
            round(random.uniform(100.0, 200.0), 4) # feature_4
        )
        for _ in range(n_rows)
    ]

    conn.executemany(
        "INSERT INTO input_data (feature_1, feature_2, feature_3, feature_4) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info("Seeded input_data with %d rows.", n_rows)


def main() -> None:
    with get_connection() as conn:
        create_tables(conn)
        seed_input_data(conn)
    logger.info("Database initialisation complete: '%s'.", DATABASE_PATH)


if __name__ == "__main__":
    main()
