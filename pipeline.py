"""
pipeline.py
-----------
Core batch prediction pipeline.

Responsibilities
~~~~~~~~~~~~~~~~
1. Connect to the SQLite database.
2. Read rows from input_data that have NOT yet been predicted.
3. Load the trained model from disk.
4. Generate predictions.
5. Write results (id, prediction, timestamp) to the predictions table.

Designed to be called directly or imported by the scheduler:
    python pipeline.py
"""

import logging
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np

from config import DATABASE_PATH, MODEL_PATH, FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row-dict access enabled."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_unpredicted_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """
    Return all rows from input_data that don't yet have a matching
    entry in the predictions table.
    """
    query = """
        SELECT i.id, {features}
        FROM   input_data i
        LEFT JOIN predictions p ON i.id = p.id
        WHERE  p.id IS NULL
    """.format(features=", ".join(f"i.{col}" for col in FEATURE_COLUMNS))

    rows = conn.execute(query).fetchall()
    logger.info("Fetched %d unpredicted row(s) from input_data.", len(rows))
    return rows


def write_predictions(conn: sqlite3.Connection, results: list[tuple]) -> None:
    """
    Insert prediction rows into the predictions table.

    Each tuple in `results` must be (id, prediction, prediction_timestamp).
    """
    if not results:
        logger.info("No predictions to write.")
        return

    conn.executemany(
        "INSERT INTO predictions (id, prediction, prediction_timestamp) VALUES (?, ?, ?)",
        results,
    )
    conn.commit()
    logger.info("Wrote %d prediction(s) to the database.", len(results))


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(path: str):
    """Load a joblib-serialised model from disk."""
    model = joblib.load(path)
    logger.info("Model loaded from '%s'.", path)
    return model


def generate_predictions(model, rows: list[sqlite3.Row]) -> list[tuple]:
    """
    Run the model on the feature columns and return a list of
    (id, prediction, prediction_timestamp) tuples.
    """
    ids = [row["id"] for row in rows]
    X = np.array([[row[col] for col in FEATURE_COLUMNS] for row in rows])

    preds = model.predict(X)

    # Single UTC timestamp for the whole batch — keeps it consistent
    batch_time = datetime.now(timezone.utc).isoformat()

    return [(id_, int(pred), batch_time) for id_, pred in zip(ids, preds)]


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """
    Execute one full batch prediction cycle.
    Logs a warning and exits cleanly if there is nothing to predict.
    """
    logger.info("--- Batch prediction run started ---")

    # 1. Load model first — fail fast if it's missing
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        logger.error(
            "Model file '%s' not found. Run 'python train_model.py' first.",
            MODEL_PATH,
        )
        return

    # 2. Connect and fetch unpredicted rows
    with get_connection() as conn:
        rows = fetch_unpredicted_rows(conn)

        if not rows:
            logger.info("All rows are already predicted. Nothing to do.")
            logger.info("--- Batch prediction run finished ---")
            return

        # 3. Generate predictions
        results = generate_predictions(model, rows)

        # 4. Persist results
        write_predictions(conn, results)

    logger.info("--- Batch prediction run finished ---")


if __name__ == "__main__":
    run_pipeline()
