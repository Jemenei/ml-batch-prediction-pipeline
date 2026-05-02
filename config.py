"""
config.py
---------
Central place for all configuration constants.
Override any value with the matching environment variable.
"""

import os

# --- Database ---
# Path to the SQLite database file
DATABASE_PATH = os.getenv("DATABASE_PATH", "pipeline.db")

# --- Model ---
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

# --- Features ---
# Must match the columns in input_data AND the order the model was trained on.
FEATURE_COLUMNS = ["feature_1", "feature_2", "feature_3", "feature_4"]

# --- Scheduler ---
# How often (in minutes) the batch pipeline should run.
SCHEDULE_INTERVAL_MINUTES = int(os.getenv("SCHEDULE_INTERVAL_MINUTES", 5))
