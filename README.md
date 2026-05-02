# Batch Prediction Pipeline

A lightweight, production-style ML batch-prediction system built with Python, SQLite, scikit-learn, and the `schedule` library.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        scheduler.py                     │
│          Runs pipeline every N minutes (default: 5)     │
└───────────────────────────┬─────────────────────────────┘
                            │ calls
                            ▼
┌─────────────────────────────────────────────────────────┐
│                        pipeline.py                      │
│  1. Fetch unpredicted rows from  input_data             │
│  2. Load model.joblib                                   │
│  3. Generate predictions                                │
│  4. Write results → predictions table                   │
└───────────┬──────────────────────────────┬──────────────┘
            │ reads                        │ writes
            ▼                              ▼
┌───────────────────────┐      ┌───────────────────────────┐
│  SQLite: input_data   │      │  SQLite: predictions      │
│  id, feature_1 … _4  │      │  id, prediction, timestamp│
└───────────────────────┘      └───────────────────────────┘
```

## Project Structure

```
batch_prediction_pipeline/
├── config.py          # All constants and env-var overrides
├── init_db.py         # Create tables + seed dummy data
├── train_model.py     # Train & save RandomForestClassifier
├── pipeline.py        # Core batch prediction logic
├── scheduler.py       # Periodic runner
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### 1. Clone & set up a virtual environment

```bash
git clone <your-repo-url>
cd batch_prediction_pipeline

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialise the database

Creates `pipeline.db` with `input_data` and `predictions` tables and seeds 50 dummy rows.

```bash
python init_db.py
```

### 4. Train the model

Trains a `RandomForestClassifier` on synthetic data and saves it as `model.joblib`.

```bash
python train_model.py
```

### 5. Run the scheduler

Executes the pipeline immediately, then every 5 minutes.

```bash
python scheduler.py
```

**Background (Unix / macOS):**
```bash
mkdir -p logs
nohup python scheduler.py > logs/scheduler.log 2>&1 &
```

**Background (Windows PowerShell):**
```powershell
Start-Process python -ArgumentList "scheduler.py" -WindowStyle Hidden
```

### 6. Run the pipeline manually (one-shot)

```bash
python pipeline.py
```

## Configuration

All settings live in `config.py` and can be overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///pipeline.db` | SQLAlchemy database URL |
| `MODEL_PATH` | `model.joblib` | Path to the saved model |
| `SCHEDULE_INTERVAL_MINUTES` | `5` | How often the pipeline runs |

Example — run every 2 minutes against a custom DB:

```bash
DATABASE_URL=sqlite:///test.db SCHEDULE_INTERVAL_MINUTES=2 python scheduler.py
```

## Verifying Results

Open the database with the SQLite CLI:

```bash
sqlite3 pipeline.db "SELECT * FROM predictions LIMIT 10;"
```

Or with Python:

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///pipeline.db")
print(pd.read_sql("SELECT * FROM predictions", engine))
```

## Notes

- The pipeline only processes rows that don't yet have a prediction (LEFT JOIN filter), so re-running it is safe and idempotent until new rows are added.
- `model.joblib` and `pipeline.db` are listed in `.gitignore` — they are runtime artefacts, not source code.
