"""
train_model.py
--------------
Trains a simple RandomForestClassifier on synthetic data and saves
the fitted model to disk as model.joblib.

The synthetic dataset deliberately uses the same feature names and
value ranges as the rows seeded in init_db.py so that the pipeline
can make meaningful predictions on that data.

Run once (or whenever you want to retrain):
    python train_model.py
"""

import logging

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config import MODEL_PATH, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 500, random_state: int = 42):
    """
    Build a labelled dataset that mirrors the shape of input_data.

    Label rule (simple and deterministic):
        class 1  if  feature_1 > 5  AND  feature_3 > 0.5
        class 0  otherwise
    """
    rng = np.random.default_rng(random_state)

    feature_1 = rng.uniform(0.0, 10.0, n_samples)
    feature_2 = rng.uniform(-5.0, 5.0, n_samples)
    feature_3 = rng.uniform(0.0, 1.0, n_samples)
    feature_4 = rng.uniform(100.0, 200.0, n_samples)

    X = np.column_stack([feature_1, feature_2, feature_3, feature_4])
    y = ((feature_1 > 5) & (feature_3 > 0.5)).astype(int)

    return X, y


def train(X_train, y_train) -> RandomForestClassifier:
    """Fit a RandomForestClassifier and return it."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> None:
    """Log a classification report so we have a quick sanity check."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["class_0", "class_1"])
    logger.info("Evaluation results:\n%s", report)


def save_model(model, path: str) -> None:
    """Persist the trained model to disk."""
    joblib.dump(model, path)
    logger.info("Model saved to '%s'.", path)


def main() -> None:
    logger.info("Generating synthetic training data …")
    X, y = generate_synthetic_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Training RandomForestClassifier on %d samples …", len(X_train))
    model = train(X_train, y_train)

    evaluate(model, X_test, y_test)
    save_model(model, MODEL_PATH)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
