"""
scheduler.py
------------
Runs the batch prediction pipeline on a fixed interval using
Python's built-in threading and time modules — no extra dependencies.

Usage
~~~~~
    # Foreground (Ctrl-C to stop)
    python scheduler.py

    # Background (Unix / macOS)
    nohup python scheduler.py > logs/scheduler.log 2>&1 &

    # Background (Windows PowerShell)
    Start-Process python -ArgumentList "scheduler.py" -WindowStyle Hidden

The interval is controlled by SCHEDULE_INTERVAL_MINUTES in config.py
(default: 5 minutes). Override with the env var of the same name:
    SCHEDULE_INTERVAL_MINUTES=2 python scheduler.py
"""

import logging
import signal
import threading
import time

from config import SCHEDULE_INTERVAL_MINUTES
from pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Shared flag — set to True by the SIGINT/SIGTERM handler to stop the loop.
_stop_event = threading.Event()


def _handle_shutdown(signum, frame) -> None:  # noqa: ANN001
    logger.info("Shutdown signal received — stopping scheduler …")
    _stop_event.set()


def scheduled_job() -> None:
    """Wrapper so an unhandled exception in the pipeline never kills the scheduler."""
    try:
        run_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during pipeline run: %s", exc)


def main() -> None:
    # Graceful shutdown on Ctrl-C or kill
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    interval_seconds = SCHEDULE_INTERVAL_MINUTES * 60
    logger.info(
        "Scheduler started. Pipeline will run every %d minute(s).",
        SCHEDULE_INTERVAL_MINUTES,
    )

    # Run immediately on startup, then wait for the interval
    scheduled_job()

    while not _stop_event.is_set():
        # Sleep in short bursts so we can react to the stop flag quickly
        for _ in range(interval_seconds):
            if _stop_event.is_set():
                break
            time.sleep(1)

        if not _stop_event.is_set():
            scheduled_job()

    logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
