"""Run with `kip-worker [--once]` or `python -m kip.worker`."""

import argparse
import logging
import signal
import threading

from sqlalchemy.exc import OperationalError

from kip.db.session import make_engine, session_factory
from kip.settings import Settings
from kip.storage import LocalObjectStore
from kip.worker.runner import Worker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process ingestion jobs: parse, embed, index in Qdrant and publish."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Claim at most one job, then exit.")
    mode.add_argument(
        "--until-idle", action="store_true", help="Process due jobs until none remain, then exit."
    )
    args = parser.parse_args()
    settings = Settings()
    engine = make_engine(settings.database_url.get_secret_value())
    worker = Worker(session_factory(engine), LocalObjectStore(settings.storage_root), settings)
    stop = threading.Event()

    def shutdown(_signal: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    logging.basicConfig(level=logging.INFO)
    # httpx logs every Qdrant request URL at INFO; keep worker logs to job-level events.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        while not stop.is_set():
            try:
                worked = worker.run_once()
            except OperationalError:
                if args.once or args.until_idle:
                    # Do not print SQL parameters / connection credentials in a traceback.
                    raise SystemExit("database unavailable") from None
                logging.warning("Database unavailable; waiting to reconnect.")
                worked = False
            if args.once or (args.until_idle and not worked):
                break
            if not worked:
                stop.wait(settings.poll_seconds)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
