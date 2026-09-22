import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER_NAME = "stock_price_lstm"


def configure_logging() -> None:
    """Configure terminal and file logging once per process."""
    project_logger = logging.getLogger(LOGGER_NAME)

    if project_logger.handlers:
        return

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_dir / "project.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    project_logger.setLevel(logging.INFO)
    project_logger.propagate = False
    project_logger.addHandler(console_handler)
    project_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the project's logging hierarchy."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def run_logged(
    main_function: Callable[[], None],
    logger: logging.Logger,
) -> None:
    """Run a command, recording its start, completion, or failure."""
    configure_logging()
    logger.info("Started")

    try:
        main_function()
    except Exception:
        logger.exception("Failed")
        raise
    else:
        logger.info("Completed successfully")