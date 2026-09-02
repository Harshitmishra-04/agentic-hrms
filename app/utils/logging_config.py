import logging
import sys

LOGGER_NAME = "hr_ai"
_configured = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure process-wide logging for the FastAPI app lifecycle."""
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not _configured:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        _configured = True

    return logger


def get_logger() -> logging.Logger:
    if not _configured:
        return setup_logging()
    return logging.getLogger(LOGGER_NAME)
