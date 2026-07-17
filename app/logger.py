"""
Logging configuration for Railway RAG Assistant.

Usage:
    from app.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Message here")
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(levelname)s] %(name)s: %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger
