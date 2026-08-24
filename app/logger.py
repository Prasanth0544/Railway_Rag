"""
Logging configuration for Railway RAG Assistant.

Usage:
    from app.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Message here")
"""

import logging
import sys
import io


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Force UTF-8 on Windows — default console codec is ASCII/cp1252
        # which crashes when logging em-dash, arrows, or emoji characters.
        try:
            stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        except Exception:
            stream = sys.stdout  # fallback: hope stdout is already UTF-8

        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                "[%(levelname)s] %(name)s: %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger
