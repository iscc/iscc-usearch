"""Utility functions for iscc-usearch."""

import os
import time
from contextlib import contextmanager
from pathlib import Path

from loguru import logger


__all__ = ["atomic_write", "timer"]


@contextmanager
def atomic_write(target):
    """Context manager for atomic file writes via temp file + rename.

    Writes to a temporary sibling file, then atomically replaces the target
    via os.replace(). Prevents corrupt partial writes from process crashes.

    Does not call fsync — provides process-crash safety, not power-loss durability.
    Assumes single-writer (no concurrent saves to the same target).

    :param target: Final destination path (str or Path)
    """
    target = Path(target)
    tmp = target.parent / (target.name + ".tmp")
    try:
        yield tmp
        if tmp.exists():
            os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


class timer:
    """Context manager for timing code blocks and logging elapsed duration.

    Logs a message with the elapsed time on exit using loguru.

    :param message: Description of the operation being timed.
    :param log_start: If True, log a "started" message on entry.
    :param level: Log level for messages (default: "DEBUG").
    """

    def __init__(self, message: str, log_start=False, level="DEBUG"):
        self.message = message
        self.log_start = log_start
        self.level = level

    def __enter__(self):
        """Start the timer."""
        # Record the start time first to ensure it's set before any potential errors
        self.start_time = time.perf_counter()
        if self.log_start:
            logger.log(self.level, f"{self.message} - started")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the timer and log elapsed duration."""
        # Calculate the elapsed time
        elapsed_time = time.perf_counter() - self.start_time
        # Log the message with the elapsed time
        logger.log(self.level, f"{self.message} - completed ({elapsed_time:.4f} seconds)")
        return False
