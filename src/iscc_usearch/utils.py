"""Utility functions for iscc-usearch."""

import time
from loguru import logger


__all__ = ["timer"]


class timer:
    """Context manager for timing code blocks and logging elapsed duration.

    Logs a message with the elapsed time on exit using loguru.

    :param message: Description of the operation being timed.
    :param log_start: If True, log a "started" message on entry.
    """

    def __init__(self, message: str, log_start=False):
        self.message = message
        self.log_start = log_start

    def __enter__(self):
        """Start the timer."""
        # Record the start time first to ensure it's set before any potential errors
        self.start_time = time.perf_counter()
        if self.log_start:
            logger.info(f"{self.message} - started")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the timer and log elapsed duration."""
        # Calculate the elapsed time
        elapsed_time = time.perf_counter() - self.start_time
        # Log the message with the elapsed time
        logger.info(f"{self.message} - completed ({elapsed_time:.4f} seconds)")
        return False
