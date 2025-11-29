import time
from loguru import logger


__all__ = ["timer"]


class timer:
    def __init__(self, message: str, log_start=False):
        self.message = message
        self.log_start = log_start

    def __enter__(self):
        # Record the start time first to ensure it's set before any potential errors
        self.start_time = time.perf_counter()
        if self.log_start:
            logger.info(f"{self.message} - started")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Calculate the elapsed time
        elapsed_time = time.perf_counter() - self.start_time
        # Log the message with the elapsed time
        logger.info(f"{self.message} - completed ({elapsed_time:.4f} seconds)")
        return False
