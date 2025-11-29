"""Unit tests for timer context manager."""

import time

from loguru import logger

from iscc_usearch.utils import timer


def test_timer_logs_completion_message(capsys):
    # type: () -> None
    """Test that timer logs completion message with elapsed time."""
    log_messages = []
    handler_id = logger.add(lambda msg: log_messages.append(msg), format="{message}")

    with timer("Test operation"):
        time.sleep(0.01)

    logger.remove(handler_id)

    assert len(log_messages) == 1
    assert "Test operation - completed" in log_messages[0]
    assert "seconds" in log_messages[0]


def test_timer_with_log_start_true():
    # type: () -> None
    """Test that timer logs start message when log_start=True."""
    log_messages = []
    handler_id = logger.add(lambda msg: log_messages.append(msg), format="{message}")

    with timer("Test operation", log_start=True):
        time.sleep(0.01)

    logger.remove(handler_id)

    assert len(log_messages) == 2
    assert "Test operation - started" in log_messages[0]
    assert "Test operation - completed" in log_messages[1]


def test_timer_returns_self():
    # type: () -> None
    """Test that timer context manager returns self for 'as' syntax."""
    with timer("Test operation") as t:
        assert isinstance(t, timer)
        assert t.message == "Test operation"
        assert hasattr(t, "start_time")


def test_timer_exit_returns_false():
    # type: () -> None
    """Test that __exit__ returns False (doesn't suppress exceptions)."""
    t = timer("Test operation")
    t.__enter__()
    result = t.__exit__(None, None, None)
    assert result is False
