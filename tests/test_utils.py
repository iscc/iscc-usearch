"""Unit tests for utility functions."""

import os
import time
from unittest.mock import patch

from loguru import logger

from iscc_usearch.utils import durable_write, timer


def test_durable_write_creates_file(tmp_path):
    """durable_write writes data and the target file exists afterward."""
    target = tmp_path / "out.bin"
    data = b"\x00\x0a\x0d\xff" * 100
    durable_write(data, target)
    assert target.read_bytes() == data
    assert not (tmp_path / "out.bin.tmp").exists()


def test_durable_write_cleans_up_on_failure(tmp_path):
    """Temp file is removed when os.write raises."""
    target = tmp_path / "out.bin"
    with patch("iscc_usearch.utils.os.write", side_effect=OSError("disk full")):
        try:
            durable_write(b"data", target)
        except OSError:
            pass
    assert not target.exists()
    assert not (tmp_path / "out.bin.tmp").exists()


def test_durable_write_handles_short_writes(tmp_path):
    """Write loop retries on partial os.write results."""
    target = tmp_path / "out.bin"
    data = b"abcdef"
    real_write = os.write
    call_count = 0

    def short_write(fd, buf):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return real_write(fd, buf[:3])
        return real_write(fd, buf)

    with patch("iscc_usearch.utils.os.write", side_effect=short_write):
        durable_write(data, target)
    assert target.read_bytes() == data
    assert call_count == 2


def test_durable_write_skips_dir_fsync_on_windows(tmp_path):
    """When _IS_WIN is True, durable_write skips the parent directory fsync."""
    import iscc_usearch.utils as utils

    target = tmp_path / "out.bin"
    saved_is_win = utils._IS_WIN
    try:
        utils._IS_WIN = True
        durable_write(b"data", target)
        assert target.read_bytes() == b"data"
    finally:
        utils._IS_WIN = saved_is_win


def test_durable_write_dir_fsync_on_posix(tmp_path):
    """When _IS_WIN is False, durable_write fsyncs the parent directory."""
    import iscc_usearch.utils as utils

    target = tmp_path / "out.bin"
    saved_is_win = utils._IS_WIN
    try:
        utils._IS_WIN = False
        real_open = os.open
        real_close = os.close
        real_fsync = os.fsync
        dir_synced = False

        def patched_open(path, flags, *args, **kwargs):
            if flags == os.O_RDONLY:
                nonlocal dir_synced
                dir_synced = True
                return 999
            return real_open(path, flags, *args, **kwargs)

        with (
            patch.object(os, "open", side_effect=patched_open),
            patch.object(os, "fsync", side_effect=lambda fd: None if fd == 999 else real_fsync(fd)),
            patch.object(os, "close", side_effect=lambda fd: None if fd == 999 else real_close(fd)),
        ):
            durable_write(b"posix", target)
        assert dir_synced
        assert target.read_bytes() == b"posix"
    finally:
        utils._IS_WIN = saved_is_win


def test_durable_write_path_bound_index(tmp_path):
    """Index.save(path) works correctly even when self.path is set."""
    from iscc_usearch.index import Index
    from usearch.index import MetricKind, ScalarKind
    import numpy as np

    bound_path = tmp_path / "bound.usearch"
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, path=str(bound_path))
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    explicit_path = tmp_path / "explicit.usearch"
    idx.save(str(explicit_path))

    assert explicit_path.exists()
    loaded = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    loaded.load(str(explicit_path))
    assert 1 in loaded


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
