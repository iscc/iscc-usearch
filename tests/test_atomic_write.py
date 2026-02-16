"""Tests for atomic_write context manager."""

from unittest.mock import patch

from iscc_usearch.utils import atomic_write


def test_atomic_write_happy_path(tmp_path):
    """Writer succeeds, target contains expected content, no temp file remains."""
    target = tmp_path / "output.dat"

    with atomic_write(target) as tmp:
        tmp.write_bytes(b"hello")

    assert target.read_bytes() == b"hello"
    assert not (tmp_path / "output.dat.tmp").exists()


def test_atomic_write_overwrites_existing_target(tmp_path):
    """Existing file at target is atomically replaced with new content."""
    target = tmp_path / "output.dat"
    target.write_bytes(b"old content")

    with atomic_write(target) as tmp:
        tmp.write_bytes(b"new content")

    assert target.read_bytes() == b"new content"
    assert not (tmp_path / "output.dat.tmp").exists()


def test_atomic_write_writer_failure_leaves_target_untouched(tmp_path):
    """Writer raises, target is untouched, temp file cleaned up."""
    target = tmp_path / "output.dat"
    target.write_bytes(b"original")

    try:
        with atomic_write(target) as tmp:
            tmp.write_bytes(b"partial")
            raise RuntimeError("simulated crash")
    except RuntimeError:
        pass

    assert target.read_bytes() == b"original"
    assert not (tmp_path / "output.dat.tmp").exists()


def test_atomic_write_replace_failure_cleans_up_tmp(tmp_path):
    """os.replace raises PermissionError, temp file is cleaned up, target untouched."""
    target = tmp_path / "output.dat"
    target.write_bytes(b"original")

    try:
        with patch("iscc_usearch.utils.os.replace", side_effect=PermissionError("denied")):
            with atomic_write(target) as tmp:
                tmp.write_bytes(b"new data")
    except PermissionError:
        pass

    assert target.read_bytes() == b"original"
    assert not (tmp_path / "output.dat.tmp").exists()


def test_atomic_write_stale_tmp_overwritten(tmp_path):
    """Pre-existing .tmp from prior crash is overwritten by new save."""
    target = tmp_path / "output.dat"
    stale_tmp = tmp_path / "output.dat.tmp"
    stale_tmp.write_bytes(b"stale from crash")

    with atomic_write(target) as tmp:
        assert tmp == stale_tmp
        tmp.write_bytes(b"fresh data")

    assert target.read_bytes() == b"fresh data"
    assert not stale_tmp.exists()


def test_atomic_write_no_target_on_new_file(tmp_path):
    """Target does not exist before write, created after successful write."""
    target = tmp_path / "brand_new.dat"
    assert not target.exists()

    with atomic_write(target) as tmp:
        tmp.write_bytes(b"created")

    assert target.exists()
    assert target.read_bytes() == b"created"
    assert not (tmp_path / "brand_new.dat.tmp").exists()


def test_atomic_write_writer_creates_no_file(tmp_path):
    """Writer yields but never creates the temp file — no-op, target unchanged."""
    target = tmp_path / "output.dat"
    target.write_bytes(b"original")

    with atomic_write(target) as tmp:
        pass  # Writer does nothing

    assert target.read_bytes() == b"original"
    assert not tmp.exists()
