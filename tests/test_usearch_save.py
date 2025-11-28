"""
Confirm the expected behavior of usearch Index.save() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- Saving to file path
- Saving to buffer
"""

import os

import numpy as np
from numpy.testing import assert_array_equal
from usearch.index import Index, MetricKind, ScalarKind


def test_save_to_path_creates_file(tmp_path):
    """Saving to a file path creates the file on disk."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "test.usearch"
    idx.save(str(file_path))

    assert os.path.exists(file_path)


def test_save_to_buffer_returns_bytes():
    """Saving without a path returns bytes buffer."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.save()

    assert isinstance(result, (bytes, bytearray))
    assert len(result) > 0


def test_save_empty_index(tmp_path):
    """Saving an empty index works without error."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    file_path = tmp_path / "empty.usearch"
    idx.save(str(file_path))

    assert os.path.exists(file_path)


def test_save_with_progress_callback(tmp_path):
    """Saving with progress callback invokes the callback with processed and total counts."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    progress_calls = []

    def progress_callback(processed: int, total: int) -> bool:
        progress_calls.append((processed, total))
        return True

    file_path = tmp_path / "progress.usearch"
    idx.save(str(file_path), progress=progress_callback)

    # Verify callback was called at least once
    assert len(progress_calls) > 0
    # Verify parameters are integers
    for processed, total in progress_calls:
        assert isinstance(processed, int)
        assert isinstance(total, int)
        assert processed <= total


def test_save_preserves_all_vectors(tmp_path):
    """Saving and loading preserves all vectors with their keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    vector1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vector2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    vector3 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(1, vector1)
    idx.add(2, vector2)
    idx.add(3, vector3)

    file_path = tmp_path / "vectors.usearch"
    idx.save(str(file_path))

    loaded = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    loaded.load(str(file_path))

    assert_array_equal(loaded.get(1), vector1)
    assert_array_equal(loaded.get(2), vector2)
    assert_array_equal(loaded.get(3), vector3)


def test_save_preserves_metadata(tmp_path):
    """Saving preserves index metadata (dimensions, metric, dtype)."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "metadata.usearch"
    idx.save(str(file_path))

    metadata = Index.metadata(str(file_path))

    assert metadata is not None
    assert metadata["dimensions"] == 32
    assert metadata["kind_metric"] == MetricKind.Hamming
    assert metadata["kind_scalar"] == ScalarKind.B1


def test_save_to_path_with_existing_file_overwrites(tmp_path):
    """Saving to an existing file path overwrites the file."""
    file_path = tmp_path / "overwrite.usearch"

    idx1 = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx1.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx1.save(str(file_path))

    idx2 = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx2.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx2.save(str(file_path))

    loaded = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    loaded.load(str(file_path))

    # Should only contain key 2 (from second save)
    expected_vector = np.array([100, 150, 200, 250], dtype=np.uint8)
    assert_array_equal(loaded.get(2), expected_vector)


def test_save_returns_none_when_saving_to_path(tmp_path):
    """Saving to a path returns None, not bytes."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "test.usearch"
    result = idx.save(str(file_path))

    assert result is None
