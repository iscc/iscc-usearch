"""
Confirm the expected behavior of usearch Index.load() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- Loading from file path
- Loading from buffer
- Loading with progress callback
- Loading then performing operations
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from usearch.index import Index, MetricKind, ScalarKind


def test_load_from_path_restores_vectors(tmp_path):
    """Loading from a file path restores all vectors with their keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "test.usearch"
    idx.save(str(file_path))

    loaded = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    loaded.load(str(file_path))

    result = loaded.get(1)
    expected = np.array([178, 204, 60, 240], dtype=np.uint8)

    assert_array_equal(result, expected)


def test_load_from_buffer_restores_vectors():
    """Loading from a buffer restores all vectors with their keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    buffer = idx.save()

    loaded = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    loaded.load(buffer)

    result = loaded.get(1)
    expected = np.array([178, 204, 60, 240], dtype=np.uint8)

    assert_array_equal(result, expected)


def test_load_restores_correct_key_vector_mappings(tmp_path):
    """Loading restores correct mappings between keys and vectors."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    vector1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vector2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    vector3 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(10, vector1)
    idx.add(20, vector2)
    idx.add(30, vector3)

    file_path = tmp_path / "mappings.usearch"
    idx.save(str(file_path))

    loaded = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    loaded.load(str(file_path))

    assert_array_equal(loaded.get(10), vector1)
    assert_array_equal(loaded.get(20), vector2)
    assert_array_equal(loaded.get(30), vector3)


def test_load_with_progress_callback(tmp_path):
    """Loading with progress callback invokes the callback with processed and total counts."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    file_path = tmp_path / "progress.usearch"
    idx.save(str(file_path))

    progress_calls = []

    def progress_callback(processed: int, total: int) -> bool:
        progress_calls.append((processed, total))
        return True

    loaded = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    loaded.load(str(file_path), progress=progress_callback)

    # Verify callback was called at least once
    assert len(progress_calls) > 0
    # Verify parameters are integers
    for processed, total in progress_calls:
        assert isinstance(processed, int)
        assert isinstance(total, int)
        assert processed <= total


def test_load_nonexistent_file_raises_error(tmp_path):
    """Loading from a nonexistent file raises an error."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    nonexistent_path = tmp_path / "nonexistent.usearch"

    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        idx.load(str(nonexistent_path))


def test_load_then_add_works(tmp_path):
    """Loading an existing index and then adding new vectors works correctly."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    vector1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vector1)

    file_path = tmp_path / "add_after_load.usearch"
    idx.save(str(file_path))

    loaded = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    loaded.load(str(file_path))

    vector2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    loaded.add(2, vector2)

    # Verify both old and new vectors are accessible
    assert_array_equal(loaded.get(1), vector1)
    assert_array_equal(loaded.get(2), vector2)


def test_load_then_search_works(tmp_path):
    """Loading an existing index and then searching returns expected results."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    vector1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vector2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    idx.add(1, vector1)
    idx.add(2, vector2)

    file_path = tmp_path / "search_after_load.usearch"
    idx.save(str(file_path))

    loaded = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    loaded.load(str(file_path))

    matches = loaded.search(vector1, count=1)

    assert len(matches.keys) == 1
    assert matches.keys[0] == 1


def test_load_then_remove_works(tmp_path):
    """Loading an existing index and then removing a key works correctly."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    vector1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vector2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    idx.add(1, vector1)
    idx.add(2, vector2)

    file_path = tmp_path / "remove_after_load.usearch"
    idx.save(str(file_path))

    loaded = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    loaded.load(str(file_path))

    removed_count = loaded.remove(1)

    assert removed_count == 1
    # Key 2 should still be accessible
    assert_array_equal(loaded.get(2), vector2)


def test_load_uses_instance_path_when_none_passed(tmp_path):
    """Loading with no arguments uses the instance's path attribute."""
    file_path = tmp_path / "instance_path.usearch"

    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, path=str(file_path))
    vector = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vector)
    idx.save()

    loaded = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, path=str(file_path))
    loaded.load()

    result = loaded.get(1)
    expected = np.array([178, 204, 60, 240], dtype=np.uint8)

    assert_array_equal(result, expected)
