"""
Confirm the expected behavior of usearch Index.view() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- Viewing from file path (memory-mapped)
- Viewing from buffer
- Read-only behavior after view
- Progress callback support
- Exact search capability after view
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from usearch.index import Index, MetricKind, ScalarKind


def test_view_from_path_enables_search(tmp_path):
    """Viewing index from file path enables search operations."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    file_path = tmp_path / "test.usearch"
    idx.save(str(file_path))

    viewed = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    viewed.view(str(file_path))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = viewed.search(query, count=2)

    assert result.keys[0] == 1
    assert result.distances[0] == 0.0
    assert len(result) == 2


def test_view_from_buffer_enables_search():
    """Viewing index from buffer enables search operations."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    buffer = idx.save()

    viewed = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    viewed.view(buffer)

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = viewed.search(query, count=1)

    assert result.keys[0] == 1
    assert result.distances[0] == 0.0


def test_view_is_read_only_add_raises(tmp_path):
    """After view(), attempting to add() raises exception documenting read-only nature."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "readonly.usearch"
    idx.save(str(file_path))

    viewed = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    viewed.view(str(file_path))

    with pytest.raises(Exception):
        viewed.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))


def test_view_preserves_key_vector_mappings(tmp_path):
    """View preserves all key-vector mappings from saved index."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    vector1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vector2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    vector3 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(1, vector1)
    idx.add(2, vector2)
    idx.add(3, vector3)

    file_path = tmp_path / "mappings.usearch"
    idx.save(str(file_path))

    viewed = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    viewed.view(str(file_path))

    assert_array_equal(viewed.get(1), vector1)
    assert_array_equal(viewed.get(2), vector2)
    assert_array_equal(viewed.get(3), vector3)


def test_view_with_progress_callback(tmp_path):
    """Viewing with progress callback invokes callback with processed and total counts."""
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

    viewed = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    viewed.view(str(file_path), progress=progress_callback)

    # Verify callback was called at least once
    assert len(progress_calls) > 0
    # Verify parameters are integers
    for processed, total in progress_calls:
        assert isinstance(processed, int)
        assert isinstance(total, int)
        assert processed <= total


def test_view_then_search_exact_works(tmp_path):
    """Viewed index supports exact search with exact=True."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    file_path = tmp_path / "exact.usearch"
    idx.save(str(file_path))

    viewed = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    viewed.view(str(file_path))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = viewed.search(query, count=2, exact=True)

    assert result.keys[0] == 1
    assert result.distances[0] == 0.0
    assert len(result) == 2


def test_view_uses_instance_path_when_none_passed(tmp_path):
    """View with no arguments uses path from Index initialization."""
    file_path = tmp_path / "instance_path.usearch"

    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, path=str(file_path))
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.save()

    viewed = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, path=str(file_path))
    viewed.view()

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = viewed.search(query, count=1)

    assert result.keys[0] == 1
    assert result.distances[0] == 0.0
