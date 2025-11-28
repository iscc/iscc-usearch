"""
Confirm the expected behavior of usearch Index.restore() static method with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- Restoring from file path
- Restoring from buffer
- view=True (memory-mapped read-only)
- view=False (loaded into memory, writable)
"""

import numpy as np
from numpy.testing import assert_array_equal
from usearch.index import Index, MetricKind, ScalarKind


def test_restore_from_path_creates_index(tmp_path):
    """Restoring from file path creates new Index instance."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "test.usearch"
    idx.save(str(file_path))

    restored = Index.restore(str(file_path))

    assert restored is not None
    assert isinstance(restored, Index)


def test_restore_from_buffer_creates_index():
    """Restoring from buffer creates new Index instance."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    buffer = idx.save()
    restored = Index.restore(buffer)

    assert restored is not None
    assert isinstance(restored, Index)


def test_restore_with_view_true(tmp_path):
    """Index.restore with view=True creates read-only memory-mapped index."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    vector = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vector)

    file_path = tmp_path / "view.usearch"
    idx.save(str(file_path))

    restored = Index.restore(str(file_path), view=True)

    assert restored is not None
    # Search works on read-only view
    result = restored.search(vector, count=1)
    assert result.keys[0] == 1
    assert result.distances[0] == 0.0


def test_restore_with_view_false(tmp_path):
    """Index.restore with view=False creates writable index."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    vector1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vector1)

    file_path = tmp_path / "load.usearch"
    idx.save(str(file_path))

    restored = Index.restore(str(file_path), view=False)

    assert restored is not None
    # Add works on loaded writable index
    vector2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    result = restored.add(2, vector2)
    assert result[0] == 2
    assert 2 in restored


def test_restore_preserves_metric(tmp_path):
    """Restored index has same metric_kind as original."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "metric.usearch"
    idx.save(str(file_path))

    restored = Index.restore(str(file_path))

    assert restored is not None
    assert restored.metric_kind == MetricKind.Hamming


def test_restore_nonexistent_returns_none(tmp_path):
    """Index.restore with nonexistent file returns None."""
    nonexistent_path = tmp_path / "nonexistent.usearch"

    restored = Index.restore(str(nonexistent_path))

    assert restored is None


def test_restore_preserves_vectors(tmp_path):
    """Restored index has all vectors accessible via get()."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    vector1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vector2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    vector3 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(1, vector1)
    idx.add(2, vector2)
    idx.add(3, vector3)

    file_path = tmp_path / "vectors.usearch"
    idx.save(str(file_path))

    restored = Index.restore(str(file_path))

    assert restored is not None
    assert_array_equal(restored.get(1), vector1)
    assert_array_equal(restored.get(2), vector2)
    assert_array_equal(restored.get(3), vector3)


def test_restore_preserves_ndim(tmp_path):
    """Restored index has same ndim as original."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "ndim.usearch"
    idx.save(str(file_path))

    restored = Index.restore(str(file_path))

    assert restored is not None
    assert restored.ndim == 32


def test_restore_default_is_load_not_view(tmp_path):
    """By default view=False, so add() should work after restore."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    vector1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vector1)

    file_path = tmp_path / "default.usearch"
    idx.save(str(file_path))

    # Restore without specifying view parameter (defaults to view=False)
    restored = Index.restore(str(file_path))

    assert restored is not None
    # Should be able to add since view=False is the default
    vector2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    result = restored.add(2, vector2)
    assert result[0] == 2
