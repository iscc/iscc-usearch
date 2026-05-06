"""
Test usearch Index constructor parameters.

Confirms expected behavior when creating Index with various parameter combinations:
- ndim (dimensionality)
- metric (string or MetricKind)
- dtype (string or ScalarKind)
- connectivity
- expansion_add
- expansion_search
- multi (single vs multiple vectors per key)
- enable_key_lookups
- path (creating and loading indexes from files)
"""

import numpy as np
from numpy.testing import assert_array_equal
from usearch.index import Index, MetricKind, ScalarKind


def test_init_with_ndim():
    """Creating index with specific ndim sets dimensionality correctly."""
    idx = Index(ndim=64, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    assert idx.ndim == 64


def test_init_with_metric_string():
    """Creating index with lowercase metric string works correctly."""
    idx = Index(ndim=32, metric="hamming", dtype=ScalarKind.B1)

    vec = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vec)

    matches = idx.search(vec, count=1)
    assert matches.keys[0] == 1


def test_init_with_metric_kind():
    """Creating index with MetricKind enum works correctly."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    vec = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vec)

    matches = idx.search(vec, count=1)
    assert matches.keys[0] == 1


def test_init_with_dtype_string():
    """Creating index with lowercase dtype string works correctly."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype="b1")

    vec = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vec)

    stored = idx.get(1)
    assert stored.dtype == np.uint8
    assert np.array_equal(stored, vec)


def test_init_with_dtype_scalarkind():
    """Creating index with ScalarKind enum works correctly."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    vec = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vec)

    stored = idx.get(1)
    assert stored.dtype == np.uint8
    assert np.array_equal(stored, vec)


def test_init_with_connectivity():
    """Creating index with connectivity parameter sets connectivity property."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, connectivity=16)

    assert idx.connectivity == 16


def test_init_with_expansion_add():
    """Creating index with expansion_add parameter sets expansion_add property."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, expansion_add=128)

    assert idx.expansion_add == 128


def test_init_with_expansion_search():
    """Creating index with expansion_search parameter sets expansion_search property."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, expansion_search=64)

    assert idx.expansion_search == 64


def test_init_with_multi_true():
    """Creating index with multi=True sets multi property to True."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    assert idx.multi is True

    # Verify we can add multiple vectors to same key
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))

    stored = idx.get(1)
    assert stored.shape == (2, 4)


def test_init_with_multi_false():
    """Creating index with multi=False sets multi property to False."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    assert idx.multi is False

    # Verify duplicate keys are silently skipped
    vector1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vector1)
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))

    assert len(idx) == 1
    assert_array_equal(idx.get(1), vector1)


def test_init_with_enable_key_lookups_false():
    """Creating index with enable_key_lookups=False makes contains() return False for existing keys."""
    idx = Index(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        enable_key_lookups=False,
    )

    vec = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vec)

    # With key lookups disabled, contains() returns False even for existing keys
    assert 1 not in idx


def test_init_defaults():
    """Creating index with defaults has reasonable default values."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    assert idx.ndim == 32
    assert idx.multi is False
    assert idx.connectivity > 0  # Should have a default connectivity
    assert idx.expansion_add > 0  # Should have a default expansion_add
    assert idx.expansion_search > 0  # Should have a default expansion_search


def test_init_with_path_creates_file(tmp_path):
    """Creating index with path parameter creates file when saved."""
    index_path = tmp_path / "test.usearch"
    idx = Index(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=str(index_path),
    )

    vec = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vec)

    idx.save()

    assert index_path.exists()


def test_init_with_path_loads_existing(tmp_path):
    """Creating index with path to existing file loads the saved index."""
    index_path = tmp_path / "test.usearch"

    # Create and save an index
    idx1 = Index(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=str(index_path),
    )

    vec1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vec2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    idx1.add(1, vec1)
    idx1.add(2, vec2)

    idx1.save()

    # Create new index with same path - should load existing data
    idx2 = Index(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=str(index_path),
    )

    # Verify vectors are present
    assert 1 in idx2
    assert 2 in idx2

    stored1 = idx2.get(1)
    stored2 = idx2.get(2)

    assert np.array_equal(stored1, vec1)
    assert np.array_equal(stored2, vec2)
