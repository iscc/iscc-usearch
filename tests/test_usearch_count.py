"""
Confirm the expected behavior of usearch Index.count() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- multi=False (single vector per key) and multi=True (multiple vectors per key)
- Counting existing vs non-existent keys
- Batch count operations
"""

import numpy as np
from numpy.testing import assert_array_equal
from usearch.index import Index, MetricKind, ScalarKind


# Tests for Index.count() with single keys


def test_count_single_existing_key_returns_one():
    """Counting an existing key with multi=False returns 1."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.count(1)

    expected = 1
    assert result == expected
    assert isinstance(result, (int, np.integer))


def test_count_single_missing_key_returns_zero():
    """Counting a non-existent key returns 0."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.count(999)

    expected = 0
    assert result == expected


def test_count_empty_index_returns_zero():
    """Counting any key in an empty index returns 0."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    result = idx.count(1)

    expected = 0
    assert result == expected


def test_count_after_remove_returns_zero():
    """After removing a key, count() returns 0 for that key."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.remove(1)

    result = idx.count(1)

    expected = 0
    assert result == expected


# Tests for Index.count() with batch keys


def test_count_batch_keys_returns_array():
    """Counting batch of keys returns numpy array of counts."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.count([1, 2, 3])

    expected = np.array([1, 1, 1], dtype=np.uint64)

    assert isinstance(result, np.ndarray)
    assert_array_equal(result, expected)


def test_count_batch_mixed_existing_and_missing():
    """Counting batch with some existing and some missing keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    result = idx.count([1, 999, 2, 888])

    expected = np.array([1, 0, 1, 0], dtype=np.uint64)

    assert isinstance(result, np.ndarray)
    assert_array_equal(result, expected)


def test_count_batch_all_missing_keys():
    """Counting batch of non-existent keys returns array of zeros."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.count([10, 20, 30])

    expected = np.array([0, 0, 0], dtype=np.uint64)

    assert isinstance(result, np.ndarray)
    assert_array_equal(result, expected)


def test_count_empty_batch_returns_empty_array():
    """Counting empty batch returns empty numpy array."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.count([])

    assert isinstance(result, np.ndarray)
    assert len(result) == 0


# Tests for Index.count() with multi=True


def test_count_with_multi_true_returns_vector_count_per_key():
    """With multi=True, count() returns number of vectors for that key."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Add 3 vectors to key 1
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.count(1)

    expected = 3
    assert result == expected


def test_count_with_multi_true_single_vector():
    """With multi=True, key with single vector returns count of 1."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.count(1)

    expected = 1
    assert result == expected


def test_count_with_multi_true_batch_keys():
    """With multi=True, batch count returns array with vector counts per key."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Key 1: 3 vectors
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    # Key 2: 2 vectors
    idx.add(2, np.array([50, 60, 70, 80], dtype=np.uint8))
    idx.add(2, np.array([90, 100, 110, 120], dtype=np.uint8))

    # Key 3: 1 vector
    idx.add(3, np.array([255, 254, 253, 252], dtype=np.uint8))

    result = idx.count([1, 2, 3, 999])

    expected = np.array([3, 2, 1, 0], dtype=np.uint64)

    assert isinstance(result, np.ndarray)
    assert_array_equal(result, expected)


def test_count_with_multi_true_after_partial_remove():
    """With multi=True, count after adding duplicate vectors to same key."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    vector = np.array([178, 204, 60, 240], dtype=np.uint8)

    # Add same vector 5 times to key 1
    for _ in range(5):
        idx.add(1, vector)

    result = idx.count(1)

    expected = 5
    assert result == expected


# Edge cases


def test_count_with_large_key_values():
    """Count works with large uint64 key values."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    large_key = 2**63 - 1  # Near max uint64
    idx.add(large_key, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.count(large_key)

    expected = 1
    assert result == expected


def test_count_with_zero_key():
    """Count works with key value 0."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(0, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.count(0)

    expected = 1
    assert result == expected


# Integration tests


def test_count_batch_vs_loop_equivalence():
    """Batch count() should give same results as individual counts."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Key 1: 3 vectors
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    # Key 2: 1 vector
    idx.add(2, np.array([50, 60, 70, 80], dtype=np.uint8))

    keys = [1, 999, 2]

    # Batch count
    batch_result = idx.count(keys)

    # Individual counts
    individual_results = [idx.count(k) for k in keys]

    # Should be equivalent
    assert_array_equal(batch_result, np.array(individual_results))
