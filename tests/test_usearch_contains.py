"""
Confirm the expected behavior of usearch Index.contains() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- multi=False (single vector per key)
- Checking existing vs non-existent keys
- Batch containment checks
"""

import numpy as np
from numpy.testing import assert_array_equal
from usearch.index import Index, MetricKind, ScalarKind


# Tests for Index.contains() with single keys


def test_contains_single_existing_key_returns_true():
    """Checking if an existing key is in the index returns True."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.contains(1)

    expected = True
    assert result is expected
    assert isinstance(result, bool)


def test_contains_single_missing_key_returns_false():
    """Checking if a non-existent key is in the index returns False."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.contains(999)

    expected = False
    assert result is expected
    assert isinstance(result, bool)


def test_contains_in_empty_index_returns_false():
    """Checking for any key in empty index returns False."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    result = idx.contains(1)

    expected = False
    assert result is expected


def test_contains_after_remove_returns_false():
    """After removing a key, contains() returns False."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.remove(1)

    result = idx.contains(1)

    expected = False
    assert result is expected


# Tests for Index.contains() with batch keys


def test_contains_batch_all_existing_keys():
    """Checking batch of existing keys returns array of True values."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.contains([1, 2, 3])

    expected = np.array([True, True, True], dtype=bool)

    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert_array_equal(result, expected)


def test_contains_batch_all_missing_keys():
    """Checking batch of non-existent keys returns array of False values."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.contains([10, 20, 30])

    expected = np.array([False, False, False], dtype=bool)

    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert_array_equal(result, expected)


def test_contains_batch_mixed_existing_and_missing():
    """Checking batch with some existing and some missing keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    result = idx.contains([1, 999, 2, 888])

    expected = np.array([True, False, True, False], dtype=bool)

    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert_array_equal(result, expected)


def test_contains_empty_batch_returns_empty_array():
    """Checking empty batch returns empty boolean array."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.contains([])

    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert len(result) == 0


# Tests for Index.contains() with multi=True


def test_contains_with_multi_true_single_vector():
    """Checking key with single vector (multi=True) returns True."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.contains(1)

    expected = True
    assert result is expected


def test_contains_with_multi_true_multiple_vectors():
    """Checking key with multiple vectors (multi=True) returns True."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.contains(1)

    expected = True
    assert result is expected


# Tests for __contains__ operator (Python's 'in' syntax)


def test_in_operator_uses_contains():
    """Python 'in' operator works via __contains__ method."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    # Using 'in' operator
    assert 1 in idx
    assert 999 not in idx


# Performance characteristics tests


def test_contains_batch_vs_loop_equivalence():
    """Batch contains() should give same results as individual checks."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    keys = [1, 999, 2, 888, 3]

    # Batch check
    batch_result = idx.contains(keys)

    # Individual checks
    individual_results = [idx.contains(k) for k in keys]

    # Should be equivalent
    assert_array_equal(batch_result, np.array(individual_results))


# Tests with enable_key_lookups=False


def test_contains_returns_false_when_key_lookups_disabled():
    """
    When enable_key_lookups=False, contains() returns False for all keys.

    Similar to get(), disabling key lookups means the reverse mapping
    from keys to vectors doesn't exist, so contains() cannot determine
    if a key is present.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, enable_key_lookups=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    # Both existing and missing keys return False
    result_exists = idx.contains(1)
    result_missing = idx.contains(999)

    expected = False

    assert result_exists is expected
    assert result_missing is expected


# Edge case tests


def test_contains_with_large_key_values():
    """Contains works with large uint64 key values."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    large_key = 2**63 - 1  # Near max uint64
    idx.add(large_key, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.contains(large_key)

    expected = True
    assert result is expected


def test_contains_with_zero_key():
    """Contains works with key value 0."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(0, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.contains(0)

    expected = True
    assert result is expected
