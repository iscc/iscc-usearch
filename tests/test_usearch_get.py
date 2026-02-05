"""
Confirm the expected behavior of usearch Index.get() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- multi=False (single vector per key)
- multi=True (multiple vectors per key)
"""

import numpy as np
from numpy.testing import assert_array_equal
from usearch.index import Index, MetricKind, ScalarKind


def test_index_get_single_empty_returns_none():
    """This is a regression test for https://github.com/unum-cloud/USearch/issues/663."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result_exists = idx.get(1)
    expected_exists = np.array([178, 204, 60, 240], dtype=np.uint8)
    assert_array_equal(result_exists, expected_exists)

    result_missing = idx.get(2)
    expected_missing = None

    assert result_missing is expected_missing


# Tests for Index.get() with multi=False (single vector per key)


def test_get_single_key_exists_returns_1d_array():
    """Single key that exists returns 1D array."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.get(1)

    expected = np.array([178, 204, 60, 240], dtype=np.uint8)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert_array_equal(result, expected)


def test_get_single_key_missing_returns_none():
    """Single key that doesn't exist returns None."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    result = idx.get(999)

    expected = None

    assert result is expected


def test_get_multiple_keys_all_exist_returns_list_of_1d_arrays():
    """Multiple existing keys return list of 1D arrays."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.get([1, 2, 3])

    expected = [
        np.array([178, 204, 60, 240], dtype=np.uint8),
        np.array([100, 150, 200, 250], dtype=np.uint8),
        np.array([1, 2, 3, 4], dtype=np.uint8),
    ]
    expected_length = 3

    assert isinstance(result, list)
    assert len(result) == expected_length
    assert all(isinstance(r, np.ndarray) and r.ndim == 1 for r in result)
    assert_array_equal(result[0], expected[0])
    assert_array_equal(result[1], expected[1])
    assert_array_equal(result[2], expected[2])


def test_get_multiple_keys_mixed_returns_list_with_none():
    """Multiple keys with some missing return list with None values."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.get([1, 2, 3])

    expected = [
        np.array([178, 204, 60, 240], dtype=np.uint8),
        None,
        np.array([1, 2, 3, 4], dtype=np.uint8),
    ]
    expected_length = 3

    assert isinstance(result, list)
    assert len(result) == expected_length
    assert isinstance(result[0], np.ndarray)
    assert result[1] is None
    assert isinstance(result[2], np.ndarray)
    assert_array_equal(result[0], expected[0])
    assert_array_equal(result[2], expected[2])


def test_get_multiple_keys_all_missing_returns_list_of_none():
    """Multiple non-existing keys return list of None values."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    result = idx.get([10, 20, 30])

    expected = [None, None, None]
    expected_length = 3

    assert isinstance(result, list)
    assert len(result) == expected_length
    assert all(r is None for r in result)
    assert result == expected


# Tests for Index.get() with multi=True (multiple vectors per key)


def test_get_single_key_one_vector_returns_2d_array():
    """Single key with one vector returns 2D array with one row."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.get(1)

    expected = np.array([[178, 204, 60, 240]], dtype=np.uint8)
    expected_shape = (1, 4)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == expected_shape
    assert_array_equal(result, expected)


def test_get_single_key_multiple_vectors_returns_2d_array():
    """Single key with multiple vectors returns 2D array with multiple rows."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.get(1)

    expected = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
            [1, 2, 3, 4],
        ],
        dtype=np.uint8,
    )
    expected_shape = (3, 4)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == expected_shape
    assert_array_equal(result, expected)


def test_get_single_key_missing_returns_none_multi():
    """Single key that doesn't exist returns None (multi=True)."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    result = idx.get(999)

    expected = None

    assert result is expected


def test_get_multiple_keys_all_exist_returns_list_of_2d_arrays():
    """Multiple existing keys return list of 2D arrays (multi=True)."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.get([1, 2, 3])

    expected = [
        np.array([[178, 204, 60, 240]], dtype=np.uint8),
        np.array([[100, 150, 200, 250]], dtype=np.uint8),
        np.array([[1, 2, 3, 4]], dtype=np.uint8),
    ]
    expected_length = 3

    assert isinstance(result, list)
    assert len(result) == expected_length
    assert all(isinstance(r, np.ndarray) and r.ndim == 2 for r in result)
    assert_array_equal(result[0], expected[0])
    assert_array_equal(result[1], expected[1])
    assert_array_equal(result[2], expected[2])


def test_get_multiple_keys_mixed_returns_list_with_none_multi():
    """Multiple keys with some missing return list with None (multi=True)."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))  # Add second vector to key 1
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.get([1, 2, 3])

    expected = [
        np.array([[178, 204, 60, 240], [178, 204, 60, 240]], dtype=np.uint8),
        None,
        np.array([[1, 2, 3, 4]], dtype=np.uint8),
    ]
    expected_length = 3
    expected_shape_key_1 = (2, 4)  # Two vectors for key 1
    expected_shape_key_3 = (1, 4)  # One vector for key 3

    assert isinstance(result, list)
    assert len(result) == expected_length
    assert isinstance(result[0], np.ndarray) and result[0].ndim == 2
    assert result[1] is None
    assert isinstance(result[2], np.ndarray) and result[2].ndim == 2
    assert result[0].shape == expected_shape_key_1
    assert result[2].shape == expected_shape_key_3
    assert_array_equal(result[0], expected[0])
    assert_array_equal(result[2], expected[2])


# Tests for Index.get() with enable_key_lookups=False


def test_get_single_key_returns_none_when_key_lookups_disabled():
    """
    When enable_key_lookups=False, Index.get() returns None for all keys.

    This is expected behavior because disabling key lookups optimizes for
    lower RAM consumption by not storing the reverse mapping from keys to vectors.
    Without this mapping, the index cannot retrieve vectors by key, so get()
    returns None even for keys that were successfully added to the index.

    Important: When enable_key_lookups=False, there is NO way to distinguish
    between a missing key and an existing key - both return None.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, enable_key_lookups=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    # Both existing and missing keys return None
    result_exists = idx.get(1)
    result_missing = idx.get(999)

    expected = None

    assert result_exists is expected
    assert result_missing is expected
