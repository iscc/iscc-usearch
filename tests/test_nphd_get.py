"""
Test NphdIndex/ShardedNphdIndex.get() method with variable-length vectors.

Verifies that get() returns properly unpadded vectors with correct lengths.
"""

import numpy as np
from numpy.testing import assert_array_equal

from iscc_usearch.nphd import NphdIndex


# Tests for get() with multi=False (single vector per key)


def test_get_single_key_exists_returns_unpadded_1d(nphd_index_factory):
    """Single key that exists returns 1D unpadded array."""
    idx = nphd_index_factory(max_dim=256)
    vector = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vector)

    result = idx.get(1)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == 4  # Original length preserved
    assert_array_equal(result, vector)


def test_get_single_key_missing_returns_none(nphd_index_factory):
    """Single key that doesn't exist returns None."""
    idx = nphd_index_factory(max_dim=256)

    result = idx.get(999)

    assert result is None


def test_get_multiple_keys_all_exist_returns_list_of_unpadded(nphd_index_factory):
    """Multiple existing keys return list of 1D unpadded arrays."""
    idx = nphd_index_factory(max_dim=256)
    v1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    v2 = np.array([100, 150, 200], dtype=np.uint8)  # Different length
    v3 = np.array([1, 2, 3, 4, 5], dtype=np.uint8)  # Different length

    idx.add(1, v1)
    idx.add(2, v2)
    idx.add(3, v3)

    result = idx.get([1, 2, 3])

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(r, np.ndarray) and r.ndim == 1 for r in result)
    assert len(result[0]) == 4
    assert len(result[1]) == 3
    assert len(result[2]) == 5
    assert_array_equal(result[0], v1)
    assert_array_equal(result[1], v2)
    assert_array_equal(result[2], v3)


def test_get_multiple_keys_mixed_returns_list_with_none(nphd_index_factory):
    """Multiple keys with some missing return list with None values."""
    idx = nphd_index_factory(max_dim=256)
    v1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    v3 = np.array([1, 2, 3, 4, 5], dtype=np.uint8)

    idx.add(1, v1)
    idx.add(3, v3)

    result = idx.get([1, 2, 3])

    assert isinstance(result, list)
    assert len(result) == 3
    assert isinstance(result[0], np.ndarray)
    assert result[1] is None
    assert isinstance(result[2], np.ndarray)
    assert_array_equal(result[0], v1)
    assert_array_equal(result[2], v3)


def test_get_multiple_keys_all_missing_returns_list_of_none(nphd_index_factory):
    """Multiple non-existing keys return list of None values."""
    idx = nphd_index_factory(max_dim=256)

    result = idx.get([10, 20, 30])

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(r is None for r in result)


# Tests for NphdIndex.get() with multi=True (multiple vectors per key)
# Note: multi=True tests are NphdIndex-specific and not parametrized


def test_get_single_key_one_vector_multi_returns_list():
    """Single key with one vector returns list with one unpadded array (multi=True)."""
    idx = NphdIndex(max_dim=256, multi=True)
    vector = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, vector)

    result = idx.get(1)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], np.ndarray)
    assert result[0].ndim == 1
    assert len(result[0]) == 4
    assert_array_equal(result[0], vector)


def test_get_single_key_multiple_vectors_multi_returns_list():
    """Single key with multiple vectors returns list of unpadded arrays (multi=True)."""
    idx = NphdIndex(max_dim=256, multi=True)
    v1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    v2 = np.array([100, 150, 200], dtype=np.uint8)  # Different length
    v3 = np.array([1, 2, 3, 4, 5], dtype=np.uint8)  # Different length

    idx.add(1, v1)
    idx.add(1, v2)
    idx.add(1, v3)

    result = idx.get(1)

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(r, np.ndarray) and r.ndim == 1 for r in result)
    assert len(result[0]) == 4
    assert len(result[1]) == 3
    assert len(result[2]) == 5
    assert_array_equal(result[0], v1)
    assert_array_equal(result[1], v2)
    assert_array_equal(result[2], v3)


def test_get_single_key_missing_multi_returns_none():
    """Single key that doesn't exist returns None (multi=True)."""
    idx = NphdIndex(max_dim=256, multi=True)

    result = idx.get(999)

    assert result is None


def test_get_multiple_keys_all_exist_multi_returns_list_of_lists():
    """Multiple existing keys return list of lists (multi=True)."""
    idx = NphdIndex(max_dim=256, multi=True)
    v1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    v2 = np.array([100, 150, 200], dtype=np.uint8)
    v3 = np.array([1, 2, 3, 4, 5], dtype=np.uint8)

    idx.add(1, v1)
    idx.add(2, v2)
    idx.add(3, v3)

    result = idx.get([1, 2, 3])

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(r, list) for r in result)
    assert len(result[0]) == 1  # One vector for key 1
    assert len(result[1]) == 1  # One vector for key 2
    assert len(result[2]) == 1  # One vector for key 3
    assert_array_equal(result[0][0], v1)
    assert_array_equal(result[1][0], v2)
    assert_array_equal(result[2][0], v3)


def test_get_multiple_keys_mixed_multi_returns_list_with_none():
    """Multiple keys with some missing return list with None (multi=True)."""
    idx = NphdIndex(max_dim=256, multi=True)
    v1a = np.array([178, 204, 60, 240], dtype=np.uint8)
    v1b = np.array([100, 150], dtype=np.uint8)
    v3 = np.array([1, 2, 3, 4, 5], dtype=np.uint8)

    idx.add(1, v1a)
    idx.add(1, v1b)
    idx.add(3, v3)

    result = idx.get([1, 2, 3])

    assert isinstance(result, list)
    assert len(result) == 3
    assert isinstance(result[0], list)
    assert result[1] is None
    assert isinstance(result[2], list)
    assert len(result[0]) == 2  # Two vectors for key 1
    assert len(result[2]) == 1  # One vector for key 3
    assert_array_equal(result[0][0], v1a)
    assert_array_equal(result[0][1], v1b)
    assert_array_equal(result[2][0], v3)
