"""Test NphdIndex.add() with variable-length binary vectors."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from iscc_usearch.nphd import NphdIndex


def test_add_single_vector_returns_key():
    """Adding a single variable-length vector returns the key."""
    idx = NphdIndex(max_dim=256)

    vector = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.add(1, vector)

    expected = np.array([1], dtype=np.uint64)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint64
    assert result.shape == (1,)
    assert_array_equal(result, expected)
    assert len(idx) == 1


def test_add_single_vector_auto_key():
    """Adding with key=None auto-generates key starting from 0."""
    idx = NphdIndex(max_dim=256)

    vector = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.add(None, vector)

    expected = np.array([0], dtype=np.uint64)

    assert_array_equal(result, expected)
    assert len(idx) == 1


def test_add_batch_uniform_vectors():
    """Adding 2D array of uniform-length vectors."""
    idx = NphdIndex(max_dim=256)

    keys = [1, 2, 3]
    vectors = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
            [1, 2, 3, 4],
        ],
        dtype=np.uint8,
    )

    result = idx.add(keys, vectors)

    expected = np.array([1, 2, 3], dtype=np.uint64)

    assert_array_equal(result, expected)
    assert len(idx) == 3


def test_add_batch_variable_vectors():
    """Adding list of variable-length vectors."""
    idx = NphdIndex(max_dim=256)

    keys = [1, 2, 3]
    vectors = [
        np.array([178, 204], dtype=np.uint8),  # 2 bytes
        np.array([100, 150, 200, 250], dtype=np.uint8),  # 4 bytes
        np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8),  # 6 bytes
    ]

    result = idx.add(keys, vectors)

    expected = np.array([1, 2, 3], dtype=np.uint64)

    assert_array_equal(result, expected)
    assert len(idx) == 3


def test_add_batch_auto_keys():
    """Batch add with key=None generates sequential keys."""
    idx = NphdIndex(max_dim=256)

    vectors = [
        np.array([178, 204], dtype=np.uint8),
        np.array([100, 150, 200], dtype=np.uint8),
        np.array([1, 2, 3, 4], dtype=np.uint8),
    ]

    result = idx.add(None, vectors)

    expected = np.array([0, 1, 2], dtype=np.uint64)

    assert_array_equal(result, expected)
    assert len(idx) == 3


def test_add_duplicate_key_raises_error():
    """Adding to same key twice raises RuntimeError."""
    idx = NphdIndex(max_dim=256)

    vector1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vector2 = np.array([100, 150, 200, 250], dtype=np.uint8)

    idx.add(1, vector1)

    with pytest.raises(RuntimeError, match="Duplicate keys not allowed"):
        idx.add(1, vector2)


def test_add_multiple_keys_returns_respective_keys():
    """Adding vectors to different keys returns their keys."""
    idx = NphdIndex(max_dim=256)

    result1 = idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    result2 = idx.add(2, np.array([100, 150, 200], dtype=np.uint8))
    result3 = idx.add(3, np.array([1, 2], dtype=np.uint8))

    expected1 = np.array([1], dtype=np.uint64)
    expected2 = np.array([2], dtype=np.uint64)
    expected3 = np.array([3], dtype=np.uint64)

    assert_array_equal(result1, expected1)
    assert_array_equal(result2, expected2)
    assert_array_equal(result3, expected3)
    assert len(idx) == 3


def test_add_mixed_auto_and_explicit_keys():
    """Mixing auto-generated and explicit keys."""
    idx = NphdIndex(max_dim=256)

    # Auto-key should be 0
    result1 = idx.add(None, np.array([1, 1, 1, 1], dtype=np.uint8))
    expected1 = np.array([0], dtype=np.uint64)
    assert_array_equal(result1, expected1)

    # Explicit key 100
    idx.add(100, np.array([2, 2, 2, 2], dtype=np.uint8))
    assert len(idx) == 2

    # Auto-key should be 2 (current size)
    result2 = idx.add(None, np.array([3, 3, 3, 3], dtype=np.uint8))
    expected2 = np.array([2], dtype=np.uint64)
    assert_array_equal(result2, expected2)


def test_add_batch_uniform_vectors_with_auto_keys():
    """Batch add of uniform 2D array with auto-generated keys."""
    idx = NphdIndex(max_dim=256)

    vectors = np.array(
        [
            [10, 10, 10, 10],
            [11, 11, 11, 11],
            [12, 12, 12, 12],
        ],
        dtype=np.uint8,
    )

    result = idx.add(None, vectors)

    expected = np.array([0, 1, 2], dtype=np.uint64)

    assert_array_equal(result, expected)
    assert len(idx) == 3


def test_add_single_byte_vectors():
    """Adding single-byte vectors."""
    idx = NphdIndex(max_dim=64)

    keys = [1, 2, 3]
    vectors = [
        np.array([255], dtype=np.uint8),
        np.array([128], dtype=np.uint8),
        np.array([0], dtype=np.uint8),
    ]

    result = idx.add(keys, vectors)

    expected = np.array([1, 2, 3], dtype=np.uint64)

    assert_array_equal(result, expected)
    assert len(idx) == 3


def test_add_max_length_vectors():
    """Adding vectors at maximum allowed length."""
    idx = NphdIndex(max_dim=256)

    # Maximum is 32 bytes (256 bits / 8)
    vector = np.array([i for i in range(32)], dtype=np.uint8)
    result = idx.add(1, vector)

    expected = np.array([1], dtype=np.uint64)

    assert_array_equal(result, expected)
    assert len(idx) == 1


def test_add_empty_then_add_more():
    """Adding to an initially empty index then adding more vectors."""
    idx = NphdIndex(max_dim=128)

    # Start with empty index
    assert len(idx) == 0

    # Add first vector
    idx.add(1, np.array([1, 2, 3], dtype=np.uint8))
    assert len(idx) == 1

    # Add more vectors
    idx.add(2, np.array([4, 5, 6], dtype=np.uint8))
    idx.add(3, np.array([7, 8, 9], dtype=np.uint8))
    assert len(idx) == 3


def test_add_returns_numpy_array():
    """Verify add() always returns np.ndarray, not list or int."""
    idx = NphdIndex(max_dim=256)

    # Single key
    result1 = idx.add(1, np.array([1, 2, 3], dtype=np.uint8))
    assert isinstance(result1, np.ndarray)
    assert result1.dtype == np.uint64

    # Batch keys - use high keys to avoid collision with auto-generated keys
    result2 = idx.add([10, 20], [np.array([4, 5], dtype=np.uint8), np.array([6, 7], dtype=np.uint8)])
    assert isinstance(result2, np.ndarray)
    assert result2.dtype == np.uint64

    # Auto key - should be 3 (current size)
    result3 = idx.add(None, np.array([8, 9], dtype=np.uint8))
    assert isinstance(result3, np.ndarray)
    assert result3.dtype == np.uint64
