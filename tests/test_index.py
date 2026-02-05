"""Tests for iscc_usearch.Index wrapper class (get, search, upsert methods)."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from usearch.index import MetricKind, ScalarKind

from iscc_usearch.index import Index


def create_index():
    """Create test index with 32-bit Hamming vectors."""
    return Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)


# Tests for upsert()


def test_upsert_keys_none_raises():
    """upsert() raises ValueError when keys=None."""
    idx = create_index()
    with pytest.raises(ValueError, match="explicit keys"):
        idx.upsert(None, np.array([1, 2, 3, 4], dtype=np.uint8))


def test_upsert_mismatched_lengths_raises():
    """upsert() raises ValueError when keys/vectors lengths differ."""
    idx = create_index()
    with pytest.raises(ValueError, match="must match"):
        idx.upsert([1], np.zeros((2, 4), dtype=np.uint8))


def test_upsert_empty_batch():
    """upsert() with empty batch returns empty array."""
    idx = create_index()
    result = idx.upsert(
        np.array([], dtype=np.uint64),
        np.array([], dtype=np.uint8).reshape(0, 4),
    )
    assert_array_equal(result, np.array([], dtype=np.uint64))
    assert len(idx) == 0


def test_upsert_single_new():
    """upsert() adds new single key."""
    idx = create_index()
    vec = np.array([1, 2, 3, 4], dtype=np.uint8)

    result = idx.upsert(1, vec)

    assert_array_equal(result, np.array([1], dtype=np.uint64))
    assert len(idx) == 1
    assert_array_equal(idx.get(1), vec)


def test_upsert_single_same_vector_noop():
    """upsert() with same vector does not modify index."""
    idx = create_index()
    vec = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.upsert(1, vec)

    result = idx.upsert(1, vec)  # same vector

    assert_array_equal(result, np.array([1], dtype=np.uint64))
    assert len(idx) == 1
    assert_array_equal(idx.get(1), vec)


def test_upsert_single_different_vector_updates():
    """upsert() with different vector updates it."""
    idx = create_index()
    vec1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    vec2 = np.array([5, 6, 7, 8], dtype=np.uint8)

    idx.upsert(1, vec1)
    result = idx.upsert(1, vec2)

    assert_array_equal(result, np.array([1], dtype=np.uint64))
    assert len(idx) == 1
    assert_array_equal(idx.get(1), vec2)


def test_upsert_batch_all_new():
    """upsert() batch with all new keys adds all."""
    idx = create_index()
    keys = np.array([1, 2, 3], dtype=np.uint64)
    vectors = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.uint8)

    result = idx.upsert(keys, vectors)

    assert_array_equal(result, keys)
    assert len(idx) == 3
    assert_array_equal(idx.get(1), vectors[0])
    assert_array_equal(idx.get(2), vectors[1])
    assert_array_equal(idx.get(3), vectors[2])


def test_upsert_batch_mixed():
    """upsert() handles mix of new, same, and different vectors."""
    idx = create_index()

    # Initial state: keys 1 and 3
    idx.upsert(
        np.array([1, 3], dtype=np.uint64),
        np.array([[1, 1, 1, 1], [3, 3, 3, 3]], dtype=np.uint8),
    )

    # Upsert: key 1 (update), key 2 (new), key 3 (same), key 4 (new)
    keys = np.array([1, 2, 3, 4], dtype=np.uint64)
    vectors = np.array(
        [
            [9, 9, 9, 9],  # key 1: different -> update
            [2, 2, 2, 2],  # key 2: new
            [3, 3, 3, 3],  # key 3: same -> no-op
            [4, 4, 4, 4],  # key 4: new
        ],
        dtype=np.uint8,
    )

    result = idx.upsert(keys, vectors)

    assert_array_equal(result, keys)
    assert len(idx) == 4
    assert_array_equal(idx.get(1), np.array([9, 9, 9, 9], dtype=np.uint8))
    assert_array_equal(idx.get(2), np.array([2, 2, 2, 2], dtype=np.uint8))
    assert_array_equal(idx.get(3), np.array([3, 3, 3, 3], dtype=np.uint8))
    assert_array_equal(idx.get(4), np.array([4, 4, 4, 4], dtype=np.uint8))


def test_upsert_internal_duplicates_keeps_last():
    """upsert() with internal duplicates keeps last occurrence."""
    idx = create_index()

    # Key 1 appears at index 0 and 2 - should use index 2's vector
    keys = np.array([1, 2, 1, 3], dtype=np.uint64)
    vectors = np.array(
        [
            [1, 1, 1, 1],  # key 1 first - ignored
            [2, 2, 2, 2],  # key 2
            [9, 9, 9, 9],  # key 1 last - used
            [3, 3, 3, 3],  # key 3
        ],
        dtype=np.uint8,
    )

    result = idx.upsert(keys, vectors)

    assert_array_equal(result, keys)
    assert len(idx) == 3
    assert_array_equal(idx.get(1), np.array([9, 9, 9, 9], dtype=np.uint8))
    assert_array_equal(idx.get(2), np.array([2, 2, 2, 2], dtype=np.uint8))
    assert_array_equal(idx.get(3), np.array([3, 3, 3, 3], dtype=np.uint8))


def test_upsert_idempotency():
    """Multiple upserts with same data produce identical state."""
    idx = create_index()
    keys = np.array([1, 2, 3], dtype=np.uint64)
    vectors = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.uint8)

    idx.upsert(keys, vectors)
    state1 = [idx.get(int(k)).tolist() for k in keys]

    idx.upsert(keys, vectors)
    state2 = [idx.get(int(k)).tolist() for k in keys]

    idx.upsert(keys, vectors)
    state3 = [idx.get(int(k)).tolist() for k in keys]

    assert state1 == state2 == state3
    assert len(idx) == 3


def test_upsert_list_keys():
    """upsert() accepts Python list as keys."""
    idx = create_index()
    result = idx.upsert(
        [1, 2, 3],
        np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.uint8),
    )
    assert_array_equal(result, np.array([1, 2, 3], dtype=np.uint64))
    assert len(idx) == 3
