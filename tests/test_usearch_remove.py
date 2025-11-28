"""
Confirm the expected behavior of usearch Index.remove() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- multi=False (single vector per key)
- Removing existing vs non-existent keys
- Batch removal operations
"""

import numpy as np
from numpy.testing import assert_array_equal
from usearch.index import Index, MetricKind, ScalarKind


# Tests for Index.remove() with single keys


def test_remove_single_existing_key_returns_count():
    """Removing an existing key returns the count of removed vectors."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.remove(1)

    # For multi=False, should return 1 (one vector removed)
    expected = 1
    assert result == expected

    # Verify key was actually removed
    assert not idx.contains(1)
    assert len(idx) == 0


def test_remove_single_missing_key_returns_zero():
    """Removing a non-existent key returns 0 without raising error."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.remove(999)

    # Should return 0 for non-existent key (no error)
    expected = 0
    assert result == expected

    # Verify original key still exists
    assert idx.contains(1)
    assert len(idx) == 1


def test_remove_single_key_from_empty_index_returns_zero():
    """Removing from empty index returns 0 without raising error."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    result = idx.remove(1)

    expected = 0
    assert result == expected
    assert len(idx) == 0


# Tests for Index.remove() with batch keys


def test_remove_batch_all_existing_keys():
    """Removing batch of existing keys returns total count or per-key counts."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.remove([1, 2, 3])

    # Result could be either:
    # - Single int total count: 3
    # - Array of per-key counts: [1, 1, 1]
    # Document actual behavior:
    if isinstance(result, (int, np.integer)):
        # Single total count
        expected_total = 3
        assert result == expected_total
    else:
        # Per-key counts
        expected_array = np.array([1, 1, 1], dtype=np.uint64)
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, expected_array)

    # Verify all keys were removed
    assert not idx.contains(1)
    assert not idx.contains(2)
    assert not idx.contains(3)
    assert len(idx) == 0


def test_remove_batch_all_missing_keys():
    """Removing batch of non-existent keys returns 0 or array of zeros."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.remove([10, 20, 30])

    # Result could be either:
    # - Single int: 0
    # - Array of zeros: [0, 0, 0]
    if isinstance(result, (int, np.integer)):
        expected = 0
        assert result == expected
    else:
        expected_array = np.array([0, 0, 0], dtype=np.uint64)
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, expected_array)

    # Verify original key still exists
    assert idx.contains(1)
    assert len(idx) == 1


def test_remove_batch_mixed_existing_and_missing_keys():
    """Removing batch with some existing and some missing keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    # Remove mix: 1 exists, 999 doesn't, 2 exists
    result = idx.remove([1, 999, 2])

    # Result could be either:
    # - Single int total: 2 (two keys removed)
    # - Array of per-key: [1, 0, 1]
    if isinstance(result, (int, np.integer)):
        expected_total = 2
        assert result == expected_total
    else:
        expected_array = np.array([1, 0, 1], dtype=np.uint64)
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, expected_array)

    # Verify only existing keys were removed
    assert not idx.contains(1)
    assert not idx.contains(2)
    assert len(idx) == 0


def test_remove_empty_batch_returns_zero_or_empty_array():
    """Removing empty batch returns 0 or empty array without error."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.remove([])

    # Result could be either:
    # - Single int: 0
    # - Empty array: []
    if isinstance(result, (int, np.integer)):
        expected = 0
        assert result == expected
    else:
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    # Verify original key still exists
    assert idx.contains(1)
    assert len(idx) == 1


# Tests for Index.remove() with multi=True


def test_remove_single_key_multi_true_returns_bool():
    """Removing key with multiple vectors (multi=True) returns bool."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.remove(1)

    # remove_one() returns bool (True if removed)
    expected = True
    assert result is expected
    assert isinstance(result, bool)

    # Verify key was completely removed
    assert not idx.contains(1)
    assert len(idx) == 0


# Tests for compact parameter


def test_remove_with_compact_false():
    """Remove with compact=False (default) marks entries deleted but doesn't compact."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    result = idx.remove(1, compact=False)

    expected = 1
    assert result == expected
    assert not idx.contains(1)

    # Index size should reflect remaining key
    assert len(idx) == 1


def test_remove_with_compact_true():
    """Remove with compact=True removes entries and compacts the index."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    result = idx.remove(1, compact=True)

    expected = 1
    assert result == expected
    assert not idx.contains(1)

    # Index should be compacted
    assert len(idx) == 1


# Integration tests: remove and re-add


def test_remove_then_readd_same_key():
    """Removing a key allows re-adding with same key (for updates)."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    original_vector = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(1, original_vector)

    # Remove key
    idx.remove(1)
    assert not idx.contains(1)

    # Re-add with different vector
    new_vector = np.array([100, 150, 200, 250], dtype=np.uint8)
    idx.add(1, new_vector)

    # Verify new vector was stored
    stored = idx.get(1)
    assert_array_equal(stored, new_vector)


def test_remove_batch_then_readd_for_update_pattern():
    """Pattern used in add_assets: remove then re-add for updates."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    # Initial add
    keys = [1, 2, 3]
    vectors = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
            [1, 2, 3, 4],
        ],
        dtype=np.uint8,
    )
    idx.add(keys, vectors)
    assert len(idx) == 3

    # Update pattern: remove then re-add with new vectors
    new_vectors = np.array(
        [
            [255, 255, 255, 255],
            [128, 128, 128, 128],
            [64, 64, 64, 64],
        ],
        dtype=np.uint8,
    )

    # Remove all keys (some may not exist in real usage)
    idx.remove(keys)

    # Re-add with new vectors
    idx.add(keys, new_vectors)

    # Verify all vectors were updated
    assert_array_equal(idx.get(1), new_vectors[0])
    assert_array_equal(idx.get(2), new_vectors[1])
    assert_array_equal(idx.get(3), new_vectors[2])
    assert len(idx) == 3
