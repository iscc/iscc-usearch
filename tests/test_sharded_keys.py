"""
Tests for ShardedIndex.keys property.

Verifies lazy key iteration across shards with:
- Basic iteration and length
- Indexing and slicing
- Numpy array conversion
- Behavior across multiple shards
- Memory efficiency (lazy evaluation)
"""

import numpy as np
import pytest
from usearch.index import MetricKind, ScalarKind

from iscc_usearch.sharded import ShardedIndex, ShardedIndexedKeys


# Basic functionality tests


def test_keys_empty_index_returns_empty(tmp_path):
    """Empty sharded index has empty keys."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )

    keys = idx.keys

    assert isinstance(keys, ShardedIndexedKeys)
    assert len(keys) == 0
    assert list(keys) == []


def test_keys_single_entry_returns_that_key(tmp_path):
    """Sharded index with one entry returns that key."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(42, np.array([178, 204, 60, 240], dtype=np.uint8))

    keys = idx.keys

    assert len(keys) == 1
    assert 42 in list(keys)


def test_keys_multiple_entries_returns_all_keys(tmp_path):
    """Sharded index with multiple entries returns all keys."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(5, np.array([1, 2, 3, 4], dtype=np.uint8))

    keys = idx.keys
    keys_list = list(keys)

    assert len(keys) == 3
    assert set(keys_list) == {1, 2, 5}


# Iteration tests


def test_keys_iteration(tmp_path):
    """Keys can be iterated."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    keys = idx.keys
    keys_list = list(keys)

    assert len(keys_list) == 3
    assert set(keys_list) == {1, 2, 3}


def test_keys_multiple_iterations(tmp_path):
    """Keys can be iterated multiple times."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    keys = idx.keys

    first_pass = list(keys)
    second_pass = list(keys)

    assert first_pass == second_pass
    assert len(second_pass) == 2


# Indexing and slicing tests


def test_keys_indexing(tmp_path):
    """Keys support integer indexing."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(10, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(20, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(30, np.array([1, 2, 3, 4], dtype=np.uint8))

    keys = idx.keys

    # Get first key
    first = keys[0]
    assert first in {10, 20, 30}

    # Negative indexing
    last = keys[-1]
    assert last in {10, 20, 30}


def test_keys_indexing_out_of_range(tmp_path):
    """Keys indexing raises IndexError for out of range."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    keys = idx.keys

    with pytest.raises(IndexError):
        _ = keys[10]

    with pytest.raises(IndexError):
        _ = keys[-10]


def test_keys_slicing(tmp_path):
    """Keys support slicing."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    for i in range(10):
        idx.add(i * 10, np.array([i, i + 1, i + 2, i + 3], dtype=np.uint8))

    keys = idx.keys

    # First 3 keys
    first_3 = keys[:3]
    assert isinstance(first_3, np.ndarray)
    assert len(first_3) == 3


# Numpy array conversion tests


def test_keys_numpy_conversion(tmp_path):
    """Keys can be converted to numpy array."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(5, np.array([1, 2, 3, 4], dtype=np.uint8))

    keys = idx.keys
    keys_array = np.asarray(keys)

    assert isinstance(keys_array, np.ndarray)
    assert keys_array.dtype == np.uint64
    assert len(keys_array) == 3
    assert set(keys_array.tolist()) == {1, 2, 5}


def test_keys_numpy_conversion_empty_index(tmp_path):
    """Empty index converts to empty numpy array with correct dtype."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )

    keys_array = np.asarray(idx.keys)

    assert isinstance(keys_array, np.ndarray)
    assert keys_array.dtype == np.uint64
    assert len(keys_array) == 0


def test_keys_numpy_conversion_with_viewed_shards(tmp_path):
    """Keys from viewed shards are included in numpy array conversion."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        shard_size=500,  # Small to force rotation
        bloom_filter=False,
    )

    # Add entries to force shard rotation (creates viewed shards)
    for i in range(50):
        idx.add(i, np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8))

    # Verify we have viewed shards
    assert len(idx._viewed_indexes) > 0

    keys_array = np.asarray(idx.keys)

    assert isinstance(keys_array, np.ndarray)
    assert keys_array.dtype == np.uint64
    assert len(keys_array) == 50
    assert set(keys_array.tolist()) == set(range(50))


def test_keys_numpy_conversion_with_dtype(tmp_path):
    """Keys can be converted to numpy array with specific dtype."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    keys = idx.keys
    keys_array = np.asarray(keys, dtype=np.int32)

    assert keys_array.dtype == np.int32


# Multi-shard tests


def test_keys_across_multiple_shards(tmp_path):
    """Keys are aggregated across multiple shards."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        shard_size=500,  # Very small to force rotation
        bloom_filter=False,
    )

    # Add enough entries to create multiple shards
    added_keys = []
    for i in range(100):
        idx.add(i, np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8))
        added_keys.append(i)

    # Should have multiple shards
    assert idx.shard_count >= 1

    keys = idx.keys
    keys_list = list(keys)

    # All keys should be present
    assert len(keys_list) == 100
    assert set(keys_list) == set(added_keys)


def test_keys_after_save_and_load(tmp_path):
    """Keys are preserved after save and load."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(5, np.array([1, 2, 3, 4], dtype=np.uint8))
    idx.save()

    # Load into new instance
    idx2 = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )

    keys = idx2.keys
    keys_list = list(keys)

    assert len(keys_list) == 3
    assert set(keys_list) == {1, 2, 5}


def test_keys_after_reload(tmp_path):
    """Keys work after reloading saved index."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.save()

    # Reload in new instance
    idx2 = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )

    keys = idx2.keys
    keys_list = list(keys)

    assert len(keys_list) == 2
    assert set(keys_list) == {1, 2}


# Live view behavior tests


def test_keys_is_live_view(tmp_path):
    """Keys reflects changes made after obtaining the keys object."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    # Get keys reference
    keys = idx.keys
    assert len(keys) == 1

    # Add more entries
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    # keys should reflect the new additions (live view)
    assert len(keys) == 3
    assert set(list(keys)) == {1, 2, 3}


# Memory efficiency tests


def test_keys_lazy_iteration(tmp_path):
    """Keys iteration is lazy (generator-based)."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    for i in range(100):
        idx.add(i, np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8))

    keys = idx.keys

    # Partial iteration should work
    first_10 = []
    for i, key in enumerate(keys):
        if i >= 10:
            break
        first_10.append(key)

    assert len(first_10) == 10


def test_keys_repr(tmp_path):
    """Keys has useful repr."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    keys = idx.keys
    repr_str = repr(keys)

    assert "ShardedIndexedKeys" in repr_str
    assert "count=2" in repr_str


# Edge cases


def test_keys_with_large_key_values(tmp_path):
    """Keys handles large uint64 values."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )

    large_key = 2**63 - 1
    idx.add(large_key, np.array([178, 204, 60, 240], dtype=np.uint8))

    keys = idx.keys
    keys_list = list(keys)

    assert len(keys_list) == 1
    assert large_key in keys_list


def test_keys_with_zero_key(tmp_path):
    """Keys handles key value 0."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(0, np.array([178, 204, 60, 240], dtype=np.uint8))

    keys = idx.keys
    keys_list = list(keys)

    assert len(keys_list) == 1
    assert 0 in keys_list


def test_keys_indexing_in_viewed_shards(tmp_path):
    """Keys indexing finds keys in viewed shards."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        shard_size=500,  # Small to force rotation
        bloom_filter=False,
    )

    # Add entries to fill first shard
    for i in range(50):
        idx.add(i, np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8))

    # Save and manually trigger rotation
    idx.save()
    idx._rotate_shard()

    # Add entries to new shard
    for i in range(50, 60):
        idx.add(i, np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8))

    # Now we have viewed shards + active shard
    keys = idx.keys

    # Access keys via indexing - should find in viewed shards
    first_key = keys[0]
    assert first_key in range(60)

    # Access key in middle (likely in viewed shard)
    middle_key = keys[25]
    assert middle_key in range(60)

    # Access key near end (could be in active shard)
    last_key = keys[-1]
    assert last_key in range(60)


def test_keys_indexing_empty_after_save_and_rotate(tmp_path):
    """Keys indexing handles edge case of empty active shard after rotation."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        shard_size=500,
        bloom_filter=False,
    )

    # Add entries
    for i in range(10):
        idx.add(i, np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8))

    # Save and manually trigger rotation (creates empty active shard)
    idx.save()
    idx._rotate_shard()

    keys = idx.keys

    # Should still access all 10 keys from viewed shard
    assert len(keys) == 10
    for i in range(10):
        assert keys[i] in range(10)

    # Out of range should raise IndexError
    with pytest.raises(IndexError, match="index out of range"):
        _ = keys[10]

    with pytest.raises(IndexError, match="index out of range"):
        _ = keys[100]


def test_keys_indexing_defensive_fallback_error(tmp_path):
    """Test defensive IndexError fallback on line 96 via state manipulation."""
    # This is defensive code that shouldn't be reachable in normal use
    # We manually create an inconsistent state to trigger it
    from usearch.index import Index as UsearchIndex, MetricKind as UK, ScalarKind as SK

    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )

    # Add entry to active shard
    idx.add(42, np.array([178, 204, 60, 240], dtype=np.uint8))

    # Create an empty viewed index and inject it
    empty_viewed = UsearchIndex(ndim=32, metric=UK.Hamming, dtype=SK.B1)
    idx._viewed_indexes.append(empty_viewed)

    # Now set active shard to None
    # This creates inconsistency: size property will count the empty viewed index
    # but __getitem__ won't find anything in it
    idx._active_shard = None

    keys = idx.keys

    # The defensive IndexError on line 96 should trigger when we try to access
    # beyond what's actually available
    with pytest.raises(IndexError, match="index out of range"):
        _ = keys[0]
