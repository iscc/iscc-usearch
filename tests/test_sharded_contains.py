"""
Test ShardedIndex contains operations.

Confirms expected behavior for checking key membership:
- Single key contains check
- Multiple keys contains check
- 'in' operator support
- Handling no active shard
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_contains_single_key(tmp_path):
    """Test contains with single key."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(42, np.random.rand(64).astype(np.float32))

    assert index.contains(42) is True
    assert index.contains(999) is False


def test_contains_multiple_keys(tmp_path):
    """Test contains with multiple keys."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add([1, 2, 3], np.random.rand(3, 64).astype(np.float32))

    result = index.contains([1, 2, 999])

    assert result[0]
    assert result[1]
    assert not result[2]


def test_contains_no_active_shard_single(tmp_path):
    """Test contains returns False when no active shard (single key)."""
    index = ShardedIndex(ndim=64, path=tmp_path, view=True)

    assert index.contains(42) is False


def test_contains_no_active_shard_multiple(tmp_path):
    """Test contains returns array of False when no active shard."""
    index = ShardedIndex(ndim=64, path=tmp_path, view=True)

    result = index.contains([1, 2, 3])

    assert not np.any(result)


def test_in_operator(tmp_path):
    """Test 'in' operator works."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(42, np.random.rand(64).astype(np.float32))

    assert 42 in index
    assert 999 not in index


def test_contains_across_shards_single(tmp_path):
    """Test contains finds keys across multiple shards (single key)."""
    # Create index with small shard size to force rotation
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.add(2, np.random.rand(64).astype(np.float32))
    index.save()

    # Reload to have view shards + active shard
    index2 = ShardedIndex(ndim=64, path=tmp_path)
    assert index2.shard_count >= 2

    # Key 1 should be in view shard, key 2 might be in active or view
    assert index2.contains(1) is True
    assert index2.contains(2) is True
    assert index2.contains(999) is False


def test_contains_across_shards_batch(tmp_path):
    """Test contains finds keys across multiple shards (batch)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.add(2, np.random.rand(64).astype(np.float32))
    index.add(3, np.random.rand(64).astype(np.float32))
    index.save()

    index2 = ShardedIndex(ndim=64, path=tmp_path)

    result = index2.contains([1, 2, 3, 999])

    assert result[0]  # key 1
    assert result[1]  # key 2
    assert result[2]  # key 3
    assert not result[3]  # key 999 doesn't exist


def test_contains_view_mode_across_shards(tmp_path):
    """Test contains works in view mode across shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.add(2, np.random.rand(64).astype(np.float32))
    index.save()

    # Open in view mode (read-only)
    index2 = ShardedIndex(ndim=64, path=tmp_path, view=True)
    assert index2._active_shard is None

    assert index2.contains(1) is True
    assert index2.contains(2) is True
    assert index2.contains(999) is False


def test_contains_enable_key_lookups_false(tmp_path):
    """Test contains returns False for all keys when enable_key_lookups=False."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.add(2, np.random.rand(64).astype(np.float32))
    index.save()

    # Reload with enable_key_lookups=False
    index2 = ShardedIndex(ndim=64, path=tmp_path, enable_key_lookups=False)

    # All contains() calls should return False (matches usearch behavior)
    assert index2.contains(1) is False
    assert index2.contains(2) is False
    assert index2.contains(999) is False

    # Batch version should return array of False
    result = index2.contains([1, 2, 999])
    assert not np.any(result)


def test_contains_empty_keys_array(tmp_path):
    """Test contains with empty keys array returns empty array."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))

    result = index.contains([])

    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_contains_early_exit_all_keys_found(tmp_path):
    """Test contains early exit when all keys found before processing all shards."""
    # type: () -> None
    # shard_size is in bytes. Use 1 byte to force rotation after each vector add.
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)

    # Add 3 vectors to create multiple shards (one per shard due to tiny shard_size)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.add(2, np.random.rand(64).astype(np.float32))
    index.add(3, np.random.rand(64).astype(np.float32))
    index.save()

    # Verify we have multiple view shards
    assert len(index._viewed_indexes) >= 2

    # Request only key 1 which is in the first view shard.
    # After processing first view shard, result.all() = True.
    # The break statement skips remaining view shards.
    result = index.contains([1])

    assert result[0]
