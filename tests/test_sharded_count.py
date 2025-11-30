"""
Test ShardedIndex count operations.

Confirms expected behavior for counting key occurrences:
- Single key count
- Multiple keys count
- Handling no active shard
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_count_single_key(tmp_path):
    """Test count with single key."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(42, np.random.rand(64).astype(np.float32))

    assert index.count(42) == 1
    assert index.count(999) == 0


def test_count_multiple_keys(tmp_path):
    """Test count with multiple keys."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add([1, 2], np.random.rand(2, 64).astype(np.float32))

    result = index.count([1, 2, 999])

    assert result[0] == 1
    assert result[1] == 1
    assert result[2] == 0


def test_count_no_active_shard_single(tmp_path):
    """Test count returns 0 when no active shard (single key)."""
    index = ShardedIndex(ndim=64, path=tmp_path, view=True)

    assert index.count(42) == 0


def test_count_no_active_shard_multiple(tmp_path):
    """Test count returns array of zeros when no active shard."""
    index = ShardedIndex(ndim=64, path=tmp_path, view=True)

    result = index.count([1, 2, 3])

    assert np.all(result == 0)


def test_count_across_shards_single(tmp_path):
    """Test count aggregates across shards (single key)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.add(2, np.random.rand(64).astype(np.float32))
    index.save()

    # Reload to have view shards
    index2 = ShardedIndex(ndim=64, path=tmp_path)
    assert index2.shard_count >= 2

    # Both keys should have count 1
    assert index2.count(1) == 1
    assert index2.count(2) == 1
    assert index2.count(999) == 0


def test_count_across_shards_batch(tmp_path):
    """Test count aggregates across shards (batch)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.add(2, np.random.rand(64).astype(np.float32))
    index.add(3, np.random.rand(64).astype(np.float32))
    index.save()

    index2 = ShardedIndex(ndim=64, path=tmp_path)

    result = index2.count([1, 2, 3, 999])

    assert result[0] == 1
    assert result[1] == 1
    assert result[2] == 1
    assert result[3] == 0


def test_count_view_mode_across_shards(tmp_path):
    """Test count works in view mode across shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.add(2, np.random.rand(64).astype(np.float32))
    index.save()

    # Open in view mode
    index2 = ShardedIndex(ndim=64, path=tmp_path, view=True)
    assert index2._active_shard is None

    assert index2.count(1) == 1
    assert index2.count(2) == 1


def test_count_empty_keys_array(tmp_path):
    """Test count with empty keys array returns empty array."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))

    result = index.count([])

    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_count_enable_key_lookups_false_single(tmp_path):
    """Test count returns 0 for single key when enable_key_lookups=False."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.save()

    # Reload with enable_key_lookups=False
    index2 = ShardedIndex(ndim=64, path=tmp_path, enable_key_lookups=False)

    assert index2.count(1) == 0


def test_count_enable_key_lookups_false_batch(tmp_path):
    """Test count returns array of zeros when enable_key_lookups=False."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add([1, 2], np.random.rand(2, 64).astype(np.float32))
    index.save()

    # Reload with enable_key_lookups=False
    index2 = ShardedIndex(ndim=64, path=tmp_path, enable_key_lookups=False)

    result = index2.count([1, 2])
    assert np.all(result == 0)
