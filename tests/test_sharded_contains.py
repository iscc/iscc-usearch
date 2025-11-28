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
