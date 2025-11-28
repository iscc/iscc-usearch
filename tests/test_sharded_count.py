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
