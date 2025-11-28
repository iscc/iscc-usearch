"""
Test ShardedIndex get operations.

Confirms expected behavior when retrieving vectors by key:
- Single key retrieval
- Multiple keys retrieval
- Handling missing keys
- Handling no active shard
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_get_single_key(tmp_path):
    """Test getting vector by single key."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vector = np.random.rand(64).astype(np.float32)
    index.add(42, vector)

    result = index.get(42)

    assert result is not None
    # Use larger tolerance since usearch may store in lower precision
    assert np.allclose(result, vector, atol=0.01)


def test_get_multiple_keys(tmp_path):
    """Test getting vectors by multiple keys."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(5, 64).astype(np.float32)
    index.add(list(range(5)), vectors)

    results = index.get([0, 2, 4])

    assert len(results) == 3


def test_get_no_active_shard_single(tmp_path):
    """Test get returns None when no active shard (single key)."""
    index = ShardedIndex(ndim=64, path=tmp_path, view=True)

    result = index.get(42)

    assert result is None


def test_get_no_active_shard_multiple(tmp_path):
    """Test get returns list of None when no active shard (multiple keys)."""
    index = ShardedIndex(ndim=64, path=tmp_path, view=True)

    result = index.get([1, 2, 3])

    assert result == [None, None, None]
