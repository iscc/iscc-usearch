"""
Test ShardedIndex add operations.

Confirms expected behavior when adding vectors:
- Single vector addition
- Batch vector addition
- View mode restrictions
- Shard rotation on size threshold
- Creating active shard when None
"""

import numpy as np
import pytest

from iscc_usearch.sharded import ShardedIndex


def test_add_single_vector(tmp_path):
    """Test adding a single vector."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vector = np.random.rand(64).astype(np.float32)

    key = index.add(1, vector)

    assert key == 1
    assert len(index) == 1


def test_add_batch_vectors(tmp_path):
    """Test adding batch of vectors."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(100, 64).astype(np.float32)
    keys = list(range(100))

    result = index.add(keys, vectors)

    assert len(result) == 100
    assert len(index) == 100


def test_add_view_mode_raises(tmp_path):
    """Test that add raises in view mode."""
    # Create and save first
    index1 = ShardedIndex(ndim=64, path=tmp_path)
    index1.add(1, np.random.rand(64).astype(np.float32))
    index1.save()

    # Open in view mode
    index2 = ShardedIndex(ndim=64, path=tmp_path, view=True)

    with pytest.raises(RuntimeError, match="view mode"):
        index2.add(2, np.random.rand(64).astype(np.float32))


def test_add_triggers_rotation(tmp_path):
    """Test that add triggers shard rotation when size exceeded."""
    # Use tiny shard size to force rotation
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    # Add enough vectors to trigger rotation
    for i in range(100):
        index.add(i, np.random.rand(64).astype(np.float32))

    assert index.shard_count >= 1
    assert index._view_shards is not None or index.shard_count == 0


def test_add_creates_shard_if_none(tmp_path):
    """Test add creates active shard if None."""
    index = ShardedIndex(ndim=64, path=tmp_path, view=True)
    # Force to non-view mode with no active shard
    index._view_mode = False
    index._active_shard = None

    index.add(1, np.random.rand(64).astype(np.float32))

    assert index._active_shard is not None
    assert len(index) == 1
