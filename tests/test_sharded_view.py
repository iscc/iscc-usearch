"""
Test ShardedIndex view operations.

Confirms expected behavior for read-only view mode:
- View empty directory
- View existing shards
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_view_empty_directory(tmp_path):
    """Test view on empty directory."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.view()

    assert index._view_mode is True
    assert index._active_shard is None
    assert index._view_shards is None


def test_view_existing_shards(tmp_path):
    """Test view on directory with shards."""
    # Create and save
    index1 = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index1.add(list(range(10)), vectors)
    index1.save()

    # View
    index2 = ShardedIndex(ndim=64, path=tmp_path)
    index2.view()

    assert index2._view_mode is True
    assert len(index2) == 10
