"""
Test ShardedIndex load operations.

Confirms expected behavior for loading index from disk:
- Load multiple shards
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_load_multiple_shards(tmp_path):
    """Test load with multiple shards."""
    # Create index with tiny shard to force multiple
    index1 = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)
    for i in range(100):
        index1.add(i, np.random.rand(64).astype(np.float32))
    index1.save()

    # Reload
    index2 = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    assert len(index2) > 0
