"""
Test ShardedIndex save operations.

Confirms expected behavior for persisting index to disk:
- Save and reload roundtrip
- Empty index handling
- No duplicate shards on repeated saves
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_save_and_load(tmp_path):
    """Test save and load roundtrip."""
    # Create and save
    index1 = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index1.add(list(range(10)), vectors)
    index1.save()

    # Load
    index2 = ShardedIndex(ndim=64, path=tmp_path)

    assert len(index2) == 10


def test_save_empty_index(tmp_path):
    """Test save does nothing for empty index."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.save()

    assert index.shard_count == 0


def test_save_no_duplicate_shards(tmp_path):
    """Test save doesn't create duplicate shards (P1 fix)."""
    # Create and save
    index1 = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index1.add(list(range(10)), vectors)
    index1.save()

    # Reopen and add more
    index2 = ShardedIndex(ndim=64, path=tmp_path)
    index2.add(100, np.random.rand(64).astype(np.float32))
    index2.save()

    # Should still be 1 shard, not 2
    assert index2.shard_count == 1
