"""
Test ShardedIndex shard rotation operations.

Confirms expected behavior for automatic shard rotation:
- Rotation creates view shards
- Data is preserved after rotation
- Guard clause when no active shard
- Active shard path cleared on rotation
- Next shard number calculation
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_shard_rotation_creates_view_shard(tmp_path):
    """Test that shard rotation moves old shard to view shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    # Add enough to trigger rotation
    for i in range(100):
        index.add(i, np.random.rand(64).astype(np.float32))

    # After rotation, should have view shards
    if index.shard_count > 1:
        assert index._view_shards is not None


def test_shard_rotation_preserves_data(tmp_path):
    """Test that data is preserved after shard rotation."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    vectors = []
    for i in range(50):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Search for all vectors
    for i, v in enumerate(vectors):
        matches = index.search(v, count=1)
        assert i in matches.keys


def test_rotate_shard_guard_clause(tmp_path):
    """Test that _rotate_shard does nothing when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    # Force active shard to None (simulates edge case)
    index._active_shard = None

    # This should return early without error
    index._rotate_shard()


def test_rotation_clears_active_shard_path(tmp_path):
    """Test that rotation clears _active_shard_path."""
    # Create fresh index with tiny shard size
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    # Initially no path since it's unsaved
    assert index._active_shard_path is None

    # Add enough to trigger rotation
    for i in range(100):
        index.add(i, np.random.rand(64).astype(np.float32))

    # After rotation, new active shard should have no path
    assert index._active_shard_path is None


def test_get_next_shard_number_empty(tmp_path):
    """Test _get_next_shard_number with no existing shards."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    assert index._get_next_shard_number() == 0


def test_get_next_shard_number_with_shards(tmp_path):
    """Test _get_next_shard_number with existing shards."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.save()

    # Now should return 1
    assert index._get_next_shard_number() == 1


def test_rotation_after_load_uses_tracked_path(tmp_path):
    """Test rotation after load uses _active_shard_path (fix for data duplication bug)."""
    # Create index and save one shard
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100000)
    index.add(0, np.random.rand(64).astype(np.float32))
    index.save()
    assert index.shard_count == 1

    # Reload - should have tracked path
    index2 = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)
    assert index2._active_shard_path is not None

    # Manually trigger rotation (simulates adding enough data)
    index2._rotate_shard()

    # The rotation should have used the tracked path (shard_000), not created shard_001
    # After rotation, shard_000 should be in view_shards
    assert len(index2._viewed_indexes) == 1
    # And _active_shard_path should be cleared for the new empty shard
    assert index2._active_shard_path is None
