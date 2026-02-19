"""
Test ShardedIndex remove operations.

Confirms expected behavior for removing vectors by key:
- Remove from active shard (no tombstone)
- Remove from view shard (tombstoned)
- Batch remove across shards
- Tombstone persistence across save/load
- Read operations respect tombstones
"""

import numpy as np
import pytest

from iscc_usearch.sharded import ShardedIndex


def test_remove_single_from_active_shard(tmp_path):
    """Remove a key from the active shard uses USearch remove (no tombstone)."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vec = np.random.rand(64).astype(np.float32)
    index.add(1, vec)

    index.remove(1)

    assert not index.contains(1)
    assert index.get(1) is None
    assert index.tombstone_count == 0  # no tombstone for active-only key


def test_remove_single_from_view_shard(tmp_path):
    """Remove a key that only exists in a view shard creates tombstone."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vec = np.random.rand(64).astype(np.float32)
    index.add(1, vec)
    index.add(2, np.random.rand(64).astype(np.float32))  # force rotation
    index.save()

    # Reload to get view shards
    index2 = ShardedIndex(ndim=64, path=tmp_path)
    assert len(index2._viewed_indexes) >= 1

    index2.remove(1)

    assert not index2.contains(1)
    assert index2.get(1) is None
    assert index2.tombstone_count == 1


def test_remove_batch_mixed(tmp_path):
    """Batch remove across active + view shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)

    # Add keys that go into view shards
    for i in range(3):
        index.add(i, np.random.rand(64).astype(np.float32))

    # Key 3 stays in active shard
    index.add(3, np.random.rand(64).astype(np.float32))

    # Remove mix of view + active keys
    index.remove([0, 1, 3])

    assert not index.contains(0)
    assert not index.contains(1)
    assert index.contains(2)
    assert not index.contains(3)


def test_remove_nonexistent_key(tmp_path):
    """Removing a nonexistent key is a no-op."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))

    index.remove(999)  # should not raise

    assert index.contains(1)
    assert index.tombstone_count == 0


def test_remove_read_only_raises(tmp_path):
    """Remove raises RuntimeError on read-only index."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.save()

    ro_index = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
    with pytest.raises(RuntimeError, match="read-only"):
        ro_index.remove(1)


def test_remove_multi_raises(tmp_path):
    """Remove raises ValueError when multi=True."""
    index = ShardedIndex(ndim=64, path=tmp_path, multi=True)
    index.add(1, np.random.rand(64).astype(np.float32))

    with pytest.raises(ValueError, match="multi=False"):
        index.remove(1)


def test_delitem(tmp_path):
    """del index[key] delegates to remove()."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))

    del index[1]

    assert not index.contains(1)


def test_remove_updates_size(tmp_path):
    """len() reflects removal from active and view shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)

    # Add 3 keys, some will rotate to view shards
    for i in range(3):
        index.add(i, np.random.rand(64).astype(np.float32))

    initial_size = len(index)
    index.remove(0)
    assert len(index) == initial_size - 1


def test_remove_active_only_no_tombstone(tmp_path):
    """Key that exists only in active shard does not create a tombstone."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(42, np.random.rand(64).astype(np.float32))

    index.remove(42)

    assert index.tombstone_count == 0


def test_remove_then_readd(tmp_path):
    """Remove key K, add K with new vector, verify get returns new value."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vec_old = np.ones(64, dtype=np.float32)
    index.add(1, vec_old)
    index.add(2, np.random.rand(64).astype(np.float32))  # force rotation

    index.remove(1)
    vec_new = np.ones(64, dtype=np.float32) * 2.0
    index.add(1, vec_new)

    result = index.get(1)
    assert result is not None
    assert np.allclose(result, vec_new, atol=0.01)
    assert index.tombstone_count == 0  # tombstone cleared by add


def test_remove_from_search_results(tmp_path):
    """Removed entries don't appear in search results."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vecs = np.eye(64, dtype=np.float32)[:5]
    index.add(list(range(5)), vecs)

    index.remove(2)

    results = index.search(vecs[2], count=5)
    assert 2 not in results.keys.tolist()


def test_remove_from_contains(tmp_path):
    """Removed key returns False from contains."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))

    index.remove(1)

    assert not index.contains(1)


def test_remove_from_keys_iterator(tmp_path):
    """Removed keys are not yielded by keys iterator."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.rand(64).astype(np.float32))

    index.remove(1)

    keys_list = list(index.keys)
    assert 1 not in keys_list


def test_tombstone_persistence(tmp_path):
    """Tombstones survive save/load cycle."""
    # shard_size=1 forces rotation after each add, leaving active shard empty
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.rand(64).astype(np.float32))

    # Remove before first save — active shard is empty so save() only writes
    # bloom + tombstones (avoids Windows file lock on active shard)
    index.remove(0)
    index.save()

    # Reload and verify tombstone persisted
    index2 = ShardedIndex(ndim=64, path=tmp_path)
    assert not index2.contains(0)
    assert index2.contains(1)
    assert index2.tombstone_count == 1


def test_tombstone_file_cleanup(tmp_path):
    """Tombstone file is removed after compact resolves all overlap."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.rand(64).astype(np.float32))

    # Remove and save — creates tombstone file
    index.remove(0)
    index.save()
    assert (tmp_path / "tombstones.npy").exists()

    # Re-add clears the tombstone but creates cross-shard overlap
    # (key 0 now in both active shard and view shard), so the file persists
    # as a compaction-needed flag
    index.add(0, np.random.rand(64).astype(np.float32))
    index.save()
    assert (tmp_path / "tombstones.npy").exists()

    # Compact resolves overlap and removes tombstone file
    index.compact()
    assert not (tmp_path / "tombstones.npy").exists()


def test_remove_batch_empty(tmp_path):
    """Remove with empty batch is a no-op."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))

    index.remove([])

    assert index.contains(1)


def test_remove_count_reflects_tombstones(tmp_path):
    """count() returns 0 for tombstoned keys in view shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.rand(64).astype(np.float32))

    index.remove(0)

    assert index.count(0) == 0
    assert index.count(1) >= 1


def test_search_truncates_oversampled_with_tombstones(tmp_path):
    """Search returns at most `count` results when tombstones cause oversampling."""
    # shard_size=1 forces rotation: each add goes to a view shard, active is empty
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vecs = np.eye(64, dtype=np.float32)[:10]
    index.add(list(range(10)), vecs)

    # Active shard is empty after rotation; tombstone triggers oversampling
    index.remove(0)
    assert index._active_shard is None or len(index._active_shard) == 0
    assert index.tombstone_count == 1

    # Single query with small count — triggers truncation
    results = index.search(vecs[1], count=3)
    assert len(results.keys) <= 3

    # Batch query with small count — triggers truncation
    batch = index.search(vecs[1:4], count=3)
    assert batch.keys.shape[1] <= 3

    # Single query with large count — no truncation needed (early return)
    results_large = index.search(vecs[1], count=100)
    assert len(results_large.keys) <= 100

    # Batch query with large count — no truncation needed (early return)
    batch_large = index.search(vecs[1:4], count=100)
    assert batch_large.keys.shape[1] <= 100


def test_remove_no_resurrection(tmp_path):
    """Upsert creates cross-shard duplicate, remove must eliminate key everywhere."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vec = np.ones(64, dtype=np.float32)

    # Add key 1, rotation pushes it to view shard
    index.add(1, vec)
    index.add(2, np.random.rand(64).astype(np.float32))

    # Upsert key 1 — now in active shard AND view shard (cross-shard dup)
    index.upsert(1, vec * 2.0)

    # Remove key 1 — must be gone from both active and view
    index.remove(1)

    assert not index.contains(1)
    assert index.get(1) is None
    assert 1 not in list(index.keys)

    # Search must not return key 1
    results = index.search(vec, count=10)
    assert 1 not in results.keys.tolist()
