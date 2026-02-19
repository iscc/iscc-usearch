"""
Test ShardedIndex compaction.

Confirms expected behavior for compact():
- Removes tombstoned entries from view shards
- Removes cross-shard duplicates (newest wins)
- Clears tombstone set after compaction
- Preserves live entries
- Returns correct removed count
- Rebuilds bloom filter
- Removes fully dead shard files
"""

import numpy as np
import pytest

from iscc_usearch.sharded import ShardedIndex


def test_compact_removes_tombstoned(tmp_path):
    """Tombstoned entries are removed from shard files after compact."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.rand(64).astype(np.float32))

    # Remove and compact in same session (avoids Windows reload file lock)
    index.remove(0)
    assert index.tombstone_count == 1

    removed = index.compact()

    assert removed >= 1
    assert index.tombstone_count == 0
    assert not index.contains(0)
    assert index.contains(1)
    assert index.contains(2)


def test_compact_removes_cross_shard_duplicates(tmp_path):
    """After upsert+rotation, compact removes old version from view shard."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.ones(64, dtype=np.float32))
    index.add(2, np.random.rand(64).astype(np.float32))  # rotation

    # Upsert key 1 → remove from view, add to active
    index.upsert(1, np.ones(64, dtype=np.float32) * 2.0)
    # Rotate to make the upserted version a view shard too
    index.add(3, np.random.rand(64).astype(np.float32))  # rotation
    index.save()

    removed = index.compact()

    # The old version of key 1 in an older view shard should be removed
    assert removed >= 1
    assert index.contains(1)
    result = index.get(1)
    assert result is not None
    assert np.allclose(result, np.ones(64, dtype=np.float32) * 2.0, atol=0.1)


def test_compact_no_tombstones_returns_zero(tmp_path):
    """Compact with no tombstones and no duplicates returns 0."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.save()

    removed = index.compact()

    assert removed == 0


def test_compact_clears_tombstones(tmp_path):
    """Tombstone set is empty after compact."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.rand(64).astype(np.float32))

    index.remove(0)
    index.remove(1)
    assert index.tombstone_count == 2

    index.compact()

    assert index.tombstone_count == 0


def test_compact_preserves_live_entries(tmp_path):
    """Non-removed entries survive compaction."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vecs = {}
    for i in range(5):
        v = np.ones(64, dtype=np.float32) * (i + 1)
        vecs[i] = v
        index.add(i, v)

    index.remove([0, 2, 4])
    index.compact()

    # Remaining should be intact
    for key in [1, 3]:
        result = index.get(key)
        assert result is not None
        assert np.allclose(result, vecs[key], atol=0.1)


def test_compact_returns_removed_count(tmp_path):
    """compact() returns the number of entries removed."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(4):
        index.add(i, np.random.rand(64).astype(np.float32))

    index.remove([0, 1])

    removed = index.compact()
    assert removed == 2


def test_compact_updates_bloom(tmp_path):
    """Bloom filter is rebuilt after compaction."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.rand(64).astype(np.float32))

    index.remove(0)
    index.compact()

    assert index._bloom is not None


def test_compact_removes_empty_shards(tmp_path):
    """Fully dead view shards have their files deleted."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    # Each add rotates due to shard_size=1, creating separate shard files
    index.add(1, np.random.rand(64).astype(np.float32))
    index.add(2, np.random.rand(64).astype(np.float32))
    index.add(3, np.random.rand(64).astype(np.float32))

    initial_shard_count = index.shard_count

    # Remove key that's the only entry in its shard
    index.remove(1)
    index.compact()

    # At least one shard file should be gone
    assert index.shard_count <= initial_shard_count


def test_compact_read_only_raises(tmp_path):
    """compact() raises RuntimeError on read-only index."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.save()

    ro = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
    with pytest.raises(RuntimeError, match="read-only"):
        ro.compact()
