"""
Test CRUD operations for ShardedIndex128 (128-bit UUID keys).

Confirms remove, upsert, tombstone persistence, and compaction
work correctly with bytes(16) keys and V16 arrays.
"""

import numpy as np
import pytest

from iscc_usearch.sharded import ShardedIndex128


def _uuid(n: int) -> bytes:
    """Create a deterministic 16-byte UUID key from an integer."""
    return n.to_bytes(16, "big")


def test_remove_uuid_key(tmp_path):
    """Remove with bytes(16) key."""
    index = ShardedIndex128(ndim=64, path=tmp_path)
    key = _uuid(1)
    index.add(key, np.random.rand(64).astype(np.float32))

    assert index.contains(key)
    index.remove(key)
    assert not index.contains(key)


def test_remove_uuid_batch(tmp_path):
    """Batch remove with UUID keys."""
    index = ShardedIndex128(ndim=64, path=tmp_path, shard_size=1)
    keys = [_uuid(i) for i in range(4)]
    for k in keys:
        index.add(k, np.random.rand(64).astype(np.float32))

    index.remove([keys[0], keys[2]])

    assert not index.contains(keys[0])
    assert index.contains(keys[1])
    assert not index.contains(keys[2])
    assert index.contains(keys[3])


def test_upsert_uuid_key(tmp_path):
    """Upsert with bytes(16) key."""
    index = ShardedIndex128(ndim=64, path=tmp_path)
    key = _uuid(1)
    vec_old = np.ones(64, dtype=np.float32)
    vec_new = np.ones(64, dtype=np.float32) * 2.0

    index.add(key, vec_old)
    index.upsert(key, vec_new)

    result = index.get(key)
    assert result is not None
    assert np.allclose(result, vec_new, atol=0.01)


def test_tombstone_persistence_uuid(tmp_path):
    """Roundtrip V16 tombstones through save/load."""
    index = ShardedIndex128(ndim=64, path=tmp_path, shard_size=1)
    keys = [_uuid(i) for i in range(3)]
    for k in keys:
        index.add(k, np.random.rand(64).astype(np.float32))

    # Remove before first save — active shard is empty after rotations
    index.remove(keys[0])
    index.save()

    index2 = ShardedIndex128(ndim=64, path=tmp_path)
    assert not index2.contains(keys[0])
    assert index2.contains(keys[1])
    assert index2.tombstone_count == 1


def test_compact_uuid(tmp_path):
    """Compaction with V16 keys."""
    index = ShardedIndex128(ndim=64, path=tmp_path, shard_size=1)
    keys = [_uuid(i) for i in range(3)]
    for k in keys:
        index.add(k, np.random.rand(64).astype(np.float32))

    # Remove and compact in same session (avoids Windows reload+save lock)
    index.remove(keys[0])
    removed = index.compact()

    assert removed >= 1
    assert not index.contains(keys[0])
    assert index.contains(keys[1])
    assert index.tombstone_count == 0


def test_delitem_uuid(tmp_path):
    """del index[key] works with UUID keys."""
    index = ShardedIndex128(ndim=64, path=tmp_path)
    key = _uuid(42)
    index.add(key, np.random.rand(64).astype(np.float32))

    del index[key]

    assert not index.contains(key)


def test_upsert_none_key_raises_uuid(tmp_path):
    """Upsert with None key raises ValueError."""
    index = ShardedIndex128(ndim=64, path=tmp_path)
    with pytest.raises(ValueError):
        index.upsert(None, np.random.rand(64).astype(np.float32))
