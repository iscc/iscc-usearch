"""Tests for ShardedIndex read-only mode.

Verifies that read_only=True opens all shards as memory-mapped views,
blocks write operations, and allows all read operations to work correctly.
"""

import numpy as np
import pytest

from iscc_usearch import ShardedNphdIndex
from iscc_usearch.sharded import ShardedIndex, ShardedIndex128
from iscc_usearch.sharded_nphd import ShardedNphdIndex128


# === Helpers ===


def _create_and_save(tmp_path, cls=ShardedIndex, n=20, **kwargs):
    """Create an index with data and save it to disk.

    :param tmp_path: Directory for shard files
    :param cls: Index class to use
    :param n: Number of vectors to add
    :param kwargs: Extra keyword arguments for the index constructor
    :return: Directory path
    """
    path = tmp_path / "idx"
    idx = cls(path=path, **kwargs)
    ndim_val = kwargs.get("ndim", kwargs.get("max_dim", 64))
    if cls in (ShardedNphdIndex, ShardedNphdIndex128):
        vectors = [np.random.randint(0, 256, ndim_val // 8, dtype=np.uint8) for _ in range(n)]
    else:
        vectors = np.random.rand(n, ndim_val).astype(np.float32)
    if cls in (ShardedIndex128, ShardedNphdIndex128):
        keys = [i.to_bytes(16, "big") for i in range(n)]
    else:
        keys = list(range(n))
    idx.add(keys, vectors)
    idx.save()
    return path


# === Core read-only tests ===


def test_read_only_opens_all_shards_as_views(tmp_path):
    """Read-only mode views all shards, no active shard."""
    path = _create_and_save(tmp_path, ndim=64)

    ro = ShardedIndex(path=path, read_only=True)

    assert ro._active_shard is None
    assert ro._active_shard_path is None
    assert len(ro._viewed_indexes) == 1
    assert len(ro) == 20


def test_read_only_search(tmp_path):
    """Search works across viewed shards."""
    path = _create_and_save(tmp_path, ndim=64)
    rw = ShardedIndex(path=path)

    ro = ShardedIndex(path=path, read_only=True)
    # Search with the first vector from the writable index
    query = np.asarray(rw.vectors[0])
    result = ro.search(query, count=5)

    assert len(result.keys) > 0
    assert result.distances[0] == pytest.approx(0.0, abs=5e-3)


def test_read_only_get(tmp_path):
    """Get retrieves vectors from viewed shards."""
    path = _create_and_save(tmp_path, ndim=64)

    ro = ShardedIndex(path=path, read_only=True)
    vec = ro.get(0)

    assert vec is not None
    assert len(vec) == 64


def test_read_only_contains(tmp_path):
    """Contains works including bloom filter fast path."""
    path = _create_and_save(tmp_path, ndim=64)

    ro = ShardedIndex(path=path, read_only=True)

    assert ro.contains(0) is True
    assert ro.contains(9999) is False
    # Batch
    result = ro.contains([0, 1, 9999])
    assert result[0] == True  # noqa: E712 - numpy bool
    assert result[1] == True  # noqa: E712 - numpy bool
    assert result[2] == False  # noqa: E712 - numpy bool


def test_read_only_count(tmp_path):
    """Count works across viewed shards."""
    path = _create_and_save(tmp_path, ndim=64)

    ro = ShardedIndex(path=path, read_only=True)

    assert ro.count(0) == 1
    assert ro.count(9999) == 0
    # Batch
    result = ro.count([0, 1, 9999])
    assert result[0] == 1
    assert result[1] == 1
    assert result[2] == 0


def test_read_only_keys_vectors(tmp_path):
    """Iteration over keys and vectors works."""
    path = _create_and_save(tmp_path, ndim=64)

    ro = ShardedIndex(path=path, read_only=True)

    keys = list(ro.keys)
    assert len(keys) == 20

    vectors = list(ro.vectors)
    assert len(vectors) == 20


# === Write operation guards ===


def test_read_only_add_raises(tmp_path):
    """Add raises IndexError on read-only index."""
    path = _create_and_save(tmp_path, ndim=64)
    ro = ShardedIndex(path=path, read_only=True)

    with pytest.raises(RuntimeError, match="read-only"):
        ro.add(100, np.random.rand(64).astype(np.float32))


def test_read_only_add_once_raises(tmp_path):
    """add_once raises IndexError on read-only index."""
    path = _create_and_save(tmp_path, ndim=64)
    ro = ShardedIndex(path=path, read_only=True)

    with pytest.raises(RuntimeError, match="read-only"):
        ro.add_once(100, np.random.rand(64).astype(np.float32))


def test_read_only_save_raises(tmp_path):
    """Save raises IndexError on read-only index."""
    path = _create_and_save(tmp_path, ndim=64)
    ro = ShardedIndex(path=path, read_only=True)

    with pytest.raises(RuntimeError, match="read-only"):
        ro.save()


def test_read_only_rebuild_bloom_raises(tmp_path):
    """rebuild_bloom raises IndexError on read-only index."""
    path = _create_and_save(tmp_path, ndim=64)
    ro = ShardedIndex(path=path, read_only=True)

    with pytest.raises(RuntimeError, match="read-only"):
        ro.rebuild_bloom()


def test_read_only_reset_raises(tmp_path):
    """Reset raises IndexError on read-only index."""
    path = _create_and_save(tmp_path, ndim=64)
    ro = ShardedIndex(path=path, read_only=True)

    with pytest.raises(RuntimeError, match="read-only"):
        ro.reset()


def test_read_only_expansion_add_setter_raises(tmp_path):
    """Setting expansion_add raises IndexError on read-only index."""
    path = _create_and_save(tmp_path, ndim=64)
    ro = ShardedIndex(path=path, read_only=True)

    with pytest.raises(RuntimeError, match="read-only"):
        ro.expansion_add = 256


# === Edge cases ===


def test_read_only_no_existing_shards_raises(tmp_path):
    """ValueError when read_only=True but no shards exist."""
    path = tmp_path / "empty"

    with pytest.raises(ValueError, match="read_only=True requires existing shards"):
        ShardedIndex(ndim=64, path=path, read_only=True)


def test_read_only_property(tmp_path):
    """read_only property returns correct value."""
    path = _create_and_save(tmp_path, ndim=64)

    rw = ShardedIndex(path=path)
    assert rw.read_only is False

    ro = ShardedIndex(path=path, read_only=True)
    assert ro.read_only is True


def test_read_only_repr(tmp_path):
    """repr includes read_only indicator."""
    path = _create_and_save(tmp_path, ndim=64)

    ro = ShardedIndex(path=path, read_only=True)
    r = repr(ro)

    assert "read_only" in r


def test_read_only_len_and_properties(tmp_path):
    """Size, ndim, dtype, and other properties work in read-only mode."""
    path = _create_and_save(tmp_path, ndim=64)

    ro = ShardedIndex(path=path, read_only=True)

    assert ro.size == 20
    assert len(ro) == 20
    assert ro.ndim == 64
    assert ro.dtype is not None
    assert ro.connectivity > 0
    assert ro.expansion_add > 0
    assert ro.expansion_search > 0
    assert ro.shard_count == 1
    assert ro.memory_usage >= 0
    assert ro.serialized_length == 0  # No active shard
    assert ro.capacity == 0  # No active shard


def test_read_only_expansion_search_settable(tmp_path):
    """expansion_search setter works on read-only (tuning search quality is safe)."""
    path = _create_and_save(tmp_path, ndim=64)
    ro = ShardedIndex(path=path, read_only=True)

    ro.expansion_search = 256
    assert ro.expansion_search == 256


# === Subclass variants ===


def test_read_only_nphd(tmp_path):
    """ShardedNphdIndex works in read-only mode."""
    path = _create_and_save(tmp_path, cls=ShardedNphdIndex, max_dim=256)

    ro = ShardedNphdIndex(path=path, read_only=True)

    assert ro.read_only is True
    assert ro._active_shard is None
    assert len(ro) == 20
    assert ro.max_dim == 256
    assert ro.max_bytes == 32
    assert "read_only" in repr(ro)

    # Search works
    query = np.random.randint(0, 256, 32, dtype=np.uint8)
    result = ro.search(query, count=5)
    assert len(result.keys) > 0

    # Get works
    vec = ro.get(0)
    assert vec is not None


def test_read_only_128(tmp_path):
    """ShardedIndex128 works in read-only mode."""
    path = _create_and_save(tmp_path, cls=ShardedIndex128, ndim=64)

    ro = ShardedIndex128(path=path, read_only=True)

    assert ro.read_only is True
    assert len(ro) == 20
    assert "read_only" in repr(ro)

    # Get with bytes key
    key = (0).to_bytes(16, "big")
    vec = ro.get(key)
    assert vec is not None

    # Contains
    assert ro.contains(key) is True


def test_read_only_nphd_128(tmp_path):
    """ShardedNphdIndex128 works in read-only mode."""
    path = _create_and_save(tmp_path, cls=ShardedNphdIndex128, max_dim=256)

    ro = ShardedNphdIndex128(path=path, read_only=True)

    assert ro.read_only is True
    assert len(ro) == 20
    assert ro.max_dim == 256
    assert "read_only" in repr(ro)

    # Search works
    query = np.random.randint(0, 256, 32, dtype=np.uint8)
    result = ro.search(query, count=5)
    assert len(result.keys) > 0


def test_read_only_multiple_shards(tmp_path):
    """Read-only mode correctly handles multiple shards."""
    path = tmp_path / "idx"
    # Add vectors one-by-one with tiny shard_size to trigger rotation
    idx = ShardedIndex(ndim=64, path=path, shard_size=1)
    vectors = np.random.rand(20, 64).astype(np.float32)
    for i in range(20):
        idx.add(i, vectors[i])
    idx.save()

    assert idx.shard_count >= 2, "Need multiple shards for this test"

    ro = ShardedIndex(path=path, read_only=True)

    assert ro._active_shard is None
    assert len(ro._viewed_indexes) == idx.shard_count
    assert len(ro) == 20

    # Search across all shards
    result = ro.search(vectors[0], count=5)
    assert result.keys[0] == 0
    assert result.distances[0] == pytest.approx(0.0, abs=5e-3)

    # Get from different shards
    assert ro.get(0) is not None
    assert ro.get(19) is not None


def test_read_only_batch_search(tmp_path):
    """Batch search works in read-only mode."""
    path = _create_and_save(tmp_path, ndim=64)
    rw = ShardedIndex(path=path)

    ro = ShardedIndex(path=path, read_only=True)
    queries = np.asarray(rw.vectors[:3])
    results = ro.search(queries, count=5)

    assert len(results) == 3
