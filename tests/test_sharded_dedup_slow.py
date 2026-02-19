"""Coverage tests for dedup/tombstone slow paths with _needs_compact=True.

Targets uncovered code in sharded.py: iterator, getitem, and array slow paths
when active shard has entries and _needs_compact=True; add with non-matching
tombstones; contains loop exhaustion; search active-shard filtering; and UUID
_tombstoned_mask guard.
"""

import numpy as np
import pytest

from iscc_usearch.sharded import ShardedIndex, ShardedIndex128


def _uuid(n: int) -> bytes:
    """Create a deterministic 16-byte key from an integer."""
    return n.to_bytes(16, "big")


def _make_needs_compact_index(tmp_path):
    # type: (...) -> ShardedIndex
    """Create a ShardedIndex with _needs_compact=True and active shard entries.

    Adds two keys to a view shard, then upserts key 0 so it exists in both
    the active shard and the view shard.
    """
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(0, np.ones(64, dtype=np.float32))
    index.add(1, np.ones(64, dtype=np.float32) * 2)
    index._rotate_shard()
    index.upsert(0, np.ones(64, dtype=np.float32) * 3)
    assert index._needs_compact
    assert index._active_shard is not None
    assert len(index._active_shard) > 0
    return index


# --- Keys slow paths ---


def test_keys_iter_slow_path_with_active_shard(tmp_path):
    # type: () -> None
    """Keys iterator yields from active shard in dedup path."""
    index = _make_needs_compact_index(tmp_path)
    keys = list(index.keys)
    assert len(keys) == 2
    assert keys.count(0) == 1


def test_keys_getitem_slow_path_needs_compact(tmp_path):
    # type: () -> None
    """Keys getitem slow path handles negative and out-of-range indexes."""
    index = _make_needs_compact_index(tmp_path)

    key = index.keys[0]
    assert key is not None

    key_neg = index.keys[-1]
    assert key_neg is not None

    with pytest.raises(IndexError):
        _ = index.keys[-100]

    with pytest.raises(IndexError):
        _ = index.keys[100]


def test_keys_array_slow_path_non_empty(tmp_path):
    # type: () -> None
    """Keys array slow path materializes via iterator."""
    index = _make_needs_compact_index(tmp_path)
    arr = np.asarray(index.keys)
    assert len(arr) == 2

    arr_f = np.asarray(index.keys, dtype=np.float64)
    assert arr_f.dtype == np.float64


def test_keys_array_slow_path_empty(tmp_path):
    # type: () -> None
    """Keys array slow path yields empty when all keys tombstoned."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(0, np.ones(64, dtype=np.float32))
    index.remove(0)
    arr = np.asarray(index.keys)
    assert len(arr) == 0


# --- Vectors slow paths ---


def test_vectors_iter_slow_path_with_active_shard(tmp_path):
    # type: () -> None
    """Vectors iterator yields from active shard in dedup path."""
    index = _make_needs_compact_index(tmp_path)
    vecs = list(index.vectors)
    assert len(vecs) == 2


def test_vectors_getitem_slow_path_needs_compact(tmp_path):
    # type: () -> None
    """Vectors getitem slow path handles negative and out-of-range indexes."""
    index = _make_needs_compact_index(tmp_path)

    vec = index.vectors[0]
    assert len(vec) == 64

    vec_neg = index.vectors[-1]
    assert len(vec_neg) == 64

    with pytest.raises(IndexError):
        _ = index.vectors[-100]

    with pytest.raises(IndexError):
        _ = index.vectors[100]


def test_vectors_array_slow_path_non_empty(tmp_path):
    # type: () -> None
    """Vectors array slow path materializes via iterator."""
    index = _make_needs_compact_index(tmp_path)
    arr = np.asarray(index.vectors)
    assert arr.shape[0] == 2
    assert arr.shape[1] == 64

    arr_f = np.asarray(index.vectors, dtype=np.float64)
    assert arr_f.dtype == np.float64


def test_vectors_array_slow_path_empty(tmp_path):
    # type: () -> None
    """Vectors array slow path yields empty when all keys tombstoned."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(0, np.ones(64, dtype=np.float32))
    index.remove(0)
    arr = np.asarray(index.vectors)
    assert arr.shape[0] == 0


# --- Add with non-matching tombstones ---


def test_add_with_tombstones_no_match(tmp_path):
    # type: () -> None
    """Add when tombstones exist but added key does not match any tombstone."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(0, np.ones(64, dtype=np.float32))
    index.add(1, np.ones(64, dtype=np.float32) * 2)
    index.remove(0)
    assert index.tombstone_count == 1

    index.add(99, np.ones(64, dtype=np.float32) * 3)
    assert index.tombstone_count == 1


# --- Contains loop exhaustion ---


def test_contains_tombstone_loop_exhausts(tmp_path):
    # type: () -> None
    """Contains with tombstones exhausts all view shards without early break."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1, bloom_filter=False)
    index.add(0, np.ones(64, dtype=np.float32))
    index.add(1, np.ones(64, dtype=np.float32) * 2)
    index.remove(0)

    result = index.contains([1, 999])
    assert result[0]
    assert not result[1]


# --- Search: active shard excludes view results ---


def test_search_active_excludes_view_results(tmp_path):
    # type: () -> None
    """Single search filters view keys present in active shard."""
    index = _make_needs_compact_index(tmp_path)
    results = index.search(np.ones(64, dtype=np.float32), count=5)
    assert results is not None
    assert len(results.keys) >= 1


# --- UUID _tombstoned_mask guard ---


def test_uuid_tombstoned_mask_empty_tombstones(tmp_path):
    # type: () -> None
    """UUID compact triggers _tombstoned_mask with empty tombstones."""
    index = ShardedIndex128(ndim=64, path=tmp_path)
    k0, k1 = _uuid(0), _uuid(1)
    index.add(k0, np.ones(64, dtype=np.float32))
    index.add(k1, np.ones(64, dtype=np.float32) * 2)
    index._rotate_shard()
    index.upsert(k0, np.ones(64, dtype=np.float32) * 3)
    assert index._needs_compact
    assert not index._tombstones

    removed = index.compact()
    assert removed >= 1
