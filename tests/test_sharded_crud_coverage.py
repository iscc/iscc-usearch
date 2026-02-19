"""
Coverage tests for CRUD-related code paths.

Targets specific uncovered branches: iterator fast/slow paths, batch operations
with tombstones, search result filtering, count with tombstones, and UUID mixin
validation edges.
"""

import numpy as np
import pytest

from usearch.index import BatchMatches

from iscc_usearch.sharded import ShardedIndex, ShardedIndex128


# --- Iterator fast paths (read-only, no active shard, no tombstones) ---


def test_keys_getitem_fast_path_read_only(tmp_path):
    """Keys __getitem__ fast path when no filtering needed (read-only)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)
    index.save()

    ro = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
    assert ro._active_shard is None
    assert not ro._tombstones

    # Fast path __getitem__
    key = ro.keys[0]
    assert key is not None

    # Negative indexing
    key_neg = ro.keys[-1]
    assert key_neg is not None


def test_vectors_getitem_fast_path_negative_and_bounds(tmp_path):
    """Vectors __getitem__ fast path negative index and bounds check (read-only)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)
    index.save()

    ro = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
    assert ro._active_shard is None
    assert not ro._tombstones

    # Fast path negative indexing
    vec = ro.vectors[-1]
    assert isinstance(vec, np.ndarray)

    # Fast path out-of-range
    with pytest.raises(IndexError, match="out of range"):
        _ = ro.vectors[100]

    with pytest.raises(IndexError, match="out of range"):
        _ = ro.vectors[-100]


def test_keys_array_fast_path_read_only(tmp_path):
    """Keys __array__ fast path when no filtering needed (read-only)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)
    index.save()

    ro = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
    arr = np.asarray(ro.keys)
    assert len(arr) == 3

    # With dtype override
    arr_float = np.asarray(ro.keys, dtype=np.float64)
    assert arr_float.dtype == np.float64


def test_vectors_getitem_fast_path_read_only(tmp_path):
    """Vectors __getitem__ fast path when no filtering needed (read-only)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)
    index.save()

    ro = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
    vec = ro.vectors[0]
    assert vec is not None
    assert len(vec) == 64


def test_vectors_array_fast_path_read_only(tmp_path):
    """Vectors __array__ fast path when no filtering needed (read-only)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)
    index.save()

    ro = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
    arr = np.asarray(ro.vectors)
    assert arr.shape[0] == 3
    assert arr.shape[1] == 64


def test_vectors_array_fast_path_with_dtype(tmp_path):
    """Vectors __array__ fast path with dtype conversion."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(2):
        index.add(i, np.ones(64, dtype=np.float32) * i)
    index.save()

    ro = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
    arr = np.asarray(ro.vectors, dtype=np.float64)
    assert arr.dtype == np.float64


# --- Iterator slow paths (with active shard + dedup) ---


def test_keys_dedup_across_shards(tmp_path):
    """Keys iterator deduplicates when same key exists in active + view shard."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.ones(64, dtype=np.float32))
    index.add(2, np.ones(64, dtype=np.float32) * 2)

    # Upsert key 1 → now in active + view shard
    index.upsert(1, np.ones(64, dtype=np.float32) * 3)

    keys = list(index.keys)
    # Key 1 should appear only once
    assert keys.count(1) == 1


def test_keys_array_slow_path_empty(tmp_path):
    """Keys __array__ slow path when index is empty but has active shard."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    arr = np.asarray(index.keys)
    assert len(arr) == 0


def test_keys_array_slow_path_with_dtype(tmp_path):
    """Keys __array__ slow path with dtype override."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.ones(64, dtype=np.float32))
    arr = np.asarray(index.keys, dtype=np.float64)
    assert arr.dtype == np.float64


def test_vectors_array_slow_path_empty(tmp_path):
    """Vectors __array__ slow path when index is empty with active shard."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    arr = np.asarray(index.vectors)
    assert arr.shape[0] == 0


# --- Batch get with tombstones ---


def test_batch_get_with_tombstones(tmp_path):
    """Batch get filters tombstoned keys from view shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vecs = {}
    for i in range(4):
        v = np.ones(64, dtype=np.float32) * (i + 1)
        vecs[i] = v
        index.add(i, v)

    # Tombstone some
    index.remove([0, 2])

    results = index.get([0, 1, 2, 3])
    assert results[0] is None  # tombstoned
    assert results[1] is not None
    assert results[2] is None  # tombstoned
    assert results[3] is not None


# --- Batch contains with tombstones ---


def test_batch_contains_with_tombstones(tmp_path):
    """Batch contains filters tombstoned keys."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(4):
        index.add(i, np.ones(64, dtype=np.float32) * i)

    index.remove([0, 2])

    result = index.contains([0, 1, 2, 3])
    assert not result[0]
    assert result[1]
    assert not result[2]
    assert result[3]


# --- Count batch with tombstones ---


def test_count_batch_with_tombstones(tmp_path):
    """Batch count skips view shard counts for tombstoned keys."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(4):
        index.add(i, np.ones(64, dtype=np.float32) * i)

    index.remove([0, 2])

    counts = index.count([0, 1, 2, 3])
    assert counts[0] == 0  # tombstoned
    assert counts[1] >= 1
    assert counts[2] == 0  # tombstoned
    assert counts[3] >= 1


# --- Batch search with tombstone/active filtering ---


def test_batch_search_with_tombstones(tmp_path):
    """Batch search filters tombstoned + active keys from view results."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vecs = np.eye(64, dtype=np.float32)[:5]
    for i in range(5):
        index.add(i, vecs[i])

    index.remove(2)

    # Batch query
    results = index.search(vecs[:3], count=5)
    for row_idx in range(3):
        row_keys = results.keys[row_idx].tolist()
        assert 2 not in row_keys or results.distances[row_idx][row_keys.index(2)] == float("inf")


def test_single_search_all_view_filtered(tmp_path):
    """Single search where all view results are filtered returns empty or active-only."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vec = np.ones(64, dtype=np.float32)
    index.add(1, vec)

    # Upsert key 1 → active has it, view has stale version
    index.upsert(1, vec * 2)

    results = index.search(vec, count=5)
    # Should get active version, view version filtered
    assert len(results.keys) >= 1


def test_batch_search_all_view_filtered(tmp_path):
    """Batch search where all view results are filtered returns None view."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vec = np.ones(64, dtype=np.float32)
    index.add(1, vec)
    index.remove(1)

    # Only tombstoned entries in view shards
    results = index.search(np.vstack([vec, vec * 2]), count=5)
    assert results is not None


# --- Compact edge cases ---


def test_compact_mixed_actions(tmp_path):
    """Compact with mixed keep/rebuild/delete actions."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    # Add 5 keys → 5 view shards
    for i in range(5):
        index.add(i, np.ones(64, dtype=np.float32) * i)

    # Remove keys 0, 4 (first and last view shards become dead)
    # Keys 1, 2, 3 remain (their shards are kept)
    index.remove([0, 4])

    removed = index.compact()
    assert removed == 2
    assert index.contains(1)
    assert index.contains(2)
    assert index.contains(3)


def test_compact_with_active_shard_dedup(tmp_path):
    """Compact deduplicates keys present in both active and view shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.ones(64, dtype=np.float32))
    index.add(2, np.ones(64, dtype=np.float32) * 2)

    # Upsert key 1 into active shard (also exists in view shard)
    index.upsert(1, np.ones(64, dtype=np.float32) * 3)

    removed = index.compact()
    # Old version of key 1 in view shard should be removed
    assert removed >= 1
    result = index.get(1)
    assert result is not None


# --- UUID mixin remove/upsert validation ---


def _uuid(n: int) -> bytes:
    return n.to_bytes(16, "big")


def test_uuid_remove_invalid_key_length(tmp_path):
    """UUID remove with wrong-length bytes raises ValueError."""
    index = ShardedIndex128(ndim=64, path=tmp_path)
    index.add(_uuid(1), np.random.rand(64).astype(np.float32))

    with pytest.raises(ValueError, match="16 bytes"):
        index.remove(b"short")


def test_uuid_upsert_invalid_key_length(tmp_path):
    """UUID upsert with wrong-length bytes raises ValueError."""
    index = ShardedIndex128(ndim=64, path=tmp_path)

    with pytest.raises(ValueError, match="16 bytes"):
        index.upsert(b"short", np.random.rand(64).astype(np.float32))


def test_uuid_upsert_batch_validation(tmp_path):
    """UUID upsert validates batch key dtype."""
    index = ShardedIndex128(ndim=64, path=tmp_path)

    with pytest.raises(ValueError, match="V16"):
        index.upsert(np.array([1, 2], dtype=np.uint64), np.random.rand(2, 64).astype(np.float32))


def test_uuid_remove_batch_iterable(tmp_path):
    """UUID remove with iterable of bytes keys."""
    index = ShardedIndex128(ndim=64, path=tmp_path)
    keys = [_uuid(i) for i in range(3)]
    for k in keys:
        index.add(k, np.random.rand(64).astype(np.float32))

    index.remove([keys[0], keys[1]])

    assert not index.contains(keys[0])
    assert not index.contains(keys[1])
    assert index.contains(keys[2])


def test_uuid_upsert_batch(tmp_path):
    """UUID upsert with batch of V16 keys."""
    index = ShardedIndex128(ndim=64, path=tmp_path)
    keys = [_uuid(i) for i in range(3)]
    for k in keys:
        index.add(k, np.ones(64, dtype=np.float32))

    # Upsert batch
    new_vecs = np.ones((3, 64), dtype=np.float32) * 2
    index.upsert(keys, new_vecs)

    for k in keys:
        result = index.get(k)
        assert result is not None
        assert np.allclose(result, np.ones(64, dtype=np.float32) * 2, atol=0.1)


# --- Search filter edge cases ---


def test_search_active_suppresses_view_batch(tmp_path):
    """Batch search: active shard keys suppress view shard results per row."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vecs = np.eye(64, dtype=np.float32)[:4]
    for i in range(4):
        index.add(i, vecs[i])

    # Upsert key 0 into active shard (view still has old version)
    index.upsert(0, vecs[0] * 2)

    # Batch query should filter stale view versions of key 0
    results = index.search(vecs[:2], count=10)
    assert results is not None


def test_search_only_tombstoned_entries(tmp_path):
    """Search with only tombstoned entries in view shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vec = np.ones(64, dtype=np.float32)
    index.add(1, vec)

    index.remove(1)

    # All view entries tombstoned, active empty
    results = index.search(vec, count=5)
    assert results is not None


# --- Iterator slow path edge cases ---


def test_keys_getitem_out_of_range_slow_path(tmp_path):
    """Keys __getitem__ out of range in slow path."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.ones(64, dtype=np.float32))

    with pytest.raises(IndexError):
        _ = index.keys[100]


def test_vectors_getitem_out_of_range_slow_path(tmp_path):
    """Vectors __getitem__ out of range in slow path."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.ones(64, dtype=np.float32))

    with pytest.raises(IndexError):
        _ = index.vectors[100]


def test_vectors_array_slow_path_with_active(tmp_path):
    """Vectors __array__ slow path with active shard + view shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)

    # Active shard present → slow path
    arr = np.asarray(index.vectors)
    assert arr.shape[0] == 3


def test_keys_iter_with_tombstones_no_active(tmp_path):
    """Keys iterator slow path: tombstones but no active shard (read-only after save)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)
    index.remove(1)
    index.save()

    # Reload as read-only — has tombstones, no active shard
    ro = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
    keys = list(ro.keys)
    assert 1 not in keys
    assert 0 in keys
    assert 2 in keys


# --- Compact path: rebuild with no view changes (all keep) ---


def test_compact_empty_shard_skip(tmp_path):
    """Compact handles shards gracefully when view shards exist with no tombstones."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)

    # No tombstones — compact should be a no-op (all keep)
    removed = index.compact()
    assert removed == 0
    assert len(index) == 3


def test_count_single_tombstoned_in_active_and_view(tmp_path):
    """Count returns 0 for tombstoned key even if active has been removed."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.ones(64, dtype=np.float32))
    index.add(2, np.ones(64, dtype=np.float32) * 2)

    # Key 1 in view shard, upsert puts it in active, then remove
    index.upsert(1, np.ones(64, dtype=np.float32) * 3)
    index.remove(1)

    assert index.count(1) == 0


# --- Vectors iterator slow path: no active shard (tombstones-only) ---


def test_vectors_iter_slow_path_no_active_shard(tmp_path):
    """Vectors __iter__ slow path with tombstones but no active shard."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)
    index.remove(1)
    index.save()

    # Reload read-only: has tombstones, no active shard → slow path, 249->255 branch
    ro = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
    assert ro._active_shard is None
    assert ro._tombstones

    vecs = list(ro.vectors)
    assert len(vecs) == 2


def test_vectors_iter_slow_path_no_view_shards(tmp_path):
    """Vectors __iter__ slow path with active shard only (no view shards)."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.ones(64, dtype=np.float32))

    # Active shard present, no view shards → slow path, 258->256 branch
    assert len(index._viewed_indexes) == 0
    vecs = list(index.vectors)
    assert len(vecs) == 1


# --- Vectors __getitem__ fast path multi-shard ---


def test_vectors_getitem_fast_path_multi_shard(tmp_path):
    """Vectors __getitem__ fast path accesses item in second shard."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * (i + 1))
    index.save()

    ro = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
    # Access item in second view shard (line 289 = current += shard_len)
    vec = ro.vectors[1]
    assert vec is not None
    assert len(vec) == 64


# --- Contains batch early break ---


def test_contains_batch_early_break(tmp_path):
    """Batch contains resolves all keys before checking all view shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    # Create 3 view shards with 1 key each
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)

    # All keys exist in view shards, early break should trigger (line 901)
    result = index.contains([0, 1, 2])
    assert all(result)


# --- Count batch all tombstoned ---


def test_count_batch_all_tombstoned(tmp_path):
    """Batch count where all keys are tombstoned skips view shard counting."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)

    # Tombstone all keys in view shards
    index.remove([0, 1, 2])

    # All tombstoned → not_tombstoned.any() is False → 981->989 branch
    counts = index.count([0, 1, 2])
    assert all(c == 0 for c in counts)


# --- Compact rebuild action (partial shard) ---


def test_compact_rebuild_partial_shard(tmp_path):
    """Compact rebuilds a shard where only some keys are tombstoned."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    # Add 3 keys to active shard (default shard_size: no auto-rotation)
    index.add(0, np.ones(64, dtype=np.float32) * 1)
    index.add(1, np.ones(64, dtype=np.float32) * 2)
    index.add(2, np.ones(64, dtype=np.float32) * 3)
    # Force rotation to create multi-key view shard
    index._rotate_shard()
    assert len(index._viewed_indexes) == 1
    assert len(index._viewed_indexes[0]) == 3

    # Tombstone key 0 (in multi-key view shard) → partial rebuild action
    index.remove(0)

    removed = index.compact()
    assert removed == 1
    assert not index.contains(0)
    assert index.contains(1)
    assert index.contains(2)

    # Verify rebuilt shard has correct data
    result = index.get(1)
    assert result is not None
    assert np.allclose(result, np.ones(64, dtype=np.float32) * 2, atol=0.1)


# --- Remove single bloom rejection ---


def test_remove_single_bloom_rejection(tmp_path):
    """Remove non-existent key rejected by bloom filter fast path."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add(1, np.ones(64, dtype=np.float32))

    # Key 999 not in bloom → _remove_single bloom rejection
    index.remove(999)
    assert index.contains(1)
    assert len(index) == 1


def test_remove_single_no_bloom(tmp_path):
    """Remove single key with bloom filter disabled (1478->1483 False branch)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1, bloom_filter=False)
    index.add(1, np.ones(64, dtype=np.float32))
    index.add(2, np.ones(64, dtype=np.float32) * 2)

    # Single remove skips bloom check entirely
    index.remove(1)
    assert not index.contains(1)
    assert index.contains(2)


# --- Remove batch no bloom ---


def test_remove_batch_no_bloom(tmp_path):
    """Batch remove with bloom disabled (line 1506)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1, bloom_filter=False)
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)

    index.remove([0, 1])
    assert not index.contains(0)
    assert not index.contains(1)
    assert index.contains(2)


# --- Remove batch all found in view break ---


def test_remove_batch_all_found_in_view_break(tmp_path):
    """Batch remove finds all keys early, skipping remaining view shards (line 1523)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    # Create 5 view shards (shard_size=1), active is empty
    for i in range(5):
        index.add(i, np.ones(64, dtype=np.float32) * i)

    # Remove keys 3 and 4 — in newest 2 view shards (checked first in reversed order)
    # After finding both, in_view.all()=True, breaks before checking remaining 3 shards
    index.remove([3, 4])
    assert not index.contains(3)
    assert not index.contains(4)
    assert index.contains(0)


# --- Remove batch no keys in view ---


def test_remove_batch_no_keys_in_view(tmp_path):
    """Batch remove where keys not in any view shard (line 1528->exit)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1, bloom_filter=False)
    # Create view shards with keys 0, 1, 2
    for i in range(3):
        index.add(i, np.ones(64, dtype=np.float32) * i)

    # Batch remove keys that don't exist in any view shard
    # bloom_filter=False ensures no early rejection; view shards exist but keys not found
    index.remove([100, 200])
    # Should be a no-op — original keys still present
    assert index.contains(0)
    assert index.contains(1)
    assert index.contains(2)


# --- Batch search filter: all view results filtered ---


def test_batch_filter_partial_row_empty(tmp_path):
    """_filter_batch_view_results with one row fully filtered and one with valid results."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.ones(64, dtype=np.float32))
    index._rotate_shard()
    # Tombstone keys 1 and 2 so crafted batch rows can be fully excluded
    index._tombstones = {1, 2}

    # Craft BatchMatches: row 0 has only tombstoned keys, row 1 has a valid key
    keys = np.array([[1, 2], [1, 5]], dtype=np.uint64)
    dists = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    batch = BatchMatches(keys=keys, distances=dists, counts=np.array([2, 2], dtype=np.int64))
    result = index._filter_batch_view_results(batch)
    assert result is not None
    assert result.counts[0] == 0  # row 0 fully filtered
    assert result.counts[1] == 1  # row 1 has key 5 (valid)
    assert result.keys[1, 0] == 5
    assert result.distances[1, 0] == 0.4


def test_batch_search_all_view_results_filtered(tmp_path):
    """Batch search where all view results are filtered (tombstoned + active)."""
    vec = np.ones(64, dtype=np.float32)
    index = ShardedIndex(ndim=64, path=tmp_path)
    # Add key 0 to active, force to view
    index.add(0, vec)
    index._rotate_shard()
    # Upsert key 0: tombstones view copy, adds to active shard
    index.upsert(0, vec * 2)
    # Now: view has key 0 (tombstoned), active has key 0
    # Batch search: view returns key 0 → filtered (tombstoned + in active)
    # Padding key 0 → also in active → filtered
    # All view results filtered → _filter_batch_view_results returns None (line 1987)
    results = index.search(np.vstack([vec, vec * 2]), count=5)
    assert results is not None


# --- UUID upsert with iterable of bytes keys ---


def test_uuid_upsert_iterable_keys(tmp_path):
    """UUID upsert with list of bytes keys (iterable path)."""
    index = ShardedIndex128(ndim=64, path=tmp_path)
    keys = [_uuid(i) for i in range(3)]
    vecs = np.random.rand(3, 64).astype(np.float32)
    for k, v in zip(keys, vecs):
        index.add(k, v)

    # Upsert with list of bytes — triggers normalize_batch_keys path (line 2127)
    new_vecs = np.ones((3, 64), dtype=np.float32) * 5
    index.upsert(keys, new_vecs)

    for k in keys:
        result = index.get(k)
        assert result is not None
        assert np.allclose(result, np.ones(64, dtype=np.float32) * 5, atol=0.1)


def test_uuid_upsert_ndarray_valid_dtype(tmp_path):
    """UUID upsert with ndarray of valid V16 dtype (2124->2128 False branch)."""
    index = ShardedIndex128(ndim=64, path=tmp_path)
    keys_list = [_uuid(i) for i in range(3)]
    for k in keys_list:
        index.add(k, np.ones(64, dtype=np.float32))

    # Create V16 ndarray — valid dtype, skips error branch
    keys_arr = np.array(keys_list, dtype="V16")
    new_vecs = np.ones((3, 64), dtype=np.float32) * 7
    index.upsert(keys_arr, new_vecs)

    for k in keys_list:
        result = index.get(k)
        assert result is not None
        assert np.allclose(result, np.ones(64, dtype=np.float32) * 7, atol=0.1)
