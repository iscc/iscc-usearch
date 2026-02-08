"""Tests for ShardedIndex128 — 128-bit UUID key support."""

import struct

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from iscc_usearch.index import Index
from iscc_usearch.sharded import ShardedIndex, ShardedIndex128, UUID_DTYPE


# --- Helpers ---


def make_key(i: int) -> bytes:
    """Create a deterministic 16-byte key from an integer."""
    return i.to_bytes(16, "big")


def make_keys(n: int, offset: int = 0) -> np.ndarray:
    """Create a V16 array of n deterministic keys."""
    return np.array([make_key(i + offset) for i in range(n)], dtype=UUID_DTYPE)


def random_vectors(n: int, ndim: int = 32) -> np.ndarray:
    """Create n random float32 vectors."""
    rng = np.random.default_rng(42)
    return rng.random((n, ndim), dtype=np.float32)


# === Core Operations ===


def test_add_single_key(tmp_path):
    """Add a single vector with a bytes(16) key."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    key = make_key(1)
    vec = np.ones(8, dtype=np.float32)
    result = idx.add(key, vec)
    assert len(idx) == 1
    assert bytes(result[0]) == key


def test_add_batch_keys(tmp_path):
    """Add a batch of vectors with V16 key array."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = make_keys(5)
    vecs = random_vectors(5, ndim=8)
    result = idx.add(keys, vecs)
    assert len(idx) == 5
    assert result.dtype == UUID_DTYPE
    for i in range(5):
        assert bytes(result[i]) == bytes(keys[i])


def test_search_returns_v16_keys(tmp_path):
    """Search results contain V16 keys."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    key = make_key(42)
    vec = np.ones(8, dtype=np.float32)
    idx.add(key, vec)
    matches = idx.search(vec, count=1)
    assert matches.keys.dtype == UUID_DTYPE
    assert bytes(matches.keys[0]) == key
    assert matches.distances[0] == pytest.approx(0.0, abs=1e-5)


def test_search_batch_returns_v16_keys(tmp_path):
    """Batch search results contain V16 keys."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = make_keys(3)
    vecs = random_vectors(3, ndim=8)
    idx.add(keys, vecs)
    batch = idx.search(vecs, count=2)
    assert batch.keys.dtype == UUID_DTYPE


def test_get_single(tmp_path):
    """Get a single vector by bytes key."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    key = make_key(7)
    vec = np.arange(8, dtype=np.float32)
    idx.add(key, vec)
    result = idx.get(key)
    assert result is not None
    assert_array_equal(result, vec)


def test_get_single_missing(tmp_path):
    """Get returns None for missing key."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    key = make_key(1)
    vec = np.ones(8, dtype=np.float32)
    idx.add(key, vec)
    assert idx.get(make_key(999)) is None


def test_get_batch(tmp_path):
    """Get multiple vectors by batch of bytes keys."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = make_keys(3)
    vecs = random_vectors(3, ndim=8)
    idx.add(keys, vecs)
    results = idx.get([make_key(0), make_key(1), make_key(999)])
    assert results[0] is not None
    assert results[1] is not None
    assert results[2] is None


def test_contains_single(tmp_path):
    """Contains returns True/False for single bytes key."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    key = make_key(5)
    idx.add(key, np.ones(8, dtype=np.float32))
    assert idx.contains(key) is True
    assert idx.contains(make_key(999)) is False


def test_contains_batch(tmp_path):
    """Contains batch returns boolean array for V16 keys."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = make_keys(3)
    idx.add(keys, random_vectors(3, ndim=8))
    check = np.array([make_key(0), make_key(1), make_key(999)], dtype=UUID_DTYPE)
    result = idx.contains(check)
    assert_array_equal(result, [True, True, False])


def test_contains_in_operator(tmp_path):
    """The 'in' operator works with bytes keys."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    key = make_key(1)
    idx.add(key, np.ones(8, dtype=np.float32))
    assert key in idx
    assert make_key(999) not in idx


def test_count_single(tmp_path):
    """Count returns integer for single bytes key."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    key = make_key(1)
    idx.add(key, np.ones(8, dtype=np.float32))
    assert idx.count(key) == 1
    assert idx.count(make_key(999)) == 0


def test_count_batch(tmp_path):
    """Count batch returns uint64 array for V16 keys."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = make_keys(3)
    idx.add(keys, random_vectors(3, ndim=8))
    check = np.array([make_key(0), make_key(2), make_key(999)], dtype=UUID_DTYPE)
    result = idx.count(check)
    assert_array_equal(result, [1, 1, 0])


# === Validation ===


def test_add_keys_none_raises(tmp_path):
    """keys=None raises ValueError."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    with pytest.raises(ValueError, match="Auto-key generation not supported"):
        idx.add(None, np.ones(8, dtype=np.float32))


def test_add_wrong_key_length_raises(tmp_path):
    """Wrong bytes length raises ValueError on add."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    with pytest.raises(ValueError, match="exactly 16 bytes"):
        idx.add(b"\x01" * 8, np.ones(8, dtype=np.float32))


def test_add_batch_wrong_dtype_raises(tmp_path):
    """Non-V16 ndarray batch raises ValueError on add."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    with pytest.raises(ValueError, match="dtype 'V16'"):
        idx.add(np.array([1, 2, 3], dtype=np.uint64), random_vectors(3, ndim=8))


def test_get_wrong_key_length_raises(tmp_path):
    """Wrong bytes length raises ValueError on get."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    idx.add(make_key(1), np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="bytes of length 16"):
        idx.get(b"\x01" * 8)


def test_contains_wrong_key_length_raises(tmp_path):
    """Wrong bytes length raises ValueError on contains."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    idx.add(make_key(1), np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="bytes of length 16"):
        idx.contains(b"\x01" * 8)


def test_count_wrong_key_length_raises(tmp_path):
    """Wrong bytes length raises ValueError on count."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    idx.add(make_key(1), np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="bytes of length 16"):
        idx.count(b"\x01" * 8)


def test_batch_keys_non_v16_ndarray_raises(tmp_path):
    """Non-V16 ndarray passed to _normalize_batch_keys raises ValueError."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    idx.add(make_key(1), np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="dtype 'V16'"):
        idx.contains(np.array([1, 2], dtype=np.uint64))


def test_batch_keys_wrong_length_bytes_raises(tmp_path):
    """Sequence[bytes] with wrong-length elements raises ValueError."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    idx.add(make_key(1), np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="bytes of length 16"):
        idx.contains([b"\x01" * 8, b"\x02" * 16])


# === Key Iteration ===


def test_keys_iteration(tmp_path):
    """ShardedIndexedKeys iteration yields V16 elements."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = make_keys(5)
    idx.add(keys, random_vectors(5, ndim=8))
    collected = list(idx.keys)
    assert len(collected) == 5
    # usearch doesn't guarantee insertion order — compare as sets
    collected_set = {bytes(k) for k in collected}
    expected_set = {bytes(k) for k in keys}
    assert collected_set == expected_set


def test_keys_indexing(tmp_path):
    """ShardedIndexedKeys supports integer indexing."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = make_keys(3)
    idx.add(keys, random_vectors(3, ndim=8))
    assert bytes(idx.keys[0]) == bytes(keys[0])
    assert bytes(idx.keys[-1]) == bytes(keys[2])


def test_keys_slicing(tmp_path):
    """ShardedIndexedKeys supports slicing."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = make_keys(5)
    idx.add(keys, random_vectors(5, ndim=8))
    sliced = idx.keys[:3]
    assert len(sliced) == 3
    assert sliced.dtype == UUID_DTYPE


def test_keys_numpy_conversion(tmp_path):
    """np.asarray(idx.keys) returns V16-dtype array."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = make_keys(4)
    idx.add(keys, random_vectors(4, ndim=8))
    arr = np.asarray(idx.keys)
    assert arr.dtype == UUID_DTYPE
    assert len(arr) == 4


# === Persistence ===


def test_save_load_roundtrip(tmp_path):
    """Save and reload preserves uuid key kind and data."""
    path = tmp_path / "idx"
    idx = ShardedIndex128(ndim=8, path=path, dtype="f32")
    key = make_key(42)
    vec = np.arange(8, dtype=np.float32)
    idx.add(key, vec)
    idx.save()

    loaded = ShardedIndex128(ndim=8, path=path, dtype="f32")
    assert len(loaded) == 1
    assert loaded.contains(key)
    result = loaded.get(key)
    assert result is not None
    assert_array_equal(result, vec)


def test_save_load_batch_roundtrip(tmp_path):
    """Save and reload with batch data preserves all keys and vectors."""
    path = tmp_path / "idx"
    idx = ShardedIndex128(ndim=8, path=path, dtype="f32")
    keys = make_keys(10)
    vecs = random_vectors(10, ndim=8)
    idx.add(keys, vecs)
    idx.save()

    loaded = ShardedIndex128(ndim=8, path=path, dtype="f32")
    assert len(loaded) == 10
    for i in range(10):
        assert loaded.contains(make_key(i))


def test_reopen_uuid_on_uint64_shards_fails(tmp_path):
    """Opening ShardedIndex128 on uint64 shards raises error."""
    path = tmp_path / "idx"
    # Create a uint64 index
    plain = ShardedIndex(ndim=8, path=path, dtype="f32")
    plain.add(1, np.ones(8, dtype=np.float32))
    plain.save()

    # Try to open as uuid — usearch should reject the mismatch
    with pytest.raises(Exception):
        ShardedIndex128(path=path, dtype="f32")


# === Multi-shard ===


def test_shard_rotation(tmp_path):
    """Shard rotation works with uuid keys, cross-shard search succeeds."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32", shard_size=500)
    vecs = random_vectors(50, ndim=8)
    for i in range(50):
        idx.add(make_key(i), vecs[i])

    assert idx.shard_count >= 2
    assert len(idx) == 50
    assert len(idx.keys) == 50

    # Search should find vectors across shards
    matches = idx.search(vecs[0], count=5)
    assert len(matches) > 0
    assert matches.keys.dtype == UUID_DTYPE


def test_bloom_fast_path_rejection(tmp_path):
    """Bloom filter rejects non-existent bytes keys without shard access."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = make_keys(10)
    idx.add(keys, random_vectors(10, ndim=8))

    # This key was never added — bloom should reject it
    missing_key = make_key(999)
    assert idx.contains(missing_key) is False
    assert idx.get(missing_key) is None
    assert idx.count(missing_key) == 0


def test_batch_search_with_radius_filter(tmp_path):
    """Batch search with radius filter produces valid V16 masked results."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32", shard_size=500)
    vecs = random_vectors(30, ndim=8)
    for i in range(30):
        idx.add(make_key(i), vecs[i])

    results = idx.search(vecs[:3], count=5, radius=0.5)
    assert results.keys.dtype == UUID_DTYPE


def test_multi_shard_search_merge(tmp_path):
    """Search result merging works correctly with V16 keys across shards."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32", shard_size=500)
    vecs = random_vectors(50, ndim=8)
    for i in range(50):
        idx.add(make_key(i), vecs[i])

    assert idx.shard_count >= 2

    # Exact self-search should return the query vector's key first
    matches = idx.search(vecs[25], count=3)
    assert bytes(matches.keys[0]) == make_key(25)


def test_multi_shard_contains_batch(tmp_path):
    """Batch contains works across multiple shards."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32", shard_size=500)
    for i in range(50):
        idx.add(make_key(i), random_vectors(1, ndim=8)[0])

    assert idx.shard_count >= 2
    assert len(idx) == 50

    check = np.array([make_key(0), make_key(25), make_key(49), make_key(999)], dtype=UUID_DTYPE)
    result = idx.contains(check)
    assert_array_equal(result, [True, True, True, False])


def test_multi_shard_get_batch(tmp_path):
    """Batch get works across multiple shards."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32", shard_size=500)
    vecs = random_vectors(50, ndim=8)
    for i in range(50):
        idx.add(make_key(i), vecs[i])

    assert idx.shard_count >= 2
    assert len(idx) == 50

    results = idx.get([make_key(0), make_key(49), make_key(999)])
    assert results[0] is not None
    assert results[1] is not None
    assert results[2] is None


def test_multi_shard_count_batch(tmp_path):
    """Batch count works across multiple shards."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32", shard_size=500)
    for i in range(50):
        idx.add(make_key(i), random_vectors(1, ndim=8)[0])

    assert idx.shard_count >= 2
    assert len(idx) == 50

    check = np.array([make_key(0), make_key(49), make_key(999)], dtype=UUID_DTYPE)
    result = idx.count(check)
    assert_array_equal(result, [1, 1, 0])


# === Edge Cases ===


def test_empty_index(tmp_path):
    """Empty uuid index has correct behavior."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    assert len(idx) == 0
    assert idx.contains(make_key(1)) is False
    assert idx.get(make_key(1)) is None
    assert idx.count(make_key(1)) == 0

    # Search on empty index
    result = idx.search(np.ones(8, dtype=np.float32), count=5)
    assert len(result) == 0
    assert result.keys.dtype == UUID_DTYPE


def test_large_key_values(tmp_path):
    """Full 128-bit range key (all 0xFF) works correctly."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    max_key = b"\xff" * 16
    min_key = b"\x00" * 16
    vecs = random_vectors(2, ndim=8)
    idx.add(max_key, vecs[0])
    idx.add(min_key, vecs[1])

    assert idx.contains(max_key)
    assert idx.contains(min_key)
    assert_array_equal(idx.get(max_key), vecs[0])
    assert_array_equal(idx.get(min_key), vecs[1])


def test_composite_key_roundtrip(tmp_path):
    """Pack body(8B) + chunk_idx(8B) big-endian, retrieve and unpack."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")

    iscc_body = 0xDEADBEEFCAFEBABE
    chunk_idx = 42
    composite_key = struct.pack(">QQ", iscc_body, chunk_idx)
    assert len(composite_key) == 16

    vec = np.arange(8, dtype=np.float32)
    idx.add(composite_key, vec)

    # Retrieve and unpack
    result = idx.get(composite_key)
    assert result is not None
    assert_array_equal(result, vec)

    # Verify key decomposition via search
    matches = idx.search(vec, count=1)
    recovered_key = bytes(matches.keys[0])
    body_out, chunk_out = struct.unpack(">QQ", recovered_key)
    assert body_out == iscc_body
    assert chunk_out == chunk_idx


def test_bloom_rebuild_with_uuid_keys(tmp_path):
    """Bloom rebuild populates filter from uuid keys."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32", bloom_filter=False)
    keys = make_keys(20)
    idx.add(keys, random_vectors(20, ndim=8))

    assert idx._bloom is None

    count = idx.rebuild_bloom(save=False, log_progress=False)
    assert count == 20
    assert idx._bloom is not None

    # Bloom now works for rejection
    assert idx.contains(make_key(0)) is True
    assert idx.contains(make_key(999)) is False


def test_empty_batch_operations(tmp_path):
    """Empty batch operations return appropriate empty results."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    idx.add(make_key(1), np.ones(8, dtype=np.float32))

    empty_keys = np.array([], dtype=UUID_DTYPE)
    assert idx.get(empty_keys) == []
    assert_array_equal(idx.contains(empty_keys), np.array([], dtype=bool))
    assert_array_equal(idx.count(empty_keys), np.array([], dtype=np.uint64))


def test_save_load_with_bloom_roundtrip(tmp_path):
    """Save/load preserves bloom filter with uuid keys."""
    path = tmp_path / "idx"
    idx = ShardedIndex128(ndim=8, path=path, dtype="f32")
    keys = make_keys(10)
    idx.add(keys, random_vectors(10, ndim=8))
    idx.save()

    loaded = ShardedIndex128(ndim=8, path=path, dtype="f32")
    # Bloom filter should have been persisted and reloaded
    assert loaded._bloom is not None
    assert loaded.contains(make_key(0)) is True
    assert loaded.contains(make_key(999)) is False


def test_keys_numpy_conversion_empty_index(tmp_path):
    """np.asarray(idx.keys) on empty index returns empty V16 array."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    arr = np.asarray(idx.keys)
    assert arr.dtype == UUID_DTYPE
    assert len(arr) == 0


def test_upsert_guard_uuid_index():
    """upsert() raises NotImplementedError on uuid-keyed Index."""
    idx = Index(ndim=8, dtype="f32", key_kind="uuid")
    vec = np.ones(8, dtype=np.float32)
    with pytest.raises(NotImplementedError, match="128-bit UUID keys"):
        idx.upsert(b"\x01" * 16, vec)


def test_size_and_keys_after_rotation(tmp_path):
    """len(idx) and len(idx.keys) are correct after shard rotation."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    vecs = random_vectors(5, ndim=8)
    idx.add(make_keys(5), vecs)
    assert len(idx) == 5

    idx._rotate_shard()
    assert len(idx._viewed_indexes) == 1
    assert len(idx._active_shard) == 0

    # Size must still include viewed shard data
    assert len(idx) == 5
    assert len(idx.keys) == 5
    # Keys must be indexable on viewed-shard-only state
    _ = idx.keys[0]


def test_add_batch_v16_dtype_passthrough(tmp_path):
    """Batch add with correct V16 ndarray passes through validation."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = np.array([make_key(0), make_key(1)], dtype=UUID_DTYPE)
    vecs = random_vectors(2, ndim=8)
    result = idx.add(keys, vecs)
    assert len(idx) == 2
    assert result.dtype == UUID_DTYPE


def test_add_list_keys_raises(tmp_path):
    """List of bytes keys on add raises ValueError."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    with pytest.raises(ValueError, match="UUID keys must be bytes"):
        idx.add([make_key(0), make_key(1)], random_vectors(2, ndim=8))


def test_add_int_key_raises(tmp_path):
    """Integer key on add raises ValueError for uuid index."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    with pytest.raises(ValueError, match="UUID keys must be bytes"):
        idx.add(42, np.ones(8, dtype=np.float32))


def test_get_int_key_raises(tmp_path):
    """Integer key on get raises ValueError for uuid index."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    idx.add(make_key(1), np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="UUID keys must be bytes"):
        idx.get(42)


def test_contains_int_key_raises(tmp_path):
    """Integer key on contains raises ValueError for uuid index."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    idx.add(make_key(1), np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="UUID keys must be bytes"):
        idx.contains(42)


def test_count_int_key_raises(tmp_path):
    """Integer key on count raises ValueError for uuid index."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    idx.add(make_key(1), np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="UUID keys must be bytes"):
        idx.count(42)


def test_search_single_viewed_shard_path(tmp_path):
    """Search with exactly 1 viewed shard and empty active exercises single-result shortcut."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    vecs = random_vectors(5, ndim=8)
    idx.add(make_keys(5), vecs)

    # Force rotation: data moves to 1 viewed shard, active is empty
    idx._rotate_shard()
    assert len(idx._viewed_indexes) == 1
    assert len(idx._active_shard) == 0

    matches = idx.search(vecs[0], count=3)
    assert len(matches) > 0
    assert matches.keys.dtype == UUID_DTYPE


def test_search_view_shards_merge_multiple(tmp_path):
    """Search with 2+ viewed shards exercises multi-result merge path."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32", shard_size=500)

    vecs = random_vectors(80, ndim=8)
    for i in range(80):
        idx.add(make_key(i), vecs[i])

    assert idx.shard_count >= 3, f"Expected >= 3 shards, got {idx.shard_count}"

    matches = idx.search(vecs[0], count=5)
    assert len(matches) > 0
