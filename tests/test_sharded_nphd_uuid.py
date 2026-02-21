"""Tests for ShardedNphdIndex128 — NPHD metric with 128-bit UUID keys."""

import struct

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from iscc_usearch.sharded import UUID_DTYPE
from iscc_usearch.sharded_nphd import ShardedNphdIndex128


# --- Helpers ---


def make_key(i: int) -> bytes:
    """Create a deterministic 16-byte key from an integer."""
    return i.to_bytes(16, "big")


def make_keys(n: int, offset: int = 0) -> np.ndarray:
    """Create a V16 array of n deterministic keys."""
    return np.array([make_key(i + offset) for i in range(n)], dtype=UUID_DTYPE)


# === Core Operations ===


def test_add_single_and_search(tmp_path):
    """Add a single variable-length vector with bytes key, search returns it."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    key = make_key(1)
    vec = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(key, vec)

    assert len(idx) == 1

    matches = idx.search(vec, count=1)
    assert matches.keys.dtype == UUID_DTYPE
    assert bytes(matches.keys[0]) == key
    assert matches.distances[0] == 0.0


def test_add_batch_and_search(tmp_path):
    """Add a batch of vectors with V16 keys, search returns correct results."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    keys = make_keys(3)
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    v2 = np.array([1, 2, 3, 5], dtype=np.uint8)
    v3 = np.array([255, 254, 253, 252], dtype=np.uint8)
    idx.add(keys, [v1, v2, v3])

    assert len(idx) == 3

    matches = idx.search(v1, count=3)
    assert matches.keys.dtype == UUID_DTYPE
    # Exact match should be first
    assert bytes(matches.keys[0]) == make_key(0)
    assert matches.distances[0] == 0.0


def test_variable_length_vectors_with_uuid_keys(tmp_path):
    """Variable-length vectors work with 128-bit keys."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    v_short = np.array([1, 2], dtype=np.uint8)
    v_medium = np.array([200, 201, 202, 203, 204, 205], dtype=np.uint8)
    v_long = np.array([128] * 10, dtype=np.uint8)

    keys = make_keys(3)
    idx.add(keys, [v_short, v_medium, v_long])

    # Search with short vector — exact match
    matches = idx.search(v_short, count=3)
    assert bytes(matches.keys[0]) == make_key(0)
    assert matches.distances[0] == 0.0


def test_get_single_returns_unpadded(tmp_path):
    """Get single key returns unpadded vector."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    key = make_key(7)
    vec = np.array([10, 20, 30], dtype=np.uint8)
    idx.add(key, vec)

    result = idx.get(key)
    assert result is not None
    assert_array_equal(result, vec)


def test_get_single_missing(tmp_path):
    """Get returns None for missing key."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    idx.add(make_key(1), np.array([1, 2, 3], dtype=np.uint8))
    assert idx.get(make_key(999)) is None


def test_get_batch_returns_unpadded(tmp_path):
    """Get batch returns unpadded vectors with None for missing keys."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    v2 = np.array([10, 20], dtype=np.uint8)
    idx.add(make_key(1), v1)
    idx.add(make_key(2), v2)

    result = idx.get(make_keys(3))  # key 0 doesn't exist
    assert len(result) == 3
    assert result[0] is None
    assert_array_equal(result[1], v1)
    assert_array_equal(result[2], v2)


def test_contains_single(tmp_path):
    """Contains with single bytes key."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    idx.add(make_key(1), np.array([1, 2], dtype=np.uint8))
    assert idx.contains(make_key(1)) is True
    assert idx.contains(make_key(99)) is False


def test_contains_batch(tmp_path):
    """Contains with batch of keys."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    idx.add(make_key(1), np.array([1, 2], dtype=np.uint8))
    idx.add(make_key(2), np.array([3, 4], dtype=np.uint8))

    result = idx.contains(make_keys(3))
    assert_array_equal(result, [False, True, True])


def test_count_single(tmp_path):
    """Count with single bytes key."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    idx.add(make_key(1), np.array([1, 2], dtype=np.uint8))
    assert idx.count(make_key(1)) == 1
    assert idx.count(make_key(99)) == 0


def test_in_operator(tmp_path):
    """The 'in' operator works with bytes keys."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    idx.add(make_key(5), np.array([1, 2, 3], dtype=np.uint8))
    assert make_key(5) in idx
    assert make_key(99) not in idx


# === Persistence ===


def test_save_load_roundtrip(tmp_path):
    """Save and load preserves data, NPHD metric, and uuid key kind."""
    path = tmp_path / "idx"
    idx = ShardedNphdIndex128(max_dim=256, path=path)

    v1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    v2 = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=np.uint8)
    k1, k2 = make_key(100), make_key(200)
    idx.add(k1, v1)
    idx.add(k2, v2)

    distances_before = idx.search(v1, count=2).distances.copy()
    idx.save()

    # Reload
    loaded = ShardedNphdIndex128(max_dim=256, path=path)
    assert len(loaded) == 2
    assert loaded.max_dim == 256

    # NPHD metric preserved
    distances_after = loaded.search(v1, count=2).distances
    np.testing.assert_array_almost_equal(distances_before, distances_after, decimal=6)

    # Get returns unpadded vectors
    assert_array_equal(loaded.get(k1), v1)
    assert_array_equal(loaded.get(k2), v2)


def test_save_load_batch(tmp_path):
    """Save/load with batch add preserves all data."""
    path = tmp_path / "idx"
    idx = ShardedNphdIndex128(max_dim=256, path=path)
    keys = make_keys(5)
    vecs = [np.array([i, i + 1, i + 2, i + 3], dtype=np.uint8) for i in range(5)]
    idx.add(keys, vecs)
    idx.save()

    loaded = ShardedNphdIndex128(max_dim=256, path=path)
    assert len(loaded) == 5
    for i in range(5):
        result = loaded.get(make_key(i))
        assert result is not None
        assert_array_equal(result, vecs[i])


# === Shard Rotation ===


def test_shard_rotation(tmp_path):
    """Shard rotates and search works across shards."""
    idx = ShardedNphdIndex128(max_dim=64, path=tmp_path / "idx", shard_size=500)

    keys_added = []
    vecs_added = []
    for i in range(100):
        key = make_key(i)
        vec = np.random.RandomState(i).randint(0, 256, 8, dtype=np.uint8)
        idx.add(key, vec)
        keys_added.append(key)
        vecs_added.append(vec)

    assert idx.shard_count >= 2

    # Search for first vector (in view shards)
    matches = idx.search(vecs_added[0], count=1)
    assert bytes(matches.keys[0]) == keys_added[0]
    assert matches.distances[0] == 0.0

    # Search for last vector (in active shard)
    matches = idx.search(vecs_added[-1], count=1)
    assert bytes(matches.keys[0]) == keys_added[-1]
    assert matches.distances[0] == 0.0

    # Verify size and keys
    assert len(idx) == 100
    assert len(idx.keys) == 100


def test_get_across_shards_after_rotation(tmp_path):
    """Get retrieves unpadded vectors from both active and view shards."""
    idx = ShardedNphdIndex128(max_dim=64, path=tmp_path / "idx", shard_size=500)

    vecs = {}
    for i in range(100):
        key = make_key(i)
        vec = np.random.RandomState(i).randint(0, 256, 8, dtype=np.uint8)
        idx.add(key, vec)
        vecs[i] = vec

    assert idx.shard_count >= 2

    # Get from first entries (view shards) and last entries (active shard)
    result_first = idx.get(make_key(0))
    result_last = idx.get(make_key(99))
    assert result_first is not None
    assert result_last is not None
    assert_array_equal(result_first, vecs[0])
    assert_array_equal(result_last, vecs[99])


def test_contains_across_shards(tmp_path):
    """Contains works across shards after rotation."""
    idx = ShardedNphdIndex128(max_dim=64, path=tmp_path / "idx", shard_size=500)

    for i in range(100):
        idx.add(make_key(i), np.random.RandomState(i).randint(0, 256, 8, dtype=np.uint8))

    assert idx.shard_count >= 2
    assert idx.contains(make_key(0)) is True
    assert idx.contains(make_key(99)) is True
    assert idx.contains(make_key(999)) is False


# === Validation ===


def test_add_keys_none_raises(tmp_path):
    """Auto-key generation raises ValueError."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    with pytest.raises(ValueError, match="Auto-key generation not supported"):
        idx.add(None, np.array([1, 2], dtype=np.uint8))


def test_add_wrong_key_length_raises(tmp_path):
    """Wrong key length raises ValueError on add."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    with pytest.raises(ValueError, match="exactly 16 bytes"):
        idx.add(b"short", np.array([1, 2], dtype=np.uint8))


def test_add_int_key_raises(tmp_path):
    """Int key raises ValueError on uuid index."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    with pytest.raises(ValueError, match="bytes"):
        idx.add(42, np.array([1, 2], dtype=np.uint8))


def test_get_wrong_key_type_raises(tmp_path):
    """Wrong key type on get raises ValueError."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    idx.add(make_key(1), np.array([1, 2], dtype=np.uint8))
    with pytest.raises(ValueError, match="bytes of length 16"):
        idx.get(b"short")


def test_contains_wrong_key_type_raises(tmp_path):
    """Wrong key type on contains raises ValueError."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    idx.add(make_key(1), np.array([1, 2], dtype=np.uint8))
    with pytest.raises(ValueError, match="bytes of length 16"):
        idx.contains(b"short")


def test_count_wrong_key_type_raises(tmp_path):
    """Wrong key type on count raises ValueError."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    idx.add(make_key(1), np.array([1, 2], dtype=np.uint8))
    with pytest.raises(ValueError, match="bytes of length 16"):
        idx.count(b"short")


# === Edge Cases ===


def test_empty_index(tmp_path):
    """Empty uuid NPHD index behaves correctly."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    assert len(idx) == 0
    assert idx.get(make_key(1)) is None
    assert idx.contains(make_key(1)) is False
    assert idx.count(make_key(1)) == 0


def test_composite_key_roundtrip(tmp_path):
    """Pack body(8B) + chunk_idx(8B) as composite key, retrieve and unpack."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")

    body = b"\xde\xad\xbe\xef\x01\x02\x03\x04"
    chunk_idx = 42
    key = body + struct.pack(">Q", chunk_idx)
    assert len(key) == 16

    vec = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(key, vec)

    # Retrieve and unpack
    result = idx.get(key)
    assert result is not None
    assert_array_equal(result, vec)

    # Verify key structure in search results
    matches = idx.search(vec, count=1)
    returned_key = bytes(matches.keys[0])
    returned_body = returned_key[:8]
    returned_chunk = struct.unpack(">Q", returned_key[8:])[0]
    assert returned_body == body
    assert returned_chunk == chunk_idx


def test_large_key_value(tmp_path):
    """Full 128-bit range key (0xFF * 16) works."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    key = b"\xff" * 16
    vec = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(key, vec)

    assert idx.contains(key) is True
    assert_array_equal(idx.get(key), vec)


def test_repr(tmp_path):
    """String representation includes key info."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    idx.add(make_key(1), np.array([1, 2, 3, 4], dtype=np.uint8))
    repr_str = repr(idx)
    assert "ShardedNphdIndex128" in repr_str
    assert "1 vectors" in repr_str
    assert "max_dim=256" in repr_str


def test_bloom_rebuild(tmp_path):
    """Bloom rebuild works with uuid NPHD keys."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    for i in range(10):
        idx.add(make_key(i), np.array([i, i + 1, i + 2], dtype=np.uint8))

    count = idx.rebuild_bloom(save=False, log_progress=False)
    assert count == 10

    # Bloom filter works after rebuild
    assert idx.contains(make_key(0)) is True
    assert idx.contains(make_key(999)) is False


def test_batch_search(tmp_path):
    """Batch search returns BatchMatches with V16 keys."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    keys = make_keys(5)
    vecs = [np.array([i, i + 1, i + 2, i + 3], dtype=np.uint8) for i in range(5)]
    idx.add(keys, vecs)

    query = np.array([vecs[0], vecs[1]])
    batch = idx.search(query, count=3)
    assert batch.keys.dtype == UUID_DTYPE
    assert batch.keys.shape[0] == 2


def test_max_dim_autodetect_on_reload(tmp_path):
    """Reloading without max_dim auto-detects from existing shards."""
    path = tmp_path / "idx"
    idx = ShardedNphdIndex128(max_dim=192, path=path)
    idx.add(make_key(1), np.array([1, 2, 3], dtype=np.uint8))
    idx.save()

    loaded = ShardedNphdIndex128(path=path)
    assert loaded.max_dim == 192
    assert loaded.max_bytes == 24
    assert len(loaded) == 1


def test_key_kind_kwarg_absorbed(tmp_path):
    """Passing key_kind kwarg is silently absorbed (not passed to base)."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx", key_kind="uuid")
    idx.add(make_key(1), np.array([1, 2], dtype=np.uint8))
    assert len(idx) == 1


def test_uuid_on_uint64_shards_recovers_empty(tmp_path):
    """Opening uuid index on uint64 shards recovers gracefully with size=0."""
    from iscc_usearch import ShardedNphdIndex

    path = tmp_path / "idx"
    idx = ShardedNphdIndex(max_dim=256, path=path)
    idx.add(1, np.array([1, 2, 3], dtype=np.uint8))
    idx.save()

    # Key kind mismatch is treated as corruption — index opens with size=0
    idx2 = ShardedNphdIndex128(max_dim=256, path=path)
    assert len(idx2) == 0


# === Vectors Property ===


def test_vectors_iteration(tmp_path):
    """Vectors iteration returns unpadded vectors via key-based retrieval."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    v2 = np.array([10, 20], dtype=np.uint8)
    idx.add(make_key(1), v1)
    idx.add(make_key(2), v2)

    vectors_list = list(idx.vectors)
    assert len(vectors_list) == 2
    lengths = sorted(len(v) for v in vectors_list)
    assert lengths == [2, 4]


def test_vectors_indexing(tmp_path):
    """Vectors indexing returns unpadded vector."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    vec = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(make_key(1), vec)

    result = idx.vectors[0]
    assert len(result) == 4
    assert_array_equal(result, vec)


def test_vectors_slicing(tmp_path):
    """Vectors slicing returns list of unpadded vectors."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    for i in range(5):
        idx.add(make_key(i), np.array([i, i + 1, i + 2], dtype=np.uint8))

    sliced = idx.vectors[:3]
    assert isinstance(sliced, list)
    assert len(sliced) == 3


def test_vectors_numpy_conversion(tmp_path):
    """Vectors converts to numpy array when all vectors have same length."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    for i in range(5):
        idx.add(make_key(i), np.array([i, i + 1, i + 2, i + 3], dtype=np.uint8))

    arr = np.asarray(idx.vectors)
    assert arr.shape == (5, 4)


def test_vectors_len(tmp_path):
    """Vectors len matches index size."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    idx.add(make_key(1), np.array([1, 2], dtype=np.uint8))
    idx.add(make_key(2), np.array([3, 4], dtype=np.uint8))
    assert len(idx.vectors) == 2
