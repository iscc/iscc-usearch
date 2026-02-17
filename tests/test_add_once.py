"""Tests for add_once() — skip-if-exists semantics on all sharded index variants."""

import numpy as np
import pytest

from iscc_usearch.sharded import ShardedIndex, ShardedIndex128, UUID_DTYPE
from iscc_usearch.sharded_nphd import ShardedNphdIndex, ShardedNphdIndex128


# --- Helpers ---


def make_key(i: int) -> bytes:
    """Create a deterministic 16-byte key from an integer."""
    return i.to_bytes(16, "big")


def make_keys(n: int, offset: int = 0) -> np.ndarray:
    """Create a V16 array of n deterministic keys."""
    return np.array([make_key(i + offset) for i in range(n)], dtype=UUID_DTYPE)


# === ShardedIndex (uint64 keys) ===


def test_sharded_add_once_single_new(tmp_path):
    """Single key that does not exist is added."""
    idx = ShardedIndex(ndim=8, path=tmp_path / "idx", dtype="f32")
    vec = np.ones(8, dtype=np.float32)
    result = idx.add_once(1, vec)
    assert result == 1
    assert len(idx) == 1


def test_sharded_add_once_single_skip(tmp_path):
    """Single key that already exists returns None and is not overwritten."""
    idx = ShardedIndex(ndim=8, path=tmp_path / "idx", dtype="f32")
    vec_a = np.ones(8, dtype=np.float32)
    vec_b = np.zeros(8, dtype=np.float32)
    idx.add(1, vec_a)
    result = idx.add_once(1, vec_b)
    assert result is None
    assert len(idx) == 1
    # Original vector preserved
    retrieved = idx.get(1)
    np.testing.assert_array_equal(retrieved, vec_a)


def test_sharded_add_once_batch_mixed(tmp_path):
    """Batch with some existing and some new keys — only new keys added."""
    idx = ShardedIndex(ndim=8, path=tmp_path / "idx", dtype="f32")
    rng = np.random.default_rng(42)
    # Pre-add keys 0, 1, 2
    vecs_existing = rng.random((3, 8), dtype=np.float32)
    idx.add(np.array([0, 1, 2], dtype=np.uint64), vecs_existing)
    assert len(idx) == 3
    # add_once with keys 1, 2, 3, 4 (1, 2 exist; 3, 4 new)
    vecs_batch = rng.random((4, 8), dtype=np.float32)
    result = idx.add_once(np.array([1, 2, 3, 4], dtype=np.uint64), vecs_batch)
    assert len(result) == 2
    assert set(result) == {3, 4}
    assert len(idx) == 5


def test_sharded_add_once_batch_all_existing(tmp_path):
    """Batch where all keys exist returns empty array."""
    idx = ShardedIndex(ndim=8, path=tmp_path / "idx", dtype="f32")
    vec = np.ones((2, 8), dtype=np.float32)
    idx.add(np.array([10, 20], dtype=np.uint64), vec)
    result = idx.add_once(np.array([10, 20], dtype=np.uint64), vec)
    assert len(result) == 0
    assert result.dtype == np.uint64


def test_sharded_add_once_batch_dedup(tmp_path):
    """Within-batch duplicates: first occurrence wins."""
    idx = ShardedIndex(ndim=8, path=tmp_path / "idx", dtype="f32")
    vec_first = np.ones(8, dtype=np.float32)
    vec_second = np.zeros(8, dtype=np.float32)
    vecs = np.stack([vec_first, vec_second])
    keys = np.array([42, 42], dtype=np.uint64)
    idx.add_once(keys, vecs)
    assert len(idx) == 1
    # First occurrence vector preserved
    retrieved = idx.get(42)
    np.testing.assert_array_equal(retrieved, vec_first)


def test_sharded_add_once_keys_none_raises(tmp_path):
    """keys=None raises ValueError."""
    idx = ShardedIndex(ndim=8, path=tmp_path / "idx", dtype="f32")
    vec = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError, match="add_once.*requires explicit keys"):
        idx.add_once(None, vec)


def test_sharded_add_once_length_mismatch_raises(tmp_path):
    """Mismatched keys/vectors lengths raises ValueError."""
    idx = ShardedIndex(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = np.array([1, 2, 3], dtype=np.uint64)
    vecs = np.ones((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="must match"):
        idx.add_once(keys, vecs)


# === ShardedNphdIndex (variable-length vectors) ===


def test_nphd_add_once_mixed_length_batch(tmp_path):
    """Batch of mixed-length vectors with some existing keys skipped."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "idx")
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(1, v1)
    # add_once: key 1 exists, key 2 is new
    v3 = np.array([7, 8, 9], dtype=np.uint8)
    result = idx.add_once(np.array([1, 2], dtype=np.uint64), [v1, v3])
    assert len(result) == 1
    assert len(result) == 1
    assert result[0] == 2
    assert len(idx) == 2


def test_nphd_add_once_single_skip(tmp_path):
    """Single existing key returns None for NPHD index."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "idx")
    vec = np.array([1, 2, 3], dtype=np.uint8)
    idx.add(1, vec)
    result = idx.add_once(1, vec)
    assert result is None
    assert len(idx) == 1


def test_nphd_add_once_batch_dedup(tmp_path):
    """Within-batch duplicates with mixed-length vectors: first occurrence wins."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "idx")
    v1 = np.array([1, 2], dtype=np.uint8)
    v2 = np.array([3, 4, 5, 6], dtype=np.uint8)
    idx.add_once(np.array([10, 10], dtype=np.uint64), [v1, v2])
    assert len(idx) == 1
    # Verify first vector was kept (retrieve and check length via unpadded)
    retrieved = idx.get(10)
    # First vector was [1, 2] — padded to max_dim but first byte encodes length
    assert retrieved is not None


def test_nphd_add_once_keys_none_raises(tmp_path):
    """keys=None raises ValueError for NPHD index."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "idx")
    vec = np.array([1, 2, 3], dtype=np.uint8)
    with pytest.raises(ValueError, match="add_once.*requires explicit keys"):
        idx.add_once(None, vec)


def test_nphd_add_once_length_mismatch_raises(tmp_path):
    """Mismatched keys/vectors lengths raises ValueError for NPHD index."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "idx")
    keys = np.array([1, 2], dtype=np.uint64)
    vecs = [np.array([1], dtype=np.uint8)]
    with pytest.raises(ValueError, match="must match"):
        idx.add_once(keys, vecs)


def test_nphd_add_once_1d_ndarray_normalized_as_single_vector(tmp_path):
    """1D ndarray in batch mode is treated as a single vector, matching add() behavior."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "idx")
    vec = np.array([1, 2, 3, 4], dtype=np.uint8)
    # One batch key + one 1D vector — should work (mismatch would fail without normalization)
    result = idx.add_once(np.array([1], dtype=np.uint64), vec)
    assert len(result) == 1
    assert len(idx) == 1


# === ShardedIndex128 (128-bit UUID keys) ===


def test_uuid_add_once_single_skip(tmp_path):
    """Single bytes(16) key that exists returns None."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    key = make_key(1)
    vec = np.ones(8, dtype=np.float32)
    idx.add(key, vec)
    result = idx.add_once(key, vec)
    assert result is None
    assert len(idx) == 1


def test_uuid_add_once_single_new(tmp_path):
    """Single bytes(16) key that does not exist is added."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    key = make_key(1)
    vec = np.ones(8, dtype=np.float32)
    result = idx.add_once(key, vec)
    assert len(idx) == 1
    assert bytes(result[0]) == key


def test_uuid_add_once_batch_mixed(tmp_path):
    """list[bytes] batch with some existing and some new keys."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    rng = np.random.default_rng(42)
    # Pre-add keys 0, 1
    existing_keys = [make_key(0), make_key(1)]
    existing_vecs = rng.random((2, 8), dtype=np.float32)
    idx.add(existing_keys, existing_vecs)
    assert len(idx) == 2
    # add_once: keys 1, 2 (1 exists, 2 new)
    batch_keys = [make_key(1), make_key(2)]
    batch_vecs = rng.random((2, 8), dtype=np.float32)
    result = idx.add_once(batch_keys, batch_vecs)
    assert len(result) == 1
    assert bytes(result[0]) == make_key(2)
    assert len(idx) == 3


def test_uuid_add_once_keys_none_raises(tmp_path):
    """keys=None raises ValueError with UUID-specific message."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    vec = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError, match="Auto-key generation not supported"):
        idx.add_once(None, vec)


def test_uuid_add_once_wrong_key_length_raises(tmp_path):
    """Wrong bytes length raises ValueError."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    vec = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError, match="exactly 16 bytes"):
        idx.add_once(b"short", vec)


def test_uuid_add_once_wrong_ndarray_dtype_raises(tmp_path):
    """ndarray with wrong dtype raises ValueError."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    keys = np.array([1, 2], dtype=np.int64)
    vecs = np.ones((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="dtype 'V16'"):
        idx.add_once(keys, vecs)


def test_uuid_add_once_v16_ndarray_batch(tmp_path):
    """V16 ndarray batch passes dtype check and adds correctly."""
    idx = ShardedIndex128(ndim=8, path=tmp_path / "idx", dtype="f32")
    rng = np.random.default_rng(42)
    keys = make_keys(3)
    vecs = rng.random((3, 8), dtype=np.float32)
    idx.add(keys[:1], vecs[:1])
    # add_once with V16 ndarray — key 0 exists, keys 1-2 new
    result = idx.add_once(keys, vecs)
    assert len(result) == 2
    assert len(idx) == 3


# === ShardedNphdIndex (uniform-length 2D ndarray batch) ===


def test_nphd_add_once_uniform_2d_batch(tmp_path):
    """2D ndarray batch (uniform-length vectors) exercises ndarray branch."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "idx")
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))
    # Uniform-length batch as 2D ndarray — key 1 exists, key 2 new
    vecs = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.uint8)
    result = idx.add_once(np.array([1, 2], dtype=np.uint64), vecs)
    assert len(result) == 1
    assert result[0] == 2
    assert len(idx) == 2


def test_nphd_add_once_uniform_2d_batch_dedup(tmp_path):
    """2D ndarray batch with within-batch duplicates exercises ndarray dedup branch."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "idx")
    vecs = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
    idx.add_once(np.array([10, 10], dtype=np.uint64), vecs)
    assert len(idx) == 1


def test_nphd_add_once_batch_all_existing(tmp_path):
    """NPHD batch where all keys exist returns empty array."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "idx")
    v1 = np.array([1, 2], dtype=np.uint8)
    v2 = np.array([3, 4, 5], dtype=np.uint8)
    idx.add(np.array([1, 2], dtype=np.uint64), [v1, v2])
    result = idx.add_once(np.array([1, 2], dtype=np.uint64), [v1, v2])
    assert len(result) == 0
    assert result.dtype == np.uint64


# === ShardedNphdIndex128 (UUID + variable-length vectors) ===


def test_uuid_nphd_add_once_batch(tmp_path):
    """128-bit keys with variable-length vectors — existing keys skipped."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    key_a = make_key(1)
    vec_a = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(key_a, vec_a)
    # add_once: key_a exists, key_b new
    key_b = make_key(2)
    vec_b = np.array([5, 6], dtype=np.uint8)
    result = idx.add_once([key_a, key_b], [vec_a, vec_b])
    assert len(result) == 1
    assert bytes(result[0]) == key_b
    assert len(idx) == 2


def test_uuid_nphd_add_once_single_new(tmp_path):
    """Single new UUID key with variable-length vector is added."""
    idx = ShardedNphdIndex128(max_dim=256, path=tmp_path / "idx")
    key = make_key(1)
    vec = np.array([10, 20, 30], dtype=np.uint8)
    result = idx.add_once(key, vec)
    assert len(idx) == 1
    assert bytes(result[0]) == key
