"""
Test CRUD operations for ShardedNphdIndex (variable-length vectors).

Confirms remove, upsert, and compaction work correctly with
variable-length binary bit-vectors and NPHD padding.
"""

import numpy as np
import pytest

from iscc_usearch.sharded_nphd import ShardedNphdIndex


def test_remove_nphd(tmp_path):
    """Remove key, verify get returns None."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path)
    vec = np.random.randint(0, 256, size=16, dtype=np.uint8)
    index.add(1, vec)

    # Verify get returns unpadded vector before remove
    result = index.get(1)
    assert result is not None
    assert len(result) == len(vec)

    index.remove(1)

    assert not index.contains(1)
    assert index.get(1) is None


def test_upsert_nphd(tmp_path):
    """Upsert with different-length vector, verify padding correct."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path)
    vec_short = np.random.randint(0, 256, size=8, dtype=np.uint8)
    vec_long = np.random.randint(0, 256, size=24, dtype=np.uint8)

    index.add(1, vec_short)
    index.upsert(1, vec_long)

    result = index.get(1)
    assert result is not None
    assert len(result) == len(vec_long)
    assert np.array_equal(result, vec_long)


def test_upsert_nphd_batch(tmp_path):
    """Batch upsert with NPHD vectors."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path)
    # Add initial vectors
    for i in range(3):
        index.add(i, np.random.randint(0, 256, size=16, dtype=np.uint8))

    # Upsert with new vectors
    new_vecs = [np.random.randint(0, 256, size=16, dtype=np.uint8) for _ in range(3)]
    index.upsert([0, 1, 2], new_vecs)

    for i, expected in enumerate(new_vecs):
        result = index.get(i)
        assert result is not None
        assert np.array_equal(result, expected)


def test_compact_nphd(tmp_path):
    """Compaction preserves correct padded/unpadded semantics."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path, shard_size=1)
    vecs = {}
    for i in range(4):
        v = np.random.randint(0, 256, size=16, dtype=np.uint8)
        vecs[i] = v
        index.add(i, v)

    index.remove([0, 2])
    index.compact()

    # Verify remaining entries
    for key in [1, 3]:
        result = index.get(key)
        assert result is not None
        assert len(result) == len(vecs[key])
        assert np.array_equal(result, vecs[key])

    # Verify removed entries are gone
    assert index.get(0) is None
    assert index.get(2) is None


def test_remove_nphd_view_shard(tmp_path):
    """Remove from view shard in NPHD index."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path, shard_size=1)
    vec = np.random.randint(0, 256, size=16, dtype=np.uint8)
    index.add(1, vec)
    index.add(2, np.random.randint(0, 256, size=16, dtype=np.uint8))

    # Key 1 is in a view shard after rotation
    index.remove(1)

    assert not index.contains(1)
    assert index.tombstone_count >= 1


def test_nphd_vectors_iterator_after_remove(tmp_path):
    """Vectors iterator excludes removed entries for NPHD."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.randint(0, 256, size=16, dtype=np.uint8))

    index.remove(1)

    keys_list = list(index.keys)
    vecs_list = list(index.vectors)
    assert len(keys_list) == len(vecs_list)
    assert 1 not in keys_list


def test_nphd_vectors_getitem_fast_path_read_only(tmp_path):
    """NPHD vectors __getitem__ fast path in read-only mode."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.randint(0, 256, size=16, dtype=np.uint8))
    index.save()

    ro = ShardedNphdIndex(max_dim=256, path=tmp_path, read_only=True)
    vec = ro.vectors[0]
    assert vec is not None
    assert len(vec) == 16


def test_nphd_vectors_iter_fast_path_read_only(tmp_path):
    """NPHD vectors __iter__ fast path in read-only mode."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.randint(0, 256, size=16, dtype=np.uint8))
    index.save()

    ro = ShardedNphdIndex(max_dim=256, path=tmp_path, read_only=True)
    vecs = list(ro.vectors)
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 16


def test_nphd_upsert_multi_raises(tmp_path):
    """NPHD upsert raises ValueError when multi=True."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path, multi=True)
    with pytest.raises(ValueError, match="multi=False"):
        index.upsert(1, np.random.randint(0, 256, size=16, dtype=np.uint8))


def test_nphd_upsert_none_key_raises(tmp_path):
    """NPHD upsert raises ValueError when keys is None."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path)
    with pytest.raises(ValueError, match="requires explicit keys"):
        index.upsert(None, np.random.randint(0, 256, size=16, dtype=np.uint8))


def test_nphd_upsert_batch_dedup(tmp_path):
    """NPHD batch upsert deduplicates (last wins) with variable-length vecs."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path)
    vec_a = np.array([1, 2, 3, 4], dtype=np.uint8)
    vec_b = np.array([5, 6, 7, 8], dtype=np.uint8)

    index.upsert([1, 1], [vec_a, vec_b])

    result = index.get(1)
    assert result is not None
    assert np.array_equal(result, vec_b)


# --- NPHD vectors iterator: slow path with no active shard ---


def test_nphd_vectors_iter_slow_path_no_active(tmp_path):
    """NPHD vectors __iter__ slow path with tombstones, no active shard (66->72)."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.randint(0, 256, size=16, dtype=np.uint8))
    index.remove(1)
    index.save()

    # Reload read-only: has tombstones, no active shard
    ro = ShardedNphdIndex(max_dim=256, path=tmp_path, read_only=True)
    assert ro._active_shard is None
    assert ro._tombstones

    vecs = list(ro.vectors)
    assert len(vecs) == 2


# --- NPHD vectors __getitem__: fast path multi-shard ---


def test_nphd_vectors_getitem_fast_path_multi_shard(tmp_path):
    """NPHD vectors __getitem__ fast path accesses second shard (line 102)."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path, shard_size=1)
    for i in range(3):
        index.add(i, np.random.randint(0, 256, size=16, dtype=np.uint8))
    index.save()

    ro = ShardedNphdIndex(max_dim=256, path=tmp_path, read_only=True)
    # Access item in second view shard
    vec = ro.vectors[1]
    assert vec is not None
    assert len(vec) == 16


# --- NPHD upsert batch with 2D ndarray vectors ---


def test_nphd_upsert_batch_2d_ndarray(tmp_path):
    """NPHD batch upsert with 2D ndarray vectors (lines 380, 391)."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path)
    for i in range(3):
        index.add(i, np.array([i, i + 1, i + 2, i + 3], dtype=np.uint8))

    # Upsert with 2D ndarray (uniform length) — hits line 380
    new_vecs = np.array([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]], dtype=np.uint8)
    index.upsert([0, 1, 2], new_vecs)

    for i, expected in enumerate(new_vecs):
        result = index.get(i)
        assert result is not None
        assert np.array_equal(result, expected)


def test_nphd_upsert_batch_2d_ndarray_dedup(tmp_path):
    """NPHD batch upsert dedup with 2D ndarray vectors (line 391)."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path)
    index.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    # Duplicate key with 2D ndarray — dedup should keep last, hitting line 391
    vecs = np.array([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=np.uint8)
    index.upsert([1, 1], vecs)

    result = index.get(1)
    assert result is not None
    assert np.array_equal(result, vecs[1])


# --- NPHD upsert with 1D vector in batch path ---


def test_nphd_upsert_single_1d_vector(tmp_path):
    """NPHD upsert with 1D ndarray vector (line 378)."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path)
    index.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    # Pass 1D ndarray as vectors with list of keys — triggers hasattr check (line 377-378)
    index.upsert([1], np.array([10, 11, 12, 13], dtype=np.uint8))

    result = index.get(1)
    assert result is not None
    assert np.array_equal(result, np.array([10, 11, 12, 13], dtype=np.uint8))


def test_nphd_upsert_batch_mismatched_lengths(tmp_path):
    """NPHD batch upsert raises ValueError when keys/vectors count differs."""
    index = ShardedNphdIndex(max_dim=256, path=tmp_path)
    with pytest.raises(ValueError, match="must match"):
        index.upsert([1, 2, 3], [np.array([1, 2], dtype=np.uint8), np.array([3, 4], dtype=np.uint8)])
