"""Tests for ShardedNphdIndexedVectors edge cases for complete coverage.

Tests edge cases and branches not covered by test_sharded_nphd.py:
- Iteration with view shards
- Negative indexing edge cases
- Empty vectors array conversion
- dtype conversion in array
"""

import numpy as np
import pytest

from iscc_usearch import ShardedNphdIndex


def test_vectors_iteration_with_view_shards(tmp_path):
    # type: () -> None
    """Test ShardedNphdIndexedVectors iteration includes view shards."""
    path = tmp_path / "shards"
    idx = ShardedNphdIndex(max_dim=256, path=path, shard_size=100)

    # Add enough vectors to trigger rotation and create view shards
    vectors = []
    for i in range(50):
        v = np.array([i, i + 1, i + 2], dtype=np.uint8)
        vectors.append(v)
        idx.add(i, v)

    idx.save()

    # Reload to have view shards
    idx2 = ShardedNphdIndex(max_dim=256, path=path)
    assert len(idx2._viewed_indexes) > 0

    # Iterate through vectors - should include view shards
    vectors_list = list(idx2.vectors)

    # Should have all vectors
    assert len(vectors_list) == 50


def test_vectors_getitem_negative_index(tmp_path):
    # type: () -> None
    """Test ShardedNphdIndexedVectors negative indexing."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    v2 = np.array([5, 6, 7, 8], dtype=np.uint8)
    v3 = np.array([9, 10, 11, 12], dtype=np.uint8)
    idx.add(1, v1)
    idx.add(2, v2)
    idx.add(3, v3)

    # Test negative indexing
    last = idx.vectors[-1]
    second_last = idx.vectors[-2]

    assert len(last) == 4
    assert len(second_last) == 4
    # Last vector should be v3
    np.testing.assert_array_equal(last, v3)


def test_vectors_getitem_negative_index_out_of_range(tmp_path):
    # type: () -> None
    """Test ShardedNphdIndexedVectors negative index out of range raises."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    idx.add(1, np.array([1, 2, 3], dtype=np.uint8))

    # Test out of range negative index
    try:
        _ = idx.vectors[-10]
        assert False, "Should have raised IndexError"
    except IndexError as e:
        assert "out of range" in str(e)


def test_vectors_getitem_from_view_shards(tmp_path):
    # type: () -> None
    """Test ShardedNphdIndexedVectors indexing retrieves from view shards."""
    path = tmp_path / "shards"
    idx = ShardedNphdIndex(max_dim=256, path=path, shard_size=100)

    # Add vectors and save to create view shards
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    v2 = np.array([5, 6, 7, 8], dtype=np.uint8)

    for i in range(30):
        idx.add(i, np.array([i, i + 1], dtype=np.uint8))

    idx.add(100, v1)
    idx.add(101, v2)
    idx.save()

    # Reload to have view shards
    idx2 = ShardedNphdIndex(max_dim=256, path=path)
    assert len(idx2._viewed_indexes) > 0

    # Active shard vectors come first, then view shards newest-to-oldest.
    # Verify both active and view shard vectors are accessible.
    vec0 = idx2.vectors[0]
    assert len(vec0) in (2, 4)

    # Ensure view shard 2-byte vectors are accessible via iteration
    all_vecs = list(idx2.vectors)
    lengths = {len(v) for v in all_vecs}
    assert 2 in lengths, "2-byte vectors from view shards should be accessible"


def test_vectors_array_empty_index(tmp_path):
    # type: () -> None
    """Test ShardedNphdIndexedVectors __array__ with empty index."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")

    # Convert empty vectors to array
    arr = np.asarray(idx.vectors)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (0,)


def test_vectors_array_dtype_conversion(tmp_path):
    # type: () -> None
    """Test ShardedNphdIndexedVectors __array__ with dtype conversion."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    # Add vectors with same length for array conversion
    for i in range(3):
        idx.add(i, np.array([i, i + 1, i + 2, i + 3], dtype=np.uint8))

    # Convert to array with different dtype
    arr = np.asarray(idx.vectors, dtype=np.uint16)

    assert arr.dtype == np.uint16
    assert arr.shape == (3, 4)


def test_vectors_getitem_fast_path_read_only(tmp_path):
    # type: () -> None
    """NPHD vectors fast path negative indexing and bounds check (read-only)."""
    path = tmp_path / "shards"
    idx = ShardedNphdIndex(max_dim=256, path=path, shard_size=100)
    for i in range(10):
        idx.add(i, np.array([i, i + 1, i + 2], dtype=np.uint8))
    idx.save()

    ro = ShardedNphdIndex(max_dim=256, path=path, read_only=True)
    assert ro._active_shard is None
    assert not ro._tombstones

    # Fast path negative indexing
    vec = ro.vectors[-1]
    assert len(vec) == 3

    # Fast path out-of-range
    try:
        _ = ro.vectors[999]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass


def test_vectors_getitem_slow_path_out_of_range(tmp_path):
    # type: () -> None
    """NPHD vectors slow path raises IndexError for out-of-range positive index."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    idx.add(1, np.array([1, 2, 3], dtype=np.uint8))
    idx.add(2, np.array([4, 5, 6], dtype=np.uint8))

    # Active shard present -> slow path; positive index beyond actual count
    try:
        _ = idx.vectors[999]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass


def test_vectors_iteration_with_no_active_shard(tmp_path):
    # type: () -> None
    """Test ShardedNphdIndexedVectors iteration with active_shard = None."""
    path = tmp_path / "shards"
    idx = ShardedNphdIndex(max_dim=256, path=path, shard_size=100)

    # Add vectors and save to create view shards
    for i in range(10):
        idx.add(i, np.array([i, i + 1, i + 2], dtype=np.uint8))

    idx.save()

    # Set active_shard to None to test the exit branch
    idx._active_shard = None

    # Should still iterate over view shards
    vectors_list = list(idx.vectors)

    assert len(vectors_list) >= 10


def test_nphd_vectors_iter_slow_path_with_active_shard(tmp_path):
    # type: () -> None
    """NPHD vectors iterator dedup path with active shard entries."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    idx.add(0, np.array([1, 2, 3], dtype=np.uint8))
    idx.add(1, np.array([4, 5, 6], dtype=np.uint8))
    idx._rotate_shard()
    idx.upsert(0, np.array([7, 8, 9], dtype=np.uint8))
    assert idx._needs_compact

    vecs = list(idx.vectors)
    assert len(vecs) == 2


def test_nphd_vectors_getitem_slow_path_needs_compact(tmp_path):
    # type: () -> None
    """NPHD vectors getitem slow path handles negative and out-of-range indexes."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    idx.add(0, np.array([1, 2, 3], dtype=np.uint8))
    idx.add(1, np.array([4, 5, 6], dtype=np.uint8))
    idx._rotate_shard()
    idx.upsert(0, np.array([7, 8, 9], dtype=np.uint8))
    assert idx._needs_compact

    vec = idx.vectors[0]
    assert vec is not None

    vec_neg = idx.vectors[-1]
    assert vec_neg is not None

    with pytest.raises(IndexError):
        _ = idx.vectors[-100]

    with pytest.raises(IndexError):
        _ = idx.vectors[100]
