"""Tests for ShardedNphdIndex - sharding-specific behavior with NPHD metric."""

import numpy as np
import pytest

from iscc_usearch import ShardedNphdIndex


def test_init_creates_empty_index(tmp_path):
    """Initialize empty ShardedNphdIndex."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")

    assert len(idx) == 0
    assert idx.max_dim == 256
    assert idx.max_bytes == 32
    assert idx.shard_count == 0


def test_add_and_search_basic(tmp_path):
    """Basic add and search with NPHD vectors."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")

    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    v2 = np.array([1, 2, 3, 5], dtype=np.uint8)  # Similar to v1
    v3 = np.array([255, 254, 253, 252], dtype=np.uint8)  # Different

    idx.add(1, v1)
    idx.add(2, v2)
    idx.add(3, v3)

    result = idx.search(v1, count=3)

    assert result.keys[0] == 1  # Exact match
    assert result.distances[0] == 0.0


def test_add_variable_length_vectors(tmp_path):
    """Add and search variable-length vectors."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")

    v1 = np.array([1, 2], dtype=np.uint8)  # 2 bytes
    v2 = np.array([255, 255, 255, 255, 255, 255], dtype=np.uint8)  # 6 bytes, different
    v3 = np.array([128, 128, 128, 128, 128, 128, 128, 128, 128, 128], dtype=np.uint8)  # 10 bytes, different

    idx.add([1, 2, 3], [v1, v2, v3])

    # Search with short vector - v1 should be exact match (distance 0)
    result = idx.search(v1, count=3)
    assert result.keys[0] == 1
    assert result.distances[0] == 0.0


def test_shard_rotation(tmp_path):
    """Shard rotates when size limit exceeded."""
    # Small shard size to trigger rotation
    idx = ShardedNphdIndex(max_dim=64, path=tmp_path / "shards", shard_size=500)

    # Add enough vectors to trigger rotation
    for i in range(100):
        vector = np.random.randint(0, 256, 8, dtype=np.uint8)
        idx.add(i, vector)

    assert idx.shard_count >= 2  # Should have rotated at least once


def test_search_across_shards(tmp_path):
    """Search finds vectors across multiple shards."""
    idx = ShardedNphdIndex(max_dim=64, path=tmp_path / "shards", shard_size=500)

    # Add vectors to trigger rotation
    vectors = []
    for i in range(100):
        vector = np.random.randint(0, 256, 8, dtype=np.uint8)
        vectors.append(vector)
        idx.add(i, vector)

    assert idx.shard_count >= 2

    # Search for first vector (should be in view shards)
    result = idx.search(vectors[0], count=1)
    assert result.keys[0] == 0
    assert result.distances[0] == 0.0

    # Search for last vector (should be in active shard)
    result = idx.search(vectors[-1], count=1)
    assert result.keys[0] == 99
    assert result.distances[0] == 0.0


def test_save_and_load(tmp_path):
    """Save and load preserves data and NPHD metric."""
    path = tmp_path / "shards"
    idx = ShardedNphdIndex(max_dim=256, path=path)

    # Add vectors with known small NPHD distance
    v1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    v2 = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=np.uint8)  # 1 bit different
    idx.add([100, 200], [v1, v2])

    # Get distances before save
    results_before = idx.search(v1, count=2)
    distances_before = results_before.distances.copy()

    # Save
    idx.save()

    # Create new index and reload
    loaded = ShardedNphdIndex(max_dim=256, path=path)

    assert len(loaded) == 2
    assert loaded.max_dim == 256

    # Verify NPHD metric preserved
    results_after = loaded.search(v1, count=2)
    np.testing.assert_array_almost_equal(
        distances_before,
        results_after.distances,
        decimal=6,
        err_msg="NPHD metric not preserved after load",
    )


def test_repr(tmp_path):
    """String representation includes key info."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    repr_str = repr(idx)

    assert "ShardedNphdIndex" in repr_str
    assert "1 vectors" in repr_str
    assert "max_dim=256" in repr_str


def test_get_single_key(tmp_path):
    """Get retrieves single vector by key."""
    path = tmp_path / "shards"
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx = ShardedNphdIndex(max_dim=256, path=path)
    idx.add(1, v1)

    result = idx.get(1)

    np.testing.assert_array_equal(result, v1)


def test_get_multiple_keys_with_none_results(tmp_path):
    """Get multiple keys where some don't exist."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(1, v1)

    # Get both existing and non-existing keys
    result = idx.get([1, 999])

    assert len(result) == 2
    np.testing.assert_array_equal(result[0], v1)
    assert result[1] is None


def test_properties(tmp_path):
    """Properties return correct values."""
    idx = ShardedNphdIndex(max_dim=192, path=tmp_path / "shards")
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    assert idx.max_dim == 192
    assert idx.max_bytes == 24
    assert idx.ndim == 200  # max_dim + 8
    assert idx.size == 1
    assert len(idx) == 1


def test_init_without_max_dim_raises_when_no_shards(tmp_path):
    """Test that init without max_dim raises error when no existing shards."""
    with pytest.raises(ValueError, match="max_dim is required"):
        ShardedNphdIndex(path=tmp_path)


def test_init_without_max_dim_autodetects_from_existing_shards(tmp_path):
    """Test that init without max_dim auto-detects from existing shards."""
    path = tmp_path / "shards"

    # Create and save an index with known max_dim
    idx1 = ShardedNphdIndex(max_dim=192, path=path)
    v1 = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    idx1.add(1, v1)
    idx1.save()

    # Reopen without specifying max_dim - should auto-detect
    idx2 = ShardedNphdIndex(path=path)

    assert idx2.max_dim == 192
    assert idx2.max_bytes == 24
    assert len(idx2) == 1


def test_get_across_shards(tmp_path):
    """Get retrieves vectors from view shards after reload."""
    path = tmp_path / "shards"
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx = ShardedNphdIndex(max_dim=256, path=path)
    idx.add(1, v1)
    idx.save()

    # Reload index - data is now in view shards
    idx2 = ShardedNphdIndex(max_dim=256, path=path)

    # get() should retrieve vectors from view shards
    result_single = idx2.get(1)
    result_multi = idx2.get([1, 999])

    np.testing.assert_array_equal(result_single, v1)
    assert len(result_multi) == 2
    np.testing.assert_array_equal(result_multi[0], v1)
    assert result_multi[1] is None


# Tests for ShardedNphdIndex.vectors property


def test_vectors_returns_unpadded_vectors(tmp_path):
    """Vectors property returns unpadded variable-length vectors."""
    from iscc_usearch.sharded_nphd import ShardedNphdIndexedVectors

    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)  # 4 bytes
    v2 = np.array([10, 20], dtype=np.uint8)  # 2 bytes
    v3 = np.array([100, 101, 102, 103, 104, 105], dtype=np.uint8)  # 6 bytes
    idx.add(1, v1)
    idx.add(2, v2)
    idx.add(3, v3)

    vectors = idx.vectors

    assert isinstance(vectors, ShardedNphdIndexedVectors)
    assert len(vectors) == 3

    vectors_list = list(vectors)
    # Check each vector is unpadded (original length)
    lengths = sorted([len(v) for v in vectors_list])
    assert lengths == [2, 4, 6]


def test_vectors_iteration_returns_original_vectors(tmp_path):
    """Vectors iteration returns original unpadded vectors."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    v2 = np.array([10, 20], dtype=np.uint8)
    idx.add(1, v1)
    idx.add(2, v2)

    vectors_list = list(idx.vectors)

    # Should find both original vectors
    found_v1 = any(np.array_equal(v, v1) for v in vectors_list)
    found_v2 = any(np.array_equal(v, v2) for v in vectors_list)
    assert found_v1
    assert found_v2


def test_vectors_indexing_returns_unpadded(tmp_path):
    """Vectors indexing returns unpadded vectors."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(1, v1)

    vec = idx.vectors[0]

    assert len(vec) == 4
    np.testing.assert_array_equal(vec, v1)


def test_vectors_slicing_returns_list(tmp_path):
    """Vectors slicing returns list of unpadded vectors."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    for i in range(5):
        idx.add(i, np.array([i, i + 1, i + 2], dtype=np.uint8))

    sliced = idx.vectors[:3]

    assert isinstance(sliced, list)
    assert len(sliced) == 3
    for v in sliced:
        assert len(v) == 3  # All vectors have same length


def test_vectors_numpy_conversion_uniform_length(tmp_path):
    """Vectors converts to numpy array when all vectors have same length."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    for i in range(5):
        idx.add(i, np.array([i, i + 1, i + 2, i + 3], dtype=np.uint8))

    vectors_array = np.asarray(idx.vectors)

    assert isinstance(vectors_array, np.ndarray)
    assert vectors_array.shape == (5, 4)


def test_vectors_numpy_conversion_variable_length_raises(tmp_path):
    """Vectors numpy conversion raises when vectors have different lengths."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))  # 4 bytes
    idx.add(2, np.array([10, 20], dtype=np.uint8))  # 2 bytes

    with pytest.raises(ValueError, match="different lengths"):
        np.asarray(idx.vectors)


def test_vectors_consistent_with_get(tmp_path):
    """Vectors returns same data as get() for each key."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    v2 = np.array([10, 20, 30], dtype=np.uint8)
    idx.add(100, v1)
    idx.add(200, v2)

    # Get vectors from both APIs
    vec_from_get_1 = idx.get(100)
    vec_from_get_2 = idx.get(200)

    vectors_list = list(idx.vectors)

    # Each vector from vectors should match one from get
    assert any(np.array_equal(v, vec_from_get_1) for v in vectors_list)
    assert any(np.array_equal(v, vec_from_get_2) for v in vectors_list)


def test_vectors_repr(tmp_path):
    """Vectors has useful repr."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))
    idx.add(2, np.array([10, 20], dtype=np.uint8))

    repr_str = repr(idx.vectors)

    assert "ShardedNphdIndexedVectors" in repr_str
    assert "count=2" in repr_str
