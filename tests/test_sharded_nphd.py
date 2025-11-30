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

    # Create new index and load
    loaded = ShardedNphdIndex(max_dim=256, path=path)
    loaded.load()

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


def test_view_mode(tmp_path):
    """View mode opens index read-only with NPHD metric."""
    path = tmp_path / "shards"
    idx = ShardedNphdIndex(max_dim=256, path=path)

    v1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    v2 = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=np.uint8)
    idx.add([100, 200], [v1, v2])

    distances_before = idx.search(v1, count=2).distances.copy()
    idx.save()

    # View mode
    viewed = ShardedNphdIndex(max_dim=256, path=path, view=True)

    assert len(viewed) == 2

    # Verify NPHD metric preserved
    distances_after = viewed.search(v1, count=2).distances
    np.testing.assert_array_almost_equal(
        distances_before,
        distances_after,
        decimal=6,
    )

    # View mode should reject writes
    with pytest.raises(RuntimeError):
        viewed.add(300, np.array([1, 2, 3, 4], dtype=np.uint8))


def test_restore_from_path(tmp_path):
    """Restore index from directory path."""
    path = tmp_path / "shards"
    idx = ShardedNphdIndex(max_dim=128, path=path)

    vector = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    idx.add(42, vector)
    idx.save()

    # Restore using static method
    restored = ShardedNphdIndex.restore(path)

    assert restored is not None
    assert len(restored) == 1
    assert restored.max_dim == 128
    assert 42 in restored


def test_restore_nonexistent_path_returns_none(tmp_path):
    """Restore from non-existent path returns None."""
    result = ShardedNphdIndex.restore(tmp_path / "nonexistent")
    assert result is None


def test_restore_empty_dir_returns_none(tmp_path):
    """Restore from empty directory returns None."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = ShardedNphdIndex.restore(empty_dir)
    assert result is None


def test_restore_file_not_dir_returns_none(tmp_path):
    """Restore from file (not directory) returns None."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("test")

    result = ShardedNphdIndex.restore(file_path)
    assert result is None


def test_restore_view_mode(tmp_path):
    """Restore in view mode."""
    path = tmp_path / "shards"
    idx = ShardedNphdIndex(max_dim=256, path=path)
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))
    idx.save()

    restored = ShardedNphdIndex.restore(path, view=True)

    assert restored is not None
    assert len(restored) == 1

    # View mode should reject writes
    with pytest.raises(RuntimeError):
        restored.add(2, np.array([5, 6, 7, 8], dtype=np.uint8))


def test_repr(tmp_path):
    """String representation includes key info."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path / "shards")
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    repr_str = repr(idx)

    assert "ShardedNphdIndex" in repr_str
    assert "1 vectors" in repr_str
    assert "max_dim=256" in repr_str


def test_get_on_view_only_index(tmp_path):
    """Get on view-only index retrieves vectors from view shards."""
    path = tmp_path / "shards"
    v1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx = ShardedNphdIndex(max_dim=256, path=path)
    idx.add(1, v1)
    idx.save()

    # Open in view mode - no active shard
    viewed = ShardedNphdIndex(max_dim=256, path=path, view=True)
    assert viewed._active_shard is None

    # get() on view-only should retrieve vectors from view shards
    result_single = viewed.get(1)
    result_multi = viewed.get([1, 999])

    np.testing.assert_array_equal(result_single, v1)
    assert len(result_multi) == 2
    np.testing.assert_array_equal(result_multi[0], v1)
    assert result_multi[1] is None


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


def test_load_syncs_max_dim(tmp_path):
    """Load correctly syncs max_dim from saved shard."""
    path = tmp_path / "shards"

    # Create with max_dim=192
    idx = ShardedNphdIndex(max_dim=192, path=path)
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))
    idx.save()

    # Create with different max_dim and load
    loaded = ShardedNphdIndex(max_dim=64, path=path)
    loaded.load()

    # max_dim should be synced from loaded shard
    assert loaded.max_dim == 192
    assert loaded.max_bytes == 24


def test_load_empty_directory(tmp_path):
    """Load from empty directory (no shards) keeps initial max_dim."""
    path = tmp_path / "empty_shards"
    path.mkdir()

    # Create index with specific max_dim and load from empty dir
    idx = ShardedNphdIndex(max_dim=128, path=path)
    idx.load()  # Should not raise, active_shard will be None or new

    # max_dim should remain as initialized
    assert idx.max_dim == 128
    assert idx.max_bytes == 16


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


def test_init_without_max_dim_view_mode_autodetects(tmp_path):
    """Test that init without max_dim auto-detects in view mode."""
    path = tmp_path / "shards"

    # Create and save an index
    idx1 = ShardedNphdIndex(max_dim=128, path=path)
    v1 = np.array([1, 2, 3], dtype=np.uint8)
    idx1.add(1, v1)
    idx1.save()

    # Reopen in view mode without max_dim
    idx2 = ShardedNphdIndex(path=path, view=True)

    assert idx2.max_dim == 128
    assert idx2.max_bytes == 16
    assert len(idx2) == 1
    assert idx2._view_mode is True
