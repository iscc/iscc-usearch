"""Tests for ShardedIndex.reset() — reset-to-empty semantics.

Verifies that reset() releases all in-memory resources (view shards, active
shard, bloom filter) without deleting files on disk. After reset the index
is empty but immediately usable for new add() calls.
"""

import numpy as np
from pathlib import Path

from iscc_usearch import ShardedIndex, ShardedNphdIndex


def test_reset_empty_index(tmp_path: Path):
    """Reset on a fresh empty index keeps it empty and usable."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    idx.reset()

    assert len(idx) == 0
    assert idx._active_shard is not None
    assert idx._view_shards is None
    assert idx._viewed_indexes == []


def test_reset_clears_data(tmp_path: Path):
    """Reset after adding vectors yields an empty index."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    vectors = np.random.rand(50, 32).astype(np.float32)
    idx.add(list(range(50)), vectors)
    assert len(idx) == 50

    idx.reset()

    assert len(idx) == 0
    assert not idx.contains(0)
    assert idx.get(0) is None


def test_reset_allows_new_adds(tmp_path: Path):
    """After reset, new vectors can be added and searched."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    vectors = np.random.rand(10, 32).astype(np.float32)
    idx.add(list(range(10)), vectors)

    idx.reset()

    # Add new data
    new_vectors = np.random.rand(5, 32).astype(np.float32)
    idx.add(list(range(100, 105)), new_vectors)

    assert len(idx) == 5
    assert idx.contains(100)
    assert not idx.contains(0)  # old data gone

    # Search works
    results = idx.search(new_vectors[0], count=1)
    assert len(results.keys) > 0


def test_reset_does_not_delete_files(tmp_path: Path):
    """Reset does not delete shard or bloom files on disk."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    vectors = np.random.rand(50, 32).astype(np.float32)
    idx.add(list(range(50)), vectors)
    idx.save()

    shard_files_before = list(tmp_path.glob("shard_*.usearch"))
    bloom_path = tmp_path / "bloom.isbf"
    assert len(shard_files_before) > 0
    assert bloom_path.exists()

    idx.reset()

    # Files still on disk
    shard_files_after = list(tmp_path.glob("shard_*.usearch"))
    assert len(shard_files_after) == len(shard_files_before)
    assert bloom_path.exists()


def test_reset_with_multiple_shards(tmp_path: Path):
    """Reset releases all view shards and active shard."""
    idx = ShardedIndex(ndim=32, path=tmp_path, shard_size=500)

    # Add enough to force shard rotation
    for i in range(100):
        vector = np.random.rand(32).astype(np.float32)
        idx.add(i, vector)

    assert idx.shard_count >= 2
    assert len(idx._viewed_indexes) >= 1

    idx.reset()

    assert len(idx) == 0
    assert idx._view_shards is None
    assert idx._viewed_indexes == []
    assert idx._active_shard is not None


def test_reset_clears_bloom_filter(tmp_path: Path):
    """Reset clears bloom filter to initial state."""
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=True)
    vectors = np.random.rand(50, 32).astype(np.float32)
    idx.add(list(range(50)), vectors)
    assert idx._bloom.count == 50

    idx.reset()

    assert idx._bloom is not None
    assert idx._bloom.count == 0
    # Bloom should not report old keys
    assert not idx._bloom.contains(0)


def test_reset_without_bloom_filter(tmp_path: Path):
    """Reset works when bloom filter is disabled."""
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)
    vectors = np.random.rand(10, 32).astype(np.float32)
    idx.add(list(range(10)), vectors)

    idx.reset()

    assert len(idx) == 0
    assert idx._bloom is None


def test_reset_invalidates_shard_cache(tmp_path: Path):
    """Reset invalidates the internal shard cache."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    vectors = np.random.rand(10, 32).astype(np.float32)
    idx.add(list(range(10)), vectors)

    idx.reset()

    assert idx._cached_shards is None


def test_reset_resets_size_check_countdown(tmp_path: Path):
    """Reset resets the amortized size check countdown."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    vectors = np.random.rand(100, 32).astype(np.float32)
    idx.add(list(range(100)), vectors)

    idx.reset()

    assert idx._adds_until_size_check == 0


def test_reset_preserves_config(tmp_path: Path):
    """Reset preserves index configuration for creating new shards."""
    idx = ShardedIndex(ndim=32, path=tmp_path, connectivity=16, expansion_add=128)
    idx.reset()

    assert idx._config["ndim"] == 32
    assert idx._config["connectivity"] == 16
    assert idx._config["expansion_add"] == 128


def test_reset_nphd_index(tmp_path: Path):
    """Reset works on ShardedNphdIndex with variable-length vectors."""
    idx = ShardedNphdIndex(max_dim=256, path=tmp_path)

    # Add variable-length vectors
    v1 = np.random.randint(0, 256, size=8, dtype=np.uint8)
    v2 = np.random.randint(0, 256, size=16, dtype=np.uint8)
    idx.add(1, v1)
    idx.add(2, v2)
    assert len(idx) == 2

    idx.reset()

    assert len(idx) == 0
    assert not idx.contains(1)
    assert idx.max_dim == 256

    # Can add new vectors after reset
    v3 = np.random.randint(0, 256, size=12, dtype=np.uint8)
    idx.add(3, v3)
    assert len(idx) == 1
    assert idx.contains(3)


def test_reset_active_shard_path_cleared(tmp_path: Path):
    """Reset clears the tracked active shard path."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    vectors = np.random.rand(10, 32).astype(np.float32)
    idx.add(list(range(10)), vectors)
    idx.save()
    assert idx._active_shard_path is not None

    idx.reset()

    assert idx._active_shard_path is None


def test_reset_then_save(tmp_path: Path):
    """Save after reset works correctly."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    vectors = np.random.rand(10, 32).astype(np.float32)
    idx.add(list(range(10)), vectors)
    idx.save()

    idx.reset()

    # Add new data and save
    new_vectors = np.random.rand(5, 32).astype(np.float32)
    idx.add(list(range(100, 105)), new_vectors)
    idx.save()

    # Reload and verify only new data exists in memory
    idx2 = ShardedIndex(ndim=32, path=tmp_path)
    # Old shard files still on disk, so loaded index sees both old and new
    assert idx2.contains(100)
