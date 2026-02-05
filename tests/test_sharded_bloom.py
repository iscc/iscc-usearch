"""Tests for ShardedIndex bloom filter integration."""

import numpy as np
import pytest
from pathlib import Path
from iscc_usearch import ShardedIndex


def test_bloom_enabled_by_default(tmp_path: Path):
    """Test that bloom filter is enabled by default."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    assert idx._use_bloom is True
    assert idx._bloom is not None


def test_bloom_can_be_disabled(tmp_path: Path):
    """Test that bloom filter can be disabled."""
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)
    assert idx._use_bloom is False
    assert idx._bloom is None


def test_bloom_tracks_added_keys(tmp_path: Path):
    """Test that bloom filter tracks keys added via add()."""
    idx = ShardedIndex(ndim=32, path=tmp_path)

    # Add some vectors
    vectors = np.random.rand(10, 32).astype(np.float32)
    keys = idx.add(None, vectors)

    # Bloom filter should contain all added keys
    for key in keys:
        assert idx._bloom.contains(int(key))


def test_bloom_tracks_single_add_with_explicit_key(tmp_path: Path):
    """Test that bloom filter tracks single vector add with explicit key."""
    idx = ShardedIndex(ndim=32, path=tmp_path)

    # Add a single vector with explicit key
    vector = np.random.rand(32).astype(np.float32)
    key = idx.add(42, vector)

    # Bloom filter should contain the added key
    assert key == 42
    assert idx._bloom.contains(42)


def test_bloom_speeds_up_nonexistent_key_lookup(tmp_path: Path):
    """Test that bloom filter correctly rejects non-existent keys."""
    idx = ShardedIndex(ndim=32, path=tmp_path)

    # Add some vectors with specific keys
    vectors = np.random.rand(100, 32).astype(np.float32)
    keys = list(range(100))
    idx.add(keys, vectors)

    # Check that existing keys are found
    assert idx.contains(0)
    assert idx.contains(50)
    assert idx.contains(99)

    # Check that non-existent keys return False quickly via bloom filter
    assert not idx.contains(1000)
    assert not idx.contains(9999)


def test_bloom_contains_batch(tmp_path: Path):
    """Test bloom filter with batch contains."""
    idx = ShardedIndex(ndim=32, path=tmp_path)

    vectors = np.random.rand(100, 32).astype(np.float32)
    keys = list(range(100))
    idx.add(keys, vectors)

    # Mix of existing and non-existing keys
    test_keys = [0, 50, 99, 1000, 2000]
    results = idx.contains(test_keys)

    assert results[0]  # 0 exists
    assert results[1]  # 50 exists
    assert results[2]  # 99 exists
    assert not results[3]  # 1000 doesn't exist
    assert not results[4]  # 2000 doesn't exist


def test_bloom_get_nonexistent_key(tmp_path: Path):
    """Test that get() uses bloom filter for non-existent keys."""
    idx = ShardedIndex(ndim=32, path=tmp_path)

    vectors = np.random.rand(10, 32).astype(np.float32)
    keys = list(range(10))
    idx.add(keys, vectors)

    # Get non-existent key (should be rejected by bloom filter)
    result = idx.get(9999)
    assert result is None


def test_bloom_get_batch_mixed_keys(tmp_path: Path):
    """Test get() batch with bloom filter for mixed existing/non-existing keys."""
    idx = ShardedIndex(ndim=32, path=tmp_path)

    vectors = np.random.rand(10, 32).astype(np.float32)
    keys = list(range(10))
    idx.add(keys, vectors)

    # Mix of existing and non-existing keys
    test_keys = [0, 5, 9, 1000, 2000]
    results = idx.get(test_keys)

    assert results[0] is not None  # 0 exists
    assert results[1] is not None  # 5 exists
    assert results[2] is not None  # 9 exists
    assert results[3] is None  # 1000 doesn't exist
    assert results[4] is None  # 2000 doesn't exist


def test_bloom_persisted_on_save(tmp_path: Path):
    """Test that bloom filter is saved to disk."""
    idx = ShardedIndex(ndim=32, path=tmp_path)

    vectors = np.random.rand(100, 32).astype(np.float32)
    keys = list(range(100))
    idx.add(keys, vectors)
    idx.save()

    # Check bloom file exists
    bloom_path = tmp_path / "bloom.isbf"
    assert bloom_path.exists()


def test_bloom_loaded_on_load(tmp_path: Path):
    """Test that bloom filter is loaded from disk."""
    # Create and save index with bloom filter
    idx = ShardedIndex(ndim=32, path=tmp_path)
    vectors = np.random.rand(100, 32).astype(np.float32)
    keys = list(range(100))
    idx.add(keys, vectors)
    idx.save()

    # Load index
    idx2 = ShardedIndex(ndim=32, path=tmp_path)

    # Bloom filter should be loaded and work
    assert idx2._bloom is not None
    assert idx2._bloom.count == 100

    # Should find existing keys
    assert idx2.contains(0)
    assert idx2.contains(50)

    # Should reject non-existing keys via bloom
    assert not idx2.contains(9999)


def test_bloom_missing_file_disables_bloom(tmp_path: Path):
    """Test that missing bloom file disables bloom in load mode (legacy index).

    This tests the regression where loading a legacy index without a bloom.isbf
    file would create an empty bloom filter, causing all lookups to return
    False/None because an empty bloom filter rejects all keys.
    """
    # Create and save index with bloom filter
    idx = ShardedIndex(ndim=32, path=tmp_path)
    vectors = np.random.rand(10, 32).astype(np.float32)
    keys = list(range(10))
    idx.add(keys, vectors)
    idx.save()

    # Delete bloom file to simulate legacy index
    bloom_path = tmp_path / "bloom.isbf"
    bloom_path.unlink()

    # Load in load mode (default) - should work without bloom
    idx2 = ShardedIndex(ndim=32, path=tmp_path)
    assert idx2._bloom is None  # Bloom should be disabled, not empty

    # Critical: lookups should still work by scanning shards
    assert len(idx2) == 10
    assert idx2.contains(0)  # Must find existing keys
    assert idx2.contains(5)
    assert idx2.contains(9)
    assert not idx2.contains(999)  # Non-existent key

    # get() should also work
    result = idx2.get(5)
    assert result is not None

    result = idx2.get(999)
    assert result is None


def test_bloom_disabled_index_works_normally(tmp_path: Path):
    """Test that index with bloom disabled works correctly."""
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)

    vectors = np.random.rand(10, 32).astype(np.float32)
    keys = list(range(10))
    idx.add(keys, vectors)

    # All operations should work
    assert idx.contains(0)
    assert idx.contains(5)
    assert not idx.contains(999)

    result = idx.get(5)
    assert result is not None

    result = idx.get(999)
    assert result is None


def test_bloom_no_false_negatives(tmp_path: Path):
    """Test that bloom filter never causes false negatives."""
    idx = ShardedIndex(ndim=32, path=tmp_path)

    vectors = np.random.rand(1000, 32).astype(np.float32)
    keys = list(range(1000))
    idx.add(keys, vectors)

    # All added keys must be found (no false negatives)
    for key in keys:
        assert idx.contains(key), f"False negative for key {key}"


def test_bloom_save_empty_index(tmp_path: Path):
    """Test saving bloom filter for empty index doesn't crash."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    idx.save()  # Should not crash with empty bloom filter


def test_bloom_batch_add_tracks_keys(tmp_path: Path):
    """Test that batch add updates bloom filter with all keys."""
    idx = ShardedIndex(ndim=32, path=tmp_path)

    # Batch add with auto-generated keys
    vectors = np.random.rand(50, 32).astype(np.float32)
    keys = idx.add(None, vectors)

    # All auto-generated keys should be in bloom filter
    assert len(keys) == 50
    for key in keys:
        assert idx._bloom.contains(int(key))


def test_bloom_contains_batch_all_nonexistent(tmp_path: Path):
    """Test batch contains when ALL keys are definitely not present (early exit path)."""
    idx = ShardedIndex(ndim=32, path=tmp_path)

    # Add some vectors
    vectors = np.random.rand(10, 32).astype(np.float32)
    keys = list(range(10))
    idx.add(keys, vectors)

    # Query only non-existent keys - should hit early exit path
    test_keys = [1000, 2000, 3000, 4000, 5000]
    results = idx.contains(test_keys)

    # All should be False
    assert all(not r for r in results)


# Tests for rebuild_bloom()


def test_rebuild_bloom_basic(tmp_path: Path):
    """Test basic rebuild_bloom functionality."""
    # Create index without bloom filter
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)

    vectors = np.random.rand(100, 32).astype(np.float32)
    keys = list(range(100))
    idx.add(keys, vectors)

    # Initially no bloom filter
    assert idx._bloom is None

    # Rebuild bloom filter
    count = idx.rebuild_bloom(save=False, log_progress=False)

    # Bloom filter should now exist and contain all keys
    assert idx._use_bloom is True
    assert idx._bloom is not None
    assert count == 100
    assert idx._bloom.count == 100

    # All keys should be in the bloom filter
    for key in keys:
        assert idx._bloom.contains(key)


def test_rebuild_bloom_saves_to_disk(tmp_path: Path):
    """Test that rebuild_bloom saves the bloom filter to disk by default."""
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)

    vectors = np.random.rand(50, 32).astype(np.float32)
    idx.add(None, vectors)

    # Rebuild with save=True (default)
    idx.rebuild_bloom(log_progress=False)

    # Bloom file should exist
    bloom_path = tmp_path / "bloom.isbf"
    assert bloom_path.exists()


def test_rebuild_bloom_no_save(tmp_path: Path):
    """Test that rebuild_bloom respects save=False."""
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)

    vectors = np.random.rand(50, 32).astype(np.float32)
    idx.add(None, vectors)

    # Rebuild without saving
    idx.rebuild_bloom(save=False, log_progress=False)

    # Bloom file should NOT exist
    bloom_path = tmp_path / "bloom.isbf"
    assert not bloom_path.exists()


def test_rebuild_bloom_enables_bloom_lookups(tmp_path: Path):
    """Test that rebuild_bloom enables bloom filter for subsequent lookups."""
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)

    vectors = np.random.rand(100, 32).astype(np.float32)
    keys = list(range(100))
    idx.add(keys, vectors)

    # Rebuild bloom filter
    idx.rebuild_bloom(save=False, log_progress=False)

    # Now lookups should use bloom filter
    # Existing keys should be found
    assert idx.contains(0)
    assert idx.contains(50)
    assert idx.contains(99)

    # Non-existing keys should be rejected (quickly via bloom)
    assert not idx.contains(1000)
    assert not idx.contains(9999)


def test_rebuild_bloom_for_existing_index(tmp_path: Path):
    """Test rebuilding bloom for an index loaded without bloom."""
    # Create index with bloom and save
    idx1 = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=True)
    vectors = np.random.rand(100, 32).astype(np.float32)
    keys = list(range(100))
    idx1.add(keys, vectors)
    idx1.save()

    # Delete bloom file to simulate old index without bloom
    bloom_path = tmp_path / "bloom.isbf"
    bloom_path.unlink()

    # Load index without bloom
    idx2 = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)
    assert idx2._bloom is None

    # Rebuild bloom
    count = idx2.rebuild_bloom(log_progress=False)
    assert count == 100

    # Bloom should work now
    assert idx2.contains(0)
    assert not idx2.contains(9999)


def test_rebuild_bloom_across_multiple_shards(tmp_path: Path):
    """Test rebuild_bloom works across multiple shards."""
    # Create index with very small shard size to force rotation
    idx = ShardedIndex(ndim=32, path=tmp_path, shard_size=500, bloom_filter=False)

    # Add enough entries to create multiple shards
    for i in range(100):
        vector = np.random.rand(32).astype(np.float32)
        idx.add(i, vector)

    # Should have multiple shards
    assert idx.shard_count >= 1

    # Rebuild bloom
    count = idx.rebuild_bloom(save=False, log_progress=False)

    # All keys should be in bloom
    assert count == 100
    for i in range(100):
        assert idx._bloom.contains(i)


def test_rebuild_bloom_empty_index(tmp_path: Path):
    """Test rebuild_bloom on empty index."""
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)

    count = idx.rebuild_bloom(save=False, log_progress=False)

    assert count == 0
    assert idx._bloom is not None
    assert idx._bloom.count == 0


def test_rebuild_bloom_returns_correct_count(tmp_path: Path):
    """Test that rebuild_bloom returns accurate count."""
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)

    vectors = np.random.rand(42, 32).astype(np.float32)
    idx.add(None, vectors)

    count = idx.rebuild_bloom(save=False, log_progress=False)

    assert count == 42


def test_rebuild_bloom_persists_across_load(tmp_path: Path):
    """Test that rebuilt bloom filter persists and loads correctly."""
    # Create index without bloom
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)
    vectors = np.random.rand(100, 32).astype(np.float32)
    keys = list(range(100))
    idx.add(keys, vectors)
    idx.save()

    # Rebuild and save bloom
    idx.rebuild_bloom(log_progress=False)

    # Load fresh instance
    idx2 = ShardedIndex(ndim=32, path=tmp_path)

    # Bloom should be loaded and functional
    assert idx2._bloom is not None
    assert idx2._bloom.count == 100
    assert idx2.contains(0)
    assert not idx2.contains(9999)


def test_rebuild_bloom_with_logging(tmp_path: Path):
    """Test rebuild_bloom with log_progress=True."""
    # Create index with multiple shards
    idx = ShardedIndex(ndim=32, path=tmp_path, shard_size=500, bloom_filter=False)

    # Add entries to create multiple shards
    for i in range(100):
        vector = np.random.rand(32).astype(np.float32)
        idx.add(i, vector)

    # Rebuild with logging - should cover log lines 599, 608, 619, 625
    count = idx.rebuild_bloom(save=False, log_progress=True)

    # Verify it worked
    assert count == 100
    assert idx._bloom is not None
    for i in range(100):
        assert idx._bloom.contains(i)


def test_rebuild_bloom_skips_empty_shards(tmp_path: Path):
    """Test rebuild_bloom handles empty shards correctly (line 608)."""
    # Create index and manually set up scenario with empty shard
    idx = ShardedIndex(ndim=32, path=tmp_path, shard_size=1000, bloom_filter=False)

    # Add some entries
    for i in range(50):
        vector = np.random.rand(32).astype(np.float32)
        idx.add(i, vector)

    idx.save()
    idx._rotate_shard()

    # Verify we have an empty active shard (this forces line 608 to execute)
    assert idx._active_shard is not None
    assert len(idx._active_shard) == 0

    # Now we have a viewed shard with data and empty active shard
    # Rebuild bloom - should skip the empty active shard (line 608)
    count = idx.rebuild_bloom(save=False, log_progress=False)

    assert count == 50
    assert idx._bloom is not None
    for i in range(50):
        assert idx._bloom.contains(i)


def test_bloom_stays_consistent_when_toggled(tmp_path: Path):
    """Test bloom filter stays in sync when bloom_filter setting is toggled.

    Regression test for issue where:
    1. Index created with bloom_filter=True
    2. Reopened with bloom_filter=False, keys added
    3. Reopened with bloom_filter=True - keys added while bloom was disabled were not found
    """
    # Step 1: Create index with bloom enabled, add initial keys
    idx1 = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=True)
    vectors1 = np.random.rand(10, 32).astype(np.float32)
    initial_keys = list(range(10))
    idx1.add(initial_keys, vectors1)
    idx1.save()

    # Verify bloom file exists and contains initial keys
    bloom_path = tmp_path / "bloom.isbf"
    assert bloom_path.exists()
    assert idx1._bloom.count == 10

    # Step 2: Reopen with bloom_filter=False, add more keys
    idx2 = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)
    assert idx2._use_bloom is False
    # The bloom filter should still be loaded to keep it in sync
    assert idx2._bloom is not None, "Bloom should be loaded even when _use_bloom=False"

    vectors2 = np.random.rand(10, 32).astype(np.float32)
    new_keys = list(range(10, 20))
    idx2.add(new_keys, vectors2)
    idx2.save()

    # Bloom should now contain all 20 keys
    assert idx2._bloom.count == 20

    # Step 3: Reopen with bloom_filter=True (default)
    idx3 = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=True)

    # Bloom should contain all 20 keys
    assert idx3._bloom is not None
    assert idx3._bloom.count == 20

    # Critical: ALL keys must be found (no false negatives)
    for key in initial_keys:
        assert idx3.contains(key), f"False negative for initial key {key}"
        assert idx3.get(key) is not None, f"get() returned None for initial key {key}"

    for key in new_keys:
        assert idx3.contains(key), f"False negative for key {key} added while bloom disabled"
        assert idx3.get(key) is not None, f"get() returned None for key {key} added while bloom disabled"

    # Non-existent keys should still return False/None
    assert not idx3.contains(999)
    assert idx3.get(999) is None
