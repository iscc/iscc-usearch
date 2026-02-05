"""
Test ShardedIndex get operations.

Confirms expected behavior when retrieving vectors by key:
- Single key retrieval
- Multiple keys retrieval
- Handling missing keys
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_get_single_key(tmp_path):
    """Test getting vector by single key."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vector = np.random.rand(64).astype(np.float32)
    index.add(42, vector)

    result = index.get(42)

    assert result is not None
    # Use larger tolerance since usearch may store in lower precision
    assert np.allclose(result, vector, atol=0.01)


def test_get_multiple_keys(tmp_path):
    """Test getting vectors by multiple keys."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(5, 64).astype(np.float32)
    index.add(list(range(5)), vectors)

    results = index.get([0, 2, 4])

    assert len(results) == 3


def test_get_across_shards_single(tmp_path):
    """Test get retrieves vectors from any shard (single key)."""
    # Create index and add vectors with specific values we can verify
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vec1 = np.arange(64, dtype=np.float32)
    vec2 = np.arange(64, dtype=np.float32) * 2
    index.add(1, vec1)
    index.add(2, vec2)
    index.save()

    # Reload to have view shards
    index2 = ShardedIndex(ndim=64, path=tmp_path)
    assert index2.shard_count >= 2

    # Both vectors should be retrievable regardless of which shard they're in
    result1 = index2.get(1)
    result2 = index2.get(2)

    assert result1 is not None
    assert result2 is not None
    # Verify we got the right vectors (allowing for precision loss)
    assert np.allclose(result1, vec1, atol=0.01)
    assert np.allclose(result2, vec2, atol=0.01)


def test_get_across_shards_batch(tmp_path):
    """Test get retrieves vectors from multiple shards (batch)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vectors = {
        1: np.arange(64, dtype=np.float32),
        2: np.arange(64, dtype=np.float32) * 2,
        3: np.arange(64, dtype=np.float32) * 3,
    }
    for key, vec in vectors.items():
        index.add(key, vec)
    index.save()

    index2 = ShardedIndex(ndim=64, path=tmp_path)

    # Get all keys including one that doesn't exist
    results = index2.get([1, 2, 3, 999])

    assert len(results) == 4
    assert np.allclose(results[0], vectors[1], atol=0.01)
    assert np.allclose(results[1], vectors[2], atol=0.01)
    assert np.allclose(results[2], vectors[3], atol=0.01)
    assert results[3] is None  # Missing key


def test_get_missing_key_returns_none(tmp_path):
    """Test get returns None for missing single key."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))

    result = index.get(999)

    assert result is None


def test_get_empty_keys_array(tmp_path):
    """Test get with empty keys array returns empty list."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))

    result = index.get([])

    assert result == []


def test_get_early_exit_all_keys_found(tmp_path):
    """Test get early exit when all keys found before processing all view shards."""
    # type: () -> None
    # shard_size is in bytes. Use 1 byte to force rotation after each vector add.
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vec1 = np.arange(64, dtype=np.float32)
    vec2 = np.arange(64, dtype=np.float32) * 2
    vec3 = np.arange(64, dtype=np.float32) * 3

    # Add 3 vectors to create multiple shards (one per shard due to tiny shard_size)
    index.add(1, vec1)
    index.add(2, vec2)
    index.add(3, vec3)
    index.save()

    # Verify we have multiple view shards
    assert len(index._viewed_indexes) >= 2

    # Request only key 1 which is in the first view shard.
    # After processing first view shard, found.all() = True.
    # The break statement skips remaining view shards.
    results = index.get([1])

    assert len(results) == 1
    assert results[0] is not None
    assert np.allclose(results[0], vec1, atol=0.01)
