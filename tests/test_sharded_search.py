"""
Test ShardedIndex search operations.

Confirms expected behavior for vector search:
- Single query search
- Batch query search
- Empty index handling
- Count parameter validation
- Search across multiple shards
- Radius filtering
- Exact search flag
"""

import numpy as np
import pytest

from iscc_usearch.sharded import ShardedIndex


def test_search_single_vector(tmp_path):
    """Test searching with single query vector."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index.add(list(range(10)), vectors)

    matches = index.search(vectors[0], count=5)

    assert len(matches.keys) <= 5
    assert matches.keys[0] == 0  # Should find itself


def test_search_batch_vectors(tmp_path):
    """Test searching with batch of query vectors."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index.add(list(range(10)), vectors)

    batch_matches = index.search(vectors[:3], count=5)

    assert len(batch_matches) == 3


def test_search_empty_index(tmp_path):
    """Test searching empty index returns empty results."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    # Single query
    matches = index.search(np.random.rand(64).astype(np.float32), count=5)
    assert len(matches.keys) == 0

    # Batch query
    batch_matches = index.search(np.random.rand(3, 64).astype(np.float32), count=5)
    assert len(batch_matches) == 3


def test_search_count_validation(tmp_path):
    """Test that search raises for count < 1."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(ValueError, match="count must be >= 1"):
        index.search(np.random.rand(64).astype(np.float32), count=0)


def test_search_merges_results_across_shards(tmp_path):
    """Test that search merges results from multiple shards."""
    # Create index with tiny shard size
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    # Add vectors to trigger multiple shards
    vectors = np.random.rand(50, 64).astype(np.float32)
    for i, v in enumerate(vectors):
        index.add(i, v)

    # Search should find results across all shards
    matches = index.search(vectors[0], count=10)
    assert len(matches.keys) > 0


def test_search_with_radius(tmp_path):
    """Test search with radius constraint."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index.add(list(range(10)), vectors)

    # Search with very small radius
    matches = index.search(vectors[0], count=5, radius=0.0001)

    # Should find at least itself (distance 0)
    assert len(matches.keys) >= 1


def test_search_batch_with_radius(tmp_path):
    """Test batch search with radius constraint."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index.add(list(range(10)), vectors)

    # Batch search with radius
    batch_matches = index.search(vectors[:3], count=5, radius=0.0001)
    assert len(batch_matches) == 3


def test_search_active_only_no_view(tmp_path):
    """Test search with active shard only (no view shards)."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index.add(list(range(10)), vectors)

    # Should work with only active shard
    matches = index.search(vectors[0], count=5)
    assert len(matches.keys) > 0


def test_search_with_exact_flag(tmp_path):
    """Test search with exact=True."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index.add(list(range(10)), vectors)

    matches = index.search(vectors[0], count=5, exact=True)
    assert len(matches.keys) > 0


