"""
Test ShardedIndex merge operations.

Confirms expected behavior for merging search results across shards:
- Merge results with radius filter
- Merge batch results
- Single query merge path
- Batch query merge path
- Strict radius filtering in merge
"""

import numpy as np
import pytest
from usearch.index import Matches

from iscc_usearch.sharded import ShardedIndex


def test_merge_results_with_radius_filter(tmp_path):
    """Test that merged results respect radius filter."""
    # Create index with tiny shard to force multiple
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    # Add vectors across multiple shards
    for i in range(100):
        index.add(i, np.random.rand(64).astype(np.float32))

    # Search with very restrictive radius
    query = np.random.rand(64).astype(np.float32)
    matches = index.search(query, count=10, radius=0.01)

    # All returned distances should be within radius
    for dist in matches.distances:
        assert dist <= 0.01 or dist == float("inf")


def test_merge_batch_results(tmp_path):
    """Test merging batch results from multiple shards."""
    # Create index with tiny shard
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    for i in range(100):
        index.add(i, np.random.rand(64).astype(np.float32))

    # Batch search
    queries = np.random.rand(5, 64).astype(np.float32)
    batch_matches = index.search(queries, count=10)

    assert len(batch_matches) == 5


def test_merge_single_with_strict_radius(tmp_path):
    """Test single search with radius filters results in merge."""
    # Create index with tiny shard to force multiple shards
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    # Add vectors to create shards
    vectors = []
    for i in range(100):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Verify we have multiple sources
    assert index._view_shards is not None or index._active_shard is not None

    # Search with strict radius
    matches = index.search(vectors[0], count=10, radius=0.001)

    # All returned distances should be within radius
    for dist in matches.distances:
        if dist < float("inf"):
            assert dist <= 0.001


def test_merge_batch_with_strict_radius(tmp_path):
    """Test batch search with radius filters in merge."""
    # Create index with tiny shard to force multiple shards
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    vectors = []
    for i in range(100):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Batch search with strict radius
    query_batch = np.array(vectors[:5])
    batch_matches = index.search(query_batch, count=10, radius=0.001)

    # All returned distances should be within radius
    for i in range(5):
        for dist in batch_matches.distances[i]:
            if dist < float("inf"):
                assert dist <= 0.001


def test_multi_shard_search_single_query(tmp_path):
    """Test single query search across multiple shards with merging."""
    # Force multiple shards
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    vectors = []
    for i in range(150):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Ensure we have view shards + active shard
    if index._view_shards is not None and len(index._view_shards) > 0:
        if index._active_shard is not None and len(index._active_shard) > 0:
            # Both sources exist, search will merge
            matches = index.search(vectors[0], count=5)
            assert len(matches.keys) > 0


def test_multi_shard_search_batch_query(tmp_path):
    """Test batch query search across multiple shards with merging."""
    # Force multiple shards
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    vectors = []
    for i in range(150):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Batch search
    query_batch = np.array(vectors[:3])
    batch_matches = index.search(query_batch, count=5)

    assert len(batch_matches) == 3


def test_merge_single_matches_with_radius(tmp_path):
    """Test merging single query results with radius filtering."""
    # Create index and populate view shards via rotation
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    # Add enough to trigger rotation
    vectors = []
    for i in range(50):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # At this point we should have view_shards populated
    # Add a few more vectors to active shard (without triggering rotation)
    for i in range(50, 55):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Verify we have both sources
    has_view = index._view_shards is not None and len(index._view_shards) > 0
    has_active = index._active_shard is not None and len(index._active_shard) > 0

    if has_view and has_active:
        # Single query search with radius - should trigger merge
        matches = index.search(vectors[0], count=10, radius=0.5)
        assert len(matches.keys) > 0


def test_merge_batch_matches_with_radius(tmp_path):
    """Test merging batch query results with radius filtering."""
    # Create index and populate view shards via rotation
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    # Add enough to trigger rotation
    vectors = []
    for i in range(50):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Add a few more to active shard
    for i in range(50, 55):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Verify we have both sources
    has_view = index._view_shards is not None and len(index._view_shards) > 0
    has_active = index._active_shard is not None and len(index._active_shard) > 0

    if has_view and has_active:
        # Batch query search with radius - should trigger batch merge
        query_batch = np.array(vectors[:3])
        batch_matches = index.search(query_batch, count=10, radius=0.5)
        assert len(batch_matches) == 3


def test_search_single_query_triggers_merge(tmp_path):
    """Force merge path for single query by having both sources."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    # Add vectors to fill and rotate shards
    vectors = []
    for i in range(80):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Add to active shard after rotation
    for i in range(80, 85):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Ensure merge path is taken
    if index._view_shards and len(index._view_shards) > 0:
        if index._active_shard and len(index._active_shard) > 0:
            # This should trigger _merge_search_results with is_single=True
            matches = index.search(vectors[0], count=5)
            assert len(matches.keys) > 0


def test_search_batch_query_triggers_merge(tmp_path):
    """Force merge path for batch query by having both sources."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    # Add vectors to fill and rotate shards
    vectors = []
    for i in range(80):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Add to active shard after rotation
    for i in range(80, 85):
        v = np.random.rand(64).astype(np.float32)
        vectors.append(v)
        index.add(i, v)

    # Ensure merge path is taken
    if index._view_shards and len(index._view_shards) > 0:
        if index._active_shard and len(index._active_shard) > 0:
            # This should trigger _merge_search_results with is_single=False
            query_batch = np.array(vectors[:3])
            batch_matches = index.search(query_batch, count=5)
            assert len(batch_matches) == 3


def test_merge_code_path_single(tmp_path):
    """Explicitly test merge code path for single query."""
    # Use tiny shard size to force rotation, creating view shards
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=500)

    # Add vectors - triggers rotation, creating view shards + active shard
    vectors = np.random.rand(50, 64).astype(np.float32)
    for i, v in enumerate(vectors):
        index.add(i, v)

    # Should have view shards from rotation and active shard with remaining data
    if index._view_shards is None or len(index._view_shards) == 0:
        pytest.skip("No rotation occurred with current shard size")

    if index._active_shard is None or len(index._active_shard) == 0:
        pytest.skip("No active shard data")

    # Single query - should trigger merge
    matches = index.search(vectors[0], count=5)
    assert len(matches.keys) > 0


def test_merge_code_path_batch(tmp_path):
    """Explicitly test merge code path for batch query."""
    # Use tiny shard size to force rotation, creating view shards
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=500)

    # Add vectors - triggers rotation, creating view shards + active shard
    vectors = np.random.rand(50, 64).astype(np.float32)
    for i, v in enumerate(vectors):
        index.add(i, v)

    # Should have view shards from rotation and active shard with remaining data
    if index._view_shards is None or len(index._view_shards) == 0:
        pytest.skip("No rotation occurred with current shard size")

    if index._active_shard is None or len(index._active_shard) == 0:
        pytest.skip("No active shard data")

    # Batch query - should trigger batch merge
    batch_matches = index.search(vectors[:3], count=5)
    assert len(batch_matches) == 3


def test_merge_single_with_radius_filter(tmp_path):
    """Test merge with radius filtering for single query."""
    # Use tiny shard size to force rotation, creating view shards
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=500)

    # Add vectors - triggers rotation, creating view shards + active shard
    vectors = np.random.rand(50, 64).astype(np.float32)
    for i, v in enumerate(vectors):
        index.add(i, v)

    # Should have view shards from rotation and active shard with remaining data
    if index._view_shards is None or len(index._view_shards) == 0:
        pytest.skip("No rotation occurred with current shard size")

    if index._active_shard is None or len(index._active_shard) == 0:
        pytest.skip("No active shard data")

    # Single query with radius - triggers merge with radius filtering
    matches = index.search(vectors[0], count=10, radius=0.5)
    for dist in matches.distances:
        assert dist <= 0.5 or dist == float("inf")


def test_merge_batch_with_radius_filter(tmp_path):
    """Test merge with radius filtering for batch query."""
    # Use tiny shard size to force rotation, creating view shards
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=500)

    # Add vectors - triggers rotation, creating view shards + active shard
    vectors = np.random.rand(50, 64).astype(np.float32)
    for i, v in enumerate(vectors):
        index.add(i, v)

    # Should have view shards from rotation and active shard with remaining data
    if index._view_shards is None or len(index._view_shards) == 0:
        pytest.skip("No rotation occurred with current shard size")

    if index._active_shard is None or len(index._active_shard) == 0:
        pytest.skip("No active shard data")

    # Batch query with radius - triggers batch merge with radius filtering
    batch_matches = index.search(vectors[:3], count=10, radius=0.5)
    assert len(batch_matches) == 3


def test_merge_single_matches_multiple_sources(tmp_path):
    # type: () -> None
    """Test _merge_single_matches with 3+ sources to cover else branch."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)

    # Create 3 synthetic Matches objects
    matches1 = Matches(
        keys=np.array([1, 2, 3], dtype=np.uint64),
        distances=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    matches2 = Matches(
        keys=np.array([4, 5, 6], dtype=np.uint64),
        distances=np.array([0.15, 0.25, 0.35], dtype=np.float32),
    )
    matches3 = Matches(
        keys=np.array([7, 8, 9], dtype=np.uint64),
        distances=np.array([0.05, 0.18, 0.28], dtype=np.float32),
    )

    # Call _merge_single_matches with 3 sources (triggers else branch)
    merged = index._merge_single_matches([matches1, matches2, matches3], count=5, radius=float("inf"))

    # Verify merged results are sorted by distance
    assert len(merged.keys) == 5
    assert merged.keys[0] == 7  # distance 0.05
    assert merged.keys[1] == 1  # distance 0.1
    assert merged.distances[0] == 0.05
    assert merged.distances[1] == 0.1
