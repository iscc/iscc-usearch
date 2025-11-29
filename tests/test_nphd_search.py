"""
Tests for NphdIndex/ShardedNphdIndex.search() method.

Verifies that search() correctly handles:
- Single and batch query vectors
- Variable-length ISCC vectors
- Padding of input vectors
- Passthrough of search parameters
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal


def test_search_single_query_returns_matches_object(nphd_index_factory):
    """Single query vector returns Matches object."""
    idx = nphd_index_factory(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=3)

    # Should return Matches object with expected attributes
    assert hasattr(result, "keys")
    assert hasattr(result, "distances")
    assert hasattr(result, "visited_members")
    assert hasattr(result, "computed_distances")
    assert len(result) == 3


def test_search_batch_queries_returns_batch_matches_object(nphd_index_factory):
    """Batch query vectors returns BatchMatches object."""
    idx = nphd_index_factory(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=2)

    # Should return BatchMatches object with expected attributes
    assert hasattr(result, "keys")
    assert hasattr(result, "distances")
    assert hasattr(result, "counts")
    assert len(result) == 2


def test_search_finds_exact_match_with_zero_distance(nphd_index_factory):
    """Search finds exact match with distance 0."""
    idx = nphd_index_factory(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=1)

    assert result.keys[0] == 1
    assert result.distances[0] == 0.0


def test_search_results_ordered_by_increasing_distance(nphd_index_factory):
    """Results are ordered by increasing distance."""
    idx = nphd_index_factory(max_dim=256)
    idx.add(1, np.array([255, 255, 255, 255], dtype=np.uint8))
    idx.add(2, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(3, np.array([178, 204, 60, 241], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=3)

    assert result.keys[0] == 2  # Exact match
    assert result.distances[0] < result.distances[1] < result.distances[2]


def test_search_with_variable_length_vectors(nphd_index_factory):
    """Search works correctly with variable-length ISCC vectors."""
    idx = nphd_index_factory(max_dim=256)
    # Add vectors of different lengths
    idx.add(1, np.array([178, 204, 60, 240, 1, 2, 3, 4], dtype=np.uint8))  # 8 bytes
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))  # 4 bytes
    idx.add(3, np.array([1, 2], dtype=np.uint8))  # 2 bytes

    query = np.array([100, 150, 200, 250], dtype=np.uint8)
    result = idx.search(query, count=3)

    assert result.keys[0] == 2  # Exact match to 4-byte vector
    assert result.distances[0] == 0.0


def test_search_count_parameter_limits_results(nphd_index_factory):
    """count parameter limits number of results returned."""
    idx = nphd_index_factory(max_dim=256)
    for i in range(10):
        idx.add(i, np.array([i, i, i, i], dtype=np.uint8))

    query = np.array([5, 5, 5, 5], dtype=np.uint8)
    result = idx.search(query, count=3)

    assert len(result) == 3


def test_search_empty_index_returns_empty_matches(nphd_index_factory):
    """Search on empty index returns empty results."""
    idx = nphd_index_factory(max_dim=256)
    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=10)

    assert len(result) == 0


def test_search_exact_parameter_passed_to_parent(nphd_index_factory):
    """exact parameter is passed to parent search method."""
    idx = nphd_index_factory(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    result_approx = idx.search(query, count=2, exact=False)
    result_exact = idx.search(query, count=2, exact=True)

    # Both should find same results for small dataset
    assert_array_equal(result_approx.keys, result_exact.keys)


def test_search_threads_parameter_passed_via_kwargs(nphd_index_factory):
    """threads parameter is passed through kwargs to parent."""
    idx = nphd_index_factory(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    # Pass threads parameter via kwargs - should not raise
    result = idx.search(query, count=1, threads=1)

    assert len(result) == 1


def test_search_with_count_zero_raises_value_error(nphd_index_factory):
    # type: () -> None
    """Search with count=0 raises ValueError to prevent usearch segfault."""
    idx = nphd_index_factory(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    with pytest.raises(ValueError, match="count must be >= 1"):
        idx.search(query, count=0)


def test_search_with_negative_count_raises_value_error(nphd_index_factory):
    # type: () -> None
    """Search with negative count raises ValueError."""
    idx = nphd_index_factory(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    with pytest.raises(ValueError, match="count must be >= 1"):
        idx.search(query, count=-5)
