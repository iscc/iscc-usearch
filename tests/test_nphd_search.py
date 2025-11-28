"""
Tests for NphdIndex.search() method.

Verifies that NphdIndex.search() correctly handles:
- Single and batch query vectors
- Variable-length ISCC vectors
- Padding of input vectors
- Passthrough of search parameters
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from iscc_usearch.nphd import NphdIndex


def test_search_single_query_returns_matches_object():
    """Single query vector returns Matches object."""
    idx = NphdIndex(max_dim=256)
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


def test_search_batch_queries_returns_batch_matches_object():
    """Batch query vectors returns BatchMatches object."""
    idx = NphdIndex(max_dim=256)
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


def test_search_finds_exact_match_with_zero_distance():
    """Search finds exact match with distance 0."""
    idx = NphdIndex(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=1)

    assert result.keys[0] == 1
    assert result.distances[0] == 0.0


def test_search_results_ordered_by_increasing_distance():
    """Results are ordered by increasing distance."""
    idx = NphdIndex(max_dim=256)
    idx.add(1, np.array([255, 255, 255, 255], dtype=np.uint8))
    idx.add(2, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(3, np.array([178, 204, 60, 241], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=3)

    assert result.keys[0] == 2  # Exact match
    assert result.distances[0] < result.distances[1] < result.distances[2]


def test_search_with_variable_length_vectors():
    """Search works correctly with variable-length ISCC vectors."""
    idx = NphdIndex(max_dim=256)
    # Add vectors of different lengths
    idx.add(1, np.array([178, 204, 60, 240, 1, 2, 3, 4], dtype=np.uint8))  # 8 bytes
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))  # 4 bytes
    idx.add(3, np.array([1, 2], dtype=np.uint8))  # 2 bytes

    query = np.array([100, 150, 200, 250], dtype=np.uint8)
    result = idx.search(query, count=3)

    assert result.keys[0] == 2  # Exact match to 4-byte vector
    assert result.distances[0] == 0.0


def test_search_count_parameter_limits_results():
    """count parameter limits number of results returned."""
    idx = NphdIndex(max_dim=256)
    for i in range(10):
        idx.add(i, np.array([i, i, i, i], dtype=np.uint8))

    query = np.array([5, 5, 5, 5], dtype=np.uint8)
    result = idx.search(query, count=3)

    assert len(result) == 3


def test_search_empty_index_returns_empty_matches():
    """Search on empty index returns empty results."""
    idx = NphdIndex(max_dim=256)
    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=10)

    assert len(result) == 0


def test_search_exact_parameter_passed_to_parent():
    """exact parameter is passed to parent search method."""
    idx = NphdIndex(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    result_approx = idx.search(query, count=2, exact=False)
    result_exact = idx.search(query, count=2, exact=True)

    # Both should find same results for small dataset
    assert_array_equal(result_approx.keys, result_exact.keys)


def test_search_threads_parameter_passed_via_kwargs():
    """threads parameter is passed through kwargs to parent."""
    idx = NphdIndex(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    # Pass threads parameter via kwargs - should not raise
    result = idx.search(query, count=1, threads=1)

    assert len(result) == 1


def test_search_with_count_zero_raises_value_error():
    # type: () -> None
    """Search with count=0 raises ValueError to prevent usearch segfault."""
    idx = NphdIndex(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    with pytest.raises(ValueError, match="count must be >= 1"):
        idx.search(query, count=0)


def test_search_with_negative_count_raises_value_error():
    # type: () -> None
    """Search with negative count raises ValueError."""
    idx = NphdIndex(max_dim=256)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    with pytest.raises(ValueError, match="count must be >= 1"):
        idx.search(query, count=-5)
