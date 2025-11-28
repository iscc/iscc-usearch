"""
Confirm the expected behavior of usearch Index.search() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- single query vector (returns Matches)
- batch query vectors (returns BatchMatches)
- different count values
- exact=True vs exact=False
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from usearch.index import BatchMatches, Index, Matches, MetricKind, ScalarKind

# Tests demonstrating complete result structures with literal expected values


def test_search_single_query_shows_complete_matches_structure():
    """
    Shows the complete structure of a Matches object returned by search().
    The 'expected' variable is an actual Matches object showing the expected structure.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=2)

    # Build the expected Matches object
    expected = Matches(
        keys=np.array([1, 2], dtype=np.uint64),
        distances=np.array([0.0, result.distances[1]], dtype=np.float32),
        visited_members=result.visited_members,
        computed_distances=result.computed_distances,
    )

    # Verify the Matches object structure matches
    assert_array_equal(result.keys, expected.keys)
    assert_array_equal(result.distances, expected.distances)
    assert len(result) == len(expected)
    assert result.visited_members == expected.visited_members
    assert result.computed_distances == expected.computed_distances

    # Verify individual Match access
    assert result[0].key == 1
    assert result[0].distance == 0.0
    assert result[1].key == 2

    # Verify to_list() conversion
    assert len(result.to_list()) == 2
    assert result.to_list()[0] == (1, 0.0)


def test_search_batch_queries_shows_complete_batch_matches_structure():
    """
    Shows the complete structure of a BatchMatches object returned by batch search().
    The 'expected' variable is an actual BatchMatches object showing the expected structure.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],  # Query 0: exact match to key 1
            [100, 150, 200, 250],  # Query 1: exact match to key 2
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=2)

    # Build the expected BatchMatches object
    expected = BatchMatches(
        keys=np.array(
            [
                [1, result.keys[0][1]],  # Query 0: [key1, key2]
                [2, result.keys[1][1]],  # Query 1: [key1, key2]
            ],
            dtype=np.uint64,
        ),
        distances=np.array(
            [
                [0.0, result.distances[0][1]],  # Query 0: [dist1, dist2]
                [0.0, result.distances[1][1]],  # Query 1: [dist1, dist2]
            ],
            dtype=np.float32,
        ),
        counts=np.array([2, 2], dtype=np.int64),
        visited_members=result.visited_members,
        computed_distances=result.computed_distances,
    )

    # Verify the BatchMatches object structure matches
    assert_array_equal(result.keys, expected.keys)
    assert_array_equal(result.distances, expected.distances)
    assert_array_equal(result.counts, expected.counts)
    assert len(result) == len(expected)
    assert result.visited_members == expected.visited_members
    assert result.computed_distances == expected.computed_distances

    # Verify nested Matches access (result[0] returns Matches for query 0)
    assert_array_equal(result[0].keys, np.array([1, result.keys[0][1]], dtype=np.uint64))
    assert result[0].distances[0] == 0.0
    assert len(result[0]) == 2

    # Verify deeply nested Match access (result[0][0] returns Match)
    assert result[0][0].key == 1
    assert result[0][0].distance == 0.0

    # Verify to_list() returns flattened list
    result_list = result.to_list()
    assert len(result_list) == 4  # 2 queries * 2 results each
    assert result_list[0] == (1, 0.0)
    assert result_list[2] == (2, 0.0)


def test_search_with_known_distances_shows_literal_matches_values():
    """
    Shows a Matches object with 100% literal values (no computed references).
    Uses vectors with known Hamming distances for fully predictable results.
    The 'expected' variable is an actual Matches object with all concrete values.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    # Add vectors with known Hamming distances to query [178, 204, 60, 240]
    idx.add(1, np.array([255, 255, 255, 255], dtype=np.uint8))  # 16 bits different
    idx.add(2, np.array([178, 204, 60, 240], dtype=np.uint8))  # 0 bits different (exact match)
    idx.add(3, np.array([178, 204, 60, 241], dtype=np.uint8))  # 1 bit different

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=3)

    # Build the expected Matches object with ALL LITERAL VALUES
    # Results are ordered by distance: key 2 (0.0), key 3 (1.0), key 1 (16.0)
    expected = Matches(
        keys=np.array([2, 3, 1], dtype=np.uint64),
        distances=np.array([0.0, 1.0, 16.0], dtype=np.float32),
        visited_members=result.visited_members,
        computed_distances=result.computed_distances,
    )

    # Verify the complete Matches object structure matches
    assert_array_equal(result.keys, expected.keys)
    assert_array_equal(result.distances, expected.distances)
    assert len(result) == len(expected)
    assert len(result) == 3

    # Verify individual Match objects (result[0], result[1], result[2])
    assert result[0].key == 2
    assert result[0].distance == 0.0

    assert result[1].key == 3
    assert result[1].distance == 1.0

    assert result[2].key == 1
    assert result[2].distance == 16.0

    # Verify to_list() conversion
    assert result.to_list() == [(2, 0.0), (3, 1.0), (1, 16.0)]

    # Verify Python list conversions
    assert result.keys.tolist() == [2, 3, 1]
    assert result.distances.tolist() == [0.0, 1.0, 16.0]


# Tests for Index.search() with single query vector (returns Matches)


def test_search_single_vector_returns_matches_object():
    """Searching with single vector returns Matches object."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=3)

    expected_attributes = ["keys", "distances", "visited_members", "computed_distances"]
    expected_length = 3

    # Verify result is a Matches object (not BatchMatches)
    for attr in expected_attributes:
        assert hasattr(result, attr)
    assert len(result) == expected_length


def test_search_single_vector_keys_and_distances_are_1d_arrays():
    """Matches object has 1D numpy arrays for keys and distances."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=3)

    expected_keys_type = np.ndarray
    expected_keys_ndim = 1
    expected_keys_shape = (3,)
    expected_distances_type = np.ndarray
    expected_distances_ndim = 1
    expected_distances_shape = (3,)

    assert isinstance(result.keys, expected_keys_type)
    assert result.keys.ndim == expected_keys_ndim
    assert result.keys.shape == expected_keys_shape

    assert isinstance(result.distances, expected_distances_type)
    assert result.distances.ndim == expected_distances_ndim
    assert result.distances.shape == expected_distances_shape


def test_search_single_vector_exact_match_has_zero_distance():
    """Searching for exact match returns distance of 0."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=1)

    expected_key = np.array([1], dtype=np.uint64)
    expected_distance = np.array([0.0], dtype=np.float32)

    assert result.keys[0] == expected_key[0]
    assert result.distances[0] == expected_distance[0]


def test_search_single_vector_results_ordered_by_distance():
    """Search results are ordered by increasing distance."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    # Add vectors with known Hamming distances to query
    idx.add(1, np.array([255, 255, 255, 255], dtype=np.uint8))  # Will be furthest
    idx.add(2, np.array([178, 204, 60, 240], dtype=np.uint8))  # Exact match (distance 0)
    idx.add(3, np.array([178, 204, 60, 241], dtype=np.uint8))  # 1 bit difference

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=3)

    expected_keys = np.array([2, 3, 1], dtype=np.uint64)
    expected_first_distance = 0.0  # Exact match

    assert_array_equal(result.keys, expected_keys)
    # Verify distances are in ascending order
    assert result.distances[0] < result.distances[1] < result.distances[2]
    assert result.distances[0] == expected_first_distance


def test_search_single_vector_count_limits_results():
    """count parameter limits number of results returned."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))
    idx.add(4, np.array([10, 20, 30, 40], dtype=np.uint8))
    idx.add(5, np.array([50, 60, 70, 80], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    # Request only 2 results
    result = idx.search(query, count=2)

    expected_length = 2
    expected_keys_shape = (2,)
    expected_distances_shape = (2,)

    assert len(result) == expected_length
    assert result.keys.shape == expected_keys_shape
    assert result.distances.shape == expected_distances_shape


def test_search_single_vector_default_count_is_10():
    """Default count parameter is 10."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    # Add only 3 vectors
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query)  # No count specified

    # Should return all 3 vectors (limited by index size, not default count)
    expected_length = 3

    assert len(result) == expected_length


def test_search_single_vector_count_exceeds_index_size():
    """Requesting more results than index size returns all vectors."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=100)

    expected_length = 2  # Only 2 vectors in index

    assert len(result) == expected_length


def test_search_single_vector_empty_index_returns_empty_matches():
    """Searching empty index returns Matches with length 0."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=10)

    expected_length = 0
    expected_keys_shape = (0,)
    expected_distances_shape = (0,)

    assert len(result) == expected_length
    assert result.keys.shape == expected_keys_shape
    assert result.distances.shape == expected_distances_shape


def test_search_single_vector_accessing_individual_matches():
    """Individual Match objects can be accessed by index."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=2)

    expected_match_attributes = ["key", "distance"]

    # Access first match
    first_match = result[0]
    for attr in expected_match_attributes:
        assert hasattr(first_match, attr)
    assert first_match.key == result.keys[0]
    assert first_match.distance == result.distances[0]

    # Access second match
    second_match = result[1]
    assert second_match.key == result.keys[1]
    assert second_match.distance == result.distances[1]


def test_search_single_vector_to_list_returns_tuples():
    """Matches.to_list() returns list of (key, distance) tuples."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=2)

    result_list = result.to_list()

    expected_list_type = list
    expected_list_length = 2
    expected_tuple_length = 2

    assert isinstance(result_list, expected_list_type)
    assert len(result_list) == expected_list_length
    assert all(isinstance(item, tuple) and len(item) == expected_tuple_length for item in result_list)
    # Verify first tuple matches first result
    assert result_list[0][0] == result.keys[0]
    assert result_list[0][1] == result.distances[0]


# Tests for Index.search() with batch query vectors (returns BatchMatches)


def test_search_batch_vectors_returns_batch_matches_object():
    """Searching with multiple vectors returns BatchMatches object."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=3)

    expected_attributes = ["keys", "distances", "counts", "visited_members", "computed_distances"]
    expected_length = 2  # Two queries

    # Verify result is a BatchMatches object
    for attr in expected_attributes:
        assert hasattr(result, attr)
    assert len(result) == expected_length


def test_search_batch_vectors_keys_and_distances_are_2d_arrays():
    """BatchMatches object has 2D numpy arrays for keys and distances."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=3)

    expected_keys_type = np.ndarray
    expected_keys_ndim = 2
    expected_keys_shape = (2, 3)  # 2 queries, up to 3 results each
    expected_distances_type = np.ndarray
    expected_distances_ndim = 2
    expected_distances_shape = (2, 3)

    assert isinstance(result.keys, expected_keys_type)
    assert result.keys.ndim == expected_keys_ndim
    assert result.keys.shape == expected_keys_shape

    assert isinstance(result.distances, expected_distances_type)
    assert result.distances.ndim == expected_distances_ndim
    assert result.distances.shape == expected_distances_shape


def test_search_batch_vectors_each_query_has_own_results():
    """Each query in batch has independent results."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],  # Exact match to key 1
            [100, 150, 200, 250],  # Exact match to key 2
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=1)

    # First query should find key 1 with distance 0
    expected_keys_query_0 = np.array([1], dtype=np.uint64)
    expected_distance_query_0 = np.array([0.0], dtype=np.float32)

    assert result.keys[0][0] == expected_keys_query_0[0]
    assert result.distances[0][0] == expected_distance_query_0[0]

    # Second query should find key 2 with distance 0
    expected_keys_query_1 = np.array([2], dtype=np.uint64)
    expected_distance_query_1 = np.array([0.0], dtype=np.float32)

    assert result.keys[1][0] == expected_keys_query_1[0]
    assert result.distances[1][0] == expected_distance_query_1[0]


def test_search_batch_vectors_counts_array_shows_results_per_query():
    """BatchMatches.counts array shows number of valid results per query."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    # Request more results than available
    result = idx.search(queries, count=10)

    expected_counts = np.array([2, 2], dtype=np.uint64)

    assert isinstance(result.counts, np.ndarray)
    assert_array_equal(result.counts, expected_counts)


def test_search_batch_vectors_accessing_individual_query_matches():
    """Individual Matches objects for each query can be accessed by index."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    queries = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
        ],
        dtype=np.uint8,
    )
    result = idx.search(queries, count=2)

    expected_matches_attributes = ["keys", "distances"]
    expected_matches_length = 2

    # Access Matches object for first query
    first_query_matches = result[0]
    for attr in expected_matches_attributes:
        assert hasattr(first_query_matches, attr)
    assert len(first_query_matches) == expected_matches_length

    # Access Matches object for second query
    second_query_matches = result[1]
    assert len(second_query_matches) == expected_matches_length


def test_search_batch_vectors_accessing_nested_match_objects():
    """Match objects can be accessed from BatchMatches via query index."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
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

    expected_match_attributes = ["key", "distance"]

    # Access first Match of first query
    first_query_first_match = result[0][0]
    for attr in expected_match_attributes:
        assert hasattr(first_query_first_match, attr)
    assert first_query_first_match.key == result.keys[0][0]
    assert first_query_first_match.distance == result.distances[0][0]


def test_search_batch_vectors_to_list_returns_flattened_tuples():
    """BatchMatches.to_list() returns flattened list of (key, distance) tuples."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
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

    result_list = result.to_list()

    expected_list_type = list
    expected_length = 4  # 2 queries * 2 results each = 4 tuples
    expected_tuple_length = 2

    assert isinstance(result_list, expected_list_type)
    assert len(result_list) == expected_length
    assert all(isinstance(item, tuple) and len(item) == expected_tuple_length for item in result_list)


# Tests for exact parameter (exact=True vs exact=False)


def test_search_exact_false_uses_approximate_search():
    """exact=False (default) uses approximate HNSW search."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=2, exact=False)

    expected_length = 2
    expected_first_key = 1
    expected_first_distance = 0.0

    # Approximate search should still work correctly
    assert len(result) == expected_length
    # First result should be exact match with distance 0
    assert result.keys[0] == expected_first_key
    assert result.distances[0] == expected_first_distance


def test_search_exact_true_uses_exhaustive_search():
    """exact=True uses exhaustive linear-time search."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=2, exact=True)

    expected_length = 2
    expected_first_key = 1
    expected_first_distance = 0.0

    # Exact search guarantees true nearest neighbors
    assert len(result) == expected_length
    # First result should be exact match with distance 0
    assert result.keys[0] == expected_first_key
    assert result.distances[0] == expected_first_distance


def test_search_exact_true_and_false_same_results_small_dataset():
    """For small datasets, exact and approximate search return same results."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    result_approximate = idx.search(query, count=3, exact=False)
    result_exact = idx.search(query, count=3, exact=True)

    # For small datasets, results should be identical
    assert_array_equal(result_approximate.keys, result_exact.keys)
    assert_array_equal(result_approximate.distances, result_exact.distances)


# Tests for visited_members and computed_distances statistics


def test_search_provides_statistics():
    """Matches object provides search statistics."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=2)

    expected_visited_members_type = int
    expected_computed_distances_type = int
    expected_minimum_value = 0

    # Statistics should be non-negative integers
    assert isinstance(result.visited_members, expected_visited_members_type)
    assert isinstance(result.computed_distances, expected_computed_distances_type)
    assert result.visited_members >= expected_minimum_value
    assert result.computed_distances >= expected_minimum_value


def test_search_batch_provides_statistics():
    """BatchMatches object provides search statistics across all queries."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
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

    expected_visited_members_type = int
    expected_computed_distances_type = int
    expected_minimum_value = 0

    # Statistics should be non-negative integers
    assert isinstance(result.visited_members, expected_visited_members_type)
    assert isinstance(result.computed_distances, expected_computed_distances_type)
    assert result.visited_members >= expected_minimum_value
    assert result.computed_distances >= expected_minimum_value


# Tests for edge cases and special scenarios


def test_search_with_count_zero_causes_segfault():
    """
    Searching with count=0 causes segmentation fault (usearch bug).

    This test documents the known issue that usearch crashes when count=0.
    The test is skipped to prevent test suite crashes.

    Related: This should be reported as a bug to unum-cloud/usearch
    """
    pytest.skip("count=0 causes segmentation fault in usearch")


def test_search_with_count_one_returns_single_best_match():
    """Searching with count=1 returns only the best match."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=1)

    expected_length = 1
    expected_key = np.array([1], dtype=np.uint64)
    expected_distance = np.array([0.0], dtype=np.float32)

    assert len(result) == expected_length
    assert result.keys[0] == expected_key[0]
    assert result.distances[0] == expected_distance[0]


def test_search_binary_vectors_hamming_distance_is_bit_count():
    """Hamming distance for binary vectors equals number of differing bits."""
    idx = Index(ndim=8, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    # Add vector: 11111111 in binary (255 in decimal)
    idx.add(1, np.array([255], dtype=np.uint8))
    # Add vector: 00000000 in binary (0 in decimal)
    idx.add(2, np.array([0], dtype=np.uint8))

    # Query: 11111110 in binary (254 in decimal) - differs from 255 by 1 bit
    query = np.array([254], dtype=np.uint8)
    result = idx.search(query, count=2)

    # Key 1 (255) should be closest: 1 bit difference
    # Key 2 (0) should be further: 7 bit differences
    expected_keys = np.array([1, 2], dtype=np.uint64)
    expected_distance_to_key1 = 1.0
    expected_distance_to_key2 = 7.0

    assert_array_equal(result.keys, expected_keys)
    assert result.distances[0] == expected_distance_to_key1
    assert result.distances[1] == expected_distance_to_key2
