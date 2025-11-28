"""
Comprehensive tests for usearch Index with multi=True parameter.

These tests document the exact behavior of search() when keys have multiple
vectors, which is critical for implementing a simprint index where each asset
(ISCC-ID) can have many simprints (chunks).

Key Questions Answered:
1. When a key has multiple vectors, which one determines the distance in search results?
2. How do you identify which specific vector matched?
3. What's the workflow for going from search results back to specific vectors?
"""

import numpy as np
from numpy.testing import assert_array_equal
from usearch.index import Index, MetricKind, ScalarKind


# Core multi=True search behavior tests


def test_multi_search_returns_key_not_specific_vector():
    """
    Search returns keys, not individual vector identifiers.

    When key 1 has 3 vectors, search() returns key 1 with a single distance.
    You cannot determine which of the 3 vectors was the closest match
    without calling get() and comparing manually.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Add 3 different vectors to key 1
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))  # Vector A
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))  # Vector B
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))  # Vector C

    # Search for exact match to Vector A
    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=1)

    # Result contains key 1, but doesn't tell us which of the 3 vectors matched
    assert result.keys[0] == 1
    assert result.distances[0] == 0.0  # Exact match to Vector A

    # To find which vector matched, must call get() and compare
    stored_vectors = idx.get(1)
    expected_shape = (3, 4)  # 3 vectors, 4 bytes each
    assert stored_vectors.shape == expected_shape


def test_multi_search_uses_closest_vector_for_distance():
    """
    When key has multiple vectors, search distance is to the CLOSEST vector.

    If key 1 has vectors at distances [0, 5, 10] from query,
    the search result shows distance 0 (the minimum).
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Add 3 vectors to key 1 with known distances to query
    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))  # Distance 0 (exact)
    idx.add(1, np.array([178, 204, 60, 245], dtype=np.uint8))  # Distance 3 (3 bits different)
    idx.add(1, np.array([255, 255, 255, 255], dtype=np.uint8))  # Distance 16 (many bits different)

    result = idx.search(query, count=1)

    # Search returns the minimum distance (0) from all vectors for key 1
    assert result.keys[0] == 1
    assert result.distances[0] == 0.0


def test_multi_search_multiple_keys_each_with_multiple_vectors():
    """
    With multiple keys each having multiple vectors, search returns best distance per key.

    Key 1: vectors at distances [1, 5, 10] -> result distance: 1
    Key 2: vectors at distances [0, 8] -> result distance: 0
    Results ordered by minimum distance: key 2 (0), then key 1 (1)
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    # Key 1: best match has distance 1
    idx.add(1, np.array([178, 204, 60, 241], dtype=np.uint8))  # Distance 1
    idx.add(1, np.array([178, 204, 60, 245], dtype=np.uint8))  # Distance 3
    idx.add(1, np.array([255, 255, 255, 255], dtype=np.uint8))  # Distance 16

    # Key 2: best match has distance 0
    idx.add(2, np.array([178, 204, 60, 240], dtype=np.uint8))  # Distance 0 (exact)
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))  # Distance 8

    result = idx.search(query, count=2)

    # Results ordered by best distance per key
    assert_array_equal(result.keys, np.array([2, 1], dtype=np.uint64))
    assert result.distances[0] == 0.0  # Key 2's best distance
    assert result.distances[1] == 1.0  # Key 1's best distance


def test_multi_get_after_search_retrieves_all_vectors():
    """
    Workflow: search() -> get() to find which specific vector matched.

    1. Search returns key and best distance
    2. Call get(key) to retrieve ALL vectors for that key
    3. Manually compare each vector to query to find which one matched
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    # Add 3 vectors to key 1, with second one being exact match
    vector_a = np.array([100, 150, 200, 250], dtype=np.uint8)
    vector_b = np.array([178, 204, 60, 240], dtype=np.uint8)  # Exact match
    vector_c = np.array([1, 2, 3, 4], dtype=np.uint8)

    idx.add(1, vector_a)
    idx.add(1, vector_b)
    idx.add(1, vector_c)

    # Step 1: Search returns key 1 with distance 0
    search_result = idx.search(query, count=1)
    assert search_result.keys[0] == 1
    assert search_result.distances[0] == 0.0

    # Step 2: Get all vectors for key 1
    stored_vectors = idx.get(1)
    expected_shape = (3, 4)
    assert stored_vectors.shape == expected_shape

    # Step 3: Find which vector(s) match by manual comparison
    # Note: usearch doesn't guarantee order, so we check if exact match exists
    matches_found = False
    for stored_vec in stored_vectors:
        if np.array_equal(stored_vec, vector_b):
            matches_found = True
            break

    assert matches_found, "Exact match vector should be in stored vectors"


def test_multi_batch_search_with_multiple_vectors_per_key():
    """
    Batch search with multi=True: each query finds best distance per key.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Key 1 has 2 vectors
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))

    # Key 2 has 2 vectors
    idx.add(2, np.array([1, 2, 3, 4], dtype=np.uint8))
    idx.add(2, np.array([10, 20, 30, 40], dtype=np.uint8))

    # Two queries: first matches key 1, second matches key 2
    queries = np.array(
        [
            [178, 204, 60, 240],  # Exact match to key 1's first vector
            [1, 2, 3, 4],  # Exact match to key 2's first vector
        ],
        dtype=np.uint8,
    )

    result = idx.search(queries, count=2)

    # Query 0 should find key 1 first (distance 0)
    assert result.keys[0][0] == 1
    assert result.distances[0][0] == 0.0

    # Query 1 should find key 2 first (distance 0)
    assert result.keys[1][0] == 2
    assert result.distances[1][0] == 0.0


# Tests for practical simprint index use case


def test_multi_simprint_use_case_asset_with_many_chunks():
    """
    Simulates the simprint use case: one asset (ISCC-ID) with many chunk simprints.

    Asset 1 (ISCC-ID body = 1): 5 chunk simprints
    Asset 2 (ISCC-ID body = 2): 3 chunk simprints

    Query: Find assets that have similar chunks
    """
    idx = Index(ndim=64, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Asset 1: 5 chunks (simprints)
    asset_1_chunks = [
        np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8),
        np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint8),
        np.array([100, 101, 102, 103, 104, 105, 106, 107], dtype=np.uint8),
        np.array([200, 201, 202, 203, 204, 205, 206, 207], dtype=np.uint8),
        np.array([255, 254, 253, 252, 251, 250, 249, 248], dtype=np.uint8),
    ]

    for chunk in asset_1_chunks:
        idx.add(1, chunk)

    # Asset 2: 3 chunks
    asset_2_chunks = [
        np.array([11, 12, 13, 14, 15, 16, 17, 18], dtype=np.uint8),
        np.array([21, 22, 23, 24, 25, 26, 27, 28], dtype=np.uint8),
        np.array([31, 32, 33, 34, 35, 36, 37, 38], dtype=np.uint8),
    ]

    for chunk in asset_2_chunks:
        idx.add(2, chunk)

    # Query: Search for asset 1's third chunk
    query = np.array([100, 101, 102, 103, 104, 105, 106, 107], dtype=np.uint8)
    result = idx.search(query, count=2)

    # Should find asset 1 first (exact match to one of its chunks)
    assert result.keys[0] == 1
    assert result.distances[0] == 0.0

    # Verify we can retrieve all chunks for asset 1
    asset_1_retrieved = idx.get(1)
    expected_shape = (5, 8)  # 5 chunks, 8 bytes each
    assert asset_1_retrieved.shape == expected_shape


def test_multi_query_with_multiple_simprints_finds_best_asset():
    """
    Query with multiple simprints (representing a document) finds assets with similar chunks.

    This simulates: "I have a document with 3 chunks, find similar documents"
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Asset 1: Has chunks A, B, C
    chunk_a = np.array([1, 1, 1, 1], dtype=np.uint8)
    chunk_b = np.array([2, 2, 2, 2], dtype=np.uint8)
    chunk_c = np.array([3, 3, 3, 3], dtype=np.uint8)

    idx.add(1, chunk_a)
    idx.add(1, chunk_b)
    idx.add(1, chunk_c)

    # Asset 2: Has chunks D, E, F (different from asset 1)
    chunk_d = np.array([100, 100, 100, 100], dtype=np.uint8)
    chunk_e = np.array([200, 200, 200, 200], dtype=np.uint8)
    chunk_f = np.array([255, 255, 255, 255], dtype=np.uint8)

    idx.add(2, chunk_d)
    idx.add(2, chunk_e)
    idx.add(2, chunk_f)

    # Query with 3 chunks: exact matches to asset 1's chunks
    queries = np.array(
        [
            [1, 1, 1, 1],  # Matches asset 1's chunk A
            [2, 2, 2, 2],  # Matches asset 1's chunk B
            [3, 3, 3, 3],  # Matches asset 1's chunk C
        ],
        dtype=np.uint8,
    )

    result = idx.search(queries, count=2)

    # All 3 queries should find asset 1 first with distance 0
    for query_idx in range(3):
        assert result.keys[query_idx][0] == 1
        assert result.distances[query_idx][0] == 0.0


# Edge cases and corner cases


def test_multi_single_vector_per_key_behaves_same_as_multi_false():
    """
    When each key has only 1 vector, multi=True behaves like multi=False.
    """
    # Index with multi=True
    idx_multi = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx_multi.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx_multi.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    # Index with multi=False
    idx_single = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx_single.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx_single.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    query = np.array([178, 204, 60, 240], dtype=np.uint8)

    result_multi = idx_multi.search(query, count=2)
    result_single = idx_single.search(query, count=2)

    # Search results should be identical
    assert_array_equal(result_multi.keys, result_single.keys)
    assert_array_equal(result_multi.distances, result_single.distances)


def test_multi_empty_index_returns_empty_results():
    """Empty index with multi=True returns empty Matches."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result = idx.search(query, count=10)

    expected_length = 0
    assert len(result) == expected_length
    assert result.keys.shape == (0,)
    assert result.distances.shape == (0,)


def test_multi_remove_key_removes_all_vectors():
    """
    Removing a key removes ALL vectors associated with it.

    After removal, the key no longer appears in search results.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Add 3 vectors to key 1
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    # Add 1 vector to key 2
    idx.add(2, np.array([50, 60, 70, 80], dtype=np.uint8))

    # Search finds both keys
    query = np.array([178, 204, 60, 240], dtype=np.uint8)
    result_before = idx.search(query, count=2)
    assert 1 in result_before.keys

    # Remove key 1 (removes all 3 vectors)
    idx.remove(1)

    # Search now only finds key 2
    result_after = idx.search(query, count=2)
    assert len(result_after) == 1
    assert result_after.keys[0] == 2
    assert 1 not in result_after.keys


def test_multi_duplicate_vectors_same_key_both_stored():
    """
    Adding the same vector twice to the same key stores both copies.

    This is expected behavior with multi=True - there's no deduplication.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    vector = np.array([178, 204, 60, 240], dtype=np.uint8)

    # Add same vector twice to key 1
    idx.add(1, vector)
    idx.add(1, vector)

    # Both copies are stored
    stored = idx.get(1)
    expected_shape = (2, 4)  # 2 copies, 4 bytes each
    assert stored.shape == expected_shape

    # Both should be identical
    assert_array_equal(stored[0], stored[1])


# Tests documenting return type differences


def test_multi_get_returns_2d_array_multi_false_returns_1d():
    """
    Document the different return types for get() based on multi parameter.

    multi=True: Always returns 2D array (even for 1 vector)
    multi=False: Returns 1D array for single vector
    """
    # multi=True
    idx_multi = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx_multi.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result_multi = idx_multi.get(1)
    assert result_multi.ndim == 2
    assert result_multi.shape == (1, 4)

    # multi=False
    idx_single = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx_single.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result_single = idx_single.get(1)
    assert result_single.ndim == 1
    assert result_single.shape == (4,)


def test_multi_batch_add_with_duplicate_keys():
    """
    Batch add with duplicate keys in the batch stores all vectors.

    This is useful for efficiently adding multiple simprints for one asset.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Add 3 vectors to key 1 in a single batch operation
    keys = [1, 1, 1]
    vectors = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
            [1, 2, 3, 4],
        ],
        dtype=np.uint8,
    )

    result = idx.add(keys, vectors)

    expected_keys = np.array([1, 1, 1], dtype=np.uint64)
    assert_array_equal(result, expected_keys)

    # All 3 vectors stored for key 1
    stored = idx.get(1)
    expected_shape = (3, 4)
    assert stored.shape == expected_shape


# Performance and scalability tests


def test_multi_many_vectors_per_key_search_performance():
    """
    Document search behavior when keys have many vectors (100+).

    This simulates a real simprint use case where documents have many chunks.
    """
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)

    # Add 100 vectors to key 1
    for i in range(100):
        vector = np.array([i, i, i, i], dtype=np.uint8)
        idx.add(1, vector)

    # Add 50 vectors to key 2
    for i in range(50):
        vector = np.array([i + 100, i + 100, i + 100, i + 100], dtype=np.uint8)
        idx.add(2, vector)

    # Search for exact match to key 1's 50th vector
    query = np.array([50, 50, 50, 50], dtype=np.uint8)
    result = idx.search(query, count=2)

    # Should find key 1 first with distance 0
    assert result.keys[0] == 1
    assert result.distances[0] == 0.0

    # Verify we can retrieve all 100 vectors for key 1
    stored = idx.get(1)
    expected_shape = (100, 4)
    assert stored.shape == expected_shape
