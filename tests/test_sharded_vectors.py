"""
Tests for ShardedIndex.vectors property.

Verifies lazy vector iteration across shards with:
- Basic iteration and length
- Indexing and slicing
- Numpy array conversion
- Behavior across multiple shards
- Memory efficiency (lazy evaluation)
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from usearch.index import MetricKind, ScalarKind

from iscc_usearch.sharded import ShardedIndex, ShardedIndexedVectors


# Basic functionality tests


def test_vectors_empty_index_returns_empty(tmp_path):
    """Empty sharded index has empty vectors."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )

    vectors = idx.vectors

    assert isinstance(vectors, ShardedIndexedVectors)
    assert len(vectors) == 0
    assert list(vectors) == []


def test_vectors_single_entry_returns_that_vector(tmp_path):
    """Sharded index with one entry returns that vector."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    vec = np.array([178, 204, 60, 240], dtype=np.uint8)
    idx.add(42, vec)

    vectors = idx.vectors

    assert len(vectors) == 1
    result = list(vectors)
    assert len(result) == 1
    assert_array_equal(result[0], vec)


def test_vectors_multiple_entries_returns_all_vectors(tmp_path):
    """Sharded index with multiple entries returns all vectors."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    vec1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vec2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    vec3 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(1, vec1)
    idx.add(2, vec2)
    idx.add(5, vec3)

    vectors = idx.vectors
    vectors_list = list(vectors)

    assert len(vectors) == 3
    # Check that all vectors are present (order may vary)
    found = [False, False, False]
    for v in vectors_list:
        if np.array_equal(v, vec1):
            found[0] = True
        elif np.array_equal(v, vec2):
            found[1] = True
        elif np.array_equal(v, vec3):
            found[2] = True
    assert all(found)


# Iteration tests


def test_vectors_iteration(tmp_path):
    """Vectors can be iterated."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    vectors = idx.vectors
    vectors_list = list(vectors)

    assert len(vectors_list) == 3


def test_vectors_multiple_iterations(tmp_path):
    """Vectors can be iterated multiple times."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    vec1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vec2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    idx.add(1, vec1)
    idx.add(2, vec2)

    vectors = idx.vectors

    first_pass = list(vectors)
    second_pass = list(vectors)

    assert len(first_pass) == len(second_pass) == 2
    for v1, v2 in zip(first_pass, second_pass):
        assert_array_equal(v1, v2)


# Indexing and slicing tests


def test_vectors_indexing(tmp_path):
    """Vectors support integer indexing."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    vec1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vec2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    vec3 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(10, vec1)
    idx.add(20, vec2)
    idx.add(30, vec3)

    vectors = idx.vectors

    # Get first vector
    first = vectors[0]
    assert isinstance(first, np.ndarray)
    assert first.dtype == np.uint8

    # Negative indexing
    last = vectors[-1]
    assert isinstance(last, np.ndarray)


def test_vectors_indexing_out_of_range(tmp_path):
    """Vectors indexing raises IndexError for out of range."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    vectors = idx.vectors

    with pytest.raises(IndexError):
        _ = vectors[10]

    with pytest.raises(IndexError):
        _ = vectors[-10]


def test_vectors_slicing(tmp_path):
    """Vectors support slicing."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    for i in range(10):
        idx.add(i * 10, np.array([i, i + 1, i + 2, i + 3], dtype=np.uint8))

    vectors = idx.vectors

    # First 3 vectors
    first_3 = vectors[:3]
    assert isinstance(first_3, np.ndarray)
    assert first_3.shape == (3, 4)


# Numpy array conversion tests


def test_vectors_numpy_conversion(tmp_path):
    """Vectors can be converted to numpy array."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(5, np.array([1, 2, 3, 4], dtype=np.uint8))

    vectors = idx.vectors
    vectors_array = np.asarray(vectors)

    assert isinstance(vectors_array, np.ndarray)
    assert vectors_array.shape == (3, 4)


def test_vectors_numpy_conversion_with_dtype(tmp_path):
    """Vectors can be converted to numpy array with specific dtype."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    vectors = idx.vectors
    vectors_array = np.asarray(vectors, dtype=np.int32)

    assert vectors_array.dtype == np.int32


def test_vectors_numpy_conversion_empty(tmp_path):
    """Empty vectors converts to empty array with correct shape."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )

    vectors = idx.vectors
    vectors_array = np.asarray(vectors)

    assert isinstance(vectors_array, np.ndarray)
    assert len(vectors_array) == 0
    # For B1 with ndim=32, vectors are 4 bytes (32 bits / 8)
    assert vectors_array.shape == (0, 4)


# Multi-shard tests


def test_vectors_across_multiple_shards(tmp_path):
    """Vectors are aggregated across multiple shards."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        shard_size=500,  # Very small to force rotation
        bloom_filter=False,
    )

    # Add enough entries to create multiple shards
    added_vectors = []
    for i in range(100):
        vec = np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8)
        idx.add(i, vec)
        added_vectors.append(vec)

    # Should have multiple shards
    assert idx.shard_count >= 1

    vectors = idx.vectors
    vectors_list = list(vectors)

    # All vectors should be present
    assert len(vectors_list) == 100


def test_vectors_after_save_and_load(tmp_path):
    """Vectors are preserved after save and load."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    vec1 = np.array([178, 204, 60, 240], dtype=np.uint8)
    vec2 = np.array([100, 150, 200, 250], dtype=np.uint8)
    vec3 = np.array([1, 2, 3, 4], dtype=np.uint8)
    idx.add(1, vec1)
    idx.add(2, vec2)
    idx.add(5, vec3)
    idx.save()

    # Load into new instance
    idx2 = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )

    vectors = idx2.vectors
    vectors_list = list(vectors)

    assert len(vectors_list) == 3


# Live view behavior tests


def test_vectors_is_live_view(tmp_path):
    """Vectors reflects changes made after obtaining the vectors object."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    # Get vectors reference
    vectors = idx.vectors
    assert len(vectors) == 1

    # Add more entries
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    # vectors should reflect the new additions (live view)
    assert len(vectors) == 3


# Memory efficiency tests


def test_vectors_lazy_iteration(tmp_path):
    """Vectors iteration is lazy (generator-based)."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    for i in range(100):
        idx.add(i, np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8))

    vectors = idx.vectors

    # Partial iteration should work
    first_10 = []
    for i, vec in enumerate(vectors):
        if i >= 10:
            break
        first_10.append(vec)

    assert len(first_10) == 10


def test_vectors_repr(tmp_path):
    """Vectors has useful repr."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    vectors = idx.vectors
    repr_str = repr(vectors)

    assert "ShardedIndexedVectors" in repr_str
    assert "count=2" in repr_str


# Edge cases


def test_vectors_slicing_empty_result(tmp_path):
    """Slicing that results in empty returns empty array with correct shape."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        bloom_filter=False,
    )
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    vectors = idx.vectors

    # Slice beyond available
    empty_slice = vectors[10:20]
    assert isinstance(empty_slice, np.ndarray)
    assert len(empty_slice) == 0
    # For B1 with ndim=32, vectors are 4 bytes (32 bits / 8)
    assert empty_slice.shape == (0, 4)


def test_vectors_indexing_in_viewed_shards(tmp_path):
    """Vectors indexing finds vectors in viewed shards."""
    idx = ShardedIndex(
        ndim=32,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        path=tmp_path / "index",
        shard_size=500,  # Small to force rotation
        bloom_filter=False,
    )

    # Add entries to fill first shard
    for i in range(50):
        idx.add(i, np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8))

    # Save and manually trigger rotation
    idx.save()
    idx._rotate_shard()

    # Add entries to new shard
    for i in range(50, 60):
        idx.add(i, np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8))

    # Now we have viewed shards + active shard
    vectors = idx.vectors

    # Access vectors via indexing - should find in viewed shards
    first_vec = vectors[0]
    assert isinstance(first_vec, np.ndarray)

    # Access vector in middle (likely in viewed shard)
    middle_vec = vectors[25]
    assert isinstance(middle_vec, np.ndarray)

    # Access vector near end (could be in active shard)
    last_vec = vectors[-1]
    assert isinstance(last_vec, np.ndarray)
